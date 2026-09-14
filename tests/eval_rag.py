"""
RAG 评估脚本 - LLM-as-Judge
使用 DeepSeek 作为评判模型，评估 RAG 系统的回答质量

评估指标：
1. Faithfulness（忠实度）：回答的实质论断是否基于检索到的上下文
2. Answer Relevancy（回答相关性）：回答是否切题
3. Context Precision（上下文精确度）：检索的上下文是否与问题相关
4. Key Points Coverage（要点覆盖率）：回答是否覆盖了关键要点

项目定制（2026-08-30）：
- 被测系统是「名人角色扮演对话」应用，舞台指令/角色口吻/文学比喻是预期产品行为，
  判规明确指示风格化表达不参与评分，只核查实质论断（概念/事实/归属/经历）。
- 判官调用走 ainvoke_nonempty + 解析失败重试：此前裸 ainvoke 撞上中转空响应时，
  解析失败被静默记为 5 分，曾把忠实度"评"出 5.8（事后发现 18/23 个低分全是
  解析失败默认分）。现解析失败重试一次，仍失败记 None 并从平均分中剔除、单独计数。
- 结果文件保存 contexts，支持事后用不同判规对同一批答案重评（对照实验）。
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.llm import get_chat_llm, ainvoke_nonempty
from src.core.llm_json import extract_json

from tests.eval_dataset import EVAL_DATASET, get_dataset_by_difficulty

_EXPANDED_DATASET_PATH = Path(__file__).parent / "eval_dataset_expanded.json"


def load_full_dataset() -> list[dict]:
    """内置 12 题 + 扩充数据集（tests/eval_dataset_expanded.json，存在则合并）"""
    items = list(EVAL_DATASET)
    if _EXPANDED_DATASET_PATH.exists():
        try:
            expanded = json.loads(_EXPANDED_DATASET_PATH.read_text(encoding="utf-8"))
            if isinstance(expanded, list):
                seen = {it["question"] for it in items}
                for it in expanded:
                    if isinstance(it, dict) and it.get("question") and it["question"] not in seen:
                        items.append(it)
                        seen.add(it["question"])
        except Exception as e:
            print(f"⚠️ 扩充数据集加载失败（仅用内置 12 题）: {e}")
    return items


# ============================================================
# 评估用 LLM
# ============================================================
def get_judge_llm() -> ChatOpenAI:
    """获取评判用 LLM。max_retries=0：SDK 内置重试会把每次 429 变成 3 连击，
    在中转限额紧张时火上浇油——重试统一交给 _judge 的长间隔策略"""
    return get_chat_llm(
        temperature=0.1,  # 低温度，评判更稳定
        max_tokens=1500,
        max_retries=0,
    )


# ============================================================
# 评估 Prompt
# ============================================================
COMBINED_JUDGE_PROMPT = """你是一个严格的评估专家。被测系统是一个「名人角色扮演对话」应用：
回答以历史人物的第一人称口吻生成，舞台化的表达是产品的核心体验，不是缺陷。

## 问题
{question}

## 检索到的上下文
{contexts}

## 系统的回答
{answer}

## 关键要点
{key_points}

## 评判规则（四个维度，全部针对本项目定制）

### 1. 忠实度 faithfulness（0-10）
只考察回答的**实质论断**是否忠于上下文，**表达风格完全不参与评分**：
- 不计分、绝不扣分：舞台指令与动作描写（如 *荣格轻轻叩了叩桌面*）、角色语气词、
  开场白/客套、戏剧化修辞与文学比喻、对上下文已有概念的通俗化复述、公认常识
- 需逐条核查的实质论断：概念解释是否与上下文一致；具体事实（数字/年份/著作名/事件）；
  著作归属（"我在《X》中写过"——上下文有 → 忠实，完全找不到 → 幻觉）；
  第一人称经历（上下文有记载 → 忠实；完全没有且属具体事实陈述 → 幻觉）；与上下文矛盾 → 幻觉
- 锚点：9-10 全部有依据 | 7-8 个别模糊但无编造 | 4-6 有 1-2 处无依据论断 | 0-3 大量编造

### 2. 回答相关性 relevancy（0-10）
只看实质内容是否回应了问题本身；角色口吻、舞台指令、寒暄式开场都不算跑题。

### 3. 上下文精确度 context_precision（0-10）
检索到的上下文是否包含回答该问题所需的信息、有无大量无关内容（这是在评检索质量）。

### 4. 要点覆盖率 key_points_coverage（0-10）
要点以**意思覆盖**为准（表述不同但意思到位即算覆盖）；舞台指令、修辞风格不影响判断。

## 输出格式（严格 JSON，不要输出 JSON 以外的任何文字）
{{
    "faithfulness": {{"score": 0-10整数, "reason": "实质论断核查结论（风格不参与）", "unsupported_claims": ["无依据论断，没有则空列表"]}},
    "relevancy": {{"score": 0-10整数, "reason": "简短理由"}},
    "context_precision": {{"score": 0-10整数, "reason": "简短理由"}},
    "key_points_coverage": {{"score": 0-10整数, "reason": "简短理由", "covered_points": [], "missed_points": []}}
}}
"""


# ============================================================
# 评估函数
# ============================================================
async def _judge(llm: ChatOpenAI, prompt: str, attempts: int = 2) -> Optional[dict]:
    """
    判官调用 + JSON 解析，带两层容错：
    1. 空响应：ainvoke_nonempty 立即重试（上游中转的空壳 200 故障）
    2. 格式坏：重问一次；仍失败返回 None —— 该指标记「无效」，
       绝不默默记 5 分污染平均（旧版曾因此把忠实度"评"出 5.8，事后发现
       18/23 个低分全是解析失败默认分）
    """
    judge_gap = float(os.environ.get("EVAL_JUDGE_GAP_SEC", "20"))
    if judge_gap > 0:
        # 距离上一次 LLM 调用（取答案）强制留出间隔，否则必撞中转 rpm 限额
        await asyncio.sleep(judge_gap)
    for i in range(attempts):
        try:
            resp = await ainvoke_nonempty(llm, [("user", prompt)])
        except Exception as e:
            # 429/网络异常同样按"该次失败"处理，等一个间隔再试——
            # 若让异常炸出去，整道题的评估就废了
            print(f"     ⚠️ 判官调用失败（第 {i + 1}/{attempts} 次）: {str(e)[:60]}")
            if i + 1 < attempts and judge_gap > 0:
                await asyncio.sleep(judge_gap)
            continue
        text = getattr(resp, "content", "") or ""
        try:
            obj = extract_json(text)
            if isinstance(obj, dict) and ("score" in obj or "faithfulness" in obj):
                return obj
        except Exception:
            pass
        print(f"     ⚠️ 判官输出无法解析（第 {i + 1}/{attempts} 次），重试…")
        if i + 1 < attempts and judge_gap > 0:
            await asyncio.sleep(judge_gap)
    return None


async def evaluate_single(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    contexts: list[str],
    key_points: list[str],
) -> dict:
    """评估单个问题：单次判官调用返回四维评分（原 4 连发会自撞中转 rpm 限额）"""
    contexts_text = "\n\n---\n\n".join(contexts) if contexts else "（无上下文）"
    key_points_text = "\n".join(f"- {kp}" for kp in key_points)

    prompt = COMBINED_JUDGE_PROMPT.format(
        question=question, contexts=contexts_text, answer=answer, key_points=key_points_text,
    )
    obj = await _judge(llm, prompt)

    def sub(metric: str) -> dict:
        v = (obj or {}).get(metric)
        return v if isinstance(v, dict) else {}

    def norm_score(v) -> int:
        """judge 可能返回 int / float / '8分' / '8/10'，统一归一为 0-10 整数"""
        if isinstance(v, bool):
            return 5
        if isinstance(v, (int, float)):
            return max(0, min(10, int(v)))
        if isinstance(v, str):
            m = re.search(r"(\d{1,2})\s*/?\s*10?", v)
            if m:
                return max(0, min(10, int(m.group(1))))
        return 5

    def metric(sd: dict) -> Optional[int]:
        return norm_score(sd.get("score")) if sd and "score" in sd else None

    f, r, c, k = sub("faithfulness"), sub("relevancy"), sub("context_precision"), sub("key_points_coverage")
    return {
        "faithfulness": metric(f),
        "relevancy": metric(r),
        "context_precision": metric(c),
        "key_points_coverage": metric(k),
        "details": {
            "faithfulness_reason": f.get("reason", "判官输出无法解析"),
            "faithfulness_unsupported_claims": f.get("unsupported_claims", []),
            "faithfulness_parse": "ok" if f else "failed",
            "relevancy_reason": r.get("reason", "判官输出无法解析"),
            "relevancy_parse": "ok" if r else "failed",
            "context_precision_reason": c.get("reason", "判官输出无法解析"),
            "context_precision_parse": "ok" if c else "failed",
            "key_points_reason": k.get("reason", "判官输出无法解析"),
            "covered_points": k.get("covered_points", []),
            "missed_points": k.get("missed_points", []),
            "key_points_parse": "ok" if k else "failed",
        },
    }


# ============================================================
# 调用系统获取回答
# ============================================================
async def get_system_answer(question: str, scene: str = "persona", character: str = "jung") -> dict:
    """
    调用评估专用端点获取回答和上下文

    返回: {answer: str, contexts: list[str]}
    """
    import httpx

    # trust_env=False：评估永远直连本机服务。若走系统代理（HTTP_PROXY），
    # 代理会劫持 localhost 请求——服务没跑时代理直接回 502，
    # 症状与"中转站挂了"一模一样，排查时极易误判（2026-09-10 踩过）
    # timeout=240：晚间中转变慢时单题管线可超 120s，240s 只影响极端慢题
    async with httpx.AsyncClient(timeout=240.0, trust_env=False) as client:
        response = await client.post(
            "http://localhost:8000/persona/eval_query",
            json={
                "query": question,
                "character_id": character,
            }
        )

        if response.status_code != 200:
            return {"answer": f"ERROR: HTTP {response.status_code}", "contexts": []}

        data = response.json()
        return {
            "answer": data.get("answer", ""),
            "contexts": data.get("contexts", []),
        }


# ============================================================
# 主评估流程
# ============================================================
async def run_evaluation(difficulty: str = None, limit: int = None, pace_sec: float = 0.0):
    """运行完整评估"""
    print("=" * 60)
    print("🎯 RAG 评估开始")
    print("=" * 60)

    # 获取测试集
    dataset = get_dataset_by_difficulty(difficulty) if difficulty else load_full_dataset()
    if limit:
        dataset = dataset[:limit]

    print(f"\n📋 测试集大小: {len(dataset)} 个问题")
    if difficulty:
        print(f"📊 难度筛选: {difficulty}")

    # 初始化评判 LLM
    llm = get_judge_llm()
    print(f"🤖 评判模型: {settings.llm_model}")

    # 存储结果（断点续跑：加载已完成的有效记录，本次只补缺口）
    results = []
    done_questions: set[str] = set()
    _prev_path = Path(__file__).parent / "eval_results.json"
    if _prev_path.exists() and not difficulty:
        try:
            _prev = json.loads(_prev_path.read_text(encoding="utf-8")).get("results", [])
            for r in _prev:
                if isinstance(r, dict) and r.get("question") and r.get("evaluation"):
                    results.append(r)
                    done_questions.add(r["question"])
            if done_questions:
                print(f"🔁 断点续跑：已有 {len(done_questions)} 题有效结果，本次只补缺口")
        except Exception as e:
            print(f"⚠️ 断点文件加载失败，从头评估: {e}")
    total_start = time.time()

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        key_points = item["key_points"]
        difficulty = item["difficulty"]

        print(f"\n[{i}/{len(dataset)}] 📝 {question[:40]}...")
        print(f"     难度: {difficulty}")

        if question in done_questions:
            print("     ✅ 已有结果，跳过（断点续跑）")
            continue

        # 获取系统回答（扩充数据集带 character 字段；内置数据集默认 jung）
        start = time.time()
        try:
            system_response = await get_system_answer(question, character=item.get("character", "jung"))
        except Exception as e:
            # 管线超时/服务异常只跳过该题，不炸整场（2026-09-10：ReadTimeout 曾炸掉全场评估）
            print(f"     ❌ 管线调用异常，跳过该题: {type(e).__name__}: {str(e)[:80]}")
            continue
        answer = system_response["answer"]
        contexts = system_response["contexts"]
        elapsed = time.time() - start

        print(f"     ⏱️  回答耗时: {elapsed:.1f}s")
        print(f"     📄 回答长度: {len(answer)} 字符")

        if not answer or answer.startswith("ERROR"):
            print(f"     ❌ 获取回答失败: {answer}")
            continue

        # 评估回答（单题容错：Judge LLM 走外部中转，偶发异常不应炸掉整个评估）
        eval_start = time.time()
        try:
            eval_result = await evaluate_single(llm, question, answer, contexts, key_points)
        except Exception as e:
            print(f"     ❌ 单题评估异常，跳过该题: {e}")
            continue
        eval_elapsed = time.time() - eval_start

        print(f"     ⏱️  评估耗时: {eval_elapsed:.1f}s")
        print(f"     📊 评分: 忠实度={eval_result['faithfulness']}, "
              f"相关性={eval_result['relevancy']}, "
              f"上下文={eval_result['context_precision']}, "
              f"要点={eval_result['key_points_coverage']}")

        results.append({
            "question": question,
            "difficulty": difficulty,
            "answer": answer,
            "ground_truth": item["ground_truth"],
            "key_points": key_points,
            "contexts": contexts,  # 保存检索上下文：支持事后用不同判规对同一批答案重评
            "evaluation": eval_result,
            "timing": {
                "answer_seconds": elapsed,
                "eval_seconds": eval_elapsed,
            }
        })

        # 题间节流：每题 5 次 LLM 调用（1 生成 + 4 评判），12 题连打会触发
        # 中转服务的 rpm/tpm 配额（429），整轮报废。需要时用 --pace-sec 控制节奏
        if pace_sec > 0:
            await asyncio.sleep(pace_sec)

    total_elapsed = time.time() - total_start

    # 计算汇总统计（None 安全：判官彻底失败的指标不计入平均，单独计数）
    print("\n" + "=" * 60)
    print("📊 评估结果汇总")
    print("=" * 60)

    if not results:
        print("❌ 没有有效的评估结果")
        return

    METRICS = ["faithfulness", "relevancy", "context_precision", "key_points_coverage"]
    METRIC_NAMES = {
        "faithfulness": "忠实度 (Faithfulness)",
        "relevancy": "回答相关性 (Relevancy)",
        "context_precision": "上下文精确度 (Context)",
        "key_points_coverage": "要点覆盖率 (Key Points)",
    }

    def avg_of(metric: str) -> tuple[Optional[float], int]:
        vals = [r["evaluation"][metric] for r in results if r["evaluation"].get(metric) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    avgs: dict[str, Optional[float]] = {}
    judged: dict[str, int] = {}
    n = len(results)
    print(f"\n📈 总体指标（共 {n} 个问题，括号内为该指标的有效评判数）:")
    for m in METRICS:
        avg, cnt = avg_of(m)
        avgs[m], judged[m] = avg, cnt
        label = METRIC_NAMES[m]
        if avg is None:
            print(f"   • {label:<28} 无有效评判")
        else:
            note = "" if cnt == n else f"（⚠️ {n - cnt} 题判官输出无法解析，未计入）"
            print(f"   • {label:<28} {avg:.1f}/10  [{cnt}/{n}]{note}")

    failed_metrics = sum(n - judged[m] for m in METRICS)
    if failed_metrics:
        print(f"\n   ⚠️ 共 {failed_metrics} 个指标因判官输出无法解析被剔除"
              f"（旧版会静默记 5 分污染平均分——那不是评分，是故障）")

    # 按难度分组统计（None 安全：分母只数有效值）
    print(f"\n📊 按难度分组:")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            vals = [
                r["evaluation"][m]
                for r in diff_results for m in METRICS
                if r["evaluation"].get(m) is not None
            ]
            if vals:
                print(f"   • {diff:6s}: {sum(vals) / len(vals):.1f}/10 ({len(diff_results)} 题)")

    # 耗时统计
    avg_answer_time = sum(r["timing"]["answer_seconds"] for r in results) / len(results)
    total_answer_time = sum(r["timing"]["answer_seconds"] for r in results)
    print(f"\n⏱️  耗时统计:")
    print(f"   • 总耗时: {total_elapsed:.1f}s")
    print(f"   • 平均回答耗时: {avg_answer_time:.1f}s")
    print(f"   • 总回答耗时: {total_answer_time:.1f}s")

    # 保存详细结果
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": len(results),
                "avg_faithfulness": round(avgs["faithfulness"], 2) if avgs["faithfulness"] is not None else None,
                "avg_relevancy": round(avgs["relevancy"], 2) if avgs["relevancy"] is not None else None,
                "avg_context_precision": round(avgs["context_precision"], 2) if avgs["context_precision"] is not None else None,
                "avg_key_points_coverage": round(avgs["key_points_coverage"], 2) if avgs["key_points_coverage"] is not None else None,
                "judged_counts": judged,
                "total_time_seconds": round(total_elapsed, 1),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存到: {output_path}")

    # 输出改进建议（仅在有效评判充足时给出，避免被解析失败误导）
    print("\n" + "=" * 60)
    print("💡 改进建议")
    print("=" * 60)

    def warn_if_low(metric: str, msg: str) -> None:
        avg, cnt = avg_of(metric)
        if avg is not None and cnt >= max(3, n // 2) and avg < 7:
            print(f"   ⚠️  {msg}")

    warn_if_low("faithfulness", "忠实度偏低：回答可能包含幻觉，建议优化 prompt 或增加检索量")
    warn_if_low("relevancy", "相关性偏低：回答可能跑题，建议优化 prompt 聚焦问题")
    warn_if_low("context_precision", "上下文精确度偏低：检索质量需要优化，建议调整 chunk_size 或 reranker")
    warn_if_low("key_points_coverage", "要点覆盖率偏低：回答不够完整，建议增加 retrieval_top_k")

    if (avgs["faithfulness"] or 0) >= 8 and (avgs["relevancy"] or 0) >= 8:
        print("   ✅ 整体表现良好！可以考虑增加更多测试问题来验证稳定性")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="按难度筛选")
    parser.add_argument("--limit", type=int, help="限制评估问题数量")
    parser.add_argument("--pace-sec", type=float, default=0.0, help="题间等待秒数（防中转 429 限流）")
    args = parser.parse_args()

    asyncio.run(run_evaluation(difficulty=args.difficulty, limit=args.limit, pace_sec=args.pace_sec))
