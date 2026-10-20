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
            print(f"扩充数据集加载失败（仅用内置 12 题）: {e}")
    return items


# ============================================================
# 评估用 LLM
# ============================================================
def get_judge_llm() -> ChatOpenAI:
    """获取评判用 LLM。max_retries=0：SDK 内置重试会把每次 429 变成 3 连击，
    在中转限额紧张时火上浇油——重试统一交给 _judge 的长间隔策略"""
    return get_chat_llm(
        temperature=0.1,  # 低温度，评判更稳定
        # 2026-10-09：1500 会截断长答案的判官 JSON（四维+幻觉清单+理由），
        # 表现为"判官输出无法解析"的系统性失败（两轮各 35/25 题）——实测 4000 稳定
        max_tokens=4000,
        max_retries=0,
        # 推理模型的 reasoning_content 会吃输出预算导致 JSON 截断（同 knowledge_graph/
        # legend 口径），结构化判官必须关思考
        reasoning_effort="none",
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
只看**实质内容**是否回应了问题本身；表达形式完全不参与评分：
- 不计分、绝不扣分：角色口吻与语气词、舞台指令与动作描写、寒暄式开场/收尾、
  口语化重复与口头禅、为了入戏而做的场景铺垫（前提是随后回到问题本身）
- 需逐条核查的实质回应：问题有几个可答的子问（如"什么是积极想象？荣格自己是如何
  使用这个技巧的？"是两问，每一问都应有对应回应）；回应是否触到问题的**所问角度**
  （问"原因"就要给原因、问"区别"就要给区别，只复述概念定义而回避所问角度 → 该问算漏答）；
  延伸与反问不算跑题，但延伸占比过半且挤掉核心回应 → 按低档评
- 锚点：9-10 实质内容直接回应问题的每一个子问 | 7-8 主体回应，夹杂少量延伸但无漏答 | 4-6 漏掉问题的一个核心子问，或回应浮于表面 | 0-3 答非所问

### 3. 上下文精确度 context_precision（0-10）
评的是**检索质量**：注入的上下文里有没有回答该问题所需的信息、无关内容占多少：
- 不算无关、不扣分：与问题同一话题下的相邻观点、人物/概念的背景铺垫、
  对已检索到概念的其他角度论述、二手文献中对原著观点的转述（内容本身相关即算）
- 算无关：与问题话题明显不相干的片段（如问"共时性"却整段讲心理类型分类、
  问某事件却整段是另一本书的美学论述）；仅因个别关键词撞车而命中的段落
- 需逐条核查的两点：①全部片段中至少 1-2 条含回答所需的核心信息（定义/观点/事例/数字）；
  ②通读后估计无关内容占比（不足两成 → 不影响；两成到一半 → 7-8；过半 → 4-6）
- 锚点：9-10 所需信息齐备，注入内容几乎全部相关 | 7-8 所需信息齐备或仅缺边角，夹杂少量无关内容 | 4-6 缺少回答所需的关键信息，或过半内容与问题无关 | 0-3 几乎没有回答该问题所需的信息

### 4. 要点覆盖率 key_points_coverage（0-10）
把关键要点**逐条**与回答对照，只数"意思到位没有"，不数"表述像不像"：
- 意思覆盖判定：要点被同义表述、更口语的说法、更具体的展开覆盖 → 算覆盖；
  顺序不同、拆开说、合并说都不影响
- 不计分、绝不扣分：舞台指令、修辞与文学比喻本身、礼貌性收尾
- 例证豁免：缺失要点若只是对**已覆盖观点**的举例、比喻或类比补充（如"钻石因共识才有价值"
  "像石头从斜坡滚落"），视为已覆盖，不扣分；missed_points 只列独立要点
- 算漏：独立的新观点、新事实、新论断（如"人脉的本质是价值交换"）未在回答中出现任何等价表述
- 锚点：9-10 独立要点全部覆盖（例证类缺失不算漏） | 7-8 缺 1-2 个独立要点 | 4-6 缺 3 个及以上独立要点，或核心要点缺失 | 0-3 绝大多数要点未覆盖

### 5. 口吻一致性 voice_consistency（0-10）
评「这个回答像不像语料里的那个人在说话」。风格参照 = 上下文语料中人物的**第一人称原话
与论述方式**（用词习惯、比喻偏好、立论方式、立场），以及人物的时代与身份设定。
与忠实度的分工：忠实度管「说的事实有没有依据」，本维管「说话的方式像不像本人」——
同一句话可以既忠实又不像本人，两维独立打分。
- 不参与评分：舞台指令与动作描写本身（如 *荣格轻轻叩了叩桌面*）、开场白/收尾客套、
  角色语气词——这些是演出形式，其余四维已豁免
- 算跳出角色（计入 out_of_character）：①现代咨询师/AI 助手腔（如"我们可以这样思考"
  "建议您""希望对你有帮助"）；②人物时代或身份不可能出现的概念与工具；③通用助手式
  表达（列要点模板腔、"首先其次最后"的教程腔）；④立场与语料中人物的公开立场相反
- 锚点：9-10 语感、用词与比喻方式与语料原话如出一手，第一人称经验表述与语料记载一致，
  挑不出跳角色之处 | 7-8 口吻与立场总体一致、专业内容转述准确，但语气偏平、缺少人物
  的标志性表达方式 | 4-6 有 1-2 处跳出角色（AI 腔/现代腔/立场漂移），人物感时有时无 |
  0-3 通篇 AI 助手腔，人物只剩名字

## 输出格式（严格 JSON，不要输出 JSON 以外的任何文字）
{{
    "faithfulness": {{"score": 0-10整数, "reason": "实质论断核查结论（风格不参与）", "unsupported_claims": ["无依据论断，没有则空列表"]}},
    "relevancy": {{"score": 0-10整数, "reason": "简短理由"}},
    "context_precision": {{"score": 0-10整数, "reason": "简短理由"}},
    "key_points_coverage": {{"score": 0-10整数, "reason": "简短理由", "covered_points": [], "missed_points": []}},
    "voice_consistency": {{"score": 0-10整数, "reason": "口吻一致性简评", "out_of_character": ["跳出角色的表述，没有则空列表"]}}
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
            print(f"     判官调用失败（第 {i + 1}/{attempts} 次）: {str(e)[:60]}")
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
        print(f"     判官输出无法解析（第 {i + 1}/{attempts} 次），重试…")
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
    v = sub("voice_consistency")
    return {
        "faithfulness": metric(f),
        "relevancy": metric(r),
        "context_precision": metric(c),
        "key_points_coverage": metric(k),
        "voice_consistency": metric(v),
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
            "voice_reason": v.get("reason", "判官输出无法解析"),
            "voice_out_of_character": v.get("out_of_character", []),
            "voice_parse": "ok" if v else "failed",
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
            return {"answer": f"ERROR: HTTP {response.status_code}", "contexts": [], "retrieval_trace": {}}

        data = response.json()
        return {
            "answer": data.get("answer", ""),
            "contexts": data.get("contexts", []),
            "retrieval_trace": data.get("retrieval_trace", {}),
        }


# ============================================================
# 主评估流程
# ============================================================
def _save_progress(results: list) -> None:
    """中途落盘：每题完成即写。

    110 题连跑要 1.5-3 小时，中途被中断（手动停 / 崩溃 / 断电）不应丢掉
    已完成的题。只保证 results 可被断点续跑读取，summary 留到结尾统一写。
    """
    path = Path(__file__).parent / "eval_results.json"
    try:
        path.write_text(
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"     [落盘失败] {e}")


async def run_evaluation(difficulty: str = None, limit: int = None, pace_sec: float = 0.0):
    """运行完整评估"""
    print("=" * 60)
    print("RAG 评估开始")
    print("=" * 60)

    # 获取测试集
    dataset = get_dataset_by_difficulty(difficulty) if difficulty else load_full_dataset()
    if limit:
        dataset = dataset[:limit]

    print(f"\n测试集大小: {len(dataset)} 个问题")
    if difficulty:
        print(f"难度筛选: {difficulty}")

    # 初始化评判 LLM
    llm = get_judge_llm()
    print(f"评判模型: {settings.llm_model}")

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
                print(f"断点续跑：已有 {len(done_questions)} 题有效结果，本次只补缺口")
        except Exception as e:
            print(f"断点文件加载失败，从头评估: {e}")
    total_start = time.time()

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        key_points = item["key_points"]
        difficulty = item["difficulty"]

        print(f"\n[{i}/{len(dataset)}] {question[:40]}...")
        print(f"     难度: {difficulty}")

        if question in done_questions:
            print("     已有结果，跳过（断点续跑）")
            continue

        # 获取系统回答（扩充数据集带 character 字段；内置数据集默认 jung）
        start = time.time()
        try:
            system_response = await get_system_answer(question, character=item.get("character", "jung"))
        except Exception as e:
            # 管线超时/服务异常只跳过该题，不炸整场（2026-09-10：ReadTimeout 曾炸掉全场评估）
            print(f"     管线调用异常，跳过该题: {type(e).__name__}: {str(e)[:80]}")
            continue
        answer = system_response["answer"]
        contexts = system_response["contexts"]
        elapsed = time.time() - start

        print(f"     回答耗时: {elapsed:.1f}s")
        print(f"     回答长度: {len(answer)} 字符")

        if not answer or answer.startswith("ERROR"):
            print(f"     获取回答失败: {answer}")
            continue

        # 评估回答（单题容错：Judge LLM 走外部中转，偶发异常不应炸掉整个评估）
        eval_start = time.time()
        try:
            eval_result = await evaluate_single(llm, question, answer, contexts, key_points)
        except Exception as e:
            print(f"     单题评估异常，跳过该题: {e}")
            continue
        eval_elapsed = time.time() - eval_start

        print(f"     评估耗时: {eval_elapsed:.1f}s")
        print(f"     评分: 忠实度={eval_result['faithfulness']}, "
              f"相关性={eval_result['relevancy']}, "
              f"上下文={eval_result['context_precision']}, "
              f"要点={eval_result['key_points_coverage']}, "
              f"口吻={eval_result['voice_consistency']}")

        results.append({
            "question": question,
            "difficulty": difficulty,
            "answer": answer,
            "ground_truth": item["ground_truth"],
            "key_points": key_points,
            "contexts": contexts,  # 保存检索上下文：支持事后用不同判规对同一批答案重评
            # 过程记录（每路召回名次等）：判官不用，只给 scripts/analyze_injection.py 归因
            "retrieval_trace": system_response.get("retrieval_trace", {}),
            "evaluation": eval_result,
            "timing": {
                "answer_seconds": elapsed,
                "eval_seconds": eval_elapsed,
            }
        })

        # 每题落盘：中途中断不丢已完成的结果（见 _save_progress）
        _save_progress(results)
        done_n = len([r for r in results if r.get("evaluation")])
        print(f"     [进度] 已落盘 {done_n} 题 → tests/eval_results.json")

        # 题间节流：每题 2 次 LLM 调用（1 生成 + 1 判官五维合一），连打会触发
        # 中转服务的 rpm/tpm 配额（429），整轮报废。需要时用 --pace-sec 控制节奏
        if pace_sec > 0:
            await asyncio.sleep(pace_sec)

    total_elapsed = time.time() - total_start

    # 计算汇总统计（None 安全：判官彻底失败的指标不计入平均，单独计数）
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    if not results:
        print("没有有效的评估结果")
        return

    METRICS = ["faithfulness", "relevancy", "context_precision", "key_points_coverage", "voice_consistency"]
    METRIC_NAMES = {
        "faithfulness": "忠实度 (Faithfulness)",
        "relevancy": "回答相关性 (Relevancy)",
        "context_precision": "上下文精确度 (Context)",
        "key_points_coverage": "要点覆盖率 (Key Points)",
        "voice_consistency": "口吻一致性 (Voice)",
    }

    def avg_of(metric: str) -> tuple[Optional[float], int]:
        vals = [r["evaluation"][metric] for r in results if r["evaluation"].get(metric) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    avgs: dict[str, Optional[float]] = {}
    judged: dict[str, int] = {}
    n = len(results)
    print(f"\n总体指标（共 {n} 个问题，括号内为该指标的有效评判数）:")
    for m in METRICS:
        avg, cnt = avg_of(m)
        avgs[m], judged[m] = avg, cnt
        label = METRIC_NAMES[m]
        if avg is None:
            print(f"   • {label:<28} 无有效评判")
        else:
            note = "" if cnt == n else f"（{n - cnt} 题判官输出无法解析，未计入）"
            print(f"   • {label:<28} {avg:.1f}/10  [{cnt}/{n}]{note}")

    failed_metrics = sum(n - judged[m] for m in METRICS)
    if failed_metrics:
        print(f"\n   共 {failed_metrics} 个指标因判官输出无法解析被剔除"
              f"（旧版会静默记 5 分污染平均分——那不是评分，是故障）")

    # 按难度分组统计（None 安全：分母只数有效值）
    print(f"\n按难度分组:")
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
    print(f"\n耗时统计:")
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

    print(f"\n详细结果已保存到: {output_path}")

    # 输出改进建议（仅在有效评判充足时给出，避免被解析失败误导）
    print("\n" + "=" * 60)
    print("改进建议")
    print("=" * 60)

    def warn_if_low(metric: str, msg: str) -> None:
        avg, cnt = avg_of(metric)
        if avg is not None and cnt >= max(3, n // 2) and avg < 7:
            print(f"   {msg}")

    warn_if_low("faithfulness", "忠实度偏低：回答可能包含幻觉，建议优化 prompt 或增加检索量")
    warn_if_low("relevancy", "相关性偏低：回答可能跑题，建议优化 prompt 聚焦问题")
    warn_if_low("context_precision", "上下文精确度偏低：检索质量需要优化，建议调整 chunk_size 或 reranker")
    warn_if_low("key_points_coverage", "要点覆盖率偏低：回答不够完整，建议增加 retrieval_top_k")

    if (avgs["faithfulness"] or 0) >= 8 and (avgs["relevancy"] or 0) >= 8:
        print("   整体表现良好！可以考虑增加更多测试问题来验证稳定性")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="按难度筛选")
    parser.add_argument("--limit", type=int, help="限制评估问题数量")
    parser.add_argument("--pace-sec", type=float, default=0.0, help="题间等待秒数（防中转 429 限流）")
    args = parser.parse_args()

    asyncio.run(run_evaluation(difficulty=args.difficulty, limit=args.limit, pace_sec=args.pace_sec))
