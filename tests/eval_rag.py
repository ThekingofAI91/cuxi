"""
RAG 评估脚本 - LLM-as-Judge
使用 DeepSeek 作为评判模型，评估 RAG 系统的回答质量

评估指标：
1. Faithfulness（忠实度）：回答是否基于检索到的上下文
2. Answer Relevancy（回答相关性）：回答是否切题
3. Context Precision（上下文精确度）：检索的上下文是否与问题相关
4. Key Points Coverage（要点覆盖率）：回答是否覆盖了关键要点
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.llm import get_chat_llm
from src.core.llm_json import extract_json

from tests.eval_dataset import EVAL_DATASET, get_dataset_by_difficulty


# ============================================================
# 评估用 LLM
# ============================================================
def get_judge_llm() -> ChatOpenAI:
    """获取评判用 LLM"""
    return get_chat_llm(
        temperature=0.1,  # 低温度，评判更稳定
        max_tokens=1000,
    )


# ============================================================
# 评估 Prompt
# ============================================================
FAITHFULNESS_PROMPT = """你是一个严格的评估专家。请评估以下回答是否基于提供的上下文信息。

## 问题
{question}

## 检索到的上下文
{contexts}

## 系统的回答
{answer}

## 评估标准
- 回答中的事实性陈述是否能在上下文中找到依据？
- 是否有回答中提到了但上下文中没有的信息（幻觉）？
- 回答是否忠实于原始材料？

## 输出格式
请输出 JSON 格式：
{{
    "score": 0-10 的整数分数,
    "reason": "简短的评判理由"
}}
"""

RELEVANCY_PROMPT = """你是一个严格的评估专家。请评估以下回答是否与问题相关。

## 问题
{question}

## 系统的回答
{answer}

## 评估标准
- 回答是否直接回应了问题？
- 是否有跑题或无关内容？
- 回答是否切中要害？

## 输出格式
请输出 JSON 格式：
{{
    "score": 0-10 的整数分数,
    "reason": "简短的评判理由"
}}
"""

CONTEXT_PRECISION_PROMPT = """你是一个严格的评估专家。请评估检索到的上下文是否与问题相关。

## 问题
{question}

## 检索到的上下文
{contexts}

## 评估标准
- 上下文是否包含回答问题所需的信息？
- 上下文中是否有大量无关内容？
- 检索质量如何？

## 输出格式
请输出 JSON 格式：
{{
    "score": 0-10 的整数分数,
    "reason": "简短的评判理由"
}}
"""

KEY_POINTS_PROMPT = """你是一个严格的评估专家。请评估回答是否覆盖了所有关键要点。

## 问题
{question}

## 关键要点
{key_points}

## 系统的回答
{answer}

## 评估标准
- 每个关键要点是否在回答中被提及或体现？
- 要点的表述是否准确？

## 输出格式
请输出 JSON 格式：
{{
    "score": 0-10 的整数分数,
    "covered_points": ["已覆盖的要点1", "已覆盖的要点2"],
    "missed_points": ["未覆盖的要点1"],
    "reason": "简短的评判理由"
}}
"""


# ============================================================
# 评估函数
# ============================================================
async def evaluate_single(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    contexts: list[str],
    key_points: list[str],
) -> dict:
    """评估单个问题的回答质量"""
    contexts_text = "\n\n---\n\n".join(contexts) if contexts else "（无上下文）"
    key_points_text = "\n".join(f"- {kp}" for kp in key_points)

    # 并行评估四个指标
    tasks = [
        llm.ainvoke([("user", FAITHFULNESS_PROMPT.format(
            question=question, contexts=contexts_text, answer=answer
        ))]),
        llm.ainvoke([("user", RELEVANCY_PROMPT.format(
            question=question, answer=answer
        ))]),
        llm.ainvoke([("user", CONTEXT_PRECISION_PROMPT.format(
            question=question, contexts=contexts_text
        ))]),
        llm.ainvoke([("user", KEY_POINTS_PROMPT.format(
            question=question, key_points=key_points_text, answer=answer
        ))]),
    ]

    results = await asyncio.gather(*tasks)

    # 解析结果
    def parse_json_response(text: str) -> dict:
        try:
            obj = extract_json(text)  # 兼容纯 JSON / ```json 代码块 / 混在文本中的平衡大括号
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {"score": 5, "reason": "解析失败"}

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

    faithfulness = parse_json_response(results[0].content)
    relevancy = parse_json_response(results[1].content)
    context_precision = parse_json_response(results[2].content)
    key_points_result = parse_json_response(results[3].content)

    return {
        "faithfulness": norm_score(faithfulness.get("score", 5)),
        "relevancy": norm_score(relevancy.get("score", 5)),
        "context_precision": norm_score(context_precision.get("score", 5)),
        "key_points_coverage": norm_score(key_points_result.get("score", 5)),
        "details": {
            "faithfulness_reason": faithfulness.get("reason", ""),
            "relevancy_reason": relevancy.get("reason", ""),
            "context_precision_reason": context_precision.get("reason", ""),
            "covered_points": key_points_result.get("covered_points", []),
            "missed_points": key_points_result.get("missed_points", []),
        }
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

    async with httpx.AsyncClient(timeout=120.0) as client:
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
async def run_evaluation(difficulty: str = None, limit: int = None):
    """运行完整评估"""
    print("=" * 60)
    print("🎯 RAG 评估开始")
    print("=" * 60)

    # 获取测试集
    dataset = get_dataset_by_difficulty(difficulty)
    if limit:
        dataset = dataset[:limit]

    print(f"\n📋 测试集大小: {len(dataset)} 个问题")
    if difficulty:
        print(f"📊 难度筛选: {difficulty}")

    # 初始化评判 LLM
    llm = get_judge_llm()
    print(f"🤖 评判模型: {settings.llm_model}")

    # 存储结果
    results = []
    total_start = time.time()

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        key_points = item["key_points"]
        difficulty = item["difficulty"]

        print(f"\n[{i}/{len(dataset)}] 📝 {question[:40]}...")
        print(f"     难度: {difficulty}")

        # 获取系统回答
        start = time.time()
        system_response = await get_system_answer(question)
        answer = system_response["answer"]
        contexts = system_response["contexts"]
        elapsed = time.time() - start

        print(f"     ⏱️  回答耗时: {elapsed:.1f}s")
        print(f"     📄 回答长度: {len(answer)} 字符")

        if not answer or answer.startswith("ERROR"):
            print(f"     ❌ 获取回答失败: {answer}")
            continue

        # 评估回答
        eval_start = time.time()
        eval_result = await evaluate_single(llm, question, answer, contexts, key_points)
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
            "evaluation": eval_result,
            "timing": {
                "answer_seconds": elapsed,
                "eval_seconds": eval_elapsed,
            }
        })

    total_elapsed = time.time() - total_start

    # 计算汇总统计
    print("\n" + "=" * 60)
    print("📊 评估结果汇总")
    print("=" * 60)

    if not results:
        print("❌ 没有有效的评估结果")
        return

    # 总体平均分
    avg_faithfulness = sum(r["evaluation"]["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["evaluation"]["relevancy"] for r in results) / len(results)
    avg_context = sum(r["evaluation"]["context_precision"] for r in results) / len(results)
    avg_keypoints = sum(r["evaluation"]["key_points_coverage"] for r in results) / len(results)

    print(f"\n📈 总体指标（共 {len(results)} 个问题）:")
    print(f"   • 忠实度 (Faithfulness):     {avg_faithfulness:.1f}/10")
    print(f"   • 回答相关性 (Relevancy):    {avg_relevancy:.1f}/10")
    print(f"   • 上下文精确度 (Context):    {avg_context:.1f}/10")
    print(f"   • 要点覆盖率 (Key Points):   {avg_keypoints:.1f}/10")

    # 按难度分组统计
    print(f"\n📊 按难度分组:")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            diff_avg = sum(
                r["evaluation"]["faithfulness"] +
                r["evaluation"]["relevancy"] +
                r["evaluation"]["context_precision"] +
                r["evaluation"]["key_points_coverage"]
                for r in diff_results
            ) / (len(diff_results) * 4)
            print(f"   • {diff:6s}: {diff_avg:.1f}/10 ({len(diff_results)} 题)")

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
                "avg_faithfulness": round(avg_faithfulness, 2),
                "avg_relevancy": round(avg_relevancy, 2),
                "avg_context_precision": round(avg_context, 2),
                "avg_key_points_coverage": round(avg_keypoints, 2),
                "total_time_seconds": round(total_elapsed, 1),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存到: {output_path}")

    # 输出改进建议
    print("\n" + "=" * 60)
    print("💡 改进建议")
    print("=" * 60)

    if avg_faithfulness < 7:
        print("   ⚠️  忠实度偏低：回答可能包含幻觉，建议优化 prompt 或增加检索量")
    if avg_relevancy < 7:
        print("   ⚠️  相关性偏低：回答可能跑题，建议优化 prompt 聚焦问题")
    if avg_context < 7:
        print("   ⚠️  上下文精确度偏低：检索质量需要优化，建议调整 chunk_size 或 reranker")
    if avg_keypoints < 7:
        print("   ⚠️  要点覆盖率偏低：回答不够完整，建议增加 retrieval_top_k")

    if avg_faithfulness >= 8 and avg_relevancy >= 8:
        print("   ✅ 整体表现良好！可以考虑增加更多测试问题来验证稳定性")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="按难度筛选")
    parser.add_argument("--limit", type=int, help="限制评估问题数量")
    args = parser.parse_args()

    asyncio.run(run_evaluation(difficulty=args.difficulty, limit=args.limit))
