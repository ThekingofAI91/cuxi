"""
rejudge_faithfulness_legacy.py — 新旧判规成对对比（同一批答案，只换忠实度判规）

用法：跑完 tests/eval_rag.py 后执行
    .venv/Scripts/python.exe scripts/rejudge_faithfulness_legacy.py
读取 tests/eval_results.json 中保存的 (question, answer, contexts)，
用**旧版通用判规**重评忠实度，与新判规得分并排输出。

目的：把「判规变更」的影响从「答案变更」中剥离——答案相同，分数差异全部来自判规。
旧判规特征：无角色扮演语境说明、无风格豁免条款、解析失败默认记 5 分（复刻其行为）。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm import get_chat_llm, ainvoke_nonempty
from src.core.llm_json import extract_json

# 旧版通用忠实度判规快照（2026-08-30 前的原文，逐字保留）
LEGACY_FAITHFULNESS_PROMPT = """你是一个严格的评估专家。请评估以下回答是否基于提供的上下文信息。

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


async def judge_legacy(llm, question: str, answer: str, contexts: list[str], attempts: int = 2):
    """用旧判规评一次（与旧版一致：解析失败记 5 分；外加一次重试以对齐新管线容错）"""
    contexts_text = "\n\n---\n\n".join(contexts) if contexts else "（无上下文）"
    prompt = LEGACY_FAITHFULNESS_PROMPT.format(
        question=question, contexts=contexts_text, answer=answer
    )
    for i in range(attempts):
        resp = await ainvoke_nonempty(llm, [("user", prompt)])
        text = getattr(resp, "content", "") or ""
        try:
            obj = extract_json(text)
            if isinstance(obj, dict) and "score" in obj:
                return int(obj["score"]), str(obj.get("reason", ""))[:60]
        except Exception:
            pass
        print(f"  [legacy] 第 {i + 1}/{attempts} 次解析失败，重试…")
    return 5, "解析失败（旧版会默认记 5 分）"


async def main() -> None:
    path = Path("tests/eval_results.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    print(f"[rejudge] 载入 {len(results)} 题的新判规结果，用旧判规对同一批答案重评忠实度\n")

    llm = get_chat_llm(temperature=0.1, max_tokens=1000)

    new_scores, legacy_scores = [], []
    for i, r in enumerate(results, 1):
        q, a, ctx = r["question"], r["answer"], r.get("contexts", [])
        new = r["evaluation"].get("faithfulness")
        old, reason = await judge_legacy(llm, q, a, ctx)
        if new is not None:
            new_scores.append(new)
            legacy_scores.append(old)
        delta = "" if new is None else f"  Δ{old - new:+d}"
        print(f"[{i:>2}] 新判规={new if new is not None else '无效'} | 旧判规={old} | {q[:24]}{delta}")
        print(f"     旧判管理由: {reason}")
        await asyncio.sleep(4)  # 节流防 429

    if new_scores:
        n = len(new_scores)
        print("\n" + "=" * 60)
        print(f"[rejudge] 同一批 {n} 个有效答案的忠实度对比")
        print(f"  新判规（项目定制）: {sum(new_scores) / n:.2f}")
        print(f"  旧判规（通用规则）: {sum(legacy_scores) / n:.2f}")
        print(f"  判规带来的分差   : {(sum(new_scores) - sum(legacy_scores)) / n:+.2f}")
    else:
        print("[rejudge] 新结果中没有有效忠实度评分可对比")


if __name__ == "__main__":
    asyncio.run(main())
