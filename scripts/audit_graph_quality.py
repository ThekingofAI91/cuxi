"""
audit_graph_quality.py — 知识图谱质量审计：抽样三元组让 LLM 判"是否忠于证据"

用法（建图完成后）：
    .venv/Scripts/python.exe scripts/audit_graph_quality.py            # 每库抽 20 条
    .venv/Scripts/python.exe scripts/audit_graph_quality.py --sample 30

判官对每条三元组给出 忠实/不忠实/无法判定 + 理由，汇总各库忠实率。
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.retrieval.knowledge_graph import graph_path

AUDIT_PROMPT = """你是知识图谱质量审计员。判断下面这条"实体-关系"三元组是否忠于给定的证据片段。

三元组：{head} --[{relation}]--> {tail}
证据片段：{evidence}

判定标准：
- faithful：证据片段能直接或合理推出该关系
- unsupported：证据片段推不出该关系（三元组可能来自别处的幻觉）
- unclear：证据片段过短/指代不明，无法判定

输出严格 JSON：
{{"verdict": "faithful|unsupported|unclear", "reason": "30 字以内理由"}}"""


async def audit_one(llm, judge, rel: dict) -> dict:
    from src.core.llm import ainvoke_nonempty
    from src.core.llm_json import extract_json

    prompt = AUDIT_PROMPT.format(
        head=rel.get("h", ""), relation=rel.get("r", ""),
        tail=rel.get("t", ""), evidence=(rel.get("e", "") or "（无证据）")[:300],
    )
    for _ in range(2):
        try:
            resp = await ainvoke_nonempty(judge, [("user", prompt)])
            obj = extract_json(getattr(resp, "content", "") or "")
            if isinstance(obj, dict) and obj.get("verdict") in ("faithful", "unsupported", "unclear"):
                return obj
        except Exception:
            pass
        await asyncio.sleep(2)
    return {"verdict": "unknown", "reason": "判官输出解析失败"}


async def main() -> None:
    parser = argparse.ArgumentParser(description="图谱质量审计")
    parser.add_argument("--sample", type=int, default=20, help="每库抽样条数")
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    import asyncio as _a

    from src.core.llm import get_chat_llm

    judge = get_chat_llm(temperature=0.1, max_tokens=200)
    rng = random.Random(42)
    report: dict[str, dict] = {}

    for col in ("persona_jung", "persona_wangyangming", "persona_adler"):
        path = graph_path(col)
        if not path.exists():
            print(f"[audit] ⏭️ {col}: 图谱不存在，跳过")
            continue
        graph = json.loads(path.read_text(encoding="utf-8"))
        rels = graph.get("relations", [])
        sample = rng.sample(rels, min(args.sample, len(rels)))
        print(f"\n[audit] === {col}（共 {len(rels)} 条关系，抽 {len(sample)} 条）===")
        counts = {"faithful": 0, "unsupported": 0, "unclear": 0, "unknown": 0}
        for i, rel in enumerate(sample, 1):
            v = await audit_one(judge, judge, rel)
            counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
            mark = {"faithful": "✅", "unsupported": "❌", "unclear": "❓", "unknown": "⚠️"}[v["verdict"]]
            print(f"  {mark} {rel.get('h','')[:14]} --[{rel.get('r','')[:12]}]--> {rel.get('t','')[:14]} | {v['reason']}")
            if args.delay > 0:
                await _a.sleep(args.delay)
        total = len(sample)
        rate = counts["faithful"] / total if total else 0
        report[col] = {"sampled": total, **counts, "faithful_rate": round(rate, 3)}
        print(f"  → 忠实率 {rate:.0%}（faithful {counts['faithful']}/{total}）")

    print("\n[audit] ===== 汇总 =====")
    for col, r in report.items():
        print(f"  {col}: 忠实率 {r['faithful_rate']:.0%} "
              f"(✅{r['faithful']} ❌{r['unsupported']} ❓{r['unclear']} ⚠️{r['unknown']})")
    out = Path("output/graph_audit.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] 明细已存 {out}")


if __name__ == "__main__":
    asyncio.run(main())
