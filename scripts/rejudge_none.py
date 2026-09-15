"""对 eval_results.json 中判官评分缺失（None）的题重新评判。

背景：管线答案完整落盘，但 judge 阶段撞上中转故障窗口的题记了 None。
本脚本只补 judge（不重跑管线），成功后回填 evaluation 并重算 summary。

用法：python scripts/rejudge_none.py [--interval 12]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULT_PATH = Path(__file__).parent.parent / "tests" / "eval_results.json"
METRICS = ["faithfulness", "relevancy", "context_precision", "key_points_coverage"]


def recompute_summary(data: dict) -> None:
    recs = data["results"]
    judged = {}
    avgs = {}
    for m in METRICS:
        vals = [r["evaluation"][m] for r in recs if r["evaluation"].get(m) is not None]
        judged[m] = len(vals)
        avgs[m] = round(sum(vals) / len(vals), 2) if vals else None
    data["summary"]["avg_faithfulness"] = avgs["faithfulness"]
    data["summary"]["avg_relevancy"] = avgs["relevancy"]
    data["summary"]["avg_context_precision"] = avgs["context_precision"]
    data["summary"]["avg_key_points_coverage"] = avgs["key_points_coverage"]
    data["summary"]["judged_counts"] = judged


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=12.0, help="题间等待秒数（防中转限流）")
    args = parser.parse_args()

    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    targets = [r for r in data["results"] if r["evaluation"].get("faithfulness") is None]
    print(f"需重新评判: {len(targets)} 题", flush=True)
    if not targets:
        return

    from tests.eval_rag import evaluate_single, get_judge_llm

    llm = get_judge_llm()
    ok = 0
    for i, r in enumerate(targets, 1):
        try:
            ev = await evaluate_single(llm, r["question"], r["answer"], r["contexts"], r["key_points"])
            if ev.get("faithfulness") is not None:
                ev["details"] = {**r["evaluation"].get("details", {}), **ev.get("details", {})}
                r["evaluation"] = ev
                ok += 1
                print(
                    f"[{i}/{len(targets)}] {r['question'][:24]}… "
                    f"忠{ev['faithfulness']} 相{ev['relevancy']} 检{ev['context_precision']} 要{ev['key_points_coverage']}",
                    flush=True,
                )
            else:
                print(f"[{i}/{len(targets)}] 判官输出仍无法解析: {r['question'][:24]}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(targets)}] {type(e).__name__}: {str(e)[:70]}", flush=True)
        # 每题落盘一次：中途崩了不丢已回填的结果
        recompute_summary(data)
        RESULT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if i < len(targets):
            await asyncio.sleep(args.interval)

    recompute_summary(data)
    RESULT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    s = data["summary"]
    print(
        f"\n完成：新回填 {ok}/{len(targets)} 题 | "
        f"忠{s['avg_faithfulness']} 相{s['avg_relevancy']} 检{s['avg_context_precision']} 要{s['avg_key_points_coverage']} "
        f"({s['judged_counts']})",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
