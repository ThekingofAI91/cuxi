"""对当前五分制结果中判官评分缺失（None）的题重新评判。

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

from src.core.evaluation import (
    METRICS, SCORE_MAX, SCORE_SCALE, get_eval_result_path, require_current_score_scale,
)

RESULT_PATH = get_eval_result_path()


def recompute_summary(data: dict) -> None:
    require_current_score_scale(data, RESULT_PATH)
    data["score_scale"] = SCORE_SCALE
    data.setdefault("summary", {})
    recs = data["results"]
    data["summary"]["total_questions"] = len(recs)
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
    data["summary"]["avg_voice_consistency"] = avgs["voice_consistency"]
    data["summary"]["judged_counts"] = judged


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=12.0, help="题间等待秒数（防中转限流）")
    args = parser.parse_args()

    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    require_current_score_scale(data, RESULT_PATH)
    targets = [
        r for r in data["results"]
        if any(r["evaluation"].get(m) is None for m in METRICS)
    ]
    print(f"需重新评判: {len(targets)} 题", flush=True)
    if not targets:
        return

    from tests.eval_rag import evaluate_single, get_judge_llm

    llm = get_judge_llm()
    ok = 0
    for i, r in enumerate(targets, 1):
        try:
            ev = await evaluate_single(llm, r["question"], r["answer"], r["contexts"], r["key_points"])
            previous = r["evaluation"]
            missing = [m for m in METRICS if previous.get(m) is None]
            recovered = [m for m in missing if ev.get(m) is not None]
            if recovered:
                details = previous.setdefault("details", {})
                detail_prefixes = {"key_points_coverage": "key_points", "voice_consistency": "voice"}
                for m in recovered:
                    previous[m] = ev[m]
                    prefix = detail_prefixes.get(m, m)
                    for key, value in ev.get("details", {}).items():
                        if key.startswith(prefix + "_") or (
                            m == "key_points_coverage" and key in ("covered_points", "missed_points")
                        ):
                            details[key] = value
                ok += 1
                print(
                    f"[{i}/{len(targets)}] {r['question'][:24]}… （1–{SCORE_MAX} 分）"
                    f"忠{previous.get('faithfulness')} 相{previous.get('relevancy')} "
                    f"检{previous.get('context_precision')} 要{previous.get('key_points_coverage')} "
                    f"口吻{previous.get('voice_consistency')}",
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
        f"\n完成（1–{SCORE_MAX} 分）：新回填 {ok}/{len(targets)} 题 | "
        f"忠{s['avg_faithfulness']} 相{s['avg_relevancy']} 检{s['avg_context_precision']} 要{s['avg_key_points_coverage']} "
        f"口吻{s['avg_voice_consistency']} "
        f"({s['judged_counts']})",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
