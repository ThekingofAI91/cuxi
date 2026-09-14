"""带重试的评估包装：中转站间歇性 502 的对策。

逻辑：跑全量评估 → 检查有效题数（faithfulness 已评判数 ≥ 85%）→
达标归档到 output/eval_final_N.json；不达标等 5 分钟重跑，最多 5 次。

用法：python scripts/run_eval_with_retry.py [--total 54]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RES_PATH = ROOT / "tests" / "eval_results.json"


def judged_count() -> tuple[int, int]:
    try:
        summary = json.loads(RES_PATH.read_text(encoding="utf-8"))["summary"]
        return (
            summary.get("judged_counts", {}).get("faithfulness", 0),
            summary.get("total_questions", 0),
        )
    except Exception:
        return 0, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=54)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    threshold = int(args.total * 0.85)
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n===== 第 {attempt}/{args.max_attempts} 次评估 =====", flush=True)
        env = {**__import__("os").environ}
        env.setdefault("EVAL_JUDGE_GAP_SEC", "25")  # 外部可用环境变量覆盖
        subprocess.run(
            [sys.executable, str(ROOT / "tests" / "eval_rag.py"), "--pace-sec", "8"],
            cwd=ROOT,
            env=env,
        )
        judged, total = judged_count()
        print(f">>> 有效题数: {judged}/{total}（达标线 {threshold}）", flush=True)
        if judged >= threshold:
            out = ROOT / "output" / f"eval_final_{total}.json"
            out.write_text(RES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"✅ 达标，已归档 → {out}", flush=True)
            return
        if attempt < args.max_attempts:
            print("中转故障窗口未过，5 分钟后重试…", flush=True)
            time.sleep(300)
    print("❌ 重试耗尽仍未达标", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
