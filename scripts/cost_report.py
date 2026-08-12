"""
打印今日 / 本周的成本与错误统计。

用法：python scripts/cost_report.py
"""

import json
import time

from src.core.monitor import get_monitor


def main() -> None:
    monitor = get_monitor()
    report = {
        "today": monitor.summary(monitor.since_start_of_day()),
        "week": monitor.summary(time.time() - 7 * 86400),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
