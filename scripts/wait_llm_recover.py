"""LLM 中转站探活：恢复后返回 0，供外层脚本拉起评估。

用法：
    python scripts/wait_llm_recover.py --max-hours 6
每 10 分钟用与评估完全相同的链路（openai client + settings）发一次 1-token 探测。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def probe() -> bool:
    from openai import OpenAI

    from src.core.config import settings

    client = OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=20,
        max_retries=0,
    )
    try:
        r = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        ok = bool(r.choices)
        print(f"[{time.strftime('%H:%M:%S')}] {'恢复' if ok else '异常响应'}", flush=True)
        return ok
    except Exception as e:
        print(
            f"[{time.strftime('%H:%M:%S')}] 未恢复: {type(e).__name__}: {str(e)[:120]}",
            flush=True,
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=6.0, help="最长等待小时数")
    parser.add_argument("--interval-sec", type=int, default=600)
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        if probe():
            sys.exit(0)
        time.sleep(args.interval_sec)
    print("等待超时，中转仍未恢复", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
