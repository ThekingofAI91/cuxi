"""
load_test.py — 并发压测：打真实 /persona/eval_query（检索+生成全链路），测 p50/p95 延迟

用法（先启动服务）：
    .venv/Scripts/python.exe scripts/load_test.py --concurrency 20 --total 40
    .venv/Scripts/python.exe scripts/load_test.py --concurrency 50 --total 100 --timeout 180

注意：真实调用 LLM API，注意上游配额（429 会在结果中体现）。
评估端点无会话历史，测的是"单轮检索+生成"的纯链路延迟。
"""

import argparse
import asyncio
import statistics
import time

import httpx

QUESTIONS = [
    "什么是集体无意识？",
    "荣格和弗洛伊德为什么决裂？",
    "什么是积极想象？",
    "阴影在心理学中代表什么？",
    "什么是共时性？",
    "阿尼玛和阿尼姆斯有什么区别？",
    "荣格如何理解梦？",
    "什么是人格面具？",
    "心理类型有哪几种？",
    "什么是自性化？",
]


async def one(client: httpx.AsyncClient, sem: asyncio.Semaphore, q: str, timeout: float, results: list):
    async with sem:
        t0 = time.perf_counter()
        err = ""
        try:
            r = await client.post(
                "http://localhost:8000/persona/eval_query",
                json={"query": q, "character_id": "jung"},
                timeout=timeout,
            )
            if r.status_code != 200:
                err = f"HTTP {r.status_code}"
            else:
                data = r.json()
                if not data.get("answer"):
                    err = "空回答"
        except Exception as e:
            err = str(e)[:60]
        dt = (time.perf_counter() - t0) * 1000
        results.append({"ms": dt, "err": err})
        flag = "❌ " + err if err else "✅"
        print(f"  [{len(results):>3}] {dt / 1000:6.1f}s {flag} {q[:20]}")


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0
    s = sorted(values)
    k = min(int(len(s) * p), len(s) - 1)
    return s[k]


async def main() -> None:
    parser = argparse.ArgumentParser(description="并发压测")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--total", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    results: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.total)]

    print(f"[load] 并发={args.concurrency} 总请求={args.total} 超时={args.timeout}s")
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, sem, q, args.timeout, results) for q in questions))
    wall = time.perf_counter() - t0

    ok = [r["ms"] for r in results if not r["err"]]
    errs = [r for r in results if r["err"]]
    print("\n" + "=" * 62)
    print(f"[load] 压测汇总：总 {len(results)} | 成功 {len(ok)} | 失败 {len(errs)} | 墙钟 {wall:.0f}s")
    print(f"       吞吐 ≈ {len(ok) / wall:.2f} 请求/秒")
    if ok:
        print(f"       延迟  p50={statistics.median(ok) / 1000:.1f}s  "
              f"p95={pct(ok, 0.95) / 1000:.1f}s  max={max(ok) / 1000:.1f}s  mean={statistics.mean(ok) / 1000:.1f}s")
    err_kinds: dict[str, int] = {}
    for r in errs:
        err_kinds[r["err"]] = err_kinds.get(r["err"], 0) + 1
    for k, v in sorted(err_kinds.items(), key=lambda x: -x[1]):
        print(f"       失败 {v}× {k}")


if __name__ == "__main__":
    asyncio.run(main())
