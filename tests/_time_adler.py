# -*- coding: utf-8 -*-
"""
对比 adler / jung 在 8001 上的检索耗时（首 token 时间 + 总时间）。
场景：
 A. adler 首次请求（可能触发 BM25 缓存加载/索引构建）
 B. adler 第二次请求（应命中缓存）
 C. jung 对照请求
"""
import asyncio
import json
import time
import httpx

BASE = "http://127.0.0.1:8001"
OUT = "tests/_time_adler.txt"


async def run_query(session_id: str, character_id: str, query: str) -> dict:
    body = {"query": query, "session_id": session_id, "character_id": character_id}
    t0 = time.time()
    first_token = None
    total = None
    got_result = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        async with client.stream("POST", f"{BASE}/persona/query", json=body) as resp:
            buffer = ""
            async for chunk in resp.aiter_bytes():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    part, buffer = buffer.split("\n\n", 1)
                    if not part.strip().startswith("data: "):
                        continue
                    try:
                        data = json.loads(part.strip()[6:])
                    except Exception:
                        continue
                    if data["type"] == "token" and first_token is None:
                        first_token = round(time.time() - t0, 2)
                    if data["type"] in ("result", "questions"):
                        got_result = True
                        total = round(time.time() - t0, 2)
    return {"first_token_s": first_token, "total_s": total, "got_result": got_result}


async def main():
    lines = []
    q = "失恋后总是走不出来，该怎么放下？"

    print(">> 场景 A：adler 首次请求...")
    a = await run_query("time-adler-a-001", "adler", q)
    print(f"    first_token={a['first_token_s']}s total={a['total_s']}s got_result={a['got_result']}")
    lines.append(("A:adler首次", a))

    print(">> 场景 B：adler 第二次请求（应命中缓存）...")
    b = await run_query("time-adler-b-001", "adler", q)
    print(f"    first_token={b['first_token_s']}s total={b['total_s']}s got_result={b['got_result']}")
    lines.append(("B:adler第二次", b))

    print(">> 场景 C：jung 对照...")
    c = await run_query("time-jung-c-001", "jung", q)
    print(f"    first_token={c['first_token_s']}s total={c['total_s']}s got_result={c['got_result']}")
    lines.append(("C:jung对照", c))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in lines))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
