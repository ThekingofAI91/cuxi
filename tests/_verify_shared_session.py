# -*- coding: utf-8 -*-
"""
复现"两人同时使用，前端收不到回答"问题（8001 独立端口）：
场景 A：两个标签页共享 localStorage -> 相同 session_id 并发（同一浏览器）
场景 B：两台设备 -> 不同 session_id 并发（对照）
检查每个请求是否都收到完整 result、是否发生串扰/丢失。
"""
import asyncio
import json
import time
import httpx

BASE = "http://127.0.0.1:8001"
OUT = "tests/_verify_shared_session.txt"


async def sse_request(tag: str, session_id: str, query: str, results: dict):
    body = {"query": query, "session_id": session_id, "character_id": "jung"}
    t0 = time.time()
    first_token = None
    token_count = 0
    result_content = None
    error_msg = None
    events = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            async with client.stream("POST", f"{BASE}/persona/query", json=body) as resp:
                if resp.status_code != 200:
                    error_msg = f"HTTP {resp.status_code}"
                else:
                    buffer = ""
                    async for chunk in resp.aiter_bytes():
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n\n" in buffer:
                            part, buffer = buffer.split("\n\n", 1)
                            line = part.strip()
                            if not line.startswith("data: "):
                                continue
                            try:
                                data = json.loads(line[6:])
                            except Exception:
                                continue
                            events.append(data["type"])
                            if data["type"] == "token":
                                token_count += 1
                                if first_token is None:
                                    first_token = time.time() - t0
                            elif data["type"] == "result":
                                result_content = data.get("content") or ""
                            elif data["type"] == "error":
                                error_msg = data.get("content")
    except Exception as e:
        error_msg = f"EXC: {e!r}"
    total = time.time() - t0
    results[tag] = {
        "total": round(total, 2),
        "first_token": round(first_token, 2) if first_token else None,
        "token_count": token_count,
        "events": events,
        "result_len": len(result_content) if result_content else 0,
        "result_head": (result_content or "")[:40],
        "error": error_msg,
    }


async def main():
    print(">> 场景 A：相同 session_id 并发（模拟同一浏览器两个标签页）...")
    results = {}
    shared = "shared-session-test-001"
    await asyncio.gather(
        sse_request("A_q1", shared, "我总是害怕被拒绝所以不敢表白，怎么办？", results),
        sse_request("A_q2", shared, "我和男朋友吵架后他总是冷暴力，该怎么打破局面？", results),
    )
    for k in ("A_q1", "A_q2"):
        r = results.get(k)
        print(f"  {k}: total={r['total']}s first_token={r['first_token']}s "
              f"tokens={r['token_count']} result_len={r['result_len']} err={r['error']}")
        print(f"     head: {r['result_head']}")

    print(">> 场景 B：不同 session_id 并发（模拟两台设备）...")
    results2 = {}
    await asyncio.gather(
        sse_request("B_u1", "device-user-111", "我总是害怕被拒绝所以不敢表白，怎么办？", results2),
        sse_request("B_u2", "device-user-222", "我和男朋友吵架后他总是冷暴力，该怎么打破局面？", results2),
    )
    for k in ("B_u1", "B_u2"):
        r = results2.get(k)
        print(f"  {k}: total={r['total']}s first_token={r['first_token']}s "
              f"tokens={r['token_count']} result_len={r['result_len']} err={r['error']}")
        print(f"     head: {r['result_head']}")

    lines = ["== 场景 A：相同 session_id 并发 ==", json.dumps(results, ensure_ascii=False, indent=1),
             "== 场景 B：不同 session_id 并发 ==", json.dumps(results2, ensure_ascii=False, indent=1)]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
