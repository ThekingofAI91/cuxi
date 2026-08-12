# -*- coding: utf-8 -*-
"""
验证"刷新页面后恢复回答"功能（8001 独立端口）：
1. 发起 persona/query SSE 请求，读到首 token 后立即断开（模拟用户刷新页面）
2. 后端 run_graph 作为后台任务应继续执行并写入 _session_store
3. 轮询 /conversation/pending/{session_id}，应能捞回 final_answer 且 query 匹配
4. 对照：未断开的正常请求不受影响
"""
import asyncio
import json
import time
import httpx

BASE = "http://127.0.0.1:8001"
OUT = "tests/_verify_pending_recovery.txt"


async def disconnect_mid_stream(session_id: str, query: str, results: dict):
    """模拟刷新：SSE 读到首个 token 后直接断开连接"""
    body = {"query": query, "session_id": session_id, "character_id": "jung"}
    t0 = time.time()
    got_token = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
            async with client.stream("POST", f"{BASE}/persona/query", json=body) as resp:
                print(f"    [断开模拟] HTTP {resp.status_code}, 等待首 token...")
                buffer = ""
                async for chunk in resp.aiter_bytes():
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        part, buffer = buffer.split("\n\n", 1)
                        if part.strip().startswith("data: "):
                            try:
                                data = json.loads(part.strip()[6:])
                            except Exception:
                                continue
                            if data["type"] == "token":
                                got_token = True
                                print(f"    [断开模拟] 收到首 token 于 {time.time()-t0:.1f}s，立即断开连接（模拟刷新页面）")
                                return  # 直接断开 SSE
    except Exception as e:
        print(f"    [断开模拟] 断开时异常(预期内): {e!r}")
    finally:
        results["disconnected"] = got_token


async def poll_pending(session_id: str, query: str, results: dict):
    """轮询 pending 端点，等待后端后台任务完成"""
    url = f"{BASE}/conversation/pending/{session_id}"
    t0 = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        while True:
            resp = await client.get(url)
            data = resp.json()
            if data.get("final_answer"):
                results["recovered"] = True
                results["query_match"] = (data["query"] == query)
                results["elapsed"] = round(time.time() - t0, 2)
                results["result_len"] = len(data["final_answer"])
                results["result_head"] = data["final_answer"][:40]
                results["route_history"] = data.get("route_history", [])
                return
            if time.time() - t0 > 120:
                results["recovered"] = False
                results["elapsed"] = round(time.time() - t0, 2)
                return
            await asyncio.sleep(2)


async def normal_request(results: dict):
    """对照：正常完整接收 SSE 的请求不受影响"""
    session_id = "pending-recovery-normal-001"
    query = "我总是害怕被拒绝所以不敢表白，怎么办？"
    body = {"query": query, "session_id": session_id, "character_id": "jung"}
    t0 = time.time()
    result_content = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        async with client.stream("POST", f"{BASE}/persona/query", json=body) as resp:
            buffer = ""
            async for chunk in resp.aiter_bytes():
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    part, buffer = buffer.split("\n\n", 1)
                    if part.strip().startswith("data: "):
                        try:
                            data = json.loads(part.strip()[6:])
                        except Exception:
                            continue
                        if data["type"] == "result":
                            result_content = data.get("content") or ""
    results["normal_ok"] = bool(result_content)
    results["normal_len"] = len(result_content) if result_content else 0
    results["normal_total"] = round(time.time() - t0, 2)


async def main():
    print(">> 场景 1：SSE 中途断开（模拟思考中刷新页面）...")
    session_id = "pending-recovery-test-001"
    query = "我总是害怕被拒绝所以不敢表白，怎么办？"
    r1, r2 = {}, {}
    await disconnect_mid_stream(session_id, query, r1)
    await poll_pending(session_id, query, r2)
    print(f"    断开成功: {r1.get('disconnected')}")
    print(f"    恢复结果: recovered={r2.get('recovered')} query_match={r2.get('query_match')} "
          f"elapsed={r2.get('elapsed')}s result_len={r2.get('result_len')}")
    print(f"    head: {r2.get('result_head')}")
    print(f"    route_history: {r2.get('route_history')}")

    print(">> 场景 2：正常完整请求对照（不受影响）...")
    r3 = {}
    await normal_request(r3)
    print(f"    正常请求: ok={r3.get('normal_ok')} len={r3.get('normal_len')} total={r3.get('normal_total')}s")

    lines = ["== 场景 1：SSE 中途断开恢复 ==", json.dumps({"disconnect": r1, "recover": r2}, ensure_ascii=False, indent=1),
             "== 场景 2：正常请求对照 ==", json.dumps(r3, ensure_ascii=False, indent=1)]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
