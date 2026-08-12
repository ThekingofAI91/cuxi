"""
验证 persona/query 按角色切换检索库修复（8001 独立端口）：
1. character_id=adler 的请求应检索 persona_adler（37613 条），而非 persona_jung（109758 条）
   （通过服务端日志 "文档库中共 N 条文档片段" 确认）
2. 记录首 token / 总时长
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8001"


def read_sse(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    events = []
    t_start = time.time()
    t_first_token = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            if data.get("type") == "token" and t_first_token is None:
                t_first_token = time.time() - t_start
            events.append(data)
    except Exception as e:
        events.append({"type": "stream_error", "content": str(e)})
    total = time.time() - t_start
    return events, {
        "total": round(total, 1),
        "first_token": round(t_first_token, 3) if t_first_token is not None else None,
    }


print(">> 1. adler SSE 请求（应检索 persona_adler 37613 条）...")
events, timings = read_sse(
    "/persona/query",
    {
        "query": "我总是害怕被拒绝所以不敢表白，怎么办？",
        "character_id": "adler",
        "session_id": "verify_adler_fix",
    },
)
print(f"  adler total={timings['total']}s first_token={timings['first_token']}s")

print(">> 2. jung SSE 请求（应检索 persona_jung 109758 条）...")
events2, timings2 = read_sse(
    "/persona/query",
    {
        "query": "我总是害怕被拒绝所以不敢表白，怎么办？",
        "character_id": "jung",
        "session_id": "verify_jung_fix",
    },
)
print(f"  jung total={timings2['total']}s first_token={timings2['first_token']}s")

result = next((e for e in events if e.get("type") in ("result", "questions")), None)
head = (result.get("content") or "")[:150] if result else "NO_RESULT"
print(f"adler 回答开头: {head}")
result2 = next((e for e in events2 if e.get("type") in ("result", "questions")), None)
head2 = (result2.get("content") or "")[:150] if result2 else "NO_RESULT"
print(f"jung 回答开头: {head2}")

with open("tests/_verify_adler_fix.txt", "w", encoding="utf-8") as f:
    f.write(f"adler: {timings}\n")
    f.write(f"jung: {timings2}\n")
    f.write(f"adler_head: {head}\n")
    f.write(f"jung_head: {head2}\n")

print("done -> tests/_verify_adler_fix.txt")
