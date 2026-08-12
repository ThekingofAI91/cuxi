"""单请求 SSE 计时：首 token / 各事件时间线"""
import json
import time
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8000/persona/query",
    data=json.dumps({
        "query": "我男朋友总是不回我消息，我是不是应该分手？",
        "character_id": "adler",
        "session_id": "verify3_solo",
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
t_first_token = None
t_first_result = None
events = []
resp = urllib.request.urlopen(req, timeout=300)
for raw_line in resp:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line.startswith("data: "):
        continue
    data = json.loads(line[6:])
    now = time.time() - t0
    if data.get("type") == "token" and t_first_token is None:
        t_first_token = now
    if data.get("type") in ("result", "questions") and t_first_result is None:
        t_first_result = now
    events.append((data.get("type"), round(now, 2)))

total = time.time() - t0
print(f"total: {round(total, 1)}s, first_token: {round(t_first_token, 1)}s, first_result: {round(t_first_result, 1)}s")
# 汇总事件时间线（去重连续同类）
timeline = []
for t, ts in events:
    if not timeline or timeline[-1][0] != t:
        timeline.append((t, ts))
print("timeline:", timeline)
