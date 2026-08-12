"""
第二轮验证：SSE 新链路 + 热缓存速度 + 并发

1. 线程 A：POST /persona/query（SSE 流式）→ 记录 total、首 token 时间、event 序列、route_history、result
2. 线程 B：POST /persona/eval_query（热缓存）→ 验证第二次调用显著快于首次 56.1s
3. 主线程：并发监视 /health + /persona/characters（验证不被重请求阻塞）
"""

import json
import threading
import time
import urllib.request
from http.client import HTTPResponse
from urllib.error import HTTPError

BASE = "http://127.0.0.1:8000"
RESULTS = {}


def post_json(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return time.time() - t0, body


def read_sse(path, payload, timeout=300):
    """POST 并逐行读 SSE 流，返回 (事件列表, 各阶段耗时 dict)"""
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    events = []
    t_start = time.time()
    t_first_event = None
    t_first_token = None
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            if t_first_event is None:
                t_first_event = time.time() - t_start
            if data.get("type") == "token" and t_first_token is None:
                t_first_token = time.time() - t_start
            events.append(data)
    except Exception as e:
        events.append({"type": "stream_error", "content": str(e)})
    total = time.time() - t_start
    return events, {
        "total": round(total, 1),
        "first_event": round(t_first_event, 3) if t_first_event is not None else None,
        "first_token": round(t_first_token, 3) if t_first_token is not None else None,
    }


def get(path, timeout=10):
    req = urllib.request.Request(BASE + path)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return time.time() - t0
    except Exception:
        return time.time() - t0


def thread_a_sse():
    """用户 A：SSE 流式 persona/query（新链路验证）"""
    events, timings = read_sse(
        "/persona/query",
        {
            "query": "我和男朋友吵架后他总是冷暴力，我该怎么打破这种局面？",
            "character_id": "adler",
            "session_id": "verify2_sse",
        },
    )
    RESULTS["sse_timings"] = timings
    RESULTS["sse_types"] = [e.get("type") for e in events]
    trace = next((e for e in events if e.get("type") == "trace"), None)
    RESULTS["sse_route_history"] = trace.get("route_history") if trace else None
    result = next((e for e in events if e.get("type") in ("result", "questions")), None)
    RESULTS["sse_result_head"] = (result.get("content") or "")[:120] if result else "NO_RESULT"
    RESULTS["sse_end_time"] = time.time()


def thread_b_eval():
    """用户 B：eval_query（热缓存，应显著快于首次 56.1s）"""
    dur, body = post_json(
        "/persona/eval_query",
        {
            "query": "我总是害怕被拒绝所以不敢表白，怎么办？",
            "character_id": "adler",
        },
    )
    RESULTS["eval_dur"] = round(dur, 1)
    try:
        obj = json.loads(body)
        RESULTS["eval_answer_head"] = (obj.get("answer") or "")[:100]
    except Exception:
        RESULTS["eval_answer_head"] = body[:100]


# 用户 A + 用户 B 同时发起重请求
tA = threading.Thread(target=thread_a_sse)
tB = threading.Thread(target=thread_b_eval)
tA.start()
tB.start()

# 用户 C：并发监视轻量端点
time.sleep(0.8)
delays = []
monitor_start = time.time()
while (tA.is_alive() or tB.is_alive()) and time.time() - monitor_start < 240:
    d1 = get("/health")
    d2 = get("/persona/characters")
    delays.append((round(d1, 3), round(d2, 3)))
    time.sleep(0.5)
tA.join()
tB.join()

RESULTS["monitor_count"] = len(delays)
if delays:
    flat = [x for pair in delays for x in pair]
    RESULTS["monitor_max"] = max(flat)
    RESULTS["monitor_avg"] = round(sum(flat) / len(flat), 3)
    RESULTS["monitor_slow_over_1s"] = sum(1 for x in flat if x > 1.0)
else:
    RESULTS["monitor_max"] = RESULTS["monitor_avg"] = RESULTS["monitor_slow_over_1s"] = 0

with open("tests/_verify_fast2.txt", "w", encoding="utf-8") as f:
    for k, v in RESULTS.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nmonitor_delays: {delays}\n")

print("done -> tests/_verify_fast2.txt")
