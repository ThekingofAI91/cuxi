"""
并发隔离 + 速度验证脚本

模拟两个用户同时使用：
- 用户 A：发 persona eval_query（阿德勒感情问题，重请求）
- 用户 B：在 A 请求进行中反复访问 /health、/persona/characters（轻量请求）

验证：
1. persona 回答耗时（应显著快于之前的 50-70 秒）
2. 并发期间轻量端点不被阻塞（延迟应 < 0.2 秒）
3. 回答内容质量不回归
"""

import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


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


def get(path, timeout=10):
    req = urllib.request.Request(BASE + path)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return time.time() - t0, body
    except Exception as e:
        return time.time() - t0, f"ERROR: {e}"


results = {}


def run_persona():
    results["persona_start"] = time.time()
    dur, body = post_json(
        "/persona/eval_query",
        {
            "query": "我和男朋友吵架后他总是冷暴力，我该怎么打破这种局面？",
            "character_id": "adler",
        },
    )
    results["persona_dur"] = round(dur, 1)
    results["persona_body_len"] = len(body)
    results["persona_body"] = body
    results["persona_end"] = time.time()


t = threading.Thread(target=run_persona)
t.start()

# 用户 B：在 persona 请求进行中持续访问轻量端点
time.sleep(1.0)  # 等 persona 进入检索/生成阶段
monitor_delays = []
monitor_start = time.time()
while t.is_alive() and time.time() - monitor_start < 240:
    d1, b1 = get("/health")
    d2, b2 = get("/persona/characters")
    monitor_delays.append((round(d1, 3), round(d2, 3)))
    time.sleep(0.5)
t.join()

results["monitor_count"] = len(monitor_delays)
if monitor_delays:
    results["monitor_max_health"] = max(d for d, _ in monitor_delays)
    results["monitor_max_chars"] = max(d for _, d in monitor_delays)
    all_d = [x for pair in monitor_delays for x in pair]
    results["monitor_avg"] = round(sum(all_d) / len(all_d), 3)
    results["monitor_slow_over_1s"] = sum(1 for x in all_d if x > 1.0)
else:
    results["monitor_max_health"] = 0
    results["monitor_max_chars"] = 0
    results["monitor_avg"] = 0
    results["monitor_slow_over_1s"] = 0

with open("tests/_verify_fast.txt", "w", encoding="utf-8") as f:
    for k, v in results.items():
        if k == "persona_body":
            f.write(f"persona_body:\n{v}\n\n")
        else:
            f.write(f"{k}: {v}\n")
    f.write(f"\nmonitor_delays(全部): {monitor_delays}\n")

print("done -> tests/_verify_fast.txt")
