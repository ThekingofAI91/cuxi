"""稳定性诊断：连续请求测试，判断 8000 端口分发是否正常"""
import time
import urllib.request

ok = 0
fail = 0
times = []

for i in range(20):
    t0 = time.time()
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10)
        dt = time.time() - t0
        times.append(dt)
        ok += 1
        status = r.status
    except Exception as e:
        dt = time.time() - t0
        times.append(dt)
        fail += 1
        status = f"FAIL {type(e).__name__}"
    print(f"req{i:02d}: {status} {dt:.3f}s")

print(f"\n成功 {ok}/20, 失败 {fail}/20")
if times:
    print(f"平均耗时: {sum(times)/len(times):.3f}s, 最大耗时: {max(times):.3f}s, 最小耗时: {min(times):.3f}s")

# 首页加载测试
t0 = time.time()
try:
    r = urllib.request.urlopen("http://127.0.0.1:8000/", timeout=15)
    body = r.read()
    print(f"\n首页 GET /: {r.status}, 耗时 {time.time()-t0:.3f}s, 大小 {len(body)} 字节")
except Exception as e:
    print(f"\n首页 GET /: FAIL {type(e).__name__}: {e}")
