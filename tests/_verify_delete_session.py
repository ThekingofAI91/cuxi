"""
验证：删除会话 API 是否真正清除后端历史，
以及删除后重新问同样的问题不再被误判为"重复提问"。

流程（模拟用户操作）：
1. 旧对话：session 问问题 X（第一次）
2. 删除对话：调用 DELETE /conversation/{session_id}（前端删除时同步通知后端）
3. 重新提问：同一 session 再次发送同样的问题 X
   → 后端历史已清空，回答不应再出现"重复问了"之类的误判
"""
import json
import sys
import httpx

# Windows 控制台默认 GBK，无法打印 emoji，统一输出 ASCII 标记
OK = "[PASS]"
FAIL = "[FAIL]"

BASE = "http://127.0.0.1:8000"
SESSION = "del_verify_s1"
X = "我最近总是做一个重复的噩梦：自己一个人走在一个巨大的黑暗洞穴里，怎么都找不到出口，心里非常害怕。你能帮我分析一下这个梦吗？"


def ask(session_id: str, query: str, timeout: int = 300) -> str:
    """调用 persona/query，解析 SSE 流返回最终回答"""
    body = {
        "query": query,
        "session_id": session_id,
        "character_id": "jung",
    }
    with httpx.stream("POST", f"{BASE}/persona/query", json=body, timeout=timeout) as r:
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}")
            return ""
        full = ""
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except Exception:
                continue
            if data.get("type") == "result":
                full = data.get("content", "")
    return full


def main() -> int:
    # 健康检查
    try:
        r = httpx.get(f"{BASE}/health", timeout=10)
        assert r.status_code == 200
    except Exception as e:
        print(f"[FAIL] 服务未启动: {e}")
        return 1
    print("健康检查: OK")

    # ---- 第 1 步：旧对话 ----
    print("=" * 60)
    print("第 1 步：模拟旧对话，session 发送问题 X（模拟用户之前问过）")
    a1 = ask(SESSION, X)
    if not a1:
        print(f"{FAIL} 第一问无回答，中止")
        return 1
    print(f"第一问回答(前120字): {a1[:120]}")
    print(f"回答总长度: {len(a1)}")

    # ---- 第 2 步：删除对话 ----
    print("=" * 60)
    print("第 2 步：模拟前端删除对话 → 调用 DELETE /conversation/{session_id}")
    r = httpx.delete(f"{BASE}/conversation/{SESSION}", timeout=10)
    data = r.json()
    print(f"DELETE 响应: HTTP {r.status_code} {data}")
    # 关键断言：第 1 步的历史必须真实存在于后端并被删除
    if r.status_code != 200 or not data.get("deleted"):
        print(f"{FAIL} DELETE 未真正删除后端历史（deleted != True），验证无效")
        return 1
    print(f"{OK} 后端历史确实存在且已删除")

    # ---- 第 3 步：同一 session 重新问同样的问题 ----
    print("=" * 60)
    print("第 3 步：重新传入之前问过的问题 X（同一 session）")
    a2 = ask(SESSION, X)
    if not a2:
        print(f"{FAIL} 第二问无回答，中止")
        return 1
    print(f"第二问回答(前150字): {a2[:150]}")
    print(f"回答总长度: {len(a2)}")

    # ---- 检查是否误判重复 ----
    print("=" * 60)
    dup_markers = ["重复问", "问过这个", "之前已经", "已经回答", "又问了一遍", "之前问过"]
    hit = [m for m in dup_markers if m in a2]
    if hit:
        print(f"{FAIL} 仍被误判为重复提问！命中关键词: {hit}")
        print(f"回答全文: {a2}")
        return 1
    print(f"{OK} 删除后重新提问，未被误判为'重复提问'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
