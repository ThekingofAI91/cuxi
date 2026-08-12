"""端到端验证：名人对话多轮上下文能力

模拟 3 轮对话：
1. 用户描述噩梦（洞穴/迷宫）
2. 用户追问梦的象征意义（延续话题）
3. 用户指代"这个梦"问与童年的关系（验证第 3 轮能否关联第 1 轮内容）

通过 /persona/query 流式端点验证。
"""
import json
import sys
import httpx

BASE = "http://127.0.0.1:8000"
SESSION = "e2e_ctx_test_001"


def ask(query: str, session_id: str, history: list[dict]) -> str:
    """发送一轮对话，解析 SSE 返回最终回答"""
    body = {
        "query": query,
        "session_id": session_id,
        "character_id": "jung",
        "history": history,
    }
    with httpx.Client(timeout=180.0) as client:
        with client.stream("POST", f"{BASE}/persona/query", json=body) as resp:
            print(f"  HTTP {resp.status_code}")
            if resp.status_code != 200:
                return f"ERROR HTTP {resp.status_code}"
            final = ""
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except Exception:
                    continue
                if data.get("type") == "result":
                    final = data.get("content", "")
                if data.get("type") == "error":
                    final = f"ERROR: {data.get('content')}"
    return final


def main():
    # 先健康检查
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        print(f"健康检查: {r.json().get('message', 'ok')}")
    except Exception as e:
        print(f"❌ 服务未启动: {e}")
        sys.exit(1)

    history: list[dict] = []
    q1 = "我最近总是做一个重复的噩梦：自己一个人走在一个巨大的黑暗洞穴里，怎么都找不到出口，心里非常害怕。"
    q2 = "这个洞穴里我还看见了一面古老的镜子，镜子里的人不是我，而是一个衰老的我。这让我很不安。"
    q3 = "结合我之前跟你说的那个梦，镜子里的衰老的我到底意味着什么？我该怎么面对？"

    print("\n" + "=" * 60)
    print(f"第 1 轮: {q1[:40]}...")
    a1 = ask(q1, SESSION, history)
    history.append({"type": "user", "content": q1})
    history.append({"type": "assistant", "content": a1})
    print(f"荣格回复: {a1[:200]}...\n")

    print("=" * 60)
    print(f"第 2 轮: {q2[:40]}...")
    a2 = ask(q2, SESSION, history)
    history.append({"type": "user", "content": q2})
    history.append({"type": "assistant", "content": a2})
    print(f"荣格回复: {a2[:200]}...\n")

    print("=" * 60)
    print(f"第 3 轮（指代前文）: {q3[:40]}...")
    a3 = ask(q3, SESSION, history)
    print(f"荣格回复: {a3}\n")

    print("=" * 60)
    print("上下文关联检查（第 3 轮回答应提及前文内容）:")
    checks = {
        "提到'洞穴'（第1轮内容）": "洞穴" in a3,
        "提到'镜子'（第2轮内容）": "镜子" in a3,
        "提到'衰老'或'老化'（第2轮内容）": ("衰老" in a3 or "老" in a3),
        "延续性回应（以角色口吻，无分析报告痕迹）": ("分析总结" not in a3 and "可信度" not in a3 and "置信度" not in a3),
    }
    passed = True
    for name, ok in checks.items():
        print(f"  {'✅' if ok else '❌'} {name}")
        if not ok:
            passed = False
    print(f"\n{'🎉 端到端上下文验证通过' if passed else '⚠️ 部分检查未通过，请查看上方回答内容'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
