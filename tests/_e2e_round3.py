"""端到端验证：第 3 轮指代性提问是否关联前两轮内容（结果写入文件避免终端截断）"""
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
SESSION = "e2e_ctx_round3_001"
OUT = Path(__file__).parent / "_e2e_round3_out.txt"

q1 = "我最近总是做一个重复的噩梦：自己一个人走在一个巨大的黑暗洞穴里，怎么都找不到出口，心里非常害怕。"
q2 = "这个洞穴里我还看见了一面古老的镜子，镜子里的人不是我，而是一个衰老的我。这让我很不安。"
q3 = "结合我之前跟你说的那个梦，镜子里的衰老的我到底意味着什么？我该怎么面对？"


def ask(query: str, history: list[dict], timeout: float = 300.0) -> str:
    body = {"query": query, "session_id": SESSION, "character_id": "jung", "history": history}
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", f"{BASE}/persona/query", json=body) as resp:
            status = resp.status_code
            if status != 200:
                return f"ERROR HTTP {status}"
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
    lines = []
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
        lines.append(f"健康检查: {r.json().get('message', 'ok')}")
    except Exception as e:
        lines.append(f"❌ 服务未启动: {e}")
        OUT.write_text("\n".join(lines), encoding="utf-8")
        return 1

    # 第 1 轮
    a1 = ask(q1, [])
    lines.append("\n" + "=" * 60)
    lines.append(f"第 1 轮回答:\n{a1}")
    history = [
        {"type": "user", "content": q1},
        {"type": "assistant", "content": a1},
    ]

    # 第 2 轮
    a2 = ask(q2, history)
    lines.append("\n" + "=" * 60)
    lines.append(f"第 2 轮回答:\n{a2}")
    history.append({"type": "user", "content": q2})
    history.append({"type": "assistant", "content": a2})

    # 第 3 轮（指代前文）
    a3 = ask(q3, history)
    lines.append("\n" + "=" * 60)
    lines.append(f"第 3 轮回答:\n{a3}")

    lines.append("\n" + "=" * 60)
    lines.append("上下文关联检查（第 3 轮回答应提及前文内容）:")
    checks = {
        "提到'洞穴'（第1轮内容）": "洞穴" in a3,
        "提到'镜子'（第2轮内容）": "镜子" in a3,
        "提到'衰老'或'老'（第2轮内容）": ("衰老" in a3 or "老" in a3),
        "延续性回应（无分析报告痕迹）": ("分析总结" not in a3 and "可信度" not in a3 and "置信度" not in a3),
        "无路由失败痕迹": ("未找到相关信息" not in a3 and "ERROR" not in a3),
    }
    passed = True
    for name, ok in checks.items():
        lines.append(f"  {'✅' if ok else '❌'} {name}")
        if not ok:
            passed = False
    lines.append(f"\n{'🎉 端到端上下文验证通过' if passed else '⚠️ 部分检查未通过，请查看上方回答内容'}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"结果已写入: {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
