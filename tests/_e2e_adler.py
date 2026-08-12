# -*- coding: utf-8 -*-
"""端到端验证：阿德勒角色感情问答（结果写入文件，避免控制台编码问题）"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"
OUT = "tests/_adler_e2e_result.txt"


def post_json(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


with open(OUT, "w", encoding="utf-8") as f:
    f.write("== 角色列表 ==\n")
    with urllib.request.urlopen(BASE + "/persona/characters", timeout=30) as resp:
        chars = json.loads(resp.read().decode("utf-8"))["characters"]
        for c in chars:
            f.write(f"  {c['id']} | {c['name']} | {c['description']}\n")

    for q in ["和男朋友吵架后他总是冷暴力，我该怎么办", "我总是不敢主动表达爱意，怕被拒绝"]:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Q: {q}\n")
        result = post_json("/persona/eval_query", {"query": q, "character_id": "adler"})
        answer = result.get("answer", "")
        if answer.startswith("ERROR"):
            f.write("ERROR: " + answer + "\n")
            continue
        f.write(f"A (length={len(answer)}, contexts={len(result.get('contexts', []))}):\n")
        f.write(answer + "\n")

print("done ->", OUT)
