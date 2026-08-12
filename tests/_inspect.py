# 临时：查看 routes.py 会话与角色关联结构
import re
t = open("src/api/routes.py", encoding="utf-8").read()
lines = t.splitlines()
keys = ["_session_store", "_conversation_history_store", "session_id", "character_id"]
for i, l in enumerate(lines, 1):
    if any(k in l for k in keys):
        print(i, l.strip()[:130])
