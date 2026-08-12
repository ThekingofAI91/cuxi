# 临时验证脚本：检查 characters 接口返回全名与 tagline
import json
import urllib.request

d = json.loads(urllib.request.urlopen(
    "http://localhost:8000/persona/characters", timeout=10).read())
lines = []
for c in d["characters"]:
    lines.append(f"{c['id']} | {c['name']} | tagline={c.get('tagline', '')}")
open("tests/_api_result.txt", "w", encoding="utf-8").write("\n".join(lines))
print("DONE")
