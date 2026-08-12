# -*- coding: utf-8 -*-
"""临时状态检查：验证脚本输出 + 进程状态（写文件，绕过终端回放）"""
import os
import shutil

result = []

# 1. 验证输出文件大小与内容副本
out_path = os.path.join(os.path.dirname(__file__), "_verify_out.txt")
if os.path.exists(out_path):
    size = os.path.getsize(out_path)
    result.append(f"verify_out size: {size}")
    if size > 0:
        shutil.copy2(out_path, os.path.join(os.path.dirname(__file__), "_verify_out_copy.txt"))
        result.append("copied to _verify_out_copy.txt")
else:
    result.append("verify_out NOT FOUND")

# 2. 进程状态
result.append("--- process ---")
try:
    import subprocess
    ps = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
        capture_output=True, text=True, encoding="gbk", errors="replace",
    )
    result.append(ps.stdout)
except Exception as e:
    result.append(f"tasklist failed: {e}")

report = "\n".join(result)
report_path = os.path.join(os.path.dirname(__file__), "_check_result.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(report)
