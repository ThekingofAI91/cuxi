# -*- coding: utf-8 -*-
"""临时状态检查 v2：绝对路径写文件"""
import os

BASE = r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant"
out_path = os.path.join(BASE, "tests", "_verify_out.txt")
report = []

if os.path.exists(out_path):
    report.append(f"verify_out size: {os.path.getsize(out_path)}")
    with open(out_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    report.append("=== content ===")
    report.append(content)
else:
    report.append("verify_out NOT FOUND")

with open(os.path.join(BASE, "tests", "_report_v2.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("written:", os.path.join(BASE, "tests", "_report_v2.txt"))
