# -*- coding: utf-8 -*-
"""核查全部语料 PDF 的作者/版权信息，验证 source_profile 分类是否准确"""
import sys

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

import fitz
from pathlib import Path

data_root = Path(r"data\persona_chat")
files = sorted(data_root.rglob("*.pdf"))

print(f"共 {len(files)} 个 PDF\n")
for f in files:
    try:
        doc = fitz.open(str(f))
        author_lines = []
        for i in range(min(8, len(doc))):
            t = doc[i].get_text()
            for line in t.splitlines():
                s = line.strip().replace(" ", "")
                # 抓 CIP 行/作者行/书名行
                if any(k in s for k in ("著", "编译", "编著", "主编", "编")) and len(s) < 80:
                    if s not in author_lines:
                        author_lines.append(s)
                if s.startswith("ISBN") and len(s) < 60:
                    author_lines.append(s)
            if len(author_lines) >= 4:
                break
        doc.close()
        print(f"【{f.parent.name}/{f.name}】")
        for l in author_lines[:4]:
            print(f"    {l[:70]}")
    except Exception as e:
        print(f"【{f.name}】读取失败: {e}")
