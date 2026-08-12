# -*- coding: utf-8 -*-
"""抽查现有语料的对话体形态：《大师思想集萃》《红书》《被讨厌的勇气》"""
import sys

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

import fitz
from collections import Counter

FILES = {
    "大师思想集萃": r"data\persona_chat\jung\荣格\13088290_大师思想集萃  荣格说潜意识与生存.pdf",
    "红书": r"data\persona_chat\jung\荣格\红书.pdf",
    "被讨厌的勇气": r"data\persona_chat\adler\阿德勒\被讨厌的勇气.pdf",
}

def probe(path, name, pages=(0, 30)):
    doc = fitz.open(path)
    total = len(doc)
    text_pages = 0
    sample_lines = []
    quote_count = 0      # 含引号的行
    qa_count = 0         # 问：/答： 模式
    line_count = 0
    for i in range(min(total, pages[1])):
        t = doc[i].get_text()
        if len(t.strip()) < 30:
            continue
        text_pages += 1
        for line in t.splitlines():
            s = line.strip()
            if not s:
                continue
            line_count += 1
            if s.startswith(("问", "答", "Q", "A", "问：", "答：")) or s[:1] in ("问", "答") and s[1:2] in "：:：" :
                qa_count += 1
            if "“" in s or "”" in s or '"' in s or "'" in s:
                quote_count += 1
            if len(sample_lines) < 25:
                sample_lines.append(s[:50])
    doc.close()
    print(f"\n===== {name}（共 {total} 页）=====")
    print(f"文本页(前{pages[1]}页内): {text_pages}, 行数: {line_count}, 含引号行: {quote_count}, 问/答开头行: {qa_count}")
    print("--- 文本样例 ---")
    for l in sample_lines:
        print(f"  {l}")

for k, v in FILES.items():
    probe(v, k)
