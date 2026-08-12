# -*- coding: utf-8 -*-
"""看《大师思想集萃》正文文体"""
import sys

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

import fitz

doc = fitz.open(r"data\persona_chat\jung\荣格\13088290_大师思想集萃  荣格说潜意识与生存.pdf")
n = 0
for i in range(3, 45):
    t = doc[i].get_text().strip()
    if len(t) < 30:
        continue
    n += 1
    if n <= 14:
        print(f"--- p{i+1} ---")
        for line in t.splitlines()[:14]:
            s = line.strip()
            if s:
                print("  ", s[:64])
