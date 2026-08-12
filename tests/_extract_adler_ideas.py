# -*- coding: utf-8 -*-
"""临时脚本：从 adler 5 本书提取感情关键词原文片段，供提炼 core_ideas.md"""
import re
from pathlib import Path

ROOT = Path(r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")
DATA = ROOT / "data" / "persona_chat" / "adler" / "阿德勒"
OUT = ROOT / "tests" / "_adler_ideas_out.txt"

from pypdf import PdfReader

KW = ["婚姻", "爱情", "伴侣", "恋人", "夫妻", "嫉妒", "吸引", "亲密", "结婚", "妻子", "丈夫", "恋爱", "离婚", "性爱", "情侣"]
lines = []

def clean(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

for fp in sorted(DATA.glob("*.pdf")):
    lines.append(f"\n{'='*70}\n### {fp.name} ###")
    reader = PdfReader(str(fp))
    # 前 3 页当目录预览
    toc = []
    for i in range(min(3, len(reader.pages))):
        t = clean(reader.pages[i].extract_text() or "")
        if t:
            toc.append(f"[页{i+1}] {t[:400]}")
    lines.append("【开头/目录】" + " | ".join(toc)[:900])
    # 全文搜索关键词片段
    full_pages = [clean(p.extract_text() or "") for p in reader.pages]
    hits = []
    for pi, pt in enumerate(full_pages):
        for kw in KW:
            for m in re.finditer(kw, pt):
                s = max(0, m.start() - 60)
                e = min(len(pt), m.end() + 160)
                hits.append((kw, pi + 1, pt[s:e]))
    # 去重相近片段
    seen = set()
    picked = []
    for kw, pi, seg in hits:
        key = seg[:40]
        if key in seen:
            continue
        seen.add(key)
        picked.append((kw, pi, seg))
        if len(picked) >= 22:
            break
    lines.append(f"【感情片段 {len(hits)} 命中，展示 {len(picked)}】")
    for kw, pi, seg in picked:
        lines.append(f"  [{kw}|p{pi}] {seg[:220]}")

lines.append("\n--- DONE ---")
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print(f"OK -> {OUT} ({len(lines)} lines)")
