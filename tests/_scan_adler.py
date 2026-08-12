# -*- coding: utf-8 -*-
"""临时脚本v3：仅用 pypdf 快速检测 PDF 文本层，不触发 OCR"""
import sys, io
from pathlib import Path

ROOT = Path(r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")
sys.path.insert(0, str(ROOT))

out_fp = ROOT / "tests" / "_adler_scan_result.txt"
lines = []

BASE = ROOT / "8、阿德勒全套文集PDF电子版-13部"
from pypdf import PdfReader

KW = ["爱情", "婚姻", "伴侣", "恋人", "夫妻", "嫉妒", "吸引", "分手", "失恋", "亲密", "追求", "异性", "婚", "恋"]

for fp in sorted(BASE.rglob("*.pdf")):
    try:
        reader = PdfReader(str(fp))
        total = 0
        full = []
        for page in reader.pages:
            t = page.extract_text() or ""
            total += len(t.strip())
            full.append(t)
        text = "\n".join(full)
        kw_hits = {k: text.count(k) for k in KW if k in text}
        top_kw = sorted(kw_hits.items(), key=lambda x: -x[1])[:8]
        flag = "TEXT" if total > 5000 else "SCAN-IMG"
        lines.append(f"[{flag}] {fp.name[:60]} | 页数={len(reader.pages)} | 字符={total}")
        if flag == "TEXT":
            lines.append(f"     感情词命中: {top_kw}")
    except Exception as e:
        lines.append(f"[ERR] {fp.name[:60]} | {e}")
lines.append("--- DONE ---")
open(out_fp, "w", encoding="utf-8").write("\n".join(lines))
