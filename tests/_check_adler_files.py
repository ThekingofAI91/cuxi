# -*- coding: utf-8 -*-
"""临时脚本：确认 6、性格的塑造.pdf 内容，复制剩余文件到 adler 数据目录"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader

# 1. 预览 6、性格的塑造.pdf 开头
p1 = ROOT / "8、阿德勒全套文集PDF电子版-13部" / "6、性格的塑造" / "6、性格的塑造.pdf"
r = PdfReader(str(p1))
t0 = (r.pages[0].extract_text() or "")[:150]
t1 = (r.pages[1].extract_text() or "")[:150]
print("【6、性格的塑造.pdf】第1页:", t0.replace("\n", " "))
print("第2页:", t1.replace("\n", " "))

# 2. 查找被讨厌的勇气 pdf 真实路径（目录名含中文引号）
base13 = ROOT / "8、阿德勒全套文集PDF电子版-13部" / "13、被讨厌的勇气：“自我启发之父”阿德勒的哲学课PDF"
print("\n【13目录】存在:", base13.exists())
if base13.exists():
    for f in base13.iterdir():
        print("  -", f.name, f.stat().st_size if f.is_file() else "(dir)")
