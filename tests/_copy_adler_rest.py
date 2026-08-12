# -*- coding: utf-8 -*-
"""临时脚本：复制剩余 2 本阿德勒书到 data/persona_chat/adler/阿德勒/"""
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")
BASE = ROOT / "8、阿德勒全套文集PDF电子版-13部"
DEST = ROOT / "data" / "persona_chat" / "adler" / "阿德勒"
DEST.mkdir(parents=True, exist_ok=True)

out = []

# 1. 6、性格的塑造.pdf（= 套装4册合集：超越自卑+洞察人性+理解生命+性格的塑造）
src1 = BASE / "6、性格的塑造" / "6、性格的塑造.pdf"
dst1 = DEST / "超越自卑与洞察人性（阿德勒四大名著合集）.pdf"
if src1.exists():
    shutil.copy2(src1, dst1)
    out.append(f"OK  6、性格的塑造.pdf -> {dst1.name} ({dst1.stat().st_size} bytes)")

# 2. 被讨厌的勇气（有文本层版，2.1MB）
src2 = BASE / "13、被讨厌的勇气：“自我启发之父”阿德勒的哲学课PDF" / "被讨厌的勇气：“自我启发之父”阿德勒的哲学课.pdf"
dst2 = DEST / "被讨厌的勇气.pdf"
if src2.exists():
    shutil.copy2(src2, dst2)
    out.append(f"OK  被讨厌的勇气.pdf -> {dst2.name} ({dst2.stat().st_size} bytes)")

# 列出最终目录
out.append("")
out.append("=== 最终数据目录 ===")
for f in sorted(DEST.iterdir()):
    out.append(f"  - {f.name} | {f.stat().st_size} bytes")

out.append("--- DONE ---")
open(ROOT / "tests" / "_copy_adler_out.txt", "w", encoding="utf-8").write("\n".join(out))
