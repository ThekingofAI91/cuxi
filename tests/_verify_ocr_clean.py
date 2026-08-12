# -*- coding: utf-8 -*-
"""OCR 噪声清洗验证：单元测试噪声判定 + 真实 OCR 前 8 页清洗前后对比"""
import sys

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

import fitz
from src.document_processing.parser import _is_ocr_noise_line, _looks_like_toc_page

# ---- 1. 单元测试：真实噪声行样例（来自 ChromaDB 抽样与常见版权页）----
print("===== 1. 噪声判定单元测试 =====")
noise_samples = [
    "版权信息",
    "书名： 走出孤独",
    "图书在版编目（CIP）数据",
    "ISBN 978-7-201-17082-4",
    "出版发行：天津人民出版社",
    "责任编辑：张三",
    "定价：58.00元",
    "12",
    "2021.4",
    "……",
    "目录：",
]
normal_samples = [
    "青年与哲人的对话开始了，他们坐在书房里。",
    "人生道路最终还是由你自己决定的。",
    "我今年三十五岁，一直觉得自己活在别人的期待里。",
    "第 3 章 课题分离与自由",
]
for s in noise_samples:
    assert _is_ocr_noise_line(s), f"应判为噪声但未判出: {s}"
print(f"噪声行判定通过: {len(noise_samples)} 条全部正确识别")
for s in normal_samples:
    assert not _is_ocr_noise_line(s), f"不应判为噪声但被误删: {s}"
print(f"正文行保留通过: {len(normal_samples)} 条全部保留")

# 目录页启发式
toc_page = [
    "第一章 为何讨厌自己 1",
    "第二章 一切烦恼来自人际关系 45",
    "第三章 让干涉你生活的人见鬼去 89",
    "第四章 要有被讨厌的勇气 132",
    "第五章 认真的人生活在当下 178",
    "后记 190",
]
assert _looks_like_toc_page(toc_page), "目录页应被识别"
print("目录页识别通过")

# ---- 2. 真实 OCR 对比（被讨厌的勇气 前 8 页）----
print("\n===== 2. 真实 OCR 清洗前后对比 =====")
from rapidocr_onnxruntime import RapidOCR

path = r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant\data\persona_chat\adler\阿德勒\被讨厌的勇气.pdf"
doc = fitz.open(path)
ocr = RapidOCR()

total_raw = 0
total_kept = 0
dropped_examples = []
for page_num in range(8):
    pix = doc[page_num].get_pixmap(dpi=200)
    result, _ = ocr(pix.tobytes("png"))
    if not result:
        continue
    page_texts = []
    for item in result:
        text = item[1].strip()
        try:
            confidence = float(item[2]) if len(item) > 2 else 0
        except (ValueError, TypeError):
            confidence = 0
        if text and confidence > 0.5:
            page_texts.append(text)

    raw_count = len(page_texts)
    total_raw += raw_count
    if _looks_like_toc_page(page_texts):
        print(f"  p{page_num+1}: 目录页整体跳过 ({raw_count} 行)")
        continue
    cleaned = [t for t in page_texts if not _is_ocr_noise_line(t)]
    total_kept += len(cleaned)
    dropped = [t for t in page_texts if _is_ocr_noise_line(t)]
    for d in dropped[:4]:
        dropped_examples.append((page_num + 1, d))

print(f"\n前 8 页: 原始 {total_raw} 行 -> 清洗后保留 {total_kept} 行（丢弃 {total_raw - total_kept} 行）")
print(f"清洗掉的样例:")
for p, d in dropped_examples[:10]:
    print(f"  p{p}: {d[:40]!r}")
doc.close()
