"""
parser.py — 文档解析器

支持解析多种文档格式：PDF / Markdown / 代码文件
输出结构化的 ParsedElement 列表，供 chunker 进一步处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ParsedElement:
    """
    文档解析后的基本单元。
    
    element_type 类型:
    - "heading": 标题
    - "paragraph": 段落
    - "code": 代码块
    - "list": 列表项
    - "table": 表格
    - "figure": 图片/图表
    """
    content: str
    element_type: str = "paragraph"
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """返回字符数，用于分块计算"""
        return len(self.content)


# ============================================================
# Markdown 解析器
# ============================================================

class MarkdownParser:
    """解析 Markdown 文件为结构化元素列表"""

    # 代码块正则
    _CODE_BLOCK_RE = re.compile(r"^```(\w*)\s*\n(.*?)\n```", re.DOTALL | re.MULTILINE)
    # 标题正则
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    # 列表项正则
    _LIST_ITEM_RE = re.compile(r"^(\s*[-*+]\s|\s*\d+\.\s)", re.MULTILINE)
    # 表格正则（简易：行中包含 |）
    _TABLE_RE = re.compile(r"^(\|.+\|)$", re.MULTILINE)

    def parse(self, text: str, source: str = "") -> list[ParsedElement]:
        """
        将 Markdown 文本解析为元素列表
        
        Args:
            text: Markdown 原始文本
            source: 来源标识（文件名）
        
        Returns:
            ParsedElement 列表
        """
        elements: list[ParsedElement] = []
        lines = text.split("\n")

        # 剥离代码块（单独处理）
        code_blocks = []
        processed_lines = self._extract_code_blocks(lines, code_blocks)

        # 按段落解析非代码内容
        i = 0
        current_paragraph: list[str] = []
        current_list: list[str] = []
        in_table = False
        table_lines: list[str] = []

        while i < len(processed_lines):
            line = processed_lines[i]
            stripped = line.strip()

            # 空行 → 结束当前段落/列表
            if not stripped:
                elements = self._flush_paragraph(elements, current_paragraph, source)
                elements = self._flush_list(elements, current_list, source)
                i += 1
                continue

            # 标题行
            heading_match = self._HEADING_RE.match(stripped)
            if heading_match:
                elements = self._flush_paragraph(elements, current_paragraph, source)
                elements = self._flush_list(elements, current_list, source)
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                elements.append(ParsedElement(
                    content=title,
                    element_type="heading",
                    metadata={"source": source, "heading_level": level, "heading": title},
                ))
                i += 1
                continue

            # 表格行
            if self._TABLE_RE.match(stripped):
                if not in_table:
                    elements = self._flush_paragraph(elements, current_paragraph, source)
                    elements = self._flush_list(elements, current_list, source)
                    in_table = True
                    table_lines = [stripped]
                else:
                    table_lines.append(stripped)
                # 下一行若不是表格行，结束表格
                if i + 1 >= len(processed_lines) or not self._TABLE_RE.match(processed_lines[i + 1].strip()):
                    elements.append(ParsedElement(
                        content="\n".join(table_lines),
                        element_type="table",
                        metadata={"source": source},
                    ))
                    in_table = False
                    table_lines = []
                i += 1
                continue

            # 列表项
            if self._LIST_ITEM_RE.match(stripped):
                if not current_list:
                    elements = self._flush_paragraph(elements, current_paragraph, source)
                current_list.append(stripped)
                i += 1
                continue

            # 普通文本 → 段落
            if current_list:
                elements = self._flush_list(elements, current_list, source)
            current_paragraph.append(line)
            i += 1

        # 处理剩余的段落/列表
        elements = self._flush_paragraph(elements, current_paragraph, source)
        elements = self._flush_list(elements, current_list, source)

        # 插入代码块（在它们原本的位置附近）
        elements = self._insert_code_blocks(elements, code_blocks, source)

        return elements

    # ---- 内部辅助方法 ----

    def _extract_code_blocks(self, lines: list[str], code_blocks: list[dict]) -> list[str]:
        """从行列表中剥离代码块，返回剩余行"""
        result: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("```"):
                fence = lines[i].strip()
                lang = fence[3:].strip()
                code_lines: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                # i 指向 ```，跳过
                code_blocks.append({
                    "language": lang,
                    "code": "\n".join(code_lines),
                    "line_number": len(result),  # 在结果中的大致位置
                })
                i += 1
            else:
                result.append(lines[i])
                i += 1
        return result

    def _flush_paragraph(
        self, elements: list[ParsedElement], paragraph: list[str], source: str
    ) -> list[ParsedElement]:
        """将缓存的段落行刷新为元素"""
        if not paragraph:
            return elements
        text = "\n".join(paragraph).strip()
        if text:
            elements.append(ParsedElement(
                content=text,
                element_type="paragraph",
                metadata={"source": source},
            ))
        paragraph.clear()
        return elements

    def _flush_list(
        self, elements: list[ParsedElement], list_items: list[str], source: str
    ) -> list[ParsedElement]:
        """将缓存的列表刷新为元素"""
        if not list_items:
            return elements
        text = "\n".join(list_items).strip()
        if text:
            elements.append(ParsedElement(
                content=text,
                element_type="list",
                metadata={"source": source},
            ))
        list_items.clear()
        return elements

    def _insert_code_blocks(
        self, elements: list[ParsedElement], code_blocks: list[dict], source: str
    ) -> list[ParsedElement]:
        """将代码块插入回元素列表中"""
        if not code_blocks:
            return elements

        result = list(elements)
        for block in code_blocks:
            result.append(ParsedElement(
                content=block["code"],
                element_type="code",
                metadata={
                    "source": source,
                    "language": block["language"] or "unknown",
                },
            ))
        return result


# ============================================================
# OCR 噪声清洗（版权页/出版信息/目录页/孤立字符）
# ============================================================

# 出版/版权信息模式（匹配前先压缩空白，模式内不含空格）
_OCR_NOISE_PATTERNS = [
    r"版权信息",
    r"版权所有",
    r"图书在版编目",
    r"版本图书馆",
    r"CIP",
    r"ISBN",
    r"书名[:：]",
    r"出版发行[:：]",
    r"出版人[:：]",
    r"出版社",
    r"印刷",
    r"定价[:：]",
    r"开本",
    r"字数",
    r"印张",
    r"印次",
    r"策划编辑",
    r"责任编辑",
    r"装帧设计",
    r"封面设计",
    r"电子邮箱",
    r"邮购电话",
    r"目录[:：]",
    # 电子书制作者署名（封面常见，如 "Playinart制作"）：
    # 只匹配「纯 ASCII 前缀 + 制作」，正文里绝不会出现这种行
    r"^[A-Za-z0-9]+制作$",
]

# 装饰性页码：`~ 2 ~`、`— 12 —`、`· 7 ·` 这类。实测《王阳明大传》每页一条，
# 577 页共 577 条，占全书行数 5%；且因为落在页面 90.8% 处（边距判据只吃
# 上下 6%），穿过了边距过滤，又被拼进正文段落。
_DECORATED_PAGE_NUM_RE = re.compile(r"^[~～\-—·*※＝=]+(\d{1,4})[~～\-—·*※＝=]+$")


def _is_ocr_noise_line(line: str) -> bool:
    """判断 OCR 行是否为噪声（版权页/出版信息/孤立页码/符号串）"""
    compact = re.sub(r"\s+", "", line)
    if not compact:
        return True
    # 孤立页码/数字（≤6 位）
    if re.fullmatch(r"\d{1,6}", compact):
        return True
    # 装饰性页码（~ 2 ~ / — 12 — / · 7 ·）
    if _DECORATED_PAGE_NUM_RE.fullmatch(compact):
        return True
    # 孤立日期/版本号（如 2021.4、2020.12.1）
    if re.fullmatch(r"\d{2,4}[./]\d{1,2}([./]\d{1,2})?", compact):
        return True
    # 孤立符号/标点串
    if len(compact) <= 8 and re.fullmatch(r"[\W_]+", compact):
        return True
    # 出版/版权信息（限定短行，避免误伤正文长句）
    if len(compact) < 50:
        for pattern in _OCR_NOISE_PATTERNS:
            if re.search(pattern, compact):
                return True
    return False


def _looks_like_toc_page(lines: list[str]) -> bool:
    """目录页启发式：多数行以数字结尾、无句末标点、行较短"""
    if len(lines) < 5:
        return False
    toc_like = 0
    for line in lines:
        compact = re.sub(r"\s+", "", line)
        if not compact:
            continue
        if (
            len(compact) < 40
            and re.search(r"\d+$", compact)
            and not re.search(r"[。！？.!?]$", compact)
        ):
            toc_like += 1
    return toc_like >= max(3, int(len(lines) * 0.4))


# ============================================================
# PDF 解析器
# ============================================================

# ---- 标题识别：基于字号，取代旧的"短行且无句末标点"启发式 ----
# 旧启发式的致命问题：文字型 PDF 是**逐行**输出的，短行遍地都是，
# 实测 84% 的行被判成标题（连"·"、"01"、"CIP"都是）；而 AdaptiveChunker
# 的 _group_by_heading 见 heading 就断组，于是分块碎成数千块
# （一本 94 页的书 → 3063 块、块长中位 47 字）。
# 新判据三条同时成立才算标题：
#   ① 字号 ≥ 正文字号 × 1.12
#   ② 该字号在全文的行数占比 ≤ 6%
#   ③ 行不长于 45 字
# ② 是专门用来治「用大一号字排引文」的书的：引文字号行数占比高，会被排除
# （实测《王阳明心学口诀》19.5 号字占 34.7% 行，是真引文；真标题只有 30 号那 7 行）。
_PDF_HEADING_SIZE_RATIO = 1.12
_PDF_HEADING_SIZE_SHARE_MAX = 0.06
_PDF_HEADING_MAX_CHARS = 45

# 页眉页脚：贴在页面顶部/底部 6% 区域内的短行（页码、书名通常在这里）
_PDF_MARGIN_RATIO = 0.06
_PDF_MARGIN_MAX_CHARS = 30

_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def _norm_font_size(size) -> float:
    """字号归一化到 1 位小数。

    PDF 里的字号常带浮点尾差（30.000000953674316），不归一化会出现
    「同一个字号两种写法」，字号集合比对会静默失败（写这个修复时踩过）。
    """
    try:
        return round(float(size or 0), 1)
    except (TypeError, ValueError):
        return 0.0


def _join_pdf_lines(parts: list[str]) -> str:
    """把同一段落的多行拼成连续文本：中文之间直接接，西文单词之间补空格。

    PDF 的换行是排版换行、不是语义换行；逐行保留会把这些碎行一路带进
    向量库和提示词。
    """
    out = ""
    for text in parts:
        text = text.strip()
        if not text:
            continue
        if out and not _CJK_RE.search(out[-1]) and not _CJK_RE.search(text[0]):
            out += " "
        out += text
    return out


def _pick_heading_sizes(size_chars: dict, size_lines: dict) -> tuple[float, set]:
    """按字号分布挑出「标题字号」，返回 (正文字号, 标题字号集合)。

    正文字号 = 覆盖字符数最多的字号（正文永远占字符量的大头）。
    """
    if not size_chars:
        return 0.0, set()
    body_size = max(size_chars, key=lambda s: (size_chars[s], s))
    total_lines = sum(size_lines.values()) or 1
    heading_sizes = {
        size for size, n in size_lines.items()
        if size >= body_size * _PDF_HEADING_SIZE_RATIO
        and n / total_lines <= _PDF_HEADING_SIZE_SHARE_MAX
    }
    return body_size, heading_sizes


def _heading_level(size: float, heading_sizes: set) -> int:
    """标题层级：字号越大层级越浅（1 = 最大字号）"""
    ordered = sorted(heading_sizes, reverse=True)
    return ordered.index(size) + 1 if size in ordered else 2


class PDFParser:
    """解析 PDF 文件为结构化元素列表

    三级降级：
    1. PyMuPDF（主路径）—— 唯一能拿到每行字号的方式，"标题 vs 正文"才判得准；
       段落按 block 聚合，标题按字号识别。
    2. pypdf（纯文本兜底）—— 拿不到字号，整页合成一个段落，不做行级标题猜测。
    3. OCR（扫描型）—— 前两条都取不到文本层时才启用，整页合并为段落。

    为什么主路径从 pypdf 换成 PyMuPDF：pypdf 只给纯文本，没有字号信息，
    只能靠"短行长不长、末尾有没有标点"去猜标题，实测误判 84%。
    """

    def parse(self, file_path: str | Path, allow_ocr: bool = True) -> list[ParsedElement]:
        """
        解析 PDF 文件

        Args:
            file_path: PDF 文件路径
            allow_ocr: 文本层取不到时是否允许回退 OCR。
                OCR 很贵（CPU 上约 3 秒/页，一本 466 页的扫描书要 20 分钟以上），
                批量入库时建议显式决定要不要开，而不是默认被拖住。

        Returns:
            ParsedElement 列表（标题 + 按 block 聚合的段落）
        """
        file_path = Path(file_path)
        source = file_path.name

        # ---- 主路径：字号感知（PyMuPDF）----
        elements = self._parse_with_pymupdf(file_path, source)

        # ---- 兜底 1：PyMuPDF 不可用或没取到文本 → pypdf 纯文本（整页一段）----
        if not elements:
            print(f"[PDFParser] PyMuPDF 无产出，回退 pypdf 纯文本路径: {source}")
            elements = self._parse_with_pypdf(file_path, source)

        # ---- 兜底 2：整本都没有文本层 → OCR ----
        if not elements and allow_ocr:
            print(f"[PDFParser] 无文本层，尝试 OCR 解析: {source}")
            elements = self._parse_with_ocr(file_path, source)
        elif not elements:
            print(f"[PDFParser] 无文本层且未开 OCR，跳过: {source}"
                  f"（该文件需要 OCR，加 --ocr 才会处理）")

        if not elements:
            print(f"[PDFParser] 未能从 {source} 提取到文本内容")
            elements.append(ParsedElement(
                content=f"[PDF 文件: {source} — 未能提取文本内容]",
                element_type="paragraph",
                metadata={"source": source, "warning": "no_text_extracted"},
            ))

        return elements

    # ---------------------------------------------------------------
    # 主路径：PyMuPDF 字号感知
    # ---------------------------------------------------------------

    def _extract_page_lines(self, page) -> list[tuple[float, str]]:
        """按行提取 (字号, 文本)，顺手丢掉页眉页脚、页码与版权噪声"""
        try:
            data = page.get_text("dict")
        except Exception:
            return []

        page_height = float(getattr(page.rect, "height", 0) or 0)
        kept: list[tuple[float, str]] = []
        margin: list[tuple[float, str]] = []

        for block in data.get("blocks", []):
            if block.get("type") != 0:      # 0 = 文本块，1 = 图片
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue

                size = _norm_font_size(
                    max((s.get("size", 0) for s in spans), default=0)
                )

                # 版权页/出版信息/孤立页码：复用 OCR 那套噪声判据
                if _is_ocr_noise_line(text):
                    continue

                # 页眉页脚：贴页面上下边缘的短行（页码、书名通常在这里）
                if page_height and len(text) <= _PDF_MARGIN_MAX_CHARS:
                    y0 = (line.get("bbox") or [0, 0, 0, 0])[1]
                    if (y0 <= page_height * _PDF_MARGIN_RATIO
                            or y0 >= page_height * (1 - _PDF_MARGIN_RATIO)):
                        margin.append((size, text))
                        continue

                kept.append((size, text))

        # 边距判据的前提是「页面还有正文」——页眉页脚之所以是噪声，是因为
        # 正文另有其文。若整页文本都落在边距带内（实测图页配文就是这样，
        # 整页仅顶部一行 y0/h=0.02），那它就不是页眉，必须留下。
        return kept if kept else margin

    def _page_to_elements(
        self,
        lines: list[tuple[float, str]],
        heading_sizes: set,
        source: str,
        page_num: int,
    ) -> list[ParsedElement]:
        """一页的行 → 元素：字号命中的短行进 heading，其余按顺序聚成段落"""
        elements: list[ParsedElement] = []
        buffer: list[str] = []

        def flush():
            if not buffer:
                return
            text = _join_pdf_lines(buffer)
            buffer.clear()
            if text:
                elements.append(ParsedElement(
                    content=text,
                    element_type="paragraph",
                    metadata={"source": source, "page_num": page_num},
                ))

        for size, text in lines:
            if size in heading_sizes and len(text) <= _PDF_HEADING_MAX_CHARS:
                flush()
                elements.append(ParsedElement(
                    content=text,
                    element_type="heading",
                    metadata={
                        "source": source,
                        "page_num": page_num,
                        "heading_level": _heading_level(size, heading_sizes),
                    },
                ))
            else:
                buffer.append(text)
        flush()
        return elements

    def _parse_with_pymupdf(self, file_path: Path, source: str) -> list[ParsedElement]:
        """PyMuPDF 主路径：先摸清全文字号分布，再逐页把行组装成标题/段落"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("[PDFParser] PyMuPDF 未安装，跳过字号感知路径")
            return []

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            print(f"[PDFParser] PyMuPDF 打开失败: {e}")
            return []

        try:
            pages: list[tuple[int, list[tuple[float, str]]]] = []
            size_chars: dict = {}
            size_lines: dict = {}
            blank_pages = 0

            for page_num, page in enumerate(doc, start=1):
                lines = self._extract_page_lines(page)
                if not lines:
                    blank_pages += 1
                    continue
                pages.append((page_num, lines))
                for size, text in lines:
                    size_chars[size] = size_chars.get(size, 0) + len(text)
                    size_lines[size] = size_lines.get(size, 0) + 1

            if not pages:
                print(f"[PDFParser] {source} 全书无文本层（{blank_pages} 页），需要 OCR")
                return []

            body_size, heading_sizes = _pick_heading_sizes(size_chars, size_lines)
            print(f"[PDFParser] {source}: {len(pages)} 页有文本层"
                  f"（{blank_pages} 页无文本层未纳入，如需请走 OCR）；"
                  f"正文字号 {body_size}，标题字号 {sorted(heading_sizes, reverse=True)}")

            elements: list[ParsedElement] = []
            for page_num, lines in pages:
                elements.extend(
                    self._page_to_elements(lines, heading_sizes, source, page_num)
                )
            return elements
        finally:
            doc.close()

    # ---------------------------------------------------------------
    # 兜底 1：pypdf 纯文本（拿不到字号，整页一段，不猜标题）
    # ---------------------------------------------------------------

    def _parse_with_pypdf(self, file_path: Path, source: str) -> list[ParsedElement]:
        """pypdf 纯文本兜底：一页合成一个段落元素。

        这里刻意**不做**行级标题猜测——没有字号信息时，任何"短行=标题"的
        启发式都会把正文切碎（这正是本次修复要根治的问题）。
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            print("[PDFParser] pypdf 未安装")
            return []

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            print(f"[PDFParser] pypdf 打开失败: {e}")
            return []

        elements: list[ParsedElement] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            lines = [ln for ln in lines if not _is_ocr_noise_line(ln)]
            if not lines or _looks_like_toc_page(lines):
                continue
            content = _join_pdf_lines(lines)
            if content:
                elements.append(ParsedElement(
                    content=content,
                    element_type="paragraph",
                    metadata={"source": source, "page_num": page_num},
                ))

        if elements:
            print(f"[PDFParser] pypdf 纯文本路径: {source} 提取 {len(elements)} 页（整页成段，未做标题猜测）")
        return elements

    # ---------------------------------------------------------------
    # 兜底 2：OCR（扫描型 PDF）
    # ---------------------------------------------------------------

    def _parse_with_ocr(self, file_path: Path, source: str) -> list[ParsedElement]:
        """
        使用 OCR 解析扫描型 PDF

        需要 PyMuPDF (渲染页面为图片) 和 rapidocr-onnxruntime (OCR引擎)。
        解析时自动清洗噪声：版权页/出版信息行、孤立页码/符号、目录页。
        OCR 结果没有字号信息，因此整页合并为一个段落（不做标题猜测）。
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("[PDFParser] OCR 需要 PyMuPDF，请先安装: pip install PyMuPDF")
            return []

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            print("[PDFParser] OCR 需要 rapidocr-onnxruntime，请先安装: pip install rapidocr-onnxruntime")
            return []

        print(f"[PDFParser] 正在 OCR 解析: {source}（这可能需要几分钟）")

        elements: list[ParsedElement] = []
        ocr_engine = RapidOCR()

        try:
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            print(f"[PDFParser] 共 {total_pages} 页需要 OCR")
            total_raw_lines = 0
            total_dropped_lines = 0
            dropped_pages = 0

            for page_num, page in enumerate(doc, start=1):
                # 渲染页面为图片（DPI=200 平衡质量和速度）
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")

                # OCR 识别
                result, _ = ocr_engine(img_bytes)
                if not result:
                    continue

                # result 是 [[box, text, confidence], ...] 的列表
                page_texts = []
                for item in result:
                    text = item[1].strip()
                    try:
                        confidence = float(item[2]) if len(item) > 2 else 0
                    except (ValueError, TypeError):
                        confidence = 0
                    if text and confidence > 0.5:
                        page_texts.append(text)

                # ---- OCR 噪声清洗 ----
                raw_count = len(page_texts)
                total_raw_lines += raw_count
                if page_texts and _looks_like_toc_page(page_texts):
                    # 目录页整体跳过
                    dropped_pages += 1
                    total_dropped_lines += raw_count
                    page_texts = []
                else:
                    cleaned = [t for t in page_texts if not _is_ocr_noise_line(t)]
                    total_dropped_lines += raw_count - len(cleaned)
                    page_texts = cleaned

                if page_texts:
                    # 整页合并为一段：OCR 无字号信息，逐行判标题必然碎片化
                    elements.append(ParsedElement(
                        content=_join_pdf_lines(page_texts),
                        element_type="paragraph",
                        metadata={"source": source, "page_num": page_num, "ocr": True},
                    ))

                # 每 10 页打印进度
                if page_num % 10 == 0 or page_num == total_pages:
                    print(f"[PDFParser] OCR 进度: {page_num}/{total_pages} 页")

            doc.close()
            print(f"[PDFParser] OCR 完成，共提取 {len(elements)} 个元素"
                  f"（噪声清洗: 丢弃 {total_dropped_lines}/{total_raw_lines} 行，跳过 {dropped_pages} 个目录/噪声页）")

        except Exception as e:
            print(f"[PDFParser] OCR 解析失败: {e}")

        return elements


# ============================================================
# 代码文件解析器
# ============================================================

class CodeParser:
    """解析代码文件为结构化元素"""

    # 常见扩展名 → 语言名映射
    EXTENSION_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c_header",
        ".hpp": "cpp_header",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".rst": "rst",
        ".tex": "latex",
    }

    def parse(self, file_path: str | Path) -> list[ParsedElement]:
        """
        解析代码文件
        
        Args:
            file_path: 代码文件路径
        
        Returns:
            ParsedElement 列表（包含函数/类定义块和注释块）
        """
        file_path = Path(file_path)
        source = file_path.name
        ext = file_path.suffix.lower()
        language = self.EXTENSION_MAP.get(ext, "unknown")

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding="gbk")
            except UnicodeDecodeError:
                print(f"[CodeParser] 无法解码文件: {source}")
                return []

        elements: list[ParsedElement] = []

        # 整个文件作为一个代码元素
        elements.append(ParsedElement(
            content=text,
            element_type="code",
            metadata={
                "source": source,
                "language": language,
                "file_path": str(file_path),
            },
        ))

        # 如果代码较长，额外提取顶层函数/类定义作为独立元素
        if language == "python":
            elements.extend(self._extract_python_defs(text, source, language))
        elif language in ("javascript", "typescript", "java", "go", "rust"):
            elements.extend(self._extract_generic_defs(text, source, language))

        return elements

    def _extract_python_defs(
        self, text: str, source: str, language: str
    ) -> list[ParsedElement]:
        """提取 Python 函数和类定义"""
        elements: list[ParsedElement] = []
        pattern = re.compile(
            r"^((?:@\w+\s*\n)*)((?:class|def)\s+\w+[^\n]*\n)(.*?)(?=\n(?:@\w+\s*\n)?(?:class|def)\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for match in pattern.finditer(text):
            decorators = match.group(1).strip()
            header = match.group(2).strip()
            body = match.group(3).strip()
            content = f"{decorators}\n{header}\n{body}" if decorators else f"{header}\n{body}"
            # 提取定义名称
            name_match = re.match(r"(?:class|def)\s+(\w+)", header)
            def_name = name_match.group(1) if name_match else "unknown"
            elements.append(ParsedElement(
                content=content.strip(),
                element_type="code",
                metadata={
                    "source": source,
                    "language": language,
                    "definition": def_name,
                    "definition_type": "class" if header.startswith("class") else "function",
                },
            ))
        return elements

    def _extract_generic_defs(
        self, text: str, source: str, language: str
    ) -> list[ParsedElement]:
        """提取通用语言的函数/方法定义（简化版）"""
        elements: list[ParsedElement] = []
        # 匹配 function 关键字或类似定义
        patterns = [
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
            r"^(?:public|private|protected)?\s*(?:static\s+)?\w+\s+(\w+)\s*\([^)]*\)\s*\{",
            r"^(?:pub\s+)?(?:fn|func)\s+(\w+)",
        ]
        for pat in patterns:
            for match in re.finditer(pat, text, re.MULTILINE):
                # 简单提取：从定义行到下一个空行或同级缩进
                start = match.start()
                lines = text[start:].split("\n")
                def_lines = []
                for line in lines:
                    def_lines.append(line)
                    if line.strip() == "" and len(def_lines) > 1:
                        break
                content = "\n".join(def_lines).strip()
                elements.append(ParsedElement(
                    content=content,
                    element_type="code",
                    metadata={
                        "source": source,
                        "language": language,
                        "definition": match.group(1),
                        "definition_type": "function",
                    },
                ))
        return elements


# ============================================================
# 统一解析器入口
# ============================================================

class DocumentParser:
    """
    统一文档解析入口
    
    根据文件扩展名自动选择解析器。
    """

    # 文本文件扩展名列表（直接用 MarkdownParser 读取）
    TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".tex", ".json", ".yaml", ".yml", ".toml", ".csv"}

    # 中文语料里常见的老编码。UTF-8 解不开就按这个顺序试——
    # 实测王阳明语料里有 GBK 的 txt（'utf-8' codec can't decode byte 0xb7），
    # 之前直接 read_text(utf-8) 会让整本书解析失败、静默丢失。
    _TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "big5")

    @classmethod
    def _read_text(cls, file_path: Path) -> str:
        """按候选编码读文本，全失败才抛错"""
        last_err: Exception | None = None
        for enc in cls._TEXT_ENCODINGS:
            try:
                return file_path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError) as e:
                last_err = e
        raise ValueError(f"无法用 {cls._TEXT_ENCODINGS} 解码文本文件: {file_path}（{last_err}）")

    def __init__(self):
        self.markdown_parser = MarkdownParser()
        self.pdf_parser = PDFParser()
        self.code_parser = CodeParser()

    def parse(self, file_path: str | Path, allow_ocr: bool = True) -> list[ParsedElement]:
        """
        解析文档，自动识别格式
        
        Args:
            file_path: 文件路径
            allow_ocr: 仅对 PDF 生效；文本层取不到时是否回退 OCR（默认允许）
        
        Returns:
            ParsedElement 列表
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()

        # PDF
        if ext == ".pdf":
            return self.pdf_parser.parse(file_path, allow_ocr=allow_ocr)

        # Markdown / 文本文件
        if ext == ".md":
            text = self._read_text(file_path)
            return self.markdown_parser.parse(text, source=file_path.name)

        # 代码文件
        if ext in CodeParser.EXTENSION_MAP:
            return self.code_parser.parse(file_path)

        # 纯文本（按 Markdown 解析，但作为纯文本段落处理）
        if ext in self.TEXT_EXTENSIONS:
            text = self._read_text(file_path)
            return [ParsedElement(
                content=text.strip(),
                element_type="paragraph",
                metadata={"source": file_path.name},
            )]

        # 遇到未知文件扩展名，尝试作为文本读取
        try:
            text = self._read_text(file_path)
            return [ParsedElement(
                content=text.strip(),
                element_type="paragraph",
                metadata={"source": file_path.name, "extension": ext},
            )]
        except Exception:
            raise ValueError(
                f"不支持的文件格式: {ext}，且无法作为文本读取。"
                f"支持的格式: PDF, Markdown, 代码文件, TXT"
            )


# ============================================================
# 便捷函数
# ============================================================

def parse_document(file_path: str | Path) -> list[ParsedElement]:
    """解析文档的便捷函数"""
    parser = DocumentParser()
    return parser.parse(file_path)
