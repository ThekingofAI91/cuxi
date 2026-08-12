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
]


def _is_ocr_noise_line(line: str) -> bool:
    """判断 OCR 行是否为噪声（版权页/出版信息/孤立页码/符号串）"""
    compact = re.sub(r"\s+", "", line)
    if not compact:
        return True
    # 孤立页码/数字（≤6 位）
    if re.fullmatch(r"\d{1,6}", compact):
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

class PDFParser:
    """解析 PDF 文件为结构化元素列表（按页面提取）
    
    支持文本型 PDF 和扫描型 PDF（通过 OCR）。
    扫描型 PDF 需要安装 PyMuPDF 和 rapidocr-onnxruntime。
    """

    def parse(self, file_path: str | Path) -> list[ParsedElement]:
        """
        解析 PDF 文件
        
        Args:
            file_path: PDF 文件路径
        
        Returns:
            ParsedElement 列表（每页一个段落元素）
        """
        file_path = Path(file_path)
        source = file_path.name

        try:
            from pypdf import PdfReader
        except ImportError:
            print("[PDFParser] pypdf 未安装，尝试使用 PyMuPDF...")
            return self._parse_with_pymupdf(file_path, source)

        elements: list[ParsedElement] = []
        reader = PdfReader(str(file_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # 尝试识别标题（通常是大字号或居中的文本）
            lines = text.strip().split("\n")
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                # 简单的标题启发式：短行（<50字）且不包含句号
                if len(stripped) < 50 and not stripped.endswith(("。", "？", "！", ".", "?", "!")):
                    elements.append(ParsedElement(
                        content=stripped,
                        element_type="heading",
                        metadata={
                            "source": source,
                            "page_num": page_num,
                            "heading_level": 2,
                        },
                    ))
                else:
                    elements.append(ParsedElement(
                        content=stripped,
                        element_type="paragraph",
                        metadata={
                            "source": source,
                            "page_num": page_num,
                        },
                    ))

        if not elements:
            # 文本提取失败，尝试 OCR
            print(f"[PDFParser] ⚠️ 文本提取失败，尝试 OCR 解析: {source}")
            elements = self._parse_with_ocr(file_path, source)

        if not elements:
            print(f"[PDFParser] ⚠️ 未能从 {source} 提取到文本内容")
            elements.append(ParsedElement(
                content=f"[PDF 文件: {source} — 未能提取文本内容]",
                element_type="paragraph",
                metadata={"source": source, "warning": "no_text_extracted"},
            ))

        return elements

    def _parse_with_pymupdf(self, file_path: Path, source: str) -> list[ParsedElement]:
        """使用 PyMuPDF (fitz) 作为备选解析引擎"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("[PDFParser] PyMuPDF 也未安装，返回空结果")
            return []

        elements: list[ParsedElement] = []
        doc = fitz.open(str(file_path))

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text or not text.strip():
                continue

            lines = text.strip().split("\n")
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if len(stripped) < 50 and not stripped.endswith(("。", "？", "！", ".", "?", "!")):
                    elements.append(ParsedElement(
                        content=stripped, element_type="heading",
                        metadata={"source": source, "page_num": page_num, "heading_level": 2},
                    ))
                else:
                    elements.append(ParsedElement(
                        content=stripped, element_type="paragraph",
                        metadata={"source": source, "page_num": page_num},
                    ))

        doc.close()
        
        if not elements:
            elements = self._parse_with_ocr(file_path, source)
        
        return elements

    def _parse_with_ocr(self, file_path: Path, source: str) -> list[ParsedElement]:
        """
        使用 OCR 解析扫描型 PDF
        
        需要 PyMuPDF (渲染页面为图片) 和 rapidocr-onnxruntime (OCR引擎)。
        解析时自动清洗噪声：版权页/出版信息行、孤立页码/符号、目录页。
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

        print(f"[PDFParser] 🔍 正在 OCR 解析: {source}（这可能需要几分钟）")
        
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
                    page_text = "\n".join(page_texts)
                    # 简单的标题启发式
                    for line in page_texts:
                        if len(line) < 50 and not line.endswith(("。", "？", "！", ".", "?", "!")):
                            elements.append(ParsedElement(
                                content=line,
                                element_type="heading",
                                metadata={"source": source, "page_num": page_num, "heading_level": 2, "ocr": True},
                            ))
                        else:
                            elements.append(ParsedElement(
                                content=line,
                                element_type="paragraph",
                                metadata={"source": source, "page_num": page_num, "ocr": True},
                            ))

                # 每 10 页打印进度
                if page_num % 10 == 0 or page_num == total_pages:
                    print(f"[PDFParser] OCR 进度: {page_num}/{total_pages} 页")

            doc.close()
            print(f"[PDFParser] ✅ OCR 完成，共提取 {len(elements)} 个元素"
                  f"（噪声清洗: 丢弃 {total_dropped_lines}/{total_raw_lines} 行，跳过 {dropped_pages} 个目录/噪声页）")

        except Exception as e:
            print(f"[PDFParser] ❌ OCR 解析失败: {e}")

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
                print(f"[CodeParser] ⚠️ 无法解码文件: {source}")
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

    def __init__(self):
        self.markdown_parser = MarkdownParser()
        self.pdf_parser = PDFParser()
        self.code_parser = CodeParser()

    def parse(self, file_path: str | Path) -> list[ParsedElement]:
        """
        解析文档，自动识别格式
        
        Args:
            file_path: 文件路径
        
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
            return self.pdf_parser.parse(file_path)

        # Markdown / 文本文件
        if ext == ".md":
            text = file_path.read_text(encoding="utf-8")
            return self.markdown_parser.parse(text, source=file_path.name)

        # 代码文件
        if ext in CodeParser.EXTENSION_MAP:
            return self.code_parser.parse(file_path)

        # 纯文本（按 Markdown 解析，但作为纯文本段落处理）
        if ext in self.TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8")
            return [ParsedElement(
                content=text.strip(),
                element_type="paragraph",
                metadata={"source": file_path.name},
            )]

        # 遇到未知文件扩展名，尝试作为文本读取
        try:
            text = file_path.read_text(encoding="utf-8")
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
