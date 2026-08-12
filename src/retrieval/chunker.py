"""
chunker.py — 自适应文档分块

核心思路：
  不搞"固定 N 个字符一刀切"的 naive 分块，
  而是基于文档结构（标题/段落/列表/代码块）智能分块。

策略：
  1. 以语义段落（ParsedElement）为基本单位
  2. 短段落向上合并（直到达到 min_chunk_size）
  3. 长段落向下分割（在句子边界截断，不超过 max_chunk_size）
  4. 相邻 chunk 之间添加重叠（overlap）
  5. 保留完整标题层级链（方便溯源）
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document

from src.document_processing.parser import ParsedElement
from src.core.config import settings


# ============================================================
# 句子分割工具（用于长段落切割）
# ============================================================

# 中文/英文句子边界
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?\n])\s*")


def _split_sentences(text: str) -> list[str]:
    """将文本按句子边界分割"""
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# 自适应 Chunker
# ============================================================

class AdaptiveChunker:
    """
    自适应文档分块器
    
    用法:
        chunker = AdaptiveChunker(min_size=200, max_size=1500, overlap=100)
        chunks = chunker.chunk(elements, source="example.pdf")
    """

    def __init__(
        self,
        min_size: int = 200,
        max_size: int = 1500,
        overlap: int = 100,
    ):
        """
        Args:
            min_size: 最小分块大小（字符数），短于此的段落会向上合并
            max_size: 最大分块大小（字符数），长于此的段落会向下分割
            overlap: 相邻 chunk 之间的重叠字符数
        """
        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap

    def chunk(
        self,
        elements: list[ParsedElement],
        source: str = "",
    ) -> list[Document]:
        """
        将解析后的元素列表分块为 LangChain Document 列表
        
        Args:
            elements: ParsedElement 列表（来自 DocumentParser）
            source: 来源标识（文件名），覆盖元素中的 source
        
        Returns:
            Document 列表，每个 Document 包含 content 和 metadata
        """
        if not elements:
            return []

        # ---- Step 1: 按标题层级分组 ----
        groups = self._group_by_heading(elements)

        # ---- Step 2: 每组内部分块 ----
        chunks: list[Document] = []
        current_heading_chain: list[str] = []  # 当前标题链

        for group in groups:
            heading_chain = self._extract_heading_chain(group, current_heading_chain)
            paragraphs = [e for e in group if e.element_type != "heading"]

            if not paragraphs:
                # 纯标题组（标题后无内容）
                heading_text = " > ".join(heading_chain) if heading_chain else ""
                if heading_text:
                    chunks.append(self._make_chunk(
                        content=heading_text,
                        heading_chain=heading_chain,
                        source=source or (group[0].metadata.get("source", "") if group else ""),
                        element_types=["heading"],
                    ))
                continue

            # 对段落分组分块
            group_chunks = self._chunk_paragraphs(paragraphs, heading_chain, source)
            chunks.extend(group_chunks)

        # ---- Step 3: 添加重叠 ----
        chunks = self._add_overlap(chunks)

        return chunks

    # ---------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------

    def _group_by_heading(self, elements: list[ParsedElement]) -> list[list[ParsedElement]]:
        """
        按标题分组：每个标题及其后续内容为一组
        连续的标题只保留最后一个（前面的作为父标题链）
        """
        groups: list[list[ParsedElement]] = []
        current_group: list[ParsedElement] = []

        for elem in elements:
            if elem.element_type == "heading":
                # 遇到新标题，保存当前组
                if current_group:
                    groups.append(current_group)
                    current_group = []
                current_group.append(elem)
            else:
                current_group.append(elem)

        if current_group:
            groups.append(current_group)

        return groups

    def _extract_heading_chain(
        self, group: list[ParsedElement], current_heading_chain: list[str]
    ) -> list[str]:
        """
        从一组元素中提取标题链（包含父标题）
        更新 current_heading_chain 并返回新的链
        """
        headings = [e for e in group if e.element_type == "heading"]

        if not headings:
            return list(current_heading_chain)

        # 获取标题层级
        heading_levels = []
        for h in headings:
            level = h.metadata.get("heading_level", 2)
            heading_levels.append((level, h.content))

        # 从 current_heading_chain 中截断：保留比第一个标题层级更低的父标题
        if heading_levels:
            first_level = heading_levels[0][0]
            # 截断 current_heading_chain 到对应层级深度
            current_heading_chain = current_heading_chain[: first_level - 1]

        # 追加新标题
        for level, title in heading_levels:
            if level <= len(current_heading_chain) + 1:
                # 替换同级标题
                pos = level - 1
                if pos < len(current_heading_chain):
                    current_heading_chain[pos] = title
                    current_heading_chain = current_heading_chain[: pos + 1]
                else:
                    current_heading_chain.append(title)
            else:
                current_heading_chain.append(title)

        return list(current_heading_chain)

    def _chunk_paragraphs(
        self,
        paragraphs: list[ParsedElement],
        heading_chain: list[str],
        source: str,
    ) -> list[Document]:
        """
        将一组段落分块
        
        策略：
        - 累积段落直到达到 min_size
        - 达到 max_size 时强制分割
        - 尽量在段落边界分割
        """
        chunks: list[Document] = []
        buffer: list[ParsedElement] = []
        buffer_size = 0

        def flush_buffer():
            """将缓冲区写入一个 chunk"""
            nonlocal buffer, buffer_size
            if not buffer:
                return
            content = "\n\n".join(e.content for e in buffer)
            element_types = list(dict.fromkeys(e.element_type for e in buffer))
            src = source or (buffer[0].metadata.get("source", "") if buffer else "")
            chunks.append(self._make_chunk(
                content=content,
                heading_chain=heading_chain,
                source=src,
                element_types=element_types,
            ))
            buffer = []
            buffer_size = 0

        for para in paragraphs:
            para_len = len(para)

            # 单个段落超过 max_size → 强制分割
            if para_len > self.max_size:
                # 先 flush 当前缓冲区
                if buffer:
                    flush_buffer()
                # 将长段落按句子分割
                sub_chunks = self._split_long_paragraph(para, heading_chain, source)
                chunks.extend(sub_chunks)
                continue

            # 如果加入当前段落会超过 max_size → flush
            if buffer_size + para_len > self.max_size:
                flush_buffer()

            buffer.append(para)
            buffer_size += para_len

        # 最后一段
        if buffer:
            flush_buffer()

        return chunks

    def _split_long_paragraph(
        self,
        paragraph: ParsedElement,
        heading_chain: list[str],
        source: str,
    ) -> list[Document]:
        """将长段落按句子边界分割为多个 chunk"""
        sentences = _split_sentences(paragraph.content)
        chunks: list[Document] = []
        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # 单个句子就超过 max_size → 强制截断
            if sentence_len > self.max_size:
                if current_chunk:
                    content = "".join(current_chunk)
                    src = source or paragraph.metadata.get("source", "")
                    chunks.append(self._make_chunk(
                        content=content,
                        heading_chain=heading_chain,
                        source=src,
                        element_types=[paragraph.element_type],
                    ))
                    current_chunk = []
                    current_size = 0
                # 暴力截断长句
                start = 0
                while start < sentence_len:
                    end = min(start + self.max_size, sentence_len)
                    content = sentence[start:end]
                    src = source or paragraph.metadata.get("source", "")
                    chunks.append(self._make_chunk(
                        content=content,
                        heading_chain=heading_chain,
                        source=src,
                        element_types=[paragraph.element_type],
                    ))
                    start = end
                continue

            if current_size + sentence_len > self.max_size:
                content = "".join(current_chunk)
                src = source or paragraph.metadata.get("source", "")
                chunks.append(self._make_chunk(
                    content=content,
                    heading_chain=heading_chain,
                    source=src,
                    element_types=[paragraph.element_type],
                ))
                current_chunk = [sentence]
                current_size = sentence_len
            else:
                current_chunk.append(sentence)
                current_size += sentence_len

        if current_chunk:
            content = "".join(current_chunk)
            src = source or paragraph.metadata.get("source", "")
            chunks.append(self._make_chunk(
                content=content,
                heading_chain=heading_chain,
                source=src,
                element_types=[paragraph.element_type],
            ))

        return chunks

    def _add_overlap(self, chunks: list[Document]) -> list[Document]:
        """
        在相邻 chunk 之间添加重叠内容
        
        策略：将前一个 chunk 末尾的 overlap 字符添加到当前 chunk 开头
        """
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: list[Document] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # 从前一个 chunk 末尾取 overlap 字符
            prev_content = prev_chunk.page_content
            overlap_text = prev_content[-self.overlap:] if len(prev_content) > self.overlap else prev_content

            if overlap_text:
                # 合并 metadata
                merged_metadata = dict(curr_chunk.metadata)
                merged_metadata["has_overlap"] = True
                merged_metadata["overlap_source"] = prev_chunk.metadata.get("heading_chain", "")

                result.append(Document(
                    page_content=overlap_text + "\n" + curr_chunk.page_content,
                    metadata=merged_metadata,
                ))
            else:
                result.append(curr_chunk)

        return result

    def _make_chunk(
        self,
        content: str,
        heading_chain: list[str],
        source: str,
        element_types: list[str],
    ) -> Document:
        """创建一个带完整 metadata 的 Document"""
        content = content.strip()
        # ChromaDB 不支持空列表作为 metadata 值，将列表转为字符串
        heading_str = " > ".join(heading_chain) if heading_chain else ""
        element_types_str = ",".join(element_types) if element_types else ""

        if not content:
            # 返回一个占位 Document，避免空片段
            return Document(
                page_content="",
                metadata={"source": source, "heading": heading_str, "empty": True},
            )

        return Document(
            page_content=content,
            metadata={
                "source": source,
                "heading": heading_str,
                "element_types": element_types_str,
                "char_length": len(content),
            },
        )


# ============================================================
# 便捷函数
# ============================================================

def chunk_elements(
    elements: list[ParsedElement],
    source: str = "",
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Document]:
    """
    自适应分块的便捷函数
    
    Args:
        elements: 解析后的元素列表
        source: 来源标识
        min_size: 最小分块大小（默认 config.chunk_size 的一半）
        max_size: 最大分块大小（默认 config.chunk_size）
        overlap: 重叠大小（默认 config.chunk_overlap）
    
    Returns:
        Document 列表
    """
    chunker = AdaptiveChunker(
        min_size=min_size or (settings.chunk_size // 2),
        max_size=max_size or settings.chunk_size,
        overlap=overlap or settings.chunk_overlap,
    )
    return chunker.chunk(elements, source=source)


def chunk_text(
    text: str,
    source: str = "",
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Document]:
    """
    直接分块纯文本的便捷函数（跳过解析步骤）
    
    适用于不需要结构化解析的场景。
    """
    from src.document_processing.parser import ParsedElement

    elements = [ParsedElement(
        content=text.strip(),
        element_type="paragraph",
        metadata={"source": source},
    )]
    return chunk_elements(elements, source=source, min_size=min_size, max_size=max_size, overlap=overlap)
