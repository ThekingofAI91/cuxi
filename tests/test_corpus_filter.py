# -*- coding: utf-8 -*-
"""
test_corpus_filter.py — 入库非正文块过滤的行为契约

这些块有两个坏性质：书名/核心概念重复十几遍（向量与 BM25 都判它最相关），
对生成却零价值。实测《原型与集体无意识》的版权页在一道问「原型和集体潜意识
是什么关系」的题里排到 `[1]`。

契约四条：
1. 强标记（ISBN / 图书在版编目 / CIP / 版权所有 / 出版发行）→ 判版权页，整块丢；
2. 点线+页码行 ≥ 3 行 → 判目录，只删行不删块，删完不足 200 字才丢；
3. 正文里顺口提到「出版社 / 定价」不能被误伤——这类词只对极短块生效；
4. 清洗必须发生在向量化之前，所以文本变了要同步 char_length。
"""

from langchain_core.documents import Document

from src.retrieval.corpus_filter import (
    clean_chunk,
    clean_chunks,
    front_matter_kind,
    is_front_matter,
)

COPYRIGHT = """中国华侨出版社
传习录：一本书读懂阳明心学
图书在版编目（CIP）数据
传习录：一本书读懂阳明心学／（明）王阳明著；陆东风编．—北京：中国华侨出版社，2013.9
ISBN 978-7-5113-4096-2
Ⅰ．①传… Ⅱ．①王…②陆… Ⅲ．①心学－中国－明代 Ⅳ．①B248.2
"""

TOC = """目  录
第一讲 潜意识下的人生追求
一、没有历史就没有心理学…… 5
二、探讨基督教………………… 9
三、继续摸索…………………… 14
"""

BODY = "荣格认为，原型是集体潜意识的内容之一。原型本身不是具体的意象，而是意象的先天形式。" * 5


class TestFrontMatterKind:
    def test_isbn_block_is_copyright_page(self):
        assert front_matter_kind(COPYRIGHT) == "版权页"
        assert is_front_matter(COPYRIGHT) is True

    def test_toc_block(self):
        assert front_matter_kind(TOC) == "目录"

    def test_body_mentioning_publisher_is_safe(self):
        """正文里顺口提出版社/定价，不能被当成版权页误删"""
        body = "他后来把书稿交给出版社，定价并不高，但销量一直上不去。" * 6
        assert front_matter_kind(body) == ""
        assert is_front_matter(body) is False

    def test_long_body_with_one_toc_line_is_safe(self):
        """只有一两行像目录，不能判目录（阈值 3 行）"""
        text = BODY + "\n第一章 原型…… 5\n" + BODY
        assert front_matter_kind(text) == ""

    def test_short_pub_fragment(self):
        assert front_matter_kind("作者：（奥）阿尔弗雷德·阿德勒出版时间：2019年11月") == "出版信息碎片"

    def test_tiny_block_is_debris(self):
        assert front_matter_kind("hello") == "碎片"
        assert front_matter_kind("   ") == "碎片"

    def test_plain_body_passes(self):
        assert front_matter_kind(BODY) == ""


class TestCleanChunk:
    def test_copyright_page_dropped_whole(self):
        text, reason = clean_chunk(COPYRIGHT)
        assert text == ""
        assert reason == "版权页"

    def test_toc_lines_stripped_body_kept(self):
        """目录页常和正文同块（章首导读 + 目录），只删目录行"""
        mixed = "这本书讲的是潜意识。" * 30 + "\n" + TOC
        text, reason = clean_chunk(mixed)
        assert reason == "目录"
        assert "这本书讲的是潜意识" in text
        assert "第一讲" not in text
        assert "……" not in text

    def test_toc_only_block_dropped(self):
        text, reason = clean_chunk(TOC)
        assert text == ""
        assert reason == "目录"

    def test_body_untouched(self):
        text, reason = clean_chunk(BODY)
        assert text == BODY
        assert reason == ""


class TestCleanChunks:
    def _docs(self):
        return [
            Document(page_content=COPYRIGHT, metadata={"char_length": len(COPYRIGHT)}),
            Document(page_content=BODY, metadata={"char_length": len(BODY)}),
            Document(page_content=TOC, metadata={"char_length": len(TOC)}),
            Document(page_content="hello", metadata={"char_length": 5}),
        ]

    def test_drops_and_counts(self):
        kept, dropped = clean_chunks(self._docs())
        assert len(kept) == 1
        assert kept[0].page_content == BODY
        assert dropped == {"版权页": 1, "目录": 1, "碎片": 1}

    def test_char_length_synced_after_clean(self):
        mixed = "正文内容。" * 40 + "\n" + TOC
        doc = Document(page_content=mixed, metadata={"char_length": len(mixed)})
        kept, _ = clean_chunks([doc])
        assert kept[0].metadata["char_length"] == len(kept[0].page_content)

    def test_empty_input(self):
        assert clean_chunks([]) == ([], {})
