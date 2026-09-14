# -*- coding: utf-8 -*-
"""
test_light_context.py — 娱乐区轻量上下文的行为契约

娱乐区检索回来的资料必须以"角色自己的记忆"身份进入 prompt。
这份测试锁住三件容易被后续改动悄悄破坏的事：
1. 不能出现"来源 / 章节"这类学术标注（会诱导模型说出"根据《xxx》"，人设当场崩）
2. 长度必须远小于教育区（撑大 prefill 会拖慢首字，且容易把回答带成长篇）
3. 空结果返回空串而不是"（无参考资料）"——娱乐区不该出现这种系统腔
"""

from langchain_core.documents import Document

from framework.analysis_agent import _build_context, _build_light_context


def _docs(n: int = 3, length: int = 900) -> list[Document]:
    return [
        Document(
            page_content=("内容" * length)[:length],
            metadata={"source": "某访谈.md", "heading": "第三章 底层叙事"},
        )
        for _ in range(n)
    ]


class TestBuildLightContext:
    def test_no_source_or_heading_leak(self):
        ctx = _build_light_context(_docs())
        assert "来源" not in ctx
        assert "章节" not in ctx
        assert "某访谈.md" not in ctx

    def test_education_keeps_source(self):
        """对照组：教育区必须保留来源，否则无法溯源"""
        ctx = _build_context(_docs(1))
        assert "某访谈.md" in ctx
        assert "第三章" in ctx

    def test_much_shorter_than_education(self):
        docs = _docs(3)
        light = _build_light_context(docs)
        edu = _build_context(docs)
        assert len(light) < len(edu) * 0.4

    def test_respects_per_doc_limit(self):
        # 3 条 × 150 字上限，加上"· "前缀与换行，总量应控制在 500 字内
        assert len(_build_light_context(_docs(3))) < 500

    def test_collapses_whitespace(self):
        doc = [Document(page_content="第一行\n\n第二行   有很多空格", metadata={})]
        ctx = _build_light_context(doc)
        assert "\n" not in ctx
        assert "第一行 第二行 有很多空格" in ctx

    def test_empty_returns_empty_string(self):
        assert _build_light_context([]) == ""

    def test_skips_blank_content(self):
        docs = [
            Document(page_content="", metadata={}),
            Document(page_content="有内容", metadata={}),
        ]
        ctx = _build_light_context(docs)
        assert ctx.strip() == "· 有内容"

    def test_tolerates_missing_page_content(self):
        docs = [Document(page_content="", metadata={})]
        assert _build_light_context(docs) == ""
