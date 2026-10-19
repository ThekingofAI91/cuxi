# -*- coding: utf-8 -*-
"""
test_citation_grading.py — 引用分级的行为契约

背景：旧实现把「检索到的原文块」和「图谱推断证据」混编成一个 1..33 的序列，
模型看到 `[31] 来源: 荣格自传` 就当成原著引用了，读者也无从分辨哪句是原话、
哪句是三元组归纳出来的关系。

现在两套编号并存：
  正文块   → [1]..[N]，头标「原文｜来源: …」
  图谱证据 → [G1]..[Gn]，头标「推断｜非原文，依据: …」

这份测试锁住四件事：
1. 两套编号各自从 1 开始，互不占用对方的号码；
2. 前端 doc_map 的编号与 _build_context 的编号逐条对齐（错位就会指错出处）；
3. 推断块必须带「非原文」字样，正文块保留来源/章节；
4. 正文块仍按 1000 字截断（注入口径不变）。
"""

from langchain_core.documents import Document

from framework.analysis_agent import _build_context
from src.api.routes import _build_doc_map


def _plain(text: str = "正文内容", source: str = "荣格自传.pdf", heading: str = "第二章") -> Document:
    return Document(page_content=text, metadata={"source": source, "heading": heading})


def _infer(relation: str = "原型 —属于→ 集体潜意识") -> Document:
    return Document(
        page_content=f"{relation}。（依据：原型与集体无意识.pdf）",
        metadata={
            "kg_inferred": True,
            "kg_relation": relation,
            "source": "原型与集体无意识.pdf",
            "heading": "第二章 原型",
        },
    )


class TestTwoSetNumbering:
    def test_plain_start_at_one(self):
        ctx = _build_context([_plain(), _plain()])
        assert "[1] 原文" in ctx
        assert "[2] 原文" in ctx
        assert "[G1]" not in ctx

    def test_infer_uses_g_prefix(self):
        ctx = _build_context([_infer()])
        assert "[G1] 推断" in ctx
        assert "[1] 原文" not in ctx

    def test_series_do_not_share_numbers(self):
        """混排时两套编号各自独立：3 条正文 + 2 条推断 → [1][2][3] + [G1][G2]"""
        docs = [_plain("甲"), _infer("甲—乙"), _plain("乙"), _infer("乙—丙"), _plain("丙")]
        ctx = _build_context(docs)
        for tag in ("[1] 原文", "[2] 原文", "[3] 原文", "[G1] 推断", "[G2] 推断"):
            assert tag in ctx
        # 不存在第 4 条正文，也不存在第 3 条推断
        assert "[4] 原文" not in ctx
        assert "[G3]" not in ctx

    def test_infer_order_follows_doc_order(self):
        """推断编号按文档出现顺序给，不按关系名排序"""
        ctx = _build_context([_infer("第一条"), _infer("第二条")])
        assert ctx.index("第一条") < ctx.index("第二条")
        first = ctx.split("\n\n---\n\n")[0]
        assert first.startswith("[G1] 推断")


class TestHeaderContent:
    def test_plain_keeps_source_and_heading(self):
        ctx = _build_context([_plain()])
        assert "来源: 荣格自传.pdf" in ctx
        assert "章节: 第二章" in ctx

    def test_infer_marks_not_original(self):
        ctx = _build_context([_infer()])
        assert "非原文" in ctx
        assert "依据:" in ctx

    def test_plain_not_truncated_under_limit(self):
        """块的实际长度是 1000~1499（max_size + overlap + 短尾合并），
        注入上限 1500 是保险丝——正常块不该被切到。"""
        ctx = _build_context([_plain("字" * 1499)])
        assert ctx.count("字") == 1499

    def test_plain_cut_at_sentence_boundary(self):
        """超上限时截在句末，不在半句处硬砍"""
        text = "甲" * 800 + "。" + "乙" * 900 + "。" + "丙" * 300
        ctx = _build_context([_plain(text)])
        assert "丙" not in ctx              # 超出 1500 的部分丢掉
        assert ctx.rstrip().endswith("。")   # 结尾是句号，不是半句
        assert ctx.count("乙") == 0          # 第二句超窗口，整句不进

    def test_plain_hard_cut_without_punctuation(self):
        """整段没有句读时才硬切到上限"""
        ctx = _build_context([_plain("字" * 3000)])
        assert ctx.count("字") == 1500

    def test_infer_not_truncated(self):
        """推断证据来自 graph_evidence_chars=260 的短句，无需再截"""
        doc = _infer()
        doc.page_content = "甲" * 400
        assert _build_context([doc]).count("甲") == 400

    def test_empty_returns_placeholder(self):
        assert _build_context([]) == "（无参考资料）"

    def test_missing_metadata_tolerated(self):
        ctx = _build_context([Document(page_content="无元数据", metadata={})])
        assert "未知来源" in ctx
        assert "未知章节" in ctx


class TestDocMapParity:
    """doc_map 与 _build_context 的编号必须逐条对齐，否则角标指错出处"""

    def test_numbers_match_context(self):
        docs = [_plain("甲"), _infer("甲—乙"), _plain("乙")]
        ctx = _build_context(docs)
        dm = _build_doc_map(docs)
        assert [d["n"] for d in dm] == ["1", "G1", "2"]
        assert dm[0]["source"] in ctx
        assert dm[1]["kind"] == "infer"
        assert dm[2]["kind"] == "source"

    def test_kind_field(self):
        dm = _build_doc_map([_infer(), _plain()])
        assert [d["kind"] for d in dm] == ["infer", "source"]
        assert [d["n"] for d in dm] == ["G1", "1"]

    def test_source_ext_stripped(self):
        dm = _build_doc_map([_plain(source="荣格自传.pdf")])
        assert dm[0]["source"] == "荣格自传"

    def test_empty(self):
        assert _build_doc_map([]) == []
        assert _build_doc_map(None) == []
