"""块尾续接（方案 C）的单元测试。

两层：判据与拼接（运行期），回填脚本的两个纯函数（离线）。
"""

import importlib.util
from pathlib import Path

from langchain_core.documents import Document

from framework.analysis_agent import _build_context
from src.retrieval.stitch import needs_continuation, with_continuation
from src.retrieval.trace import RetrievalTrace


def _load_backfill():
    path = Path(__file__).resolve().parent.parent / "scripts" / "backfill_cont_tail.py"
    spec = importlib.util.spec_from_file_location("backfill_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestNeedsContinuation:
    def test_sentence_end_needs_nothing(self):
        assert needs_continuation("这句话是完整的。") is False
        assert needs_continuation("他问：真的吗？") is False
        assert needs_continuation("还没说完……") is False

    def test_mid_sentence_is_flagged(self):
        assert needs_continuation("我则以听天由命的态度处之：不然又能怎么办呢？我们恐") is True
        assert needs_continuation("重新聚焦这位伟大的思想家、哲学家、教育家、") is True

    def test_empty_and_punctuation_only(self):
        assert needs_continuation("") is False
        assert needs_continuation("   ") is False
        assert needs_continuation("，，、") is False


class TestWithContinuation:
    def test_appends_tail(self):
        text = "我们恐"
        out, cont = with_continuation(text, {"cont_tail": "怕得在这儿困上好几年呢！"})
        assert out == "我们恐\n怕得在这儿困上好几年呢！"
        assert cont == "怕得在这儿困上好几年呢！"

    def test_no_tail_keeps_text(self):
        assert with_continuation("我们恐", {}) == ("我们恐", "")
        assert with_continuation("我们恐", {"cont_tail": "  "}) == ("我们恐", "")

    def test_does_not_touch_complete_blocks(self):
        text = "完整的句子。"
        assert with_continuation(text, {"cont_tail": "不该被贴上"}) == (text, "")

    def test_markdown_is_skipped(self):
        """markdown 一行就是一条记录，没有「半句」这回事"""
        text = "我们恐"
        md = {"source": "01-writings.md", "cont_tail": "怕得困上几年！"}
        assert with_continuation(text, md) == (text, "")
        pdf = {"source": "荣格自传.pdf", "cont_tail": "怕得困上几年！"}
        assert with_continuation(text, pdf)[0] == "我们恐\n怕得困上几年！"


class TestBackfill:
    # 生产里相邻块重叠恰好 100 字，测试数据必须同量级，否则会被 next_sentence 判成不连续
    OV100 = ("我则以听天由命的态度处之不然又能怎么办呢我们恐" + "重叠" * 40)[:100]

    def test_next_sentence_needs_exact_overlap(self):
        mod = _load_backfill()
        prev = "前面的一大段原文，" + self.OV100
        nxt = self.OV100 + "\n怕得在这儿困上好几年呢！”接着我又有个想法："
        assert mod.next_sentence(prev, nxt) == "怕得在这儿困上好几年呢！"

    def test_next_sentence_rejects_discontinuity(self):
        """id 相邻但中间删过块 —— 前缀对不上，不能接"""
        mod = _load_backfill()
        prev = "字" * 200 + "目录"
        nxt = "荣格说潜意识与生存\n\n第一讲\n\n潜意识下的人生追求"
        assert mod.next_sentence(prev, nxt) == ""

    def test_next_sentence_rejects_short_prev(self):
        mod = _load_backfill()
        assert mod.next_sentence("只有几十个字。", "只有几十个字。后面还有") == ""

    def test_first_sentence_without_punct_returns_all(self):
        mod = _load_backfill()
        assert mod.first_sentence("  没有句号的整段文字") == "没有句号的整段文字"
        assert mod.first_sentence("第一句。第二句。") == "第一句。"

    def test_build_updates_is_idempotent(self):
        """每个块都写：不需要接的写成空串，旧值自然被覆盖"""
        mod = _load_backfill()
        prev = "开头，" + self.OV100
        nxt = self.OV100 + "怕得困上几年！"
        ids = ["x_0", "x_1"]
        docs = [prev, nxt]
        metas = [{"source": "a.pdf", "cont_tail": "旧的脏值"}, {"source": "a.pdf"}]
        updates, stat = mod.build_updates(ids, docs, metas)

        assert len(updates) == 2                       # 两块都被写
        assert updates[0][1]["cont_tail"] == "怕得困上几年！"
        assert updates[0][1]["cont_from"] == "x_1"
        assert updates[1][1]["cont_tail"] == ""         # 末块没有下一块（且它本身以句号结尾）
        assert stat["filled"] == 1 and stat["no_next"] == 0

    def test_markdown_source_gets_no_tail(self):
        """markdown 语料整类不参与拼接，即使两块文本连续"""
        mod = _load_backfill()
        prev = "开头，" + self.OV100
        nxt = self.OV100 + "怕得困上几年！"
        updates, stat = mod.build_updates(
            ["m_0", "m_1"], [prev, nxt],
            [{"source": "01-writings.md"}, {"source": "01-writings.md"}],
        )
        assert stat["skipped_md"] == 2 and stat["filled"] == 0
        assert [u[1]["cont_tail"] for u in updates] == ["", ""]


class TestBuildContextStitches:
    def test_stitched_text_enters_prompt_and_keeps_numbering(self):
        docs = [
            Document(page_content="我们恐", metadata={"source": "a.pdf", "cont_tail": "怕得困上几年！"}),
            Document(page_content="另一段完整的原文。", metadata={"source": "b.pdf"}),
        ]
        trace = RetrievalTrace()
        ctx = _build_context(docs, trace)

        assert "我们恐\n怕得困上几年！" in ctx
        # 续接句不新开编号：仍然只有 [1][2]
        assert "[1] 原文" in ctx and "[2] 原文" in ctx and "[3]" not in ctx
        info = [v for v in trace.boundary.values() if v["kind"] == "source"]
        assert sorted(v["stitched_chars"] for v in info) == [0, len("怕得困上几年！")]

    def test_graph_evidence_is_not_stitched(self):
        docs = [Document(page_content="推断片段", metadata={"kg_inferred": True, "cont_tail": "不该接"})]
        assert "不该接" not in _build_context(docs)
