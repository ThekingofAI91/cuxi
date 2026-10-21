# -*- coding: utf-8 -*-
"""
test_retrieval_trace.py — 检索轨迹的契约测试

轨迹是「评测归因」的地基：如果 boards 记的不是融合真正吃进去的那批名次，
后面所有「为什么这个配置没赚分」的分析都是废的。所以这里锁三条：

1. 每一路召回都要有一条 boards 记录，且标清所属类（vector/bm25）与产出它的查询；
2. 打开改写后，变体与 HyDE 的榜要分别标成 variant1/variant2/hyde，不能糊成一条；
3. 传不传 trace 都不改变检索结果——采集必须是纯旁路。
"""

import asyncio

import pytest
from langchain_core.documents import Document

from src.retrieval import advanced_search as AS
from src.retrieval import knowledge_graph as KG
from src.retrieval.advanced_search import fuse_by_class
from src.retrieval.trace import RetrievalTrace
from src.core.config import settings
from framework.analysis_agent import _build_context


class _FakeEmbedder:
    """不做真编码：只保证 embed_documents 的返回条数与入参对齐"""

    def embed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]


class _FakeBM25:
    """BM25 替身：命中顺序就是文档原序，document_count 供日志打印"""

    def __init__(self, docs):
        self._docs = docs
        self.document_count = len(docs)

    def search(self, query, top_k):
        return [(d, 1.0) for d in self._docs[:top_k]]


class _FakeCollection:
    """向量库替身：每条查询都回同一批文档，名次 = 原序"""

    name = "persona_test"

    def __init__(self, docs):
        self._docs = docs

    def query(self, query_embeddings=None, n_results=10, include=None):
        chosen = self._docs[:n_results]
        groups = [[d.page_content for d in chosen] for _ in query_embeddings]
        metas = [[d.metadata for d in chosen] for _ in query_embeddings]
        dists = [[0.01 * i for i in range(len(chosen))] for _ in query_embeddings]
        return {"documents": groups, "metadatas": metas, "distances": dists}


def _docs(n: int = 8):
    return [
        Document(
            page_content=f"文档{i}：" + "正文内容" * 40,
            metadata={"source": f"book{i}.pdf", "heading": f"章{i}"},
        )
        for i in range(n)
    ]


def _run(rewrite=None, trace=None, n_docs=8, use_rerank=True, rerank_fn=None,
         top_k=5, anchor_k=0):
    """跑一次 advanced_retrieval（全替身，不碰真模型与真向量库）"""
    docs = _docs(n_docs)
    collection = _FakeCollection(docs)
    identity_rerank = lambda q, ds, top_k=None: list(ds)  # noqa: E731

    async def _fake_rewrite(question, llm, num_variants):
        return rewrite

    async def _go():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(AS, "_ensure_bm25_ready", lambda coll: (docs, _FakeBM25(docs)))
            mp.setattr("src.retrieval.embedder.get_embedder", lambda: _FakeEmbedder())
            mp.setattr(
                "src.retrieval.reranker.rerank_documents",
                rerank_fn or identity_rerank,
            )
            if rewrite is not None:
                mp.setattr(AS, "_rewrite_with_budget", _fake_rewrite)
            return await AS.advanced_retrieval(
                question="测试问题",
                collection=collection,
                llm=None,
                top_k=top_k,
                use_multi_query=rewrite is not None,
                use_hyde=rewrite is not None,
                use_rerank=use_rerank,
                anchor_k=anchor_k,
                trace_out=trace,
            )

    return asyncio.run(_go())


class TestBoards:
    def test_one_record_per_recall_board(self):
        """关改写：向量类 1 条榜 + BM25 类 1 条榜 = 2 条记录"""
        trace = RetrievalTrace()
        _run(trace=trace)

        assert len(trace.boards) == 2
        assert [b["class"] for b in trace.boards] == ["vector", "bm25"]
        assert [b["route"] for b in trace.boards] == ["original", "original"]
        assert all(b["hits"] > 0 for b in trace.boards)

    def test_records_are_ranked_doc_ids(self):
        """ids 必须是完整名次（不是截断的头几名），且与 hits 对得上"""
        trace = RetrievalTrace()
        docs, _ = _run(trace=trace)

        board = trace.boards[0]
        assert len(board["ids"]) == board["hits"]
        # 名次第一的 id 必须就是最终候选池里的那条（同源文本哈希）
        assert board["ids"][0] == AS._get_doc_id(docs[0].page_content)

    def test_variants_and_hyde_get_own_routes(self):
        """开改写：原问 + 2 变体 + HyDE 各算一条向量榜，变体再各算一条 BM25 榜"""
        trace = RetrievalTrace()
        _run(rewrite=(["变体一", "变体二"], "一段假设答案"), trace=trace)

        vector_routes = [b["route"] for b in trace.boards if b["class"] == "vector"]
        bm25_routes = [b["route"] for b in trace.boards if b["class"] == "bm25"]
        assert vector_routes == ["original", "variant1", "variant2", "hyde"]
        assert bm25_routes == ["original", "variant1", "variant2"]

    def test_question_is_recorded(self):
        trace = RetrievalTrace()
        _run(trace=trace)
        assert trace.question == "测试问题"


class TestTraceIsPureBypass:
    def test_results_identical_with_and_without_trace(self):
        """采集不能改变检索结果：同样的输入，带不带 trace 都要吐出同一批上下文"""
        docs_without, ctx_without = _run()
        docs_with, ctx_with = _run(trace=RetrievalTrace())

        assert ctx_without == ctx_with
        assert [d.page_content for d in docs_without] == [d.page_content for d in docs_with]

    def test_to_dict_round_trips(self):
        trace = RetrievalTrace(question="q", character="jung")
        _run(trace=trace)
        payload = trace.to_dict()

        assert payload["question"] == "q"
        assert payload["character"] == "jung"
        assert isinstance(payload["boards"], list) and payload["boards"]


class TestFused:
    def test_scores_match_fusion_recomputed_from_boards(self):
        """fused 里的分数必须能用 boards 复算出来——两处口径一旦分叉，归因就失效"""
        trace = RetrievalTrace()
        _run(trace=trace)

        boards = [(b["class"], b["ids"]) for b in trace.boards]
        meta = {doc_id: {"source": info["source"]} for doc_id, info in trace.fused.items()}
        recomputed = fuse_by_class(
            boards, meta,
            {"vector": settings.rrf_vector_weight, "bm25": settings.rrf_bm25_weight},
            use_source_weight=True,
        )

        for doc_id, info in trace.fused.items():
            assert info["score"] == pytest.approx(recomputed[doc_id], abs=1e-5)

    def test_contrib_sums_to_score(self):
        """contrib 是分数拆项，加回去必须等于原分（否则「谁捞的」对不上账）"""
        trace = RetrievalTrace()
        _run(trace=trace)

        for doc_id, info in trace.fused.items():
            assert sum(info["contrib"].values()) == pytest.approx(info["score"], abs=1e-5)

    def test_best_rank_is_min_rank_within_class(self):
        """best_rank 取类内最好名次：同一类多条榜时不能拿更差的那个"""
        trace = RetrievalTrace()
        _run(rewrite=(["变体一", "变体二"], "一段假设答案"), trace=trace)

        expected: dict[str, dict] = {}
        for board in trace.boards:
            for rank, doc_id in enumerate(board["ids"]):
                cls = board["class"]
                prev = expected.setdefault(doc_id, {}).get(cls)
                if prev is None or rank < prev:
                    expected[doc_id][cls] = rank

        for doc_id, info in trace.fused.items():
            assert info["best_rank"] == expected[doc_id]

    def test_only_pool_members_are_recorded(self):
        """只记进候选池的：池子大小由 rerank_pool_size 决定，没进池的不记"""
        trace = RetrievalTrace()
        _run(n_docs=40, trace=trace)

        assert len(trace.fused) == settings.rerank_pool_size
        assert sorted(i["rrf_rank"] for i in trace.fused.values()) == list(
            range(settings.rerank_pool_size)
        )


class TestRerank:
    def test_identity_rerank_keeps_order(self):
        """精排不换人时：前后名次一致、top_k 交集满、kept 数等于注入条数"""
        trace = RetrievalTrace()
        _run(trace=trace)

        info = trace.rerank
        assert info["enabled"] is True
        assert info["topk_overlap"] == 5
        assert sum(1 for d in info["docs"].values() if d["kept"]) == 5
        assert all(d["rrf_rank"] == d["rerank_rank"] for d in info["docs"].values())

    def test_reversing_rerank_shows_what_got_swapped(self):
        """精排反序时：交集应显著小于 top_k，这才叫「真换了人」"""
        def _reverse(q, ds, top_k=None):
            return list(reversed(ds))

        trace = RetrievalTrace()
        _run(n_docs=8, rerank_fn=_reverse, trace=trace)

        info = trace.rerank
        assert info["pool_size"] == 8
        assert info["topk_overlap"] < 5
        # RRF 第 0 名被精排压到最后 → 不再进注入
        first = min(info["docs"].values(), key=lambda d: d["rrf_rank"])
        assert first["kept"] is False

    def test_disabled_when_rerank_off(self):
        """轻量档不精排：enabled=False，名次原样保留"""
        trace = RetrievalTrace()
        _run(use_rerank=False, trace=trace)

        info = trace.rerank
        assert info["enabled"] is False
        assert all(d["rrf_rank"] == d["rerank_rank"] for d in info["docs"].values())


class TestAnchorBypass:
    """RRF 头部直通 + 尾部精排（2026-10-20 的策略）"""

    def test_anchor_docs_ignore_rerank_order(self):
        """直通段按 RRF 名次进，精排怎么排都改不了它们"""
        def _reverse(q, ds, top_k=None):
            return list(reversed(ds))

        trace = RetrievalTrace()
        docs, _ = _run(rerank_fn=_reverse, trace=trace, top_k=6, anchor_k=3)

        info = trace.rerank
        assert info["anchor_k"] == 3
        assert info["select_k"] == 3
        # 注入 6 条，其中前 3 条必须是 RRF 名次 0/1/2
        injected = [d for d in info["docs"].values() if d["kept"]]
        assert len(injected) == 6
        heads = sorted(d["rrf_rank"] for d in injected)[:3]
        assert heads == [0, 1, 2]
        assert all(d["role"] == "anchor" for d in info["docs"].values() if d["rrf_rank"] < 3)

    def test_tail_is_what_rerank_actually_ranks(self):
        """精确契约：精排只应该看到池子减掉 anchor 之后的那几条"""
        seen = {}

        def _spy(q, ds, top_k=None):
            seen["n"] = len(ds)
            return list(ds)

        trace = RetrievalTrace()
        _run(n_docs=20, rerank_fn=_spy, trace=trace, top_k=6, anchor_k=3)

        assert trace.rerank["pool_size"] == settings.rerank_pool_size
        assert seen["n"] == settings.rerank_pool_size - 3
        assert trace.rerank["tail_size"] == settings.rerank_pool_size - 3

    def test_total_injection_is_top_k(self):
        """anchor + 精排选 = 总注入条数，不多不少"""
        trace = RetrievalTrace()
        docs, _ = _run(n_docs=20, trace=trace, top_k=6, anchor_k=3)

        assert len(docs) == 6
        assert trace.rerank["anchor_k"] + trace.rerank["select_k"] == 6

    def test_anchor_zero_keeps_legacy_behaviour(self):
        """anchor_k=0 必须与旧口径完全一致：整池精排，交集照旧算"""
        trace = RetrievalTrace()
        _run(trace=trace, top_k=5, anchor_k=0)

        info = trace.rerank
        assert info["anchor_k"] == 0
        assert info["tail_size"] == info["pool_size"]
        assert info["topk_overlap"] == 5


class _NamedCollection:
    def __init__(self, name="persona_test"):
        self.name = name


def _fake_graph():
    return {
        "entities": {
            "阿尼玛": {"name": "阿尼玛", "freq": 5},
            "阴影": {"name": "阴影", "freq": 3},
        },
        "relations": [
            {
                "h": "阿尼玛", "r": "投射于", "t": "阴影",
                "e": "阿尼玛常以阴影的形式投射到他人身上",
                "w": 2, "s": "荣格自传.pdf", "hd": "第六章",
            },
        ],
    }


class TestGraphTrace:
    def test_records_seeds_edges_and_evidence(self, monkeypatch):
        """触发后的图谱轨迹：种子实体名、关系条数、证据条数都要能对上"""
        monkeypatch.setattr(KG, "load_graph", lambda name: _fake_graph())
        monkeypatch.setattr(
            KG, "_match_query_entities",
            lambda q, g, n=5: [("阿尼玛", 3.0), ("阴影", 3.0)],
        )
        trace: dict = {}
        asyncio.run(
            KG.retrieve_graph_context("阿尼玛是什么", _NamedCollection(), trace_out=trace)
        )

        assert trace["available"] is True
        assert trace["matched"] == 2
        assert trace["seeds"] == ["阿尼玛", "阴影"]
        assert trace["edges"] == 1
        assert trace["evidence"] == 1

    def test_marks_unavailable_without_graph_file(self, monkeypatch):
        """没有图谱文件时也要写一条 available=False，别让下游把「没图」当成「没命中」"""
        monkeypatch.setattr(KG, "load_graph", lambda name: {})
        trace: dict = {}
        text, docs = asyncio.run(
            KG.retrieve_graph_context("随便问问", _NamedCollection(), trace_out=trace)
        )

        assert text == "" and docs == []
        assert trace["available"] is False


class TestBoundary:
    def test_long_block_is_injected_in_full(self):
        """去掉注入上限的核心契约：3000 字的块必须一字不少地进上下文"""
        text = "这是一段很长的正文。" * 300
        assert len(text) == 3000

        context = _build_context([Document(page_content=text, metadata={"source": "d.pdf"})])

        assert text in context
        assert context.count("这是一段很长的正文。") == 300

    def test_records_length_and_sentence_end(self):
        """块长度按原文记，结尾是句末标点就标 True"""
        text = "完整的一句话。"
        trace = RetrievalTrace()

        _build_context([Document(page_content=text, metadata={"source": "e.pdf"})], trace=trace)
        info = trace.boundary[AS._get_doc_id(text)]

        assert info["chars"] == len(text)
        assert info["ends_at_sentence"] is True
        assert info["kind"] == "source"

    def test_block_ending_mid_sentence_is_flagged(self):
        """断在句中的块要能认出来——这是分块/OCR 的问题，不是注入的问题"""
        text = "这段话没有写完就断了"
        trace = RetrievalTrace()

        _build_context([Document(page_content=text, metadata={"source": "f.pdf"})], trace=trace)
        info = trace.boundary[AS._get_doc_id(text)]

        assert info["ends_at_sentence"] is False

    def test_infer_evidence_marked(self):
        """图谱证据标 kind=infer，与正文块区分开"""
        trace = RetrievalTrace()
        doc = Document(
            page_content="阿尼玛 —投射于→ 阴影",
            metadata={"source": "荣格自传.pdf", "kg_inferred": True},
        )

        _build_context([doc], trace=trace)
        info = trace.boundary[AS._get_doc_id(doc.page_content)]

        assert info["kind"] == "infer"

    def test_without_trace_context_still_builds(self):
        """生产路径不传 trace：上下文照旧构建，不能因为采集分支炸掉"""
        doc = Document(page_content="正文。" * 10, metadata={"source": "c.pdf"})
        context = _build_context([doc])
        assert "原文｜来源: c.pdf" in context
