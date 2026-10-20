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
from src.retrieval.advanced_search import fuse_by_class
from src.retrieval.trace import RetrievalTrace
from src.core.config import settings


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


def _run(rewrite=None, trace=None, n_docs=8):
    """跑一次 advanced_retrieval（全替身，不碰真模型与真向量库）"""
    docs = _docs(n_docs)
    collection = _FakeCollection(docs)

    async def _fake_rewrite(question, llm, num_variants):
        return rewrite

    async def _go():
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(AS, "_ensure_bm25_ready", lambda coll: (docs, _FakeBM25(docs)))
            mp.setattr("src.retrieval.embedder.get_embedder", lambda: _FakeEmbedder())
            mp.setattr(
                "src.retrieval.reranker.rerank_documents",
                lambda q, ds, top_k=None: list(ds),
            )
            if rewrite is not None:
                mp.setattr(AS, "_rewrite_with_budget", _fake_rewrite)
            return await AS.advanced_retrieval(
                question="测试问题",
                collection=collection,
                llm=None,
                top_k=5,
                use_multi_query=rewrite is not None,
                use_hyde=rewrite is not None,
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
