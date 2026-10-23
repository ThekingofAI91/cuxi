"""
RRF 融合「每类一票 + 类间权重」的回归测试

锁住三条契约：
1. 同一类里同一文档出现多次，只按它在该类里的最好名次计一票（不重复加分）；
2. 打开查询改写（向量榜 1→4、BM25 榜 1→3）不改变两类之间的票数比；
3. 类间权重与来源权重各自独立生效，且都只缩放分数、不改变名次语义。
"""

import pytest

from src.retrieval.advanced_search import fuse_by_class

RRF_K = 60


def _no_source_meta(*doc_ids: str) -> dict[str, dict]:
    """无来源标记的 metadata：来源权重恒为 1.0，便于隔离类间逻辑"""
    return {doc_id: {} for doc_id in doc_ids}


def _score(rank: int, weight: float = 1.0, rrf_k: int = RRF_K) -> float:
    return weight / (rrf_k + rank + 1)


class TestClassInternalVoting:
    def test_same_doc_in_two_boards_of_one_class_counts_once(self):
        """doc a 在向量类两条榜都是第 1 名，只按 rank 0 记一票"""
        boards = [
            ("vector", ["a", "b"]),
            ("vector", ["a", "c"]),
        ]
        scores = fuse_by_class(
            boards, _no_source_meta("a", "b", "c"), {"vector": 1.0, "bm25": 1.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert scores["a"] == pytest.approx(_score(0))
        assert scores["a"] != pytest.approx(_score(0) * 2)
        assert scores["b"] == pytest.approx(_score(1))
        assert scores["c"] == pytest.approx(_score(1))

    def test_best_rank_wins_within_class(self):
        """同一 doc 在一类里名次 3 和名次 0 都出现，取 0 而不是相加"""
        boards = [
            ("vector", ["x", "y", "z", "a"]),
            ("vector", ["a"]),
        ]
        scores = fuse_by_class(
            boards, _no_source_meta("a", "x", "y", "z"), {"vector": 1.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert scores["a"] == pytest.approx(_score(0))

    def test_different_classes_still_add_up(self):
        """跨类命中照旧累加：向量一票 + BM25 一票"""
        boards = [("vector", ["a"]), ("bm25", ["a"])]
        scores = fuse_by_class(
            boards, _no_source_meta("a"), {"vector": 1.0, "bm25": 1.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert scores["a"] == pytest.approx(_score(0) * 2)


class TestMultiBoardVoting:
    """同类多榜不能放大票数——改写时代踩过的坑，规则留在融合层。"""

    def test_vote_ratio_stable_when_boards_added(self):
        """向量 4 榜、BM25 3 榜时，同文档两类票数比仍是 1:1"""
        baseline = {
            "vector": [("vector", ["doc", "other1"])],
            "bm25": [("bm25", ["doc", "other2"])],
        }
        many_boards = {
            "vector": [("vector", ["doc", "other1"])] * 4,
            "bm25": [("bm25", ["doc", "other2"])] * 3,
        }
        meta = _no_source_meta("doc", "other1", "other2")

        base_scores = fuse_by_class(
            baseline["vector"] + baseline["bm25"], meta, {"vector": 1.0, "bm25": 1.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        rw_scores = fuse_by_class(
            many_boards["vector"] + many_boards["bm25"], meta, {"vector": 1.0, "bm25": 1.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert base_scores == pytest.approx(rw_scores)

    def test_new_docs_from_extra_board_still_retrieved(self):
        """多出来的榜引入的新文档照样能被召回，只是不会挤掉同类老文档的票数"""
        boards = [
            ("vector", ["a"]),
            ("vector", ["b"]),   # 第二条榜召回到的新文档
        ]
        scores = fuse_by_class(
            boards, _no_source_meta("a", "b"), {"vector": 1.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert scores["a"] == pytest.approx(_score(0))
        assert scores["b"] == pytest.approx(_score(0))


class TestClassWeights:
    def test_class_weight_scales_score(self):
        """BM25 类权重 2.0，只有 BM25 命中的文档分数翻倍"""
        boards = [("bm25", ["a"]), ("vector", ["b"])]
        scores = fuse_by_class(
            boards, _no_source_meta("a", "b"), {"vector": 1.0, "bm25": 2.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert scores["a"] == pytest.approx(_score(0, 2.0))
        assert scores["b"] == pytest.approx(_score(0, 1.0))

    def test_unknown_class_falls_back_to_unit_weight(self):
        scores = fuse_by_class(
            [("graph", ["a"])], _no_source_meta("a"), {"vector": 3.0},
            rrf_k=RRF_K, use_source_weight=False,
        )
        assert scores["a"] == pytest.approx(_score(0))


class TestSourceWeight:
    def test_anchor_beats_secondary_in_same_rank(self):
        """同名次下，anchor（2.0）压过 secondary（0.25）"""
        boards = [("vector", ["secondary_doc", "anchor_doc"])]
        meta = {
            "secondary_doc": {"source": "王阳明大传.pdf"},
            "anchor_doc": {"source": "wangyangming_core_ideas.md"},
        }
        scores = fuse_by_class(
            boards, meta, {"vector": 1.0}, rrf_k=RRF_K, use_source_weight=True,
        )
        # anchor 在 rank 1 但权重 2.0，仍需按公式计算后比较
        assert scores["anchor_doc"] > scores["secondary_doc"]

    def test_source_weight_can_be_disabled(self):
        boards = [("vector", ["secondary_doc"])]
        meta = {"secondary_doc": {"source": "王阳明大传.pdf"}}
        weighted = fuse_by_class(boards, meta, {"vector": 1.0}, use_source_weight=True)
        raw = fuse_by_class(boards, meta, {"vector": 1.0}, use_source_weight=False)
        assert weighted["secondary_doc"] < raw["secondary_doc"]
