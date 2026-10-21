"""子图选边规则的单测：一跳优先、二跳支撑度门槛、去重累加。

这些规则直接决定「注入哪 30 条图谱关系」，之前没有覆盖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.knowledge_graph import _bfs_subgraph


def _graph(edges, names=None):
    """edges: [(h, r, t, w)]"""
    ents = {}
    rels = []
    for h, r, t, w in edges:
        for k in (h, t):
            ents.setdefault(k, {"name": k, "freq": 1})
        rels.append({"h": h, "r": r, "t": t, "e": f"{h}{r}{t}", "w": w})
    if names:
        for k, v in names.items():
            ents[k]["name"] = v
    return {"entities": ents, "relations": rels}


def _pairs(edges):
    return {(e["h"], e["r"], e["t"]) for e in edges}


def test_one_hop_wins_over_two_hop_even_with_lower_weight():
    """一跳边优先：即便二跳边的 w 更高，也排在一跳之后并被截断掉。"""
    g = _graph([("A", "r", "B", 1), ("B", "r", "C", 9), ("B", "r", "D", 9)])
    edges = _bfs_subgraph(g, ["A"], hops=2, max_edges=1)
    assert len(edges) == 1
    assert edges[0]["t"] == "B"


def test_orphan_two_hop_edge_is_dropped():
    """二跳孤边（另一端只连着跳板）被挡：它就是约翰那种把整串无关关系拖进来的入口。"""
    g = _graph([("A", "r", "B", 1), ("B", "r", "X", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=2, max_edges=30)
    assert ("B", "r", "X") not in _pairs(edges)
    assert _pairs(edges) == {("A", "r", "B")}


def test_two_hop_edge_with_second_support_is_kept():
    """二跳边的另一端还被种子邻域里另一个节点指到（支撑度 2），保留。"""
    g = _graph(
        [
            ("A1", "r", "B", 1),
            ("A2", "r", "C", 1),
            ("B", "r", "X", 1),
            ("C", "r", "X", 1),
        ]
    )
    edges = _bfs_subgraph(g, ["A1", "A2"], hops=2, max_edges=30)
    assert ("B", "r", "X") in _pairs(edges) or ("C", "r", "X") in _pairs(edges)


def test_min_support_is_configurable():
    """门槛降到 1 时，孤边也能进来——证明挡它的是门槛而不是别的条件。"""
    g = _graph([("A", "r", "B", 1), ("B", "r", "X", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=2, max_edges=30, min_support=1)
    assert ("B", "r", "X") in _pairs(edges)


def test_both_ends_in_seed_neighbourhood_count_as_supported():
    """两端都在种子一跳邻域里的边，一律算有支撑。"""
    g = _graph([("A", "r", "B", 1), ("A", "r", "C", 1), ("B", "r", "C", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=2, max_edges=30)
    assert ("B", "r", "C") in _pairs(edges) or ("C", "r", "B") in _pairs(edges)


def test_returned_edges_carry_no_ranking_fields():
    """hop / support 是内部排序用的，不能漏到下游渲染与证据文档里。"""
    g = _graph([("A", "r", "B", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=2, max_edges=30)
    assert edges
    for e in edges:
        assert "hop" not in e and "support" not in e
        assert {"h", "r", "t", "e", "w", "s", "hd"} <= set(e)


def test_same_edge_listed_twice_accumulates_weight():
    """同一条边在图里出现两次（常见于 A→B 与 B→A 都被抽到）时 w 累加，而不是出现两次。"""
    g = _graph([("A", "r", "X", 1), ("A", "r", "X", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=2, max_edges=30)
    hits = [e for e in edges if {e["h"], e["t"]} == {"A", "X"}]
    assert len(hits) == 1
    assert hits[0]["w"] == 2


def test_hops_one_ignores_two_hop_edges():
    """hops=1 时不产生任何二跳边。"""
    g = _graph([("A", "r", "B", 1), ("B", "r", "X", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=1, max_edges=30)
    assert _pairs(edges) == {("A", "r", "B")}
