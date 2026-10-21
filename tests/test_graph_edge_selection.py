"""子图选边规则的单测：一跳优先、二跳支撑度门槛、去重累加、闸门节点。

这些规则直接决定「注入哪 30 条图谱关系」，之前没有覆盖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.knowledge_graph import _bfs_subgraph, _resolve_gate_nodes


def _graph(edges, names=None, types=None):
    """edges: [(h, r, t, w)]；names/types: {key: 值}（默认 name=key、无类型）"""
    ents = {}
    rels = []
    for h, r, t, w in edges:
        for k in (h, t):
            ents.setdefault(k, {"name": k, "freq": 1})
        rels.append({"h": h, "r": r, "t": t, "e": f"{h}{r}{t}", "w": w})
    if names:
        for k, v in names.items():
            ents[k]["name"] = v
    if types:
        for k, v in types.items():
            ents[k]["type"] = v
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


# ============================================================
# 闸门：本体枢纽 + 事件节点
# ============================================================

def test_gate_drops_edges_to_entities_absent_from_question():
    """闸门节点（本体）指向的节点必须在问题命中实体里，否则这条边不收。"""
    g = _graph([("A", "r", "X", 1), ("A", "r", "Y", 1), ("A", "r", "Z", 1)])
    edges = _bfs_subgraph(
        g, ["A"], hops=1, max_edges=30,
        allowed_endpoints={"A", "X"}, gate_nodes={"A"},
    )
    pairs = _pairs(edges)
    assert ("A", "r", "X") in pairs
    assert ("A", "r", "Y") not in pairs and ("A", "r", "Z") not in pairs


def test_gate_is_off_without_allowed_endpoints():
    """不传 allowed_endpoints 时闸门关闭，保持旧行为（离线探针、单测用）。"""
    g = _graph([("A", "r", "X", 1), ("A", "r", "Y", 1), ("A", "r", "Z", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=1, max_edges=30)
    assert len(edges) == 3


def test_gate_is_off_when_gate_nodes_empty():
    """有白名单但闸门节点为空（旧图谱没有 type、也没配本体名）时，一条边都不挡。"""
    g = _graph([("A", "r", "X", 1), ("A", "r", "Y", 1)])
    edges = _bfs_subgraph(g, ["A"], hops=1, max_edges=30, allowed_endpoints={"A"})
    assert len(edges) == 2


def test_non_gate_node_is_not_gated():
    """普通节点不受闸门约束：只有 A 在闸门里，B 的邻居照收。"""
    rels = [("A", "r", f"n{i}", 1) for i in range(18)] + [("B", "r", "X", 1), ("B", "r", "Y", 1)]
    g = _graph(rels)
    edges = _bfs_subgraph(
        g, ["B"], hops=1, max_edges=30,
        allowed_endpoints={"B"}, gate_nodes={"A"},
    )
    pairs = _pairs(edges)
    assert ("B", "r", "X") in pairs and ("B", "r", "Y") in pairs


# ============================================================
# 闸门节点从哪来：本体按配置名、事件按 type 字段
# ============================================================

def test_hub_names_come_from_config_and_cover_all_aliases(monkeypatch):
    """峰哥在图里是两个节点（本名 + 「峰哥亡命天涯」），两个都要认成枢纽。"""
    g = _graph([("峰哥", "别名", "峰哥亡命天涯", 1)])
    gates = _resolve_gate_nodes(g, "persona_fengge")
    assert {"峰哥", "峰哥亡命天涯"} <= gates


def test_unknown_collection_has_no_hub():
    """配置里没有这个 collection（或名字对不上）→ 没有枢纽，闸门只剩事件那一半。"""
    g = _graph([("王阳明", "提出", "知行合一", 1)])
    assert _resolve_gate_nodes(g, "persona_test") == set()


def test_event_nodes_are_gated_by_type():
    """事件节点靠图谱里的 type 字段认出，进闸门后不再向外扩张。"""
    g = _graph(
        [
            ("王阳明", "经历", "龙场悟道", 1),
            ("龙场悟道", "同场", "某弟子", 1),
            ("某弟子", "师从", "王阳明", 1),
        ],
        types={"龙场悟道": "事件", "王阳明": "人物", "某弟子": "人物"},
    )
    gates = _resolve_gate_nodes(g, "persona_test")  # 该库没配本体名，闸门里只有事件
    assert gates == {"龙场悟道"}

    # 问题里出现事件名 → 它成为种子、进白名单；到本体的边收下，到同场人物的边挡掉
    edges = _bfs_subgraph(
        g, ["龙场悟道"], hops=1, max_edges=30,
        allowed_endpoints={"龙场悟道", "王阳明"}, gate_nodes=gates,
    )
    pairs = {frozenset((e["h"], e["t"])) for e in edges}
    assert frozenset(("王阳明", "龙场悟道")) in pairs
    assert not any("某弟子" in p for p in pairs)
