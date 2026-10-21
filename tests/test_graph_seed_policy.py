"""种子策略单测：只取字面命中、不补位、白名单同源、没命中就零产出。

2026-10-21 的决定：删掉补位种子（「查询是节点名的子串」2.5 / 词重叠 0.3-0.75 两档），
问题里没有字面命中的实体就不产出图谱证据。这里锁住四条契约：谁能当种子、白名单从哪来、
模糊命中还能不能凑数、没命中时返回什么。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import knowledge_graph as kg


def _graph(entities, relations=None):
    """entities: {key: freq}；relations: [(h, r, t)]"""
    ents = {k: {"name": k, "freq": v} for k, v in entities.items()}
    rels = []
    for h, r, t in relations or []:
        rels.append({"h": h, "r": r, "t": t, "e": f"{h}{r}{t}", "w": 1, "s": "", "hd": ""})
    return {"entities": ents, "relations": rels}


class _Collection:
    def __init__(self, name="persona_test"):
        self.name = name


# ---------- 谁能当种子 ----------

def test_literal_entity_in_question_becomes_seed():
    g = _graph({"王阳明": 30, "龙场": 5, "宁王": 8})
    seeds = kg._literal_seeds("王阳明在龙场悟道", g, top_entities=6)
    assert set(seeds) == {"王阳明", "龙场"}
    assert all(s >= 3.0 for s in seeds.values())


def test_bare_word_overlap_no_longer_fills_the_quota():
    """「王阳明讲学」「宁王」这类只靠单字重叠拿分的实体，不再占种子名额。

    它们正是王阳明 24 题里六道拿到逐字相同边缘证据的入口：6 个名额被
    本体 + 本体变体 + 宁王/王华/王通 占满，问题主题一个进不来。
    """
    g = _graph({"王阳明": 30, "王阳明讲学": 12, "王阳明弟子": 9, "宁王": 8, "王华": 6})
    seeds = kg._literal_seeds("王阳明十五岁时梦到了哪位历史人物？", g, top_entities=6)
    assert list(seeds) == ["王阳明"]


def test_query_inside_node_name_is_not_a_seed():
    """2.5 档（问题整体是节点名的子串）同样不补位。"""
    g = _graph({"荣格心理学": 20})
    assert kg._literal_seeds("荣格", g, top_entities=6) == {}


def test_seeds_are_capped_and_sorted_by_freq():
    g = _graph({"阿": 1, "阿德": 2, "阿德勒": 30, "自卑情结": 12, "课题分离": 9,
                "共同体感觉": 7, "目的论": 4})
    q = "阿德勒说的阿德、阿德勒、自卑情结、课题分离、共同体感觉、目的论分别是什么"
    seeds = kg._literal_seeds(q, g, top_entities=3)
    assert len(seeds) == 3
    assert list(seeds)[:1] == ["阿德勒"]  # 同分按节点频次降序


# ---------- 白名单与零产出 ----------

def test_whitelist_comes_from_seeds_not_from_fuzzy_hits(monkeypatch):
    """闸门白名单同源：只放种子自己，没进问题的「宁王」进不来。"""
    g = _graph(
        {"王阳明": 30, "龙场": 5, "宁王": 8},
        [("王阳明", "贬至", "龙场"), ("王阳明", "平定", "宁王")],
    )
    # 本体名按 collection 配置；测试库是 persona_test，这里临时给它配上王阳明
    monkeypatch.setattr(
        kg.settings, "graph_hub_entities", {"persona_test": ["王阳明"]}, raising=False
    )
    monkeypatch.setattr(kg, "load_graph", lambda name: g)
    text, docs = asyncio.run(
        kg.retrieve_graph_context("王阳明在龙场悟道", _Collection())
    )
    assert text and docs
    assert "龙场" in text
    assert "宁王" not in text


def test_no_literal_hit_returns_nothing(monkeypatch):
    """问题里没有字面命中的实体时，图谱通道什么都不给——不拿模糊实体凑数。"""
    g = _graph(
        {"宁王": 8, "王华": 6},
        [("宁王", "起兵", "王华")],
    )
    monkeypatch.setattr(kg, "load_graph", lambda name: g)
    trace: dict = {}
    text, docs = asyncio.run(
        kg.retrieve_graph_context("婚礼那天我去了哪里", _Collection(), trace_out=trace)
    )
    assert text == "" and docs == []
    assert trace["matched"] == 0
    assert "seeds" not in trace


def test_trigger_ignores_fuzzy_only_hits(monkeypatch):
    """触发判据与取种子同口径：只有模糊命中时不触发图谱（否则白跑一趟还零产出）。"""
    g = _graph({"荣格心理学": 20}, [("荣格心理学", "提出", "荣格心理学")])
    monkeypatch.setattr(kg, "graph_exists", lambda name: True)
    monkeypatch.setattr(kg, "load_graph", lambda name: g)
    ok, reason = kg.should_trigger_graph_retrieval("荣格", [], "persona_test")
    assert ok is False
