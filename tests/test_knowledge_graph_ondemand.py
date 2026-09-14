"""
知识图谱「按需触发」单元测试（优化十九）

覆盖本轮优化：
1. 查询实体对齐改为词法匹配（零 LLM 调用，取代每轮 LLM 抽取查询实体）
2. load_graph 按 mtime 缓存 / invalidate 清缓存
3. should_trigger_graph_retrieval 按需触发判断
   （文本召回不足 / top 相关性弱 / 关键实体未被文本覆盖 才启动图谱）
4. 图谱证据挂真实出处 + kg_inferred 标注（不再把「知识图谱」渲染成假书名冒充原著）

全部为纯内存单元测试：不连 ChromaDB、不调 LLM。
"""

import json
import time

import pytest
from langchain_core.documents import Document

from src.core.config import settings
from src.retrieval import knowledge_graph as kg


def _make_graph(entities=None, relations=None) -> dict:
    """构造最小可用的内存图谱 dict（键与 build_knowledge_graph 落盘格式一致）"""
    return {
        "collection": "test_col",
        "built_at": 0.0,
        "stats": {
            "chunks": 1,
            "entities": len(entities or {}),
            "relations": len(relations or []),
        },
        "entities": entities or {
            "原型": {"name": "原型", "freq": 5},
            "荣格": {"name": "荣格", "freq": 8},
            "集体潜意识": {"name": "集体潜意识", "freq": 3},
        },
        "relations": relations or [
            {
                "h": "原型", "r": "属于", "t": "集体潜意识",
                "e": "原型是集体潜意识的内容之一。", "w": 2,
                "c": "abc", "s": "原型与集体无意识.pdf", "hd": "第二章 原型",
            },
        ],
    }


def _doc(text: str, score: float = 0.1, source: str = "书A.pdf") -> Document:
    return Document(page_content=text, metadata={"rrf_score": score, "source": source})


@pytest.fixture(autouse=True)
def _clear_graph_cache():
    """每个测试后清空全局图谱缓存，避免用例间串扰"""
    yield
    with kg._graph_cache_lock:
        kg._graph_cache.clear()


# ============================================================
# 1) 词法实体对齐（零 LLM）
# ============================================================

class TestMatchQueryEntities:
    def test_substring_hit_weights(self):
        """节点名整体命中查询 → 最高分；中英文混合查询也成立"""
        graph = _make_graph()
        matched = kg._match_query_entities("荣格的原型理论", graph, n=6)
        scores = dict(matched)
        assert "原型" in scores and "荣格" in scores
        assert scores["原型"] == 3.0
        assert scores["荣格"] == 3.0

    def test_no_match_returns_empty(self):
        graph = _make_graph()
        assert kg._match_query_entities("完全无关的问题xyz", graph) == []

    def test_overlap_ratio(self):
        """无子串命中时按词重叠比例给分（>0 且 <3.0）"""
        graph = _make_graph(entities={"jung": {"name": "Carl Gustav Jung", "freq": 1}})
        matched = kg._match_query_entities("Jung 和弗洛伊德", graph)
        assert len(matched) == 1
        score = dict(matched)["jung"]
        assert 0 < score < 3.0
        assert score == pytest.approx(1 / 3)

    def test_ordering_by_score(self):
        """评分降序：原型（子串3.0）应排在词重叠（<3.0）之前"""
        graph = _make_graph(entities={
            "原型": {"name": "原型", "freq": 1},
            "梦": {"name": "梦", "freq": 9},
        })
        matched = kg._match_query_entities("原型和梦的关系", graph)
        keys = [k for k, _ in matched]
        assert keys.index("原型") < keys.index("梦")


# ============================================================
# 2) load_graph mtime 缓存
# ============================================================

class TestLoadGraphCache:
    def test_cache_hit_same_object(self, tmp_path, monkeypatch):
        """同一 mtime 二次加载返回同一对象（命中缓存）"""
        name = "cache_test_a"
        monkeypatch.setattr(kg, "graph_dir", lambda: tmp_path)
        kg.graph_path(name).write_text(
            json.dumps(_make_graph(), ensure_ascii=False), encoding="utf-8"
        )
        g1 = kg.load_graph(name)
        g2 = kg.load_graph(name)
        assert g1 is g2

    def test_reload_after_mtime_change(self, tmp_path, monkeypatch):
        """文件被重写（mtime 变化）后自动重载，不返回旧缓存"""
        name = "cache_test_b"
        monkeypatch.setattr(kg, "graph_dir", lambda: tmp_path)
        p = kg.graph_path(name)
        p.write_text(json.dumps(_make_graph(), ensure_ascii=False), encoding="utf-8")
        g1 = kg.load_graph(name)

        time.sleep(0.02)  # 确保文件 mtime 变化
        data = _make_graph()
        data["stats"]["entities"] = 99
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        g2 = kg.load_graph(name)
        assert g2 is not g1
        assert g2["stats"]["entities"] == 99

    def test_invalidate_clears_cache(self, tmp_path, monkeypatch):
        """invalidate_graph 删除文件并清缓存，再次加载返回 None"""
        name = "cache_test_c"
        monkeypatch.setattr(kg, "graph_dir", lambda: tmp_path)
        p = kg.graph_path(name)
        p.write_text(json.dumps(_make_graph(), ensure_ascii=False), encoding="utf-8")
        assert kg.load_graph(name) is not None
        kg.invalidate_graph(name)
        assert not p.exists()
        assert kg.load_graph(name) is None


# ============================================================
# 3) 按需触发判断
# ============================================================

class TestShouldTriggerGraphRetrieval:
    def _patch(self, monkeypatch, exists=True, graph=None):
        monkeypatch.setattr(settings, "graph_trigger_min_docs", 3)
        monkeypatch.setattr(settings, "graph_trigger_score", 0.03)
        monkeypatch.setattr(kg, "graph_exists", lambda name: exists)
        monkeypatch.setattr(
            kg, "load_graph", lambda name: graph if graph is not None else _make_graph()
        )

    def test_not_built_skips(self, monkeypatch):
        """图谱未构建 → 不触发，静默降级"""
        self._patch(monkeypatch, exists=False)
        trigger, reason = kg.should_trigger_graph_retrieval("原型", [], "x")
        assert trigger is False
        assert "未构建" in reason

    def test_few_docs_triggers(self, monkeypatch):
        """文本召回不足（< 阈值）→ 触发"""
        self._patch(monkeypatch)
        docs = [_doc("荣格提出了原型理论", 0.05), _doc("原型属于集体潜意识", 0.04)]
        trigger, reason = kg.should_trigger_graph_retrieval("原型", docs, "x")
        assert trigger is True
        assert "召回不足" in reason

    def test_low_top_score_triggers(self, monkeypatch):
        """top rrf_score 低于阈值 → 触发"""
        self._patch(monkeypatch)
        docs = [_doc("荣格提出了原型理论", 0.02) for _ in range(5)]
        trigger, reason = kg.should_trigger_graph_retrieval("原型", docs, "x")
        assert trigger is True
        assert "相关性偏弱" in reason

    def test_entity_uncovered_triggers(self, monkeypatch):
        """分数/条数达标，但查询命中的图谱实体未被文本覆盖 → 触发"""
        self._patch(monkeypatch)
        docs = [_doc("荣格提出了原型理论", 0.2) for _ in range(5)]  # 缺「集体潜意识」
        trigger, reason = kg.should_trigger_graph_retrieval(
            "原型和集体潜意识的关系", docs, "x"
        )
        assert trigger is True
        assert "未被文本覆盖" in reason
        assert "集体潜意识" in reason

    def test_quality_ok_skips(self, monkeypatch):
        """文本覆盖全部命中实体且质量达标 → 不触发（纯文本检索）"""
        self._patch(monkeypatch)
        docs = [_doc("荣格提出了原型理论，原型属于集体潜意识", 0.2) for _ in range(5)]
        trigger, reason = kg.should_trigger_graph_retrieval(
            "原型和集体潜意识的关系", docs, "x"
        )
        assert trigger is False
        assert "达标" in reason


# ============================================================
# 4) 图谱证据挂真实出处 + 推断标注
# ============================================================

class TestRetrieveGraphContext:
    @pytest.mark.asyncio
    async def test_evidence_carries_real_source_and_inferred_flag(self, monkeypatch):
        """证据文档使用真实出处/章节，并带 kg_inferred 标注（不冒充原著）"""
        graph = _make_graph()
        monkeypatch.setattr(kg, "load_graph", lambda name: graph)

        class FakeCollection:
            name = "test_col"

        ctx, ev_docs = await kg.retrieve_graph_context(
            "原型和集体潜意识的关系", FakeCollection(), top_entities=6,
        )
        assert ctx  # 渲染出结构化上下文
        assert "非人物逐字原话" in ctx  # 渲染文案明确标注推断
        assert ev_docs
        doc = ev_docs[0]
        assert doc.metadata["source"] == "原型与集体无意识.pdf"  # 真实出处
        assert doc.metadata["heading"] == "第二章 原型"           # 真实章节
        assert doc.metadata["kg_inferred"] is True
        assert doc.metadata["kg_evidence"] is True
        assert doc.metadata["source_type"] == "original"          # 按真实来源分类

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, monkeypatch):
        """查询未命中任何实体 → 返回空，不产出证据"""
        graph = _make_graph()
        monkeypatch.setattr(kg, "load_graph", lambda name: graph)

        class FakeCollection:
            name = "test_col"

        ctx, ev_docs = await kg.retrieve_graph_context(
            "完全无关的问题xyz", FakeCollection(), top_entities=6,
        )
        assert ctx == ""
        assert ev_docs == []
