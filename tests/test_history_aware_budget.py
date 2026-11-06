"""历史感知检索：相关背景由 LLM 提炼，查询总长最多 450 字符。

使用假 LLM 和检索后端，验证指代背景、完整问题、长度边界与降级路径。
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from framework import runtime as rt
from framework import supervisor_agent as sa
from src.core.config import settings


class FakeLLM:
    def __init__(self, content="", *, error=None, delay=0):
        self.content = content
        self.error = error
        self.delay = delay
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def _patch_llm(monkeypatch, **kwargs):
    llm = FakeLLM(**kwargs)
    monkeypatch.setattr(sa, "get_chat_llm", lambda **kw: llm)
    return llm


def test_no_history_uses_current_query_without_llm(monkeypatch):
    def unexpected_llm(**kwargs):
        pytest.fail("无历史的短问题不需要摘要调用")

    monkeypatch.setattr(sa, "get_chat_llm", unexpected_llm)
    query = "什么是共时性？"
    assert asyncio.run(sa.build_history_aware_query(query, [])) == query


def test_related_summary_reads_full_recent_turns_and_current_question(monkeypatch):
    summary = "用户讨论梦中的蛇与集体无意识，正在追问它们的关系。"
    llm = _patch_llm(monkeypatch, content=summary)
    query = "那它们是什么关系？"
    history = [
        ("与当前问题无关的旧话题", "旧回答"),
        ("我梦到了蛇", "寒暄" * 100 + "梦中的蛇可能涉及集体无意识。"),
        ("我想继续理解那个梦", "可以讨论其中的象征。"),
    ]
    out = asyncio.run(sa.build_history_aware_query(query, history))

    assert summary in out
    assert out.endswith(query)
    assert "寒暄" not in out
    supplied = json.loads(llm.messages[0][1][1])
    assert supplied["当前问题"] == query
    assert len(supplied["最近对话"]) == 2
    assert "集体无意识" in supplied["最近对话"][0]["助手"]
    assert "旧话题" not in llm.messages[0][1][1]


def test_oversized_summary_never_displaces_current_question(monkeypatch):
    _patch_llm(monkeypatch, content="背景" * 1000)
    query = "我在 2026 年考了 558 分，劣势感觉功能会怎样影响复习专注度？"
    out = asyncio.run(sa.build_history_aware_query(query, [("长" * 500, "答" * 500)]))

    assert len(out) <= 450
    assert query in out
    assert out.endswith(query)


@pytest.mark.parametrize("length", [440, 450])
def test_question_using_budget_is_kept_without_summary(monkeypatch, length):
    def unexpected_llm(**kwargs):
        pytest.fail("当前问题占满预算时不调用历史摘要")

    monkeypatch.setattr(sa, "get_chat_llm", unexpected_llm)
    query = "问" * length
    assert asyncio.run(sa.build_history_aware_query(query, [("历史", "回答")])) == query


@pytest.mark.parametrize("content", [None, "", "  ", '""'])
def test_empty_summary_falls_back_to_current_question(monkeypatch, content):
    _patch_llm(monkeypatch, content=content)
    query = "什么是共时性？"
    assert asyncio.run(sa.build_history_aware_query(query, [("你好", "你好")])) == query


def test_summary_failure_falls_back_to_current_question(monkeypatch):
    _patch_llm(monkeypatch, error=RuntimeError("摘要服务不可用"))
    query = "这和阴影有什么关系？"
    assert asyncio.run(sa.build_history_aware_query(query, [("梦", "梦的解释")])) == query


def test_summary_timeout_does_not_block_retrieval(monkeypatch):
    monkeypatch.setattr(settings, "history_query_summary_timeout", 0.005)
    _patch_llm(monkeypatch, content="不会等到这个摘要", delay=0.05)
    query = "这和阴影有什么关系？"
    assert asyncio.run(sa.build_history_aware_query(query, [("梦", "梦的解释")])) == query


def test_long_question_is_semantically_compacted_for_retrieval(monkeypatch):
    query = "2026 年我考了 558 分。" + "复习时反复走神，" * 100 + "劣势感觉功能怎样影响专注度？"
    compact = "2026 年考了 558 分的用户，想了解劣势感觉功能怎样影响复习专注度。"
    llm = _patch_llm(monkeypatch, content=compact)
    out = asyncio.run(sa.build_history_aware_query(query, [("心理功能", "劣势功能")]))

    assert out == compact
    assert len(out) <= 450
    assert json.loads(llm.messages[0][1][1])["当前问题"] == query


def test_long_question_fallback_preserves_both_ends(monkeypatch):
    _patch_llm(monkeypatch, error=RuntimeError("摘要服务不可用"))
    query = "2026 年我考了 558 分。" + "复习情境" * 200 + "最后想问：劣势感觉功能怎样影响专注度？"
    out = asyncio.run(sa.build_history_aware_query(query, [("心理功能", "劣势功能")]))

    assert len(out) <= 450
    assert out.startswith("2026 年我考了 558 分。")
    assert out.endswith("最后想问：劣势感觉功能怎样影响专注度？")


@pytest.mark.parametrize("strong", [True, False])
def test_retrieval_awaits_summary_in_both_modes(monkeypatch, strong):
    from src.retrieval import advanced_search

    _patch_llm(monkeypatch, content="用户追问梦中的蛇与集体无意识的关系。")
    collection = SimpleNamespace(name="test_history", count=lambda: 1)
    monkeypatch.setattr(rt, "get_chroma_client", lambda: SimpleNamespace(
        get_or_create_collection=lambda **kw: collection,
    ))
    monkeypatch.setattr(rt, "get_scene_config", lambda: SimpleNamespace(
        chroma_collection="test_history", history_aware_retrieval=True,
    ))
    monkeypatch.setattr(settings, "graph_rag_enabled", False)
    captured = {}

    async def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(advanced_search, "advanced_retrieval", fake_retrieve)
    query = "那它们是什么关系？"
    docs, graph_used, error = asyncio.run(sa.retrieve_documents(
        query, [("我梦到了蛇", "可以从集体无意识理解。")], strong=strong,
    ))

    assert (docs, graph_used, error) == ([], False, "")
    assert "集体无意识" in captured["question"]
    assert captured["question"].endswith(query)
    assert len(captured["question"]) <= 450
    assert captured["use_rerank"] is strong
    assert captured["top_k"] == (6 if strong else 3)
