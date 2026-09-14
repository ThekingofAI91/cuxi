"""
知识图谱「常驻 + AI 自主调用」单元测试（优化二十）

用户诉求：知识图谱应一直可用，由 AI 自主判断何时调用，而非用户在网页点按钮构建。
本轮改动：
1. 聊天端点角色首次被对话且图谱缺失时自动后台构建（graph_auto_build 开关，默认开）
2. AI 真正触发图谱增强时，检索链路置 graph_used=True 并随回答下发给前端（显示徽标）

本文件校验核心回归点：retrieval_agent 在 AI 决定调用图谱时正确标记 graph_used，
跳过时不标记；config 默认开启自动构建。
全部为内存单测：mock 掉 ChromaDB / LLM / 高级检索，不连真实库、不调模型。
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.core.config import settings


def _fake_state(query="什么是集体潜意识？", history=None):
    return {
        "query": query,
        "history": history or [],
        "route_history": [],
        "graph_used": False,
    }


def _bind_retrieval_mocks(graph_exists_val, trigger_val, with_kg_docs):
    """构造 retrieval_agent 运行所需的全部外部依赖 mock"""
    fake_collection = MagicMock()
    fake_collection.name = "persona_test"
    fake_collection.count.return_value = 5  # 文档库非空，能进入检索

    fake_client = MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    doc = MagicMock()
    doc.page_content = "文本检索片段"
    adv_result = ([doc], 0.1)

    kg_doc = MagicMock()
    kg_doc.page_content = "知识图谱关联证据"
    kg_ctx = ("图谱上下文", [kg_doc] if with_kg_docs else [])

    scene_cfg = MagicMock()
    scene_cfg.chroma_collection = "persona_test"

    return {
        "framework.supervisor.get_scene_config": patch(
            "framework.supervisor.get_scene_config", return_value=scene_cfg),
        "framework.supervisor.get_chroma_client": patch(
            "framework.supervisor.get_chroma_client", return_value=fake_client),
        "src.core.llm.get_chat_llm": patch(
            "src.core.llm.get_chat_llm", return_value=MagicMock()),
        "src.retrieval.advanced_search.advanced_retrieval": patch(
            "src.retrieval.advanced_search.advanced_retrieval", return_value=adv_result),
        # graph_exists 在 retrieval_agent 顶部 import，需 patch 模块属性
        "framework.retrieval_agent.graph_exists": patch(
            "framework.retrieval_agent.graph_exists", return_value=graph_exists_val),
        # 以下在函数体内动态 import，patch 源模块即可
        "src.retrieval.knowledge_graph.should_trigger_graph_retrieval": patch(
            "src.retrieval.knowledge_graph.should_trigger_graph_retrieval",
            return_value=(trigger_val, "测试触发原因")),
        "src.retrieval.knowledge_graph.retrieve_graph_context": patch(
            "src.retrieval.knowledge_graph.retrieve_graph_context",
            new=AsyncMock(return_value=kg_ctx)),
        "src.retrieval.advanced_search._get_doc_id": patch(
            "src.retrieval.advanced_search._get_doc_id", side_effect=lambda t: t),
    }


@pytest.mark.asyncio
async def test_graph_used_true_when_ai_triggers():
    """AI 判断需调用图谱（文本检索不达标）→ graph_used 标记 True"""
    mocks = _bind_retrieval_mocks(graph_exists_val=True, trigger_val=True, with_kg_docs=True)
    for m in mocks.values():
        m.start()
    try:
        from framework.retrieval_agent import retrieval_agent
        out = await retrieval_agent(_fake_state())
    finally:
        for m in mocks.values():
            m.stop()
    assert out["graph_used"] is True
    assert any("retrieval_agent" in str(r) for r in out["route_history"])


@pytest.mark.asyncio
async def test_graph_used_false_when_skipped():
    """文本检索质量达标，AI 跳过图谱 → graph_used 标记 False"""
    mocks = _bind_retrieval_mocks(graph_exists_val=True, trigger_val=False, with_kg_docs=False)
    for m in mocks.values():
        m.start()
    try:
        from framework.retrieval_agent import retrieval_agent
        out = await retrieval_agent(_fake_state())
    finally:
        for m in mocks.values():
            m.stop()
    assert out["graph_used"] is False


@pytest.mark.asyncio
async def test_graph_used_false_when_no_graph_built():
    """图谱尚未构建（graph_exists=False）→ 不触发，graph_used 为 False"""
    mocks = _bind_retrieval_mocks(graph_exists_val=False, trigger_val=True, with_kg_docs=True)
    for m in mocks.values():
        m.start()
    try:
        from framework.retrieval_agent import retrieval_agent
        out = await retrieval_agent(_fake_state())
    finally:
        for m in mocks.values():
            m.stop()
    assert out["graph_used"] is False


def test_graph_auto_build_enabled_by_default():
    """知识图谱自动构建开关默认开启（无需用户点按钮）"""
    assert settings.graph_auto_build is True
