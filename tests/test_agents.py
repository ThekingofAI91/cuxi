"""
Agent 系统测试：State 定义 / 配置加载 / 各能力函数的基本可用性

（2026-09-30：原 Supervisor 路由测试与 LangGraph 图构建测试随旧图一起删除——
一对一的编排已收敛到 framework/supervisor_agent.py，路由判据不再存在。
编排行为测试见 tests/test_supervisor_agent.py。）
"""

import pytest

from src.core.state import AgentState
from src.core.config import settings
from framework.runtime import set_scene_config
from framework.supervisor_agent import retrieve_documents
from framework.analysis_agent import analysis_agent
from framework.verification_agent import verification_agent
from scenes.persona_chat.config import persona_chat_config


# ============================================================
# State 定义测试
# ============================================================

class TestState:
    """测试 AgentState 定义"""

    def test_state_creation(self):
        """测试能否正确创建 State"""
        state: AgentState = {
            "query": "测试问题",
            "session_id": "test-session",
            "retrieved_docs": [],
            "analysis": "",
            "code_result": "",
            "verification": "",
            "final_answer": "",
            "history": [],
            "route_history": [],
            "next_agent": None,
            "error": None,
        }

        assert state["query"] == "测试问题"
        assert state["session_id"] == "test-session"
        assert isinstance(state["retrieved_docs"], list)
        assert isinstance(state["history"], list)
        assert isinstance(state["route_history"], list)

    def test_state_optional_fields(self):
        """测试可选字段"""
        state: AgentState = {
            "query": "测试",
            "session_id": "test",
            "retrieved_docs": [],
            "analysis": "",
            "code_result": "",
            "verification": "",
            "final_answer": "",
            "history": [],
            "route_history": [],
            "next_agent": "retriever",
            "error": None,
        }

        assert state["next_agent"] == "retriever"
        assert state["error"] is None


# ============================================================
# Config 测试
# ============================================================

class TestConfig:
    """测试配置加载"""

    def test_config_loaded(self):
        """测试配置是否正确加载"""
        assert settings.llm_model is not None
        assert settings.embedding_model is not None
        assert settings.chunk_size > 0
        assert settings.max_history_turns > 0


# ============================================================
# 能力函数测试
# ============================================================

def _blank_state(query: str) -> AgentState:
    return {
        "query": query,
        "session_id": "test",
        "retrieved_docs": [],
        "analysis": "",
        "code_result": "",
        "verification": "",
        "final_answer": "",
        "history": [],
        "route_history": [],
        "next_agent": None,
        "error": None,
    }


class TestAgentNodes:
    """测试各能力函数（原图节点，现为被直接调用的函数）"""

    @pytest.mark.asyncio
    async def test_retrieve_documents_returns_triple(self):
        """retrieve_documents 契约：(docs, graph_used, error)，异常一律内部消化"""
        set_scene_config(persona_chat_config)
        docs, graph_used, error = await retrieve_documents("测试检索")
        assert isinstance(docs, list)
        assert isinstance(graph_used, bool)
        assert isinstance(error, str)

    @pytest.mark.asyncio
    async def test_retrieve_documents_skip_retrieval(self):
        """skip_retrieval=True 时零检索、零错误，直接返回空（不碰向量库）"""
        docs, graph_used, error = await retrieve_documents(
            "测试检索", skip_retrieval=True
        )
        assert docs == []
        assert graph_used is False
        assert error == ""

    @pytest.mark.asyncio
    async def test_analysis_agent(self):
        """测试分析 Agent"""
        result = await analysis_agent(_blank_state("测试分析"))
        assert "analysis" in result
        assert "route_history" in result
        assert "analysis_agent" in result["route_history"]

    @pytest.mark.asyncio
    async def test_verification_agent(self):
        """测试验证 Agent"""
        result = await verification_agent(_blank_state("测试验证"))
        assert "verification" in result
        assert "route_history" in result
        assert "verification_agent" in result["route_history"]
