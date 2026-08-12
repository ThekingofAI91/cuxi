"""
Agent 系统测试
验证各个组件的基本功能
"""

import pytest

from src.core.state import AgentState
from src.core.config import settings
from framework.supervisor import (
    build_graph,
    _rule_based_routing,
    supervisor_node,
    set_scene_config,
)
from framework.retrieval_agent import retrieval_agent
from framework.analysis_agent import analysis_agent
from framework.verification_agent import verification_agent
from framework.summarizer import summarizer
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
# Supervisor 路由测试
# ============================================================

class TestSupervisorRouting:
    """测试 Supervisor 路由逻辑（名人对话场景）"""
    
    def test_rule_based_routing_keyword_hit(self):
        """测试规则路由：分析类关键词命中 analyzer"""
        set_scene_config(persona_chat_config)
        query = "你如何看待梦的象征意义？"
        result = _rule_based_routing(query)
        assert result == "analyzer"
    
    def test_rule_based_routing_default(self):
        """测试规则路由：默认走 analyzer（名人场景默认 agent，无 coder）"""
        set_scene_config(persona_chat_config)
        query = "这段代码有bug吗？"
        result = _rule_based_routing(query)
        assert result == "analyzer"


# ============================================================
# Agent Node 测试
# ============================================================

class TestAgentNodes:
    """测试各个 Agent Node"""
    
    @pytest.mark.asyncio
    async def test_retrieval_agent(self):
        """测试检索 Agent"""
        state: AgentState = {
            "query": "测试检索",
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
        
        result = await retrieval_agent(state)
        assert "retrieved_docs" in result
        assert "route_history" in result
        assert "retrieval_agent" in result["route_history"]
    
    @pytest.mark.asyncio
    async def test_analysis_agent(self):
        """测试分析 Agent"""
        state: AgentState = {
            "query": "测试分析",
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
        
        result = await analysis_agent(state)
        assert "analysis" in result
        assert "route_history" in result
        assert "analysis_agent" in result["route_history"]
    
    @pytest.mark.asyncio
    async def test_verification_agent(self):
        """测试验证 Agent"""
        state: AgentState = {
            "query": "测试验证",
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
        
        result = await verification_agent(state)
        assert "verification" in result
        assert "route_history" in result
        assert "verification_agent" in result["route_history"]
    
    @pytest.mark.asyncio
    async def test_summarizer(self):
        """测试摘要 Agent"""
        state: AgentState = {
            "query": "测试摘要",
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
        
        result = await summarizer(state)
        assert "route_history" in result
        assert "summarizer" in result["route_history"]


# ============================================================
# Graph 构建测试
# ============================================================

class TestGraph:
    """测试 LangGraph 构建"""
    
    def test_graph_build(self):
        """测试能否成功构建图"""
        graph = build_graph()
        assert graph is not None
    
    def test_graph_nodes(self):
        """测试图的节点（名人对话场景：无 coder/info_gap）"""
        graph = build_graph()
        # 检查节点是否存在
        assert "supervisor" in graph.nodes
        assert "retriever" in graph.nodes
        assert "analyzer" in graph.nodes
        assert "verifier" in graph.nodes
        assert "summarizer" in graph.nodes
        assert "coder" not in graph.nodes
