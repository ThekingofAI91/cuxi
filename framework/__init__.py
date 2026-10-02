"""
framework — 通用多智能体框架层
核心逻辑，与具体场景解耦

一对一对话的入口是 supervisor_agent.run_supervisor_agent：
supervisor 自己持有 search_library 工具（检索实现也在该文件），运行时决定查不查、查多深。
共享运行时设施（场景上下文/会话历史/消息装配/Chroma 单例）在 framework/runtime.py。
"""

from framework.runtime import (
    set_scene_config,
    get_scene_config,
    get_chroma_client,
    get_conversation_history,
    append_conversation,
)
from framework.supervisor_agent import run_supervisor_agent, retrieve_documents
from framework.analysis_agent import analysis_agent
from framework.verification_agent import verification_agent

# 默认注入名人对话场景配置（启动时自动初始化）
from scenes.persona_chat.config import persona_chat_config
set_scene_config(persona_chat_config)

__all__ = [
    "run_supervisor_agent",
    "retrieve_documents",
    "set_scene_config",
    "get_scene_config",
    "get_chroma_client",
    "get_conversation_history",
    "append_conversation",
    "analysis_agent",
    "verification_agent",
]
