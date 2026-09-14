"""
framework — 通用多智能体框架层
核心逻辑，与具体场景解耦
"""

from framework.supervisor import (
    build_graph,
    get_persona_graph,
    supervisor_node,
    set_scene_config,
    get_scene_config,
    get_conversation_history,
    append_conversation,
)
from framework.retrieval_agent import retrieval_agent
from framework.analysis_agent import analysis_agent
from framework.verification_agent import verification_agent

# 默认注入名人对话场景配置（启动时自动初始化）
from scenes.persona_chat.config import persona_chat_config
set_scene_config(persona_chat_config)

# 预编译 persona 场景图（避免首个请求现场编译）
get_persona_graph()

__all__ = [
    "build_graph",
    "get_persona_graph",
    "supervisor_node",
    "set_scene_config",
    "get_scene_config",
    "get_conversation_history",
    "append_conversation",
    "retrieval_agent",
    "analysis_agent",
    "verification_agent",
]
