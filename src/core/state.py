"""
state.py — 整个系统的数据契约
所有 Agent 共享的 State，贯穿 LangGraph StateGraph。
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """LangGraph 全局状态，所有 Node 共享读写。"""

    # ---- 输入 ----
    query: str                          # 用户输入
    session_id: str                     # 会话 ID，用于隔离状态

    # ---- 检索结果 ----
    retrieved_docs: list[Document]      # Retrieval Agent 写入

    # ---- Agent 输出 ----
    analysis: str                       # Analysis Agent 写入
    code_result: str                    # Code Agent 写入
    verification: str                   # Verification Agent 写入
    final_answer: str                   # Supervisor 最终汇总

    # ---- 对话管理 ----
    history: list[dict]                 # 对话历史，Summarizer 维护
    route_history: list[str]            # 路由记录，每个 Agent 执行完追加一条

    # ---- InfoGap 追问 ----
    info_gap_questions: Optional[list[dict]]  # InfoGap Agent 检测到的信息缺口（追问用）

    # ---- 角色人设 ----
    character_role_prompt: Optional[str]  # 角色 role_prompt（persona 场景注入，已含酒馆式构件）
    enable_verification: Optional[bool]    # 是否启用引用核查（沉浸型人设关闭）
    # 后历史指令：插在对话历史之后再钉一次角色（对应 SillyTavern Author's Note）。
    # 模型对越靠后的内容越敏感，历史一长人设就会被稀释，靠这条防长对话漂移。
    post_history_directive: Optional[str]
    # 按角色的采样覆盖：{"temperature":..,"top_p":..,"frequency_penalty":..}；
    # 仅 OpenAI 兼容参数，缺省/None 表示沿用代码默认值，让不同角色有不同的说话手感。
    sampling: Optional[dict]

    # ---- 分区与检索策略 ----
    zone: Optional[str]                   # "education" | "entertainment"
    light_retrieval: Optional[bool]       # 轻量召回：跳过改写/重排/图谱，仅原始问题 top-3
    skip_retrieval: Optional[bool]        # 酒馆式零检索：角色卡构件齐全时完全不查向量库
    graph_used: Optional[bool]            # 本轮是否触发了知识图谱增强

    # ---- 用户长期记忆 ----
    user_memory: Optional[str]            # 已渲染的记忆条目块（空串/None = 无记忆，不注入）

    # ---- 流式输出 ----
    stream_callback: Optional[Any]      # 流式 token 回调 (async callable, token: str) -> None

    # ---- 内部控制 ----
    next_agent: Optional[str]           # Supervisor 路由决策结果
    error: Optional[str]                # 错误信息
