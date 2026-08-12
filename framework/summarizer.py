"""
Summarizer — 记录员 📝
负责多轮对话压缩、历史摘要管理
使用 LLM 对过往对话进行智能摘要
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.core.state import AgentState


async def summarizer(state: AgentState) -> dict[str, Any]:
    """
    摘要 Agent：压缩对话历史，防止 Token 爆炸

    流程：
    1. 读取 state["history"]
    2. 如果对话轮次超过阈值，用 LLM 压缩旧对话
    3. 保留最近几轮完整对话
    4. 更新 state["history"]
    """
    history = state.get("history", [])
    print(f"\n[Summarizer] 📝 正在检查对话历史...")
    print(f"[Summarizer] 当前历史轮次: {len(history)}")

    # 如果对话轮次未超过阈值，不做处理
    if len(history) <= settings.max_history_turns:
        print(f"[Summarizer] 轮次未超阈值 ({settings.max_history_turns})，无需摘要")
        return {
            "history": history,
            "route_history": state.get("route_history", []) + ["summarizer"],
        }

    print(f"[Summarizer] 轮次超过阈值，开始压缩摘要...")

    try:
        # 区分：旧对话（需要摘要）vs 新对话（保留完整）
        keep_full = settings.max_history_turns // 2  # 保留最近几轮完整
        if keep_full < 1:
            keep_full = 1

        if len(history) > keep_full:
            to_summarize = history[:-keep_full]  # 旧对话
            to_keep = history[-keep_full:]       # 新对话
        else:
            to_summarize = []
            to_keep = history

        if not to_summarize:
            print("[Summarizer] 没有需要摘要的旧对话")
            return {
                "history": history,
                "route_history": state.get("route_history", []) + ["summarizer"],
            }

        # 使用 LLM 生成摘要
        summary = await _summarize_with_llm(to_summarize)

        # 构建新的历史
        new_history: list[dict] = [
            {"role": "system", "content": f"对话历史摘要（已压缩）: {summary}"},
        ]
        new_history.extend(to_keep)

        print(f"[Summarizer] ✅ 摘要完成! 从 {len(to_summarize)} 轮对话压缩为1条摘要")
        print(f"[Summarizer] 新历史轮次: {len(new_history)}")

        return {
            "history": new_history,
            "route_history": state.get("route_history", []) + ["summarizer"],
        }

    except Exception as e:
        print(f"[Summarizer] ❌ 摘要过程出错: {e}")
        # 降级：直接截断
        truncated = history[-settings.max_history_turns:]
        print(f"[Summarizer] 使用降级方案: 截断至最近 {len(truncated)} 轮")
        return {
            "history": truncated,
            "route_history": state.get("route_history", []) + ["summarizer"],
            "error": str(e),
        }


async def _summarize_with_llm(history: list[dict]) -> str:
    """
    使用 LLM 对对话历史进行智能摘要

    Args:
        history: 对话历史列表

    Returns:
        摘要文本
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,  # 降低温度，减少摘要时的数字幻觉
        max_tokens=600,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    # 格式化历史：用户输入完整保留，助手回答截断放宽
    history_text = ""
    for i, turn in enumerate(history, 1):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        # 用户消息完整保留，助手消息保留前 500 字符
        if role == "user":
            content_preview = content
        else:
            content_preview = content[:500] + "..." if len(content) > 500 else content
        history_text += f"[{i}] {role}: {content_preview}\n"

    prompt = f"""请压缩以下对话历史为一段简洁的摘要。

## 对话历史
{history_text}

## 关键要求
1. 必须原样保留用户提供的所有具体数字、分数、日期、名称等信息（如"558分"不能改成"561分"）
2. 保留用户的核心问题和关注点
3. 保留已经给出的关键回答信息
4. 去除重复的讨论
5. 保持逻辑连贯
6. 用中文输出，控制在 300 字以内

## 摘要"""

    response = llm.invoke([
        ("system", "你是一个高效的对话摘要专家。请准确、简洁地压缩对话历史，必须原样保留用户提供的具体数字和事实信息。"),
        ("user", prompt),
    ])

    return response.content.strip()
