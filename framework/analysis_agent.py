"""
Analysis Agent — 分析师 🔍
负责深度分析：对比、因果推理、归纳总结、趋势挖掘
使用 LLM 对检索结果进行结构化分析
"""

from __future__ import annotations

from typing import Any

from src.core.llm import get_chat_llm
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.state import AgentState


class AnalysisOutput(BaseModel):
    """分析 Agent 的结构化输出"""

    summary: str = Field(description="核心观点总结，2-3句话")
    key_points: list[str] = Field(description="关键发现列表，每一条一个要点")
    detailed_analysis: str = Field(description="详细分析内容，Markdown格式")
    confidence: float = Field(description="分析置信度，0-1之间", ge=0.0, le=1.0)


async def analysis_agent(state: AgentState) -> dict[str, Any]:
    """
    分析 Agent：对检索结果进行深度分析，或直接回答用户问题

    流程：
    1. 从 state 读取 query 和 retrieved_docs
    2. 如果有检索结果，基于资料进行分析
    3. 如果没有检索结果，用 LLM 自身知识直接回答
    4. 将结果写入 state["analysis"]
    """
    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])
    history = state.get("history", [])
    character_role_prompt = state.get("character_role_prompt", "")
    stream_callback = state.get("stream_callback")
    print(f"\n[Analysis Agent] 🔍 正在分析: {query}")
    print(f"[Analysis Agent] 基于 {len(retrieved_docs)} 条检索结果")
    print(f"[Analysis Agent] 对话历史轮次: {len(history)}")
    print(f"[Analysis Agent] 角色人设: {'有' if character_role_prompt else '无'}")

    try:
        # ---- 名人对话场景：直接以角色口吻回答（复用 supervisor 的生成逻辑）----
        # 基于检索资料 + 对话历史，流式输出角色口吻回答，不再产出结构化 JSON 报告；
        # supervisor 直接以 analysis 作为最终回答，省掉一次长 LLM 二次生成。
        if character_role_prompt:
            from framework.supervisor import _generate_direct_response
            context = _build_context(retrieved_docs) if retrieved_docs else None
            answer = await _generate_direct_response(
                query,
                history,
                character_role_prompt,
                stream_callback,
                session_id=state.get("session_id"),
                context=context,
            )
            print("[Analysis Agent] ✅ 角色口吻回答完成（流式）")
            return {
                "analysis": answer,
                "route_history": state.get("route_history", []) + ["analysis_agent"],
            }

        llm = get_chat_llm(
            temperature=0.5,
            max_tokens=settings.llm_max_tokens,
        )

        # ---- 有检索结果：基于资料分析 ----
        if retrieved_docs:
            context = _build_context(retrieved_docs)

            # 构建系统提示词，注入角色人设
            if character_role_prompt:
                system_prompt = f"""{character_role_prompt}

你的任务是基于下面的参考资料回答用户问题。请遵循以下原则：
1. 回答必须严格基于参考资料，不要编造事实
2. 不要添加参考资料中没有的信息，即使你认为自己知道答案
3. 如果参考资料不足，明确说明哪些内容无法从资料中得出
4. 始终保持角色设定的语气、风格和知识边界
5. 引用资料时说明出自哪本书或哪篇文章
6. 对于超出角色知识范围的问题，诚实说明

请严格按照以下 JSON 格式输出：
{{
    "summary": "2-3句话的核心总结",
    "key_points": ["关键发现1", "关键发现2", "关键发现3"],
    "detailed_analysis": "完整的Markdown格式分析报告",
    "confidence": 0.85
}}"""
            else:
                system_prompt = """你是一个学术分析专家。你的任务是对给定的查询和参考资料进行深入分析。

请遵循以下原则：
1. 分析必须基于参考资料，不要编造事实
2. 如果参考资料不足，明确说明局限性
3. 对于对比类问题，使用表格呈现差异
4. 对于因果推理，标明因果关系链
5. 指出不同来源之间的观点分歧（如有）

请严格按照以下 JSON 格式输出：
{
    "summary": "2-3句话的核心总结",
    "key_points": ["关键发现1", "关键发现2", "关键发现3"],
    "detailed_analysis": "完整的Markdown格式分析报告",
    "confidence": 0.85
}"""

            # 构建用户 prompt，包含历史上下文
            user_prompt = f"""## 用户问题
{query}

## 参考资料
{context}"""

            if history:
                from framework.supervisor import format_history_for_prompt, get_scene_config
                from src.core.config import settings as _settings
                history_text = format_history_for_prompt(history, max_turns=_settings.max_history_turns, session_id=state.get("session_id"))
                _config = get_scene_config()
                _history_instruction = getattr(_config, 'history_instruction', "") or "" if _config else ""
                if not _history_instruction:
                    _history_instruction = "以下是之前的对话历史（仅用于理解指代和背景，不要主动延伸历史话题）："
                user_prompt = f"""## 对话历史
{history_text}

{_history_instruction}

【重要】引用对话历史中的信息时，必须原样保留用户提供的具体数字、分数、日期、名称等，不得修改或近似。

""" + user_prompt

            response = await llm.ainvoke([
                ("system", system_prompt),
                ("user", user_prompt),
            ])

            # 解析 JSON 响应
            analysis = _parse_analysis_response(response.content)

            full_analysis = f"""## 分析总结
{analysis['summary']}

## 关键发现
"""
            for i, point in enumerate(analysis['key_points'], 1):
                full_analysis += f"{i}. {point}\n"

            full_analysis += f"""
## 详细分析
{analysis['detailed_analysis']}

---
*分析置信度: {analysis['confidence']:.1%}*
"""

            print(f"[Analysis Agent] ✅ 基于资料分析完成，置信度: {analysis['confidence']:.1%}")

        # ---- 无检索结果：用 LLM 自身知识直接回答 ----
        else:
            print("[Analysis Agent] ⚠️ 无检索结果，用 LLM 自身知识回答")

            # 构建消息列表，注入角色人设和对话历史
            if character_role_prompt:
                system_msg = f"""{character_role_prompt}

请始终保持角色设定的语气、风格和知识边界来回答问题。
如果问题超出你的知识范围，诚实说明。
回答要简洁友好，使用Markdown格式。
直接回答问题，不需要提及内部处理流程。"""
            else:
                system_msg = """你是一个友好、知识丰富的学术助手。

【最高原则】你必须始终围绕用户的【当前问题】来回答，不要被对话历史带偏话题。

请遵循以下原则：
1. 首先直接回答用户的当前问题，这是最重要的任务
2. 对话历史仅用于理解指代和背景，不要主动延伸历史中的其他话题
3. 回答要简洁、有针对性、有价值，不要泛泛而谈
4. 使用Markdown格式，让回答结构清晰
5. 绝对不要主动引导话题转向其他方向

直接回答问题，不需要提及内部处理流程。"""

            messages = [("system", system_msg)]

            if history:
                from framework.supervisor import format_history_for_prompt, get_scene_config
                from src.core.config import settings as _settings
                history_text = format_history_for_prompt(history, max_turns=_settings.max_history_turns, session_id=state.get("session_id"))
                _config = get_scene_config()
                _history_instruction = getattr(_config, 'history_instruction', "") or "" if _config else ""
                if not _history_instruction:
                    _history_instruction = "以下是之前的对话历史（仅用于理解指代和背景，不要主动延伸历史话题）："
                messages.append(("system", f"""{_history_instruction}

{history_text}

【重要】引用对话历史中的信息时，必须原样保留用户提供的具体数字、分数、日期、名称等，不得修改或近似。"""))

            messages.append(("user", query))

            # 流式输出
            if stream_callback:
                full_analysis = ""
                async for chunk in llm.astream(messages):
                    token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if token:
                        full_analysis += token
                        await stream_callback(token)
                print(f"[Analysis Agent] ✅ LLM 流式回答完成")
            else:
                response = await llm.ainvoke(messages)
                full_analysis = response.content.strip()
                print(f"[Analysis Agent] ✅ LLM 直接回答完成")

        return {
            "analysis": full_analysis,
            "route_history": state.get("route_history", []) + ["analysis_agent"],
        }

    except Exception as e:
        print(f"[Analysis Agent] ❌ 分析过程出错: {e}")
        fallback = f"分析过程出现错误: {e}"
        return {
            "analysis": fallback,
            "route_history": state.get("route_history", []) + ["analysis_agent"],
            "error": str(e),
        }


def _build_context(docs: list) -> str:
    """构建 LLM 上下文"""
    if not docs:
        return "（无参考资料）"

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        heading = doc.metadata.get("heading", "未知章节")
        content = doc.page_content[:450]  # 限制每条长度：检索用完整块，注入 LLM 只取头部
        # （重组后块为 ~1000 字符完整段落，全量注入会让每次 LLM 调用 prefill 12K+ 字符，
        #   单请求 40s+、高并发排队严重；截到 450 字符保留段落核心语义，速度显著回升）
        context_parts.append(f"[{i}] 来源: {source} | 章节: {heading}\n{content}")

    return "\n\n---\n\n".join(context_parts)


def _parse_analysis_response(text: str) -> dict:
    """
    解析 LLM 返回的分析结果（JSON 格式）。
    如果解析失败，回退到默认结构。
    """
    import json as _json

    # 尝试提取 JSON
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = _json.loads(text[start:end])
            return {
                "summary": data.get("summary", ""),
                "key_points": data.get("key_points", []),
                "detailed_analysis": data.get("detailed_analysis", ""),
                "confidence": float(data.get("confidence", 0.5)),
            }
    except Exception:
        pass

    # 解析失败，用原始文本作为 detailed_analysis
    print("[Analysis Agent] ⚠️ JSON 解析失败，使用原始文本")
    return {
        "summary": text[:200],
        "key_points": ["（解析失败）"],
        "detailed_analysis": text,
        "confidence": 0.5,
    }
