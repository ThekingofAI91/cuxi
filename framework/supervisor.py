"""
Supervisor Agent — 整个系统的大脑
通用框架层，通过 scene_config 注入场景特定的prompt和配置

当前项目只保留名人对话场景（persona_chat）：
图结构 = supervisor → retriever/analyzer/verifier
- 问候/闲聊直接回复；其余一律先检索再分析
- 分析完成后对引用原著的回答做引用核查（verifier），再编译最终答案
"""

import contextvars
import re
import asyncio
import threading
from typing import Any, Optional

from src.core.llm import get_chat_llm, astream_nonempty, ainvoke_nonempty
from langgraph.graph import END, START, StateGraph

from framework.analysis_agent import analysis_agent
from framework.retrieval_agent import retrieval_agent
from framework.verification_agent import verification_agent
from src.core.config import settings
from src.core.state import AgentState
from src.core.session_store import get_store as _get_session_store


# ============================================================
# 场景配置管理器（contextvars 按请求隔离）
# ============================================================
#
# 历史问题：_current_scene_config 是进程级全局单例。persona 请求处理期间
# 会把全局配置切走，另一个用户同时发请求时会拿到错误配置，
# 导致检索库、提示词全部错乱 —— 系统无法两人同时使用。
# 改用 contextvars：每个 asyncio 请求任务拥有独立上下文，set 只影响
# 当前请求，create_task 子任务会复制父上下文，天然做到请求级隔离。

_current_scene_config = contextvars.ContextVar("scene_config", default=None)

# 对话历史存储：session_id -> [(user_query, assistant_answer), ...]
_conversation_history_store: dict[str, list[tuple[str, str]]] = {}

# 对话摘要存储：session_id -> str（旧对话的压缩摘要，保留原始话题和关键信息）
_conversation_summaries: dict[str, str] = {}

# 摘要生成锁：多个旧对话压缩任务并发时串行执行，
# 避免后完成的任务基于旧的 existing 覆盖，导致早期轮次内容丢失
_summary_lock = asyncio.Lock()

# 会话历史读写锁：两个请求（如同一浏览器两个标签页共享 sessionId）
# 并发 append/删除同一会话时，防止读改写竞态导致丢轮次/历史被截断
_history_lock = threading.Lock()


def set_scene_config(config):
    """设置当前请求的场景配置（contextvar 仅对当前请求任务生效，无需全局恢复）"""
    _current_scene_config.set(config)


def get_scene_config():
    """获取当前请求的场景配置（未设置时返回 None）"""
    return _current_scene_config.get()


# ============================================================
# ChromaDB client 单例（线程安全，可复用）
# ============================================================
# 之前每次请求都新建 PersistentClient，重复打开 sqlite + 加载
# HNSW 索引有明显开销（约 0.5-2 秒/次）；PersistentClient 线程安全，
# 复用单例即可。

_chroma_client = None


def get_chroma_client():
    """获取全局 ChromaDB PersistentClient（单例）"""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_conversation_history(session_id: str) -> list[tuple[str, str]]:
    """获取指定会话的对话历史"""
    with _history_lock:
        return list(_conversation_history_store.get(session_id, []))


def delete_conversation_history(session_id: str) -> bool:
    """
    删除指定会话的全部历史（对话历史 + 摘要）。
    前端删除对话时调用，确保后端内存中的上下文同步清除，
    避免复用/遗留 session 时把旧历史注入新对话导致"重复提问"误判。
    """
    existed = False
    with _history_lock:
        if session_id in _conversation_history_store:
            del _conversation_history_store[session_id]
            existed = True
        if session_id in _conversation_summaries:
            del _conversation_summaries[session_id]
            existed = True
    if existed:
        print(f"[Supervisor] 🗑️ 已删除会话历史: session={session_id}")
        try:
            _get_session_store().delete(session_id)
        except Exception as e:
            print(f"[Supervisor] ⚠️ 会话持久化删除失败: {e}")
    return existed


def append_conversation(session_id: str, query: str, answer: str):
    """追加一轮对话到历史记录，超过阈值时自动压缩旧对话为摘要"""
    # 锁：同一 sessionId 的两个请求并发完成时，append 与截断必须原子，
    # 否则后完成的请求会基于旧的列表覆盖，导致轮次丢失
    to_summarize = []
    with _history_lock:
        if session_id not in _conversation_history_store:
            _conversation_history_store[session_id] = []
        _conversation_history_store[session_id].append((query, answer))
        # 限制最大轮次，超出部分用 LLM 压缩为摘要（防止丢失原始话题）
        max_turns = settings.max_history_turns if hasattr(settings, 'max_history_turns') else 10
        if len(_conversation_history_store[session_id]) > max_turns:
            # 取出需要压缩的旧对话
            to_summarize = _conversation_history_store[session_id][:-max_turns]
            _conversation_history_store[session_id] = _conversation_history_store[session_id][-max_turns:]
    if to_summarize:
        # 异步压缩（不阻塞当前响应）；锁保证多个压缩任务串行合并，不丢轮次
        asyncio.create_task(_summarize_old_turns(session_id, to_summarize))

    # 写穿持久化：对话历史落盘，服务重启后恢复
    try:
        with _history_lock:
            current = list(_conversation_history_store.get(session_id, []))
        _get_session_store().set_history(session_id, current)
    except Exception as e:
        print(f"[Supervisor] ⚠️ 对话历史持久化失败: {e}")


def truncate_conversation_history(session_id: str, keep_turns: int) -> int:
    """
    把会话历史截断到 keep_turns 轮（对话页"编辑历史消息"时调用）。

    前端删除被编辑消息及其之后的内容并重新发送时，后端存储同步截断，
    保证两边的上下文一致（否则被编辑轮次之后的旧回答仍留在后端，
    会作为历史注入，让模型以为那些轮次还成立）。

    返回截断后的轮数；会话不存在返回 -1。
    注意：已被压缩进摘要的早期轮次无法找回——摘要仍会注入，
    这是可接受的（摘要本就是"更早之前"的模糊记忆）。
    """
    if keep_turns < 0:
        keep_turns = 0
    with _history_lock:
        turns = _conversation_history_store.get(session_id)
        if turns is None:
            return -1
        if keep_turns < len(turns):
            del turns[keep_turns:]
        current = list(turns)
    try:
        _get_session_store().set_history(session_id, current)
    except Exception as e:
        print(f"[Supervisor] ⚠️ 截断后历史持久化失败: {e}")
    return len(current)


def pop_last_turn(session_id: str, expect_query: str | None = None) -> bool:
    """
    弹出最后一轮对话（「重新生成」时调用）。

    旧答案作废：从历史里摘掉最后一轮 (query, old_answer)，重答完成后
    append_conversation 会把新一轮写回。expect_query 用于校验弹掉的
    确实是当前要重答的那轮（防止并发时误删别人的最后一轮）。
    """
    with _history_lock:
        turns = _conversation_history_store.get(session_id)
        if not turns:
            return False
        if expect_query is not None and turns[-1][0] != expect_query:
            print(f"[Supervisor] ⚠️ pop_last_turn 校验失败，最后一轮不是待重答的问题，跳过")
            return False
        turns.pop()
        current = list(turns)
    try:
        _get_session_store().set_history(session_id, current)
    except Exception as e:
        print(f"[Supervisor] ⚠️ 弹轮后历史持久化失败: {e}")
    return True


async def _summarize_old_turns(session_id: str, turns: list[tuple[str, str]]):
    """将旧对话轮次压缩为摘要，合并到已有摘要中（串行执行，防止并发覆盖）"""
    if not turns:
        return
    async with _summary_lock:
        try:
            # 格式化旧对话：用户输入完整保留，助手回答截断放宽
            parts = []
            for q, a in turns:
                # 用户输入必须完整保留（包含分数、日期等关键信息）
                # 助手回答保留前 800 字符，足够包含核心内容
                a_truncated = a[:800] + "..." if len(a) > 800 else a
                parts.append(f"用户：{q}\n助手：{a_truncated}")
            turns_text = "\n\n".join(parts)

            # 已有摘要则拼在前面
            existing = _conversation_summaries.get(session_id, "")
            if existing:
                context = f"已有摘要：{existing}\n\n新增对话：\n{turns_text}"
            else:
                context = turns_text

            llm = get_chat_llm(
                temperature=0.1,  # 降低温度，减少摘要时的数字幻觉
                max_tokens=600,
            )
            response = await llm.ainvoke([
                ("system", """你是一个对话摘要专家。请将以下对话历史压缩为一段简洁的摘要。

【关键要求】
1. 必须原样保留用户提供的所有具体数字、分数、日期、名称等信息（如"558分"不能改成"561分"）
2. 保留用户的核心问题和关注点
3. 保留已给出的关键回答信息
4. 去除重复的讨论
5. 用中文输出，控制在 300 字以内"""),
                ("user", context),
            ])
            _conversation_summaries[session_id] = response.content.strip()
            print(f"[Supervisor] ✅ 旧对话摘要已更新: session={session_id}, 压缩了{len(turns)}轮")
            try:
                _get_session_store().set_summary(session_id, _conversation_summaries[session_id])
            except Exception as e:
                print(f"[Supervisor] ⚠️ 摘要持久化失败: {e}")
        except Exception as e:
            print(f"[Supervisor] ⚠️ 摘要生成失败: {e}")


def format_history_for_prompt(history: list[tuple[str, str]], max_turns: int = None, session_id: str = None) -> str:
    """将对话历史格式化为 prompt 文本，包含摘要（如有）"""
    if max_turns is None:
        # 默认注入轮数与存储容量一致，避免历史被静默丢弃
        max_turns = settings.max_history_turns
    parts = []

    # 先注入摘要（保留的原始话题上下文）
    if session_id and session_id in _conversation_summaries:
        parts.append(f"【早期对话摘要】\n{_conversation_summaries[session_id]}")

    # 再注入最近几轮完整对话
    if history:
        recent = history[-max_turns:]
        for i, (q, a) in enumerate(recent, 1):
            parts.append(f"第{i}轮对话：\n用户：{q}\n助手：{a}")

    return "\n\n".join(parts)


# ============================================================
# Supervisor Node 实现
# ============================================================

async def _generate_direct_response(
    query: str,
    history: list[tuple[str, str]] = None,
    character_role_prompt: str = "",
    stream_callback=None,
    session_id: str = None,
    context: str = None,
    max_tokens: int = None,
    zone: str = "education",
    post_history_directive: str = None,
    sampling: dict = None,
    user_memory: str = None,
) -> str:
    """
    当不需要调用任何 Agent 时，直接用 LLM 生成对话式回复。
    prompt从场景配置读取，并注入角色人设。
    支持流式输出：当提供 stream_callback 时，逐 token 回调。

    Args:
        context: 可选的参考资料（来自 analysis_agent 或检索结果），
                 提供后 LLM 会优先基于这些资料回答，减少幻觉。
        zone: 分区。娱乐区把资料定位成"角色自己的记忆"而非"参考资料"——
              一旦说出"根据资料""出自某篇"，AI 味立刻回来，故两区用不同的注入话术。
        post_history_directive: 后历史指令，插在对话历史**之后**（对应 SillyTavern 的
              Author's Note）。模型对越靠后的内容越敏感，历史一长人设会被稀释，
              靠这条在最后再钉一次角色。
        sampling: 按角色的采样覆盖（temperature / top_p / frequency_penalty 等
              OpenAI 兼容参数），缺省沿用代码默认值。
        user_memory: 用户长期记忆块（retrieve_memory_block 渲染产物），
              空串/None 不注入。
    """
    try:
        config = get_scene_config()
        system_prompt = config.direct_response_system_prompt if config else (
            "你是一个友好的AI助手。"
        )
        fallback = config.display_name if config else "AI助手"

        # 注入角色人设
        if character_role_prompt:
            voice = getattr(config, "persona_voice_directive", "") if config else ""
            system_prompt = f"""{character_role_prompt}

{system_prompt}

{voice}"""

        # 名人对话回答通常需要更长篇幅，放宽 token 上限（避免尾部"建议"被截断）
        # 娱乐区由调用方传入更小的 max_tokens（如 300），实现极短回答
        if max_tokens is None:
            max_tokens = 1600 if character_role_prompt else 900

        # 采样参数：角色卡可覆盖；未配置则沿用默认，保证既有手感不变
        _sampling = sampling or {}
        _llm_kwargs = {
            "temperature": _sampling.get(
                "temperature", 0.8 if character_role_prompt else 0.6
            ),
            "max_tokens": max_tokens,
        }
        for _k in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
            if _sampling.get(_k) is not None:
                _llm_kwargs[_k] = _sampling[_k]
        # 注意：不要把 repetition_penalty 等本地推理参数塞进 model_kwargs 透传——
        # DeepSeek / OpenAI 兼容 API 会以 400 拒收未知字段（角色卡该字段已弃用）
        llm = get_chat_llm(**_llm_kwargs)

        # 构建消息列表，注入对话历史
        messages = [("system", system_prompt)]

        # 注入检索上下文（基于知识库的资料），减少幻觉
        if user_memory:
            # 用户长期记忆：放在人设之后、资料之前——它是"对这位朋友的了解"，
            # 优先级低于硬约束（资料核查话术），但要在对话历史之前被看到
            from src.core.memory import format_memory_directive
            messages.append(("system", format_memory_directive(user_memory)))

        if context:
            if zone == "entertainment":
                # 娱乐区：资料 = 角色自己的记忆，不是"参考资料"。
                # 严禁暴露检索痕迹——一旦说出"根据资料""出自某篇"，人设当场崩。
                messages.append(("system", f"""【你记得的事】下面是你自己的记忆——你经历过的事、你说过的话、你熟悉的场景。
用它们帮你想起细节和语气，但：
- 绝对不要说出"根据资料""据我记载""出自某篇文章""资料显示"这类话
- 不要列来源、不要报章节名、不要标注出处
- 就当是你自己想起来了，用你平时说话的方式正常说出来
- 记忆里没有的，凭你自己的判断说；涉及具体数字、日期、名次这类硬事实，想不起来就说想不起来，别编

{context}"""))
            else:
                messages.append(("system", f"""以下是基于知识库整理的参考资料。

【重要】你必须严格基于以下参考资料来回答，不要添加资料中没有的信息。
如果参考资料不足以完整回答问题，请明确说明哪些部分基于资料、哪些部分无法从资料中得出。
不要编造资料中不存在的事实、数字或概念。
基于资料的延伸推理（如投射、自性化等概念的应用建议）可以用"我会这样看""依我的经验"等角色化口吻带出，
但不要与著作中的原话混为一谈，也不要宣称延伸内容出自某本著作。

【引用标注】回答中的关键论断、概念解释、事实与数字，请在句末标注所依据资料的编号（如 [2]）；
一个论断依据多条资料时可并列（如 [1][3]）；延伸推理部分不标注。编号与下方资料清单一一对应。

## 参考资料
{context}"""))

        if history:
            history_text = format_history_for_prompt(history, max_turns=settings.max_history_turns, session_id=session_id)
            # 历史使用指令按场景配置（名人对话鼓励延续，保持话题连贯）
            history_instruction = ""
            if config:
                history_instruction = getattr(config, 'history_instruction', "") or ""
            if not history_instruction:
                history_instruction = "以下是之前的对话历史（仅用于理解指代和背景，不要主动延伸历史话题）："
            messages.append(("system", f"""{history_instruction}

{history_text}

【重要】引用对话历史中的信息时，必须原样保留用户提供的具体数字、分数、日期、名称等，不得修改或近似。"""))

        # 后历史指令：历史之后再钉一次角色。人设指令全在最前面时，
        # 对话历史越长稀释越严重，这条是长对话不跑偏的关键（对应酒馆 Author's Note）。
        if post_history_directive:
            messages.append(("system", post_history_directive))

        messages.append(("user", query))

        # 流式输出（空响应容错：上游偶发 200 空壳，未推送任何 token 时安全重试）
        if stream_callback:
            full_text = ""
            async for token in astream_nonempty(llm, messages):
                if token:
                    full_text += token
                    await stream_callback(token)
            return full_text.strip()
        else:
            response = await ainvoke_nonempty(llm, messages)
            return (response.content if hasattr(response, "content") else "").strip()
    except Exception as e:
        print(f"[Supervisor] 直接回复生成失败: {e}")
        return f"你好！我是{fallback}，请问有什么可以帮你的？"


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Supervisor 节点：意图识别 + 路由决策（名人对话场景）

    路由策略（跳过 LLM 路由，省一次调用）：
    1. 引用核查已完成 → 编译最终答案
    2. 已有分析结果：
       - 检索到原著资料 → verifier（引用核查）
       - 无资料可对照 → 直接作为最终回答
    3. 已检索但未分析 → analyzer（防死循环）
    4. 问候/闲聊 → 直接回复
    5. 其余 → retriever（先检索，analyzer 会基于检索结果回答）

    注：历史轮次过多时的压缩由 _update_conversation_history 的异步任务
    在图外完成，不占用图内路由。
    """
    config = get_scene_config()
    query = state["query"]
    history = state.get("history", [])
    route_history = state.get("route_history", [])
    character_role_prompt = state.get("character_role_prompt", "")
    stream_callback = state.get("stream_callback")

    print(f"\n[Supervisor] 收到 query: {query}")
    print(f"[Supervisor] 当前历史轮次: {len(history)}")
    print(f"[Supervisor] 路由历史: {route_history}")
    print(f"[Supervisor] 角色人设: {'有' if character_role_prompt else '无'}")

    # 限制最大循环次数
    supervisor_count = route_history.count("supervisor")
    if supervisor_count > 6 or len(route_history) > 15:
        print(f"[Supervisor] ⚠️ 达到最大路由次数 (supervisor={supervisor_count}, total={len(route_history)})，强制结束")
        # 以角色口吻兜底，保证回答风格一致
        final_answer = await _generate_direct_response(
            query, history, character_role_prompt, stream_callback,
            session_id=state.get("session_id"),
            context=(state.get("analysis") or "").strip() or None,
        )
        if not final_answer or final_answer == "未找到相关信息。请尝试换个问法或上传相关文档后再试。":
            final_answer = await _generate_direct_response(
                query, history, character_role_prompt, stream_callback,
                session_id=state.get("session_id"),
                context=state.get("analysis") or None,
            )
        return {
            "next_agent": "__end__",
            "final_answer": final_answer,
            "route_history": route_history + ["supervisor"],
        }

    has_analysis = bool(state.get("analysis"))
    has_verification = bool(state.get("verification"))
    has_docs = bool(state.get("retrieved_docs"))

    # 引用核查已完成 → 编译最终答案（回答 + 引用可信度 + 出处）
    if "verification_agent" in route_history:
        print("[Supervisor] 引用核查已完成，编译最终答案")
        final_answer = _compile_final_answer(state)
        if not final_answer or final_answer == "未找到相关信息。请尝试换个问法或上传相关文档后再试。":
            print("[Supervisor] Agent 无有效结果，用 LLM 直接回复")
            final_answer = await _generate_direct_response(
                query, history, character_role_prompt, stream_callback,
                session_id=state.get("session_id"),
                context=state.get("analysis") or None,
            )
        return {
            "next_agent": "__end__",
            "final_answer": final_answer,
            "route_history": route_history + ["supervisor"],
        }

    # 已有分析结果：分析已以角色口吻流式生成，直接作为最终回答结束。
    # 引用核查（verifier）已移出图内关键路径，改为回答返回后异步执行（见 routes.event_stream）。
    if has_analysis and not has_verification:
        # 引用核查（verifier）不再在图内路由：其串行 LLM 调用会阻塞首字/定稿，
        # 改为 routes.event_stream 中「回答返回后异步执行」，经 type:'citations' 事件补推引用出处。
        final_answer = state.get("analysis", "").strip()
        if not final_answer:
            print("[Supervisor] 分析为空，降级为直接生成")
            final_answer = await _generate_direct_response(
                query, history, character_role_prompt, stream_callback,
                session_id=state.get("session_id"),
            )
        return {
            "next_agent": "__end__",
            "final_answer": final_answer,
            "route_history": route_history + ["supervisor"],
        }

    # 防死循环：已检索过但尚未分析时，强制路由 analyzer，
    # 避免陷入 retriever→supervisor 无限循环，analyzer 永远不被调用
    if "retrieval_agent" in route_history and not has_analysis:
        print("[Supervisor] 已检索但尚无分析结果，强制路由到 analyzer（避免重复检索死循环）")
        return {
            "next_agent": "analyzer",
            "route_history": route_history + ["supervisor"],
        }

    # ---- 轻聊快速通道：社交寒暄/语气回应直接生成，不进检索管线 ----
    # 这类消息不需要任何知识库资料，走全管线要白付检索改写 + 精排的 5-8s；
    # 快速通道一次 LLM 直出，首字延迟从 ~20s 降到 ~2s。
    # 判定用「整句精确匹配 + 去标点后全等」，绝不吞掉真实提问
    # （"你好，我想问抑郁症怎么治" 不匹配"你好"，仍走检索）。
    if getattr(settings, "light_chat_enabled", True) and _is_lightweight_turn(query):
        print("[Supervisor] 轻聊快速通道：直接生成（跳过检索）")
        final_answer = await _generate_direct_response(
            query, history, character_role_prompt, stream_callback,
            session_id=state.get("session_id"),
            max_tokens=300 if state.get("zone") == "entertainment" else None,
            zone=state.get("zone", "education"),
            post_history_directive=state.get("post_history_directive"),
            sampling=state.get("sampling"),
        )
        return {
            "next_agent": "__end__",
            "final_answer": final_answer,
            "route_history": route_history + ["supervisor"],
        }

    # 兜底：无角色人设时走规则路由（正常不会走到，persona 请求必带 role_prompt）
    if not character_role_prompt:
        next_agent = _rule_based_routing(query)
        print(f"[Supervisor] 规则路由降级: {next_agent}")
        return {
            "next_agent": next_agent,
            "route_history": route_history + ["supervisor"],
        }

    print("[Supervisor] 名人对话：规则路由到 retriever（跳过 LLM 路由）")
    return {
        "next_agent": "retriever",
        "route_history": route_history + ["supervisor"],
    }


# 轻聊快速通道的整句白名单（去空白标点、转小写后全等匹配）。
# 只收录"无论如何都不需要知识库"的社交短语；身份类（你是谁/自我介绍）
# 只靠人设就能答，也不必检索。
_LIGHT_PHRASES = frozenset({
    "你好", "您好", "嗨", "哈喽", "哈罗", "hello", "hi", "hey", "yo",
    "在吗", "在不在", "你是谁", "你叫什么", "你叫什么名字", "你是",
    "自我介绍", "自我介绍一下", "介绍一下你自己", "介绍下自己", "介绍一下自己",
    "谢谢", "谢谢啦", "感谢", "多谢", "thanks", "thank", "thx",
    "再见", "拜拜", "晚安", "早", "早上好", "中午好", "下午好", "晚上好",
    "辛苦了", "辛苦", "真的吗", "原来如此", "这样啊", "是吗", "好的",
    "好吧", "行吧", "好", "行", "可以", "嗯", "嗯嗯", "哦", "哦哦",
    "好哒", "好嘞", "ok", "okay", "哈哈", "哈哈哈", "哈哈哈哈", "666",
    "明白", "了解", "收到", "懂了", "明白了", "了解了",
})


def _is_lightweight_turn(query: str) -> bool:
    """判断是否为轻聊轮：寒暄、道谢、语气回应、身份询问。

    规则刻意保守（整句全等 + 去标点 + 长度上限），宁可漏判走检索，
    不可误判把真实提问送进无资料直答通道。
    """
    q = re.sub(r"[\s！!？?。，,、~～.．…·]+", "", (query or "").strip()).lower()
    if not q or len(q) > 12:
        return False
    if q in _LIGHT_PHRASES:
        return True
    # 叠词寒暄：哈喽哈喽 / 谢谢谢谢 / 晚安晚安（两半相同且本身是社交短语）
    half = len(q) // 2
    if len(q) % 2 == 0 and half >= 2 and q[:half] == q[half:] and q[:half] in _LIGHT_PHRASES:
        return True
    # 纯语气复读：哈哈哈…、嗯嗯嗯…、？？？…、666…
    if len(q) <= 8 and re.fullmatch(r"(.)\1{1,7}", q):
        return True
    return False


def _rule_based_routing(query: str) -> str:
    """基于规则的路由降级（persona 场景）"""
    config = get_scene_config()
    if config and hasattr(config, 'routing_keywords'):
        query_lower = query.lower()
        for agent, keywords in config.routing_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return agent
        return getattr(config, 'default_agent', 'retriever')

    # 兜底：分析类问题走 analyzer，其余一律检索
    query_lower = query.lower()
    analysis_keywords = ["对比", "分析", "区别", "异同", "总结", "趋势", "原因", "compare", "difference"]
    if any(kw in query_lower for kw in analysis_keywords):
        return "analyzer"
    return "retriever"


def _citation_block(verification: str) -> str:
    """从引用核查报告中提取「核查摘要 + 引用出处」区块，供回答返回后异步推送。

    与历史 _compile_final_answer 拼接待引用出处逻辑保持一致；verifier 移出图内后，
    routes.event_stream 用本函数把引用出处渲染成 type:'citations' 事件补推给用户。
    """
    if not verification:
        return ""
    parts = []
    summary_match = re.search(r"### 核查摘要\n(.*?)(?=\n###|\Z)", verification, re.DOTALL)
    citation_match = re.search(r"### 引用出处\n(.*?)(?=\n###|\Z)", verification, re.DOTALL)
    if summary_match:
        parts.append(f"\n\n---\n📊 {summary_match.group(1).strip()}")
    if citation_match:
        parts.append(f"\n**引用出处:**\n{citation_match.group(1).strip()}")
    return "".join(parts).strip()


def _compile_final_answer(state: AgentState) -> str:
    """编译最终答案：角色回答 + 引用核查摘要 + 引用出处"""
    analysis = state.get("analysis", "")
    verification = state.get("verification", "")

    core_answer = analysis or ""
    citation = _citation_block(verification) if verification else ""

    final = (core_answer + citation).strip() if (core_answer or citation) else ""
    if not final:
        final = "未找到相关信息。请尝试换个问法或上传相关文档后再试。"

    return final


# ============================================================
# 路由函数
# ============================================================

def route_after_supervisor(state: AgentState) -> str:
    """Supervisor 之后的路由"""
    next_agent = state.get("next_agent", "__end__")
    if next_agent == "__end__":
        return "__end__"
    return next_agent


# ============================================================
# 构建 LangGraph StateGraph
# ============================================================

def build_graph() -> StateGraph:
    """构建多智能体工作流图（名人对话场景：无 InfoGap 追问，无 Coder）"""

    workflow = StateGraph(AgentState)

    workflow.add_edge(START, "supervisor")

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retriever", retrieval_agent)
    workflow.add_node("analyzer", analysis_agent)
    workflow.add_node("verifier", verification_agent)

    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retriever": "retriever",
            "analyzer": "analyzer",
            "verifier": "verifier",
            "__end__": END,
        },
    )

    workflow.add_edge("retriever", "supervisor")
    workflow.add_edge("analyzer", "supervisor")
    workflow.add_edge("verifier", "supervisor")

    return workflow.compile()


# 全局图实例（persona 场景图，编译一次，缓存复用）
_persona_graph = None


def get_persona_graph():
    """获取 persona 场景的 LangGraph 图（缓存单例）"""
    global _persona_graph
    if _persona_graph is None:
        _persona_graph = build_graph()
    return _persona_graph
