"""
Runtime 能力层 —— 场景上下文 / 会话历史 / 消息装配 / 生成 / 向量库单例

架构变化（2026-10-02：supervisor.py 改名 runtime.py）：
    一对一对话的唯一执行体是 framework/supervisor_agent.py：它自己持有
    search_library 工具（检索实现也在该文件），"这轮查不查、查多深"由模型
    运行时决定（strong 参数）。本模块是它与其他模块依赖的"能力库"——
    消息装配、直接生成、会话历史、向量库单例、场景上下文。

    原 LangGraph 状态图（supervisor_node / route_after_supervisor /
    build_graph / get_persona_graph）以及随之失效的
    resolve_retrieval_strategy（分区路由判据）、_is_lightweight_turn
    （轻聊规则快速通道）已一并删除：一对一不需要图，圆桌争鸣另行设计。

能力清单：
- 消息装配 build_direct_messages（人设 → 用户记忆 → 资料 → 历史 → 后历史指令）
- 直接生成 _generate_direct_response（空响应重试 + 首 token 超时）
- 引用出处摘取 _citation_block（供 routes 层异步补推 type:'citations'）
- 会话历史读写与摘要压缩（_conversation_history_store + session_store 持久化）
- ChromaDB client 单例（首次构造加锁，见 get_chroma_client）
"""

import contextvars
import re
import asyncio
import threading

from src.core.llm import get_chat_llm, astream_nonempty, ainvoke_nonempty
from src.core.config import settings
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

# 上游连续空响应（首 token 超时 + 整次流空，重试全部耗尽）时的对外话术。
# 不再回退到「无资料直答」：那会丢掉本轮检索到的全部资料，给出看似有据、
# 实则凭空的回答，比如实报错更糟。routes.event_stream 会把它包成
# type:'error' 事件推给前端（抛出时保证尚未向用户推送过任何 token）。
_EMPTY_UPSTREAM_MSG = "模型服务暂时没有返回内容，请稍后重试"

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
# 首次构造必须互斥：ChromaDB 1.5.9 的多线程并发首次构造不是线程安全的。
_chroma_lock = threading.Lock()


def get_chroma_client():
    """获取全局 ChromaDB PersistentClient（单例，双重检查加锁）

    为什么要加锁（2026-09-23 实测复现）：ChromaDB 1.5.9 的
    `PersistentClient(path=...)` 在**多线程并发首次构造**时会分别抛出

        AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'
        KeyError: <持久化路径>
        ValueError: Could not connect to tenant default_tenant

    （6 线程并发构造的探针里 6/6 全失败；同一进程内第二次之后 0/6 失败，
    说明炸的是"首次构造"这个窗口，不是后续使用。）

    本项目的启动预热跑在独立 daemon 线程（main.py 的 _warmup），用户请求跑在
    uvicorn 线程，两边会同时触发首次构造 → 表现为**预热整段失败**
    （`[Warmup] 预热失败: 'RustBindingsAPI' object has no attribute 'bindings'`）
    加上**首个请求的检索静默失败**（调用方 try/except 吞掉异常、返回空上下文，
    用户只看到回答少了资料，不知道为什么）。

    加锁把首次构造串行化即可，构造完成后就是纯粹的读复用，没有性能代价。
    """
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
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
        print(f"[Supervisor] 已删除会话历史: session={session_id}")
        try:
            _get_session_store().delete(session_id)
        except Exception as e:
            print(f"[Supervisor] 会话持久化删除失败: {e}")
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
        print(f"[Supervisor] 对话历史持久化失败: {e}")


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
        print(f"[Supervisor] 截断后历史持久化失败: {e}")
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
            print(f"[Supervisor] pop_last_turn 校验失败，最后一轮不是待重答的问题，跳过")
            return False
        turns.pop()
        current = list(turns)
    try:
        _get_session_store().set_history(session_id, current)
    except Exception as e:
        print(f"[Supervisor] 弹轮后历史持久化失败: {e}")
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
            print(f"[Supervisor] 旧对话摘要已更新: session={session_id}, 压缩了{len(turns)}轮")
            try:
                _get_session_store().set_summary(session_id, _conversation_summaries[session_id])
            except Exception as e:
                print(f"[Supervisor] 摘要持久化失败: {e}")
        except Exception as e:
            print(f"[Supervisor] 摘要生成失败: {e}")


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

def build_direct_messages(
    query: str,
    history: list[tuple[str, str]] = None,
    character_role_prompt: str = "",
    session_id: str = None,
    context: str = None,
    zone: str = "education",
    post_history_directive: str = None,
    user_memory: str = None,
) -> list:
    """装配回答用的消息序列：人设 → 用户记忆 → 资料 → 历史 → 后历史指令 → 提问。

    顺序是刻意的，工具化检索路径（framework/tool_agent）复用本函数，
    让"直接回答"与"查完资料再回答"两条路的消息结构完全一致。
    - 资料放在对话历史之前：历史越长，越会稀释"必须基于资料"的硬约束
    - 后历史指令放在历史之后：长对话靠它最后再钉一次人设（对应酒馆 Author's Note）
    这两条顺序别动，动了人设就会漂。
    """
    config = get_scene_config()
    system_prompt = config.direct_response_system_prompt if config else (
        "你是一个友好的AI助手。"
    )

    # 注入角色人设
    if character_role_prompt:
        voice = getattr(config, "persona_voice_directive", "") if config else ""
        system_prompt = f"""{character_role_prompt}

{system_prompt}

{voice}"""

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
    return messages


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
        fallback = config.display_name if config else "AI助手"

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

        # 消息装配统一走 build_direct_messages：工具化检索路径复用同一份，
        # 两条路的消息顺序必须一致，否则人设表现会分叉
        messages = build_direct_messages(
            query,
            history=history,
            character_role_prompt=character_role_prompt,
            session_id=session_id,
            context=context,
            zone=zone,
            post_history_directive=post_history_directive,
            user_memory=user_memory,
        )

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


def _citation_block(verification: str) -> str:
    """从引用核查报告中提取「引用出处」列表，供回答返回后异步推送。

    只取引用出处逐条列表，不输出核查摘要与引用可信度：前端把本函数的返回值
    整体放进折叠区块正文且不解析 markdown，任何统计数字或 `**加粗**`、`---`
    都会原样显示，纯列表最干净。verifier 移出图内后，routes.event_stream 用
    本函数把引用出处渲染成 type:'citations' 事件补推给用户。
    """
    if not verification:
        return ""
    citation_match = re.search(r"### 引用出处\n(.*?)(?=\n###|\Z)", verification, re.DOTALL)
    if not citation_match:
        return ""
    return citation_match.group(1).strip()
