"""
圆桌会议（Roundtable）编排引擎

将多名名人角色拉入同一议题，按回合轮流发言、相互反驳，形成多智能体辩论。

设计要点：
- 每位发言者**独立检索自己的知识库**（各自的 ChromaDB collection），保证观点扎根于
  其原著 / 思想，而不是凭空生成；检索用项目统一的 Embedder（与入库时向量一致），
  因此不会出现"默认 embedding 函数维度不匹配、检索到垃圾"的问题。
- 每位发言者以**第一人称 + 角色人设 + 圆桌辩论指令**生成，对其他与会者的观点直接
  回应 / 交锋，制造真实的思想碰撞。
- 走 LangGraph 之外的轻量编排：每轮每位发言者一次 LLM 流式调用，事件以 async generator
  产出，与 routes.py 既有的 SSE 模式一致（start / round / speaker_start / token /
  speaker_end / end / error）。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from src.core.llm import get_chat_llm, astream_nonempty
from framework.supervisor import get_chroma_client
from scenes.persona_chat.config import persona_chat_config
from scenes.persona_chat.prompt_builder import build_character_prompt
from src.retrieval.embedder import get_embedder


# 圆桌人数上限：人太多会摊薄每位的发言篇幅，削弱交锋感，反而达不到辩论效果。
MAX_ROUNDTABLE_CHARS = 4


# 发言间隔（秒）：每位发言结束后停顿，既给用户留出阅读上一段的时间，
# 也把 LLM 调用频率压到 DeepSeek RPM 限额（默认 20/min）以内，避免 429。
ROUNDTABLE_SPEAKER_PAUSE = 4.0


# ============================================================
# 圆桌辩论专用指令（叠加在角色人设之上，约束"会议"形态）
# ============================================================
ROUNDTABLE_DEBATE_DIRECTIVE = """
你正参加一场圆桌辩论（Roundtable）。多位历史人物齐聚，就同一议题各自陈词、相互交锋。

【你的任务】
1. 始终以第一人称、用你本人的语言习惯与思维方式发言；绝不跳出角色，绝不以"AI / 主持人 / 总结者"口吻说话。
2. 鲜明表达你对该议题的立场，从你自己的学说、经历、时代与价值取向出发；允许直接、但不失礼地反驳其他与会者。
3. 若下方提供了你的思想资料库，请优先援引你本人的原著 / 观点作为论据；资料未覆盖之处，用你视角的合理推演，但不要编造具体书名或伪造引文。
4. 若提供了其他与会者的观点，请针对其中的分歧点做出回应或追问，制造真实的思想交锋；不要泛泛而谈、不要和稀泥。
5. 控制篇幅（开场陈述 150-260 字；后续交锋 120-220 字），像现场辩论一样利落有力。
""".strip()


# ============================================================
# 检索：每位发言者只检索自己的知识库
# ============================================================
async def _retrieve_for_speaker(collection_name: str, query: str, top_k: int = 4) -> str:
    """
    检索某发言者自己知识库中最相关的片段，作为其发言的参考资料。

    用项目统一 Embedder 把 query 向量化，再走 ChromaDB 的 query_embeddings，
    保证与入库时（upload 路径用同一 Embedder）的向量空间一致。
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        embedder = get_embedder()
        emb = embedder.embed_query(query)
        res = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        docs = (res.get("documents") or [[]])[0]
        if not docs:
            return ""
        # 去重 + 拼接
        seen: set[str] = set()
        parts: list[str] = []
        for d in docs:
            if d and d.strip() and d not in seen:
                seen.add(d)
                parts.append(d.strip())
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[Roundtable] 检索失败 col={collection_name}: {e}")
        return ""


def _format_debate_so_far(transcript: list[dict], exclude_id: Optional[str] = None) -> str:
    """把已有发言格式化为『其他与会者观点』，供当前发言者参考（默认排除自己）。

    截断策略：最多保留最近约 1500 字（从最早的发言开始丢弃），
    既保证交锋轮输入的上下文长度稳定（避免 DeepSeek "message too long"），
    又让发言者聚焦最新一轮的分歧点。
    """
    if not transcript:
        return ""
    lines = []
    for turn in transcript:
        if exclude_id and turn.get("character_id") == exclude_id:
            continue
        lines.append(f"【{turn.get('name', '发言人')}】{turn.get('content', '')}")
    text = "\n\n".join(lines)
    while len(text) > 1500 and lines:
        lines.pop(0)
        text = "\n\n".join(lines)
    return text


# ============================================================
# 构造单轮发言的 prompt
# ============================================================
async def build_speaker_messages(
    *,
    character,
    topic: str,
    transcript: list[dict],
    round_index: int,
    total_rounds: int,
    is_opening: bool,
) -> list[tuple[str, str]]:
    """
    拼装某发言者本轮的 LangChain 消息列表（system / user）。
    检索异步进行，故本函数为 async。
    """
    config = persona_chat_config
    voice = getattr(config, "persona_voice_directive", "") or ""
    # 酒馆式装配：示例对话与世界书让发言者在交锋中也不丢自己的语气；
    # topic 用于世界书关键词命中（开场轮与交锋轮都用议题本身匹配）。
    role_prompt = build_character_prompt(character, query=topic) or getattr(
        config, "direct_response_system_prompt", ""
    )

    # 检索：开场轮用议题本身；交锋轮加入最近他人观点，使其检索更聚焦分歧
    retrieval_query = topic
    if not is_opening:
        others = _format_debate_so_far(transcript, exclude_id=character.id)
        if others:
            retrieval_query = f"{topic}\n\n其他与会者近期观点：\n{others[-700:]}"

    context = await _retrieve_for_speaker(character.chroma_collection, retrieval_query, top_k=4)
    others_view = _format_debate_so_far(transcript, exclude_id=character.id)

    system_parts = [role_prompt, ROUNDTABLE_DEBATE_DIRECTIVE]
    if voice:
        system_parts.append(voice)
    messages: list[tuple[str, str]] = [("system", "\n\n".join(system_parts))]

    if context:
        messages.append(("system",
            "【你的思想资料库（来自你的原著 / 传记，优先援引其中的观点与论据）】\n" + context))

    if others_view:
        messages.append(("system",
            "【其他与会者的发言（请针对其中的分歧点回应或反驳，制造思想交锋）】\n" + others_view))

    if is_opening:
        user_msg = (
            f"议题：{topic}\n\n"
            f"这是圆桌的开场陈述。请首先亮明你对这一议题的核心立场与理由，"
            f"用你本人才有的视角、学说与例证，让听众一眼认出“这就是某某”。"
        )
    else:
        user_msg = (
            f"议题：{topic}\n\n"
            f"这是第 {round_index}/{total_rounds} 轮交锋。请回应其他与会者的观点，"
            f"进一步阐述、修正或捍卫你的立场；若有相左之处，直接而得体地反驳。"
        )
    messages.append(("user", user_msg))
    return messages


# ============================================================
# 流式 LLM 调用：429/限流指数退避重试
# ============================================================
async def _stream_llm_with_retry(messages, label: str = "", max_retries: int = 2):
    """流式调用 LLM，逐 token 产出；对 429/限流错误在**未产出任何 token** 时退避重试。

    已在中途产出部分内容后失败，则放弃重试（避免重复内容），由调用方兜底。
    """
    attempts = 0
    while True:
        llm = get_chat_llm(temperature=0.85, max_tokens=900)
        got = ""
        try:
            async for tok in astream_nonempty(llm, messages):
                if tok:
                    got += tok
                    yield tok
            return
        except Exception as e:
            attempts += 1
            err = str(e)
            is_quota = "429" in err or "rpm" in err.lower() or "quota" in err.lower()
            if is_quota and attempts <= max_retries and not got:
                await asyncio.sleep(2 * attempts)  # 2s / 4s 指数退避
                continue
            print(f"[Roundtable] 发言流式失败 {label}（attempts={attempts}）: {e}")
            return


# ============================================================
# 编排主流程：产出 SSE 事件流
# ============================================================
async def stream_roundtable(
    topic: str,
    character_ids: list[str],
    rounds: int,
    session_id: str,
) -> AsyncIterator[dict]:
    """
    圆桌会议主流程：开场陈述 + rounds 轮交锋，逐位发言、流式产出事件。

    事件类型（前端据此渲染）：
      start           {session_id, topic, speakers:[{id,name,avatar,theme}]}
      round           {round, total, phase: "opening"|"rebuttal"}
      speaker_start   {character_id, name, avatar, theme}
      token           {character_id, content}
      speaker_end     {character_id, content}
      end             {session_id}
      error           {content}
    """
    from src.core.content_filter import sanitize_output

    # 解析有效发言者（去重、过滤不存在的角色）
    seen_ids: set[str] = set()
    speakers = []
    for cid in character_ids:
        if cid in seen_ids:
            continue
        ch = persona_chat_config.characters.get(cid)
        if not ch:
            continue
        seen_ids.add(cid)
        speakers.append(ch)

    if len(speakers) < 2:
        yield {"type": "error", "content": "圆桌会议至少需要 2 位有效的对话者。"}
        return

    if len(speakers) > MAX_ROUNDTABLE_CHARS:
        yield {
            "type": "error",
            "content": f"圆桌人数过多（上限 {MAX_ROUNDTABLE_CHARS} 位）。人太多反而难以形成有效交锋，请精简参与人物。",
        }
        return

    rounds = max(1, min(int(rounds), 3))  # 钳制 1-3 轮，避免过度消耗

    speaker_meta = [
        {"id": ch.id, "name": ch.name, "avatar": ch.avatar or "🎭", "theme": ch.theme or "original"}
        for ch in speakers
    ]
    yield {
        "type": "start",
        "session_id": session_id,
        "topic": topic,
        "speakers": speaker_meta,
    }

    transcript: list[dict] = []  # {character_id, name, content}

    # ---- 开场陈述 ----
    yield {"type": "round", "round": 0, "total": rounds, "phase": "opening"}
    for ch in speakers:
        yield {
            "type": "speaker_start",
            "character_id": ch.id,
            "name": ch.name,
            "avatar": ch.avatar or "🎭",
            "theme": ch.theme or "original",
        }
        messages = await build_speaker_messages(
            character=ch, topic=topic, transcript=transcript,
            round_index=0, total_rounds=rounds, is_opening=True,
        )
        full = ""
        async for tok in _stream_llm_with_retry(messages, label=f"{ch.id} 开场"):
            full += tok
            yield {"type": "token", "character_id": ch.id, "content": tok}
        if not full:
            full = f"（{ch.name} 暂时无法发言。）"
            yield {"type": "token", "character_id": ch.id, "content": full}
        full = sanitize_output(full).strip()
        transcript.append({"character_id": ch.id, "name": ch.name, "content": full})
        yield {"type": "speaker_end", "character_id": ch.id, "content": full}
        await asyncio.sleep(ROUNDTABLE_SPEAKER_PAUSE)

    # ---- 交锋轮 ----
    for r in range(1, rounds + 1):
        yield {"type": "round", "round": r, "total": rounds, "phase": "rebuttal"}
        for ch in speakers:
            yield {
                "type": "speaker_start",
                "character_id": ch.id,
                "name": ch.name,
                "avatar": ch.avatar or "🎭",
                "theme": ch.theme or "original",
            }
            messages = await build_speaker_messages(
                character=ch, topic=topic, transcript=transcript,
                round_index=r, total_rounds=rounds, is_opening=False,
            )
            full = ""
            async for tok in _stream_llm_with_retry(messages, label=f"{ch.id} 交锋{r}"):
                full += tok
                yield {"type": "token", "character_id": ch.id, "content": tok}
            if not full:
                full = f"（{ch.name} 暂时无法发言。）"
                yield {"type": "token", "character_id": ch.id, "content": full}
            full = sanitize_output(full).strip()
            transcript.append({"character_id": ch.id, "name": ch.name, "content": full})
            yield {"type": "speaker_end", "character_id": ch.id, "content": full}
            await asyncio.sleep(ROUNDTABLE_SPEAKER_PAUSE)

    yield {"type": "end", "session_id": session_id}
