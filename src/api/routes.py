"""
FastAPI 路由端点
提供名人对话查询、文档上传、健康检查等 API
"""

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from framework.supervisor import (
    set_scene_config,
    get_persona_graph, get_chroma_client,
    get_conversation_history, append_conversation, _generate_direct_response,
    _conversation_history_store,
    _summarize_old_turns, delete_conversation_history,
)
from src.core.config import settings
from src.core.state import AgentState
from src.core.session_store import get_store as _get_session_store
from src.core.logger import get_logger, set_request_id
from src.core.content_filter import contains_sensitive, sanitize_output, refusal_message
from src.core.monitor import get_monitor
from src.document_processing.parser import DocumentParser
from src.retrieval.chunker import AdaptiveChunker
from src.retrieval.embedder import get_embedder

# persona_chat 场景
from scenes.persona_chat.config import persona_chat_config

router = APIRouter()


# ============================================================
# 请求/响应模型
# ============================================================

class UploadResponse(BaseModel):
    """上传响应"""
    document_id: str
    filename: str
    status: str


# ============================================================
# 会话存储（临时内存存储，后续可替换为 Redis）
# ============================================================

_session_store: dict[str, dict] = {}

# 答案缓存：character_id::query -> (expire_ts, answer, route_history)
# 只缓存"新会话首问"，避免带上下文的追问被错误命中
_ANSWER_CACHE: dict[str, tuple[float, str, list]] = {}
_ANSWER_CACHE_TTL = 3600

# 限流存储：ip -> {"min": 窗口起点, "min_count": 分钟计数, "day": 窗口起点, "day_count": 天计数}
_RATE_STORE: dict[str, dict] = {}


def _client_ip(request: Request) -> str:
    """获取客户端 IP（部署在 nginx 后需启用 uvicorn --proxy-headers 信任 X-Forwarded-For）"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate(ip: str, per_minute: int = 0, per_day: int = 0) -> None:
    """固定窗口限流：超过限制抛 429（带 Retry-After）"""
    now = time.time()
    cur = _RATE_STORE.get(ip)
    if not cur:
        cur = {"min": now, "min_count": 0, "day": now, "day_count": 0}
        _RATE_STORE[ip] = cur
    if now - cur["min"] > 60:
        cur["min"], cur["min_count"] = now, 0
    if now - cur["day"] > 86400:
        cur["day"], cur["day_count"] = now, 0
    if per_minute and cur["min_count"] >= per_minute:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试", headers={"Retry-After": "60"})
    if per_day and cur["day_count"] >= per_day:
        retry = int(86400 - (now - cur["day"]))
        raise HTTPException(status_code=429, detail="今日使用次数已达上限，请明天再来", headers={"Retry-After": str(retry)})
    cur["min_count"] += 1
    cur["day_count"] += 1


def _persist_session_db(session_id: str) -> None:
    """把会话状态（归属角色 / 最近问题 / 路由 / 答案）写入 SQLite，重启后可恢复"""
    try:
        data = _session_store.get(session_id, {})
        store = _get_session_store()
        store.ensure(session_id)
        store.set_meta(
            session_id,
            character=data.get("character", ""),
            query=data.get("query", ""),
            route_history=data.get("route_history", []),
            final_answer=data.get("final_answer", ""),
        )
    except Exception as e:
        print(f"[History] ⚠️ 写入会话持久化失败: {e}")


@router.get("/admin/stats")
async def admin_stats(token: str = ""):
    """
    成本 / 错误监控：今日与本周汇总（请求数、错误、缓存命中率、费用、平均延迟、按角色拆分）。
    配置 MONITOR_TOKEN 后需带 ?token=xxx 访问；留空则直接可看（上线前务必设置）。
    """
    if settings.monitor_token and token != settings.monitor_token:
        raise HTTPException(status_code=403, detail="token 无效")
    monitor = get_monitor()
    return {
        "today": monitor.summary(monitor.since_start_of_day()),
        "week": monitor.summary(time.time() - 7 * 86400),
    }

# 会话归属记录持久化文件（session_id -> character_id）：
# 后端只记录会话"首次提问"时的角色，后续切换角色不会覆盖，
# 因此是修复前端被污染的历史对话归属（characterId 被改写、
# 消息缺失 sceneIcon 的旧数据）的最可靠依据。
# 服务重启后仍可识别"这个会话最初是跟谁对话的"。
_SESSION_META_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "session_meta.json"


def _persist_session_meta():
    """把会话归属记录写盘（服务重启后仍可识别历史对话属于哪个角色）"""
    try:
        _SESSION_META_FILE.parent.mkdir(parents=True, exist_ok=True)
        meta = {sid: d.get("character", "") for sid, d in _session_store.items() if d.get("character")}
        _SESSION_META_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[History] ⚠️ 写入会话归属记录失败: {e}")


def _load_session_meta():
    """启动时加载持久化的会话归属记录（仅恢复 character 字段）"""
    try:
        if _SESSION_META_FILE.exists():
            raw = json.loads(_SESSION_META_FILE.read_text(encoding="utf-8"))
            for sid, char_id in raw.items():
                if char_id and sid not in _session_store:
                    _session_store[sid] = {
                        "session_id": sid,
                        "character": char_id,
                        "query": "",
                        "route_history": [],
                        "final_answer": "",
                    }
            if raw:
                print(f"[History] ✅ 已加载 {len(raw)} 条会话归属记录")
    except Exception as e:
        print(f"[History] ⚠️ 加载会话归属记录失败: {e}")


_load_session_meta()

# 已在前端删除过的会话 ID 集合（内存级）：
# 用户删除对话后，即使前端因缓存/页面恢复等原因再次携带旧历史请求，
# 后端也拒绝恢复，防止“删了对话后重新问同样的问题被误判为重复提问”。
_deleted_sessions: set[str] = set()

# 前端历史恢复锁：两个请求（如同一浏览器两个标签页共享 sessionId）
# 同时携带各自历史到达时，恢复操作必须串行，避免后写覆盖先写
_history_restore_lock = threading.Lock()


def _restore_history_from_frontend(session_id: str, history: list[dict]):
    """
    从前端发来的对话历史恢复到后端 store。
    解决服务重启后后端丢失上下文的问题。
    """
    if not history:
        return
    # 用户已在前端删除该会话：拒绝用前端历史恢复（防止重复提问误判）
    if session_id in _deleted_sessions:
        print(f"[History] ⚠️ 会话 {session_id} 已删除，忽略前端发来的历史（防止重复提问误判）")
        return
    # 只在后端历史为空时恢复（避免覆盖已有数据）
    with _history_restore_lock:
        if session_id in _conversation_history_store and _conversation_history_store[session_id]:
            return
        # 将前端历史转为 (query, answer) 元组
        restored = []
        i = 0
        while i < len(history):
            msg = history[i]
            if msg.get("type") == "user" or msg.get("role") == "user":
                user_msg = msg.get("content", "")
                # 找下一个助手消息
                assistant_msg = ""
                if i + 1 < len(history):
                    next_msg = history[i + 1]
                    if next_msg.get("type") == "assistant" or next_msg.get("role") == "assistant":
                        assistant_msg = next_msg.get("content", "")
                        i += 1
                # 跳过没有助手回答的轮次（通常是当前正在问的问题，不是历史），
                # 避免把"用户：X 助手：（空）"写入历史，导致 LLM 误判用户重复提问
                if user_msg and assistant_msg:
                    restored.append((user_msg, assistant_msg))
            i += 1
        to_summarize = []
        if restored:
            _conversation_history_store[session_id] = restored
            print(f"[History] ✅ 从前端恢复会话历史: session={session_id}, 轮次={len(restored)}")
            # 恢复的历史超过存储上限时，超出部分异步压缩为摘要，避免早期上下文直接丢失
            max_turns = settings.max_history_turns if hasattr(settings, 'max_history_turns') else 10
            if len(restored) > max_turns:
                to_summarize = restored[:-max_turns]
                _conversation_history_store[session_id] = restored[-max_turns:]
                print(f"[History] 📝 恢复历史超限，压缩 {len(to_summarize)} 轮旧对话为摘要")
    if to_summarize:
        asyncio.create_task(_summarize_old_turns(session_id, to_summarize))


# ============================================================
# API 端点
# ============================================================

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "Multi-Agent RAG Persona Chat is running"}


@router.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """
    删除指定会话的全部后端数据（对话历史 + 摘要 + 会话状态）。
    前端删除对话时调用，确保后端内存中的历史同步清除。
    """
    deleted_history = delete_conversation_history(session_id)
    deleted_session = False
    if session_id in _session_store:
        del _session_store[session_id]
        deleted_session = True
    # 无论后端是否真的存在该会话历史，都标记为“已删除”：
    # 服务重启后后端内存历史可能已丢失（deleted=False），但前端确实删除了对话，
    # 此后同一 session 再次请求时必须拒绝前端携带的旧历史，避免恢复已删除的内容。
    _deleted_sessions.add(session_id)
    _persist_session_meta()
    print(f"[History] 🗑️ 删除会话: session={session_id}, history={deleted_history}, session_store={deleted_session}")
    return {
        "session_id": session_id,
        "deleted": deleted_history or deleted_session,
        "message": "会话已删除" if (deleted_history or deleted_session) else "会话不存在",
    }


@router.get("/conversation/pending/{session_id}")
async def get_pending_answer(session_id: str):
    """
    获取“尚未被前端接收”的回答（刷新页面后恢复用）。

    前端在 SSE 流式传输中断开（刷新/关闭页面）时，后端 run_graph
    作为独立后台任务仍会继续执行，完成后把最终答案写入 _session_store。
    页面重新加载后前端轮询本端点，即可把丢失的回答捞回来。

    返回 query 字段：前端用它校验答案与当前问题是否匹配，
    防止把另一个标签页的回答错贴到当前问题下。
    """
    data = _session_store.get(session_id)
    if not data:
        return {
            "pending": False,
            "query": "",
            "final_answer": "",
            "route_history": [],
        }
    return {
        "pending": bool(data.get("final_answer")),
        "query": data.get("query", ""),
        "final_answer": data.get("final_answer", ""),
        "route_history": data.get("route_history", []),
    }


@router.get("/conversation/{session_id}/meta")
async def conversation_meta(session_id: str):
    """
    查询会话的归属角色（后端权威记录）。

    后端只在会话"首次提问"时记录角色，后续切换角色不会覆盖该记录，
    因此是修复前端被污染的历史对话归属（characterId 被改写）的最可靠依据。
    """
    data = _session_store.get(session_id)
    if not data or not data.get("character"):
        return {"session_id": session_id, "known": False, "character": None}
    return {"session_id": session_id, "known": True, "character": data["character"]}


# ============================================================
# Persona Chat 场景端点（与名人对话）
# ============================================================

class PersonaQueryRequest(BaseModel):
    """名人对话查询请求"""
    query: str
    session_id: Optional[str] = None
    character_id: Optional[str] = "jung"
    history: Optional[list[dict]] = None  # 前端发来的对话历史 [{role, content}]


class CharacterInfo(BaseModel):
    """角色信息"""
    id: str
    name: str
    description: str
    tagline: str = ""
    avatar: str = "🎭"
    theme: str = ""


class EvalQueryRequest(BaseModel):
    """评估查询请求（返回检索上下文）"""
    query: str
    character_id: Optional[str] = "jung"


@router.post("/persona/eval_query")
async def persona_eval_endpoint(request: EvalQueryRequest):
    """
    评估专用端点：返回回答 + 检索到的上下文
    用于 RAG 评估时同时获取回答和检索上下文

    直接用 ChromaDB 向量查询检索相关文档（避免加载全部文档导致 SQLite 溢出）。
    """
    character_id = request.character_id or "jung"
    character = persona_chat_config.characters.get(character_id)
    if not character:
        raise HTTPException(status_code=400, detail=f"角色 '{character_id}' 不存在")

    # 切换到 persona 场景（contextvar 按请求隔离，不影响其他用户）
    from dataclasses import replace as _dc_replace
    set_scene_config(_dc_replace(persona_chat_config, chroma_collection=character.chroma_collection))

    try:
        from langchain_core.documents import Document
        from src.retrieval.embedder import get_embedder
        from src.retrieval.advanced_search import advanced_retrieval
        from src.core.llm import get_chat_llm

        # ---- 1. 高级检索：多查询 + HyDE 组合 ----
        embedder = get_embedder()

        client = get_chroma_client()

        collection_name = character.chroma_collection  # e.g. "persona_jung"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 初始化 LLM（用于生成查询变体和 HyDE 文档；
        # max_tokens 已合并为一次调用，256 足够容纳短变体 + 3-5 句假设答案）
        retrieval_llm = get_chat_llm(
            temperature=0.3,
            max_tokens=256,
        )

        # 执行高级检索
        retrieved_docs, contexts = await advanced_retrieval(
            question=request.query,
            collection=collection,
            llm=retrieval_llm,
            top_k=15,
            use_multi_query=True,
            use_hyde=True,
            num_variants=2,
        )

        print(f"[Eval Query] 高级检索到 {len(retrieved_docs)} 条相关文档")

        # ---- 2. 分析 ----
        from framework.analysis_agent import analysis_agent
        from src.core.state import AgentState

        state: AgentState = {
            "query": request.query,
            "session_id": "eval",
            "retrieved_docs": retrieved_docs,
            "analysis": "",
            "code_result": "",
            "verification": "",
            "final_answer": "",
            "history": [],
            "route_history": [],
            "next_agent": None,
            "error": None,
            "character_role_prompt": character.role_prompt,
            "stream_callback": None,
            "info_gap_questions": None,
        }

        analysis_result = await analysis_agent(state)
        analysis = analysis_result.get("analysis", "")

        # ---- 3. 直接用分析结果作为回答（避免二次生成引入幻觉）----
        # analysis_agent 已有严格 prompt 要求基于参考资料，无需再调 _generate_direct_response
        answer = analysis

        return {
            "answer": answer,
            "contexts": contexts,  # 返回全部上下文，确保 judge 能完整评估
            "analysis": analysis[:500],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"ERROR: {str(e)}", "contexts": [], "analysis": ""}


@router.get("/persona/characters")
async def list_characters():
    """
    获取可用角色列表
    """
    characters = []
    for char_id, char_def in persona_chat_config.characters.items():
        characters.append(CharacterInfo(
            id=char_def.id,
            name=char_def.name,
            description=char_def.description,
            tagline=char_def.tagline,
            avatar=char_def.avatar,
            theme=char_def.theme,
        ))
    return {"characters": characters}


@router.post("/persona/query")
async def persona_query_endpoint(http_request: Request, request: PersonaQueryRequest):
    """
    名人对话查询端点（流式输出）

    临时切换到 persona_chat 场景配置，查询完成后恢复。
    """
    rid = uuid.uuid4().hex[:12]
    set_request_id(rid)
    logger = get_logger("persona.query")
    client_ip = _client_ip(http_request)
    logger.info(
        "start ip=%s char=%s q=%r",
        client_ip, request.character_id or "jung", request.query[:60],
    )
    t0 = time.time()
    _check_rate(client_ip, per_minute=settings.rate_limit_per_minute, per_day=settings.rate_limit_per_day)
    session_id = request.session_id or str(uuid.uuid4())
    character_id = request.character_id or "jung"

    # 输入侧内容安全：命中敏感词直接礼貌拒绝，不进入流水线
    if contains_sensitive(request.query):
        logger.info("blocked_sensitive ip=%s q=%r", client_ip, request.query[:60])

        async def _refusal_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': refusal_message()})}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

        return StreamingResponse(
            _refusal_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # 从前端恢复对话历史（解决服务重启后后端丢失上下文的问题）
    _restore_history_from_frontend(session_id, request.history or [])

    # 获取角色定义
    character = persona_chat_config.characters.get(character_id)
    if not character:
        raise HTTPException(
            status_code=400,
            detail=f"角色 '{character_id}' 不存在。可用角色: {list(persona_chat_config.characters.keys())}",
        )

    if session_id not in _session_store:
        _session_store[session_id] = {
            "session_id": session_id,
            "query": request.query,
            "route_history": [],
            "final_answer": "",
            "scene": "persona_chat",
            "character": character_id,
        }
        # 记录/更新会话归属（服务重启后仍可识别历史对话属于哪个角色）
        _persist_session_meta()
        _persist_session_db(session_id)

    async def event_stream():
        """生成 SSE 事件流（token 级流式）"""
        token_queue: asyncio.Queue = asyncio.Queue()
        graph_error: list[Exception] = []
        # 本请求独立的结果容器：即使两个请求共享同一 session_id（如同一浏览器
        # 两个标签页），也绝不从共享的 _session_store 读最终答案，
        # 避免并发时互相覆盖导致串扰/丢回答
        result_box: dict = {"final_answer": "", "route_history": [], "info_gap_questions": None}
        history_before = get_conversation_history(session_id)
        cache_key = f"{character_id}::{request.query.strip()}"
        was_cache_hit = False

        async def stream_callback(token: str):
            await token_queue.put(token)

        async def run_graph():
            """在后台任务中运行图执行"""
            # persona 场景配置（contextvar 按请求隔离，create_task 复制父上下文）
            # 按角色注入对应的 chroma_collection，避免 adler 用户误检索荣格大库
            # （修复：先前全局 persona_chat_config 的 chroma_collection 写死为 persona_jung）
            from dataclasses import replace as _dc_replace
            char_config = _dc_replace(persona_chat_config, chroma_collection=character.chroma_collection)
            set_scene_config(char_config)
            try:
                # 使用缓存的 persona 图（场景编译一次即可，避免每次请求重新编译）
                graph = get_persona_graph()
                history = get_conversation_history(session_id)
                print(f"[Persona Query] 加载会话历史: session={session_id}, 轮次={len(history)}")

                initial_state: AgentState = {
                    "query": request.query,
                    "session_id": session_id,
                    "retrieved_docs": [],
                    "analysis": "",
                    "code_result": "",
                    "verification": "",
                    "final_answer": "",
                    "history": history,
                    "route_history": [],
                    "next_agent": None,
                    "error": None,
                    "character_role_prompt": character.role_prompt,
                    "enable_verification": character.enable_verification,
                    "stream_callback": stream_callback,
                }

                accumulated = {}
                async for update in graph.astream(initial_state, config={"recursion_limit": 25}, stream_mode="updates"):
                    for node_name, node_output in update.items():
                        accumulated.update(node_output)
                        await token_queue.put({"__node_done__": node_name, "output": node_output})

                answer = accumulated.get("final_answer", "")
                info_gap_qs = accumulated.get("info_gap_questions")
                if answer:
                    answer = sanitize_output(answer)
                    append_conversation(session_id, request.query, answer)
                    result_box.update({
                        "route_history": accumulated.get("route_history", []),
                        "final_answer": answer,
                        "info_gap_questions": info_gap_qs,
                    })
                    # _session_store 仍同步更新（供刷新后恢复答案），
                    # 但 SSE 最终结果只读 result_box，与并发请求互不干扰
                    # query 同步更新：前端刷新后可用它校验“答案属于哪个问题”，
                    # 防止恢复机制把另一个标签页的回答错贴到当前问题下
                    _session_store[session_id].update({**result_box, "query": request.query})
                    _persist_session_db(session_id)
                    if not history_before:
                        _ANSWER_CACHE[cache_key] = (
                            time.time() + _ANSWER_CACHE_TTL,
                            answer,
                            accumulated.get("route_history", []),
                        )
                        print(f"[Persona Query] 💾 已缓存答案: {cache_key}")
            except GraphRecursionError:
                print("[Persona Query] ⚠️ 图执行超出递归限制")
                fallback = await _generate_direct_response(
                    request.query, get_conversation_history(session_id),
                    character.role_prompt, stream_callback,
                )
                append_conversation(session_id, request.query, fallback)
                result_box["final_answer"] = fallback
                _session_store[session_id].update({"final_answer": fallback, "query": request.query})
                _persist_session_db(session_id)
            except Exception as e:
                graph_error.append(e)
            finally:
                await token_queue.put(None)

        try:
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'{character.name}正在思考...'})}\n\n"

            # 新会话首问命中缓存：直接回放，跳过整个图执行
            cached = _ANSWER_CACHE.get(cache_key)
            if not history_before and cached and cached[0] > time.time():
                logger.info("cache_hit ip=%s key=%s", client_ip, cache_key)
                was_cache_hit = True
                cached_answer, cached_route = cached[1], cached[2]
                yield f"data: {json.dumps({'type': 'trace', 'route_history': cached_route})}\n\n"
                yield f"data: {json.dumps({'type': 'result', 'content': cached_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"
                return

            graph_task = asyncio.create_task(run_graph())

            while True:
                item = await token_queue.get()
                if item is None:
                    break
                if isinstance(item, dict) and "__node_done__" in item:
                    node_name = item["__node_done__"]
                    yield f"data: {json.dumps({'type': 'agent_done', 'agent': node_name})}\n\n"
                elif isinstance(item, str):
                    yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"

            await graph_task

            if graph_error:
                raise graph_error[0]

            final_answer = result_box.get("final_answer", "")
            route_history = result_box.get("route_history", [])
            info_gap_qs = result_box.get("info_gap_questions")

            yield f"data: {json.dumps({'type': 'trace', 'route_history': route_history})}\n\n"

            if info_gap_qs and final_answer:
                yield f"data: {json.dumps({'type': 'questions', 'content': final_answer, 'questions': info_gap_qs})}\n\n"
            elif final_answer:
                yield f"data: {json.dumps({'type': 'result', 'content': final_answer})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'result', 'content': '查询完成'})}\n\n"

            logger.info(
                "done ip=%s char=%s elapsed_ms=%d answer_chars=%d",
                client_ip, character_id,
                int((time.time() - t0) * 1000), len(final_answer),
            )
            try:
                prompt_chars = len(request.query) + sum(len(q) + len(a) for q, a in history_before)
                get_monitor().record(
                    request_id=rid,
                    character=character_id,
                    prompt_chars=prompt_chars,
                    answer_chars=len(final_answer),
                    latency_ms=int((time.time() - t0) * 1000),
                    cache_hit=was_cache_hit,
                )
            except Exception as e:
                print(f"[Monitor] ⚠️ 用量记录失败: {e}")
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

        except Exception as e:
            logger.exception("query_error ip=%s char=%s", client_ip, character_id)
            try:
                get_monitor().record(
                    request_id=rid,
                    character=character_id,
                    prompt_chars=len(request.query),
                    latency_ms=int((time.time() - t0) * 1000),
                    error=True,
                )
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/persona/upload")
async def persona_upload_document(
    http_request: Request,
    file: UploadFile = File(...),
    character_id: str = "jung",
):
    """
    上传文档到名人对话知识库

    文档会被存入对应角色的 ChromaDB collection。
    """
    rid = uuid.uuid4().hex[:12]
    set_request_id(rid)
    logger = get_logger("persona.upload")
    client_ip = _client_ip(http_request)
    logger.info("start ip=%s char=%s file=%s", client_ip, character_id, file.filename)
    _check_rate(client_ip, per_minute=5, per_day=settings.upload_limit_per_day)
    character = persona_chat_config.characters.get(character_id)
    if not character:
        raise HTTPException(
            status_code=400,
            detail=f"角色 '{character_id}' 不存在",
        )

    allowed_types = {".pdf", ".md", ".py", ".txt", ".java", ".cpp", ".c", ".js"}
    filename = file.filename or "unknown"
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(allowed_types)}",
        )

    document_id = str(uuid.uuid4())
    max_bytes = settings.upload_max_mb * 1024 * 1024

    # 先按 Content-Length 预检，避免大文件直接打满内存
    content_length = http_request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小限制（最大 {settings.upload_max_mb}MB）",
        )

    try:
        upload_dir = Path("data/raw")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{document_id}_{filename}"
        total = 0
        try:
            with file_path.open("wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"文件超过大小限制（最大 {settings.upload_max_mb}MB）",
                        )
                    buffer.write(chunk)
        except HTTPException:
            file_path.unlink(missing_ok=True)
            raise
        if total == 0:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="文件为空，请重新上传")

        print(f"[Persona Upload] 文件已保存: {file_path}")

        parser = DocumentParser()
        elements = parser.parse(file_path)
        print(f"[Persona Upload] 文档解析完成: {len(elements)} 个元素")

        chunker = AdaptiveChunker(
            min_size=settings.chunk_size // 2,
            max_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk(elements, source=filename)
        print(f"[Persona Upload] 文档分块完成: {len(chunks)} 个块")

        embedder = get_embedder()
        vectors, metadatas = embedder.embed_documents_with_metadata(chunks)

        client = get_chroma_client()

        # 使用角色对应的 collection
        collection = client.get_or_create_collection(
            name=character.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            batch_ids = [f"{document_id}_{j}" for j in range(i, batch_end)]
            batch_texts = [chunks[j].page_content for j in range(i, batch_end)]
            batch_metadatas = [chunks[j].metadata for j in range(i, batch_end)]
            batch_vectors = vectors[i:batch_end]

            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_vectors,
                metadatas=batch_metadatas,
            )

        file_path.unlink()

        # BM25 缓存失效：新文档已入库，下次检索自动重建索引
        from src.retrieval.advanced_search import invalidate_bm25_cache
        invalidate_bm25_cache(character.chroma_collection)

        print(f"[Persona Upload] ✅ 文档已存入 {character.chroma_collection}: {len(chunks)} 个向量")

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            status="completed",
        )

    except Exception as e:
        print(f"[Persona Upload] ❌ 文档处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {str(e)}",
        )
