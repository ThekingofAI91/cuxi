"""
FastAPI 路由端点
提供名人对话查询、文档上传、健康检查等 API
"""

import asyncio
import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from framework.supervisor import (
    set_scene_config, get_scene_config,
    get_persona_graph, get_chroma_client,
    get_conversation_history, append_conversation, _generate_direct_response,
    _conversation_history_store, _conversation_summaries,
    _summarize_old_turns, delete_conversation_history,
)
from src.core.config import settings
from src.core.state import AgentState
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
    return {"status": "ok", "message": "Multi-Agent RAG Academic Assistant is running"}


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


@router.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    查询端点（流式输出）
    
    输入：{query: str, session_id?: str}
    输出：SSE 流式响应（token 逐字推送）
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    # 从前端恢复对话历史（解决服务重启后后端丢失上下文的问题）
    _restore_history_from_frontend(session_id, request.history or [])
    
    if session_id not in _session_store:
        _session_store[session_id] = {
            "session_id": session_id,
            "query": request.query,
            "route_history": [],
            "final_answer": "",
        }
    
    async def event_stream():
        """生成 SSE 事件流（token 级流式）"""
        token_queue: asyncio.Queue = asyncio.Queue()
        graph_error: list[Exception] = []
        # 本请求独立的结果容器：即使两个请求共享同一 session_id（如同一浏览器
        # 两个标签页），也绝不从共享的 _session_store 读最终答案，
        # 避免并发时互相覆盖导致串扰/丢回答
        result_box: dict = {"final_answer": "", "route_history": [], "info_gap_questions": None}

        async def stream_callback(token: str):
            await token_queue.put(token)

        async def run_graph():
            """在后台任务中运行图执行"""
            try:
                # 显式注入学术场景配置（contextvar 仅对当前请求生效，
                # 与同时进行的 persona 请求互不干扰）
                set_scene_config(academic_config)
                graph = get_graph()
                history = get_conversation_history(session_id)

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
                    "info_gap_questions": None,
                    "character_role_prompt": "",
                    "stream_callback": stream_callback,
                }

                # 使用 astream 实现节点级流式（stream_mode="updates" 只返回节点输出）
                # 累积状态以提取最终结果
                accumulated = {}
                async for update in graph.astream(initial_state, config={"recursion_limit": 25}, stream_mode="updates"):
                    for node_name, node_output in update.items():
                        accumulated.update(node_output)
                        await token_queue.put({"__node_done__": node_name, "output": node_output})

                # 保存对话历史
                answer = accumulated.get("final_answer", "")
                info_gap_qs = accumulated.get("info_gap_questions")
                if answer:
                    append_conversation(session_id, request.query, answer)
                    result_box.update({
                        "route_history": accumulated.get("route_history", []),
                        "final_answer": answer,
                        "info_gap_questions": info_gap_qs,
                    })
                    # _session_store 仍同步更新（供 /trace 端点查询路由历史），
                    # 但 SSE 最终结果只读 result_box，与并发请求互不干扰
                    # query 同步更新：前端刷新后可用它校验“答案属于哪个问题”，
                    # 防止恢复机制把另一个标签页的回答错贴到当前问题下
                    _session_store[session_id].update({**result_box, "query": request.query})
            except GraphRecursionError:
                print("[Query] ⚠️ 图执行超出递归限制")
                fallback = await _generate_direct_response(request.query, get_conversation_history(session_id), "", stream_callback)
                append_conversation(session_id, request.query, fallback)
                result_box["final_answer"] = fallback
                _session_store[session_id].update({"final_answer": fallback, "query": request.query})
            except Exception as e:
                graph_error.append(e)
            finally:
                await token_queue.put(None)  # 结束信号

        try:
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'thinking', 'content': '正在分析问题...'})}\n\n"

            # 启动后台图执行
            graph_task = asyncio.create_task(run_graph())

            # 消费 token 队列，实时推送给前端
            while True:
                item = await token_queue.get()
                if item is None:
                    break
                if isinstance(item, dict) and "__node_done__" in item:
                    # 节点完成事件（可选：发送进度）
                    node_name = item["__node_done__"]
                    yield f"data: {json.dumps({'type': 'agent_done', 'agent': node_name})}\n\n"
                elif isinstance(item, str):
                    # 流式 token
                    yield f"data: {json.dumps({'type': 'token', 'content': item})}\n\n"

            await graph_task

            # 检查是否有错误
            if graph_error:
                raise graph_error[0]

            # 发送最终结果（包含完整文本，用于前端最终渲染）
            # 读本请求独立的 result_box：并发请求不会互相覆盖
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

            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

        except Exception as e:
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


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档端点
    
    输入：PDF/MD/PY/TXT 文件
    输出：文档 ID + 状态
    """
    # 验证文件类型
    allowed_types = {".pdf", ".md", ".py", ".txt", ".java", ".cpp", ".c", ".js"}
    filename = file.filename or "unknown"
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。支持的类型: {', '.join(allowed_types)}",
        )

    # 生成文档 ID
    document_id = str(uuid.uuid4())

    try:
        # ---- 1. 保存文件到本地 -===-
        upload_dir = Path("data/raw")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / f"{document_id}_{filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[Upload] 文件已保存: {file_path}")

        # ---- 2. 解析文档 ----
        parser = DocumentParser()
        elements = parser.parse(file_path)
        print(f"[Upload] 文档解析完成: {len(elements)} 个元素")

        # ---- 3. 分块 ----
        chunker = AdaptiveChunker(
            min_size=settings.chunk_size // 2,
            max_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk(elements, source=filename)
        print(f"[Upload] 文档分块完成: {len(chunks)} 个块")

        # ---- 4. 向量化并存入 ChromaDB ----
        embedder = get_embedder()
        vectors, metadatas = embedder.embed_documents_with_metadata(chunks)

        client = get_chroma_client()

        collection = client.get_or_create_collection(
            name="academic_docs",
            metadata={"hnsw:space": "cosine"},
        )

        # 批量添加到 ChromaDB
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

        # ---- 5. 清理临时文件 ----
        file_path.unlink()

        # BM25 缓存失效：新文档已入库，旧索引检索不到新内容，下次检索自动重建
        from src.retrieval.advanced_search import invalidate_bm25_cache
        invalidate_bm25_cache("academic_docs")

        print(f"[Upload] ✅ 文档处理完成: {len(chunks)} 个向量已存入 ChromaDB")

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            status="completed",
        )

    except Exception as e:
        print(f"[Upload] ❌ 文档处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {str(e)}",
        )


@router.get("/trace/{session_id}", response_model=TraceResponse)
async def get_trace(session_id: str):
    """
    获取 Agent 路由记录
    
    输入：session_id
    输出：路由历史（用于前端可视化）
    """
    if session_id not in _session_store:
        raise HTTPException(
            status_code=404,
            detail=f"会话 {session_id} 不存在",
        )
    
    session_data = _session_store[session_id]
    
    return TraceResponse(
        session_id=session_id,
        route_history=session_data.get("route_history", []),
    )


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
    avatar: str = "🎭"


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
        from langchain_openai import ChatOpenAI

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
        retrieval_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=256,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
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
            avatar=char_def.avatar,
        ))
    return {"characters": characters}


@router.post("/persona/query")
async def persona_query_endpoint(request: PersonaQueryRequest):
    """
    名人对话查询端点（流式输出）
    
    临时切换到 persona_chat 场景配置，查询完成后恢复。
    """
    session_id = request.session_id or str(uuid.uuid4())
    character_id = request.character_id or "jung"

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

    async def event_stream():
        """生成 SSE 事件流（token 级流式）"""
        token_queue: asyncio.Queue = asyncio.Queue()
        graph_error: list[Exception] = []
        # 本请求独立的结果容器：即使两个请求共享同一 session_id（如同一浏览器
        # 两个标签页），也绝不从共享的 _session_store 读最终答案，
        # 避免并发时互相覆盖导致串扰/丢回答
        result_box: dict = {"final_answer": "", "route_history": [], "info_gap_questions": None}

        async def stream_callback(token: str):
            await token_queue.put(token)

        async def run_graph():
            """在后台任务中运行图执行"""
            # persona 场景配置（contextvar 按请求隔离，create_task 复制父上下文，
            # 不影响同时进行的学术请求）
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
                    append_conversation(session_id, request.query, answer)
                    result_box.update({
                        "route_history": accumulated.get("route_history", []),
                        "final_answer": answer,
                        "info_gap_questions": info_gap_qs,
                    })
                    # _session_store 仍同步更新（供 /trace 端点查询路由历史），
                    # 但 SSE 最终结果只读 result_box，与并发请求互不干扰
                    # query 同步更新：前端刷新后可用它校验“答案属于哪个问题”，
                    # 防止恢复机制把另一个标签页的回答错贴到当前问题下
                    _session_store[session_id].update({**result_box, "query": request.query})
            except GraphRecursionError:
                print("[Persona Query] ⚠️ 图执行超出递归限制")
                fallback = await _generate_direct_response(
                    request.query, get_conversation_history(session_id),
                    character.role_prompt, stream_callback,
                )
                append_conversation(session_id, request.query, fallback)
                result_box["final_answer"] = fallback
                _session_store[session_id].update({"final_answer": fallback, "query": request.query})
            except Exception as e:
                graph_error.append(e)
            finally:
                await token_queue.put(None)

        try:
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'{character.name}正在思考...'})}\n\n"

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

            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

        except Exception as e:
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
    file: UploadFile = File(...),
    character_id: str = "jung",
):
    """
    上传文档到名人对话知识库
    
    文档会被存入对应角色的 ChromaDB collection。
    """
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

    try:
        upload_dir = Path("data/raw")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{document_id}_{filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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
