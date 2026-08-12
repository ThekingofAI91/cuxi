"""
FastAPI 应用启动入口
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.config import settings


async def _warmup():
    """
    后台预热：加载 embedding/rerank 模型 + 构建各 collection 的 BM25 索引。

    之前服务重启后的第一个请求要现场完成全部重型初始化
    （全量拉取文档 + BM25 分词构建 + 560M 模型加载，需 30~70 秒），
    该请求的用户会等得非常痛苦，并发请求还会重复构建互相拖死。
    预热在启动时异步执行，用户请求到达时直接命中缓存。
    """
    try:
        print("[Warmup] 🔥 开始后台预热（不影响服务启动与使用）...")

        # 1. 预热 embedding 模型（触发模型文件加载）
        from src.retrieval.embedder import get_embedder
        embedder = get_embedder()
        await asyncio.to_thread(embedder.embed_query, "预热")
        print("[Warmup] ✅ Embedding 模型就绪")

        # 2. 预热 Cross-Encoder 重排序模型（560M 参数，首次加载 10-30s）
        from src.retrieval.reranker import get_reranker
        await asyncio.to_thread(lambda: get_reranker().model)
        print("[Warmup] ✅ Rerank 模型就绪")

        # 3. 预热各角色 collection 的 BM25 索引（全量拉取 + 构建，耗时大头）
        from framework.supervisor import get_chroma_client
        from src.retrieval.advanced_search import _ensure_bm25_ready
        from scenes.persona_chat.config import persona_chat_config

        names = {c.chroma_collection for c in persona_chat_config.characters.values()}


        client = get_chroma_client()
        for name in names:
            try:
                collection = client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
                count = collection.count()
                if count == 0:
                    print(f"[Warmup] ⏭️ {name}: 文档库为空，跳过")
                    continue
                await asyncio.to_thread(_ensure_bm25_ready, collection)
                print(f"[Warmup] ✅ {name}: {count} 条文档，BM25 索引就绪")
            except Exception as e:
                print(f"[Warmup] ⚠️ {name} 预热失败: {e}")

        print("[Warmup] ✅ 预热全部完成")
    except Exception as e:
        print(f"[Warmup] ⚠️ 预热失败（不影响使用，首个请求可能稍慢）: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("=" * 60)
    print("🚀 Multi-Agent RAG Persona Chat 启动中...")
    print(f"🌐 打开浏览器访问: http://localhost:{settings.api_port}")
    print(f"📍 API 文档: http://localhost:{settings.api_port}/docs")
    print(f"🤖 LLM 模型: {settings.llm_model}")
    print(f"📦 ChromaDB 路径: {settings.chroma_persist_dir}")
    print("=" * 60)

    # 后台预热，不阻塞服务启动与响应
    asyncio.create_task(_warmup())

    # 启动时恢复近期会话（SQLite 写穿持久化），并启动过期清理任务
    import json as _json

    from src.core.session_store import get_store as _get_session_store
    from framework.supervisor import _conversation_history_store, _conversation_summaries
    from src.api.routes import _session_store

    try:
        rows = _get_session_store().load_recent(settings.session_ttl_days)
        for row in rows:
            sid = row["session_id"]
            if row.get("history"):
                try:
                    _conversation_history_store[sid] = _json.loads(row["history"])
                except Exception:
                    pass
            if row.get("summary"):
                _conversation_summaries[sid] = row["summary"]
            if sid not in _session_store:
                _session_store[sid] = {
                    "session_id": sid,
                    "character": row.get("character", ""),
                    "query": row.get("query", ""),
                    "route_history": _json.loads(row.get("route_history") or "[]"),
                    "final_answer": row.get("final_answer", ""),
                }
        if rows:
            print(f"[Session] ✅ 已恢复 {len(rows)} 个近期会话")
    except Exception as e:
        print(f"[Session] ⚠️ 会话恢复失败: {e}")

    async def _session_cleanup():
        while True:
            await asyncio.sleep(6 * 3600)
            try:
                removed = _get_session_store().delete_expired(settings.session_ttl_days)
                for sid in removed:
                    _conversation_history_store.pop(sid, None)
                    _conversation_summaries.pop(sid, None)
                    _session_store.pop(sid, None)
                if removed:
                    print(f"[Session] 🧹 已清理 {len(removed)} 个过期会话")
            except Exception as e:
                print(f"[Session] ⚠️ 清理任务异常: {e}")

    cleanup_task = asyncio.create_task(_session_cleanup())

    yield

    cleanup_task.cancel()
    
    # 关闭时执行
    print("=" * 60)
    print("👋 应用正在关闭...")
    print("=" * 60)


# 创建 FastAPI 应用
app = FastAPI(
    title="Multi-Agent RAG Persona Chat",
    description="多智能体RAG名人对话系统 - 基于LangGraph的智能问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（必须先注册，避免被静态文件拦截）
app.include_router(router)

# 前端静态文件服务
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    # 挂载静态资源（CSS/JS/图片等）
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    
    # 根路径返回 index.html
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        # 热重载默认关闭（避免 tests/ 等目录下 .py 文件变动触发重启）；
        # 开发调试时设置环境变量 APP_RELOAD=1 再启动即可启用
        reload=settings.app_reload,
    )
