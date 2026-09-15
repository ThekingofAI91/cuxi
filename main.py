"""
FastAPI 应用启动入口
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.api.setup import router as setup_router
from src.api.security import (
    GlobalRateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from src.core.config import settings

import logging

logger = logging.getLogger(__name__)


def _warmup_sync():
    """
    后台预热：加载 embedding/rerank 模型 + 构建各 collection 的 BM25 索引。

    之前服务重启后的第一个请求要现场完成全部重型初始化
    （全量拉取文档 + BM25 分词构建 + 560M 模型加载，需 30~70 秒），
    该请求的用户会等得非常痛苦，并发请求还会重复构建互相拖死。
    预热在启动后的独立 daemon 线程中同步执行（用户请求到达时直接命中缓存）。

    为什么不用 asyncio.to_thread：实测在 uvicorn 事件循环内用 to_thread 加载
    torch 模型 / 构建 BM25 会偶发死锁（进程挂起、CPU 归零、executor worker 空闲），
    而普通 threading.Thread 中同步执行稳定可复现。
    """
    try:
        print("[Warmup] 开始后台预热（不影响服务启动与使用）...", flush=True)

        # 1. 预热 embedding 模型（触发模型文件加载）
        from src.retrieval.embedder import get_embedder
        embedder = get_embedder()
        embedder.embed_query("预热")
        print("[Warmup] Embedding 模型就绪", flush=True)

        # 2. 预热 Cross-Encoder 重排序模型（560M 参数，首次加载 10-30s）
        from src.retrieval.reranker import get_reranker
        get_reranker().model
        print("[Warmup] Rerank 模型就绪", flush=True)

        # 3. 预热各角色 collection 的 BM25 索引（全量拉取 + 构建，耗时大头）
        from framework.supervisor import get_chroma_client
        from src.retrieval.advanced_search import _ensure_bm25_ready
        from scenes.persona_chat.config import persona_chat_config

        names = {c.chroma_collection for c in persona_chat_config.characters.values()}
        print(f"[Warmup] 待预热 collection: {sorted(names)}", flush=True)


        client = get_chroma_client()
        print("[Warmup] ChromaDB client 就绪", flush=True)
        for name in names:
            try:
                collection = client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
                count = collection.count()
                print(f"[Warmup] {name}: count={count}", flush=True)
                if count == 0:
                    print(f"[Warmup] {name}: 文档库为空，跳过")
                    continue
                _ensure_bm25_ready(collection)
                print(f"[Warmup] {name}: {count} 条文档，BM25 索引就绪")
            except Exception as e:
                print(f"[Warmup] {name} 预热失败: {e}")

        print("[Warmup] 预热全部完成", flush=True)
    except Exception as e:
        print(f"[Warmup] 预热失败（不影响使用，首个请求可能稍慢）: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    from src.core import runtime_config as _runtime_config

    _llm_cfg = _runtime_config.resolve(settings)
    _configured = bool(_llm_cfg["api_key"])

    print("=" * 60)
    print(" Multi-Agent RAG Persona Chat 启动中...")
    print(f" 打开浏览器访问: http://localhost:{settings.api_port}")
    print(f" API 文档: http://localhost:{settings.api_port}/docs")
    print(f" LLM 模型: {_llm_cfg['model']}（来源：{_llm_cfg['source']}）")
    print(f" ChromaDB 路径: {settings.chroma_persist_dir}")
    if not _configured:
        print("-" * 60)
        print(" 尚未配置 LLM API Key")
        print(f"     请打开 http://localhost:{settings.api_port} 在设置页填写自己的 Key，")
        print("     或在项目根目录 .env 中设置 LLM_API_KEY（二选一即可）")
    print("=" * 60)

    # 后台预热，不阻塞服务启动与响应；用独立 daemon 线程同步执行，
    # 避开 uvicorn 事件循环 + asyncio.to_thread 在 Windows 上的死锁问题
    threading.Thread(target=_warmup_sync, daemon=True, name="warmup").start()

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
            print(f"[Session]  已恢复 {len(rows)} 个近期会话")
    except Exception as e:
        print(f"[Session]  会话恢复失败: {e}")

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
                    print(f"[Session]  已清理 {len(removed)} 个过期会话")
            except Exception as e:
                print(f"[Session]  清理任务异常: {e}")

    cleanup_task = asyncio.create_task(_session_cleanup())

    yield

    cleanup_task.cancel()
    
    # 关闭时执行
    print("=" * 60)
    print("应用正在关闭...")
    print("=" * 60)


# 创建 FastAPI 应用
app = FastAPI(
    title="促膝 · Persona Chat",
    description="多智能体RAG名人对话系统 - 基于LangGraph的智能问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# 配置 CORS
# 前端由本服务同源提供，正常不需要放行跨域；保留配置是为了前后端分离部署的场景。
# 注意：allow_origins="*" 与 allow_credentials=True 是规范禁止的组合（浏览器会直接拒绝响应），
# 因此仅在配置了具体域名时才允许携带凭证。
_allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
_allow_all = (not _allowed_origins) or _allowed_origins == ["*"]
if _allow_all:
    logger.warning(
        "CORS 当前为通配 '*'，已自动关闭 allow_credentials；"
        "上线前请把 CORS_ORIGINS 改成具体域名（如 https://yourdomain.com）"
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 网络防护中间件（纯 ASGI：不读 body、不缓冲，对 SSE 流式透明）----
# Starlette 后添加的在外层。刻意排成「限流(内) → CORS(中) → 安全头(外)」，
# 这样被限流的 429 响应依然带得上 CORS 头与安全头，浏览器才读得到错误信息。
# 顺序说明（从外到内）：安全头 → 大小限制 → 限流 → CORS → 路由。
# 把大小限制放在限流外层，是为了让超大请求直接 413，不去消耗限流额度。
if settings.global_rate_limit_per_minute > 0:
    app.add_middleware(
        GlobalRateLimitMiddleware,
        per_minute=settings.global_rate_limit_per_minute,
    )
if settings.max_request_body_mb > 0:
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=settings.max_request_body_mb * 1024 * 1024,
    )
if settings.security_headers_enabled:
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=settings.security_hsts_enabled,
    )

# 注册路由（必须先注册，避免被静态文件拦截）
app.include_router(router)
app.include_router(setup_router)

# 前端静态文件服务
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    # 挂载静态资源（CSS/JS/图片等）
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    # 首页背景图目录：前端按相对路径 assets/bg/<id>.jpg 引用，
    # 必须单独挂载，否则通过后端访问时 404、图片回退为内置 SVG
    assets_path = frontend_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    # 根路径返回 index.html
    # 注意：必须加 no-store，否则浏览器/CDN/反向代理可能缓存旧版前端，
    # 导致用户“强制刷新”后仍拿到未修复的旧 index.html（删除/复活等相关修复不生效）。
    @app.get("/")
    async def serve_frontend():
        return FileResponse(
            str(frontend_path / "index.html"),
            headers={"Cache-Control": "no-store, max-age=0"},
        )


if __name__ == "__main__":
    import uvicorn

    # 单 worker 模式下预加载 AI 模型：实测 uvicorn 事件循环内用 asyncio.to_thread
    # 加载 torch 模型（bge-small / bge-reranker-v2-m3）在 Windows 上会偶发死锁
    # （进程挂起、CPU 归零），提前同步加载则稳定（代价是启动多等 ~1 分钟）。
    # 多 worker（>1）时跳过：父进程预加载只会白占内存，各 worker 由预热线程自行加载。
    if settings.app_workers <= 1:
        try:
            from src.retrieval.embedder import get_embedder
            get_embedder().embed_query("预热")  # 触发 bge-small-zh-v1.5 加载
            print("[Warmup] Embedding 模型预加载完成")
        except Exception as e:
            print(f"[Warmup] Embedding 模型预加载失败: {e}")
        try:
            from src.retrieval.reranker import get_reranker
            get_reranker().model
            print("[Warmup] Rerank 模型预加载完成")
        except Exception as e:
            print(f"[Warmup] Rerank 模型预加载失败: {e}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.app_workers,
        # 热重载默认关闭（避免 tests/ 等目录下 .py 文件变动触发重启）；
        # 开发调试时设置环境变量 APP_RELOAD=1 再启动即可启用
        reload=settings.app_reload,
    )
