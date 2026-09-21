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

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from framework.supervisor import (
    set_scene_config,
    get_persona_graph, get_chroma_client,
    get_conversation_history, append_conversation, _generate_direct_response,
    _conversation_history_store,
    _summarize_old_turns, delete_conversation_history,
    truncate_conversation_history, pop_last_turn,
)
from framework.roundtable import stream_roundtable, MAX_ROUNDTABLE_CHARS
from src.core.config import settings
from src.core.state import AgentState
from src.core.session_store import get_store as _get_session_store
from src.core.logger import get_logger, set_request_id, stages_begin, stage_mark, stages_snapshot
from src.core.content_filter import contains_sensitive, sanitize_output, refusal_message
from src.core.monitor import get_monitor
from src.core.feedback import get_feedback_store
from src.core import accounts
from src.document_processing.parser import DocumentParser
from src.retrieval.chunker import AdaptiveChunker
from src.retrieval.embedder import get_embedder

# persona_chat 场景
from scenes.persona_chat.config import persona_chat_config

# 酒馆式提示词装配器（人设 + 性格 + 场景 + 世界书 + 示例对话 + 分区语气 + 后历史指令）
from scenes.persona_chat.prompt_builder import (
    build_character_prompt,
    build_post_history_directive,
    has_tavern_components,
    compose_lorebook_scan_text,
)

# 角色卡导出/导入（原生 JSON 格式，含世界书与示例对话）
from scenes.persona_chat.card_io import export_card, parse_card, CARD_FORMAT, CARD_VERSION


def _sampling_for(character) -> dict | None:
    """
    挑出角色卡上的采样覆盖（temperature / top_p / frequency_penalty / presence_penalty）。

    未配置的字段一律不传，让生成代码沿用默认值——既有角色的手感不受影响。
    注意：不要透传 repetition_penalty 这类非 OpenAI 兼容参数——
    DeepSeek API 会以 400 拒收未知字段，本地推理后端的专用参数对 API 无效。
    """
    out = {}
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        v = getattr(character, key, None)
        if v is not None:
            out[key] = v
    return out or None
from scenes.persona_chat.custom_store import (
    save_custom_character,
    delete_custom_character_file,
    generate_custom_id,
    is_valid_custom_id,
    collection_name_for,
    load_custom_characters,
)
from scenes.persona_chat.persona_builder import build_persona
from src.retrieval.ingest import ingest_texts_async
from src.retrieval.knowledge_graph import graph_path, graph_exists, load_graph

# 管理后台页面（/admin）所在目录
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

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

# 用户长期记忆的异步提取任务引用集（防 GC，同 advanced_search._bg_rewrite_tasks 模式）
_MEMORY_TASKS: set = set()

# 限流存储：ip -> {"min": 窗口起点, "min_count": 分钟计数, "day": 窗口起点, "day_count": 天计数}
_RATE_STORE: dict[str, dict] = {}


def _client_ip(request: Request) -> str:
    """获取客户端 IP（部署在 nginx 后需启用 uvicorn --proxy-headers 信任 X-Forwarded-For）"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate(ip: str, per_minute: int = 0, per_day: int = 0, bucket: str = "main") -> None:
    """固定窗口限流：超过限制抛 429（带 Retry-After）。

    bucket：按用途分桶（聊天 / 建角 / 图谱构建 / 反馈各自独立计数），
    避免重操作与轻操作共享额度——比如聊了几句就把"建角"额度吃光，或反之。
    """
    key = f"{bucket}::{ip}"
    now = time.time()
    cur = _RATE_STORE.get(key)
    if not cur:
        cur = {"min": now, "min_count": 0, "day": now, "day_count": 0}
        _RATE_STORE[key] = cur
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
        print(f"[History] 写入会话持久化失败: {e}")


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


# ============================================================
# 意见反馈 + 管理后台控制台
# ============================================================

class FeedbackIn(BaseModel):
    """用户提交的反馈（意见 / bug / 建议）"""
    content: str
    character: str = ""      # 提交时正在对话的角色（便于定位上下文）
    contact: str = ""        # 可选联系方式（QQ / 微信 / 邮箱），留空则匿名
    page: str = ""           # 来源页面标记


class FeedbackStatusIn(BaseModel):
    """管理端更新反馈处理状态"""
    status: str              # pending / resolved / replied
    reply: str = ""          # 仅 replied 时需要填写回复内容


def _require_admin(token: str) -> None:
    """管理端读取接口鉴权：设置 monitor_token 后必须带 ?token=xxx"""
    if settings.monitor_token and token != settings.monitor_token:
        raise HTTPException(status_code=403, detail="token 无效")


@router.post("/admin/feedback")
async def submit_feedback(payload: FeedbackIn, request: Request):
    """
    用户意见反馈提交入口（**无需 token**，所有访客可提交）。
    落盘 SQLite，管理后台 /admin 直接查看与标记处理。

    限流独立分桶（bucket=feedback）：无鉴权端点必须限流，防脚本刷库。
    """
    ip = _client_ip(request)
    _check_rate(ip, per_minute=5, per_day=50, bucket="feedback")
    try:
        row = get_feedback_store().record(
            content=payload.content,
            character=payload.character,
            contact=payload.contact,
            page=payload.page,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"[Feedback] 收到反馈 from={ip} char={payload.character or '无'}: {payload.content[:50]}")
    return {"ok": True, "id": row["id"]}


@router.get("/admin/feedback")
async def list_feedback(token: str = "", status: str = "", limit: int = 200):
    """管理端拉取反馈列表（需 token）"""
    _require_admin(token)
    store = get_feedback_store()
    return {
        "items": store.list(status=status or None, limit=limit),
        "counts": store.count_by_status(),
    }


@router.post("/admin/feedback/{fid}/status")
async def set_feedback_status(fid: int, payload: FeedbackStatusIn, token: str = ""):
    """管理端标记反馈状态（pending/resolved/replied，需 token）"""
    _require_admin(token)
    row = get_feedback_store().set_status(fid, payload.status, payload.reply)
    if not row:
        raise HTTPException(status_code=404, detail="反馈不存在")
    return {"ok": True, "feedback": row}


@router.get("/admin/dashboard")
async def admin_dashboard(token: str = ""):
    """
    管理后台聚合数据（需 token）：用量监控 + 会话统计 + 自定义角色 + 知识图谱状态 + 反馈概览。
    前端 /admin 页面调用此接口渲染控制台。
    """
    _require_admin(token)
    now = time.time()
    # usage 见下方（圆桌数据后），先算会话与圆桌以确立 _known 过滤集

    # 会话统计（按角色拆分会话数 / 对话轮次）
    sessions = _get_session_store().stats(settings.session_ttl_days)
    # 只统计当前可用角色（教育成长区 + 娱乐区），未知会话归 unknown
    _known = set(persona_chat_config.characters.keys()) | {"unknown"}
    sessions["by_character"] = {
        k: v for k, v in sessions.get("by_character", {}).items() if k in _known
    }

    # 圆桌会议会话（议题 / 与会者 / 最近记录）
    rt_items = _get_session_store().list_roundtables(ttl_days=settings.session_ttl_days)
    roundtables = {"total": len(rt_items), "recent": rt_items[:10]}

    # 用量监控（today/week/all）——按角色拆分的"请求/费用/错误"只统计当前可用角色
    monitor = get_monitor()
    usage = {
        "today": monitor.summary(monitor.since_start_of_day()),
        "week": monitor.summary(now - 7 * 86400),
        "all": monitor.summary(0),
    }
    for _u in usage.values():
        if "per_character" in _u:
            _u["per_character"] = {
                k: v for k, v in _u["per_character"].items() if k in _known
            }

    # 自定义角色列表
    custom_objs = load_custom_characters()
    custom_list = [
        {
            "id": c.id,
            "name": c.name,
            "created_at": getattr(c, "created_at", 0),
            "is_custom": getattr(c, "is_custom", False),
        }
        for c in custom_objs.values()
        if getattr(c, "is_custom", False)
    ]

    # 知识图谱状态（遍历所有角色 collection）
    graph_list = []
    for c in persona_chat_config.characters.values():
        col = c.chroma_collection
        if graph_exists(col):
            g = load_graph(col) or {}
            graph_list.append({
                "character": c.id,
                "collection": col,
                "entities": len(g.get("entities", {}) or {}),
                "relations": len(g.get("relations", []) or []),
            })

    # 反馈概览
    fb = get_feedback_store()
    feedback = {"counts": fb.count_by_status(), "recent": fb.list(limit=20)}

    return {
        "generated_at": now,
        "usage": usage,
        "sessions": sessions,
        "roundtables": roundtables,
        "custom_characters": custom_list,
        "custom_count": len(custom_list),
        "graphs": graph_list,
        "feedback": feedback,
    }


@router.get("/admin", include_in_schema=False)
async def serve_admin():
    """管理后台页面（独立 HTML，不暴露在用户聊天界面）"""
    p = _FRONTEND_DIR / "admin.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="管理后台页面不存在，请确认 frontend/admin.html")
    return FileResponse(str(p))


# ---- 法务页面（用户协议 / 隐私政策）：独立静态页，登录弹层与使用须知处均链接 ----
@router.get("/terms", include_in_schema=False)
async def serve_terms():
    p = _FRONTEND_DIR / "terms.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="用户协议页面不存在")
    return FileResponse(str(p))


@router.get("/privacy", include_in_schema=False)
async def serve_privacy():
    p = _FRONTEND_DIR / "privacy.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="隐私政策页面不存在")
    return FileResponse(str(p))

# ============================================================
# 账号系统：注册 / 登录 / 登出 / 当前用户
# ============================================================
# 会话用不透明 token 放 HttpOnly Cookie（SameSite=Lax），服务端 SQLite 查表校验。
# 登录与否目前影响两件事：① 限流主体（登录用户按账号限流，匿名按 IP）；
# ② 后续点数/付费额度的挂载点（账号是收费系统的地基）。
# 匿名用户依旧可正常对话——登录是可选能力，不强制。

class AuthRegisterIn(BaseModel):
    account: str
    password: str
    display_name: Optional[str] = None


class AuthLoginIn(BaseModel):
    account: str
    password: str


def _current_session_token(request: Request) -> str:
    return request.cookies.get(settings.auth_cookie_name, "")


def get_current_user(request: Request) -> Optional[dict]:
    """从 Cookie 解析当前登录用户；未登录返回 None（不抛错）。"""
    try:
        return accounts.get_user_by_session(_current_session_token(request))
    except Exception as e:
        print(f"[Auth] 会话解析失败（按未登录处理）: {e}")
        return None


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=accounts.SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=settings.auth_cookie_secure,
        path="/",
    )


@router.post("/auth/register")
async def auth_register(http_request: Request, payload: AuthRegisterIn, response: Response):
    """注册并直接登录。限流独立分桶（bucket=auth），防脚本批量注册。"""
    _check_rate(_client_ip(http_request), per_minute=5, per_day=20, bucket="auth")
    try:
        user = accounts.create_user(payload.account, payload.password, payload.display_name or "")
        accounts.purge_expired_sessions()
        token = accounts.create_session(user["id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _set_auth_cookie(response, token)
    print(f"[Auth] 新用户注册: {user['account']} ip={_client_ip(http_request)}")
    return {"user": {"id": user["id"], "account": user["account"], "display_name": user["display_name"]}}


@router.post("/auth/login")
async def auth_login(http_request: Request, payload: AuthLoginIn, response: Response):
    _check_rate(_client_ip(http_request), per_minute=5, per_day=20, bucket="auth")
    try:
        user = accounts.authenticate(payload.account, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    accounts.purge_expired_sessions()
    token = accounts.create_session(user["id"])
    _set_auth_cookie(response, token)
    return {"user": {"id": user["id"], "account": user["account"], "display_name": user["display_name"]}}


@router.post("/auth/logout")
async def auth_logout(http_request: Request, response: Response):
    accounts.delete_session(_current_session_token(http_request))
    response.delete_cookie(settings.auth_cookie_name, path="/")
    return {"ok": True}


@router.get("/auth/me")
async def auth_me(http_request: Request):
    user = get_current_user(http_request)
    return {"user": user}


@router.delete("/auth/account")
async def auth_delete_account(http_request: Request):
    """
    注销账号（隐私政策承诺的"删除权"）：立即删除该用户的全部登录会话与账号记录。

    需已登录（凭 Cookie 身份操作，防越权注销他人）。对话记录保存在用户本地
    浏览器，不受影响；用量统计为无身份关联的聚合数据，保留但不指向个人。
    """
    user = get_current_user(http_request)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    accounts.delete_account(user["id"])
    # 隐私承诺同步兑现：注销即删该用户的全部长期记忆条目
    try:
        from src.core.memory import get_memory_store
        n = get_memory_store().delete_user(f"u:{user['id']}")
        if n:
            print(f"[Auth] 已删除用户长期记忆 {n} 条: {user['account']}")
    except Exception as _me:
        print(f"[Auth] 用户记忆删除失败（忽略）: {_me}")
    print(f"[Auth] 账号已注销: {user['account']} ip={_client_ip(http_request)}")
    return {"deleted": True}


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
        print(f"[History] 写入会话归属记录失败: {e}")


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
                print(f"[History] 已加载 {len(raw)} 条会话归属记录")
    except Exception as e:
        print(f"[History] 加载会话归属记录失败: {e}")


_load_session_meta()

# 已在前端删除过的会话 ID 集合：
# 用户删除对话后，即使前端因缓存/页面恢复等原因再次携带旧历史请求，
# 后端也拒绝恢复，防止“删了对话后重新问同样的问题被误判为重复提问”。
# 注意：原先是纯内存集合，服务一重启就丢失、已删会话的“禁止恢复”保护随之失效，
# 导致被删对话的历史（连同 AI 旧回答）在重启后复活。现持久化到磁盘，重启后仍拦截。
_DELETED_SESSIONS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "deleted_sessions.json"
_deleted_sessions: set[str] = set()


def _load_deleted_sessions():
    """启动时加载已删除会话集合（磁盘持久化，重启后仍然拒绝恢复旧历史）"""
    try:
        if _DELETED_SESSIONS_FILE.exists():
            raw = json.loads(_DELETED_SESSIONS_FILE.read_text(encoding="utf-8"))
            _deleted_sessions.update(raw)
            if raw:
                print(f"[History] 已加载 {len(raw)} 条已删除会话记录")
    except Exception as e:
        print(f"[History] 加载已删除会话记录失败: {e}")


def _persist_deleted_sessions():
    """把已删除会话集合写盘，服务重启后仍能拦截这些会话的恢复请求"""
    try:
        _DELETED_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DELETED_SESSIONS_FILE.write_text(
            json.dumps(sorted(_deleted_sessions), ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[History] 写入已删除会话记录失败: {e}")


_load_deleted_sessions()

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
        print(f"[History] 会话 {session_id} 已删除，忽略前端发来的历史（防止重复提问误判）")
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
            print(f"[History] 从前端恢复会话历史: session={session_id}, 轮次={len(restored)}")
            # 恢复的历史超过存储上限时，超出部分异步压缩为摘要，避免早期上下文直接丢失
            max_turns = settings.max_history_turns if hasattr(settings, 'max_history_turns') else 10
            if len(restored) > max_turns:
                to_summarize = restored[:-max_turns]
                _conversation_history_store[session_id] = restored[-max_turns:]
                print(f"[History] 恢复历史超限，压缩 {len(to_summarize)} 轮旧对话为摘要")
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
    # 真正清理 SQLite 持久化层：此前 DELETE 只清内存（_conversation_history_store /
    # _session_store），sessions 表里的行一直残留，服务重启后若前端再次发来该
    # session 的旧历史就会被重新灌回上下文，表现为“被删对话复活、带旧回答”。
    _get_session_store().delete(session_id)
    # 无论后端是否真的存在该会话历史，都标记为“已删除”并持久化：
    # 服务重启后后端内存历史可能已丢失，但前端确实删除了对话，
    # 此后同一 session 再次请求时必须拒绝前端携带的旧历史，避免恢复已删除的内容。
    _deleted_sessions.add(session_id)
    _persist_session_meta()
    _persist_deleted_sessions()
    # 同步失效答案缓存：_ANSWER_CACHE 按（用户, 角色, 问题）索引、不含 session_id，
    # 删除会话后若不清，用户重新提出同一问题会命中旧缓存并原样回放（连 route_history
    # 都是旧的），表现为「删掉的对话凭空复现」。删除接口拿不到该会话用过的全部 key，
    # 故整表清空；缓存只是省一次图执行的优化，清空的代价仅是多跑一次完整流程。
    _ANSWER_CACHE.clear()
    print(f"[History] 删除会话: session={session_id}, history={deleted_history}, session_store={deleted_session}")
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
    # 重新生成：作废本轮旧答案重答（后端弹出最后一轮，重答后写回；跳过答案缓存）
    regenerate: bool = False
    # 历史编辑：把后端存储的会话历史截断到 N 轮（前端删掉被编辑轮次之后的内容重发时携带）
    truncate_to_turns: Optional[int] = None


class CharacterInfo(BaseModel):
    """角色信息"""
    id: str
    name: str
    description: str
    tagline: str = ""
    ability: str = ""
    review: str = ""
    avatar: str = "🎭"
    theme: str = ""
    location: Optional[dict] = None
    is_custom: bool = False
    zone: str = "education"            # 分区：education（教育成长区）| entertainment（娱乐区）
    first_mes: str = ""                # 开场白（进入对话时前端直接展示，不消耗 LLM）


class EvalQueryRequest(BaseModel):
    """评估查询请求（返回检索上下文）"""
    query: str
    character_id: Optional[str] = "jung"


# ============================================================
# 自建角色：研究 / 创建 / 删除
# ============================================================

class CharacterResearchRequest(BaseModel):
    """创建前"智能收集资料"：网络搜索 + LLM 合成人设草稿，供用户预览/微调"""
    name: str                         # 人物名字（必填）
    background: str = ""              # 用户已有的背景（可空）
    web_search: bool = True           # 是否用网络搜索辅助（热门人物建议开启）


class CharacterCreateRequest(BaseModel):
    """创建一个用户自建角色"""
    name: str                         # 人物名字（必填）
    background: str = ""              # 背景知识（必填其一：背景或 role_prompt 至少给一个）
    role_prompt: Optional[str] = None  # 用户写好的角色人设（不给则经网络搜索自动生成）
    char_id: Optional[str] = None     # 可选自定义 id（须 a-z0-9_，2-40 位）；不给则自动生成
    description: str = ""             # 一句话简介（首页卡片）
    tagline: str = ""                 # 个性化标语（首页问候语）
    ability: str = ""                 # 能力标签（首页卡片一句话定位）
    review: str = ""                  # 评语（首页卡片一句点评）
    avatar: str = "🎭"               # 头像 emoji
    theme: str = ""                   # 前端主题 original/paper/noir/street/live（空则 original）
    location: Optional[dict] = None   # 经典地点 {name,lat,lon,note}
    use_web_search: bool = True       # 创建时是否用网络搜索辅助生成人设与背景
    enable_verification: bool = True # 是否启用引用核查（沉浸型人设可关）

    # ---- 酒馆式角色卡构件（均可留空：留空则随人设一同自动生成）----
    first_mes: str = ""               # 开场白
    mes_example: str = ""             # 示例对话（{{user}}/{{char}}）
    scenario: str = ""                # 场景设定
    personality: str = ""             # 性格特质
    lorebook: list = []               # 世界书：[{"keyword","content"}]


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

        # 改写开关与主链路（retrieval_agent）保持同一配置口径——此前这里硬编码
        # use_multi_query=True，导致评估测的是生产根本不跑的配置（主链路默认关改写）
        use_rewrite = settings.rewrite_enabled

        # 执行高级检索
        retrieved_docs, contexts = await advanced_retrieval(
            question=request.query,
            collection=collection,
            llm=retrieval_llm,
            top_k=15,
            use_multi_query=use_rewrite,
            use_hyde=use_rewrite,
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
            # 评估要测的是线上真实行为，故同样走酒馆式装配（人设+性格+场景+世界书+示例）
            "character_role_prompt": build_character_prompt(character, query=request.query),
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
            ability=char_def.ability,
            review=char_def.review,
            avatar=char_def.avatar,
            theme=char_def.theme,
            location=char_def.location,
            is_custom=char_def.is_custom,
            zone=char_def.zone,
            first_mes=char_def.first_mes,
        ))
    return {"characters": characters}


@router.post("/persona/characters/research")
async def research_character(http_request: Request, request: CharacterResearchRequest):
    """
    创建前的"智能收集资料"：

    给定人物名字（+可选背景），用网络搜索收集公开资料，再交给 LLM 合成
    - role_prompt：建议的人设提示词
    - background：建议的背景知识库（markdown）
    - sources：实际用到的网络片段（空表示未搜到，前端据此提示用户手动补充）

    该端点不影响任何存储，仅返回草稿供用户预览、微调后再提交 /persona/characters/create。

    限流独立分桶（bucket=research）：本端点会触发多引擎网络搜索 + LLM 合成，
    属于重操作，不能与聊天共享额度，也不可无限滥用。
    """
    _check_rate(_client_ip(http_request), per_minute=3, per_day=20, bucket="research")

    name = (request.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="请提供人物名字 name")
    if contains_sensitive(name) or contains_sensitive(request.background or ""):
        raise HTTPException(status_code=400, detail="人物名字或背景包含不允许的内容，请调整后再试")

    try:
        role_prompt, background, sources, card = await build_persona(
            name=name,
            user_background=request.background or "",
            use_web_search=request.web_search,
            provided_role_prompt=None,  # 草稿阶段不绑定用户人设，始终由 LLM 生成建议
        )
    except Exception as e:
        print(f"[Research] 生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"资料收集失败: {str(e)}")

    return {
        "name": name,
        "role_prompt": role_prompt,
        "background": background,
        "sources": sources,
        "web_search_used": bool(sources),
        # 酒馆式角色卡草稿：前端可回填进表单，用户能直接改开场白与示例对话
        "first_mes": card.get("first_mes", ""),
        "mes_example": card.get("mes_example", ""),
        "scenario": card.get("scenario", ""),
        "personality": card.get("personality", ""),
        "lorebook": card.get("lorebook", []),
    }


# ============================================================
# 知识图谱 RAG：构建（后台任务）+ 状态查询
# ============================================================
# 构建要对整个 collection 的 chunk 做 LLM 三元组抽取，可能耗时数分钟，
# 因此用后台任务跑，前端轮询状态；图谱文件已存在且未强制时直接返回既有统计。

_graph_jobs: dict[str, dict] = {}  # collection_name -> {status, started_at, stats, error}
# 构建失败时间戳（退避用）：失败后 1 小时内不自动重试，防每次对话重复触发注定失败的构建
_GRAPH_BUILD_FAIL_AT: dict[str, float] = {}
_graph_job_lock = threading.Lock()


class GraphBuildRequest(BaseModel):
    character_id: str
    force: bool = False  # 已存在时是否强制重建


def _resolve_collection(character_id: str):
    """按 character_id 解析出 ChromaDB collection，找不到抛 404"""
    if character_id not in persona_chat_config.characters:
        raise HTTPException(status_code=404, detail=f"角色 '{character_id}' 不存在")
    col_name = persona_chat_config.characters[character_id].chroma_collection
    client = get_chroma_client()
    return client.get_or_create_collection(name=col_name, metadata={"hnsw:space": "cosine"})


async def _run_graph_build(collection_name: str, force: bool = False):
    """后台构建任务：成功/失败都更新 _graph_jobs 状态"""
    if force:
        # 彻底重建：清掉旧成品与续传临时文件，从头抽取
        from src.retrieval.knowledge_graph import invalidate_graph, graph_path
        try:
            invalidate_graph(collection_name)
            tp = graph_path(collection_name).with_suffix(".json.tmp")
            if tp.exists():
                tp.unlink()
        except Exception as ex:
            print(f"[GraphBuild] force 清理失败（继续构建）: {ex}")
    with _graph_job_lock:
        _graph_jobs[collection_name] = {
            "status": "building", "started_at": time.time(), "stats": None, "error": None,
        }
    try:
        from framework.supervisor import get_chroma_client
        from src.retrieval.knowledge_graph import build_knowledge_graph
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        graph = await build_knowledge_graph(collection, resume=not force)
        with _graph_job_lock:
            _graph_jobs[collection_name] = {
                "status": "done", "started_at": _graph_jobs.get(collection_name, {}).get("started_at"),
                "stats": graph.get("stats"), "error": None,
            }
        print(f"[GraphBuild] 后台构建完成: {collection_name} -> {graph.get('stats')}")
    except Exception as e:
        print(f"[GraphBuild] 构建失败 {collection_name}: {e}")
        with _graph_job_lock:
            _graph_jobs[collection_name] = {
                "status": "error", "started_at": _graph_jobs.get(collection_name, {}).get("started_at"),
                "stats": None, "error": str(e),
            }
        _GRAPH_BUILD_FAIL_AT[collection_name] = time.time()


@router.post("/persona/graph/build")
async def build_character_graph(http_request: Request, request: GraphBuildRequest):
    """
    为某角色构建知识图谱（GraphRAG 增强层）。

    全流程异步：提交后立即可用 /persona/graph/status 轮询。
    图谱文件已存在且未 force 时直接返回既有统计，不重复构建。

    限流独立分桶（bucket=graph）：一次构建可能触发几百次 LLM 调用，
    是全站最重的操作，必须严限。
    """
    _check_rate(_client_ip(http_request), per_minute=2, per_day=10, bucket="graph")
    if not settings.graph_rag_enabled:
        raise HTTPException(status_code=400, detail="知识图谱 RAG 当前已关闭（graph_rag_enabled=false）")

    collection = _resolve_collection(request.character_id)
    col_name = collection.name

    from src.retrieval.knowledge_graph import graph_exists, load_graph
    if graph_exists(col_name) and not request.force:
        g = load_graph(col_name) or {}
        return {
            "collection": col_name,
            "status": "exists",
            "stats": g.get("stats"),
            "message": "图谱已存在，使用 force=true 可强制重建",
        }

    with _graph_job_lock:
        job = _graph_jobs.get(col_name)
        if job and job.get("status") == "building":
            return {"collection": col_name, "status": "building", "message": "已在构建中，请稍后轮询"}

    # 启动后台构建
    try:
        asyncio.create_task(_run_graph_build(col_name, force=request.force))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动构建任务失败: {str(e)}")

    return {"collection": col_name, "status": "building", "message": "已启动后台构建，请轮询 /persona/graph/status"}


@router.get("/persona/graph/status")
async def graph_build_status(character_id: str):
    """
    查询某角色知识图谱状态：是否已构建（exists）+ 后台任务进度（building/done/error）。
    """
    if character_id not in persona_chat_config.characters:
        raise HTTPException(status_code=404, detail=f"角色 '{character_id}' 不存在")
    col_name = persona_chat_config.characters[character_id].chroma_collection

    from src.retrieval.knowledge_graph import graph_exists, load_graph, graph_path
    exists = graph_exists(col_name)
    tmp_exists = graph_path(col_name).with_suffix(".json.tmp").exists()
    with _graph_job_lock:
        job = _graph_jobs.get(col_name, {})
    g = load_graph(col_name) or {}

    # 状态以最终文件为准，内存 job 仅表示“本进程内正在构建”
    if exists:
        status = job.get("status") if job.get("status") in ("done", "error") else "done"
    elif job.get("status") == "building":
        status = "building"
    elif tmp_exists:
        # 上次构建中途中断（成品未生成、仅留临时文件），可重跑续传
        status = "interrupted"
    else:
        status = "idle"

    return {
        "character_id": character_id,
        "collection": col_name,
        "exists": exists,
        "status": status,
        "stats": g.get("stats") if exists else (job.get("stats") if status == "done" else None),
        "error": job.get("error"),
    }


@router.post("/persona/characters/create")
async def create_character(http_request: Request, request: CharacterCreateRequest):
    """
    创建一个用户自建角色，并把它提交的人物设定 + 背景灌入独立 ChromaDB collection。

    流程：
    1. 校验名字；解析/生成角色 id（与内置角色命名空间隔离）
    2. 若开启网络搜索：收集公开资料
    3. 合成 role_prompt 与 background（用户提供 role_prompt 则保留之，只合并背景）
    4. 把 background 向量化写入 custom_<id> collection
    5. 落盘 <id>.json 并注册进场景配置，立即可对话

    限流独立分桶（bucket=create，建角是重操作）；总角色数有上限，防滥用撑爆磁盘/向量库。
    """
    _check_rate(_client_ip(http_request), per_minute=3, per_day=20, bucket="create")

    name = (request.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="请提供人物名字 name")
    if contains_sensitive(name) or contains_sensitive(request.background or "") or contains_sensitive(request.role_prompt or ""):
        raise HTTPException(status_code=400, detail="人物名字、背景或人设包含不允许的内容，请调整后再试")

    # 自建角色总量上限：无账号体系的阶段防"脚本批量造角"撑爆磁盘与向量库
    _custom_count = sum(1 for c in persona_chat_config.characters.values() if c.is_custom)
    if _custom_count >= settings.max_custom_characters:
        raise HTTPException(
            status_code=400,
            detail=f"自建角色已达上限（{settings.max_custom_characters} 个），请先删除一些再创建",
        )

    # 解析/生成 id
    if request.char_id and request.char_id.strip():
        cid = request.char_id.strip().lower()
        if not is_valid_custom_id(cid):
            raise HTTPException(
                status_code=400,
                detail="char_id 仅允许小写字母/数字/下划线，长度 2-40",
            )
    else:
        cid = generate_custom_id(name)

    if cid in persona_chat_config.characters:
        raise HTTPException(status_code=400, detail=f"角色 id '{cid}' 已存在，请换一个")

    provided_role = (request.role_prompt or "").strip() or None

    # 至少要有人设或背景，否则无法成角色
    if not provided_role and not request.use_web_search:
        if not (request.background or "").strip():
            raise HTTPException(
                status_code=400,
                detail="请至少提供「背景」或「角色人设 role_prompt」，或开启网络搜索自动生成",
            )

    try:
        role_prompt, background, sources, card = await build_persona(
            name=name,
            user_background=request.background or "",
            use_web_search=request.use_web_search,
            provided_role_prompt=provided_role,
        )
    except Exception as e:
        print(f"[Create] 构建人设失败: {e}")
        raise HTTPException(status_code=500, detail=f"构建人设失败: {str(e)}")

    if not role_prompt:
        raise HTTPException(status_code=400, detail="未能生成角色人设，请手动填写 role_prompt")

    if not background or not background.strip():
        background = name  # 最小背景兜底，保证检索库非空

    # 入库：背景写入独立 collection
    collection_name = collection_name_for(cid)
    try:
        n_chunks = await ingest_texts_async(collection_name, cid, background)
    except Exception as e:
        print(f"[Create] 背景入库失败: {e}")
        raise HTTPException(status_code=500, detail=f"背景入库失败: {str(e)}")

    # 组装 CharacterDef 并落盘 + 注册
    from scenes.persona_chat.models import CharacterDef
    char = CharacterDef(
        id=cid,
        name=name,
        description=(request.description or request.tagline or name),
        role_prompt=role_prompt,
        chroma_collection=collection_name,
        data_source="",  # 自建角色无文件目录，知识库来自提交的背景文本
        avatar=(request.avatar or "🎭").strip() or "🎭",
        tagline=request.tagline or "",
        ability=request.ability or "",
        review=request.review or "",
        theme=request.theme or "original",
        location=request.location,
        enable_verification=False,  # 娱乐区关引用核查（去 AI 味，不出现出处/置信度）
        is_custom=True,
        created_at=time.time(),
        # 自建角色默认归娱乐区：像人、轻检索、极短、关引用核查
        zone="entertainment",
        # 酒馆式角色卡：用户手填的优先，留空则用自动生成的结果补上。
        # 有示例对话/世界书，角色才能走"零检索 + 像本人"的路子。
        first_mes=(request.first_mes or "").strip() or card.get("first_mes", ""),
        mes_example=(request.mes_example or "").strip() or card.get("mes_example", ""),
        scenario=(request.scenario or "").strip() or card.get("scenario", ""),
        personality=(request.personality or "").strip() or card.get("personality", ""),
        lorebook=(request.lorebook or []) or card.get("lorebook", []),
    )
    try:
        save_custom_character(char)
    except Exception as e:
        print(f"[Create] 落盘失败（内存已注册）: {e}")
    persona_chat_config.characters[cid] = char

    print(f"[Create] 自建角色 '{name}' (id={cid}) 已创建，背景 {n_chunks} 块")
    return {
        "character": CharacterInfo(
            id=char.id,
            name=char.name,
            description=char.description,
            tagline=char.tagline,
            ability=char.ability,
            review=char.review,
            avatar=char.avatar,
            theme=char.theme,
            location=char.location,
            is_custom=True,
            zone=char.zone,
            first_mes=char.first_mes,
        ),
        "chunks": n_chunks,
        "web_search_used": bool(sources),
        "sources": sources,
    }


@router.delete("/persona/characters/{char_id}")
async def delete_character(char_id: str):
    """
    删除一个自建角色：从场景配置注销、删除落盘 JSON、清空其 ChromaDB collection。

    内置角色（is_custom=False）不可删除。
    """
    char = persona_chat_config.characters.get(char_id)
    if not char:
        raise HTTPException(status_code=404, detail=f"角色 '{char_id}' 不存在")
    if not char.is_custom:
        raise HTTPException(status_code=400, detail="内置角色不可删除")

    # 注销内存
    persona_chat_config.characters.pop(char_id, None)
    # 删除落盘
    delete_custom_character_file(char_id)
    # 清空向量库
    try:
        from framework.supervisor import get_chroma_client
        client = get_chroma_client()
        client.delete_collection(name=char.chroma_collection)
        from src.retrieval.advanced_search import invalidate_bm25_cache
        invalidate_bm25_cache(char.chroma_collection)
        # 一并失效该角色的知识图谱（若有）
        from src.retrieval.knowledge_graph import invalidate_graph
        invalidate_graph(char.chroma_collection)
    except Exception as e:
        print(f"[Delete] 删除 collection 失败（可忽略）: {e}")

    print(f"[Delete] 已删除自建角色: {char_id}")
    return {"deleted": True, "id": char_id}


@router.get("/persona/characters/{char_id}/export")
async def export_character_card(char_id: str):
    """
    导出角色卡（原生 JSON 格式）：人设 / 世界书 / 示例对话等软设定打包下载。

    - 自建角色：从其 collection 反查背景全文一并导出，导入方可完整重建知识库；
    - 内置角色：只导出软设定（background 留空——语料文件体积大且未必可再分发），
      导出的卡在别处导入后仍可人设驱动对话。
    """
    char = persona_chat_config.characters.get(char_id)
    if not char:
        raise HTTPException(status_code=404, detail=f"角色 '{char_id}' 不存在")

    background = ""
    if char.is_custom:
        try:
            from scenes.persona_chat.card_io import background_from_collection
            background = background_from_collection(get_chroma_client().get_collection(char.chroma_collection))
        except Exception as e:
            print(f"[CardExport] 反查背景失败（导出为空背景卡）: {e}")

    card = export_card(char, background=background)
    fname = f"{char_id}-card.json"
    return JSONResponse(
        content=card,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


class CardImportRequest(BaseModel):
    """角色卡导入请求：card=原生 JSON 卡片；png_base64=社区通用 PNG 角色卡（二选一）"""
    card: Optional[dict] = None
    png_base64: Optional[str] = None


@router.post("/persona/characters/import")
async def import_character_card(http_request: Request, request: CardImportRequest):
    """
    导入一张角色卡，创建为自建角色并立即可对话。

    - 带 background 的卡：背景切块入库，得到与导出方同等内容的知识库；
    - 不带 background 的卡：创建纯人设驱动角色（照常对话，仅无检索兜底）；
    - 支持 png_base64：社区通用 PNG 角色卡（格式互操作，仅处理用户主动导入的内容，
      不预装不分发任何第三方卡片）；
    - id 自动生成，与导入方隔离；同名不冲突。
    """
    _check_rate(_client_ip(http_request), per_minute=3, per_day=20, bucket="create")

    if request.png_base64:
        # 社区通用 PNG 角色卡（格式互操作：自行解析 PNG 文本块，仅处理用户主动导入的内容）
        import base64 as _b64
        from scenes.persona_chat.card_io import parse_png_character_card
        try:
            png_bytes = _b64.b64decode(request.png_base64, validate=False)
            native_card = parse_png_character_card(png_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PNG 解析失败：{e}")
        try:
            fields = parse_card(native_card)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"这张角色卡缺少必要内容：{e}")
    elif request.card:
        try:
            fields = parse_card(request.card)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="请提供角色卡（card 或 png_base64）")
    if contains_sensitive(fields["name"]) or contains_sensitive(fields["role_prompt"]) or contains_sensitive(fields["background"]):
        raise HTTPException(status_code=400, detail="角色卡内容包含不允许的内容，已拒绝导入")

    # 与"创建角色"共用同一个自建角色总量上限
    _custom_count = sum(1 for c in persona_chat_config.characters.values() if c.is_custom)
    if _custom_count >= settings.max_custom_characters:
        raise HTTPException(
            status_code=400,
            detail=f"自建角色已达上限（{settings.max_custom_characters} 个），请先删除一些再导入",
        )

    name = fields["name"]
    cid = generate_custom_id(name)
    while cid in persona_chat_config.characters:  # 极小概率撞 id，重生成一次
        cid = generate_custom_id(name)

    collection_name = collection_name_for(cid)
    n_chunks = 0
    if fields["background"]:
        try:
            n_chunks = await ingest_texts_async(collection_name, cid, fields["background"])
        except Exception as e:
            print(f"[CardImport] 背景入库失败（角色仍创建，无检索兜底）: {e}")

    from scenes.persona_chat.models import CharacterDef
    char = CharacterDef(
        id=cid,
        name=name,
        description=fields["description"],
        role_prompt=fields["role_prompt"] or fields["name"],
        chroma_collection=collection_name,
        data_source="",
        avatar=fields["avatar"],
        tagline=fields["tagline"],
        ability=fields["ability"],
        theme=fields["theme"],
        enable_verification=fields["enable_verification"],
        is_custom=True,
        created_at=time.time(),
        zone=fields["zone"],
        first_mes=fields["first_mes"],
        mes_example=fields["mes_example"],
        scenario=fields["scenario"],
        personality=fields["personality"],
        lorebook=fields["lorebook"],
    )
    try:
        save_custom_character(char)
    except Exception as e:
        print(f"[CardImport] 落盘失败（内存已注册）: {e}")
    persona_chat_config.characters[cid] = char

    print(f"[CardImport] 已导入角色卡 '{name}' (id={cid})，背景 {n_chunks} 块")
    return {
        "character": CharacterInfo(
            id=char.id,
            name=char.name,
            description=char.description,
            tagline=char.tagline,
            ability=char.ability,
            review=char.review,
            avatar=char.avatar,
            theme=char.theme,
            location=char.location,
            is_custom=True,
            zone=char.zone,
            first_mes=char.first_mes,
        ),
        "chunks": n_chunks,
        "lorebook_entries": len(fields["lorebook"]),
    }


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
    # 限流主体：登录用户按账号（共享出口 IP 不再互相误伤、换 IP 无法绕过），匿名按 IP
    _user = get_current_user(http_request)
    _rate_subject = f"user:{_user['id']}" if _user else client_ip
    logger.info(
        "start ip=%s user=%s char=%s q=%r",
        client_ip, _user["account"] if _user else "-", request.character_id or "jung", request.query[:60],
    )
    t0 = time.time()
    _check_rate(_rate_subject, per_minute=settings.rate_limit_per_minute, per_day=settings.rate_limit_per_day)
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

    # 历史编辑：前端已删掉被编辑轮次之后的内容，后端存储同步截断，
    # 保证两边的上下文一致（后端历史会作为 prompt 注入，必须与用户所见对齐）
    if request.truncate_to_turns is not None:
        truncate_conversation_history(session_id, int(request.truncate_to_turns))

    # 重新生成：旧答案作废——弹出最后一轮 (query, old_answer)，重答完成后写回。
    # expect_query 校验弹掉的确实是本轮要重答的问题，防止并发时误删其他轮次。
    if request.regenerate:
        pop_last_turn(session_id, expect_query=request.query)

    # 获取角色定义
    character = persona_chat_config.characters.get(character_id)
    if not character:
        raise HTTPException(
            status_code=400,
            detail=f"角色 '{character_id}' 不存在。可用角色: {list(persona_chat_config.characters.keys())}",
        )

    # 知识图谱常驻：角色首次被对话且图谱缺失时仍会自动后台构建（用户无需点按钮），
    # 但触发时机已从「请求开始」推迟到「回答返回之后」（见 event_stream 末尾），
    # 避免后台构建的成百次 LLM 调用与本次回答争夺 DeepSeek API / CPU，拉高回答延迟。
    # 图谱就绪后由 AI 自主判断何时调用（should_trigger_graph_retrieval）。

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
        stages_begin()  # 阶段耗时收集：检索/重排/首字/核查 → monitor 延迟瀑布
        token_queue: asyncio.Queue = asyncio.Queue()
        graph_error: list[Exception] = []
        # 本请求独立的结果容器：即使两个请求共享同一 session_id（如同一浏览器
        # 两个标签页），也绝不从共享的 _session_store 读最终答案，
        # 避免并发时互相覆盖导致串扰/丢回答
        result_box: dict = {"final_answer": "", "route_history": [], "info_gap_questions": None}
        history_before = get_conversation_history(session_id)
        # 缓存键必须带用户维度：答案里会注入该用户的长期记忆块（见下方 _memory_block），
        # 若只按（角色, 问题）索引，用户 A 的答案会被用户 B 原样命中。
        # 匿名用户不建记忆（下方注释），统一归入 anon 段，可安全共享。
        _user_seg = f"u:{_user['id']}" if _user else "anon"
        cache_key = f"{_user_seg}::{character_id}::{request.query.strip()}"
        was_cache_hit = False

        _first_token_at: list[float] = []

        async def stream_callback(token: str):
            # 首字延迟：只记第一次 token 到达的时刻
            if not _first_token_at:
                _first_token_at.append(time.time())
                stage_mark("first_token_ms", (_first_token_at[0] - t0) * 1000)
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

                # 酒馆式装配：人设 + 性格 + 场景 + 世界书命中词条 + 示例对话（+ 娱乐区语气指令）
                # 世界书扫描范围 = 本轮消息 + 最近几轮对话（接住"我们刚才聊的那个XX"类指代）
                _history_now = get_conversation_history(session_id)
                _scan_text = compose_lorebook_scan_text(request.query, _history_now)
                _role_prompt = build_character_prompt(
                    character, query=request.query, scan_text=_scan_text
                )

                # 检索策略：娱乐区走轻量召回（top_k=3，无改写 / 无重排 / 无图谱）。
                # 实测中位仅 23ms，相比 8-15s 的生成可忽略；换回的是"贴合角色背景资料"。
                # 真正贵的是重排（实测 10.4s）与检索改写（1 次 LLM 往返），两者娱乐区都不走。
                # 置 settings.entertainment_light_retrieval = False 可回到零检索。
                _is_ent = character.zone == "entertainment"
                _light = _is_ent and settings.entertainment_light_retrieval
                # 娱乐区但无示例对话/世界书：只能靠背景检索撑住，日志留痕便于发现"弱角色卡"
                if _is_ent and not has_tavern_components(character):
                    logger.info("ent_char_no_card char=%s（缺示例对话/世界书）", character_id)

                # 用户长期记忆：登录用户跨会话记住背景/偏好/关注主题（匿名不建）
                _user_key = f"u:{_user['id']}" if _user else None
                _memory_block = ""
                if _user_key and settings.memory_enabled:
                    try:
                        from src.core.memory import retrieve_memory_block
                        _memory_block = await retrieve_memory_block(_user_key, request.query)
                    except Exception as _me:
                        print(f"[Memory] 记忆检索失败（忽略）: {_me}")

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
                    "character_role_prompt": _role_prompt,
                    "enable_verification": character.enable_verification,
                    # 后历史指令：历史之后再钉一次角色，防长对话漂移
                    "post_history_directive": build_post_history_directive(
                        character, query=request.query, scan_text=_scan_text
                    ),
                    # 按角色的采样覆盖，未配置则 None（沿用默认）
                    "sampling": _sampling_for(character),
                    "zone": character.zone,
                    "light_retrieval": _light,
                    "skip_retrieval": _is_ent and not _light,
                    "user_memory": _memory_block,
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
                        "analysis": accumulated.get("analysis", ""),
                        "retrieved_docs": accumulated.get("retrieved_docs", []),
                        "info_gap_questions": info_gap_qs,
                    })
                    # 用户长期记忆提取：回答返回后异步执行（不占用户等待时间）
                    if _user_key and settings.memory_enabled:
                        from src.core.memory import extract_and_store as _mem_extract
                        _mt = asyncio.create_task(
                            _mem_extract(_user_key, request.query, answer)
                        )
                        _MEMORY_TASKS.add(_mt)
                        _mt.add_done_callback(_MEMORY_TASKS.discard)
                    # _session_store 仍同步更新（供刷新后恢复答案），
                    # 但 SSE 最终结果只读 result_box，与并发请求互不干扰
                    # query 同步更新：前端刷新后可用它校验“答案属于哪个问题”，
                    # 防止恢复机制把另一个标签页的回答错贴到当前问题下
                    _session_store[session_id].update({**result_box, "query": request.query})
                    _persist_session_db(session_id)
                result_box["graph_used"] = bool(accumulated.get("graph_used", False))
                if not history_before and not request.regenerate:
                    _ANSWER_CACHE[cache_key] = (
                        time.time() + _ANSWER_CACHE_TTL,
                        answer,
                        accumulated.get("route_history", []),
                    )
                    # 过期清理：字典只进不出是慢泄漏（长期运行每条 (角色,问题) 占一条），
                    # 超过软上限时先清过期项，清完仍超则按插入序淘汰最旧的
                    if len(_ANSWER_CACHE) > 512:
                        now = time.time()
                        for k in [k for k, v in _ANSWER_CACHE.items() if v[0] < now]:
                            _ANSWER_CACHE.pop(k, None)
                        while len(_ANSWER_CACHE) > 512:
                            _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)), None)
                    print(f"[Persona Query] 已缓存答案: {cache_key}")
            except GraphRecursionError:
                print("[Persona Query] 图执行超出递归限制")
                fallback = await _generate_direct_response(
                    request.query, get_conversation_history(session_id),
                    build_character_prompt(character, query=request.query), stream_callback,
                    max_tokens=300 if character.zone == "entertainment" else None,
                    zone=character.zone,
                    post_history_directive=build_post_history_directive(
                        character, query=request.query
                    ),
                    sampling=_sampling_for(character),
                )
                append_conversation(session_id, request.query, fallback)
                result_box["final_answer"] = fallback
                result_box["graph_used"] = False
                _session_store[session_id].update({"final_answer": fallback, "query": request.query})
                _persist_session_db(session_id)
            except Exception as e:
                graph_error.append(e)
            finally:
                await token_queue.put(None)

        try:
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'{character.name}正在思考...'})}\n\n"

            # 新会话首问命中缓存：直接回放，跳过整个图执行。
            # 重新生成时不读缓存——重答的意义就是给一个不同的回答。
            cached = _ANSWER_CACHE.get(cache_key)
            if not history_before and not request.regenerate and cached and cached[0] > time.time():
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
            graph_used = bool(result_box.get("graph_used", False))

            # 引用标注映射：正文 [n] 角标 → 资料（来源/章节），前端渲染悬浮出处。
            # 编号与 _build_context 的 [1]..[n] 清单顺序一致
            _doc_map = [
                {
                    "n": i + 1,
                    "source": (d.metadata or {}).get("source", ""),
                    "heading": (d.metadata or {}).get("heading", ""),
                }
                for i, d in enumerate(result_box.get("retrieved_docs", []) or [])
            ]

            yield f"data: {json.dumps({'type': 'trace', 'route_history': route_history})}\n\n"

            if info_gap_qs and final_answer:
                yield f"data: {json.dumps({'type': 'questions', 'content': final_answer, 'questions': info_gap_qs, 'graph_used': graph_used, 'doc_map': _doc_map})}\n\n"
            elif final_answer:
                yield f"data: {json.dumps({'type': 'result', 'content': final_answer, 'graph_used': graph_used, 'doc_map': _doc_map})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'result', 'content': '查询完成'})}\n\n"

            # ---- 异步引用核查：回答已流式推给用户，这里再补跑 verifier 并推送引用出处 ----
            # verifier 含一次串行 LLM 调用（论断>3 条时），移到回答定稿之后执行，不阻塞用户看到答案。
            # 但 end 事件在核查之后才发：核查不完，前端"完成"状态就一直挂着。
            # 给核查设等待预算，超时即放弃本次引用出处推送，按时收尾。
            if character.enable_verification and final_answer:
                try:
                    from framework.verification_agent import verification_agent
                    from framework.supervisor import _citation_block
                    _v_docs = result_box.get("retrieved_docs", []) or []
                    _v_analysis = result_box.get("analysis", "") or final_answer
                    if _v_docs:
                        _v_t0 = time.time()
                        _v_task = asyncio.create_task(verification_agent({
                            "analysis": _v_analysis,
                            "retrieved_docs": _v_docs,
                            "query": request.query,
                            "route_history": [],
                        }))
                        _v_out = await asyncio.wait_for(
                            _v_task, timeout=getattr(settings, "verify_deadline_sec", 15.0)
                        )
                        stage_mark("verify_ms", (time.time() - _v_t0) * 1000)
                        _v_report = _v_out.get("verification", "")
                        _v_cite = _citation_block(_v_report) if _v_report else ""
                        if _v_cite:
                            yield f"data: {json.dumps({'type': 'citations', 'content': _v_cite, 'graph_used': graph_used})}\n\n"
                except asyncio.TimeoutError:
                    logger.warning(
                        "verify_timeout char=%s deadline=%.0fs（放弃本次引用出处推送）",
                        character_id, getattr(settings, "verify_deadline_sec", 15.0),
                    )
                except Exception as _ve:
                    print(f"[Persona Query] 异步引用核查失败（忽略）: {_ve}")

            # ---- 知识图谱自动构建：推迟到回答返回之后再触发 ----
            # 避免后台构建的成百次 LLM 调用与本次回答争夺 DeepSeek API / CPU（见上方注释）。
            # 失败退避：构建失败（如上游空响应熔断）后 1 小时内不自动重试，
            # 防止每次对话都重新触发一次注定失败的几百次 LLM 调用拖垮服务。
            if settings.graph_rag_enabled and settings.graph_auto_build and character.zone == "education":
                try:
                    from src.retrieval.knowledge_graph import graph_exists as _ge
                    _col = character.chroma_collection
                    _fail_at = _GRAPH_BUILD_FAIL_AT.get(_col, 0)
                    _backoff_ok = (time.time() - _fail_at) > 3600
                    if not _ge(_col) and _backoff_ok:
                        with _graph_job_lock:
                            _jb = _graph_jobs.get(_col)
                            _building = _jb and _jb.get("status") == "building"
                        if not _building:
                            asyncio.create_task(_run_graph_build(_col, force=False))
                            logger.info("auto_build_graph char=%s col=%s", character_id, _col)
                except Exception as _ae:
                    print(f"[GraphAutoBuild] 自动构建触发失败（忽略）: {_ae}")

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
                    stages=stages_snapshot(),
                )
            except Exception as e:
                print(f"[Monitor] 用量记录失败: {e}")
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
                    stages=stages_snapshot(),
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


@router.post("/persona/roundtable")
async def persona_roundtable_endpoint(http_request: Request, request: "RoundtableRequest"):
    """
    圆桌会议端点（流式输出）：将多名名人拉入同一议题，按回合轮流发言、相互交锋。

    请求体：{ topic, character_ids:[...], rounds, session_id? }
    返回 SSE 事件流：start / round / speaker_start / token / speaker_end / end / error。
    """
    rid = uuid.uuid4().hex[:12]
    set_request_id(rid)
    logger = get_logger("persona.roundtable")
    client_ip = _client_ip(http_request)
    topic = (request.topic or "").strip()
    character_ids = [str(c).strip() for c in (request.character_ids or []) if str(c).strip()]
    rounds = int(request.rounds or 1)
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("start ip=%s topic=%r speakers=%s rounds=%d", client_ip, topic[:60], character_ids, rounds)

    if not topic:
        raise HTTPException(status_code=400, detail="请提供圆桌会议的议题 topic")
    if len(character_ids) < 2:
        raise HTTPException(status_code=400, detail="圆桌会议至少需要选择 2 位对话者")
    if len(character_ids) > MAX_ROUNDTABLE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"圆桌会议最多支持 {MAX_ROUNDTABLE_CHARS} 位对话者（人太多会削弱交锋感）",
        )
    # 只保留确实存在的角色，并去重
    valid_ids = []
    seen = set()
    for cid in character_ids:
        if cid in seen:
            continue
        if cid in persona_chat_config.characters:
            seen.add(cid)
            valid_ids.append(cid)
    if len(valid_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"有效的对话者不足 2 位。可用角色: {list(persona_chat_config.characters.keys())}",
        )

    _check_rate(
        (lambda u: f"user:{u['id']}" if u else client_ip)(get_current_user(http_request)),
        per_minute=settings.rate_limit_per_minute, per_day=settings.rate_limit_per_day,
    )

    # 输入侧内容安全：命中敏感词直接礼貌拒绝，不进入流水线
    if contains_sensitive(topic):
        logger.info("blocked_sensitive ip=%s topic=%r", client_ip, topic[:60])

        async def _refusal_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': refusal_message()})}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

        return StreamingResponse(
            _refusal_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    t0 = time.time()

    async def event_stream():
        speakers_meta: list = []
        transcript: list = []
        cur = None
        try:
            async for evt in stream_roundtable(topic, valid_ids, rounds, session_id):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
                et = evt.get("type")
                if et == "start":
                    speakers_meta = evt.get("speakers", []) or []
                elif et == "speaker_start":
                    cur = {
                        "character_id": evt.get("character_id"),
                        "name": evt.get("name", ""),
                        "content": "",
                    }
                elif et == "speaker_end" and cur is not None:
                    cur["content"] = evt.get("content", "")
                    transcript.append(cur)
                    cur = None
            # 圆桌会话落库：完整流式结束后写入 SQLite，可历史回看
            try:
                _get_session_store().save_roundtable(
                    session_id=session_id,
                    topic=topic,
                    speakers=speakers_meta,
                    rounds=rounds,
                    transcript=transcript,
                )
            except Exception:
                logger.exception("roundtable_save_failed session=%s", session_id)
            logger.info(
                "done ip=%s elapsed_ms=%d speakers=%d rounds=%d transcript_len=%d",
                client_ip, int((time.time() - t0) * 1000), len(valid_ids), rounds, len(transcript),
            )
        except Exception as e:
            logger.exception("roundtable_error ip=%s", client_ip)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 圆桌会议会话管理（历史 / 查看 / 删除）
# ============================================================
@router.get("/persona/roundtable/history")
async def roundtable_history_endpoint():
    """圆桌会议历史（最近 30 天，按时间倒序；不含完整发言，查看详情走 GET /persona/roundtable/{session_id}）"""
    items = _get_session_store().list_roundtables(ttl_days=30)
    return {"items": items}


@router.get("/persona/roundtable/{session_id}")
async def roundtable_get_endpoint(session_id: str):
    """按 session_id 查看一场圆桌会议（议题、与会者、完整发言记录）"""
    data = _get_session_store().get_roundtable(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="圆桌会议记录不存在")
    return data


@router.delete("/persona/roundtable/{session_id}")
async def roundtable_delete_endpoint(session_id: str):
    """删除一场圆桌会议记录"""
    _get_session_store().delete_roundtable(session_id)
    return {"ok": True, "session_id": session_id}


class RoundtableRequest(BaseModel):
    """圆桌会议请求"""
    topic: str
    character_ids: list[str]
    rounds: int = 1
    session_id: Optional[str] = None


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

        print(f"[Persona Upload] 文档已存入 {character.chroma_collection}: {len(chunks)} 个向量")

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            status="completed",
        )

    except Exception as e:
        print(f"[Persona Upload] 文档处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {str(e)}",
        )
