"""请求级依赖：客户端标识、限流、登录态、管理端鉴权、采样覆盖。

这些函数不含业务逻辑，只回答「这个请求是谁、还能不能用、该用什么采样参数」，
所以从 routes.py 拆出来单独承载——改限流策略时不必翻 2600 行的端点文件。

全部是同步函数，可在任意端点里直接调用；限流状态放在模块级 `_RATE_STORE`，
进程内有效（多进程部署时每进程独立计数，需要全局配额要换 Redis）。
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import HTTPException, Request, Response

from src.core import accounts
from src.core.config import settings

# 限流存储：ip -> {"min": 窗口起点, "min_count": 分钟计数, "day": 窗口起点, "day_count": 天计数}
_RATE_STORE: dict[str, dict] = {}


def client_ip(request: Request) -> str:
    """获取客户端 IP（部署在 nginx 后需启用 uvicorn --proxy-headers 信任 X-Forwarded-For）"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate(ip: str, per_minute: int = 0, per_day: int = 0, bucket: str = "main") -> None:
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


def current_session_token(request: Request) -> str:
    return request.cookies.get(settings.auth_cookie_name, "")


def get_current_user(request: Request) -> Optional[dict]:
    """从 Cookie 解析当前登录用户；未登录返回 None（不抛错）。"""
    try:
        return accounts.get_user_by_session(current_session_token(request))
    except Exception as e:
        print(f"[Auth] 会话解析失败（按未登录处理）: {e}")
        return None


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=accounts.SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=settings.auth_cookie_secure,
        path="/",
    )


def require_admin(token: str) -> None:
    """管理端读取接口鉴权：设置 monitor_token 后必须带 ?token=xxx"""
    if settings.monitor_token and token != settings.monitor_token:
        raise HTTPException(status_code=403, detail="token 无效")


def sampling_for(character) -> Optional[dict]:
    """挑出角色卡上的采样覆盖（temperature / top_p / frequency_penalty / presence_penalty）。

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
