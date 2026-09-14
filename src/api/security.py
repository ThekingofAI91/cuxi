"""上线网络安全防护（纯 ASGI 中间件，不读 body、不缓冲，对 SSE 流式透明）。

三件事：
1. 安全响应头：CSP / X-Content-Type-Options / X-Frame-Options / Referrer-Policy /
   Permissions-Policy / COOP / HSTS（可选）。此前项目安全响应头为零。
2. 全局限流兜底：业务侧只在 11 处路由点了 _check_rate（共 32 条路由），
   其余接口裸奔。这里给所有非静态请求兜一个宽松上限，防扫站与刷接口。
3. 请求体大小限制：超大 Content-Length 直接 413，不进业务。

为什么不用 BaseHTTPMiddleware：它会重新包装请求/响应流，对 SSE 流式返回
有缓冲与中断的已知风险；纯 ASGI 只包一层 send，零副作用。
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# 不做全局限流的路径：健康检查、静态资源、首页（前端同源于本服务）
_STATIC_PREFIXES = ("/static", "/assets", "/favicon", "/privacy", "/terms")
_STATIC_SUFFIXES = (".css", ".js", ".jpg", ".jpeg", ".png", ".svg", ".ico", ".webp", ".woff", ".woff2")


def _client_ip(scope: Scope) -> str:
    """取客户端 IP。

    部署在 nginx 后需 uvicorn --proxy-headers 才会信任 X-Forwarded-For，
    否则拿到的是反代地址（所有人共用一个桶，限流失效）。
    """
    headers = scope.get("headers") or []
    for k, v in headers:
        if k.lower() == b"x-forwarded-for":
            first = v.decode("latin-1").split(",")[0].strip()
            if first:
                return first
    client = scope.get("client")
    return client[0] if client else "unknown"


class SecurityHeadersMiddleware:
    """给所有 HTTP 响应加安全头。放在中间件最外层，连 429 / CORS 错误也带得上。"""

    def __init__(self, app: ASGIApp, *, enable_hsts: bool = False, server_name: str = "gkrm") -> None:
        self.app = app
        self.enable_hsts = enable_hsts
        self.server_name = server_name
        # 前端无内联脚本、无外部 CDN、无 eval，因此 CSP 可以收紧；
        # style-src 保留 unsafe-inline：存在少量 style="" 属性与 markdown 渲染出的内联样式。
        # 注意：MutableHeaders 的键必须是 str（内部会 .lower().encode()），
        # 传 bytes 会在 __contains__ 里炸 AttributeError。
        self.base_headers: list[tuple[str, str]] = [
            ("x-content-type-options", "nosniff"),
            ("x-frame-options", "SAMEORIGIN"),
            ("referrer-policy", "strict-origin-when-cross-origin"),
            ("permissions-policy", "geolocation=(), microphone=(), camera=()"),
            ("cross-origin-opener-policy", "same-origin"),
            ("content-security-policy", (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob: https:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "media-src 'self'; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )),
        ]
        if enable_hsts:
            # 仅 HTTPS 生产环境开启：HTTP 下发了会被浏览器忽略，
            # 但配了 HTTPS 再下发能让浏览器后续强制走 HTTPS。
            self.base_headers.append(
                ("strict-transport-security", "max-age=31536000; includeSubDomains")
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in self.base_headers:
                    if k not in headers:
                        headers[k] = v
                # 抹掉 uvicorn 默认 Server 头，减少版本指纹泄露
                if "server" in headers:
                    headers["server"] = self.server_name
            await send(message)

        await self.app(scope, receive, send_wrapper)


class GlobalRateLimitMiddleware:
    """全局限流兜底（滑动窗口，按 IP）。

    业务侧已经有更精细的分桶限流（聊天/建角/图谱/反馈各自计数），
    这里只兜住"没被业务限流覆盖到的接口"，阈值给得宽松，避免误伤正常用户。
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        per_minute: int = 120,
        exclude_static: bool = True,
        max_keys: int = 10000,
    ) -> None:
        self.app = app
        self.per_minute = per_minute
        self.exclude_static = exclude_static
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _is_exempt(self, path: str, method: str) -> bool:
        if method == "OPTIONS":  # CORS 预检不计入
            return True
        if path in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            return True
        if self.exclude_static:
            if path.startswith(_STATIC_PREFIXES):
                return True
            if path.endswith(_STATIC_SUFFIXES):
                return True
        return False

    def _allow(self, key: str) -> bool:
        now = time.time()
        window = self._hits[key]
        # 滑出 60 秒窗口的旧记录
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)

        # 粗粒度防内存膨胀：键数超阈值时清掉空桶
        if len(self._hits) > self.max_keys:
            for k in [k for k, v in self._hits.items() if not v]:
                self._hits.pop(k, None)
            if len(self._hits) > self.max_keys:  # 仍然超限就整体清空，宁可误放也不 OOM
                self._hits.clear()
        return True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.per_minute <= 0:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""
        method = (scope.get("method") or "GET").upper()
        if self._is_exempt(path, method):
            await self.app(scope, receive, send)
            return

        ip = _client_ip(scope)
        if not self._allow(ip):
            logger.warning("全局限流触发 ip=%s path=%s method=%s", ip, path, method)
            resp = JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
                headers={"Retry-After": "60"},
            )
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)


class RequestSizeLimitMiddleware:
    """按 Content-Length 拦截超大请求体，避免超大 payload 打进业务层。

    只看头不读 body，因此对流式上传/下载无影响。
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = 8 * 1024 * 1024) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return
        for k, v in scope.get("headers") or []:
            if k.lower() == b"content-length":
                try:
                    if int(v) > self.max_bytes:
                        logger.warning(
                            "请求体过大被拒 path=%s size=%s limit=%s",
                            scope.get("path"), v.decode("latin-1"), self.max_bytes,
                        )
                        resp = JSONResponse(
                            status_code=413,
                            content={"detail": "请求体过大"},
                        )
                        await resp(scope, receive, send)
                        return
                except ValueError:
                    break
        await self.app(scope, receive, send)
