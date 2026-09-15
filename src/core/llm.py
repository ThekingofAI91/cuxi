"""
llm.py — ChatOpenAI 统一工厂

所有 LLM 调用统一走 get_chat_llm()，保证超时、重试、模型参数一致：
- request_timeout：单次请求超时，防止上游 API 挂起时 SSE 流永久卡住
- max_retries：网络抖动/5xx 自动重试
- 备用模型：配置 llm_fallback_model 后，主模型报错（额度用完/不可用等）
  时自动切换到备用模型（with_fallbacks，对调用方透明）
- 连接池复用：异步调用共享按事件循环隔离的 httpx.AsyncClient，
  省掉每次调用重付 TCP+TLS 握手（见 _shared_async_http_client）
- 具体场景的 temperature / max_tokens 仍由调用方通过 kwargs 覆盖
"""

import asyncio
import threading
from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.core.runtime_config import resolve as resolve_llm_config


class LLMNotConfiguredError(RuntimeError):
    """尚未配置 LLM API Key。

    开源版把这个 Key 交给使用者自己填（浏览器设置页或 .env）。
    这里抛业务异常而不是让它变成 401/网络错误，是为了让上层能识别出
    「配置缺失」这一种情况并给出人话提示，而不是让用户对着一串 traceback 猜。
    """


# ============================================================
# 共享 HTTP 连接池（按事件循环隔离）
# ============================================================
# 默认每个 ChatOpenAI 实例自建一个 httpx.AsyncClient，而本项目每次请求都会
# get_chat_llm() 新建实例 → 每次调用都是全新 TCP+TLS 连接，走第三方中转时
# 每次多付 0.1-0.5s 握手。共享连接池后，同一事件循环内的所有 LLM 调用复用
# 已建立的连接。
#
# 必须按事件循环隔离：httpx 连接池里的连接绑定创建时的 event loop，
# 跨 loop 复用（如测试里多次 asyncio.run）会拿到已死亡的连接报错。
# 单次请求超时不受影响——openai SDK 层按 request_timeout 对每个请求单独计时。
# 环境代理（HTTP_PROXY 等）仍按 httpx 默认 trust_env 行为生效。

_async_http_clients: dict[int, Any] = {}
_client_pool_lock = threading.Lock()


def _shared_async_http_client():
    """返回当前事件循环专属的共享 httpx.AsyncClient；不可用时返回 None（走默认行为）"""
    if not getattr(settings, "llm_shared_http_client", True):
        return None
    try:
        import httpx

        loop = asyncio.get_running_loop()
        key = id(loop)
        with _client_pool_lock:
            client = _async_http_clients.get(key)
            if client is None:
                # 事件循环销毁后 id 可能被复用，池子超过 4 个直接清空重建
                #（生产 uvicorn 单循环只会有 1 个条目）
                if len(_async_http_clients) > 4:
                    _async_http_clients.clear()
                client = httpx.AsyncClient(
                    # 兜底超时：正常情况下 openai SDK 按 request_timeout 计时，
                    # 这里只是防止裸连接池永不超时
                    timeout=httpx.Timeout(120.0, connect=15.0),
                    # 高并发下连接池放开：LLM API 的并发受业务限流约束，不会打爆上游
                    limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
                )
                _async_http_clients[key] = client
            return client
    except RuntimeError:
        # 无运行中的事件循环（同步上下文）：同步路径本就有独立的 client，不需要共享
        return None
    except Exception as e:
        print(f"[LLM] 共享连接池创建失败（回退默认行为）: {e}")
        return None


def get_chat_llm(**kwargs: Any) -> Runnable:
    """创建 ChatOpenAI 实例，默认注入超时/重试/模型/密钥配置。

    密钥与地址每次调用时从「用户填写 > .env > 默认值」解析，所以用户在设置页
    保存后无需重启服务即可生效。

    配置了 fallback_model 且调用方未显式指定 model 时，
    返回 主模型.with_fallbacks([备用模型])，主模型调用失败自动降级。
    """
    cfg = resolve_llm_config(settings)

    kwargs.setdefault("api_key", cfg["api_key"])
    kwargs.setdefault("base_url", cfg["base_url"])
    kwargs.setdefault("request_timeout", settings.llm_request_timeout)
    kwargs.setdefault("max_retries", settings.llm_max_retries)

    # 调用方显式传了 api_key（如测试注入）就尊重调用方，此时才允许"未配置"放行
    if not kwargs.get("api_key"):
        raise LLMNotConfiguredError(
            "尚未配置 LLM API Key：请打开页面右上角「设置」填写自己的 API Key，"
            "或在项目根目录的 .env 中设置 LLM_API_KEY。"
        )

    # 调用方显式传了 http_async_client 就尊重调用方（如测试注入 mock client）
    if "http_async_client" not in kwargs:
        shared = _shared_async_http_client()
        if shared is not None:
            kwargs["http_async_client"] = shared

    explicit_model = "model" in kwargs
    kwargs.setdefault("model", cfg["model"])
    llm = ChatOpenAI(**kwargs)

    # 显式指定 model 的场景（如特殊用途模型）不套 fallback，避免行为被意外替换
    if explicit_model or not cfg["fallback_model"]:
        return llm

    fallback = ChatOpenAI(**{**kwargs, "model": cfg["fallback_model"]})
    return llm.with_fallbacks([fallback])


# ============================================================
# 空响应容错（2026-08-29 新增）
# ============================================================
# 实测（deepseek-v4-flash 经第三方中转）：上游存在随机性故障——HTTP 200 但
# content 为空（约 13s 超时后返回空壳），发生率随时段波动，同一 prompt 前一分钟
# 成功、后一分钟全空。空响应不抛异常，ChatOpenAI 内置重试（针对网络错误/5xx）
# 不会触发。危害：图谱构建每批 0 实体白烧调用；主链路分析拿到空回答 → 走
# "分析为空→降级直答"兜底，用户侧表现为延迟翻倍（13s 超时 + 8s 再生成）。
#
# 对策：关键调用点用 ainvoke_nonempty / astream_nonempty 包一层——
# 空响应视为失败立即重试（再摇一次骰子，上游随机性下重试命中率很高）；
# 流式版本仅在"一个 token 都没收到"时重试（此时尚未向用户推送任何内容，安全）。

_EMPTY_RETRY_ATTEMPTS = 2


def _is_empty_response(resp: Any) -> bool:
    content = getattr(resp, "content", None)
    return not (content and str(content).strip())


async def ainvoke_nonempty(
    llm: Runnable,
    messages: Any,
    attempts: int = _EMPTY_RETRY_ATTEMPTS,
    fallback_llm: Runnable = None,
):
    """ainvoke + 空响应重试。上游返回空壳（200 但无内容）时视为失败重试。

    fallback_llm：主模型 attempts 轮全空后切换备用模型再试一轮
    （实测中转故障时两个模型可能同时故障，但时段不同表现不同，值得一试）。
    """
    resp = None
    for i in range(attempts + 1):
        resp = await llm.ainvoke(messages)
        if not _is_empty_response(resp):
            return resp
        print(f"[LLM] 上游返回空响应（第 {i + 1} 次），重试…")
    if fallback_llm is not None:
        print("[LLM] 主模型持续空响应，切换备用模型重试")
        for i in range(attempts + 1):
            resp = await fallback_llm.ainvoke(messages)
            if not _is_empty_response(resp):
                return resp
            print(f"[LLM] 备用模型返回空响应（第 {i + 1} 次），重试…")
    return resp  # 全空则原样返回，由调用方按空内容走既有兜底


async def astream_nonempty(llm: Runnable, messages: Any, attempts: int = _EMPTY_RETRY_ATTEMPTS):
    """
    astream + 空响应重试：逐 token 产出；若整次流一个 token 都没产出
    （上游空壳响应），视为失败重试。已产出部分内容后失败不重试（避免重复推送）。
    """
    for i in range(attempts + 1):
        got = ""
        try:
            async for chunk in llm.astream(messages):
                token = getattr(chunk, "content", None) or ""
                if token:
                    got += token
                    yield token
            if got:
                return
            # 整个流空：未向调用方推送任何内容，安全重试
            print(f"[LLM] 上游返回空流（第 {i + 1} 次），重试…")
        except Exception:
            if got:
                raise  # 已推送部分内容，交由调用方既有逻辑处理
            if i >= attempts:
                raise
            print(f"[LLM] 流式调用异常且无输出（第 {i + 1} 次），重试…")
