"""
setup.py — 设置页接口（开源版：用户自己填 LLM API Key）

三个接口：
  GET  /setup/status  是否已配置、当前生效的 base_url / model、脱敏后的 Key
  POST /setup/llm     保存用户填写的配置（落盘 data/user_config.json）
  POST /setup/test    用填写的配置真实发一次最小请求，验证 Key / 地址 / 模型通不通

安全策略（重要）：
  首次配置（尚未填 Key）永远放行——否则部署到服务器后，没法从浏览器完成初始化。
  已完成配置后再改动，只允许「本机请求」或「持有 MONITOR_TOKEN 的请求」，
  避免服务暴露在公网时被人恶意替换 Key（换 Key 等于把调用成本转嫁给别人）。
"""

from __future__ import annotations

import secrets
import time
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.core import runtime_config
from src.core.config import settings

router = APIRouter(tags=["setup"])

# uvicorn 单机部署时客户端地址；testclient 是 FastAPI 测试客户端的 host
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}

# 国内用户最常用的两家；填表时点一下就带出来，省得去翻文档
PRESETS = [
    {"label": "DeepSeek 官方", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    {"label": "阿里云百炼（通义）", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    {"label": "智谱 GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    {"label": "月之暗面 Kimi", "base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
]


class LLMConfigIn(BaseModel):
    """设置页提交体。空串 = 不修改该字段（已有值保留）。"""

    base_url: str = Field(default="", max_length=300)
    api_key: str = Field(default="", max_length=300)
    model: str = Field(default="", max_length=120)
    fallback_model: str = Field(default="", max_length=120)


def _is_local(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    return host in _LOCAL_HOSTS


def _guard_write(request: Request) -> None:
    """已配置完成的服务，只允许本机或持管理员令牌的请求改配置。"""
    if not runtime_config.is_configured(settings):
        return  # 首次配置：放行

    token = (
        request.headers.get("X-Admin-Token")
        or request.query_params.get("token")
        or ""
    ).strip()
    if settings.monitor_token and token and secrets.compare_digest(token, settings.monitor_token):
        return
    if _is_local(request):
        return

    raise HTTPException(
        status_code=403,
        detail="服务已完成配置。如需修改，请在服务器本机操作，或携带管理员令牌。",
    )


def _status_payload() -> dict:
    cfg = runtime_config.resolve(settings)
    return {
        "configured": bool(cfg["api_key"]),
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "fallback_model": cfg["fallback_model"],
        # 绝不回传明文 Key：只给脱敏串，让用户确认「填的是哪一把」
        "api_key_masked": runtime_config.mask(cfg["api_key"]),
        "source": cfg["source"],  # user / env / none
        "presets": PRESETS,
    }


@router.get("/setup/status")
async def setup_status():
    """前端加载时调用：决定是否弹出设置页。"""
    return _status_payload()


@router.post("/setup/llm")
async def setup_save(payload: LLMConfigIn, request: Request):
    """保存配置。保存后立即生效，无需重启（runtime_config 按文件 mtime 热加载）。"""
    _guard_write(request)

    base_url = payload.base_url.strip()
    if base_url:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(400, "Base URL 格式不对，应形如 https://api.deepseek.com")

    api_key = payload.api_key.strip()
    if api_key and len(api_key) < 8:
        raise HTTPException(400, "API Key 看起来不完整，请检查是否复制全了")

    runtime_config.save(
        {
            "base_url": base_url,
            "api_key": api_key,
            "model": payload.model.strip(),
            "fallback_model": payload.fallback_model.strip(),
        }
    )
    print(f"[Config] 用户已在设置页更新 LLM 配置（source 变为 user）")
    return _status_payload()


def _friendly_llm_error(exc: Exception) -> str:
    """把各家的报错翻译成人话——用户看到的应该是「Key 错了」而不是 401 JSON。"""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    if status == 401:
        return "API Key 无效或已被禁用，请确认复制的是完整的 Key"
    if status == 403:
        return "该 Key 没有访问这个模型的权限"
    if status == 404:
        return "请求地址不对：请检查 Base URL 是否少了 /v1，或模型名是否写错"
    if status == 400:
        return "请求被拒绝：多半是模型名写错了，请核对服务商的模型列表"
    if status == 429:
        return "调用过于频繁或额度已用尽，稍后再试或检查账户余额"
    if status and 500 <= int(status) < 600:
        return f"服务商暂时不可用（HTTP {status}），稍后再试"

    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "连接超时：请检查 Base URL 是否正确、网络能否访问该服务商"
    if "connection" in text or "connect" in text or "name or service" in text:
        return "连不上这个地址：请检查 Base URL 与网络"
    return f"调用失败：{exc.__class__.__name__}: {str(exc)[:200]}"


@router.post("/setup/test")
async def setup_test(payload: LLMConfigIn, request: Request):
    """真实发一次最小请求验证连通性。

    只体检、不保存——用户点「测试连接」通过后再点保存，避免把错的配置写进文件。
    请求体里没给的字段回退到「已保存的 / .env 的」，所以只测模型名也行。
    """
    _guard_write(request)

    saved = runtime_config.resolve(settings)
    base_url = payload.base_url.strip() or saved["base_url"]
    api_key = payload.api_key.strip() or saved["api_key"]
    model = payload.model.strip() or saved["model"]

    if not api_key:
        raise HTTPException(400, "请先填写 API Key")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(400, "Base URL 格式不对，应形如 https://api.deepseek.com")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
        max_tokens=8,          # 只验证通不通，别真的花钱
        request_timeout=20,
        max_retries=0,         # 测试要的是真实结果，不要重试掩盖问题
    )

    started = time.perf_counter()
    try:
        await llm.ainvoke("hi")
    except Exception as exc:  # noqa: BLE001 — 这里就是要兜住所有上游异常并翻译
        return {
            "ok": False,
            "message": _friendly_llm_error(exc),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    return {
        "ok": True,
        "message": f"连接成功，模型 {model} 可正常调用",
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
