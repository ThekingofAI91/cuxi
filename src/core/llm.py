"""
llm.py — ChatOpenAI 统一工厂

所有 LLM 调用统一走 get_chat_llm()，保证超时、重试、模型参数一致：
- request_timeout：单次请求超时，防止上游 API 挂起时 SSE 流永久卡住
- max_retries：网络抖动/5xx 自动重试
- 具体场景的 temperature / max_tokens 仍由调用方通过 kwargs 覆盖
"""

from typing import Any

from langchain_openai import ChatOpenAI

from src.core.config import settings


def get_chat_llm(**kwargs: Any) -> ChatOpenAI:
    """创建 ChatOpenAI 实例，默认注入超时/重试/模型/密钥配置。"""
    kwargs.setdefault("model", settings.llm_model)
    kwargs.setdefault("api_key", settings.llm_api_key)
    kwargs.setdefault("base_url", settings.llm_base_url)
    kwargs.setdefault("request_timeout", settings.llm_request_timeout)
    kwargs.setdefault("max_retries", settings.llm_max_retries)
    return ChatOpenAI(**kwargs)
