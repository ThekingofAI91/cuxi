"""
runtime_config.py — 运行期用户配置（开源版的「用户自己填 API Key」）

背景：
  项目开源后不应内置任何 API Key。用户在浏览器首次打开时填写
  Base URL / API Key / 模型名，落到 data/user_config.json（已 gitignore），
  全程不碰文件系统，也不需要重启服务。

优先级：
  用户填写（本模块） > .env / 环境变量 > config.py 字段默认值

为什么要做 mtime 缓存：
  前端保存后要求「无需重启即生效」，所以每次取配置时按文件 mtime 判断是否需要
  重新读盘；文件没变就走内存缓存，避免每次 LLM 调用都产生一次磁盘 IO。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

# 与 config.py 里 session_db_path 等项目内路径保持同样的相对根（仓库根目录）
CONFIG_PATH = Path("./data/user_config.json")

# 允许被用户覆盖的字段（其余配置仍只由 .env 控制，避免任意键污染）
FIELDS = ("base_url", "api_key", "model", "fallback_model")

_lock = threading.Lock()
_cache: dict[str, str] = {}
_cache_mtime: float | None = None


def _read_file(path: Path) -> dict[str, str]:
    """读配置文件；不存在 / 损坏 / 非法 JSON 一律当作「未配置」，不抛异常。

    开源项目里用户手改坏 JSON 是常态，配置读取绝不该让整个服务起不来。
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        print(f"[Config] 读取用户配置失败（按未配置处理）：{e}")
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Config] 用户配置不是合法 JSON（按未配置处理）：{e}")
        return {}

    if not isinstance(data, dict):
        return {}

    out: dict[str, str] = {}
    for key in FIELDS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out


def load(force: bool = False) -> dict[str, str]:
    """返回用户填写的配置（可能为空 dict）。仅在文件 mtime 变化时重新读盘。"""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime: float | None = CONFIG_PATH.stat().st_mtime
        except OSError:
            mtime = None  # 文件不存在 / 不可读

        if force or mtime != _cache_mtime:
            _cache = _read_file(CONFIG_PATH)
            _cache_mtime = mtime

        return dict(_cache)


def save(payload: dict[str, Any]) -> dict[str, str]:
    """写入用户配置（原子替换，避免写一半被读到）。

    payload 中值为空串 / None 的字段表示「不修改」，不会把已有值抹掉——
    这样用户只改模型名时不必重新粘贴 Key。
    """
    merged = load(force=True)

    for key in FIELDS:
        if key not in payload:
            continue
        value = payload.get(key)
        value = value.strip() if isinstance(value, str) else ""
        if value:
            merged[key] = value

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)  # Windows / POSIX 上都是原子替换

    # 立即刷新缓存，避免 mtime 精度问题时保存后读到的还是旧值
    global _cache, _cache_mtime
    with _lock:
        _cache = dict(merged)
        try:
            _cache_mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            _cache_mtime = None
    return dict(merged)


def resolve(settings) -> dict[str, str]:
    """返回最终生效的 LLM 配置（用户填写 > .env > 默认值）。

    settings 由调用方传入，避免本模块反向 import config 造成循环依赖。
    """
    user = load()
    api_key = user.get("api_key") or (settings.llm_api_key or "")
    return {
        "base_url": user.get("base_url") or settings.llm_base_url,
        "api_key": api_key,
        "model": user.get("model") or settings.llm_model,
        "fallback_model": user.get("fallback_model") or (settings.llm_fallback_model or ""),
        "source": "user" if user.get("api_key") else ("env" if settings.llm_api_key else "none"),
    }


def is_configured(settings) -> bool:
    """是否已有可用的 API Key（用户填的或 .env 里的都算）。"""
    return bool(resolve(settings)["api_key"])


def mask(key: str) -> str:
    """脱敏展示：只暴露前缀 6 位与后缀 4 位，供前端确认「填的是哪一把」。"""
    if not key:
        return ""
    if len(key) <= 12:
        return key[:2] + "*" * 6
    return f"{key[:6]}{'*' * 8}{key[-4:]}"


def reset() -> dict[str, str]:
    """清空用户配置（回到 .env / 默认值）。主要给测试与「重置」入口用。"""
    try:
        CONFIG_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"[Config] 清除用户配置失败：{e}")
    return load(force=True)
