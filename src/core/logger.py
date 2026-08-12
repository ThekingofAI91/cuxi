"""
结构化日志：request_id 贯穿一次请求，控制台 + 滚动文件双输出。

用法：
    from src.core.logger import get_logger, set_request_id
    set_request_id("req_abc")
    logger = get_logger("routes")
    logger.info("...")

日志文件：logs/app.log（10MB 滚动 × 3 份），可配合 logrotate 或直接归档。
"""

import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: str) -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | rid=%(request_id)s | %(name)s | %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.addFilter(_RequestIdFilter())
    logger.addHandler(sh)

    try:
        from src.core.config import settings

        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        fh.addFilter(_RequestIdFilter())
        logger.addHandler(fh)
    except Exception:
        pass  # 文件日志失败不阻塞主流程
    return logger


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    if name not in _loggers:
        _loggers[name] = _build_logger(name)
    return _loggers[name]
