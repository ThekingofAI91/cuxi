"""
accounts.py — 账号系统（注册 / 登录 / 会话）

设计取向（配合本项目"轻量自托管"的定位）：
- 存储用 SQLite 单文件（data/accounts.db），与会话/用量库同目录，备份同策略
- 密码哈希用标准库 pbkdf2_hmac（sha256，24 万次迭代），不引入第三方依赖
- 会话用不透明随机 token 放 HttpOnly Cookie（SameSite=Lax），
  服务端查表校验；不用 JWT——单服务场景下查表更简单且可随时吊销
- 账号 = 邮箱或用户名（4-40 字符），不做邮箱验证（无 SMTP 基建时先可用；
  上 SMTP 后在同一张表上加 verified 字段即可，不改表结构主干）
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from src.core.config import settings

# 账号规则：邮箱或用户名，ASCII 字母/数字/_.@-，4-40 字符
# （显式 ASCII——Python 的 \w 会匹配中文等 Unicode 字符，不符合账号定位）
_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_.@-]{4,40}$")
_PASSWORD_MIN = 8

# 会话有效期（秒）；Cookie 与服务端记录同生命周期
SESSION_TTL = 7 * 24 * 3600

_PBKDF2_ITERATIONS = 240_000

_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _db_path() -> Path:
    p = Path(settings.accounts_db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(_db_path(), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA journal_mode=WAL")
        _CONN.execute("PRAGMA busy_timeout=5000")
        _init_tables(_CONN)
    return _CONN


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            account       TEXT UNIQUE NOT NULL,
            display_name  TEXT NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'active',
            created_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """
    )
    conn.commit()


# ============================================================
# 密码哈希（pbkdf2，格式：pbkdf2$iter$salt_hex$hash_hex）
# ============================================================

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


# ============================================================
# 账号与校验
# ============================================================

def validate_account(account: str) -> str | None:
    """校验账号格式；合法返回 None，非法返回面向用户的消息。"""
    a = (account or "").strip().lower()
    if not a:
        return "请填写账号（邮箱或用户名）"
    if not _ACCOUNT_RE.match(a):
        return "账号为 4-40 位字母/数字/下划线（或邮箱）"
    return None


def validate_password(password: str) -> str | None:
    if not password or len(password) < _PASSWORD_MIN:
        return f"密码至少 {_PASSWORD_MIN} 位"
    if len(password) > 72:
        return "密码过长（最多 72 位）"
    return None


def create_user(account: str, password: str, display_name: str = "") -> dict:
    """注册新用户；账号已存在抛 ValueError（消息面向用户）。"""
    a = (account or "").strip().lower()
    err = validate_account(a)
    if err:
        raise ValueError(err)
    err = validate_password(password)
    if err:
        raise ValueError(err)
    conn = _connect()
    now = time.time()
    try:
        with _LOCK:
            conn.execute(
                "INSERT INTO users (id, account, display_name, password_hash, status, created_at) "
                "VALUES (?, ?, ?, ?, 'active', ?)",
                (secrets.token_hex(8), a, (display_name or "").strip() or a, hash_password(password), now),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("该账号已被注册，请换一个或直接登录")
    return get_user_by_account(a)


def get_user_by_account(account: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, account, display_name, password_hash, status, created_at "
        "FROM users WHERE account = ?", ((account or "").strip().lower(),)
    ).fetchone()
    return dict(row) if row else None


def authenticate(account: str, password: str) -> dict:
    """登录校验；失败抛 ValueError（消息面向用户）。"""
    user = get_user_by_account(account)
    # 统一报错不区分"账号不存在/密码错误"，避免账号枚举
    if not user or not verify_password(password, user["password_hash"]):
        raise ValueError("账号或密码不正确")
    if user.get("status") != "active":
        raise ValueError("该账号已被停用，如有疑问请联系管理员")
    return user


# ============================================================
# 会话（不透明 token，HttpOnly Cookie）
# ============================================================

def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn = _connect()
    with _LOCK:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + SESSION_TTL),
        )
        conn.commit()
    return token


def get_user_by_session(token: str) -> dict | None:
    """按会话 token 取用户；token 无效/过期/用户停用返回 None。"""
    if not token:
        return None
    conn = _connect()
    row = conn.execute(
        "SELECT u.id, u.account, u.display_name, u.status FROM sessions s "
        "JOIN users u ON u.id = s.user_id WHERE s.token = ? AND s.expires_at > ?",
        (token, time.time()),
    ).fetchone()
    if not row or row["status"] != "active":
        return None
    return {"id": row["id"], "account": row["account"], "display_name": row["display_name"]}


def delete_session(token: str) -> None:
    if not token:
        return
    conn = _connect()
    with _LOCK:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def delete_account(user_id: str) -> bool:
    """注销账号：删除该用户全部会话与账号记录（隐私政策"删除权"的落地）。

    返回是否确实删除了账号；用户不存在返回 False（幂等）。
    """
    conn = _connect()
    with _LOCK:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    return cur.rowcount > 0


def purge_expired_sessions() -> int:
    """清理过期会话（登录时顺带触发即可，无需独立定时任务）。"""
    conn = _connect()
    with _LOCK:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        conn.commit()
    return cur.rowcount
