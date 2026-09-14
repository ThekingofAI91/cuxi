"""
用户意见反馈存储（SQLite）— 轻量落盘，管理后台 /admin 直接读取与标记处理。

设计：
- demo 规模下 SQLite 足够；与 monitor / session_store 一致的工程风格（WAL + 全局锁 + 写穿）；
- 字段：id / ts / character / contact / content / page / status / reply / replied_at；
- status: pending（待处理）-> resolved（已处理）-> replied（已回复，带回复内容）；
- 提供 record / list / get / set_status / count_by_status，均为同步、线程安全。
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

STATUS_PENDING = "pending"
STATUS_RESOLVED = "resolved"
STATUS_REPLIED = "replied"
_VALID_STATUS = {STATUS_PENDING, STATUS_RESOLVED, STATUS_REPLIED}


class FeedbackStore:
    def __init__(self, db_path: str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    character TEXT DEFAULT '',
                    contact TEXT DEFAULT '',
                    content TEXT DEFAULT '',
                    page TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    reply TEXT DEFAULT '',
                    replied_at REAL
                )
                """
            )
            self._conn.commit()

    def record(
        self,
        content: str,
        character: str = "",
        contact: str = "",
        page: str = "",
    ) -> dict[str, Any]:
        """新增一条反馈，返回完整行。content 为空直接抛 ValueError。"""
        content = (content or "").strip()
        if not content:
            raise ValueError("反馈内容不能为空")
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO feedback (ts, character, contact, content, page, status)
                VALUES (?,?,?,?,?,?)
                """,
                (now, (character or "").strip(), (contact or "").strip(),
                 content, (page or "").strip(), STATUS_PENDING),
            )
            self._conn.commit()
            fid = cur.lastrowid
            row = self._conn.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
        return dict(row)

    def list(self, status: Optional[str] = None, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        """按时间倒序列出反馈；status 可选过滤（pending/resolved/replied）。"""
        sql = "SELECT * FROM feedback"
        args: list[Any] = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def get(self, feedback_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        return dict(row) if row else None

    def set_status(self, feedback_id: int, status: str, reply: str = "") -> Optional[dict[str, Any]]:
        """更新处理状态（与可选回复）。返回更新后的行；id 不存在返回 None。"""
        if status not in _VALID_STATUS:
            raise ValueError(f"非法状态: {status}")
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
            if not row:
                return None
            self._conn.execute(
                """
                UPDATE feedback SET status=?, reply=?, replied_at=? WHERE id=?
                """,
                (status, (reply or "").strip(), now if status == STATUS_REPLIED else row["replied_at"], feedback_id),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        return dict(row)

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS c FROM feedback GROUP BY status"
            ).fetchall()
        return {r["status"]: r["c"] for r in rows}


_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """全局单例（延迟初始化）"""
    global _store
    if _store is None:
        from src.core.config import settings

        _store = FeedbackStore(settings.feedback_db_path)
    return _store
