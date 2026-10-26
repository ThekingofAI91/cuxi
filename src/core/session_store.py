"""
会话持久化（SQLite）— 服务重启后恢复对话历史 / 摘要 / 会话状态，并按 TTL 清理。

设计：
- 内存 store 仍是读取主路径（快），本模块做写穿（write-through）持久化；
- 分列更新（history / summary / meta 互不覆盖），避免并发时互相清空；
- 同步 sqlite3 + WAL + 全局锁，demo 规模下足够；并发量大后可换 Redis。
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class SessionStore:
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
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    character TEXT DEFAULT '',
                    query TEXT DEFAULT '',
                    route_history TEXT DEFAULT '[]',
                    final_answer TEXT DEFAULT '',
                    history TEXT DEFAULT '[]',
                    summary TEXT DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roundtables (
                    session_id TEXT PRIMARY KEY,
                    topic TEXT DEFAULT '',
                    speakers TEXT DEFAULT '[]',
                    rounds INTEGER DEFAULT 1,
                    transcript TEXT DEFAULT '[]',
                    summary TEXT DEFAULT '{}',
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            # 老库补列（CREATE TABLE IF NOT EXISTS 不会给已存在的表加字段）。
            # 单独一条 ALTER + 忽略异常，重复执行无副作用。
            try:
                self._conn.execute(
                    "ALTER TABLE roundtables ADD COLUMN summary TEXT DEFAULT '{}'"
                )
            except sqlite3.OperationalError:
                pass
            # version：历史写入的乐观锁版本号（2026-10-26 加）。
            # 多 worker 下同一 session 会被不同进程各自「读改写」，没有它就会
            # 出现后落盘者覆盖先落盘者。见 set_history_cas。
            try:
                self._conn.execute(
                    "ALTER TABLE sessions ADD COLUMN version INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
            self._conn.commit()

    def _ensure(self, session_id: str, now: float) -> None:
        """确保行存在（不覆盖已有字段）"""
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?,?,?)",
            (session_id, now, now),
        )

    def ensure(self, session_id: str) -> None:
        now = time.time()
        with self._lock:
            self._ensure(session_id, now)
            self._conn.commit()

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """按 session_id 取整行（历史 / 摘要 / 版本 / 元信息）。不存在返回 None。

        历史读取的唯一入口。2026-10-26 起内存不再持有会话副本，
        这样多 worker 下任何一个进程都能读到同一条最新历史。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def set_history(self, session_id: str, history: list) -> int:
        """无条件覆盖历史，返回写入后的 version。

        只用于「确定要覆盖」的场景（前端历史恢复、CAS 连续冲突兜底）。
        常规写入走 set_history_cas，否则并发下会互相冲掉。
        """
        now = time.time()
        with self._lock:
            self._ensure(session_id, now)
            self._conn.execute(
                "UPDATE sessions SET history=?, version=version+1, updated_at=? "
                "WHERE session_id=?",
                (json.dumps(history, ensure_ascii=False), now, session_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT version FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return int(row["version"]) if row else 0

    def set_history_cas(
        self, session_id: str, history: list, expected_version: int
    ) -> bool:
        """仅当 version 仍是 expected_version 时覆盖历史。

        返回 False 表示期间被别人改过 —— 调用方必须重读、重新合并、再写，
        不能当成「写失败」直接放弃，否则那一轮就丢了。
        """
        now = time.time()
        with self._lock:
            self._ensure(session_id, now)
            cur = self._conn.execute(
                "UPDATE sessions SET history=?, version=version+1, updated_at=? "
                "WHERE session_id=? AND version=?",
                (json.dumps(history, ensure_ascii=False), now, session_id, expected_version),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def set_summary(self, session_id: str, summary: str) -> None:
        now = time.time()
        with self._lock:
            self._ensure(session_id, now)
            self._conn.execute(
                "UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",
                (summary or "", now, session_id),
            )
            self._conn.commit()

    def set_meta(
        self,
        session_id: str,
        character: str = "",
        query: str = "",
        route_history: Optional[list] = None,
        final_answer: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            self._ensure(session_id, now)
            self._conn.execute(
                """
                UPDATE sessions SET character=?, query=?, route_history=?, final_answer=?, updated_at=?
                WHERE session_id=?
                """,
                (
                    character or "",
                    query or "",
                    json.dumps(route_history or [], ensure_ascii=False),
                    final_answer or "",
                    now,
                    session_id,
                ),
            )
            self._conn.commit()

    def load_recent(self, ttl_days: int) -> list[dict[str, Any]]:
        cutoff = time.time() - ttl_days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE updated_at >= ?", (cutoff,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self._conn.commit()

    def delete_expired(self, ttl_days: int) -> list[str]:
        cutoff = time.time() - ttl_days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id FROM sessions WHERE updated_at < ?", (cutoff,)
            ).fetchall()
            ids = [r["session_id"] for r in rows]
            if ids:
                self._conn.executemany(
                    "DELETE FROM sessions WHERE session_id=?", [(i,) for i in ids]
                )
            self._conn.execute(
                "DELETE FROM roundtables WHERE updated_at < ?", (cutoff,)
            )
            self._conn.commit()
        return ids

    # ---- 圆桌会议会话 ----

    def save_roundtable(
        self,
        session_id: str,
        topic: str,
        speakers: list,
        rounds: int,
        transcript: list,
        summary: Optional[dict] = None,
    ) -> None:
        """保存一场圆桌会议（topic / 与会者 / 完整发言记录 / 纪要），按 session_id 覆盖。"""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO roundtables
                    (session_id, topic, speakers, rounds, transcript, summary, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    topic or "",
                    json.dumps(speakers or [], ensure_ascii=False),
                    int(rounds or 1),
                    json.dumps(transcript or [], ensure_ascii=False),
                    json.dumps(summary or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def get_roundtable(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM roundtables WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("speakers", "transcript"):
            try:
                d[k] = json.loads(d[k] or "[]")
            except Exception:
                d[k] = []
        try:
            d["summary"] = json.loads(d.get("summary") or "{}")
        except Exception:
            d["summary"] = {}
        return d

    def list_roundtables(self, ttl_days: int = 30) -> list[dict[str, Any]]:
        """列出 TTL 内最近的圆桌会议（按更新时间倒序），transcript 不返回（省流量）。"""
        cutoff = time.time() - ttl_days * 86400
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id, topic, speakers, rounds, created_at, updated_at
                FROM roundtables WHERE updated_at >= ? ORDER BY updated_at DESC
                """,
                (cutoff,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["speakers"] = json.loads(d["speakers"] or "[]")
            except Exception:
                d["speakers"] = []
            out.append(d)
        return out

    def delete_roundtable(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM roundtables WHERE session_id=?", (session_id,)
            )
            self._conn.commit()

    def stats(self, ttl_days: int) -> dict[str, Any]:
        """管理后台用：按角色聚合会话数与对话轮次（TTL 内活跃会话）。"""
        cutoff = time.time() - ttl_days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT character, history FROM sessions WHERE updated_at >= ?", (cutoff,)
            ).fetchall()
        by_character: dict[str, dict[str, int]] = {}
        total_turns = 0
        for r in rows:
            ch = r["character"] or "unknown"
            d = by_character.setdefault(ch, {"sessions": 0, "turns": 0})
            d["sessions"] += 1
            try:
                hist = json.loads(r["history"] or "[]")
                turns = len(hist) if isinstance(hist, list) else 0
                d["turns"] += turns
                total_turns += turns
            except Exception:
                pass
        return {
            "total_sessions": len(rows),
            "total_turns": total_turns,
            "by_character": by_character,
        }


_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    """全局单例（延迟初始化，避免 import 时创建）"""
    global _store
    if _store is None:
        from src.core.config import settings

        _store = SessionStore(settings.session_db_path)
    return _store
