"""圆桌执行段的原子占用与断线保活。

SQLite 负责跨 worker 的选择去重和执行租约；进程内 Task 负责让已接受的执行段
不随 HTTP 订阅取消。租约过期后的恢复允许重跑未完成节点，不承诺外部模型请求恰好一次。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


LEASE_SECONDS = 60.0
HEARTBEAT_SECONDS = 10.0
_OWNER = uuid.uuid4().hex
_DONE = object()


class RoundtableConflictError(RuntimeError):
    def __init__(self, code: str, content: str):
        super().__init__(content)
        self.code = code
        self.content = content


@contextmanager
def _connection(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("""CREATE TABLE IF NOT EXISTS roundtable_runs (
        session_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, status TEXT NOT NULL,
        op_id TEXT NOT NULL DEFAULT '', owner TEXT NOT NULL DEFAULT '', lease_until REAL NOT NULL DEFAULT 0,
        decision_id TEXT NOT NULL DEFAULT '', accepted_decision_id TEXT NOT NULL DEFAULT '',
        accepted_choice TEXT NOT NULL DEFAULT '', start_payload TEXT NOT NULL,
        summary TEXT, updated_at REAL NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS roundtable_consumed_decisions (
        run_id TEXT NOT NULL, decision_id TEXT NOT NULL, PRIMARY KEY (run_id, decision_id)
    )""")
    try:
        yield conn
    finally:
        conn.close()


def _read(path: str, session_id: str) -> dict | None:
    with _connection(path) as conn:
        row = conn.execute("SELECT * FROM roundtable_runs WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


async def read_run(path: str, session_id: str) -> dict | None:
    return await asyncio.to_thread(_read, path, session_id)


async def adopt_checkpoint(path: str, session_id: str, *, run_id: str, status: str,
                           decision_id: str, payload: dict) -> dict:
    """把升级前仅有 checkpoint 的会议登记一次；永不覆盖已存在的新运行记录。"""
    def adopt():
        with _connection(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""INSERT OR IGNORE INTO roundtable_runs
                (session_id, run_id, status, decision_id, start_payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, run_id, status, decision_id, json.dumps(payload, ensure_ascii=False), time.time()))
            conn.commit()
        return _read(path, session_id)
    return await asyncio.to_thread(adopt)


async def invalidate_waiting(path: str, session_id: str, run_id: str):
    def invalidate():
        with _connection(path) as conn:
            conn.execute("""UPDATE roundtable_runs SET status='interrupted', decision_id='', updated_at=?
                WHERE session_id=? AND run_id=? AND status='waiting' AND op_id=''""",
                (time.time(), session_id, run_id))
    await asyncio.to_thread(invalidate)


def _claim_start(path: str, session_id: str, payload: dict) -> dict:
    with _connection(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM roundtable_runs WHERE session_id = ?", (session_id,)).fetchone()
        if row and row["status"] != "completed":
            raise RoundtableConflictError("session_active", "这场会议尚未结束，请恢复会议或另开一场。")
        run_id, op_id, now = uuid.uuid4().hex, uuid.uuid4().hex, time.time()
        if row:
            conn.execute("DELETE FROM roundtable_consumed_decisions WHERE run_id = ?", (row["run_id"],))
        conn.execute("""INSERT OR REPLACE INTO roundtable_runs
            (session_id, run_id, status, op_id, owner, lease_until, start_payload, updated_at)
            VALUES (?, ?, 'running', ?, ?, ?, ?, ?)""",
            (session_id, run_id, op_id, _OWNER, now + LEASE_SECONDS,
             json.dumps(payload, ensure_ascii=False), now))
        conn.commit()
    return _read(path, session_id)


async def claim_start(path: str, session_id: str, payload: dict) -> dict:
    return await asyncio.to_thread(_claim_start, path, session_id, payload)


def _claim_choice(path: str, session_id: str, decision_id: str, choice: str) -> dict:
    with _connection(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM roundtable_runs WHERE session_id = ?", (session_id,)).fetchone()
        if not row or row["status"] != "waiting":
            raise RoundtableConflictError("decision_unavailable", "这次选择已被处理，或会议当前不在等待点将，请刷新会议。")
        if not decision_id or row["decision_id"] != decision_id:
            raise RoundtableConflictError("stale_decision", "这份请缨已经过期，请查看当前一轮。")
        try:
            conn.execute("INSERT INTO roundtable_consumed_decisions VALUES (?, ?)",
                         (row["run_id"], decision_id))
        except sqlite3.IntegrityError as exc:
            raise RoundtableConflictError("decision_consumed", "这次选择已经提交，请查看会议进度。") from exc
        now, op_id = time.time(), uuid.uuid4().hex
        conn.execute("""UPDATE roundtable_runs SET status='running', op_id=?, owner=?, lease_until=?,
            decision_id='', accepted_decision_id=?, accepted_choice=?, updated_at=? WHERE session_id=?""",
            (op_id, _OWNER, now + LEASE_SECONDS, decision_id, choice, now, session_id))
        conn.commit()
    return _read(path, session_id)


async def claim_choice(path: str, session_id: str, decision_id: str, choice: str) -> dict:
    return await asyncio.to_thread(_claim_choice, path, session_id, decision_id, choice)


def _claim_continue(path: str, session_id: str) -> dict | None:
    with _connection(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM roundtable_runs WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            raise RoundtableConflictError("session_missing", "这场会议不存在。")
        now = time.time()
        if row["status"] in ("waiting", "completed"):
            return None
        if row["status"] == "running" and float(row["lease_until"]) > now:
            return None
        conn.execute("""UPDATE roundtable_runs SET status='running', op_id=?, owner=?, lease_until=?,
            updated_at=? WHERE session_id=?""",
            (uuid.uuid4().hex, _OWNER, now + LEASE_SECONDS, now, session_id))
        conn.commit()
    return _read(path, session_id)


async def claim_continue(path: str, session_id: str) -> dict | None:
    return await asyncio.to_thread(_claim_continue, path, session_id)


def _renew(path: str, row: dict) -> bool:
    with _connection(path) as conn:
        now = time.time()
        cursor = conn.execute("""UPDATE roundtable_runs SET lease_until=?, updated_at=?
            WHERE session_id=? AND op_id=? AND owner=? AND status='running'""",
            (now + LEASE_SECONDS, now, row["session_id"], row["op_id"], _OWNER))
        return cursor.rowcount == 1


def _finish(path: str, row: dict, status: str, decision_id: str, summary) -> bool:
    with _connection(path) as conn:
        cursor = conn.execute("""UPDATE roundtable_runs SET status=?, decision_id=?,
            op_id='', owner='', lease_until=0, summary=COALESCE(?, summary), updated_at=?
            WHERE session_id=? AND op_id=? AND owner=?""",
            (status, decision_id, json.dumps(summary, ensure_ascii=False) if summary is not None else None,
             time.time(), row["session_id"], row["op_id"], _OWNER))
        return cursor.rowcount == 1


@dataclass
class _Job:
    row: dict
    snapshot: Callable[[], Awaitable[dict | None]]
    on_complete: Callable[[dict, dict], Awaitable[None]] | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    task: asyncio.Task | None = None


_jobs: dict[tuple[str, str], _Job] = {}


def _key(path: str, session_id: str) -> tuple[str, str]:
    return (str(Path(path).resolve()), session_id)


def _subscribe(job: _Job) -> AsyncIterator[dict]:
    queue = asyncio.Queue(maxsize=128)
    job.subscribers.add(queue)

    async def events():
        try:
            while True:
                event = await queue.get()
                if event is _DONE:
                    break
                yield event
        finally:
            job.subscribers.discard(queue)

    return events()


def subscribe_running(path: str, session_id: str) -> AsyncIterator[dict] | None:
    job = _jobs.get(_key(path, session_id))
    if job is not None and job.task is not None and not job.task.done():
        return _subscribe(job)
    return None


async def _broadcast(job: _Job, event: dict):
    snapshot = None
    for queue in list(job.subscribers):
        if queue.full():
            # 慢订阅者不积累无限 token；用完整 checkpoint 快照重新对齐。
            while not queue.empty():
                queue.get_nowait()
            if snapshot is None:
                snapshot = await job.snapshot()
            queue.put_nowait({"type": "snapshot", "data": snapshot})
        queue.put_nowait(event)


async def _heartbeat(path: str, job: _Job):
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        try:
            owned = await asyncio.to_thread(_renew, path, job.row)
        except Exception:
            owned = False  # 无法续租时必须停止旧执行者，不能让它与接管者同时生成。
        if not owned:
            if job.task is not None:
                job.task.cancel()
            return


async def _run(path: str, job: _Job, driver: Callable[[], AsyncIterator[dict]]):
    heartbeat = asyncio.create_task(_heartbeat(path, job))
    pending, summary, ended = None, None, None
    summary_started = 0.0
    try:
        async for event in driver():
            if event.get("type") == "choice_request":
                # checkpoint 已挂起后先提交可选状态，再通知客户端，避免快速点击撞上 running。
                pending = event
                continue
            if event.get("type") == "summary":
                summary = event.get("data")
            elif event.get("type") == "end":
                ended = dict(event)
            elif event.get("type") == "summary_start":
                summary_started = time.perf_counter()
            elif event.get("type") == "summary_end" and ended and summary_started:
                ended["timings"] = {**(ended.get("timings") or {}),
                                    "summary_ms": (time.perf_counter() - summary_started) * 1000}
            await _broadcast(job, event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _broadcast(job, {"type": "error", "content": str(exc)})
    finally:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        try:
            status = "waiting" if pending else "completed" if ended else "interrupted"
            owned = await asyncio.to_thread(_finish, path, job.row, status,
                                            str((pending or {}).get("decision_id") or ""), summary)
            if owned:
                if pending:
                    await _broadcast(job, pending)
                else:
                    if ended and job.on_complete is not None:
                        try:
                            await job.on_complete(ended, summary or {})
                        except Exception as exc:
                            await _broadcast(job, {"type": "error", "code": "archive_failed",
                                                   "content": f"会议已保留，历史记录写入失败：{exc}"})
                    await _broadcast(job, {"type": "snapshot", "data": await job.snapshot()})
        except Exception as exc:
            # 快照读取或落库失败也不能让订阅者永久等待；租约过期后仍可继续。
            event = {"type": "error", "code": "snapshot_failed", "content": f"会议状态暂时无法读取，请重新连接：{exc}"}
            for queue in list(job.subscribers):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(event)
        finally:
            for queue in list(job.subscribers):
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(_DONE)
            if _jobs.get(_key(path, job.row["session_id"])) is job:
                _jobs.pop(_key(path, job.row["session_id"]), None)


def launch(path: str, row: dict, driver: Callable[[], AsyncIterator[dict]],
           snapshot: Callable[[], Awaitable[dict | None]],
           on_complete: Callable[[dict, dict], Awaitable[None]] | None = None) -> AsyncIterator[dict]:
    job = _Job(row=row, snapshot=snapshot, on_complete=on_complete)
    events = _subscribe(job)
    _jobs[_key(path, row["session_id"])] = job
    job.task = asyncio.create_task(_run(path, job, driver), name=f"roundtable:{row['session_id']}")
    return events


async def stop_and_delete(path: str, session_id: str):
    job = _jobs.get(_key(path, session_id))
    if job is not None and job.task is not None:
        job.task.cancel()
        await asyncio.gather(job.task, return_exceptions=True)

    def delete():
        with _connection(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT run_id FROM roundtable_runs WHERE session_id=?", (session_id,)).fetchone()
            if row:
                conn.execute("DELETE FROM roundtable_consumed_decisions WHERE run_id=?", (row["run_id"],))
                conn.execute("DELETE FROM roundtable_runs WHERE session_id=?", (session_id,))
            conn.commit()

    await asyncio.to_thread(delete)
