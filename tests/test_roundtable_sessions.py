"""会话租约、原子选择和订阅清理的不变量；不启动 API 或调用模型。"""

from __future__ import annotations

import asyncio
import importlib.util
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from framework import roundtable_sessions as sessions


def _instance(monkeypatch, name):
    """独立模块拥有不同进程标识和任务字典，仅共享 SQLite 文件。"""
    spec = importlib.util.spec_from_file_location(name, sessions.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _race(*operations):
    barrier = threading.Barrier(len(operations))

    def enter(operation):
        barrier.wait(timeout=5)
        return operation()

    with ThreadPoolExecutor(max_workers=len(operations)) as pool:
        futures = [pool.submit(enter, operation) for operation in operations]
        return [future.result(timeout=10) for future in futures]


def _attempt(module, method, *args):
    try:
        return {"module": module, "row": getattr(module, method)(*args), "error": None}
    except module.RoundtableConflictError as exc:
        return {"module": module, "row": None, "error": exc.code}


def test_independent_instances_atomically_accept_only_one_start_and_one_choice(monkeypatch, tmp_path):
    a = _instance(monkeypatch, "roundtable_sessions_test_owner_a")
    b = _instance(monkeypatch, "roundtable_sessions_test_owner_b")
    assert a._OWNER != b._OWNER
    path = str(tmp_path / "shared.db")
    starts = _race(
        lambda: _attempt(a, "_claim_start", path, "meeting", {"topic": "甲的请求"}),
        lambda: _attempt(b, "_claim_start", path, "meeting", {"topic": "乙的请求"}),
    )
    accepted = [result for result in starts if result["row"] is not None]
    assert len(accepted) == 1
    assert len([result for result in starts if result["error"]]) == 1
    initial = accepted[0]
    assert initial["module"]._finish(path, initial["row"], "waiting", "decision-1", None)

    choices = _race(
        lambda: _attempt(a, "_claim_choice", path, "meeting", "decision-1", "speaker-a"),
        lambda: _attempt(b, "_claim_choice", path, "meeting", "decision-1", "speaker-b"),
    )
    accepted_choices = [result for result in choices if result["row"] is not None]
    assert len(accepted_choices) == 1
    assert len([result for result in choices if result["error"]]) == 1
    chosen = accepted_choices[0]
    persisted = a._read(path, "meeting")
    assert persisted["status"] == "running"
    assert persisted["owner"] == chosen["module"]._OWNER
    assert persisted["accepted_decision_id"] == "decision-1"
    assert persisted["accepted_choice"] == chosen["row"]["accepted_choice"]
    assert persisted["run_id"] == initial["row"]["run_id"]

    # 即使下一轮已经等待，上一轮迟到的点击也不能消费新一轮。
    assert chosen["module"]._finish(path, chosen["row"], "waiting", "decision-2", None)
    stale = _attempt(a, "_claim_choice", path, "meeting", "decision-1", "speaker-a")
    assert stale["row"] is None and stale["error"]
    assert a._read(path, "meeting")["decision_id"] == "decision-2"


def test_expired_lease_has_one_successor_and_old_owner_cannot_change_status(monkeypatch, tmp_path):
    a = _instance(monkeypatch, "roundtable_sessions_test_lease_a")
    b = _instance(monkeypatch, "roundtable_sessions_test_lease_b")
    path = str(tmp_path / "shared.db")
    original = a._claim_start(path, "meeting", {})
    # 固定数据库时钟条件，不靠短 sleep 等真实租约到期。
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE roundtable_runs SET lease_until=0 WHERE session_id=?", ("meeting",))
    results = _race(
        lambda: _attempt(a, "_claim_continue", path, "meeting"),
        lambda: _attempt(b, "_claim_continue", path, "meeting"),
    )
    successors = [result for result in results if result["row"] is not None]
    assert len(successors) == 1
    current = successors[0]["row"]
    assert current["op_id"] != original["op_id"]
    assert not a._renew(path, original)
    assert not a._finish(path, original, "completed", "", None)
    assert a._read(path, "meeting")["op_id"] == current["op_id"]
    assert a._read(path, "meeting")["status"] == "running"


@pytest.mark.parametrize("failure", ["snapshot", "finish"])
def test_finalization_failure_still_closes_subscribers_and_removes_the_job(monkeypatch, tmp_path, failure):
    path = str(tmp_path / "runs.db")

    async def run():
        row = await sessions.claim_start(path, "meeting", {})

        async def driver():
            yield {"type": "end", "session_id": "meeting"}

        async def snapshot():
            if failure == "snapshot":
                raise RuntimeError("checkpoint read failed")
            return {"status": "completed"}

        if failure == "finish":
            def failed_finish(*args, **kwargs):
                raise sqlite3.OperationalError("database is locked")
            monkeypatch.setattr(sessions, "_finish", failed_finish)

        stream = sessions.launch(path, row, driver, snapshot)
        job = sessions._jobs[sessions._key(path, "meeting")]

        async def collect():
            return [event async for event in stream]

        # timeout 仅防止回归时测试挂死；断言关注终止和清理，不依赖具体耗时。
        events = await asyncio.wait_for(collect(), timeout=5)
        await job.task
        assert events[0]["type"] == "end"
        assert any(event["type"] == "error" for event in events)
        assert sessions._key(path, "meeting") not in sessions._jobs
        assert not job.subscribers

    asyncio.run(run())


@pytest.mark.parametrize("renewal_raises", [False, True])
def test_lost_or_failed_renewal_stops_driver_before_recovery(monkeypatch, tmp_path, renewal_raises):
    path = str(tmp_path / "runs.db")
    monkeypatch.setattr(sessions, "HEARTBEAT_SECONDS", 0)

    def cannot_renew(*args):
        if renewal_raises:
            raise sqlite3.OperationalError("database is locked")
        return False

    monkeypatch.setattr(sessions, "_renew", cannot_renew)

    async def run():
        row = await sessions.claim_start(path, "meeting", {})
        stopped = asyncio.Event()
        hold = asyncio.Event()

        async def driver():
            try:
                yield {"type": "started"}
                await hold.wait()
                pytest.fail("失去租约后不应继续生成")
            finally:
                stopped.set()

        async def snapshot():
            return {"status": "interrupted"}

        stream = sessions.launch(path, row, driver, snapshot)
        job = sessions._jobs[sessions._key(path, "meeting")]

        async def collect():
            return [event async for event in stream]

        events = await asyncio.wait_for(collect(), timeout=5)
        await asyncio.gather(job.task, return_exceptions=True)
        assert stopped.is_set() and job.task.done()
        assert events[0]["type"] == "started"
        assert sessions._key(path, "meeting") not in sessions._jobs
        successor = await sessions.claim_continue(path, "meeting")
        assert successor is not None and successor["op_id"] != row["op_id"]

    asyncio.run(run())
