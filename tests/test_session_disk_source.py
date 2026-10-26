"""会话历史的磁盘单源与多进程安全（版本号 CAS）。

背景（2026-10-26）：
历史原先以进程内 dict 为读取主路径、SQLite 只做写穿镜像。多 worker 部署时
请求没有亲和性，同一个 session 可能落到不同进程 —— 读不到上文，且落盘时用
自己那份残缺快照覆盖，把别的进程写的完整历史截断。

本文件锁两件事：
1. 读必须走磁盘：另一个 SessionStore 实例写入的内容，本实例能读到
2. 并发写不能丢轮次：拿旧 version 的写入必须被拒绝
"""
import json

import pytest

from src.core import session_store as session_store_mod
from src.core.config import settings
from src.core.session_store import SessionStore


@pytest.fixture()
def two_workers(tmp_path):
    """两个独立连接指向同一个库文件 —— 等价于两个 uvicorn worker。"""
    db = str(tmp_path / "sessions.db")
    w1 = SessionStore(db)
    w2 = SessionStore(db)
    yield w1, w2
    w1._conn.close()
    w2._conn.close()


def test_read_sees_write_from_other_connection(two_workers):
    """一个 worker 写入的历史，另一个 worker 必须能读到（证明读走磁盘）。"""
    w1, w2 = two_workers
    w1.set_history("s1", [["问题1", "回答1"]])
    row = w2.get("s1")
    assert row is not None
    assert json.loads(row["history"]) == [["问题1", "回答1"]]


def test_stale_version_write_is_rejected(two_workers):
    """旧版本写入必须被拒绝，否则先写者的轮次会被冲掉。"""
    w1, w2 = two_workers
    w1.set_history("s1", [["问题1", "回答1"]])

    seen = w2.get("s1")["version"]

    # w2 先写：成功，版本 +1
    assert w2.set_history_cas("s1", [["问题1", "回答1"], ["问题2", "回答2"]], seen) is True
    # w1 拿着同一个旧版本再写：必须失败
    assert w1.set_history_cas("s1", [["问题1", "回答1"], ["问题2", "回答2"]], seen) is False

    assert len(json.loads(w1.get("s1")["history"])) == 2


def test_unconditional_write_bumps_version(two_workers):
    """无条件覆盖也要推进版本，否则后续 CAS 的判据会失效。"""
    w1, _ = two_workers
    v0 = w1.set_history("s1", [["q", "a"]])
    v1 = w1.set_history("s1", [["q", "a"], ["q2", "a2"]])
    assert v1 > v0


def test_append_conversation_keeps_other_worker_turns(tmp_path, monkeypatch):
    """worker1 写一轮、worker2 写一轮，两轮都要在（模拟无亲和性分发）。"""
    monkeypatch.setattr(settings, "session_db_path", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(session_store_mod, "_store", None)   # 重置单例，让它用新路径

    from framework import runtime

    runtime.append_conversation("s-append", "问题1", "回答1")

    # 绕过本进程的 runtime，直接用另一个连接往同一行追加（等价于另一个 worker）
    other = SessionStore(str(tmp_path / "sessions.db"))
    row = other.get("s-append")
    turns = json.loads(row["history"])
    turns.append(["问题2", "回答2"])
    assert other.set_history_cas("s-append", turns, row["version"]) is True

    assert runtime.get_conversation_history("s-append") == [("问题1", "回答1"), ("问题2", "回答2")]
    other._conn.close()


def test_truncate_and_pop_round_trip(tmp_path, monkeypatch):
    """截断与弹轮走磁盘后语义不变；不存在的会话返回 -1。"""
    monkeypatch.setattr(settings, "session_db_path", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(session_store_mod, "_store", None)

    from framework import runtime

    runtime.append_conversation("s-ctl", "问题1", "回答1")
    runtime.append_conversation("s-ctl", "问题2", "回答2")
    runtime.append_conversation("s-ctl", "问题3", "回答3")

    assert runtime.truncate_conversation_history("s-ctl", 2) == 2
    assert runtime.get_conversation_history("s-ctl") == [("问题1", "回答1"), ("问题2", "回答2")]

    assert runtime.pop_last_turn("s-ctl", expect_query="问题2") is True
    assert runtime.get_conversation_history("s-ctl") == [("问题1", "回答1")]

    assert runtime.truncate_conversation_history("s-不存在", 3) == -1
