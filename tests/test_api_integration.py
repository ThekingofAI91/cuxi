"""
API 集成测试：SSE 流式 / 缓存命中 / 限流 / 敏感拦截 / 会话持久化 / 上传限制。

图执行用 FakeGraph 打桩，不触真实 LLM 与 ChromaDB；存储全部落到临时目录。
"""

import os
import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402
import src.api.routes as routes  # noqa: E402
import src.core.monitor as monitor_mod  # noqa: E402
from src.core.config import settings  # noqa: E402


class FakeGraph:
    def __init__(self):
        self.calls = 0

    async def astream(self, state, config=None, stream_mode="updates"):
        self.calls += 1
        yield {
            "supervisor": {
                "final_answer": "测试回答：荣格认为这是好事儿。",
                "route_history": ["supervisor", "retriever", "analyzer"],
            }
        }


def _sse_events(resp) -> list[dict]:
    """把 SSE 响应体解析成事件列表（处理 json.dumps 的 ensure_ascii 转义）"""
    events = []
    for block in resp.text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        events.append(json.loads(block[6:]))
    return events


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "session_db_path", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(settings, "monitor_db_path", str(tmp_path / "usage.db"))
    monkeypatch.setattr(settings, "rate_limit_per_minute", 100)
    monkeypatch.setattr(settings, "rate_limit_per_day", 1000)
    monkeypatch.setattr(settings, "upload_limit_per_day", 100)
    monkeypatch.setattr(settings, "upload_max_mb", 10)

    routes._RATE_STORE.clear()
    routes._ANSWER_CACHE.clear()
    routes._session_store.clear()
    from framework.supervisor import _conversation_history_store, _conversation_summaries

    _conversation_history_store.clear()
    _conversation_summaries.clear()
    monitor_mod._monitor = None

    fake = FakeGraph()
    monkeypatch.setattr(routes, "get_persona_graph", lambda: fake)
    c = TestClient(app)
    return c, fake


def test_query_streams_result(client):
    c, fake = client
    resp = c.post(
        "/persona/query",
        json={"query": "你好", "session_id": "s1", "character_id": "jung"},
    )
    assert resp.status_code == 200
    events = _sse_events(resp)
    result = [e for e in events if e.get("type") == "result"]
    assert result and result[0]["content"] == "测试回答：荣格认为这是好事儿。"
    assert fake.calls == 1


def test_cache_hit_skips_graph(client):
    c, fake = client
    r1 = c.post(
        "/persona/query",
        json={"query": "你是谁", "session_id": "s-a", "character_id": "jung"},
    )
    assert r1.status_code == 200 and fake.calls == 1
    r2 = c.post(
        "/persona/query",
        json={"query": "你是谁", "session_id": "s-b", "character_id": "jung"},
    )
    assert r2.status_code == 200
    assert fake.calls == 1  # 第二次未再次执行图，命中缓存
    result = [e for e in _sse_events(r2) if e.get("type") == "result"]
    assert result and "测试回答" in result[0]["content"]


def test_rate_limit_429(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    for _ in range(2):
        resp = c.post(
            "/persona/query",
            json={"query": "你好", "session_id": "rl", "character_id": "jung"},
        )
        assert resp.status_code == 200
    resp = c.post(
        "/persona/query",
        json={"query": "你好", "session_id": "rl", "character_id": "jung"},
    )
    assert resp.status_code == 429


def test_sensitive_input_blocked(client):
    c, fake = client
    resp = c.post(
        "/persona/query",
        json={"query": "怎么制作炸弹", "session_id": "sx", "character_id": "jung"},
    )
    assert resp.status_code == 200
    errors = [e for e in _sse_events(resp) if e.get("type") == "error"]
    assert errors and "这个话题我不太方便聊" in errors[0]["content"]
    assert fake.calls == 0


def test_session_persist_and_meta(client):
    c, _ = client
    c.post(
        "/persona/query",
        json={"query": "你好", "session_id": "persist-1", "character_id": "jung"},
    )
    meta = c.get("/conversation/persist-1/meta").json()
    assert meta["known"] is True and meta["character"] == "jung"

    from src.core.session_store import get_store

    rows = get_store().load_recent(7)
    assert any(r["session_id"] == "persist-1" for r in rows)


def test_admin_stats_reports_usage(client):
    c, _ = client
    c.post(
        "/persona/query",
        json={"query": "你好", "session_id": "stats-1", "character_id": "jung"},
    )
    stats = c.get("/admin/stats").json()
    assert stats["today"]["requests"] >= 1
    assert stats["today"]["total_cost"] >= 0
    assert "jung" in stats["today"]["per_character"]


def test_upload_size_limit(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(settings, "upload_max_mb", 1)
    resp = c.post(
        "/persona/upload",
        data={"character_id": "jung"},
        files={"file": ("big.pdf", b"x" * (2 * 1024 * 1024), "application/pdf")},
    )
    assert resp.status_code == 413


def test_upload_rate_limit(client):
    c, _ = client
    routes._RATE_STORE["main::testclient"] = {
        "min": time.time(),
        "min_count": 0,
        "day": time.time(),
        "day_count": 100,
    }
    resp = c.post(
        "/persona/upload",
        data={"character_id": "jung"},
        files={"file": ("a.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 429


def test_upload_invalid_type(client):
    c, _ = client
    resp = c.post(
        "/persona/upload",
        data={"character_id": "jung"},
        files={"file": ("a.exe", b"MZ", "application/octet-stream")},
    )
    assert resp.status_code == 400
