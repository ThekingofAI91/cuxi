"""
API 集成测试：SSE 流式 / 缓存命中 / 限流 / 敏感拦截 / 会话持久化 / 上传限制。

一对一编排用 FakePipeline 打桩，不触真实 LLM 与 ChromaDB；存储全部落到临时目录。
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
import src.api.deps as deps  # noqa: E402
import src.core.monitor as monitor_mod  # noqa: E402
from src.core.config import settings  # noqa: E402


class FakePipeline:
    """打桩一对一编排（framework/supervisor_agent.run_supervisor_agent）。

    supervisor 自己持有检索工具，本桩只需返回与真实现同构的 dict；
    on_stage 对应前端 agent_done 进度事件（阶段名 supervisor / retrieval_agent）。
    """

    def __init__(self):
        self.calls = 0

    async def __call__(self, state, on_stage=None):
        self.calls += 1
        if on_stage is not None:
            await on_stage("supervisor")
        answer = "测试回答：荣格认为这是好事儿。"
        return {
            "final_answer": answer,
            "analysis": answer,
            "retrieved_docs": [],
            "graph_used": False,
            "route_history": ["supervisor", "retriever", "analyzer"],
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

    deps._RATE_STORE.clear()
    routes._ANSWER_CACHE.clear()
    routes._session_store.clear()
    from framework.runtime import _conversation_history_store, _conversation_summaries

    _conversation_history_store.clear()
    _conversation_summaries.clear()
    monitor_mod._monitor = None

    # 编排函数在 routes 内是延迟导入（每次调用都从源模块取属性），
    # 所以打桩要落在 framework.supervisor_agent 上，而不是 routes 的模块属性。
    import framework.supervisor_agent as _sa

    fake = FakePipeline()
    monkeypatch.setattr(_sa, "run_supervisor_agent", fake)
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


@pytest.mark.parametrize("partial_output", [False, True])
def test_query_failure_ends_stream_and_allows_retry(client, monkeypatch, partial_output):
    c, fake = client
    import framework.supervisor_agent as supervisor
    from framework.runtime import get_conversation_history
    from src.core.llm import LLMEmptyResponseError

    async def failing_pipeline(state, on_stage=None):
        if partial_output:
            await state["stream_callback"]("尚未完成的回答")
            raise RuntimeError("上游连接中断")
        raise LLMEmptyResponseError("模型暂时没有返回内容")

    monkeypatch.setattr(supervisor, "run_supervisor_agent", failing_pipeline)
    payload = {"query": "什么是阴影？", "session_id": "failed-query", "character_id": "jung"}
    response = c.post("/persona/query", json=payload)
    assert response.status_code == 200
    events = _sse_events(response)
    assert [event["type"] for event in events[-2:]] == ["error", "end"]
    assert events[-1]["session_id"] == payload["session_id"]
    assert ("上游连接中断" if partial_output else "模型暂时没有返回内容") in events[-2]["content"]
    assert any(event["type"] == "token" for event in events) is partial_output
    assert not any(event["type"] == "result" for event in events)
    assert get_conversation_history(payload["session_id"]) == []
    assert not routes._ANSWER_CACHE
    assert monitor_mod.get_monitor().summary(0)["errors"] == 1

    monkeypatch.setattr(supervisor, "run_supervisor_agent", fake)
    retry = c.post("/persona/query", json=payload)
    retry_events = _sse_events(retry)
    assert retry.status_code == 200
    assert any(event["type"] == "result" for event in retry_events)
    assert retry_events[-1] == {"type": "end", "session_id": payload["session_id"]}
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
    deps._RATE_STORE["main::testclient"] = {
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
