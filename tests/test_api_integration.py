"""
API 集成测试：SSE 流式 / 缓存命中 / 限流 / 敏感拦截 / 会话持久化 / 上传限制。

一对一编排用 FakePipeline 打桩，不触真实 LLM 与 ChromaDB；存储全部落到临时目录。
"""

import os
import asyncio
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
    """打桩一对一编排（framework/chat_agent.run_chat_agent）。

    chat_agent 自己持有检索工具，本桩只需返回与真实现同构的 dict；
    on_stage 对应前端 agent_done 进度事件（阶段名 chat_agent / retrieval_agent）。
    """

    def __init__(self):
        self.calls = 0
        self.histories = []

    async def __call__(self, state, on_stage=None):
        self.calls += 1
        self.histories.append(list(state.get("history", [])))
        if on_stage is not None:
            await on_stage("chat_agent")
        answer = "测试回答：荣格认为这是好事儿。"
        return {
            "final_answer": answer,
            "analysis": answer,
            "retrieved_docs": [],
            "graph_used": False,
            "route_history": ["chat_agent", "retriever", "analyzer"],
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
    monkeypatch.setattr(routes, "_SESSION_META_FILE", tmp_path / "session_meta.json")
    monkeypatch.setattr(routes, "_DELETED_SESSIONS_FILE", tmp_path / "deleted_sessions.json")

    deps._RATE_STORE.clear()
    routes._ANSWER_CACHE.clear()
    routes._session_store.clear()
    # 历史现在只有 SQLite 一份（2026-10-26），重置单例让它用上面 monkeypatch 的临时库，
    # 否则会继续指向 ./data/sessions.db，测试之间互相污染。
    from src.core import session_store as _session_store_mod

    _session_store_mod._store = None
    monitor_mod._monitor = None

    # 编排函数在 routes 内是延迟导入（每次调用都从源模块取属性），
    # 所以打桩要落在 framework.chat_agent 上，而不是 routes 的模块属性。
    import framework.chat_agent as _sa

    fake = FakePipeline()
    monkeypatch.setattr(_sa, "run_chat_agent", fake)
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


@pytest.mark.parametrize("candidates", [[], [{"id": "jung", "name": "荣格"}]])
@pytest.mark.parametrize("choice", ["", "   ", "adler"])
def test_roundtable_rejects_non_bidder_without_resuming(client, monkeypatch, candidates, choice):
    c, _ = client
    prepared = []

    async def prepare(session_id, character_id, decision_id, **kwargs):
        prepared.append((session_id, character_id, decision_id))
        assert character_id not in {item["id"] for item in candidates}
        raise routes.RoundtableConflictError("invalid_choice", "请选择本轮请缨者")

    monkeypatch.setattr(routes, "prepare_roundtable_choice", prepare)
    response = c.post("/persona/roundtable/awaiting/choice",
                      json={"character_id": choice, "decision_id": "decision-1"})
    assert response.status_code == 400
    assert len(prepared) == (1 if choice.strip() else 0)


def test_roundtable_start_preserves_client_session_id_for_early_disconnect_recovery(client, monkeypatch):
    c, _ = client
    prepared = []

    async def events(session_id):
        yield {"type": "start", "session_id": session_id}

    async def prepare(topic, character_ids, rounds, session_id, picker="user", **kwargs):
        prepared.append((character_ids, rounds, session_id, picker))
        assert kwargs["on_complete"] is routes._persist_roundtable_result
        return events(session_id)

    monkeypatch.setattr(routes, "prepare_roundtable_start", prepare)
    response = c.post("/persona/roundtable", json={
        "topic": "理解与行动", "character_ids": ["jung", "adler"], "rounds": 6,
        "picker": "user", "session_id": "client-saved-before-start",
    })
    assert response.status_code == 200
    assert prepared == [(["jung", "adler"], 6, "client-saved-before-start", "user")]
    assert _sse_events(response)[0]["session_id"] == "client-saved-before-start"


@pytest.mark.parametrize("choice", ["jung", "__end__"])
def test_roundtable_explicit_choice_preserves_resumed_sse(client, monkeypatch, choice):
    c, _ = client
    resumed = []

    async def resume(session_id, character_id):
        yield {"type": "converged", "reason": "user" if character_id == "__end__" else "rounds"}
        yield {"type": "end", "session_id": session_id, "transcript": []}

    async def prepare(session_id, character_id, decision_id, **kwargs):
        resumed.append((session_id, character_id, decision_id))
        assert kwargs["on_complete"] is routes._persist_roundtable_result
        return resume(session_id, character_id)

    monkeypatch.setattr(routes, "prepare_roundtable_choice", prepare)
    response = c.post("/persona/roundtable/awaiting/choice",
                      json={"character_id": choice, "decision_id": "decision-1"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert resumed == [("awaiting", choice, "decision-1")]
    assert [event["type"] for event in _sse_events(response)] == ["converged", "end"]


def test_roundtable_pending_preserves_all_bid_details(client, monkeypatch):
    c, _ = client
    payload = {
        "round": 1, "total": 2, "can_end": True,
        "candidates": [{"id": "jung", "urgency": 4, "reason": "需要澄清", "move": "clarify",
                        "target": {"speaker": "阿德勒", "quote": "原话片段"}, "speeches": 1, "quota": 2}],
        "intents": [{"character_id": "jung", "want": True},
                    {"character_id": "adler", "want": False, "reason": "此刻没有补充"}],
    }

    async def pending(session_id):
        return payload

    monkeypatch.setattr(routes, "roundtable_pending_choice", pending)
    response = c.get("/persona/roundtable/awaiting/pending")
    assert response.status_code == 200
    assert response.json() == {"pending": True, **payload}


def test_roundtable_choice_requires_current_decision_id(client, monkeypatch):
    c, _ = client

    async def unexpected_prepare(*args, **kwargs):
        pytest.fail("没有决定标识的请求不能启动会议")

    monkeypatch.setattr(routes, "prepare_roundtable_choice", unexpected_prepare)
    assert c.post("/persona/roundtable/awaiting/choice", json={"character_id": "jung"}).status_code == 400


@pytest.mark.parametrize("code", ["stale_decision", "already_claimed"])
def test_roundtable_choice_conflict_returns_409_before_stream(client, monkeypatch, code):
    c, _ = client

    async def conflict(*args, **kwargs):
        raise routes.RoundtableConflictError(code, "该轮点将已经处理，请刷新会议")

    monkeypatch.setattr(routes, "prepare_roundtable_choice", conflict)
    response = c.post("/persona/roundtable/awaiting/choice",
                      json={"character_id": "jung", "decision_id": "old-decision"})
    assert response.status_code == 409
    assert not response.headers["content-type"].startswith("text/event-stream")


def test_roundtable_snapshot_preserves_transcript_and_pending_decision(client, monkeypatch):
    c, _ = client
    snapshot = {"session_id": "awaiting", "status": "waiting", "pending": True,
                "transcript": [{"turn_id": "turn-1", "character_id": "jung", "content": "完整原文"}],
                "ledger": {"jung": {"speeches": 1, "quota": 4}},
                "choice_request": {"decision_id": "current-decision", "candidates": [{"id": "jung"}]}}

    async def get_snapshot(session_id):
        return snapshot if session_id == "awaiting" else None

    monkeypatch.setattr(routes, "roundtable_snapshot", get_snapshot)
    assert c.get("/persona/roundtable/awaiting/snapshot").json() == snapshot
    assert c.get("/persona/roundtable/missing/snapshot").status_code == 404


def test_roundtable_continue_uses_saved_run_and_returns_snapshot_first(client, monkeypatch):
    c, _ = client

    async def events():
        yield {"type": "snapshot", "data": {"session_id": "awaiting", "status": "waiting"}}

    async def continue_saved(session_id, **kwargs):
        assert session_id == "awaiting"
        assert kwargs["on_complete"] is routes._persist_roundtable_result
        return events()

    monkeypatch.setattr(routes, "continue_roundtable", continue_saved)
    response = c.post("/persona/roundtable/awaiting/continue", json={})
    assert response.status_code == 200
    assert _sse_events(response) == [{"type": "snapshot", "data": {"session_id": "awaiting", "status": "waiting"}}]


def test_roundtable_archive_is_independent_of_stream_subscriptions(client):
    # 两条订阅都读到 end 也不能各记一次消耗；唯一的后台完成回调负责持久化。
    c, _ = client
    payload = {"type": "end", "session_id": "background-result", "topic": "议题", "rounds": 3,
               "speakers": [{"id": "jung", "name": "荣格", "quota": 4}],
               "transcript": [{"turn_id": "turn-1", "character_id": "jung", "content": "已完成的发言"}],
               "prompt_chars": 120, "answer_chars": 30, "meeting_ms": 100,
               "timings": {"intent_ms": 20.0, "speaker_ms": 80.0}}

    async def events():
        yield payload

    async def deliver_and_archive():
        for _ in range(2):
            chunks = [chunk async for chunk in routes._roundtable_sse(
                events(), session_id="background-result", logger=routes.get_logger("test"),
                client_ip="test", t0=0,
            )]
            assert chunks
        assert monitor_mod.get_monitor().summary(0)["requests"] == 0
        await routes._persist_roundtable_result(payload, {"closing": "已结束"})

    asyncio.run(deliver_and_archive())
    assert monitor_mod.get_monitor().summary(0)["requests"] == 1
    response = c.get("/persona/roundtable/background-result")
    assert response.status_code == 200
    assert response.json()["transcript"] == payload["transcript"]
    assert response.json()["summary"] == {"closing": "已结束"}


@pytest.mark.parametrize("partial_output", [False, True])
def test_query_failure_ends_stream_and_allows_retry(client, monkeypatch, partial_output):
    c, fake = client
    import framework.chat_agent as chat_agent_mod
    from framework.runtime import get_conversation_history
    from src.core.llm import LLMEmptyResponseError

    async def failing_pipeline(state, on_stage=None):
        if partial_output:
            await state["stream_callback"]("尚未完成的回答")
            raise RuntimeError("上游连接中断")
        raise LLMEmptyResponseError("模型暂时没有返回内容")

    monkeypatch.setattr(chat_agent_mod, "run_chat_agent", failing_pipeline)
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

    monkeypatch.setattr(chat_agent_mod, "run_chat_agent", fake)
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

    from src.core.session_store import get_store
    row = get_store().get("s-b")
    assert json.loads(row["history"]) == [["你是谁", result[0]["content"]]]
    assert row["query"] == "你是谁" and row["final_answer"] == result[0]["content"]
    assert json.loads(row["route_history"]) == ["chat_agent", "retriever", "analyzer"]
    pending = c.get("/conversation/pending/s-b").json()
    assert pending["pending"] is True and pending["final_answer"] == result[0]["content"]
    usage = monitor_mod.get_monitor().summary(0)
    assert usage["requests"] == 2 and usage["cache_hits"] == 1
    cached_usage = monitor_mod.get_monitor()._conn.execute(
        "SELECT prompt_tokens, completion_tokens, cost FROM usage_log WHERE cache_hit=1",
    ).fetchone()
    assert tuple(cached_usage) == (0, 0, 0.0)


def test_followup_after_cache_hit_uses_the_cached_turn_as_history(client):
    c, fake = client
    for sid in ("followup-source", "followup-cached"):
        response = c.post("/persona/query", json={
            "query": "什么是阴影", "session_id": sid, "character_id": "jung",
        })
        assert response.status_code == 200
    assert fake.calls == 1

    followup = c.post("/persona/query", json={
        "query": "刚才的解释可以举个例子吗", "session_id": "followup-cached", "character_id": "jung",
    })
    assert followup.status_code == 200 and fake.calls == 2
    assert fake.histories[-1] == [("什么是阴影", "测试回答：荣格认为这是好事儿。")]

    from framework.runtime import get_conversation_history
    assert len(get_conversation_history("followup-cached")) == 2
    assert len(get_conversation_history("followup-source")) == 1


@pytest.mark.parametrize("with_questions", [False, True])
def test_cache_replays_citations_graph_and_questions_without_shared_mutation(client, monkeypatch, with_questions):
    from langchain_core.documents import Document
    import framework.chat_agent as chat_agent_mod

    c, _ = client
    calls = []
    docs = [
        Document(page_content="原文资料", metadata={"source": "原著.pdf", "heading": "第一章"}),
        Document(page_content="关系证据", metadata={"source": "图谱.md", "kg_inferred": True}),
    ]
    questions = [{"question": "你更关心哪个方面？", "options": ["概念", "应用"]}] if with_questions else None

    async def rich_answer(state, on_stage=None):
        calls.append(state)
        return {"final_answer": "原文依据 [1]，图谱推断 [G1]。", "retrieved_docs": docs,
                "graph_used": True, "route_history": ["chat_agent", "retrieval_agent"],
                "info_gap_questions": questions}

    monkeypatch.setattr(chat_agent_mod, "run_chat_agent", rich_answer)
    monkeypatch.setattr(settings, "graph_auto_build", False)

    def ask(sid):
        response = c.post("/persona/query", json={
            "query": "带引用的缓存问题", "session_id": sid, "character_id": "jung",
        })
        assert response.status_code == 200
        return [event for event in _sse_events(response) if event["type"] in (
            "trace", "result", "questions", "citations",
        )]

    generated = ask("rich-source")
    replayed = ask("rich-replayed")
    assert len(calls) == 1 and replayed == generated
    answer_event = next(event for event in replayed if event["type"] in ("result", "questions"))
    assert answer_event["type"] == ("questions" if with_questions else "result")
    assert answer_event["graph_used"] is True
    assert [doc["n"] for doc in answer_event["doc_map"]] == ["1", "G1"]
    assert any(event["type"] == "citations" for event in replayed)
    if with_questions:
        assert answer_event["questions"] == questions

    # 修改某个会话的恢复副本，不能污染其他新会话的缓存回放。
    routes._session_store["rich-replayed"]["doc_map"][0]["source"] = "污染来源"
    routes._session_store["rich-replayed"]["route_history"].append("污染路由")
    if with_questions:
        routes._session_store["rich-replayed"]["info_gap_questions"][0]["options"].append("污染选项")
    assert ask("rich-third") == generated
    assert len(calls) == 1


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
