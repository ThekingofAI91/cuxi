"""滚动摘要保留完整输入、合并已有记忆，并执行可配置的字符上限。"""

import asyncio
from types import SimpleNamespace

import pytest

from framework import runtime
from src.core.config import settings
from src.core.session_store import SessionStore


@pytest.fixture()
def session_store(tmp_path, monkeypatch):
    store = SessionStore(str(tmp_path / "sessions.db"))
    monkeypatch.setattr(runtime, "_get_session_store", lambda: store)
    yield store
    store._conn.close()


class FakeLLM:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        content = next(self.contents)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(content=content)


def test_summary_keeps_more_than_300_chars_and_reads_full_answers(session_store, monkeypatch):
    monkeypatch.setattr(settings, "history_summary_max_chars", 1000)
    session_store.set_summary("s", "用户在 2026 年考了 558 分。")
    summary = "用户关注感觉功能与复习专注度。" * 30
    assert 300 < len(summary) < 1000
    llm = FakeLLM([summary])
    llm_kwargs = {}

    def fake_llm(**kwargs):
        llm_kwargs.update(kwargs)
        return llm

    monkeypatch.setattr(runtime, "get_chat_llm", fake_llm)
    long_answer = "背景解释" * 250 + "最后约定在 2026 年 11 月 10 日复盘。"
    asyncio.run(runtime._summarize_old_turns("s", [("我想改善专注度", long_answer)]))

    assert session_store.get("s")["summary"] == summary
    prompt = llm.messages[0][1][1]
    assert "558 分" in prompt
    assert long_answer in prompt
    assert "1000" in llm.messages[0][0][1]
    assert llm_kwargs["max_tokens"] > 600


def test_oversized_summary_is_recompressed_before_saving(session_store, monkeypatch):
    monkeypatch.setattr(settings, "history_summary_max_chars", 600)
    oversized = "背景信息" * 200
    compact = "用户在 2026 年考了 558 分，希望改善专注度。"
    llm = FakeLLM([oversized, compact])
    monkeypatch.setattr(runtime, "get_chat_llm", lambda **kw: llm)

    asyncio.run(runtime._summarize_old_turns("s", [("我考了 558 分", "建议分段复习。")]))

    assert session_store.get("s")["summary"] == compact
    assert len(llm.messages) == 2
    assert oversized in llm.messages[1][1][1]


def test_model_cannot_save_an_over_budget_summary(session_store, monkeypatch):
    monkeypatch.setattr(settings, "history_summary_max_chars", 1000)
    llm = FakeLLM(["用户背景" * 500, "精简背景" * 400])
    monkeypatch.setattr(runtime, "get_chat_llm", lambda **kw: llm)

    asyncio.run(runtime._summarize_old_turns("s", [("用户问题", "回答")]))

    assert len(session_store.get("s")["summary"]) <= 1000


def test_recompression_failure_still_saves_the_generated_summary(session_store, monkeypatch):
    monkeypatch.setattr(settings, "history_summary_max_chars", 1000)
    generated = "用户在 2026 年考了 558 分。" + "用户背景" * 500
    llm = FakeLLM([generated, RuntimeError("二次精简服务不可用")])
    monkeypatch.setattr(runtime, "get_chat_llm", lambda **kw: llm)

    asyncio.run(runtime._summarize_old_turns("s", [("我考了 558 分", "复习建议")]))

    saved = session_store.get("s")["summary"]
    assert saved.startswith("用户在 2026 年考了 558 分。")
    assert len(saved) <= 1000


def test_empty_model_response_does_not_erase_existing_summary(session_store, monkeypatch):
    existing = "用户在 2026 年考了 558 分。"
    session_store.set_summary("s", existing)
    llm = FakeLLM(["", "", ""])
    monkeypatch.setattr(runtime, "get_chat_llm", lambda **kw: llm)

    asyncio.run(runtime._summarize_old_turns("s", [("我想改善专注度", "可以分段复习。")]))

    assert session_store.get("s")["summary"] == existing


def test_overflow_summary_and_recent_turns_are_injected_together(session_store, monkeypatch):
    monkeypatch.setattr(settings, "max_history_turns", 5)
    recent = [(f"问题{i}", f"完整回答{i}") for i in range(1, 6)]
    session_store.set_history("s", recent)
    llm = FakeLLM(["用户最早讨论的问题1已经给出结论。"])
    monkeypatch.setattr(runtime, "get_chat_llm", lambda **kw: llm)

    async def append_and_summarize():
        runtime.append_conversation("s", "问题6", "完整回答6")
        # 等待本轮创建的异步摘要任务，而非依赖固定 sleep。
        tasks = asyncio.all_tasks() - {asyncio.current_task()}
        await asyncio.gather(*tasks)

    asyncio.run(append_and_summarize())
    history = runtime.get_conversation_history("s")
    assert history == recent[1:] + [("问题6", "完整回答6")]
    messages = runtime.build_direct_messages("当前问题", history=history, session_id="s")
    history_text = messages[1][1]
    assert "【早期对话摘要】" in history_text
    assert "最早讨论的问题1" in history_text
    assert all(f"完整回答{i}" in history_text for i in range(2, 7))
    assert messages[-1] == ("user", "当前问题")
