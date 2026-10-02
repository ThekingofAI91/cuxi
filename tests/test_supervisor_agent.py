"""
一对一编排（framework/supervisor_agent）测试。

覆盖五件事：
1. 增量累积：真实上游按 chunk 下发工具调用，拼错就整个工具调用失效
2. strong 双档：true → 全管线 + 引用标注；false → 轻量 top-3 + 记忆式话术
3. 模型选择直接回答 → 一次检索都不发生
4. 降级与兜底：工具执行失败不炸链路；模型空手但资料已查到时不白花钱
5. 空轮重试：正文和工具调用都没有才算空轮；纯工具轮不能被误判成空轮

LLM 全部打桩（按脚本产出 chunk），检索整份打桩，不触真实 API、不碰 ChromaDB。
"""

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, ToolMessage

from framework import runtime as rt
from framework import supervisor_agent as sa
from framework.supervisor_agent import (
    SEARCH_TOOL_NAME,
    _collect_tool_calls,
    _finalize_tool_calls,
    build_search_tool,
)


FAKE_DOCS = [
    Document(
        page_content="过去从外物求天理，是舍本逐末了。天下之物本无可格者，其格物之功只在身心上做。",
        metadata={"source": "王阳明大传.pdf", "heading": "卷二 格物"},
    ),
    Document(
        page_content="有路过的人死了，暴尸荒野，我和童仆去挖坑埋了他们。",
        metadata={"source": "此心光明.pdf", "heading": "卷三"},
    ),
]


class ScriptedLLM:
    """按脚本逐轮产出 chunk 的假模型；bind_tools 原样返回自身以便拿到同一实例断言"""

    def __init__(self, rounds):
        self._rounds = rounds
        self._i = 0
        self.calls = 0
        self.bound_tools = []
        self.seen_messages = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def astream(self, messages):
        self.calls += 1
        self.seen_messages.append(messages)
        idx = min(self._i, len(self._rounds) - 1)
        self._i += 1
        for chunk in self._rounds[idx]:
            yield chunk


def _text(t: str) -> AIMessageChunk:
    return AIMessageChunk(content=t)


def _tool_call_fragments(name: str, args: str, call_id: str) -> list:
    """把一次工具调用拆成多个增量 chunk（模拟真实增量下发）"""
    return [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": name, "args": args[: len(args) // 2], "id": call_id, "index": 0},
        ]),
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": "", "args": args[len(args) // 2:], "id": None, "index": 0},
        ]),
    ]


def _search_round(args: str) -> list:
    return _tool_call_fragments(SEARCH_TOOL_NAME, args, "call_1")


@pytest.fixture(autouse=True)
def _use_persona_scene():
    """注入 persona 场景配置，让 build_direct_messages 走真实装配路径"""
    from scenes.persona_chat.config import persona_chat_config

    rt.set_scene_config(persona_chat_config)
    yield


def _base_state(**over):
    state = {
        "query": "你为什么说格物致知要走内心？",
        "session_id": "t-sup-agent",
        "retrieved_docs": [],
        "analysis": "",
        "verification": "",
        "final_answer": "",
        "history": [],
        "route_history": [],
        "character_role_prompt": "你是王阳明本人，说话笃定。",
        "zone": "education",
        "sampling": None,
        "post_history_directive": None,
        "user_memory": None,
        "stream_callback": None,
    }
    state.update(over)
    return state


def _patch_llm(monkeypatch, llm):
    monkeypatch.setattr(sa, "get_chat_llm", lambda **kw: llm)
    return llm


def _patch_retrieval(monkeypatch, docs=None, light_capture=None, boom=False):
    async def fake_retrieve(query, history=None, light_retrieval=False, skip_retrieval=False):
        if light_capture is not None:
            light_capture["light"] = light_retrieval
        if boom:
            raise RuntimeError("检索后端炸了")
        return list(docs or []), False, ""

    monkeypatch.setattr(sa, "retrieve_documents", fake_retrieve)
    return fake_retrieve


# ============================================================
# 一、增量累积（真实上游按 chunk 下发，拼错就整个工具调用失效）
# ============================================================

def test_collect_tool_calls_accumulates_incremental_fragments():
    pending: dict = {}
    for chunk in _search_round('{"query": "格物致知", "strong": true}'):
        _collect_tool_calls(chunk, pending)
    calls = _finalize_tool_calls(pending)
    assert len(calls) == 1
    assert calls[0]["name"] == SEARCH_TOOL_NAME
    assert calls[0]["args"] == {"query": "格物致知", "strong": True}


def test_finalize_drops_calls_without_name_and_survives_bad_json():
    pending = {
        0: {"name": "", "args": "{}", "id": "x"},          # 无函数名 → 丢弃
        1: {"name": SEARCH_TOOL_NAME, "args": "{坏", "id": "y"},  # JSON 坏 → 空参
    }
    calls = _finalize_tool_calls(pending)
    assert len(calls) == 1
    assert calls[0]["args"] == {}


# ============================================================
# 二、strong 双档
# ============================================================

def test_strong_true_runs_full_pipeline_with_citation_rule(monkeypatch):
    """strong=true → light_retrieval=False；工具返回值带【引用标注】规则"""
    cap: dict = {}
    _patch_retrieval(monkeypatch, FAKE_DOCS, light_capture=cap)
    llm = _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "格物致知", "strong": true}'),
        [_text("格物致知的关键在于向内求。")],
    ]))

    out = asyncio.run(sa.run_supervisor_agent(_base_state()))

    assert cap["light"] is False
    assert out["final_answer"] == "格物致知的关键在于向内求。"
    assert len(out["retrieved_docs"]) == 2
    assert out["route_history"] == ["supervisor", "retrieval_agent"]
    # 回填给模型的 ToolMessage 必须带引用标注规则，否则正文不会标 [n]、
    # 前端的引用出处折叠区（判据是正文含 [n]）会整块消失
    tool_msgs = [m for m in llm.seen_messages[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "【引用标注】" in tool_msgs[0].content


def test_strong_false_runs_light_pipeline_with_memory_wording(monkeypatch):
    """strong=false → light_retrieval=True；工具返回值走"你记得的事"话术、不带来源"""
    cap: dict = {}
    _patch_retrieval(monkeypatch, FAKE_DOCS, light_capture=cap)
    llm = _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "近况", "strong": false}'),
        [_text("最近还好。")],
    ]))

    out = asyncio.run(sa.run_supervisor_agent(
        _base_state(zone="entertainment", character_role_prompt="你是峰哥。")
    ))

    assert cap["light"] is True
    tool_msgs = [m for m in llm.seen_messages[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].content.startswith("【你记得的事】")
    assert "王阳明大传.pdf" not in tool_msgs[0].content  # 不能暴露来源痕迹
    assert "【引用标注】" not in tool_msgs[0].content


def test_strong_defaults_to_true_when_model_omits_it(monkeypatch):
    """模型漏填 strong → 按强检索处理（宁可多花，不可不可溯源）"""
    cap: dict = {}
    _patch_retrieval(monkeypatch, FAKE_DOCS, light_capture=cap)
    _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "格物致知"}'),
        [_text("回答")],
    ]))

    asyncio.run(sa.run_supervisor_agent(_base_state()))
    assert cap["light"] is False


@pytest.mark.parametrize("zone, hint", [
    ("education", "教育成长区"),
    ("entertainment", "娱乐区"),
])
def test_tool_description_carries_zone_hint(zone, hint):
    """合并路由后，靠工具描述里的分区倾向把两区的默认力度分开"""
    tool = build_search_tool({}, zone=zone)
    assert hint in tool.description


# ============================================================
# 三、模型选择直接回答（寒暄不该无脑走检索）
# ============================================================

def test_model_answers_directly_without_search(monkeypatch):
    triggered = {"n": 0}

    async def fake_retrieve(*a, **k):
        triggered["n"] += 1
        return [], False, ""

    monkeypatch.setattr(sa, "retrieve_documents", fake_retrieve)
    _patch_llm(monkeypatch, ScriptedLLM([[_text("你好。坐下吧，不必拘谨。")]]))

    out = asyncio.run(sa.run_supervisor_agent(_base_state(query="你好")))

    assert triggered["n"] == 0
    assert out["retrieved_docs"] == []
    assert out["route_history"] == ["supervisor"]
    assert out["final_answer"] == "你好。坐下吧，不必拘谨。"


# ============================================================
# 四、降级与兜底
# ============================================================

def test_tool_failure_does_not_break_the_answer(monkeypatch):
    """检索后端抛异常 → 降级成一句人话回填，模型照常作答，链路不断"""
    _patch_retrieval(monkeypatch, boom=True)
    llm = _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "格物致知"}'),
        [_text("凭我自己的记忆，格物在心上做。")],
    ]))

    out = asyncio.run(sa.run_supervisor_agent(_base_state()))

    assert out["final_answer"] == "凭我自己的记忆，格物在心上做。"
    tool_msgs = [m for m in llm.seen_messages[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "检索失败" in tool_msgs[0].content


def test_docs_are_rescued_when_model_returns_nothing(monkeypatch):
    """模型空手但资料已查到 → 带资料重生成一次，检索成本不白花"""
    _patch_retrieval(monkeypatch, FAKE_DOCS)
    _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "格物致知"}'),
        [],  # 第二轮整轮无产出
    ]))

    captured: dict = {}

    async def fake_direct(query, history=None, character_role_prompt="",
                          stream_callback=None, **kw):
        captured["context"] = kw.get("context")
        return "兜底生成的角色回答"

    monkeypatch.setattr(rt, "_generate_direct_response", fake_direct)

    out = asyncio.run(sa.run_supervisor_agent(_base_state()))

    assert out["final_answer"] == "兜底生成的角色回答"
    assert captured["context"] and "王阳明大传.pdf" in captured["context"]


# ============================================================
# 五、空轮重试
# ============================================================

def test_empty_round_is_retried(monkeypatch):
    """整轮既无正文也无工具请求 → 重试，拿到内容即返回"""
    _patch_retrieval(monkeypatch, [])
    llm = _patch_llm(monkeypatch, ScriptedLLM([
        [],                          # 第 1 次尝试：空壳
        [_text("重试后拿到的回答")],   # 第 2 次尝试：正常
    ]))

    out = asyncio.run(sa.run_supervisor_agent(_base_state()))

    assert llm.calls == 2
    assert out["final_answer"] == "重试后拿到的回答"


def test_tool_only_round_is_not_treated_as_empty(monkeypatch):
    """纯工具轮（无正文但请求了工具）不是空壳，绝不能被重试掉工具调用"""
    _patch_retrieval(monkeypatch, FAKE_DOCS)
    llm = _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "格物致知"}'),
        [_text("答案")],
    ]))

    asyncio.run(sa.run_supervisor_agent(_base_state()))

    # 两轮各调一次 astream；若把纯工具轮误判成空轮，这里会变成 3 次
    assert llm.calls == 2


# ============================================================
# 六、阶段事件（前端"判断该怎么回应""翻查原著"靠它）
# ============================================================

def test_stage_events_emit_supervisor_then_retrieval(monkeypatch):
    _patch_retrieval(monkeypatch, FAKE_DOCS)
    _patch_llm(monkeypatch, ScriptedLLM([
        _search_round('{"query": "格物致知"}'),
        [_text("答案")],
    ]))
    stages: list[str] = []

    async def on_stage(name):
        stages.append(name)

    asyncio.run(sa.run_supervisor_agent(_base_state(), on_stage=on_stage))

    assert stages[0] == "supervisor"
    assert stages[-1] == "retrieval_agent"


def test_no_retrieval_stage_when_model_does_not_search(monkeypatch):
    _patch_retrieval(monkeypatch, [])
    _patch_llm(monkeypatch, ScriptedLLM([[_text("直接回答")]]))
    stages: list[str] = []

    async def on_stage(name):
        stages.append(name)

    asyncio.run(sa.run_supervisor_agent(_base_state(query="你好"), on_stage=on_stage))

    assert stages == ["supervisor"]
