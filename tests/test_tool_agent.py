"""
工具化检索（framework/tool_agent）测试。

覆盖四件事：
1. 模型请求调用 search_library → 检索被执行、结果回填、文档写回 state
2. 模型选择直接回答 → 一次检索都不发生
3. 开关 tool_retrieval_enabled 决定 supervisor 走 tool_agent 还是老的 retriever
4. 防死循环：交回空 analysis（上游空响应 / 工具路径降级）时不再被送回 tool_agent

LLM 全部打桩（按脚本产出 chunk），不触真实 API，也不碰 ChromaDB。
"""

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, ToolMessage

from src.core.config import settings
from framework import tool_agent as ta
from framework import supervisor as sup
from framework.tool_agent import (
    SEARCH_TOOL_NAME,
    _collect_tool_calls,
    _finalize_tool_calls,
    build_search_tool,
    render_docs_for_tool,
    tool_agent,
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
        self.bound_tools = []
        self.seen_messages = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def astream(self, messages):
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


@pytest.fixture(autouse=True)
def _use_persona_scene():
    """注入 persona 场景配置，让 build_direct_messages 走真实装配路径"""
    from scenes.persona_chat.config import persona_chat_config

    sup.set_scene_config(persona_chat_config)
    yield


def _base_state(**over):
    state = {
        "query": "你为什么说格物致知要走内心？",
        "session_id": "t-tool",
        "retrieved_docs": [],
        "analysis": "",
        "verification": "",
        "final_answer": "",
        "history": [],
        "route_history": ["supervisor"],
        "character_role_prompt": "你是王阳明本人，说话笃定。",
        "zone": "education",
        "stream_callback": None,
    }
    state.update(over)
    return state


# ============================================================
# 增量累积（真实上游按 chunk 下发，拼错就整个工具调用失效）
# ============================================================

def test_collect_tool_calls_accumulates_incremental_fragments():
    pending: dict = {}
    for chunk in _tool_call_fragments(SEARCH_TOOL_NAME, '{"query": "王阳明 格物"}', "call_abc"):
        _collect_tool_calls(chunk, pending)

    calls = _finalize_tool_calls(pending)
    assert len(calls) == 1
    assert calls[0]["name"] == SEARCH_TOOL_NAME
    assert calls[0]["args"] == {"query": "王阳明 格物"}
    assert calls[0]["id"] == "call_abc"


def test_finalize_drops_calls_without_name_and_survives_bad_json():
    pending = {0: {"name": "", "args": "{}", "id": "x"}}
    assert _finalize_tool_calls(pending) == []

    pending = {0: {"name": SEARCH_TOOL_NAME, "args": "{不是合法 json", "id": "y"}}
    calls = _finalize_tool_calls(pending)
    assert len(calls) == 1 and calls[0]["args"] == {}


# ============================================================
# 情形一：模型请求检索
# ============================================================

def test_model_requests_search_then_answers(monkeypatch):
    called = {}

    async def fake_retrieve(query, history=None, light_retrieval=False, skip_retrieval=False):
        called["query"] = query
        return FAKE_DOCS, False, ""

    monkeypatch.setattr(ta, "retrieve_documents", fake_retrieve)

    llm = ScriptedLLM([
        _tool_call_fragments(SEARCH_TOOL_NAME, '{"query": "王阳明 格物致知"}', "call_1"),
        [_text("格物不是去格外面的物，"), _text("是在心上学做工夫。")],
    ])
    monkeypatch.setattr(ta, "get_chat_llm", lambda **kw: llm)

    tokens: list[str] = []

    async def cb(tok):
        tokens.append(tok)

    state = _base_state(stream_callback=cb)
    out = asyncio.run(tool_agent(state))

    # 检索被真正触发，且用的是模型给的查询词
    assert called["query"] == "王阳明 格物致知"
    # 工具已绑定给模型
    assert [t.name for t in llm.bound_tools] == [SEARCH_TOOL_NAME]
    # 文档写回 state，供 citations/verifier 复用
    assert out["retrieved_docs"] == FAKE_DOCS
    assert out["analysis"] == "格物不是去格外面的物，是在心上学做工夫。"
    assert out["route_history"][-1] == "tool_agent"
    # 第二轮的输入里必须带上 ToolMessage（资料回填），否则模型答不出依据
    second_round_msgs = llm.seen_messages[1]
    tool_msgs = [m for m in second_round_msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_1"
    assert "格物" in tool_msgs[0].content
    # 正文已逐 token 推给前端
    assert "".join(tokens) == out["analysis"]


def test_search_tool_returns_prompt_when_nothing_found(monkeypatch):
    async def empty_retrieve(query, history=None, light_retrieval=False, skip_retrieval=False):
        return [], False, ""

    monkeypatch.setattr(ta, "retrieve_documents", empty_retrieve)
    sink = {}
    tool = build_search_tool(sink, history=[], zone="education")
    result = asyncio.run(tool.ainvoke({"query": "查不到的东西"}))

    assert "没有查到" in result
    assert sink["docs"] == []


def test_search_tool_result_keeps_citation_rule_for_education(monkeypatch):
    """教育区回填必须带【引用标注】——前端引用出处折叠区靠正文 [n] 触发，规则丢了就整块消失"""
    async def fake_retrieve(query, history=None, light_retrieval=False, skip_retrieval=False):
        return FAKE_DOCS, True, ""

    monkeypatch.setattr(ta, "retrieve_documents", fake_retrieve)
    sink = {}
    tool = build_search_tool(sink, history=[], zone="education")
    result = asyncio.run(tool.ainvoke({"query": "格物"}))
    assert "【引用标注】" in result
    assert "[1] 来源" in result
    assert sink["graph_used"] is True

    # 娱乐区不能出现检索痕迹与引用标注
    light = render_docs_for_tool(FAKE_DOCS, zone="entertainment")
    assert "【引用标注】" not in light
    assert "来源" not in light


# ============================================================
# 情形二：模型选择直接回答（寒暄不再无脑走检索）
# ============================================================

def test_model_answers_directly_without_search(monkeypatch):
    called = {"n": 0}

    async def fake_retrieve(query, history=None, light_retrieval=False, skip_retrieval=False):
        called["n"] += 1
        return FAKE_DOCS, False, ""

    monkeypatch.setattr(ta, "retrieve_documents", fake_retrieve)

    llm = ScriptedLLM([[_text("你来了。"), _text("山里的夜还凉。")]])
    monkeypatch.setattr(ta, "get_chat_llm", lambda **kw: llm)

    out = asyncio.run(tool_agent(_base_state(query="你好")))

    assert called["n"] == 0
    assert out["retrieved_docs"] == []
    assert out["analysis"] == "你来了。山里的夜还凉。"


def test_tool_failure_degrades_without_breaking_answer(monkeypatch):
    def boom(**kw):
        raise RuntimeError("模型不可用")

    monkeypatch.setattr(ta, "get_chat_llm", boom)
    out = asyncio.run(tool_agent(_base_state()))

    # 不抛异常，交空 analysis 让 supervisor 退回规则路径（不会被再送回 tool_agent）
    assert out["analysis"] == ""
    assert out["error"]


# ============================================================
# 情形三：开关决定路由
# ============================================================

def test_supervisor_routes_to_tool_agent_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "tool_retrieval_enabled", True)
    monkeypatch.setattr(settings, "light_chat_enabled", False)

    out = asyncio.run(sup.supervisor_node(_base_state()))
    assert out["next_agent"] == "tool_agent"


def test_supervisor_keeps_old_retriever_path_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "tool_retrieval_enabled", False)
    monkeypatch.setattr(settings, "light_chat_enabled", False)

    out = asyncio.run(sup.supervisor_node(_base_state()))
    assert out["next_agent"] == "retriever"
    assert "tool_agent" not in out["route_history"]


def test_entertainment_zone_never_uses_tool_agent(monkeypatch):
    """娱乐区检索是 23ms 软背景，工具化要给每条消息加一次 LLM 往返，必须挡住"""
    monkeypatch.setattr(settings, "tool_retrieval_enabled", True)
    monkeypatch.setattr(settings, "light_chat_enabled", False)

    out = asyncio.run(sup.supervisor_node(_base_state(zone="entertainment")))
    assert out["next_agent"] == "retriever"


# ============================================================
# 图结构：节点与边都注册了
# ============================================================

def test_graph_contains_tool_agent_node():
    graph = sup.build_graph()
    nodes = set(getattr(graph, "nodes", {}) or {})
    assert "tool_agent" in nodes
    assert "retriever" in nodes  # 旧路径同时保留


# ============================================================
# 防死循环：tool_agent 交回空 analysis（上游空响应 / 工具路径降级）
# ============================================================
# 用户实测 bug：supervisor 与 tool_agent 来回 7 轮、单次耗时 61 秒，
# 直到 supervisor_count>6 的强制结束才收手。下面三个测试把这个环钉死。


def test_supervisor_never_re_enters_tool_agent_without_analysis(monkeypatch):
    """工具化跑过但没吐出正文时，supervisor 必须换目标，不能回送 tool_agent"""
    monkeypatch.setattr(settings, "tool_retrieval_enabled", True)
    monkeypatch.setattr(settings, "light_chat_enabled", False)

    after_tool = ["supervisor", "tool_agent", "supervisor"]

    # 查到过资料 → 走 analyzer 复用（analyzer 自己读 retrieved_docs，不再重复检索）
    out = asyncio.run(sup.supervisor_node(_base_state(
        analysis="", retrieved_docs=FAKE_DOCS, route_history=after_tool,
    )))
    assert out["next_agent"] == "analyzer"

    # 一条资料都没查到 → 退回规则检索路径
    out = asyncio.run(sup.supervisor_node(_base_state(
        analysis="", retrieved_docs=[], route_history=after_tool,
    )))
    assert out["next_agent"] == "retriever"

    # 两种情况都必须继续推进 route_history（图靠它判定轮次）
    assert out["route_history"] == after_tool + ["supervisor"]


def test_graph_terminates_when_tool_agent_returns_empty(monkeypatch):
    """端到端（LLM 全打桩）：tool_agent 交回空 analysis 后必须几步内收敛

    修复前这条链会一直 supervisor→tool_agent 互送，直到 6 次上限。
    """
    monkeypatch.setattr(settings, "tool_retrieval_enabled", True)
    monkeypatch.setattr(settings, "light_chat_enabled", False)

    calls = {"tool": 0, "retriever": 0, "analyzer": 0}

    async def fake_tool(state):
        calls["tool"] += 1
        return {
            "analysis": "",
            "retrieved_docs": [],
            "route_history": state["route_history"] + ["tool_agent"],
        }

    async def fake_retriever(state):
        calls["retriever"] += 1
        return {
            "retrieved_docs": FAKE_DOCS,
            "route_history": state["route_history"] + ["retrieval_agent"],
        }

    async def fake_analyzer(state):
        calls["analyzer"] += 1
        return {
            "analysis": "格物不是去格外面的物，是在心上学做工夫。",
            "route_history": state["route_history"] + ["analysis_agent"],
        }

    monkeypatch.setattr(ta, "tool_agent", fake_tool)
    monkeypatch.setattr(sup, "retrieval_agent", fake_retriever)
    monkeypatch.setattr(sup, "analysis_agent", fake_analyzer)

    graph = sup.build_graph()

    async def run():
        final: dict = {}
        async for update in graph.astream(
            _base_state(route_history=[]),
            config={"recursion_limit": 25},
            stream_mode="updates",
        ):
            for _node, out in update.items():
                final.update(out)
        return final

    final = asyncio.run(run())

    assert calls["tool"] == 1  # 只进一次，不再被回送
    assert final["final_answer"] == "格物不是去格外面的物，是在心上学做工夫。"


def test_supervisor_routes_to_tool_agent_only_once_across_rounds(monkeypatch):
    """分析一直为空也不能再选 tool_agent（退化路径或安全网强制结束都算通过）"""
    monkeypatch.setattr(settings, "tool_retrieval_enabled", True)
    monkeypatch.setattr(settings, "light_chat_enabled", False)

    rh = ["supervisor", "tool_agent", "supervisor"]
    for _ in range(6):
        out = asyncio.run(sup.supervisor_node(
            _base_state(analysis="", retrieved_docs=[], route_history=rh)
        ))
        assert out["next_agent"] != "tool_agent"
        # analyzer / retriever 是退化路径；__end__ 是 supervisor 次数上限的安全网
        assert out["next_agent"] in {"analyzer", "retriever", "__end__"}
        rh = out["route_history"]


# ============================================================
# 空轮重试：上游吐空壳时不能把空 analysis 交出去
# ============================================================

def test_empty_round_is_retried_but_tool_only_round_is_not(monkeypatch):
    """判据是"正文和工具调用都没有"才算空轮——只有工具调用的轮次不能重试"""
    async def fake_retrieve(query, history=None, light_retrieval=False, skip_retrieval=False):
        return FAKE_DOCS, False, ""

    monkeypatch.setattr(ta, "retrieve_documents", fake_retrieve)

    # 第 1 轮吐空壳（重试），第 2 轮出正文
    llm = ScriptedLLM([[], [_text("心即理。"), _text("心外无理。")]])
    monkeypatch.setattr(ta, "get_chat_llm", lambda **kw: llm)

    out = asyncio.run(tool_agent(_base_state()))
    assert out["analysis"] == "心即理。心外无理。"
    assert len(llm.seen_messages) > 1  # 空轮被重试过

    # 对照：整轮没有正文、只有一个工具调用 —— 这是工具轮的常态，不能被当空壳重试
    llm2 = ScriptedLLM([
        _tool_call_fragments(SEARCH_TOOL_NAME, '{"query": "格物"}', "call_1"),
        [_text("格物在心。")],
    ])
    monkeypatch.setattr(ta, "get_chat_llm", lambda **kw: llm2)
    out2 = asyncio.run(tool_agent(_base_state()))

    # 第 1 轮（无正文 + 1 个工具调用）+ 第 2 轮（出正文）= 恰好 2 次流式调用，没多试
    assert len(llm2.seen_messages) == 2
    assert out2["analysis"] == "格物在心。"
    assert out2["retrieved_docs"] == FAKE_DOCS
