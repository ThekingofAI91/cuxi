"""
争鸣（圆桌会议）测试 —— LangGraph 主持人拓扑。

覆盖八件事：
1. 拓扑契约：host↔speaker 双向，host→END；**子智能体之间没有边**
2. 主持人决策解析：合法 JSON / 包在解释里 / 只有关键字段 / 收敛 / 空串
3. agent 主持完整一场：开场排班 + 逐轮定人 + 请缨理由 + 轮数收敛
4. 用户主持：挂起点推 choice_request 并挂起（不发 end），回填后从 checkpoint 续跑
5. 兜底：用户弃权/给非法 id、主持人指认不存在的人 → 规则轮转兜底，会议不空转
6. 数学收敛优先于模型：配额耗尽 / 轮数到顶 / 预算触顶
7. 角色中途消失：提前收场，不把整场拖垮
8. 纪要：只记账不评分；越界纠正重试，仍越界则零 LLM 降级

LLM 与检索全部打桩，不触真实 API，不碰 ChromaDB。
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from src.core.config import settings
from framework import roundtable as rt
from scenes.persona_chat.skill_loader import PersonaSections


CHARS = {
    "a": ("甲",),
    "b": ("乙",),
    "c": ("丙",),
    "d": ("丁",),
}


def _fake_char(cid: str, name: str):
    return SimpleNamespace(
        id=cid, name=name, avatar="🎭", theme="paper", chroma_collection=f"col_{cid}"
    )


# ============================================================
# 打桩：同一条 llm 工厂按参数分派三种用途
# ============================================================
class LLMState:
    """三类用途的打桩队列。主持裁决已不走模型（改为读投标），故没有 host 队列。"""

    def __init__(self, speeches=None, bids=None,
                 summary='{"clashes": [], "positions": [], "open": []}'):
        self.speeches = list(speeches or [])
        self.bids = list(bids or [])
        self.summary = summary
        self.calls = []          # 按到达顺序记录 bid / speech / clerk

    def kinds(self, kind):
        return self.calls.count(kind)


class ScriptedLLM:
    """roundtable 用一个 llm 工厂产出三种用途，靠温度区分：

    - 投标：temperature=_INTENT_TEMPERATURE（非流式，一个 JSON）
    - 发言：temperature=0.85（流式）
    - 纪要：temperature=0.2（非流式）
    """

    def __init__(self, kind, st):
        self.kind = kind
        self.st = st

    async def ainvoke(self, messages):
        self.st.calls.append(self.kind)
        if self.kind == "bid":
            return SimpleNamespace(
                content=self.st.bids.pop(0) if self.st.bids else '{"want": false, "urgency": 1}'
            )
        return SimpleNamespace(content=self.st.summary)

    async def astream(self, messages):
        self.st.calls.append(self.kind)
        text = self.st.speeches.pop(0) if self.st.speeches else "（无言）"
        for ch in text:
            yield AIMessageChunk(content=ch)


def _install(monkeypatch, st, *, quota=None, max_calls=None, summary_enabled=True):
    def _factory(**kwargs):
        temp = kwargs.get("temperature")
        if temp == rt._INTENT_TEMPERATURE:
            kind = "bid"
        elif temp == 0.85:
            kind = "speech"
        else:
            kind = "clerk"
        return ScriptedLLM(kind, st)

    async def _ainvoke(llm, messages, attempts=1, fallback_llm=None):
        return await llm.ainvoke(messages)

    async def _astream(llm, messages, attempts=3, ttft_timeout=None):
        # 真实 astream_nonempty 对外产出的是**字符串**，不是 AIMessageChunk；
        # 打桩必须照这个口径来，否则调用方 str 拼接会静默失败、退化成占位文本。
        async for chunk in llm.astream(messages):
            yield getattr(chunk, "content", chunk)

    monkeypatch.setattr(rt, "get_chat_llm", _factory)
    monkeypatch.setattr(rt, "ainvoke_nonempty", _ainvoke)
    monkeypatch.setattr(rt, "astream_nonempty", _astream)
    monkeypatch.setattr(rt, "ROUNDTABLE_SPEAKER_PAUSE", 0)
    monkeypatch.setattr(rt, "build_character_prompt",
                        lambda character, query=None, **kw: f"你是{character.name}。")
    monkeypatch.setattr(rt, "resolve_persona_sections", lambda character: PersonaSections(
        role_prompt=f"你是{character.name}。",
    ))
    monkeypatch.setattr(
        rt.persona_chat_config, "characters",
        {cid: _fake_char(cid, n[0]) for cid, n in CHARS.items()},
    )
    if quota is not None:
        monkeypatch.setattr(settings, "roundtable_max_speeches_per_speaker", quota, raising=False)
    if max_calls is not None:
        monkeypatch.setattr(settings, "roundtable_max_llm_calls", max_calls, raising=False)
    if not summary_enabled:
        monkeypatch.setattr(settings, "roundtable_summary_enabled", False, raising=False)
    return st


@pytest.fixture(autouse=True)
def _no_retrieval(monkeypatch):
    """发言前不做真实检索：单测不碰 ChromaDB / Embedder。"""

    async def _fake(collection_name, query):
        return ""

    monkeypatch.setattr(rt, "_retrieve_for_speaker", _fake)
    yield


@pytest.fixture(autouse=True)
def _isolated_ckpt(monkeypatch, tmp_path):
    """每例一个独立的 checkpoint 库，避免用例互相串场。"""
    monkeypatch.setattr(
        settings, "roundtable_checkpoint_db_path", str(tmp_path / "rt_ckpt.db"), raising=False
    )
    yield


def _run(gen):
    """把异步事件流跑完并返回全部事件。"""
    async def _go():
        return [evt async for evt in gen]
    return asyncio.run(_go())


def _types(events):
    return [e["type"] for e in events]


def _of(events, type_):
    return [e for e in events if e["type"] == type_]


# ============================================================
# 1. 拓扑契约
# ============================================================
def test_topology_is_host_centred_and_subagents_have_no_edges():
    """子智能体之间没有边是**结构约束**，不是约定——靠图本身保证。"""
    g = rt.build_roundtable_graph().get_graph()
    edges = {(e.source, e.target) for e in g.edges}

    assert ("__start__", "host") in edges
    assert ("host", "speaker") in edges
    assert ("speaker", "host") in edges
    assert ("host", "__end__") in edges
    # 投标节点只经 Send 到达（动态分发），get_graph 不列出它的静态入边，
    # 所以这里断言节点存在；真正验证「收齐后回 host」的是端到端用例。
    assert "collect_intent" in set(g.nodes)
    # speaker 只有一条出边，且回到 host
    assert sorted(t for s, t in edges if s == "speaker") == ["host"]
    # 不存在 speaker → speaker，也不存在 speaker → collect_intent
    assert not any(s == "speaker" and t != "host" for s, t in edges)
    # 三个节点 = 三种行为（裁决 / 收投标 / 执行发言），不是三个角色
    assert {"host", "speaker", "collect_intent"} <= set(g.nodes)


# ============================================================
# 2. agent 主持：完整一场
# ============================================================
def test_agent_host_runs_a_full_meeting_with_volunteer_reasons(monkeypatch):
    st = _install(
        monkeypatch,
        LLMState(
            # 第 1 轮 a 的 urgency 更高；第 2 轮 a 已在上轮发过言（不参与投标），b 中标
            bids=[
                '{"want": true, "urgency": 5, "target": {"speaker": "乙", "quote": "兴趣不能当饭吃"}, "reason": "适之说不必谈主义，我偏要说清楚。", "move": "rebut"}',
                '{"want": true, "urgency": 3, "target": {"speaker": "甲", "quote": "谈主义才有方向"}, "reason": "他曲解了我的意思。", "move": "clarify"}',
                '{"want": true, "urgency": 4, "target": {"speaker": "甲", "quote": "谈主义才有方向"}, "reason": "他曲解了我的意思，我必须澄清。", "move": "clarify"}',
            ],
            speeches=["甲的开场陈述", "乙的开场陈述", "甲的交锋发言", "乙的交锋发言"],
        ),
    )
    events = _run(rt.stream_roundtable("兴趣能不能当饭吃", ["a", "b"], 2, "t-full", picker="agent"))
    types = _types(events)

    # 开场是本场唯一的排班：每人一次，只有一条 opening 事件
    opening = _of(events, "round")
    assert [r["phase"] for r in opening] == ["opening", "rebuttal", "rebuttal"]
    assert opening[0]["round"] == 0

    # 两次交锋都由主持人定人，且**带请缨理由**
    choices = _of(events, "choice")
    assert [c["character_id"] for c in choices] == ["a", "b"]
    assert all(c["by"] == "bid" for c in choices)
    assert choices[0]["reason"] == "适之说不必谈主义，我偏要说清楚。"
    assert all(c["reason"] for c in choices)

    # 逐 token 流式：token 事件带 character_id，能拼回完整发言
    tokens = _of(events, "token")
    assert tokens and {t["character_id"] for t in tokens} == {"a", "b"}
    for end in _of(events, "speaker_end"):
        joined = "".join(t["content"] for t in tokens if t["character_id"] == end["character_id"])
        assert end["content"] in joined

    ends = _of(events, "speaker_end")
    # 关键：正文必须真的走完了「脚本 → 逐 token 流 → 落库」这条链路，
    # 不能是靠"发言失败"的占位文本蒙混过关（这类静默降级最容易骗过测试）
    assert [e["content"] for e in ends] == [
        "甲的开场陈述", "乙的开场陈述", "甲的交锋发言", "乙的交锋发言"
    ]
    assert all("暂时无法发言" not in e["content"] for e in ends)
    assert len(ends) == 4
    # 主持人裁决不再调模型；第 1 轮 2 人投标、第 2 轮只剩 b 可投
    assert st.kinds("host") == 0
    assert st.kinds("bid") == 3
    assert st.kinds("speech") == 4

    # 轮数到顶即收敛（数学优先于模型）
    conv = _of(events, "converged")
    assert len(conv) == 1 and conv[0]["reason"] == "rounds"
    assert types.index("converged") < types.index("end")


def test_end_payload_carries_the_authoritative_record(monkeypatch):
    _install(
        monkeypatch,
        LLMState(
            # 开场后两人都能投标；urgency 相同，按距上次发言最久再按 id 决胜 → a
            bids=['{"want": true, "urgency": 4, "reason": "有话要说", "move": "rebut"}'] * 2,
            speeches=["甲的开场", "乙的开场", "甲的交锋"],
        ),
    )
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 1, "t-end", picker="agent"))
    end = _of(events, "end")[0]

    assert end["topic"] == "议题"
    assert {c["id"] for c in end["speakers"]} == {"a", "b"}
    assert len(end["transcript"]) == 3
    assert [t["kind"] for t in end["transcript"]] == ["opening", "opening", "rebuttal"]


# ============================================================
# 3. 用户主持：挂起 + 跨请求恢复
# ============================================================
def test_user_host_suspends_then_resumes_across_requests(monkeypatch):
    """挂起靠 checkpoint 落盘，恢复靠 Command(resume)——两次调用互不相干，现场不丢。"""
    _install(
        monkeypatch,
        LLMState(speeches=["甲的开场", "乙的开场", "被点将者的发言"]),
    )
    sid = "t-suspend"

    # ---- 第一次「请求」：开场跑完，到决策点挂起 ----
    first = _run(rt.stream_roundtable("议题", ["a", "b"], 1, sid, picker="user"))
    types = _types(first)

    assert "choice_request" in types
    assert "end" not in types          # 会没开完，不能发 end
    assert "summary_start" not in types
    assert len(_of(first, "speaker_end")) == 2       # 只有开场
    req = _of(first, "choice_request")[0]
    assert req["round"] == 1 and req["total"] == 1
    assert {c["id"] for c in req["candidates"]} == {"a", "b"}
    assert types.index("choice_request") == len(types) - 1   # 挂起事件是这一流的最后一条

    # ---- 挂起状态在库里，能查回来（前端刷新页面靠它找回弹窗）----
    pending = asyncio.run(rt.roundtable_pending_choice(sid))
    assert pending and {c["id"] for c in pending["candidates"]} == {"a", "b"}

    # ---- 第二次「请求」：点将 b，从 checkpoint 续跑 ----
    second = _run(rt.resume_roundtable(sid, "b"))
    stypes = _types(second)

    assert "start" not in stypes                    # 续跑不重发 start
    choices = _of(second, "choice")
    assert len(choices) == 1
    assert choices[0]["by"] == "user" and choices[0]["character_id"] == "b"
    assert choices[0]["reason"] == ""               # 用户点将不是请缨
    assert "end" in stypes

    # 恢复后的 end 里是**整场**的完整记录，包含挂起前那两次开场
    end = _of(second, "end")[0]
    assert [t["character_id"] for t in end["transcript"]] == ["a", "b", "b"]

    # 恢复后不再挂起
    assert asyncio.run(rt.roundtable_pending_choice(sid)) is None


def test_resume_on_finished_meeting_is_rejected(monkeypatch):
    _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场"]))
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 1, "t-done", picker="agent"))
    assert "end" in _types(events)

    resumed = _run(rt.resume_roundtable("t-done", "a"))
    assert _types(resumed) == ["error"]
    assert asyncio.run(rt.roundtable_pending_choice("t-done")) is None


# ============================================================
# 4. 兜底：坏输入不空转
# ============================================================
def test_abstaining_user_falls_back_to_host_rotation(monkeypatch):
    """用户弃权（回填空串）→ 主持人规则兜底挑一位，会议不停摆。"""
    _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场", "兜底者的发言"]))
    sid = "t-abstain"
    _run(rt.stream_roundtable("议题", ["a", "b"], 1, sid, picker="user"))

    second = _run(rt.resume_roundtable(sid, ""))
    choice = _of(second, "choice")[0]
    assert choice["by"] == "host"
    assert choice["note"]                       # 说明是兜底
    assert choice["character_id"] == "a"        # 平票按名单顺序
    assert "end" in _types(second)


def test_invalid_user_choice_falls_back_to_host(monkeypatch):
    _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场", "兜底者的发言"]))
    sid = "t-badcid"
    _run(rt.stream_roundtable("议题", ["a", "b"], 1, sid, picker="user"))

    second = _run(rt.resume_roundtable(sid, "zzz"))
    choice = _of(second, "choice")[0]
    assert choice["by"] == "host" and choice["note"]


def test_bid_from_ineligible_character_is_filtered(monkeypatch):
    """投标里出现不在名单 / 超额度的 id → 被裁决过滤，绝不把它选上来。

    这是「主持人指认陌生人」那条兜底的替代：选人方从主持人换成了投标者，
    但「非法来源不得进入裁决」这条约束没变。
    """
    _install(
        monkeypatch,
        LLMState(
            # 九条投标里混进一条伪造 id；真实两人各投一条，a 的 urgency 更高
            bids=[
                '{"want": true, "urgency": 5, "reason": "我最有话说", "move": "rebut"}',
                '{"want": true, "urgency": 3, "reason": "我也想补一句", "move": "clarify"}',
            ],
            speeches=["甲的开场", "乙的开场", "甲的交锋"],
        ),
    )
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 1, "t-ghost", picker="agent"))
    choice = _of(events, "choice")[0]

    assert choice["character_id"] in {"a", "b"}
    assert choice["by"] == "bid"
    assert choice["reason"]


def test_host_can_declare_convergence(monkeypatch):
    """主持人宣布收敛 → 直接收场，不再选人。"""
    _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场"]))
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 3, "t-hostconv", picker="agent"))
    types = _types(events)

    assert "choice" not in types
    assert _of(events, "converged")[0]["reason"] == "host"
    assert len(_of(events, "speaker_end")) == 2


# ============================================================
# 5. 数学收敛优先于模型
# ============================================================
def test_quota_exhausted_converges_without_spending_host_calls(monkeypatch):
    """quota=1：开场就用完全部额度 → 一个交锋轮都不跑，且一次主持人调用都不花。"""
    st = _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场", "不该出现"]), quota=1)
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 3, "t-quota", picker="agent"))

    assert _of(events, "converged")[0]["reason"] == "quota"
    assert len(_of(events, "speaker_end")) == 2
    assert st.kinds("host") == 0
    assert "choice" not in _types(events)


def test_llm_budget_caps_the_meeting(monkeypatch):
    """没有硬上限时，超预算会以 429 的形式暴露（表现为"某位角色静默失声"）。"""
    st = _install(
        monkeypatch,
        LLMState(speeches=["甲的开场", "乙的开场", "甲的交锋", "不该出现", "不该出现"]),
        max_calls=4,
    )
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 3, "t-budget", picker="agent"))
    types = _types(events)

    # 2 次开场后还剩 2 次额度；而一轮需要 2 次投标 + 1 次发言 = 3 次，
    # 预算不够就不该发起——宁可停在开场，也不能把额度耗在投标上让发言失声。
    assert "budget_exhausted" in types
    assert st.kinds("speech") == 2
    assert st.kinds("bid") == 0
    assert len(_of(events, "speaker_end")) == 2
    assert _of(events, "converged")[0]["reason"] == "budget"
    assert types.index("budget_exhausted") < types.index("end")


def test_rounds_are_clamped(monkeypatch):
    _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场"]))
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 99, "t-clamp", picker="agent"))
    assert _of(events, "round")[0]["total"] == rt._MAX_ROUNDS


# ============================================================
# 7. 角色中途消失
# ============================================================
def test_speaker_node_bails_out_when_the_character_disappears(monkeypatch):
    """自建角色可以在会议进行中被删——提前收场，别把整场拖垮。"""
    _install(monkeypatch, LLMState())
    state = {
        "topic": "议题", "roster": [], "ledger": {},
        "pending_speaker": "ghost", "pending_kind": "rebuttal", "pending_round": 1,
    }
    out = asyncio.run(rt.speaker_node(state))
    assert out["ended"] is True
    assert out["converge_reason"] == "speaker_missing"


# ============================================================
# 8. 纪要：只记账不评分
# ============================================================
def test_summary_verdict_words_are_detected():
    assert rt.summary_verdict_hit({"open": ["孔子更胜一筹"]}) == "更胜"
    assert rt.summary_verdict_hit({"positions": [{"name": "甲", "stance": "甲更有说服力"}]}) == "更有说服力"
    assert rt.summary_verdict_hit({"positions": [{"name": "甲", "stance": "学而优则仕"}]}) == ""
    # 「回避了问题」是描述交锋的合法事实，不算越界——系统不该做的只是下结论
    assert rt.summary_verdict_hit({"clashes": [{"a": "甲", "b": "乙", "point": "甲认为乙在回避"}]}) == ""


def test_normalize_summary_drops_verdict_items_but_keeps_the_rest():
    obj = {
        "clashes": [
            {"a": "孔子", "b": "庄子", "point": "乱世出仕是任事还是共谋"},
            {"a": "甲", "b": "乙", "point": "甲更有道理"},
        ],
        "positions": [
            {"name": "孔子", "stance": "天下无道更当挺身任事"},
            {"name": "庄子", "stance": "庄子略胜一筹"},
        ],
        "open": ["君主不义时出仕算不算共谋", "谁赢了"],
    }
    data = rt.normalize_summary(obj)

    assert len(data["clashes"]) == 1 and "更有道理" not in data["clashes"][0]["point"]
    assert [p["name"] for p in data["positions"]] == ["孔子"]
    assert data["open"] == ["君主不义时出仕算不算共谋"]


def test_summary_degrades_without_extra_calls_when_it_keeps_overstepping(monkeypatch):
    """越界就纠正重试；仍越界则退成零 LLM 的降级版（只列各方主张），绝不空手而归。"""
    st = _install(
        monkeypatch,
        LLMState(
            speeches=["甲的开场立场陈述", "乙的开场立场陈述"],
            summary='{"positions": [{"name": "甲", "stance": "甲更有说服力"}]}',
        ),
    )
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 1, "t-degrade", picker="agent"))
    data = _of(events, "summary")[0]["data"]

    assert data["degraded"] is True
    assert data["positions"]                                   # 降级仍给出立场
    assert st.kinds("clerk") == rt._SUMMARY_ATTEMPTS           # 只重试到上限，不无限烧钱
    assert all("更有说服力" not in p["stance"] for p in data["positions"])


def test_summary_carries_facts_not_scores(monkeypatch):
    _install(
        monkeypatch,
        LLMState(
            speeches=["甲的开场", "乙的开场"],
            summary=(
                '{"clashes": [{"a": "甲", "b": "乙", "point": "在『德可不可教』上撞上"}],'
                ' "positions": [{"name": "甲", "stance": "德可教"}, {"name": "乙", "stance": "德不可教"}],'
                ' "open": ["若无天赋，教还有没有用"]}'
            ),
        ),
    )
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 1, "t-fact", picker="agent"))
    types = _types(events)
    data = _of(events, "summary")[0]["data"]

    assert data["degraded"] is False
    assert data["clashes"][0]["point"] == "在『德可不可教』上撞上"
    assert len(data["positions"]) == 2
    # 发言次数只作为事实随纪要一起给出（配额是公开机制），不是分数
    assert {s["id"] for s in data["speeches"]} == {"a", "b"}
    assert all(s["quota"] == 2 for s in data["speeches"])
    # 纪要必须落在 end 之后：非阻塞旁路，纪要失败也不影响会议结果已交付
    assert types.index("end") < types.index("summary_start") < types.index("summary_end")


def test_summary_can_be_switched_off(monkeypatch):
    _install(
        monkeypatch,
        LLMState(speeches=["甲的开场", "乙的开场"]),
        summary_enabled=False,
    )
    events = _run(rt.stream_roundtable("议题", ["a", "b"], 1, "t-nosum", picker="agent"))
    assert "summary_start" not in _types(events)


# ============================================================
# 兜底：人数与轮数的钳制
# ============================================================
def test_too_few_or_too_many_speakers_is_rejected(monkeypatch):
    _install(monkeypatch, LLMState(speeches=["x"]))
    events = _run(rt.stream_roundtable("议题", ["a"], 1, "t-few"))
    assert _types(events) == ["error"]

    events = _run(rt.stream_roundtable("议题", ["a", "b", "c", "d"], 1, "t-many"))
    assert _types(events) == ["error"]
    assert str(rt.MAX_ROUNDTABLE_CHARS) in events[0]["content"]


def test_reopening_a_suspended_session_is_refused(monkeypatch):
    """同一 session_id 上还挂着未完成的会议时，不能拿新会议覆盖它的现场。"""
    _install(monkeypatch, LLMState(speeches=["甲的开场", "乙的开场"]))
    sid = "t-dup"
    _run(rt.stream_roundtable("议题", ["a", "b"], 1, sid, picker="user"))

    again = _run(rt.stream_roundtable("议题", ["a", "b"], 1, sid, picker="user"))
    assert _types(again) == ["error"]
    assert "还在等待点将" in again[0]["content"]
