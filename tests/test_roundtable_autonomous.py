"""
争鸣（圆桌会议自主发言）测试。

覆盖七件事：
1. 意愿判断解析：合法 JSON / 包在解释里的 JSON / 答"是·否" / 解析不出来（一律不举手）
2. 配额闸门：配额耗尽不再主动举牌，且 eligibility 过滤在编排层兜底
3. 主持人代班：多人争抢时选「本场说得最少」的那位，平票保持原顺序
4. 点将双向交互：contention 挂起 → 用户提交选择 → 唤醒并采纳（by="user"）
5. 收敛：无人举手是合法终止条件；首轮无人交锋时主持人兜底指定一次
6. 预算硬上限：LLM 调用触顶即收束，不会把 RPM 打爆
7. 纪要：只记账不评分；命中"宣布胜负"的词就纠正重试，仍越界则零 LLM 降级

LLM 与检索全部打桩，不触真实 API，也不碰 ChromaDB。
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from src.core.config import settings
from framework import roundtable as rt


# ============================================================
# 打桩：按调用参数分派用途的假模型
# ============================================================
class LLMState:
    def __init__(self, intents=None, speeches=None, summary='{"clashes": [], "positions": [], "open": []}'):
        self.intents = list(intents or [])
        self.speeches = list(speeches or [])
        self.summary = summary
        self.calls = []           # 按到达顺序记录 intent / speech / clerk

    def kinds(self):
        return list(self.calls)


class ScriptedLLM:
    """roundtable 用同一个 llm 工厂产出三种用途，靠参数区分：

    - 意愿判断：temperature=0.2 且 max_tokens 小（流式，凑齐 JSON 即断）
    - 发言：temperature=0.85（流式）
    - 纪要：temperature=0.2 且 max_tokens 大（非流式）
    """

    def __init__(self, kind, st):
        self.kind = kind
        self.st = st

    async def ainvoke(self, messages):
        self.st.calls.append(self.kind)
        if self.kind == "intent":
            content = self.st.intents.pop(0) if self.st.intents else '{"speak": false}'
        else:
            content = self.st.summary
        return SimpleNamespace(content=content)

    async def astream(self, messages):
        self.st.calls.append(self.kind)
        if self.kind == "intent":
            # 意愿判断走流式：一边收一边判，凑齐 JSON 就断开（见 _read_intent_json）
            text = self.st.intents.pop(0) if self.st.intents else '{"speak": false}'
        else:
            text = self.st.speeches.pop(0) if self.st.speeches else "（无言）"
        for ch in text:
            yield AIMessageChunk(content=ch)


def _install(monkeypatch, st, *, quota=None, max_calls=None, summary_enabled=True):
    def _factory(**kwargs):
        if kwargs.get("max_tokens") == rt._INTENT_MAX_TOKENS:
            kind = "intent"
        elif kwargs.get("temperature") == 0.85:
            kind = "speech"
        else:
            kind = "clerk"
        return ScriptedLLM(kind, st)

    async def _ainvoke(llm, messages, attempts=1, fallback_llm=None):
        return await llm.ainvoke(messages)

    monkeypatch.setattr(rt, "get_chat_llm", _factory)
    monkeypatch.setattr(rt, "ainvoke_nonempty", _ainvoke)
    monkeypatch.setattr(rt, "ROUNDTABLE_SPEAKER_PAUSE", 0)
    monkeypatch.setattr(rt, "build_character_prompt", lambda character, query=None, **kw: f"你是{character.name}。")
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

    async def _fake(collection_name, query, top_k=4):
        return ""

    monkeypatch.setattr(rt, "_retrieve_for_speaker", _fake)
    yield


@pytest.fixture(autouse=True)
def _clean_registry():
    rt._RT_SESSIONS.clear()
    yield
    rt._RT_SESSIONS.clear()


def _run(gen, on_event=None):
    """把异步事件流跑完并返回全部事件；on_event 在收到每条事件后同步调用。"""
    async def _go():
        events = []
        async for evt in gen:
            events.append(evt)
            if on_event:
                on_event(evt)
        return events
    return asyncio.run(_go())


def _types(events):
    return [e["type"] for e in events]


def _speakers(*ids):
    return list(ids)


# ============================================================
# 1. 意愿判断解析
# ============================================================
def test_parse_intent_accepts_plain_json():
    assert rt.parse_intent('{"speak": true, "reason": "反驳孔子：德不可教"}') == (True, "反驳孔子：德不可教")
    assert rt.parse_intent('{"speak": false, "reason": "无靶子"}') == (False, "")


def test_parse_intent_accepts_json_wrapped_in_prose_or_fence():
    raw = '我的判断如下：\n```json\n{"speak": true, "reason": "他在曲解我"}\n```\n以上。'
    assert rt.parse_intent(raw) == (True, "他在曲解我")


def test_parse_intent_falls_back_to_yes_no_and_defaults_to_silence():
    assert rt.parse_intent("speak: 是")[0] is True
    assert rt.parse_intent("speak: 否")[0] is False
    # 解析不出来一律按不举手处理——宁可不说话，也不要无靶子地插话
    assert rt.parse_intent("我觉得我应该发言，因为……") == (False, "")
    assert rt.parse_intent("") == (False, "")


def test_intent_reason_is_truncated():
    long_reason = "要反驳" + "很长的理由" * 40
    speak, reason = rt.parse_intent('{"speak": true, "reason": "%s"}' % long_reason)
    assert speak is True
    assert len(reason) <= rt._INTENT_REASON_CHARS


def test_intent_is_read_as_a_stream_and_stops_at_the_first_complete_json(monkeypatch):
    """意愿判断必须走流式并尽早断开。

    这不是为了快，是为了**拿得到正文**：这个模型是推理模型，reasoning 与正文共用
    max_tokens；非流式时上游要把整段生成完才返回，预算被 reasoning 吃光就返回空正文
    （实测 max_tokens=96/300/700 正文全空，且不报错）。流式能边收边判、凑齐就断。
    """
    from scenes.persona_chat.config import persona_chat_config

    st = _install(monkeypatch, LLMState(intents=[
        '{"speak": true, "reason": "反驳甲"}',
        '{"speak": true, "reason": "这条不该被消费"}',
    ]))
    session = rt.RoundtableSession(session_id="s-intent", topic="议题")
    session.ledger["jung"] = rt.SpeakerLedger("jung", "荣格", quota=2)

    out = asyncio.run(rt.poll_speaker_intent(session, persona_chat_config.characters["jung"]))

    assert out["speak"] is True and out["reason"] == "反驳甲"
    assert st.calls == ["intent"]
    assert st.intents == ['{"speak": true, "reason": "这条不该被消费"}']  # 凑齐即断，没多读
    assert session.llm_calls == 1


# ============================================================
# 2. 配额
# ============================================================
def test_ledger_remaining_never_negative():
    led = rt.SpeakerLedger(character_id="jung", name="荣格", quota=2)
    assert led.remaining == 2
    led.speeches = 2
    assert led.remaining == 0
    led.speeches = 5
    assert led.remaining == 0


def test_quota_exhausted_means_no_more_intent_polling(monkeypatch):
    """quota=1：开场就用完全部额度 → 一个交锋轮都不跑，直接按配额收敛。"""
    st = _install(monkeypatch, LLMState(speeches=["开场一", "开场二", "不该出现"]), quota=1)
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 3, "t-quota"))

    assert "converged" in _types(events)
    assert [e for e in events if e["type"] == "converged"][0]["reason"] == "quota"
    # 配额耗尽后连"要不要发言"都不再问——省下的正是最贵的那部分调用
    assert "intent" not in _types(events)
    assert st.calls.count("intent") == 0


# ============================================================
# 3. 主持人代班
# ============================================================
def test_moderator_picks_the_least_vocal_and_keeps_order_on_tie():
    session = rt.RoundtableSession(session_id="s", topic="t")
    session.ledger = {
        "a": rt.SpeakerLedger("a", "甲", quota=3, speeches=2),
        "b": rt.SpeakerLedger("b", "乙", quota=3, speeches=0),
        "c": rt.SpeakerLedger("c", "丙", quota=3, speeches=1),
    }
    assert rt.moderator_pick(session, ["a", "b", "c"]) == "b"
    # 平票时保持传入顺序（也就是发言顺序的原样）
    session.ledger["c"].speeches = 0
    assert rt.moderator_pick(session, ["a", "b", "c"]) == "b"
    assert rt.moderator_pick(session, ["c", "b"]) == "c"


# ============================================================
# 4. 点将双向交互
# ============================================================
def test_user_pick_wakes_the_waiting_stream(monkeypatch):
    """contention 挂起 → 用户提交选择 → 采纳，by="user"。"""
    st = _install(
        monkeypatch,
        LLMState(
            intents=['{"speak": true, "reason": "反驳他"}', '{"speak": true, "reason": "澄清我"}'],
            speeches=["甲的开场", "乙的开场", "由用户点将选中的发言"],
        ),
    )
    picked = {}

    def _on_event(evt):
        if evt["type"] == "contention":
            picked["ids"] = [c["id"] for c in evt["candidates"]]
            # 模拟用户点了乙：会议正处于挂起等待，这一下会把它唤醒
            assert rt.submit_roundtable_choice("t-pick", evt["candidates"][1]["id"]) is True

    # picker=user 才会真的弹窗等待；给足超时，但仍靠提交立即返回
    monkeypatch.setattr(settings, "roundtable_pick_timeout", 20, raising=False)
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 1, "t-pick", picker="user"), _on_event)

    assert picked["ids"] == ["jung", "adler"]
    choices = [e for e in events if e["type"] == "choice"]
    assert len(choices) == 1
    assert choices[0]["by"] == "user"
    assert choices[0]["character_id"] == "adler"


def test_agent_mode_skips_the_popup_and_delegates(monkeypatch):
    """picker=agent：不弹窗，直接由主持人裁决。"""
    _install(
        monkeypatch,
        LLMState(
            intents=['{"speak": true}', '{"speak": true}'],
            speeches=["甲的开场", "乙的开场", "主持人安排的发言"],
        ),
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 1, "t-agent", picker="agent"))

    assert "contention" not in _types(events)
    choices = [e for e in events if e["type"] == "choice"]
    assert len(choices) == 1 and choices[0]["by"] == "moderator"


def test_choice_for_unknown_session_is_rejected():
    assert rt.submit_roundtable_choice("nope", "jung") is False


def test_prepare_choice_wait_does_not_swallow_an_early_pick():
    """竞态回归锁：先进入等待态、再推 contention，用户秒选的结果不能被 clear 抹掉。"""
    session = rt.RoundtableSession(session_id="s", topic="t")
    rt.register_roundtable_session(session)

    rt._prepare_choice_wait(session)
    assert rt.submit_roundtable_choice("s", "jung") is True   # 事件刚推出去用户就点了
    assert asyncio.run(rt._wait_user_choice(session, 1.0)) == "jung"


# ============================================================
# 5. 收敛与兜底
# ============================================================
def test_converged_when_nobody_raises_after_the_first_rebuttal(monkeypatch):
    """无人举手是合法的终止条件（讨论自然收敛），不该被当成故障。"""
    st = _install(
        monkeypatch,
        LLMState(intents=['{"speak": false}'] * 8, speeches=["甲的开场", "乙的开场", "主持人指定的一轮"]),
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 3, "t-conv"))
    types = _types(events)

    assert types.count("converged") == 1
    assert [e for e in events if e["type"] == "converged"][0]["reason"] == "no_raiser"
    # 首轮无人交锋时主持人兜底指定一次，保证至少有一次真实对撞；第二轮不再兜底
    speeches = [e for e in events if e["type"] == "speaker_end"]
    assert len(speeches) == 3
    # 第 1 轮问了 2 位；那次兜底发言吃掉了被指定者最后的额度，
    # 于是第 2 轮只有 1 位还在举牌资格内 → 3 次意愿判断。配额就是这样自动收窄讨论的。
    assert st.calls.count("intent") == 3
    forced = [e for e in events if e["type"] == "choice"][0]
    assert forced["by"] == "moderator" and forced["note"]


def test_single_raiser_speaks_without_asking_the_user(monkeypatch):
    st = _install(
        monkeypatch,
        LLMState(
            intents=['{"speak": true, "reason": "只有一个靶子"}', '{"speak": false}'],
            speeches=["甲的开场", "乙的开场", "唯一的抢答者"],
        ),
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 1, "t-one", picker="user"))
    choices = [e for e in events if e["type"] == "choice"]

    assert len(choices) == 1
    assert choices[0]["by"] == "auto"        # 只有一个人举手，不必打扰用户
    assert "contention" not in _types(events)
    # 想说话的那句理由被透传出来，正是用户在点将时需要的判断依据
    raisers = [e for e in events if e["type"] == "intent" and e["speak"]]
    assert len(raisers) == 1 and raisers[0]["reason"] == "只有一个靶子"


# ============================================================
# 6. 预算硬上限
# ============================================================
def test_llm_call_budget_stops_the_meeting(monkeypatch):
    """没有硬上限时，自主发言会以 429 的形式暴露（表现为"某位角色静默失声"）。"""
    st = _install(
        monkeypatch,
        LLMState(
            intents=['{"speak": true}'] * 8,
            speeches=["甲的开场", "乙的开场", "甲的反驳", "乙的反驳", "不该出现"],
        ),
        max_calls=5,
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 3, "t-budget"))
    types = _types(events)

    assert "budget_exhausted" in types
    # 2 开场 + 2 意愿 + 1 发言 = 5，到顶；后面的轮次不再发起任何调用
    assert st.calls.count("speech") == 3
    assert len([e for e in events if e["type"] == "speaker_end"]) == 3
    assert types.index("budget_exhausted") < types.index("end")


# ============================================================
# 7. 纪要：只记账，不评分
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


def test_summary_degrades_without_extra_llm_calls_when_it_keeps_overstepping(monkeypatch):
    """越界就纠正重试；仍越界则退成零 LLM 的降级版（只列各方主张），绝不空手而归。"""
    st = _install(
        monkeypatch,
        LLMState(
            intents=['{"speak": false}'],
            speeches=["甲的开场立场陈述", "乙的开场立场陈述"],
            summary='{"positions": [{"name": "甲", "stance": "甲更有说服力"}]}',
        ),
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 1, "t-degrade"))
    summary = [e for e in events if e["type"] == "summary"]
    assert len(summary) == 1
    data = summary[0]["data"]

    assert data["degraded"] is True
    assert data["positions"]                       # 降级仍给出立场，不是一块空白
    assert st.calls.count("clerk") == rt._SUMMARY_ATTEMPTS   # 只重试到上限，不无限烧钱
    assert all("更有说服力" not in p["stance"] for p in data["positions"])


def test_summary_carries_facts_not_scores(monkeypatch):
    st = _install(
        monkeypatch,
        LLMState(
            intents=['{"speak": false}'],
            speeches=["甲的开场", "乙的开场"],
            summary=(
                '{"clashes": [{"a": "甲", "b": "乙", "point": "在『德可不可教』上撞上"}],'
                ' "positions": [{"name": "甲", "stance": "德可教"}, {"name": "乙", "stance": "德不可教"}],'
                ' "open": ["若无天赋，教还有没有用"]}'
            ),
        ),
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 1, "t-fact"))
    types = _types(events)
    data = [e for e in events if e["type"] == "summary"][0]["data"]

    assert data["degraded"] is False
    assert data["clashes"][0]["point"] == "在『德可不可教』上撞上"
    assert len(data["positions"]) == 2
    # 发言次数只作为事实随纪要一起给出（配额是公开机制），不是分数
    assert {s["id"] for s in data["speeches"]} == {"jung", "adler"}
    assert all(s["quota"] == 2 for s in data["speeches"])
    # 纪要必须落在 end 之后：非阻塞旁路，纪要失败也不影响会议结果已经交付
    assert types.index("end") < types.index("summary_start") < types.index("summary_end")


def test_summary_can_be_switched_off(monkeypatch):
    _install(
        monkeypatch,
        LLMState(intents=['{"speak": false}'], speeches=["甲的开场", "乙的开场"]),
        summary_enabled=False,
    )
    events = _run(rt.stream_roundtable("议题", _speakers("jung", "adler"), 1, "t-nosum"))
    assert "summary_start" not in _types(events)


# ============================================================
# 兜底：人数与轮数的钳制（与既有约束一致）
# ============================================================
def test_too_few_or_too_many_speakers_is_rejected(monkeypatch):
    _install(monkeypatch, LLMState(speeches=["x"]))
    events = _run(rt.stream_roundtable("议题", _speakers("jung"), 1, "t-few"))
    assert _types(events) == ["error"]

    events = _run(rt.stream_roundtable(
        "议题", _speakers("jung", "adler", "wangyangming", "fengge"), 1, "t-many"))
    assert _types(events) == ["error"]
    assert str(rt.MAX_ROUNDTABLE_CHARS) in events[0]["content"]
