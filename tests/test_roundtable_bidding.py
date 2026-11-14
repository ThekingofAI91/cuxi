"""
圆桌投标（Contract Net）的解析与裁决规则

这套测试守三件事：

1. **投标必须可验证**。只接受完整合法 JSON，回应对象必须是他人的具体发言，
   原话逐字匹配。格式或引用错误交给有限纠正重试，不能伪装成角色主动弃权。
2. **裁决必须是可预测的**。urgency 高者优先；平手时距上次发言最久者优先；
   再平手按 id。**全程没有模型参与**，所以结果能复现。
3. **谁不能投标是硬规则**。配额耗尽的人、上一轮的发言者都不进投标名单——
   这是「强制轮转」，靠代码保证，不靠模型自觉。
"""

from __future__ import annotations

import asyncio

import pytest

from framework import roundtable as rt
from framework.roundtable_intents import IntentValidationError, validate_intent


# ============================================================
# 一、投标解析
# ============================================================

def test_parse_clean_json():
    bid = validate_intent(
        '{"want": true, "urgency": 4, "target": {"speaker_id": "adler", "turn_id": "turn-1", "quote": "社会兴趣是后天习得的"},'
        '"reason": "他抽掉了原型的先天基础", "move": "rebut"}',
        transcript=[{"character_id": "adler", "name": "阿德勒", "content": "我认为社会兴趣是后天习得的。"}],
        speaker_id="jung",
    )
    assert bid["want"] is True
    assert bid["urgency"] == 4
    assert bid["target"]["quote"] == "社会兴趣是后天习得的"
    assert bid["move"] == "rebut"
    assert bid["target"]["speaker"] == "阿德勒"
    assert bid["target"]["speaker_id"] == "adler"


@pytest.mark.parametrize("raw", [
    "", "   ", "我觉得可以说两句", "{}",
    '好的，我的判断如下：{"want": false, "urgency": 1, "reason": "没什么非说不可"}',
    "want: true, urgency: 3, reason: '他漏了一点', move: clarify",
    '{"want": false, "urgency": 9, "reason": "旁听"}',
    '{"want": false, "urgency": 0, "reason": "旁听"}',
])
def test_invalid_output_is_not_silently_treated_as_abstention(raw):
    with pytest.raises(IntentValidationError):
        validate_intent(raw, transcript=[], speaker_id="jung")


# ============================================================
# 二、可投标人筛选（硬规则）
# ============================================================

def _state(*, quota=2, ledger=None, last_selected="", rounds=None):
    roster = [{"id": cid, "name": cid} for cid in ("a", "b", "c")]
    return {
        "roster": roster,
        "ledger": ledger or {cid: {"name": cid, "speeches": 0, "quota": quota} for cid in ("a", "b", "c")},
        "quota": quota,
        "bid_round": 1,
        "transcript": rounds or [],
        "last_decision": {"selected": last_selected} if last_selected else {},
        "topic": "议题",
    }


def test_fan_out_excludes_previous_speaker():
    """上一轮的发言者本轮不能投标——强制轮转，防止一个人占住话语权。"""
    sends = rt.fan_out_bids(_state(last_selected="a"))
    assert {s.arg["character_id"] for s in sends} == {"b", "c"}


def test_fan_out_excludes_quota_exhausted():
    ledger = {
        "a": {"name": "a", "speeches": 2, "quota": 2},   # 用完
        "b": {"name": "b", "speeches": 0, "quota": 2},
        "c": {"name": "c", "speeches": 1, "quota": 2},
    }
    sends = rt.fan_out_bids(_state(ledger=ledger))
    assert {s.arg["character_id"] for s in sends} == {"b", "c"}


def test_fan_out_is_parallel_send_not_separate_nodes():
    """分发出来的是同一个节点 N 次，不是 N 个节点。"""
    sends = rt.fan_out_bids(_state())
    assert len(sends) == 3
    assert {s.node for s in sends} == {"collect_intent"}


# ============================================================
# 三、裁决排序
# ============================================================

def _bid(cid, urgency, *, want=True):
    return {"character_id": cid, "want": want, "urgency": urgency, "reason": f"{cid} 有话", "move": "rebut"}


def test_urgency_wins():
    """urgency 高者优先，跟先后顺序无关。"""
    bids = [_bid("b", 3), _bid("a", 5), _bid("c", 4)]
    bids.sort(key=lambda i: (-int(i["urgency"]), 0, i["character_id"]))
    assert bids[0]["character_id"] == "a"


def test_tie_breaks_by_longest_silence():
    """urgency 相同时，距上次发言最久的人优先（从没说过 = -1，排最前）。"""
    transcript = [
        {"character_id": "a", "content": "a 刚刚说过", "kind": "rebuttal", "round": 2},
        {"character_id": "b", "content": "b 上一轮说过", "kind": "rebuttal", "round": 1},
    ]
    bids = [_bid("a", 4), _bid("b", 4), _bid("c", 4)]
    bids.sort(key=lambda i: (
        -int(i["urgency"]),
        rt._last_spoke_round(transcript, i["character_id"]),
        i["character_id"],
    ))
    # c 从没发过言（-1），b 是第 1 轮，a 是第 2 轮
    assert [i["character_id"] for i in bids] == ["c", "b", "a"]


def test_last_spoke_round_reads_latest_occurrence():
    transcript = [
        {"character_id": "a", "round": 1},
        {"character_id": "b", "round": 2},
        {"character_id": "a", "round": 3},
    ]
    assert rt._last_spoke_round(transcript, "a") == 3
    assert rt._last_spoke_round(transcript, "b") == 2
    assert rt._last_spoke_round(transcript, "nobody") == -1


# ============================================================
# 四、无人可投标：强制轮转的边界
# ============================================================

def test_no_bidder_left_converges(monkeypatch):
    """只剩上一轮发言者还有配额 → 本轮无人具备投标资格 → 直接收敛。

    「上轮说过的人本轮闭嘴」是硬规则，所以这不是故障：哪怕他还有话，
    本轮也没人能投标。host_node 遇到空名单会按配额收敛收场，
    避免空转（如果不收敛，它会发起 0 份投标再回到自己，白烧一轮）。
    """
    state = {
        "roster": [{"id": "a", "name": "a"}, {"id": "b", "name": "b"}],
        "ledger": {
            "a": {"name": "a", "speeches": 0, "quota": 2},   # 还有配额
            "b": {"name": "b", "speeches": 2, "quota": 2},   # 已用完
        },
        "opening_done": ["a", "b"],
        "transcript": [{"character_id": "a", "kind": "opening", "round": 0, "content": "..."}],
        "last_decision": {"selected": "a"},   # a 正是上一轮的发言者
        "bid_round": 0,
        "round_index": 0,
        "max_rounds": 2,
        "quota": 2,
        "max_calls": 24,
        "llm_calls": 0,
        "picker": "agent",
        "topic": "议题",
    }
    out = asyncio.run(rt.host_node(state))

    assert out["converged"] is True
    assert out["converge_reason"] == "quota"
    assert out["ended"] is True


# ============================================================
# 五、单点失败不能拖垮整场
# ============================================================

def test_failed_bid_remains_a_visible_failure(monkeypatch):
    """某个角色投标抛异常（429 / 超时）→ 记进 intent_errors，不向上抛。

    网关突发即 429 是常态。如果这里让异常冒出去，一个角色撞上限流就会
    让整场辩论停摆；失败要单独记账并展示，其余人仍可正常提交请缨。
    """
    async def boom(**kwargs):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(rt, "ask_intent", boom)
    out = asyncio.run(rt.collect_intent_node({"character_id": "jung", "round": 1}))

    assert out["intents"][0]["status"] == "error"
    assert out["intents"][0]["want"] is False
    assert out["bid_usage"][0]["llm_calls"] == 1
    err = out["intent_errors"][0]
    assert err["character_id"] == "jung"
    assert "429" in err["error"]


def test_unknown_character_is_a_failure_without_spending_a_call():
    """角色 id 不在配置里（被删了 / 前端传错）同样只记错误，不抛。"""
    out = asyncio.run(rt.collect_intent_node({"character_id": "ghost", "round": 1}))
    assert out["intents"][0]["status"] == "error"
    assert out["intents"][0]["want"] is False
    assert out["bid_usage"][0]["llm_calls"] == 0
    assert out["intent_errors"][0]["error"] == "角色不可用"
