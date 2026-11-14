"""投标 JSON、署名引用与纠正预算的确定性测试，不调用模型或检索。"""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from framework.roundtable_intents import (
    IntentValidationError,
    normalize_transcript,
    request_validated_intent,
    validate_intent,
)


TRANSCRIPT = [
    {"character_id": "a", "name": "甲", "content": "这是我自己的观点。", "kind": "opening", "round": 0},
    {"character_id": "b", "name": "乙", "content": "情绪不是事实，行动仍由自己负责。\n可以先迈出一步。", "kind": "opening", "round": 0},
    {"character_id": "c", "name": "丙", "content": "我只说了另一件事。", "kind": "opening", "round": 0},
    {"character_id": "b", "name": "乙", "content": "改变需要时间。", "kind": "rebuttal", "round": 1},
]


def _proposal() -> dict:
    return {
        "want": True, "urgency": 4, "reason": "我要区分情绪的事实与事实判断", "move": "distinguish",
        "target": {"speaker_id": "b", "turn_id": "turn-2", "quote": "情绪不是事实"},
    }


def _validate(obj: dict) -> dict:
    return validate_intent(json.dumps(obj, ensure_ascii=False), transcript=TRANSCRIPT, speaker_id="a")


def test_normalization_preserves_existing_ids_and_does_not_mutate_shared_records():
    source = copy.deepcopy(TRANSCRIPT)
    source[0]["turn_id"] = "existing-a"
    before = copy.deepcopy(source)
    normalized = normalize_transcript(source)
    assert source == before
    assert normalized[0]["turn_id"] == "existing-a"
    assert normalized[1]["turn_id"] == "turn-2"
    assert normalized[1]["speaker_id"] == "b"
    assert normalized[1]["content"] == source[1]["content"]
    assert normalize_transcript(normalized) == normalized
    extended = normalize_transcript(source + [{"character_id": "c", "name": "丙", "content": "后来补充"}])
    assert extended[:len(normalized)] == normalized


def test_duplicate_turn_ids_are_rejected_instead_of_choosing_an_ambiguous_source():
    source = copy.deepcopy(TRANSCRIPT)
    source[0]["turn_id"] = "same"
    source[1]["turn_id"] = "same"
    with pytest.raises(IntentValidationError, match="重复"):
        normalize_transcript(source)


@pytest.mark.parametrize("wrapped", ["情绪不是事实", "「情绪不是事实」", '"情绪不是事实"', "“情绪不是事实”"])
def test_exact_quote_is_bound_to_the_real_speaker_and_turn(wrapped):
    proposal = _proposal()
    proposal["target"]["quote"] = wrapped
    bid = _validate(proposal)
    assert bid["target"] == {
        "speaker": "乙", "speaker_id": "b", "turn_id": "turn-2", "quote": "情绪不是事实",
    }


def test_real_short_quotes_are_not_rejected_for_artificial_length_requirements():
    proposal = _proposal()
    proposal["target"]["quote"] = "情绪"
    assert _validate(proposal)["target"]["quote"] == "情绪"


@pytest.mark.parametrize(("field", "value"), [
    ("want", "true"), ("want", 1), ("want", None),
    ("urgency", "4"), ("urgency", True), ("urgency", 4.0), ("urgency", 0), ("urgency", 6),
    ("reason", ""), ("reason", "  "), ("reason", 123),
    ("move", "attack"), ("move", ""), ("move", None),
    ("target", None), ("target", []),
])
def test_schema_errors_are_not_coerced_into_valid_bids(field, value):
    proposal = _proposal()
    proposal[field] = value
    with pytest.raises(IntentValidationError):
        _validate(proposal)


@pytest.mark.parametrize("field", ["want", "urgency", "reason", "move", "target"])
def test_willing_bids_require_every_contract_field(field):
    proposal = _proposal()
    del proposal[field]
    with pytest.raises(IntentValidationError):
        _validate(proposal)


@pytest.mark.parametrize("field", ["speaker_id", "turn_id", "quote"])
def test_targets_require_stable_speaker_and_turn_identity(field):
    proposal = _proposal()
    del proposal["target"][field]
    with pytest.raises(IntentValidationError, match=f"target.{field}"):
        _validate(proposal)


@pytest.mark.parametrize("target", [
    {"speaker_id": "ghost", "turn_id": "turn-2", "quote": "情绪不是事实"},
    {"speaker_id": "b", "turn_id": "turn-999", "quote": "情绪不是事实"},
    {"speaker_id": "c", "turn_id": "turn-2", "quote": "情绪不是事实"},
    {"speaker_id": "b", "turn_id": "turn-4", "quote": "情绪不是事实"},
    {"speaker_id": "b", "turn_id": "turn-2", "quote": "情绪不是事实", "speaker": "丙"},
    {"speaker_id": "a", "turn_id": "turn-1", "quote": "我自己的观点"},
])
def test_wrong_character_turn_name_and_self_citation_are_rejected(target):
    proposal = _proposal()
    proposal["target"] = target
    with pytest.raises(IntentValidationError):
        _validate(proposal)


@pytest.mark.parametrize("quote", [
    "情绪不等于事实", "情绪不是事实，自己负责", "情绪 不是事实", "情绪不是事实...可以先迈出一步",
    "自己负责。 可以先迈出一步。", "「」",
])
def test_quotes_must_be_contiguous_verbatim_text_without_fuzzy_repair(quote):
    proposal = _proposal()
    proposal["target"]["quote"] = quote
    with pytest.raises(IntentValidationError):
        _validate(proposal)


def test_newline_is_preserved_when_it_really_exists_in_the_quote():
    proposal = _proposal()
    proposal["target"]["quote"] = "自己负责。\n可以先迈出一步。"
    assert _validate(proposal)["target"]["quote"] == proposal["target"]["quote"]


@pytest.mark.parametrize("raw", [
    "{}", '[]', 'null', 'true',
    '{"want": false, "urgency": 1}',
    '{"want": "false", "urgency": 1, "reason": "旁听"}',
    '{"want": false, "urgency": 1, "reason": ""}',
    '{"want": false, "want": true, "urgency": 1, "reason": "旁听"}',
    '{"want": false, "urgency": NaN, "reason": "旁听"}',
    '{"want": false, "urgency": Infinity, "reason": "旁听"}',
    '{"want": false, "urgency": 1, "reason": "旁听"} 多余说明',
    '```json\n{"want": false, "urgency": 1, "reason": "旁听"}\n```',
])
def test_even_abstention_requires_an_unambiguous_complete_json_contract(raw):
    with pytest.raises(IntentValidationError):
        validate_intent(raw, transcript=TRANSCRIPT, speaker_id="a")


def test_a_valid_abstention_is_distinct_from_validation_failure():
    bid = validate_intent('{"want": false, "urgency": 1, "reason": "此刻没有新的补充"}',
                          transcript=TRANSCRIPT, speaker_id="a")
    assert bid == {"want": False, "urgency": 1, "reason": "此刻没有新的补充", "move": "", "target": {}}


def test_correction_retries_preserve_rules_and_account_for_every_attempt():
    wrong = _proposal()
    wrong["target"]["quote"] = "乙根本没说过这句话"
    outputs = ["SECRET-RAW-RESPONSE", json.dumps(wrong, ensure_ascii=False), json.dumps(_proposal(), ensure_ascii=False)]
    calls = []
    messages = [("system", "固定的角色规则"), ("user", "完整发言记录是引用数据")]
    original = copy.deepcopy(messages)
    ticks = iter([10.0, 10.025])

    async def invoke(sent, *, max_tokens):
        calls.append((sent, max_tokens))
        return SimpleNamespace(content=outputs[len(calls) - 1])

    result = asyncio.run(request_validated_intent(
        messages=messages, transcript=TRANSCRIPT, speaker_id="a", invoke=invoke, clock=lambda: next(ticks),
    ))
    assert result["bid"]["want"] is True and result["error"] == ""
    assert len(result["errors"]) == 2
    assert len(calls) == result["usage"]["llm_calls"] == 3
    assert all(tokens == 4000 for _, tokens in calls)
    assert result["usage"]["prompt_chars"] == sum(len(content) for sent, _ in calls for _, content in sent)
    assert result["usage"]["answer_chars"] == sum(map(len, outputs))
    assert result["usage"]["intent_ms"] == pytest.approx(25.0)
    assert messages == original
    assert all([m for m in sent if m[0] == "system"] == [messages[0]] for sent, _ in calls)
    assert calls[1][0][-1][0] == "user" and "JSON" in calls[1][0][-1][1]
    assert calls[2][0][-1][0] == "user" and "连续原文" in calls[2][0][-1][1]
    assert all("SECRET-RAW-RESPONSE" not in content for sent, _ in calls for _, content in sent)


@pytest.mark.parametrize(("budget", "expected"), [(0, 0), (1, 1), (2, 2), (3, 3), (20, 3), (-1, 0)])
def test_corrections_never_exceed_the_available_attempt_budget(budget, expected):
    calls = []

    async def invoke(sent, *, max_tokens):
        calls.append(sent)
        return "{}"

    result = asyncio.run(request_validated_intent(
        messages=[("system", "规则")], transcript=TRANSCRIPT, speaker_id="a", invoke=invoke, max_attempts=budget,
    ))
    assert result["bid"] is None and result["error"]
    assert len(calls) == result["usage"]["llm_calls"] == expected
    assert result["usage"]["answer_chars"] == 2 * expected
    assert len(result["errors"]) == expected


def test_a_valid_abstention_stops_immediately_without_spending_retry_budget():
    calls = []

    async def invoke(sent, *, max_tokens):
        calls.append(sent)
        return '{"want": false, "urgency": 1, "reason": "我没有新的补充"}'

    result = asyncio.run(request_validated_intent(
        messages=[], transcript=TRANSCRIPT, speaker_id="a", invoke=invoke,
    ))
    assert result["bid"]["want"] is False
    assert result["error"] == "" and result["errors"] == []
    assert len(calls) == result["usage"]["llm_calls"] == 1


def test_transient_call_error_is_counted_and_retried_without_echoing_exception_details():
    calls = []
    response = '{"want": false, "urgency": 1, "reason": "旁听"}'

    async def invoke(sent, *, max_tokens):
        calls.append(sent)
        if len(calls) == 1:
            raise TimeoutError("PRIVATE-ENDPOINT-AND-KEY")
        return response

    result = asyncio.run(request_validated_intent(
        messages=[], transcript=TRANSCRIPT, speaker_id="a", invoke=invoke,
    ))
    assert result["bid"]["want"] is False
    assert result["usage"]["llm_calls"] == 2
    assert result["usage"]["answer_chars"] == len(response)
    assert "TimeoutError" in result["errors"][0]
    assert "PRIVATE-ENDPOINT-AND-KEY" not in str(result)
    assert "PRIVATE-ENDPOINT-AND-KEY" not in str(calls)
