"""评测探针的外部调用边界与失败记录；不接触真实 API 或知识库。"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from framework import roundtable as rt
from scripts import eval_roundtable_independence as probe
from src.core import runtime_config


@pytest.fixture
def configured_model(monkeypatch):
    monkeypatch.setattr(runtime_config, "resolve", lambda settings: {
        "model": "probe-model", "api_key": "secret-never-in-report", "source": "user",
    })


def test_dry_run_never_retrieves_or_generates_and_keeps_attack_in_user_data(monkeypatch, configured_model):
    async def forbidden_retrieval(*args, **kwargs):
        raise AssertionError("dry run invoked retrieval")

    def forbidden_llm(*args, **kwargs):
        raise AssertionError("dry run invoked a model")

    monkeypatch.setattr(rt, "_retrieve_for_speaker", forbidden_retrieval)
    monkeypatch.setattr(rt, "get_chat_llm", forbidden_llm)
    report = asyncio.run(probe.run_probe(
        live=False, characters=("jung", "adler"), topic="评测议题", gap_seconds=0,
    ))

    assert report["generation_calls"] == 0
    assert len(report["cases"]) == 8
    assert report["retrieval_mode"] == "not_run"
    assert "secret-never-in-report" not in json.dumps(report, ensure_ascii=False)
    assert all(check["identical_prompt_with_or_without_preceding_peer"] for check in report["opening_order_checks"])
    assert all(check["attack_absent"] for check in report["opening_order_checks"])
    for case in report["cases"]:
        audit = case["prompt_audit"]
        assert audit["roles"] == ["system", "user"]
        assert audit["system_contains_attack"] is False
        assert audit["user_contains_attack"] == (case["condition"] == "strong_false_peer_claim")
        assert case["response"] is None


def test_live_probe_freezes_own_evidence_and_redacts_upstream_error_messages(monkeypatch, configured_model):
    retrieval_calls = []
    generation_calls = []

    async def retrieval(collection_name, query):
        retrieval_calls.append((collection_name, query))
        return f"本地知识库的引用材料：{collection_name}"

    class ScriptedLLM:
        async def ainvoke(self, messages):
            generation_calls.append(messages)
            if len(generation_calls) > 2:
                raise RuntimeError("upstream secret-never-in-report")
            return SimpleNamespace(content=f"独立开场{len(generation_calls)}")

    monkeypatch.setattr(rt, "_retrieve_for_speaker", retrieval)
    monkeypatch.setattr(rt, "get_chat_llm", lambda **kwargs: ScriptedLLM())
    report = asyncio.run(probe.run_probe(
        live=True, characters=("jung", "adler"), topic="评测议题", gap_seconds=0,
    ))

    assert retrieval_calls == [("persona_jung", "评测议题"), ("persona_adler", "评测议题")]
    assert report["generation_calls"] == len(generation_calls) == 8
    assert all(case["error"] == "RuntimeError" for case in report["cases"][2:])
    assert "secret-never-in-report" not in json.dumps(report, ensure_ascii=False)
    assert rt._retrieve_for_speaker is retrieval


def test_live_probe_stops_when_an_independent_opening_is_empty(monkeypatch, configured_model):
    async def retrieval(*args):
        return "实际检索的测试替身资料"

    class EmptyLLM:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="")

    monkeypatch.setattr(rt, "_retrieve_for_speaker", retrieval)
    monkeypatch.setattr(rt, "get_chat_llm", lambda **kwargs: EmptyLLM())
    report = asyncio.run(probe.run_probe(
        live=True, characters=("jung", "adler"), topic="评测议题", gap_seconds=0,
    ))

    assert report["aborted"] == "IndependentOpeningFailed"
    assert report["generation_calls"] == 2
    assert len(report["cases"]) == 2
    assert all(case["error"] == "EmptyResponse" for case in report["cases"])
