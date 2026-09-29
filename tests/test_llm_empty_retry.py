"""
`astream_nonempty` 的首 token 超时 + 空响应重试测试；
外加 supervisor 在「分析节点跑过一轮却无产出」时明确失败（不再回退无资料直答）。

LLM 全部打桩（按脚本产出 chunk / 挂起 / 抛异常），不触真实 API。

设计要点（回归防线）：
1. 上游空壳会挂到它自身超时才回 200+空内容（实测约 13s），等满纯属白等 →
   首 token 超时必须在阈值处掐断，而不是等流自然结束；
2. 首 token 一到就不再计时，后面慢产出不能被误杀；
3. 已经向用户推过 token 后绝不重试，否则内容会被重复推一遍；
4. analysis 为空的「无资料直答」兜底已删 —— 空产出必须明确失败，不能静默降级。
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.core.llm import LLMEmptyResponseError, astream_nonempty
from framework import supervisor as sup


def _chunk(text):
    return SimpleNamespace(content=text)


class ScriptedStreamLLM:
    """按「第几次调用 astream()」为单位产出流的假模型。

    script[i] 描述第 i 次调用的行为，元素为：
      (sleep, text)  —— 先 sleep 秒，再产出一个 content=text 的 chunk（text=None 即空 content）
      Exception 实例 —— 直接抛
    调用次数超出 script 长度时，重复最后一项的行为。
    """

    def __init__(self, script):
        self._script = script
        self.calls = 0

    def astream(self, messages):
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        steps = self._script[idx]

        async def _gen():
            for step in steps:
                if isinstance(step, Exception):
                    raise step
                sleep, text = step
                if sleep:
                    await asyncio.sleep(sleep)
                yield _chunk(text)

        return _gen()


async def _drain(llm, **kw):
    out = []
    async for tok in astream_nonempty(llm, [("user", "hi")], **kw):
        out.append(tok)
    return out


# ============================================================
# 首 token 超时
# ============================================================

def test_first_token_timeout_retries_then_succeeds():
    """第一次迟迟不出 token（模拟上游空壳挂住）→ 按 TTFT 掐断重试，第二次正常产出。

    关键是耗时：总耗时应约等于 TTFT 阈值，而不是第一次那 0.5s 的挂起时间。
    """
    llm = ScriptedStreamLLM([
        [(0.5, "太晚了")],                 # 0.5s 后才出 → 超过 0.05s 阈值
        [(0.0, "你好"), (0.0, "，世界")],
    ])
    t0 = time.perf_counter()
    out = asyncio.run(_drain(llm, attempts=1, ttft_timeout=0.05))
    elapsed = time.perf_counter() - t0

    assert out == ["你好", "，世界"]
    assert llm.calls == 2
    assert elapsed < 0.3, f"没有按 TTFT 掐断，实际等了 {elapsed:.2f}s"


def test_first_token_timeout_exhausted_returns_empty_without_raising():
    """每次都超时 → 重试耗尽后安静返回空，由调用方决定怎么兜底（本层不抛业务异常）。"""
    llm = ScriptedStreamLLM([[(0.5, "永远来不及")]])
    t0 = time.perf_counter()
    out = asyncio.run(_drain(llm, attempts=1, ttft_timeout=0.05))
    elapsed = time.perf_counter() - t0

    assert out == []
    assert llm.calls == 2
    assert elapsed < 0.3


def test_first_token_arrived_disables_timeout():
    """首 token 一到就不再计时：后面的慢产出不该被掐断。"""
    llm = ScriptedStreamLLM([[(0.0, "甲"), (0.3, "乙")]])
    assert asyncio.run(_drain(llm, attempts=0, ttft_timeout=0.05)) == ["甲", "乙"]


# ============================================================
# 整次流空 / 已推送后不重试
# ============================================================

def test_zero_token_stream_retries_then_succeeds():
    """正常结束但一个 token 都没有 → 视为失败重试。"""
    llm = ScriptedStreamLLM([[], [(0.0, "好")]])
    assert asyncio.run(_drain(llm, attempts=1, ttft_timeout=1.0)) == ["好"]
    assert llm.calls == 2


def test_empty_chunk_counts_as_no_token():
    """空 content 的 chunk 不算产出，仍按空响应重试。"""
    llm = ScriptedStreamLLM([[(0.0, None)], [(0.0, "有内容")]])
    assert asyncio.run(_drain(llm, attempts=1, ttft_timeout=1.0)) == ["有内容"]


def test_no_retry_after_tokens_already_pushed():
    """已经往用户推过 token 就不能重试（否则内容会被重复推一遍）。"""
    llm = ScriptedStreamLLM([[(0.0, "半句"), ValueError("上游断了")]])
    with pytest.raises(ValueError):
        asyncio.run(_drain(llm, attempts=2, ttft_timeout=1.0))
    assert llm.calls == 1


# ============================================================
# supervisor：分析为空 → 明确失败
# ============================================================

def _sup_state(**over):
    state = {
        "query": "你为什么说格物致知要走内心？",
        "session_id": "t-empty",
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


def test_supervisor_fails_when_analyzer_ran_but_produced_nothing(monkeypatch):
    """analyzer 已跑过一轮仍无分析 → 明确失败，不再回送 analyzer、也不走无资料直答。

    修复前这里会一路回送 analyzer，直到 supervisor_count>6 才用无资料直答收手。
    """
    monkeypatch.setattr(settings, "light_chat_enabled", False)
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(sup.supervisor_node(_sup_state(
            route_history=["supervisor", "retrieval_agent", "supervisor", "analysis_agent"],
        )))


def test_supervisor_fails_on_whitespace_only_analysis(monkeypatch):
    """analysis 非空但全是空白 → 等价于没产出，同样明确失败而不是无资料直答。"""
    monkeypatch.setattr(settings, "light_chat_enabled", False)
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(sup.supervisor_node(_sup_state(analysis="   \n\n")))


def test_supervisor_still_routes_to_analyzer_before_analyzer_ever_ran(monkeypatch):
    """边界：检索过但 analyzer 还没跑 → 仍要送 analyzer（新增的失败闸不能抢在前面）。"""
    monkeypatch.setattr(settings, "light_chat_enabled", False)
    out = asyncio.run(sup.supervisor_node(_sup_state(
        route_history=["supervisor", "retrieval_agent", "supervisor"],
    )))
    assert out["next_agent"] == "analyzer"
