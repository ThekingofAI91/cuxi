"""
`astream_nonempty` 的首 token 超时 + 空响应重试测试；
外加 supervisor_agent 在「模型整轮无产出」时明确失败（不再回退无资料直答）。

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
from framework import runtime as rt
from framework import supervisor_agent as sup_agent


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
# supervisor_agent：模型空产出 → 明确失败（绝不无资料直答）
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


class SilentLLM:
    """产出为空 / 只有空白的假模型：整轮既不吐正文也不请求工具"""

    def __init__(self, payload=None):
        self._payload = payload

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        if self._payload is not None:
            yield _chunk(self._payload)


def _patch_silent_llm(monkeypatch, payload=None):
    import framework.supervisor_agent as sa

    monkeypatch.setattr(sa, "get_chat_llm", lambda **kw: SilentLLM(payload))


def test_supervisor_agent_fails_on_persistently_empty_upstream(monkeypatch):
    """上游整轮无正文也无工具请求（重试耗尽）→ 明确失败，不无资料直答。"""
    _patch_silent_llm(monkeypatch)
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(sup_agent.run_supervisor_agent(_sup_state()))


def test_supervisor_agent_fails_on_whitespace_only_output(monkeypatch):
    """正文非空但全是空白 → 等价于没产出，同样明确失败而不是无资料直答。"""
    _patch_silent_llm(monkeypatch, payload="   \n\n")
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(sup_agent.run_supervisor_agent(_sup_state()))


# ============================================================
# 失败时带出本轮已花掉的检索资料（供排查，不推给用户）
# ============================================================

class ToolCallingThenSilentLLM:
    """第 1 轮请求一次 search_library（让资料进入 sink），之后整轮沉默。

    用来逼出"空手但有资料"这条路径——它才是兜底重生成 + 抛错的分支。
    """

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def astream(self, messages):
        idx = self.calls
        self.calls += 1

        async def _gen():
            if idx == 0:
                yield SimpleNamespace(
                    content="",
                    tool_call_chunks=[{
                        "index": 0,
                        "name": "search_library",
                        "args": '{"query": "格物致知"}',
                        "id": "call_1",
                    }],
                )

        return _gen()


async def _empty_direct(*args, **kwargs):
    """兜底重生成同样空 → 模拟上游持续空响应。"""
    return ""


def test_failure_carries_retrieved_docs_for_troubleshooting(monkeypatch):
    """空手但有资料：抛错时必须把 sink 里的资料挂到异常上带出来。

    否则 sink 随栈销毁，异常里只剩一句 message——「资料可用于排查」
    就成了一纸空承诺（注释与代码不一致，2026-10-08 修正）。
    """
    from langchain_core.documents import Document

    docs = [
        Document(page_content="格物致知者，致吾心之良知于事事物物也。",
                 metadata={"source": "传习录.pdf", "heading": "答顾东桥书"}),
        Document(page_content="心外无物，心外无理。",
                 metadata={"source": "王阳明全集.pdf", "heading": "与王纯甫书"}),
    ]

    async def _fake_retrieve(query, history=None, light_retrieval=False,
                             skip_retrieval=False):
        return docs, False, ""

    monkeypatch.setattr(sup_agent, "get_chat_llm", lambda **kw: ToolCallingThenSilentLLM())
    monkeypatch.setattr(sup_agent, "retrieve_documents", _fake_retrieve)
    monkeypatch.setattr(rt, "_generate_direct_response", _empty_direct)

    with pytest.raises(LLMEmptyResponseError) as ei:
        asyncio.run(sup_agent.run_supervisor_agent(_sup_state()))

    assert len(ei.value.docs) == 2
    assert ei.value.docs[0].metadata["source"] == "传习录.pdf"


def test_empty_response_error_defaults_to_no_docs():
    """不带 docs 构造（旧调用点/直接 raise）仍可用，docs 为空列表而非 None。"""
    err = LLMEmptyResponseError("上游空了")
    assert err.docs == []
    assert "上游空了" in str(err)
