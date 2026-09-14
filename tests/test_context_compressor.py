# -*- coding: utf-8 -*-
"""
test_context_compressor.py — 上下文智能压缩

锁住三条契约：
1. 正常压缩：解析"片段N: 摘要"，无关片段置空 → 回退原文
2. 失败兜底：LLM 挂/解析全空 → 返回原文档，绝不丢检索结果
3. 缓存：相似问题（≥0.92）第二次零 LLM 调用
"""

import asyncio
import hashlib

import numpy as np
import pytest
from langchain_core.documents import Document

from src.retrieval import context_compressor as CC


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return _Msg(
            "片段1: 集体无意识由原型组成。\n"
            "片段2: （无关）\n"
            "片段3: 自性化是整合人格的过程。"
        )


def _fake_embed(texts):
    out = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.random(8, dtype=np.float32)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(CC, "_norm_vec", lambda t: _fake_embed([t])[0])
    CC._cache.clear()
    yield
    CC._cache.clear()


def _docs():
    return [
        Document(page_content="集体无意识是…原型……", metadata={"source": "A"}),
        Document(page_content="无关内容……", metadata={"source": "B"}),
        Document(page_content="自性化过程……", metadata={"source": "C"}),
    ]


def test_compress_and_parse():
    llm = _FakeLLM()
    out = asyncio.run(CC.compress_docs("什么是集体无意识？", _docs(), llm=llm))
    assert out[0].page_content == "集体无意识由原型组成。"
    assert out[1].page_content == "无关内容……", "（无关）片段回退原文"
    assert out[2].page_content == "自性化是整合人格的过程。"
    assert out[0].metadata["source"] == "A", "metadata 必须保留"


def test_failure_falls_back_to_original(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("上游挂了")

    monkeypatch.setattr(CC, "_norm_vec", boom)
    docs = _docs()
    out = asyncio.run(CC.compress_docs("q", docs, llm=_FakeLLM()))
    assert [d.page_content for d in out] == [d.page_content for d in docs]


def test_cache_hits_skip_llm():
    llm = _FakeLLM()
    docs = _docs()
    asyncio.run(CC.compress_docs("什么是集体无意识？", docs, llm=llm))
    assert llm.calls == 1
    # 相似问题（同文本）第二次 → 缓存命中，不再调 LLM
    out = asyncio.run(CC.compress_docs("什么是集体无意识？", docs, llm=llm))
    assert llm.calls == 1
    assert out[0].page_content == "集体无意识由原型组成。"
