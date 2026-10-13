"""
圆桌发言者的检索契约（优化一百一十三）。

一句话契约：**检索由代码在发言前强制触发，固定走全管线强检索（strong=True），
只查这位发言者自己的语料库**——没有模型自持的工具，也不给模型"选不选、选多深"的机会。

本文件只钉这一条契约：把统一检索 `retrieve_documents` 换成 spy，断言
`_retrieve_for_speaker` 必须以 strong=True + 本发言者的 collection 去调它。
不打桩图、不碰 ChromaDB。
"""

import pytest

from framework import roundtable as rt


class _Doc:
    """最小 Document 替身：拼接只用到 page_content。"""

    def __init__(self, text: str):
        self.page_content = text


@pytest.mark.asyncio
async def test_retrieval_is_forced_strong_on_own_collection(monkeypatch):
    """必须以 strong=True 且带上本发言者的 collection 调用统一检索"""
    captured: dict = {}

    async def _spy(query, history=None, strong=True, collection_name=None):
        captured.update(query=query, strong=strong, collection_name=collection_name)
        return [], False, ""

    monkeypatch.setattr(rt, "retrieve_documents", _spy)

    out = await rt._retrieve_for_speaker("persona_jung", "什么是集体潜意识")

    assert captured["strong"] is True
    assert captured["collection_name"] == "persona_jung"
    assert captured["query"] == "什么是集体潜意识"
    assert out == ""


@pytest.mark.asyncio
async def test_retrieval_joins_and_dedups_page_content(monkeypatch):
    """命中片段按顺序拼接、重复内容去重"""

    async def _spy(query, history=None, strong=True, collection_name=None):
        return [_Doc("甲段"), _Doc("乙段"), _Doc("甲段")], False, ""

    monkeypatch.setattr(rt, "retrieve_documents", _spy)

    assert await rt._retrieve_for_speaker("col", "q") == "甲段\n\n乙段"


@pytest.mark.asyncio
async def test_retrieval_empty_when_no_hits(monkeypatch):
    """无命中（含检索出错）时返回空串，不向上抛"""

    async def _spy(query, history=None, strong=True, collection_name=None):
        return [], False, "检索失败占位"

    monkeypatch.setattr(rt, "retrieve_documents", _spy)

    assert await rt._retrieve_for_speaker("col", "q") == ""
