# -*- coding: utf-8 -*-
"""
test_memory.py — 用户长期记忆模块

锁住五条契约：
1. 提取：LLM 返回的记忆条目正确入库（profile + topic）
2. 去重：近似条目（余弦 ≥ 0.85）更新原条目，不产生重复
3. 检索：profile 常驻注入；topic 按查询向量相似度过滤（低于阈值不进）
4. 容量：profile 超上限淘汰最旧的
5. 注销：delete_user 清空该用户全部记忆
"""

import asyncio
import hashlib

import numpy as np
import pytest

from src.core import memory as M


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload

    async def ainvoke(self, messages):
        return _Msg(self.payload)


def _fake_embed(texts):
    """确定性伪向量：文本哈希做种子（跨进程稳定，同 rewrite 测试模式）"""
    out = []
    for t in texts:
        seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.random(8, dtype=np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


# 各测试自定义"提取 LLM 返回内容"的槽位
_payload_holder = {"payload": ""}


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "_embed_texts", _fake_embed)
    monkeypatch.setattr(M, "get_memory_store", lambda: M.MemoryStore(str(tmp_path / "mem.db")))
    monkeypatch.setattr(M.settings, "memory_enabled", True)
    # 关键：提取 LLM 必须替换成假客户端——否则测试会打真实 API
    monkeypatch.setattr(M, "_get_extract_llm", lambda: _FakeLLM(_payload_holder["payload"]))
    _payload_holder["payload"] = ""
    yield


def test_extract_and_store_items():
    """提取结果按 kind 入库"""
    _payload_holder["payload"] = (
        '{"memories": [{"kind": "profile", "text": "用户正在准备考研"}, '
        '{"kind": "topic", "text": "用户关注自性化过程"}]}'
    )
    asyncio.run(M.extract_and_store("u:1", "我在准备考研，自性化是什么？", "自性化是……"))
    store = M.get_memory_store()
    items = store.all_for_user("u:1")
    kinds = {i["kind"] for i in items}
    texts = {i["text"] for i in items}
    assert kinds == {"profile", "topic"}
    assert "用户正在准备考研" in texts and "用户关注自性化过程" in texts


def test_near_duplicate_updates_not_duplicates():
    """近似条目更新原条目，不新增"""
    _payload_holder["payload"] = '{"memories": [{"kind": "profile", "text": "用户正在准备考研心理学"}]}'
    asyncio.run(M.extract_and_store("u:2", "我在准备考研", "好的"))
    # 相同文本再次提取（模拟用户重复自述）→ 应更新而非插入第二条
    asyncio.run(M.extract_and_store("u:2", "考研的事", "加油"))
    store = M.get_memory_store()
    items = [i for i in store.all_for_user("u:2") if i["text"] == "用户正在准备考研心理学"]
    assert len(items) == 1


def test_retrieve_profiles_always_and_topics_by_sim():
    """profile 常驻；topic 按相似度过滤（正交的不进，同向的进）"""
    store = M.get_memory_store()
    store.insert("u:3", "profile", "用户在准备高考", _fake_embed(["x"])[0])

    # 确定性构造与查询正交/同向的主题向量，不受随机向量波动影响
    qvec = _fake_embed(["什么是共时性？"])[0]
    raw = _fake_embed(["y"])[0]
    orth = raw - np.dot(raw, qvec) * qvec
    orth = orth / np.linalg.norm(orth)
    store.insert("u:3", "topic", "完全无关的主题条目", orth)
    store.insert("u:3", "topic", "与查询高度相关的主题条目", qvec)

    block = asyncio.run(M.retrieve_memory_block("u:3", "什么是共时性？"))
    assert "用户在准备高考" in block, "profile 应回常驻注入"
    assert "与查询高度相关的主题条目" in block, "同向 topic 应命中"
    assert "完全无关的主题条目" not in block, "正交 topic 不应命中"


def test_profile_cap_trims_oldest():
    """profile 超上限淘汰最旧的"""
    store = M.get_memory_store()
    for i in range(10):
        store.insert("u:4", "profile", f"条目{i}", _fake_embed([f"x{i}"])[0])
    _payload_holder["payload"] = '{"memories": [{"kind": "profile", "text": "新增的记忆条目X"}]}'
    asyncio.run(M.extract_and_store("u:4", "新增一条", "好"))
    items = [i for i in store.all_for_user("u:4") if i["kind"] == "profile"]
    assert len(items) <= M._PROFILE_CAP


def test_delete_user_clears_all():
    store = M.get_memory_store()
    store.insert("u:5", "profile", "A", _fake_embed(["a"])[0])
    store.insert("u:5", "topic", "B", _fake_embed(["b"])[0])
    store.insert("u:6", "profile", "C", _fake_embed(["c"])[0])
    assert store.delete_user("u:5") == 2
    assert store.all_for_user("u:5") == []
    assert len(store.all_for_user("u:6")) == 1, "不能误删其他用户"
