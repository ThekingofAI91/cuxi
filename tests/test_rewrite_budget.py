# -*- coding: utf-8 -*-
"""
test_rewrite_budget.py — 检索改写的缓存与等待预算

这段逻辑有时序与并发（shield + 后台任务补缓存），最容易在后续重构中悄悄退化。
测试锁住四条契约：
1. 缓存命中 → 不再调用 LLM（省掉整次往返）
2. 未命中且 LLM 够快 → 正常用改写结果并写入缓存
3. 未命中且 LLM 超过预算 → 放弃改写但**不抛异常**，且后台跑完后补进缓存
4. 相似度不足 → 不复用（改写结果对问题语义敏感，不能乱套）
"""

import asyncio
import hashlib

import numpy as np
import pytest

from src.retrieval import advanced_search as AS


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """可控延迟的假 LLM，记录被调用次数"""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return _Msg("【查询】\n变体甲\n变体乙\n【假设答案】\n一段假设答案内容")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """每个用例前清空缓存与后台任务，并把 embedding 换成固定的确定性向量"""
    AS._rewrite_cache.clear()
    AS._bg_rewrite_tasks.clear()

    vectors = {}

    def fake_norm_vec(text):
        # 用文本长度造一个稳定的伪向量：长度相同 → 向量相同（相似度 1.0）
        # 种子必须跨进程稳定：内置 hash() 是进程级随机化的（PYTHONHASHSEED），
        # 用它做种子会让"两问题不相似"的断言变成小概率偶发失败
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.random(8, dtype=np.float32)
        if text in vectors:
            v = vectors[text]
        else:
            vectors[text] = v
        return v / np.linalg.norm(v)

    monkeypatch.setattr(AS, "_norm_vec", fake_norm_vec)
    monkeypatch.setattr(AS.settings, "rewrite_deadline_sec", 0.3)
    yield
    AS._rewrite_cache.clear()
    AS._bg_rewrite_tasks.clear()


class TestCacheHit:
    def test_second_identical_question_skips_llm(self):
        """同一个问题问两次：第二次应命中缓存，不再打 LLM"""
        async def run():
            llm = _FakeLLM(delay=0.01)
            v1, h1 = await AS._rewrite_with_budget("三十岁没对象正常吗", llm, 2)
            calls_after_first = llm.calls
            v2, h2 = await AS._rewrite_with_budget("三十岁没对象正常吗", llm, 2)
            return v1, v2, calls_after_first, llm.calls

        v1, v2, first, second = asyncio.run(run())
        assert len(v1) == 2, "首次应拿到 2 个查询变体"
        assert v2 == v1, "第二次应复用缓存中的改写结果"
        assert first == 1
        assert second == 1, "命中缓存时不应再调用 LLM"


class TestBudgetTimeout:
    def test_slow_llm_abandoned_without_error(self):
        """LLM 慢于预算：放弃改写、返回空，但不抛异常"""
        async def run():
            llm = _FakeLLM(delay=1.0)  # 远超 0.3s 预算
            result = await AS._rewrite_with_budget("一个比较长的复杂问题需要改写", llm, 2)
            # 必须在事件循环内检查：asyncio.run 结束时会取消未完成任务，
            # 其 done_callback 会把任务从集合里移除
            return result, len(AS._bg_rewrite_tasks)

        (variants, hyde), bg_count = asyncio.run(run())
        assert variants == [] and hyde == "", "超预算应放弃改写，退回原始查询检索"
        assert bg_count == 1, "改写任务应转入后台而非被取消"

    def test_background_task_later_fills_cache(self):
        """放弃之后，后台跑完的结果要补进缓存——这次白等，下次零等待"""
        async def run():
            llm = _FakeLLM(delay=0.5)
            await AS._rewrite_with_budget("一个比较长的复杂问题需要改写", llm, 2)
            assert len(AS._rewrite_cache) == 0, "放弃时缓存还没填充"
            # 等后台任务跑完 + 缓存写入
            for _ in range(50):
                await asyncio.sleep(0.05)
                if AS._rewrite_cache:
                    break
            return len(AS._rewrite_cache)

        n = asyncio.run(run())
        assert n == 1, "后台改写完成后应把结果写入缓存"


class TestFastPath:
    def test_fast_llm_uses_result_and_caches(self):
        """LLM 在预算内返回：直接用改写结果，并写入缓存"""
        async def run():
            llm = _FakeLLM(delay=0.01)
            variants, hyde = await AS._rewrite_with_budget("一个比较长的复杂问题需要改写", llm, 2)
            return variants, hyde, len(AS._rewrite_cache)

        variants, hyde, cached = asyncio.run(run())
        assert len(variants) == 2
        assert "假设答案" in hyde
        assert cached == 1, "成功改写后应写入缓存"


class TestSimilarityGuard:
    def test_dissimilar_question_not_reused(self):
        """语义不相近的问题不应复用改写结果"""
        # 语义缓存是模块级单例，其他测试可能已写入条目导致本用例误命中——先清空
        AS._rewrite_cache.clear()
        async def run():
            llm = _FakeLLM(delay=0.01)
            await AS._rewrite_with_budget("关于力工与亲密关系的困惑", llm, 2)
            before = llm.calls
            await AS._rewrite_with_budget("完全不同的另一个话题领域", llm, 2)
            return before, llm.calls

        before, after = asyncio.run(run())
        assert after == before + 1, "不相似的问题应重新调用 LLM，不能复用"
