"""
reranker.py — Cross-Encoder 重排序

混合检索之后做精排，确保最相关的排在前面。
使用 Cross-Encoder 对 (query, document) 对进行深度语义匹配。
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Optional

from langchain_core.documents import Document

from src.core.config import settings


# ============================================================
# 精排分数缓存（LRU）
# ============================================================
# 首页的"建议问题"、高频问题会被反复提问——同一 (query, doc) 对的精排分数
# 完全确定，缓存后命中即零推理（省 ~1s/次）。键含文档内容指纹，语料重组后
# 自动失效。进程内缓存，多 worker 各自维护（可接受）。
_SCORE_CACHE: OrderedDict[tuple, float] = OrderedDict()
_SCORE_CACHE_MAX = 2000
_SCORE_CACHE_LOCK = threading.Lock()

# 精排输入跨度（字符）与 token 上限。
#
# 2026-10-18 实测（output/_rerank_span_probe.py，98 题 × 8 候选 = 784 对，改写关）：
#   200 字 / 256 token  vs  500 字 / 512 token
#   → top-1 只一致 44.9%（44/98）、top-3 完全相同 24.5%、Spearman 0.636
#   → ≥800 字的块平均 +0.123 logit，<300 字的块 -0.017
#   即「只读开头」并不够：长块的段首多是过渡句，答案在中段。5 道题的 top1 从第
#   6-8 名直接翻到第 1，全都是 1100-1223 字的长块。
# 上限 512 是模型决定的：bge-reranker-base 的 max_position_embeddings=514，
# 写超过 512 的 max_length 不会生效（实测 1024 与 512 的分数逐位相同）。
# 成本：80ms/对 → 192ms/对；生产每请求 5 对，约 0.4s → 0.96s。
MAX_PAIR_CHARS = 500
MAX_PAIR_TOKENS = 512


def _score_cache_get(key: tuple) -> Optional[float]:
    with _SCORE_CACHE_LOCK:
        v = _SCORE_CACHE.get(key)
        if v is not None:
            _SCORE_CACHE.move_to_end(key)
        return v


def _score_cache_put(key: tuple, score: float) -> None:
    with _SCORE_CACHE_LOCK:
        _SCORE_CACHE[key] = score
        _SCORE_CACHE.move_to_end(key)
        while len(_SCORE_CACHE) > _SCORE_CACHE_MAX:
            _SCORE_CACHE.popitem(last=False)


def _pair_cache_key(query_head: str, doc: Document) -> tuple:
    doc_fp = hashlib.md5((doc.page_content or "")[:MAX_PAIR_CHARS].encode("utf-8")).hexdigest()
    return (query_head, doc_fp)


# ============================================================
# Cross-Encoder 重排序器
# ============================================================

class Reranker:
    """
    Cross-Encoder 重排序器

    用法:
        reranker = Reranker()
        ranked = reranker.rerank(query, candidates, top_k=5)
    """

    # 类级锁：
    # - 模型加载锁：防止并发请求同时加载大模型（重复加载浪费 10-30s 与数 GB 内存）
    # - 推理信号量：Cross-Encoder CPU 推理是有界并行（BoundedSemaphore），
    #   全串行浪费多核（16 核只有 1 个在跑），全放开又互相抢占把 CPU 打满；
    #   默认 rerank_max_concurrent=4，多核机器吞吐接近线性提升，单请求排队可控
    _LOAD_LOCK = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Cross-Encoder 模型名称
            device: 运行设备（cpu / cuda）
        """
        self.model_name = model_name or settings.rerank_model
        self.device = device or "cpu"
        self._model = None
        self._predict_sem = threading.BoundedSemaphore(max(1, settings.rerank_max_concurrent))

    @property
    def model(self):
        """延迟加载 Cross-Encoder 模型（线程安全，只加载一次）"""
        if self._model is None:
            with self._LOAD_LOCK:
                if self._model is None:  # 双重检查，防止并发重复加载
                    print(f"[Reranker] 加载模型: {self.model_name} (device={self.device})")
                    try:
                        import torch
                        torch.set_num_threads(max(1, settings.torch_num_threads))
                        from sentence_transformers import CrossEncoder
                        self._model = CrossEncoder(
                            self.model_name,
                            device=self.device,
                        )
                        if getattr(settings, "rerank_int8_quantize", False):
                            self._apply_int8_quantization(self._model)
                        print("[Reranker] 模型加载完成")
                    except Exception as e:
                        print(f"[Reranker] 模型加载失败: {e}")
                        print("[Reranker] 使用降级方案（直接返回原始顺序）")
                        self._model = "fallback"
        return self._model

    @staticmethod
    def _apply_int8_quantization(model) -> None:
        """
        对 encoder 层的 Linear 做 int8 动态量化（CPU 推理实测 1.57x）。

        只量化 encoder：整模型量化在 torch 2.13 上会破坏 transformers 的输入解包
        （embeddings 层把 BatchEncoding 当 tensor 用，直接 AttributeError），且
        embeddings / 分类头只占计算量的 ~1%，量化收益趋近于零。失败时静默回退
        fp32（量化是加速项，不该让加载挂掉）。
        """
        try:
            import time

            import torch
            # CrossEncoder 是包装层，transformers 模型在其 .model 属性里；
            # XLM-RoBERTa 系的 encoder 路径是 .roberta.encoder，BERT 系是 .bert.encoder
            target = getattr(model, "model", model)
            core = getattr(target, "roberta", None) or getattr(target, "bert", None) or target
            encoder = getattr(core, "encoder", None)
            if encoder is None:
                print("[Reranker] 未定位到 encoder 层，跳过 int8 量化")
                return
            t0 = time.time()
            core.encoder = torch.ao.quantization.quantize_dynamic(
                encoder, {torch.nn.Linear}, dtype=torch.qint8
            )
            print(f"[Reranker] int8 动态量化完成（转换 {time.time() - t0:.1f}s，一次性，CPU 推理 ~1.57x）")
        except Exception as e:
            print(f"[Reranker] int8 量化失败（回退 fp32）: {e}")

    def rerank(
        self,
        query: str,
        candidates: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        对候选文档进行重排序

        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回 top-K 结果（默认 config.rerank_top_k）

        Returns:
            按相关度降序排列的 Document 列表
        """
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k

        # 如果只有 1 个候选或 0 个，直接返回
        if len(candidates) <= 1:
            for doc in candidates:
                doc.metadata["rerank_score"] = 1.0
            return candidates[:top_k]

        # 尝试使用 Cross-Encoder，失败则降级
        try:
            return self._rerank_with_cross_encoder(query, candidates, top_k)
        except Exception as e:
            print(f"[Reranker] Cross-Encoder 重排序失败: {e}")
            return self._rerank_fallback(query, candidates, top_k)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: list[Document],
        top_k: int,
    ) -> list[Document]:
        """使用 Cross-Encoder 进行重排序"""
        m = self.model
        if m is None or m == "fallback":
            return self._rerank_fallback(query, candidates, top_k)

        # 构造 (query, document) 对
        # 输入跨度见模块顶部的 MAX_PAIR_CHARS / MAX_PAIR_TOKENS：
        # 曾写死 200 字符，理由是「排序主要依赖开头语义」——2026-10-18 实测推翻了它
        # （200 vs 500 字 top1 只一致 44.9%）。500 字 / 512 token 已是 bge-reranker-base
        # 的位置编码上限，再长无效；换 v2-m3 才有继续放宽的空间。
        max_pair_chars = MAX_PAIR_CHARS
        query_head = query[:max_pair_chars]

        # 分数缓存：命中的对不进推理，只精排未命中的（重复问题可整批命中 → 零推理）
        cached_scores: dict[int, float] = {}
        to_predict: list[tuple[int, tuple[str, str]]] = []
        for idx, doc in enumerate(candidates):
            key = _pair_cache_key(query_head, doc)
            hit = _score_cache_get(key)
            if hit is not None:
                cached_scores[idx] = hit
            else:
                to_predict.append((idx, (query_head, doc.page_content[:max_pair_chars])))

        if to_predict:
            pairs = [p for _, p in to_predict]
            # 推理有界并行：信号量内同时最多 rerank_max_concurrent 个 CPU 推理，
            # 超出排队；比全串行吞吐高，比无界并发稳定
            with self._predict_sem:
                # token 上限跟着输入跨度走（MAX_PAIR_TOKENS=512）：只放宽字符数、
                # 不放宽 token 上限的话，500 字会被 256 token 悄悄截回 ~250 字，
                # 等于白改。旧版 API 不支持 max_length 则全量推理。
                try:
                    scores = m.predict(
                        pairs, show_progress_bar=False, max_length=MAX_PAIR_TOKENS,
                    )
                except TypeError:
                    scores = m.predict(pairs, show_progress_bar=False)
            for (idx, _p), score in zip(to_predict, scores):
                cached_scores[idx] = float(score)
                _score_cache_put(_pair_cache_key(query_head, candidates[idx]), float(score))

        # 将分数附加到 metadata
        for idx, doc in enumerate(candidates):
            doc.metadata["rerank_score"] = cached_scores.get(idx, 0.0)

        # 按分数降序排序
        ranked = sorted(candidates, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)

        return ranked[:top_k]

    def _rerank_fallback(
        self,
        query: str,
        candidates: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        降级重排序方案

        当 Cross-Encoder 不可用时：
        1. 首先使用已有的 rrf_score（来自混合检索）
        2. 如果没有，使用文档长度与 query 的简单关键词匹配
        """
        for doc in candidates:
            # 优先使用已有的 rrf_score
            if "rrf_score" in doc.metadata:
                doc.metadata["rerank_score"] = doc.metadata["rrf_score"]
            else:
                # 简单关键词匹配降级
                content = doc.page_content.lower()
                query_terms = query.lower().split()
                match_count = sum(1 for term in query_terms if term in content)
                doc.metadata["rerank_score"] = match_count / max(len(query_terms), 1)

        ranked = sorted(candidates, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        return ranked[:top_k]


# ============================================================
# 全局单例
# ============================================================

_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """获取全局 Reranker 实例（单例，延迟加载模型）"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def rerank_documents(
    query: str,
    candidates: list[Document],
    top_k: Optional[int] = None,
) -> list[Document]:
    """
    重排序文档的便捷函数

    Args:
        query: 查询文本
        candidates: 候选文档列表
        top_k: 返回 top-K 结果

    Returns:
        重排序后的 Document 列表
    """
    reranker = get_reranker()
    return reranker.rerank(query, candidates, top_k=top_k)
