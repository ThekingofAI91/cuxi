"""
reranker.py — Cross-Encoder 重排序

混合检索之后做精排，确保最相关的排在前面。
使用 Cross-Encoder 对 (query, document) 对进行深度语义匹配。
"""

from __future__ import annotations

import threading
from typing import Optional

from langchain_core.documents import Document

from src.core.config import settings


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
    # - 模型加载锁：防止并发请求同时加载 560M 模型（重复加载浪费 10-30s 与数 GB 内存）
    # - 推理锁：Cross-Encoder CPU 推理是全局争抢点，多请求同时推理会把 CPU 打满、
    #   所有请求一起变慢 3 倍以上；串行化后排队等待（每个 ~2-4s），体验远好于互相拖死
    _LOAD_LOCK = threading.Lock()
    _PREDICT_LOCK = threading.Lock()

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
        self.model_name = model_name or "BAAI/bge-reranker-v2-m3"
        self.device = device or "cpu"
        self._model = None

    @property
    def model(self):
        """延迟加载 Cross-Encoder 模型（线程安全，只加载一次）"""
        if self._model is None:
            with self._LOAD_LOCK:
                if self._model is None:  # 双重检查，防止并发重复加载
                    print(f"[Reranker] 加载模型: {self.model_name} (device={self.device})")
                    try:
                        from sentence_transformers import CrossEncoder
                        self._model = CrossEncoder(
                            self.model_name,
                            device=self.device,
                        )
                        print("[Reranker] 模型加载完成")
                    except Exception as e:
                        print(f"[Reranker] ⚠️ 模型加载失败: {e}")
                        print("[Reranker] 使用降级方案（直接返回原始顺序）")
                        self._model = "fallback"
        return self._model

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
        pairs = [(query, doc.page_content) for doc in candidates]

        # 推理加锁：CPU 推理串行化，避免并发请求互相抢占 CPU 导致集体变慢
        # （锁内持有时间 ~2-4s，排队等待远好于争抢 CPU 使所有请求一起卡死）
        with self._PREDICT_LOCK:
            # 限制输入长度：chunk 通常数百字，尾部对排序贡献小，
            # 截断到 256 token 可显著加速 CPU 推理（512→256 约省一半时间）；
            # 旧版 API 不支持则全量推理
            try:
                scores = m.predict(pairs, show_progress_bar=False, max_length=256)
            except TypeError:
                scores = m.predict(pairs, show_progress_bar=False)

        # 将分数附加到 metadata
        for doc, score in zip(candidates, scores):
            doc.metadata["rerank_score"] = float(score)

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
