"""
hybrid_search.py — 混合检索引擎

把 BM25（关键词）和 向量检索（语义）结合起来，
用 Reciprocal Rank Fusion (RRF) 合并排序。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.retrieval.embedder import get_embedder


# ============================================================
# BM25 索引管理器
# ============================================================

class BM25Index:
    """
    BM25 索引，支持增量更新和持久化
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._documents: list[Document] = []
        self._corpus: list[list[str]] = []  # 分词后的语料

    def fit(self, documents: list[Document]) -> None:
        """
        训练 BM25 索引

        Args:
            documents: Document 列表
        """
        self._documents = list(documents)
        self._corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

    def add_documents(self, documents: list[Document]) -> None:
        """
        增量添加文档到索引

        Args:
            documents: 新文档列表
        """
        if not documents:
            return
        self._documents.extend(documents)
        self._corpus.extend([self._tokenize(doc.page_content) for doc in documents])
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int = 50) -> list[tuple[Document, float]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回 top-K 结果

        Returns:
            [(Document, score), ...] 按分数降序排列
        """
        if self._bm25 is None or not self._documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 获取 top_k 索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[Document, float]] = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的结果
                results.append((self._documents[idx], float(scores[idx])))

        return results

    @property
    def document_count(self) -> int:
        """返回索引中的文档数量"""
        return len(self._documents)

    def clear(self) -> None:
        """清空索引"""
        self._bm25 = None
        self._documents = []
        self._corpus = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        分词（中文按字符 + 空格分割，英文按空格）

        对于中文 BM25，简单的按字符分割效果已足够。
        """
        import re
        # 中英文混合分词：中文按字符切，英文按词切
        tokens: list[str] = []
        for part in re.split(r"([\u4e00-\u9fff])", text):
            part = part.strip()
            if not part:
                continue
            if re.match(r"^[\u4e00-\u9fff]+$", part):
                # 中文单字
                tokens.append(part)
            else:
                # 英文/数字按空格分词
                tokens.extend(part.split())
        return [t for t in tokens if t]


# ============================================================
# 全局 BM25 索引（单例）
# ============================================================

_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """获取全局 BM25 索引实例（单例）"""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


# ============================================================
# 向量检索
# ============================================================

def vector_search(
    query: str,
    documents: list[Document],
    top_k: int = 50,
) -> list[tuple[Document, float]]:
    """
    向量检索：计算 query 与所有文档的余弦相似度

    Args:
        query: 查询文本
        documents: 候选文档列表
        top_k: 返回 top-K 结果

    Returns:
        [(Document, score), ...] 按分数降序排列
    """
    if not documents:
        return []

    embedder = get_embedder()

    # 向量化 query
    query_vec = np.array(embedder.embed_query(query), dtype=np.float32)

    # 向量化所有文档
    texts = [doc.page_content for doc in documents]
    doc_vecs = np.array(embedder.embed_documents(texts), dtype=np.float32)

    # 计算余弦相似度（向量已 L2 归一化，所以点积即余弦相似度）
    similarities = np.dot(doc_vecs, query_vec)

    # 获取 top_k 索引
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results: list[tuple[Document, float]] = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append((documents[idx], float(similarities[idx])))

    return results


# ============================================================
# 混合检索
# ============================================================

def hybrid_search(
    query: str,
    documents: list[Document],
    bm25_index: Optional[BM25Index] = None,
    top_k: int = 50,
    bm25_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[Document]:
    """
    混合检索：BM25 + 向量检索 → RRF 合并排序

    两种策略可选：
    1. RRF (推荐): score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vec(d))
    2. 加权合并: score(d) = α * bm25_score + (1-α) * vec_score

    Args:
        query: 查询文本
        documents: 候选文档列表
        bm25_index: BM25 索引（如果为 None，则临时构建）
        top_k: 最终返回 top-K 结果
        bm25_weight: 加权合并时 BM25 的权重（仅加权合并模式使用）
        rrf_k: RRF 平滑常数（默认 60）

    Returns:
        排序后的 Document 列表
    """
    if not documents:
        return []

    # ---- 1. BM25 检索 ----
    if bm25_index is not None and bm25_index.document_count > 0:
        bm25_results = bm25_index.search(query, top_k=top_k * 2)
    else:
        # 临时构建 BM25 索引
        temp_bm25 = BM25Index()
        temp_bm25.fit(documents)
        bm25_results = temp_bm25.search(query, top_k=top_k * 2)

    # ---- 2. 向量检索 ----
    vector_results = vector_search(query, documents, top_k=top_k * 2)

    # ---- 3. RRF 合并 ----
    # 构建 doc_id → rank 映射
    doc_rank_bm25: dict[str, int] = {}
    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = _get_doc_id(doc)
        doc_rank_bm25[doc_id] = rank + 1  # 1-based rank

    doc_rank_vec: dict[str, int] = {}
    for rank, (doc, _) in enumerate(vector_results):
        doc_id = _get_doc_id(doc)
        doc_rank_vec[doc_id] = rank + 1

    # 合并文档集
    all_docs_map: dict[str, Document] = {}
    for doc, _ in bm25_results + vector_results:
        doc_id = _get_doc_id(doc)
        if doc_id not in all_docs_map:
            all_docs_map[doc_id] = doc

    # 计算 RRF 分数
    doc_rrf_scores: dict[str, float] = {}
    for doc_id in all_docs_map:
        rank_bm25 = doc_rank_bm25.get(doc_id, top_k * 3)  # 未命中的给一个高排名
        rank_vec = doc_rank_vec.get(doc_id, top_k * 3)
        score = 1.0 / (rrf_k + rank_bm25) + 1.0 / (rrf_k + rank_vec)
        doc_rrf_scores[doc_id] = score

    # 按 RRF 分数排序
    sorted_doc_ids = sorted(doc_rrf_scores.keys(), key=lambda x: doc_rrf_scores[x], reverse=True)

    # 返回 top_k
    result: list[Document] = []
    for doc_id in sorted_doc_ids[:top_k]:
        doc = all_docs_map[doc_id]
        # 附加 RRF 分数到 metadata
        doc.metadata["rrf_score"] = doc_rrf_scores[doc_id]
        result.append(doc)

    return result


def _get_doc_id(doc: Document) -> str:
    """获取文档的唯一标识"""
    # 优先使用 content 哈希作为 ID
    import hashlib
    content = doc.page_content[:200]  # 取前 200 字符
    source = doc.metadata.get("source", "")
    heading = doc.metadata.get("heading", "")
    unique_str = f"{content}|{source}|{heading}"
    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()
