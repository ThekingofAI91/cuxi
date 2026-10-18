"""
hybrid_search.py — BM25 关键词索引

仅保留被实际使用的 BM25Index（高级混合检索 hybrid_search / vector_search 已废弃，
检索主链路统一走 advanced_search.advanced_retrieval：Multi-Query + HyDE + BM25 + Cross-Encoder 重排）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


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
