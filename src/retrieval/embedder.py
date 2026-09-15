"""
embedder.py — 文本向量化模块

使用 sentence-transformers 将文本转换为向量，
同时支持查询(query)和文档(document)的嵌入。
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document

from src.core.config import settings


# ============================================================
# 嵌入器
# ============================================================

class Embedder:
    """
    文本向量化器
    
    封装 sentence-transformers 模型，提供统一的嵌入接口。
    
    用法:
        embedder = Embedder()
        vectors = embedder.embed_documents(["文本1", "文本2"])
        query_vec = embedder.embed_query("用户问题")
    """

    # 类级锁：多线程首次加载模型时防止重复初始化（模型文件大，重复加载浪费内存/时间）
    _LOAD_LOCK = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: 模型名称（默认从 config 读取）
            device: 运行设备（cpu / cuda），默认从 config 读取
            cache_dir: 嵌入缓存目录（None 表示不缓存）
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None  # 延迟加载
        self._dimension: Optional[int] = None
        # 向量化有界并行：sentence-transformers CPU encode 是资源争抢点，
        # 用信号量限制并发数，避免高并发时互相抢占（默认 embedding_max_concurrent=4）
        self._encode_sem = threading.BoundedSemaphore(max(1, settings.embedding_max_concurrent))

        # 嵌入缓存（可选）
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        """延迟加载模型（只在首次调用时加载，线程安全）"""
        if self._model is None:
            with self._LOAD_LOCK:
                if self._model is None:  # 双重检查，防止并发重复加载
                    print(f"[Embedder] 加载模型: {self.model_name} (device={self.device})")
                    try:
                        import torch
                        torch.set_num_threads(max(1, settings.torch_num_threads))
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(
                            self.model_name,
                            device=self.device,
                        )
                        self._dimension = self._model.get_embedding_dimension()
                        print(f"[Embedder] 模型加载完成，嵌入维度: {self._dimension}")
                    except Exception as e:
                        print(f"[Embedder] 模型加载失败: {e}")
                        print("[Embedder] 使用降级方案（随机向量）— 仅用于测试")
                        self._dimension = settings.embedding_dimension
        return self._model

    @property
    def dimension(self) -> int:
        """返回嵌入向量维度"""
        if self._dimension is None:
            _ = self.model  # 触发加载
        return self._dimension or settings.embedding_dimension

    def embed_query(self, text: str) -> list[float]:
        """
        对单个查询文本进行向量化
        
        Args:
            text: 查询文本
        
        Returns:
            float 列表表示的向量
        """
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        对多个文档文本进行批量向量化
        
        Args:
            texts: 文本列表
        
        Returns:
            float 二维列表 [n_texts, dimension]
        """
        if not texts:
            return []

        # 检查缓存
        if self.cache_dir:
            cached = self._batch_check_cache(texts)
            if cached is not None:
                return cached

        vectors = self._embed(texts)

        # 写入缓存
        if self.cache_dir:
            self._batch_write_cache(texts, vectors)

        return vectors

    def embed_documents_with_metadata(
        self, documents: list[Document]
    ) -> tuple[list[list[float]], list[dict]]:
        """
        对 Document 列表进行向量化，同时保留 metadata
        
        Args:
            documents: Document 列表
        
        Returns:
            (vectors, metadatas) 元组
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectors = self.embed_documents(texts)
        return vectors, metadatas

    # ---------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """实际执行嵌入的核心方法"""
        if self.model is None:
            # 降级方案：返回随机向量（仅用于测试）
            print("[Embedder] 使用降级嵌入（随机向量）")
            rng = np.random.default_rng(42)
            return rng.random((len(texts), self.dimension)).tolist()

        try:
            with self._encode_sem:
                embeddings = self.model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # L2 归一化，提高余弦相似度计算效率
                )
            return embeddings.tolist()
        except Exception as e:
            print(f"[Embedder] 嵌入过程出错: {e}")
            # 降级
            rng = np.random.default_rng(42)
            return rng.random((len(texts), self.dimension)).tolist()

    # ---- 缓存相关 ----

    def _get_text_hash(self, text: str) -> str:
        """计算文本的 SHA256 哈希（作为缓存 key）"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _get_cache_path(self, text_hash: str) -> Path:
        """获取缓存文件路径"""
        if self.cache_dir is None:
            raise ValueError("cache_dir 未设置")
        return self.cache_dir / f"{text_hash}.json"

    def _check_cache(self, text: str) -> Optional[list[float]]:
        """检查单个文本的缓存"""
        if self.cache_dir is None:
            return None
        cache_path = self._get_cache_path(self._get_text_hash(text))
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _batch_check_cache(self, texts: list[str]) -> Optional[list[list[float]]]:
        """批量检查缓存，全部命中才返回"""
        if self.cache_dir is None:
            return None
        vectors: list[list[float]] = []
        for text in texts:
            cached = self._check_cache(text)
            if cached is None:
                return None  # 任一未命中就退出
            vectors.append(cached)
        print(f"[Embedder] 缓存命中 {len(texts)} 条")
        return vectors

    def _write_cache(self, text: str, vector: list[float]) -> None:
        """写入单条缓存"""
        if self.cache_dir is None:
            return
        cache_path = self._get_cache_path(self._get_text_hash(text))
        try:
            cache_path.write_text(json.dumps(vector, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            pass  # 缓存写入失败不影响主流程

    def _batch_write_cache(self, texts: list[str], vectors: list[list[float]]) -> None:
        """批量写入缓存"""
        if self.cache_dir is None:
            return
        for text, vec in zip(texts, vectors):
            self._write_cache(text, vec)

    def clear_cache(self) -> None:
        """清空嵌入缓存"""
        if self.cache_dir and self.cache_dir.exists():
            count = 0
            for f in self.cache_dir.iterdir():
                if f.suffix == ".json":
                    f.unlink()
                    count += 1
            print(f"[Embedder] 已清除 {count} 条缓存")


# ============================================================
# 全局单例
# ============================================================

_embedder: Optional[Embedder] = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    """获取全局 Embedder 实例（单例，延迟加载模型，线程安全）"""
    global _embedder
    if _embedder is None:
        with _EMBEDDER_LOCK:
            if _embedder is None:  # 双重检查，防止并发创建多个实例
                _embedder = Embedder()
    return _embedder
