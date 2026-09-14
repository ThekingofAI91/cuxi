"""
ingest.py — 把纯文本（如自建角色的背景）灌入 ChromaDB collection

复用项目已验证的链路：
- chunk_text()：AdaptiveChunker 自适应分块（与内置角色入库同一套逻辑）
- get_embedder()：bge-small-zh 向量化
- get_chroma_client()：线程安全的 PersistentClient 单例

自建角色的背景被标记为 source_type="anchor"（权重 2.0），
因为用户提交的背景就是这个角色的"事实锚点"，应当高召回、高可信。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.documents import Document

from src.core.config import settings
from src.retrieval.chunker import chunk_text
from src.retrieval.embedder import get_embedder


def _chunk_and_embed(text: str, source: str) -> tuple[list[Document], list[list[float]]]:
    """分块 + 来源标记 + 向量化，返回 (chunks, vectors)"""
    chunks = chunk_text(
        text,
        source=source,
        min_size=settings.chunk_size // 2,
        max_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    # 自建背景即"事实锚点"，检索时高权重召回
    for c in chunks:
        c.metadata.setdefault("source_type", "anchor")
    embedder = get_embedder()
    vectors, _metadatas = embedder.embed_documents_with_metadata(chunks)
    return chunks, vectors


def ingest_texts(
    collection_name: str,
    character_id: str,
    text: str,
    source: str = "custom_background",
) -> int:
    """
    把 text 分块向量化后写入指定 ChromaDB collection（覆盖式：先清空同名 collection）。

    返回写入的文档块数量。text 为空或全空白时返回 0（不入空库）。
    """
    if not text or not text.strip():
        return 0

    chunks, vectors = _chunk_and_embed(text, source)
    if not chunks:
        return 0

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from framework.supervisor import get_chroma_client
    from src.retrieval.advanced_search import invalidate_bm25_cache

    client = get_chroma_client()

    # 覆盖式：先删旧 collection，保证重建知识库时干净
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    n = len(chunks)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_ids = [f"{character_id}_{j}" for j in range(i, end)]
        batch_texts = [chunks[j].page_content for j in range(i, end)]
        batch_metadatas = [chunks[j].metadata for j in range(i, end)]
        batch_vectors = vectors[i:end]
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_vectors,
            metadatas=batch_metadatas,
        )

    # 新库就绪：失效 BM25 磁盘缓存，下次检索自动重建
    invalidate_bm25_cache(collection_name)
    return n


async def ingest_texts_async(collection_name: str, character_id: str, text: str,
                             source: str = "custom_background") -> int:
    """异步包装：把 CPU 密集的向量化放到线程池，避免阻塞事件循环"""
    return await asyncio.to_thread(ingest_texts, collection_name, character_id, text, source)
