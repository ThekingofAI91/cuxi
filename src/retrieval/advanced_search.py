"""
advanced_search.py — 高级检索策略

检索增强：
1. BM25（关键词检索）：精确命中专有名词（如《红书》、积极想象等）
2. Cross-Encoder 重排序：对 RRF 合并后的候选做深度语义精排

向量与 BM25 两路结果通过 RRF 合并，再经重排序输出 top-K。

性能优化（优化十七）：向量与 BM25 并行、BM25 仅查询原问题、重排序截断输入。

注：Multi-Query / HyDE 查询改写已于 2026-10-18 整体删除（三轮 A/B 实测：
成功率 0%→23%→93% 三档变化下四维分数全在噪声层，且回答耗时 +3s，属结构性冗余）。
"""

from __future__ import annotations

import asyncio
import hashlib
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.core.config import settings


# ============================================================
# 全局缓存（全部文档 + BM25 索引，避免重复加载）
# ============================================================

_bm25_cache: dict[str, tuple[list[Document], Optional[object]]] = {}

# BM25 构建锁：全量拉取 + 分词构建是 CPU/IO 密集操作（数万条文档需数十秒），
# 多个并发请求同时构建会重复拉取/分词并互相抢占 CPU，必须保证进程内只构建一次。
# 其余并发请求在锁内发现缓存已就绪后直接复用（等待时间 ~0），
# 而不是像之前那样各自从头构建一遍导致全体请求一起卡死。
_bm25_build_lock = threading.Lock()


def _bm25_cache_path(name: str) -> Path:
    """BM25 索引磁盘缓存路径（按 collection 名，服务重启后可复用）"""
    return Path(settings.chroma_persist_dir) / "bm25_cache" / f"{name}.pkl"


def _load_bm25_from_disk(name: str, expected_count: Optional[int] = None):
    """
    从磁盘加载 BM25 索引。

    expected_count 为 None 时跳过文档数校验（用于预热/缓存优先路径，
    避免先全量拉取 10 万条文档才能校验）；传入具体数字时校验
    （文档更新后 count 变化会检测到，自动重建）。
    """
    try:
        path = _bm25_cache_path(name)
        if not path.exists():
            return None
        with path.open("rb") as f:
            data = pickle.load(f)
        if expected_count is not None and data.get("count") != expected_count:
            print(f"[Advanced Retrieval] BM25 磁盘缓存过期（{data.get('count')} != {expected_count}），重建")
            return None
        print(f"[Advanced Retrieval] BM25 磁盘缓存命中: {data.get('count')} 条文档")
        return data.get("index")
    except Exception as e:
        print(f"[Advanced Retrieval] BM25 磁盘缓存加载失败: {e}")
        return None


def _save_bm25_to_disk(name: str, count: int, index) -> None:
    """持久化 BM25 索引到磁盘，避免服务重启后重新构建（10 万+ 文档构建需数秒~十几秒）"""
    try:
        path = _bm25_cache_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"count": count, "index": index}, f)
        print(f"[Advanced Retrieval] BM25 索引已持久化: {path}")
    except Exception as e:
        print(f"[Advanced Retrieval] BM25 持久化失败: {e}")


def _ensure_bm25_ready(collection) -> tuple[list[Document], object]:
    """
    线程安全地确保 BM25 索引就绪：返回 (全部文档, BM25 索引)。

    在 asyncio.to_thread 中执行（全量拉取/分词/打分耗 CPU，直接跑在事件循环
    会阻塞所有并发请求）。整体加锁保证并发请求下只执行一次全量拉取 + 构建，
    其余请求在锁内直接复用缓存。

    优化：优先加载磁盘缓存。pkl 内已序列化全部文档（BM25Index._documents），
    命中时直接复用，完全跳过“分页拉取 10 万条文档”这一分钟级操作；
    只有磁盘缓存不存在时才全量拉取 + 构建。
    """
    name = collection.name
    cached = _bm25_cache.get(name)
    if cached is not None and cached[0] and cached[1] is not None:
        return cached

    with _bm25_build_lock:
        # 双重检查：等待锁期间其他线程可能已完成构建
        cached = _bm25_cache.get(name)
        if cached is not None and cached[0] and cached[1] is not None:
            return cached

        # ---- 路径 1：磁盘缓存命中（含全部文档），免全量拉取 ----
        path = _bm25_cache_path(name)
        if path.exists():
            index = _load_bm25_from_disk(name, None)
            all_docs = list(getattr(index, "_documents", []) or []) if index is not None else []
            if all_docs and index is not None:
                _bm25_cache[name] = (all_docs, index)
                print(f"[Advanced Retrieval] BM25 磁盘缓存命中（含全部文档，免全量拉取）: {len(all_docs)} 条")
                return all_docs, index
            print(f"[Advanced Retrieval] BM25 磁盘缓存不完整（缺文档），降级全量拉取重建")

        # ---- 路径 2：全量拉取 + 构建/加载 + 持久化（只执行一次）----
        all_docs = _load_all_documents(collection)
        index = _load_bm25_from_disk(name, len(all_docs))
        if index is None:
            from src.retrieval.hybrid_search import BM25Index
            index = BM25Index()
            index.fit(all_docs)
            _save_bm25_to_disk(name, len(all_docs), index)

        _bm25_cache[name] = (all_docs, index)
        return all_docs, index


def invalidate_bm25_cache(name: str):
    """上传文档后使 BM25 缓存失效（内存 + 磁盘），下次检索自动重建，
    避免检索结果不包含新上传的文档。"""
    _bm25_cache.pop(name, None)
    try:
        path = _bm25_cache_path(name)
        if path.exists():
            path.unlink()
            print(f"[Advanced Retrieval] BM25 缓存已失效: {path}")
    except Exception as e:
        print(f"[Advanced Retrieval] BM25 缓存失效失败: {e}")


def _load_all_documents(collection) -> list[Document]:
    """
    加载 collection 全部文档（带缓存，分页拉取）

    用于构建 BM25 关键词索引。文档量级为数千条 chunk，
    内存占用可接受，且只加载一次。

    注意：必须分页（limit/offset），否则 ChromaDB 内部
    用 `WHERE id IN (...)` 一次查询全部 ID，会触发
    SQLite "too many SQL variables" 错误（SQLITE_MAX_VARIABLE_NUMBER）。
    """
    name = collection.name
    if name in _bm25_cache:
        return _bm25_cache[name][0]

    docs: list[Document] = []
    offset = 0
    page_size = 500  # 10 万条文档时 200/页 需 550 次 SQLite 查询，500/页 减到 220 次
    while True:
        raw = collection.get(
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        if raw is None:
            print("[Advanced Retrieval] collection.get() 返回 None，终止全量拉取")
            break
        ids = raw.get("ids") or []
        if not ids:
            break
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        for text, meta in zip(documents, metadatas):
            docs.append(Document(page_content=text or "", metadata=meta or {}))
        offset += page_size
        if len(ids) < page_size:
            break

    _bm25_cache[name] = (docs, None)
    print(f"[Advanced Retrieval] 加载全部文档: {len(docs)} 条 (collection={name})")
    return docs


def _get_doc_id(text: str) -> str:
    """用内容哈希作为 doc_id（与向量路保持一致，保证去重/合并正确）"""
    return hashlib.md5(text[:200].encode("utf-8")).hexdigest()


# ============================================================
# ============================================================
# 高级检索引擎
# ============================================================

async def advanced_retrieval(
    question: str,
    collection,
    top_k: int = 15,
    use_bm25: bool = True,
    use_rerank: bool = True,
    use_source_weight: bool = True,
) -> tuple[list[Document], list[str]]:
    """
    高级检索：向量 + BM25 混合召回，RRF 合并后 Cross-Encoder 精排

    流程：
    1. 原始查询并行跑向量检索与 BM25 关键词检索（精确命中专有名词）
    2. 两路结果用 RRF 合并，取 top_k 条候选
    3. Cross-Encoder 对前 rerank_candidates 条精排，其余保持 RRF 顺序，最终返回 top_k 条

    Args:
        question: 原始问题
        collection: ChromaDB collection
        top_k: 最终返回数量
        use_bm25: 是否使用 BM25 关键词检索
        use_rerank: 是否使用 Cross-Encoder 重排序
        use_source_weight: 是否按语料来源类型加权（original/oral/anchor 优先，
            secondary 二手解读强降权，避免第三者评价冒充名人原话）

    Returns:
        (retrieved_docs, all_contexts) 元组
    """
    from src.retrieval.embedder import get_embedder

    embedder = get_embedder()


    fetch_k = top_k * 2  # 多取一些用于合并

    async def _vector_pass(queries: list[str]) -> list[list[str]]:
        """批量向量检索：一次批量 embed + 一次 ChromaDB 批量查询，返回每路 ranked doc_ids。"""
        if not queries:
            return []
        embeddings = await asyncio.to_thread(embedder.embed_documents, queries)
        query_result = await asyncio.to_thread(
            collection.query,
            query_embeddings=embeddings,
            n_results=min(fetch_k, 50),
            include=["documents", "metadatas", "distances"],
        )
        ranked_groups: list[list[str]] = []
        if query_result and query_result["documents"]:
            for group_idx in range(len(query_result["documents"])):
                ranked_ids: list[str] = []
                group_texts = query_result["documents"][group_idx] or []
                group_metas = query_result["metadatas"][group_idx] if query_result["metadatas"] else None
                group_dists = query_result["distances"][group_idx] if query_result["distances"] else None
                for i, text in enumerate(group_texts):
                    if text is None:
                        continue
                    metadata = (group_metas or [{}] * len(group_texts))[i] if group_metas else {}
                    distance = (group_dists or [0.0] * len(group_texts))[i] if group_dists else 0
                    doc_id = _get_doc_id(text)
                    if doc_id not in doc_candidates:
                        doc_candidates[doc_id] = {
                            "text": text,
                            "metadata": metadata or {},
                            "distance": distance,
                        }
                    ranked_ids.append(doc_id)
                ranked_groups.append(ranked_ids)
        return ranked_groups

    async def _bm25_pass(queries: list[str], bm25_index) -> list[list[str]]:
        """BM25 关键词检索（多路并行放线程池），返回每路 ranked doc_ids。"""
        ranked_groups: list[list[str]] = []
        tasks = [asyncio.to_thread(bm25_index.search, bq, top_k * 2) for bq in queries]
        for _bq, bm25_results in zip(queries, await asyncio.gather(*tasks)):
            if not bm25_results:
                continue
            ranked_ids = []
            for doc, _score in bm25_results:
                doc_id = _get_doc_id(doc.page_content)
                if doc_id not in doc_candidates:
                    doc_candidates[doc_id] = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "distance": 0.0,
                    }
                ranked_ids.append(doc_id)
            ranked_groups.append(ranked_ids)
        return ranked_groups

    doc_candidates: dict[str, dict] = {}   # doc_id -> {text, metadata, distance}
    query_results_list: list[list[str]] = []  # 每路的 ranked doc_ids（供 RRF）

    # 阶段耗时埋点（延迟瀑布；无请求上下文时 stage_mark 自动忽略）
    from src.core.logger import stage_mark

    _retrieval_t0 = time.perf_counter()

    # ---- 第一轮：原始查询的向量 + BM25（两者相互独立，并行执行）----
    # 向量路只依赖 embedder，先发车；BM25 索引加载/构建（磁盘冷启动可达数秒）
    # 放到并行任务里，不再阻塞向量路。RRF 按各路内部名次打分，两路结果的
    # 合并顺序不影响得分。并发写 doc_candidates 的代码段内无 await，事件循环
    # 单线程语义下天然互斥，无需加锁。
    vec_task = asyncio.create_task(_vector_pass([question]))

    async def _bm25_first_pass():
        bm25_ready = await asyncio.to_thread(_ensure_bm25_ready, collection)
        print(f"[Advanced Retrieval] BM25 索引就绪: {bm25_ready[1].document_count} 条文档")
        return bm25_ready[1], await _bm25_pass([question], bm25_ready[1])

    bm25_task = asyncio.create_task(_bm25_first_pass()) if use_bm25 else None

    try:
        query_results_list.extend(await vec_task)
    except Exception:
        # 向量路失败时取消仍在跑的 BM25 任务，避免"异常从未被读取"告警
        if bm25_task is not None and not bm25_task.done():
            bm25_task.cancel()
        raise

    if bm25_task is not None:
        try:
            _index, first_groups = await bm25_task
            query_results_list.extend(first_groups)
        except Exception as e:
            print(f"[Advanced Retrieval] BM25 检索失败，跳过: {e}")

    # ---- 收两路结果（向量 + BM25 均只跑原始查询）----
    stage_mark("retrieval_ms", (time.perf_counter() - _retrieval_t0) * 1000)

    # ---- Step 3: RRF 合并（按语料来源类型加权）----
    rrf_k = 60
    doc_rrf_scores: dict[str, float] = {}

    # 来源加权：oral（口述/讲座/回忆录）优先召回学说话方式，
    # original（原著）支撑观点，anchor（core_ideas）锚定关键事实，
    # secondary（二手解读）强降权防止第三者评价冒充名人原话。
    from src.retrieval.source_profile import classify_source, get_source_weight

    for ranked_ids in query_results_list:
        for rank, doc_id in enumerate(ranked_ids):
            if doc_id not in doc_rrf_scores:
                doc_rrf_scores[doc_id] = 0.0
            if use_source_weight:
                meta = doc_candidates[doc_id].get("metadata") or {}
                source = meta.get("source", "")
                weight = get_source_weight(source)
            else:
                weight = 1.0
            doc_rrf_scores[doc_id] += weight / (rrf_k + rank + 1)

    # 按 RRF 分数排序
    sorted_doc_ids = sorted(
        doc_rrf_scores.keys(),
        key=lambda x: doc_rrf_scores[x],
        reverse=True,
    )

    # ---- Step 4: 构建候选（多取一些供重排序选择）----
    candidate_docs: list[Document] = []
    # 候选数与最终 top_k 一致：RRF 前 top_k 已覆盖高相关文档，
    # 候选越少，CPU 重排（560M 参数）越快（20 对 → 15 对，约省 25% 推理时间）
    candidate_limit = top_k

    for doc_id in sorted_doc_ids[:candidate_limit]:
        candidate = doc_candidates[doc_id]
        metadata = dict(candidate["metadata"])
        metadata["rrf_score"] = round(doc_rrf_scores[doc_id], 6)
        # 附加语料来源类型（original/oral/secondary/...），便于溯源与下游提示
        if "source_type" not in metadata:
            metadata["source_type"] = classify_source(metadata.get("source", ""))

        candidate_docs.append(
            Document(
                page_content=candidate["text"],
                metadata=metadata,
            )
        )

    # ---- Step 5: Cross-Encoder 重排序（只精排 RRF 前 N 条，其余按 RRF 顺序兜底）----
    # 重组后块为完整段落（~1000 字符），若对全部候选（默认 8 条 = top_k）都做 CPU 精排会显著拖慢单请求；
    # 只精排 rerank_candidates（默认 5）对，剩余候选保持 RRF 顺序拼接，质量损失小、耗时减半。
    rerank_n = settings.rerank_candidates
    head = candidate_docs[:rerank_n]
    tail = candidate_docs[rerank_n:]
    if use_rerank and len(head) > 1:
        try:
            from src.retrieval.reranker import rerank_documents
            # CPU 推理（默认 5 对 × 256 token）需数秒，同步调用会阻塞事件循环，放线程池
            _rerank_t0 = time.perf_counter()
            head = await asyncio.to_thread(rerank_documents, question, head, top_k=rerank_n)
            stage_mark("rerank_ms", (time.perf_counter() - _rerank_t0) * 1000)
            print(f"[Advanced Retrieval] Cross-Encoder 重排序完成（前 {len(head)} 条精排 + {len(tail)} 条 RRF 兜底）")
        except Exception as e:
            print(f"[Advanced Retrieval] 重排序失败，保持 RRF 顺序: {e}")
    candidate_docs = head + tail

    retrieved_docs = candidate_docs[:top_k]
    contexts = [doc.page_content for doc in retrieved_docs]

    # 打印最终结果的来源类型分布（便于验证来源加权效果）
    type_counts: dict[str, int] = {}
    for doc in retrieved_docs:
        st = doc.metadata.get("source_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
    print(f"[Advanced Retrieval] 最终返回 {len(retrieved_docs)} 条文档 | 来源分布: {type_counts}")

    return retrieved_docs, contexts
