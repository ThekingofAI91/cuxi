"""
advanced_search.py — 高级检索策略

实现多种检索增强：
1. Multi-Query（多查询检索）：LLM 生成多个查询变体，分别检索后合并
2. HyDE（假设文档嵌入）：LLM 先生成假设答案，用答案向量辅助检索
3. BM25（关键词检索）：精确命中专有名词（如《红书》、积极想象等）
4. Cross-Encoder 重排序：对 RRF 合并后的候选做深度语义精排

多路结果通过 RRF 合并，再经重排序输出 top-K。

性能优化（优化十七）：向量检索多路并行、BM25 仅查询原问题、重排序截断输入。
"""

from __future__ import annotations

import asyncio
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

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
        print(f"[Advanced Retrieval] ⚠️ BM25 磁盘缓存加载失败: {e}")
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
        print(f"[Advanced Retrieval] ⚠️ BM25 持久化失败: {e}")


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
            print(f"[Advanced Retrieval] 🗑️ BM25 缓存已失效: {path}")
    except Exception as e:
        print(f"[Advanced Retrieval] ⚠️ BM25 缓存失效失败: {e}")


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
        ids = raw["ids"]
        if not ids:
            break
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta or {}))
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
# 查询变体生成（Multi-Query）+ HyDE 合并调用
# ============================================================
# 性能优化：Multi-Query 与 HyDE 原本是两次独立 LLM 调用（串行等待两次 API 往返），
# 合并为一次调用同时产出变体与假设文档，省掉一半 LLM 延迟。

_MERGE_QUERY_HYDE_PROMPT = """你是检索优化专家。给定下面的用户问题，请完成两项任务：

任务1：生成 {num_variants} 个不同角度的检索查询。要求：
- 每个查询从不同角度表达相同的信息需求
- 使用不同的关键词和表述方式
- 保持查询简洁（不超过15个字）
- 适合用于向量检索系统

任务2：写一段3-5句话的简短参考答案，尽量使用专业术语和关键概念
（这段文字将用于检索相关文档）。

原始问题：{question}

输出格式（严格按此格式）：
【查询】
查询1
查询2
【假设答案】
假设答案内容"""


async def generate_query_variants_and_hyde(
    question: str,
    llm: ChatOpenAI,
    num_variants: int = 2,
) -> tuple[list[str], str]:
    """
    一次 LLM 调用同时生成查询变体 + HyDE 假设文档

    Returns:
        (查询变体列表, HyDE 假设文档)；失败时返回 ([], "")
    """
    prompt = _MERGE_QUERY_HYDE_PROMPT.format(
        question=question,
        num_variants=num_variants,
    )

    try:
        response = await llm.ainvoke([("user", prompt)])
        content = response.content.strip()

        variants: list[str] = []
        hyde_doc: str = ""
        q_marker = "【查询】"
        a_marker = "【假设答案】"

        if q_marker in content and a_marker in content:
            q_part = content.split(q_marker)[1].split(a_marker)[0]
            variants = [line.strip() for line in q_part.split("\n") if line.strip()]
            hyde_doc = content.split(a_marker)[1].strip()
        else:
            # 格式解析失败：降级为把每行当查询，HyDE 留空
            variants = [line.strip() for line in content.split("\n") if line.strip()]

        return variants[:num_variants], hyde_doc
    except Exception as e:
        print(f"[Multi-Query+HyDE] 生成失败: {e}")
        return [], ""


# ============================================================
# 高级检索引擎
# ============================================================

async def advanced_retrieval(
    question: str,
    collection,
    llm: ChatOpenAI,
    top_k: int = 15,
    use_multi_query: bool = True,
    use_hyde: bool = True,
    use_bm25: bool = True,
    use_rerank: bool = True,
    num_variants: int = 2,
    use_source_weight: bool = True,
) -> tuple[list[Document], list[str]]:
    """
    高级检索：Multi-Query + HyDE + BM25 + Cross-Encoder 重排序

    流程：
    1. 并行生成查询变体 + HyDE 假设文档
    2. 对每个查询（原始 + 变体 + HyDE）分别做向量检索
    3. 对原始 + 变体查询做 BM25 关键词检索（精确命中专有名词）
    4. 全部路结果用 RRF 合并，取 top_k*2 候选
    5. Cross-Encoder 对候选做重排序，返回 top-K

    Args:
        question: 原始问题
        collection: ChromaDB collection
        llm: LLM 实例
        top_k: 最终返回数量
        use_multi_query: 是否使用多查询
        use_hyde: 是否使用 HyDE
        use_bm25: 是否使用 BM25 关键词检索
        use_rerank: 是否使用 Cross-Encoder 重排序
        num_variants: 查询变体数量
        use_source_weight: 是否按语料来源类型加权（original/oral/anchor 优先，
            secondary 二手解读强降权，避免第三者评价冒充名人原话）

    Returns:
        (retrieved_docs, all_contexts) 元组
    """
    from src.retrieval.embedder import get_embedder

    embedder = get_embedder()

    # ---- Step 1: 一次 LLM 调用同时生成查询变体和 HyDE 文档 ----
    if use_multi_query or use_hyde:
        query_variants, hyde_doc = await generate_query_variants_and_hyde(question, llm, num_variants)
    else:
        query_variants, hyde_doc = [], ""

    # 构建所有查询列表
    all_queries = [question]  # 原始查询
    all_queries.extend(query_variants)
    if hyde_doc:
        all_queries.append(hyde_doc)

    print(f"[Advanced Retrieval] 共 {len(all_queries)} 个查询:")
    for i, q in enumerate(all_queries):
        label = "原始" if i == 0 else (f"HyDE" if i == len(all_queries) - 1 and hyde_doc else "变体")
        print(f"  [{label}] {q[:60]}...")

    # ---- Step 2: 对全部查询做批量向量检索（一次批量 embed + 一次 collection.query）----
    # 性能优化：之前每路单独 embed + 单独 query（4 次 ChromaDB 调用），
    # SQLite 内部有锁导致互相竞争；合并为 1 次批量嵌入 + 1 次批量查询。
    doc_candidates: dict[str, dict] = {}  # doc_id -> {doc, metadata}
    query_results_list: list[list[str]] = []  # 每路的 ranked doc_ids

    fetch_k = top_k * 2  # 多取一些用于合并

    # 一次批量 encode 全部查询（sentence-transformers 内部 batch 并行）
    embeddings = await asyncio.to_thread(embedder.embed_documents, all_queries)

    # 一次 ChromaDB 查询全部向量（单次调用返回 len(all_queries) 组结果）
    query_result = await asyncio.to_thread(
        collection.query,
        query_embeddings=embeddings,
        n_results=min(fetch_k, 50),
        include=["documents", "metadatas", "distances"],
    )

    if query_result and query_result["documents"]:
        for group_idx in range(len(query_result["documents"])):
            ranked_ids: list[str] = []
            group_texts = query_result["documents"][group_idx] or []
            group_metas = query_result["metadatas"][group_idx] if query_result["metadatas"] else None
            group_dists = query_result["distances"][group_idx] if query_result["distances"] else None
            for i, text in enumerate(group_texts):
                metadata = (group_metas or [{}] * len(group_texts))[i] if group_metas else {}
                distance = (group_dists or [0.0] * len(group_texts))[i] if group_dists else 0

                doc_id = _get_doc_id(text)

                if doc_id not in doc_candidates:
                    doc_candidates[doc_id] = {
                        "text": text,
                        "metadata": metadata,
                        "distance": distance,
                    }

                ranked_ids.append(doc_id)
            query_results_list.append(ranked_ids)

    # ---- Step 2.5: BM25 关键词检索 ----
    if use_bm25:
        try:
            # 全量拉取与索引构建/加载都是同步 CPU/IO 密集操作，放线程池避免阻塞事件循环；
            # 内部整体加锁，并发请求只构建一次，其余直接复用缓存
            all_docs, bm25_index = await asyncio.to_thread(_ensure_bm25_ready, collection)

            print(f"[Advanced Retrieval] BM25 索引就绪: {bm25_index.document_count} 条文档")

            # BM25 查询：原始 + 变体
            # （变体含补充关键词（如“正面意义”“双重性”），BM25 精确命中可弥补
            #  向量检索漏召回的文档；6 题小评估验证：仅原查询使 Q5 阴影
            #  上下文 8→6、要点 8→6，故保留变体查询）
            bm25_queries = [question] + query_variants
            # 打分 O(10万条)，同步调用会卡死服务器，必须放线程池；
            # 3 路并行（多核 CPU）替代串行，10 万条库从 ~3-6s 降到 ~1-2s
            bm25_tasks = [
                asyncio.to_thread(bm25_index.search, bq, top_k * 2)
                for bq in bm25_queries
            ]
            for bq, bm25_results in zip(bm25_queries, await asyncio.gather(*bm25_tasks)):
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
                query_results_list.append(ranked_ids)
        except Exception as e:
            print(f"[Advanced Retrieval] ⚠️ BM25 检索不可用，跳过: {e}")

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
                source = doc_candidates[doc_id]["metadata"].get("source", "")
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
    # 重组后块为完整段落（~1000 字符），全候选精排在 CPU 上 15 对要 20s+；
    # 只精排 rerank_candidates（默认 10）对，剩余候选保持 RRF 顺序拼接，质量损失小、耗时减半。
    rerank_n = settings.rerank_candidates
    head = candidate_docs[:rerank_n]
    tail = candidate_docs[rerank_n:]
    if use_rerank and len(head) > 1:
        try:
            from src.retrieval.reranker import rerank_documents
            # CPU 推理（30 对 × 512 token）需数秒，同步调用会阻塞事件循环，放线程池
            head = await asyncio.to_thread(rerank_documents, question, head, top_k=rerank_n)
            print(f"[Advanced Retrieval] Cross-Encoder 重排序完成（前 {len(head)} 条精排 + {len(tail)} 条 RRF 兜底）")
        except Exception as e:
            print(f"[Advanced Retrieval] ⚠️ 重排序失败，保持 RRF 顺序: {e}")
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
