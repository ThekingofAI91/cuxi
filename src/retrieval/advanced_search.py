"""
advanced_search.py — 高级检索策略

实现多种检索增强：
1. Multi-Query（多查询检索）：LLM 生成多个查询变体，分别检索后合并
2. HyDE（假设文档嵌入）：LLM 先生成假设答案，用答案向量辅助检索
3. BM25（关键词检索）：精确命中专有名词（如《红书》、积极想象等）
4. Cross-Encoder 重排序：对 RRF 合并后的候选做深度语义精排

多路结果通过 RRF 合并（每类一票 + 类间权重），再经重排序输出 top-K。

性能优化（优化十七）：向量检索多路并行、BM25 仅查询原问题、重排序截断输入。
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
from langchain_openai import ChatOpenAI

from src.core.config import settings
from src.retrieval.trace import RetrievalTrace


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
        # 超时保护：中转服务抖动时改写实测可达 24.8s（约 20% 概率白等）。
        # 超过上限就放弃改写、退回原始查询检索——宁可召回略差，也不让用户干等。
        _timeout = getattr(settings, "rewrite_timeout_sec", 10.0) or 10.0
        response = await asyncio.wait_for(
            llm.ainvoke([("user", prompt)]),
            timeout=_timeout,
        )
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
    except asyncio.TimeoutError:
        print(f"[Multi-Query+HyDE] 超过 {getattr(settings, 'rewrite_timeout_sec', 10.0)}s 未返回，降级为原始查询检索")
        return [], ""
    except Exception as e:
        print(f"[Multi-Query+HyDE] 生成失败: {e}")
        return [], ""


# ============================================================
# 改写结果缓存 + 首字等待预算
# ============================================================
# 背景：改写是 1 次 LLM 往返，实测平均 8.8s、失败时 24.8s，是首字延迟的头号元凶。
# 但改写结果对问题语义高度敏感，不能无脑复用，故用 embedding 做"近似问题"命中：
# 命中即零等待复用，未命中才付费。embed 一次仅约 15ms，相对 8.8s 可以忽略。

_rewrite_cache: list[dict] = []          # [{"vec", "variants", "hyde"}]
_REWRITE_CACHE_MAX = 200                 # 上限，超出按 FIFO 淘汰
_REWRITE_CACHE_MIN_SIM = 0.88            # 余弦相似度门槛：低于此值视为不同问题
_bg_rewrite_tasks: set = set()           # 持有后台任务引用，防止被 GC


def _norm_vec(text: str):
    """把文本编码成单位向量（同步，调用方负责放线程）"""
    from src.retrieval.embedder import get_embedder
    import numpy as np
    v = np.asarray(get_embedder().embed_query(text), dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


async def _rewrite_cache_lookup(question: str):
    """语义缓存查找；命中返回 (variants, hyde, sim)，否则返回 None"""
    if not _rewrite_cache:
        return None  # 缓存为空时不做 embed，避免为首次查询付模型加载成本
    try:
        import numpy as np
        vec = await asyncio.to_thread(_norm_vec, question)
        sims = [float(np.dot(vec, it["vec"])) for it in _rewrite_cache]
        best = max(range(len(sims)), key=lambda i: sims[i])
        if sims[best] >= _REWRITE_CACHE_MIN_SIM:
            it = _rewrite_cache[best]
            return it["variants"], it["hyde"], sims[best]
    except Exception as e:
        print(f"[改写缓存] 查询失败（忽略）: {e}")
    return None


async def _rewrite_cache_store(question: str, variants: list[str], hyde: str) -> None:
    if not variants and not hyde:
        return
    try:
        vec = await asyncio.to_thread(_norm_vec, question)
        _rewrite_cache.append({"vec": vec, "variants": variants, "hyde": hyde})
        while len(_rewrite_cache) > _REWRITE_CACHE_MAX:
            _rewrite_cache.pop(0)
    except Exception as e:
        print(f"[改写缓存] 写入失败（忽略）: {e}")


def _on_background_rewrite_done(question: str, task) -> None:
    """超时放弃后，后台改写跑完的结果补进缓存——这次白等了，下次就能零等待"""
    _bg_rewrite_tasks.discard(task)
    if task.cancelled():
        return
    try:
        variants, hyde = task.result()
    except Exception:
        return
    if variants or hyde:
        asyncio.create_task(_rewrite_cache_store(question, variants, hyde))


async def _rewrite_with_budget(question: str, llm: ChatOpenAI, num_variants: int):
    """
    带缓存与等待预算的改写。

    1) 缓存命中 → 零等待复用（省掉整次 LLM 往返）
    2) 未命中 → 最多等 rewrite_deadline_sec 秒
       - 按时返回 → 用改写结果，并写入缓存
       - 超时    → 放弃改写（先用原始查询检索，保住首字延迟），
                   但任务不取消，后台跑完后补进缓存
    """
    cached = await _rewrite_cache_lookup(question)
    if cached is not None:
        variants, hyde, sim = cached
        print(f"[Advanced Retrieval] 改写缓存命中（相似度 {sim:.3f}），跳过 LLM 调用")
        return variants, hyde

    deadline = getattr(settings, "rewrite_deadline_sec", 2.0) or 2.0
    task = asyncio.create_task(
        generate_query_variants_and_hyde(question, llm, num_variants)
    )
    try:
        # shield：超时只取消外层等待，不取消改写任务本身
        variants, hyde = await asyncio.wait_for(asyncio.shield(task), timeout=deadline)
        await _rewrite_cache_store(question, variants, hyde)
        return variants, hyde
    except asyncio.TimeoutError:
        _bg_rewrite_tasks.add(task)
        task.add_done_callback(lambda t: _on_background_rewrite_done(question, t))
        print(f"[Advanced Retrieval] 改写超过 {deadline}s，改用原始查询检索（后台继续补全缓存）")
        return [], ""


# ============================================================
# 高级检索引擎
# ============================================================

def fuse_by_class(
    class_boards: list[tuple[str, list[str]]],
    doc_metadata: dict[str, dict],
    class_weights: dict[str, float],
    rrf_k: int = 60,
    use_source_weight: bool = True,
    detail: Optional[dict] = None,
) -> dict[str, float]:
    """RRF 融合：每类一票 + 类间权重。

    「一类」= 一种召回通道（vector / bm25）。同一通道下可以有多条榜（原问、查询变体、
    HyDE），它们只是同一通道的多次取样，不该各记一票——否则打开改写就等于偷偷改动
    融合权重：旧实现把榜拍平累加，改写生效时向量榜 1→4、BM25 榜 1→3，向量:BM25 的
    票数比从 1:1 变成 4:3，语义路被凭空放大。

    本函数分两步：
      1) 类内合票：同一 doc 在一类里出现多次，只保留它在该类里的最好名次；
      2) 类间加权：每类这一票乘类权重，再按 RRF 公式累加。

    只吃名次、不吃原始分数：余弦距离与 BM25 分数不同量纲，直接相加没有意义。

    Args:
        class_boards: [(类名, 该榜的 ranked doc_ids)]，类名 ∈ {"vector", "bm25"}
        doc_metadata: doc_id -> metadata（用于取 source 做来源加权）
        class_weights: 类名 -> 类权重
        rrf_k: RRF 平滑常数
        use_source_weight: 是否按语料来源类型加权
        detail: 可选的结果明细容器（评测归因用）。传了就填两张表：
            detail["best_rank"][doc_id][类名] = 该类里的最好名次
            detail["contrib"][doc_id][类名] = 该类实际贡献的分数（已含类权重与来源权重）
            两者相加即该 doc 的最终得分，用于回答「这条是向量捞的还是 BM25 捞的」

    Returns:
        doc_id -> RRF 分数（未排序）
    """
    from src.retrieval.source_profile import get_source_weight

    # 类内合票：class -> {doc_id: 该类内最好名次}
    best_rank_by_class: dict[str, dict[str, int]] = {}
    for class_name, ranked_ids in class_boards:
        ranks = best_rank_by_class.setdefault(class_name, {})
        for rank, doc_id in enumerate(ranked_ids):
            prev = ranks.get(doc_id)
            if prev is None or rank < prev:
                ranks[doc_id] = rank

    # 类间加权
    scores: dict[str, float] = {}
    for class_name, ranks in best_rank_by_class.items():
        class_weight = class_weights.get(class_name, 1.0)
        for doc_id, rank in ranks.items():
            weight = class_weight
            if use_source_weight:
                source = (doc_metadata.get(doc_id) or {}).get("source", "")
                weight *= get_source_weight(source)
            contribution = weight / (rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution
            if detail is not None:
                detail.setdefault("best_rank", {}).setdefault(doc_id, {})[class_name] = rank
                detail.setdefault("contrib", {}).setdefault(doc_id, {})[class_name] = contribution
    return scores


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
    trace_out: Optional[RetrievalTrace] = None,
) -> tuple[list[Document], list[str]]:
    """
    高级检索：Multi-Query + HyDE + BM25 + Cross-Encoder 重排序

    流程：
    1. 并行生成查询变体 + HyDE 假设文档
    2. 对每个查询（原始 + 变体 + HyDE）分别做向量检索
    3. 对原始 + 变体查询做 BM25 关键词检索（精确命中专有名词）
    4. 各路结果按「每类一票 + 类间权重」做 RRF 合并，取 rerank_pool_size 条进候选池
    5. Cross-Encoder 对整个候选池精排（池子默认 10、注入默认 5），最终返回 top_k 条

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
        trace_out: 可选的轨迹容器（评测专用）。传了就往里写每一路召回的名次，
            不传则完全无采集开销

    Returns:
        (retrieved_docs, all_contexts) 元组
    """
    from src.retrieval.embedder import get_embedder

    embedder = get_embedder()

    # ============================================================
    # 改写与"原始查询检索"重叠执行（教育区延迟优化）
    # ============================================================
    # 旧流程串行：先等改写 LLM（预算 2s）→ 再做向量/BM25。
    # 但改写只服务"变体 + HyDE"这几路查询，原始查询的向量 + BM25 根本不依赖它——
    # 完全可以并行。新流程：
    #   1) 改写任务直接 create_task 发车（不 await，不在此等待）
    #   2) 立即对原始查询跑向量 + BM25（与改写 LLM 重叠）
    #   3) 原始路跑完后再收改写结果（_rewrite_with_budget 的等待预算从任务启动起算，
    #      原始路通常几百 ms~2s，届时改写多半已就绪 → 实际额外等待趋近 0）
    #   4) 有变体/HyDE 再补第二轮向量 + BM25，与原始路结果一起 RRF 合并
    # 检索语义与旧实现完全一致（同样的查询集合、同样的合并方式），只是时序重叠。
    if use_multi_query or use_hyde:
        rewrite_task = asyncio.create_task(
            _rewrite_with_budget(question, llm, num_variants)
        )
    else:
        rewrite_task = None

    # 候选池：精排要真正参与「选人」，池子就必须大于最终返回条数。
    # 轻量档（use_rerank=False）不精排，池子就等于 top_k，行为与旧版一致。
    pool_size = max(top_k, settings.rerank_pool_size) if use_rerank else top_k
    fetch_k = max(top_k, pool_size) * 2  # 每路多取一些用于融合

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
    # 每路一条榜：(所属召回类, 该榜的 ranked doc_ids)。类只有 vector / bm25 两种，
    # 同一类下可以有多条榜（原问、查询变体、HyDE）——融合时类内取最好名次合成一票。
    class_boards: list[tuple[str, list[str]]] = []

    if trace_out is not None and not trace_out.question:
        trace_out.question = question

    def _add_boards(class_name: str, routes: list[str], groups: list[list[str]]) -> None:
        """登记一榜：同时进融合用的 class_boards 与轨迹用的 trace.boards。

        route 说明这一榜由哪条查询产出（original / variant1 / variant2 / hyde）——
        没有它就只能看到「向量类有 4 条榜」，看不出改写到底补了几路、补的那几路命中
        和原问重不重合。两处一起写，避免融合与轨迹的名次来源分叉。
        """
        for i, ids in enumerate(groups):
            route = routes[i] if i < len(routes) else f"query{i}"
            class_boards.append((class_name, ids))
            if trace_out is not None:
                trace_out.boards.append({
                    "class": class_name,
                    "route": route,
                    "hits": len(ids),
                    "ids": list(ids),
                })

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
    bm25_index = None

    try:
        _add_boards("vector", ["original"], await vec_task)
    except Exception:
        # 向量路失败时取消仍在跑的 BM25 任务，避免"异常从未被读取"告警
        if bm25_task is not None and not bm25_task.done():
            bm25_task.cancel()
        raise

    if bm25_task is not None:
        try:
            bm25_index, first_groups = await bm25_task
            _add_boards("bm25", ["original"], first_groups)
        except Exception as e:
            print(f"[Advanced Retrieval] BM25 检索失败，跳过: {e}")

    # ---- 收改写结果（预算从任务启动时已开始计算，此处通常只需极短等待）----
    if rewrite_task is not None:
        try:
            query_variants, hyde_doc = await rewrite_task
        except Exception as e:
            print(f"[Advanced Retrieval] 改写任务异常，退化为仅原始查询: {e}")
            query_variants, hyde_doc = [], ""
    else:
        query_variants, hyde_doc = [], ""

    # 构建第二路查询列表（变体 + HyDE）
    variant_routes = [f"variant{i}" for i in range(1, len(query_variants) + 1)]
    extra_queries: list[str] = list(query_variants)
    extra_routes: list[str] = list(variant_routes)
    if hyde_doc:
        extra_queries.append(hyde_doc)
        extra_routes.append("hyde")

    if extra_queries:
        print(f"[Advanced Retrieval] 改写补充 {len(extra_queries)} 路查询:")
        for q in extra_queries:
            print(f"  [改写] {q[:60]}...")

        # 第二轮：变体 + HyDE 的向量检索
        try:
            _add_boards("vector", extra_routes, await _vector_pass(extra_queries))
        except Exception as e:
            print(f"[Advanced Retrieval] 变体向量检索失败，跳过: {e}")

        # 第二轮：变体的 BM25 检索（HyDE 是长文档，不进 BM25；与旧实现一致）
        if bm25_index is not None and query_variants:
            try:
                _add_boards("bm25", variant_routes, await _bm25_pass(query_variants, bm25_index))
            except Exception as e:
                print(f"[Advanced Retrieval] 变体 BM25 检索失败，跳过: {e}")

    stage_mark("retrieval_ms", (time.perf_counter() - _retrieval_t0) * 1000)

    # ---- Step 3: RRF 合并（每类一票 + 类间权重）----
    rrf_k = 60
    class_weights = {
        "vector": settings.rrf_vector_weight,
        "bm25": settings.rrf_bm25_weight,
    }
    # 来源加权：oral（口述/讲座/回忆录）优先召回学说话方式，
    # original（原著）支撑观点，anchor（core_ideas）锚定关键事实，
    # secondary（二手解读）强降权防止第三者评价冒充名人原话。
    fusion_detail: dict = {}
    doc_rrf_scores = fuse_by_class(
        class_boards,
        {doc_id: cand.get("metadata") or {} for doc_id, cand in doc_candidates.items()},
        class_weights,
        rrf_k=rrf_k,
        use_source_weight=use_source_weight,
        detail=fusion_detail if trace_out is not None else None,
    )

    # 按 RRF 分数排序
    sorted_doc_ids = sorted(
        doc_rrf_scores.keys(),
        key=lambda x: doc_rrf_scores[x],
        reverse=True,
    )

    # ---- Step 4: 构建候选池（要比最终返回条数多，精排才有「选人」的余地）----
    candidate_docs: list[Document] = []
    candidate_limit = pool_size

    from src.retrieval.source_profile import classify_source

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

    # ---- 轨迹：候选池里每条是怎么进来的（哪类捞的、各贡献多少分）----
    # 只记进池的 candidate_limit 条：没进池的不会影响回答，记了只是噪音。
    # contrib 之和 == score，所以「这条靠向量还是靠 BM25」可以逐条对账。
    if trace_out is not None:
        for rrf_rank, doc_id in enumerate(sorted_doc_ids[:candidate_limit]):
            meta = doc_candidates[doc_id]["metadata"] or {}
            src = meta.get("source", "")
            trace_out.fused[doc_id] = {
                "rrf_rank": rrf_rank,
                "score": round(doc_rrf_scores[doc_id], 6),
                "best_rank": fusion_detail.get("best_rank", {}).get(doc_id, {}),
                "contrib": {
                    cls: round(v, 6)
                    for cls, v in fusion_detail.get("contrib", {}).get(doc_id, {}).items()
                },
                "source": src,
                "source_type": meta.get("source_type") or classify_source(src),
            }

    # ---- Step 5: Cross-Encoder 精排整个候选池，再按 top_k 截断 ----
    # 池子（rerank_pool_size，默认 10）> 注入条数（top_k，默认 5）是本步的前提：
    # 只有池子更大，精排的排序才会实际决定「哪几条被留下」。若池子 == 注入条数，
    # 精排就退化成纯排序器，改它等于白改（2026-10-18 的视野实验已证）。
    # 代价：10 对 × 500 字 × 512 token ≈ 1.9s（int8，CPU 4 并发）。
    if use_rerank and len(candidate_docs) > 1:
        try:
            from src.retrieval.reranker import rerank_documents
            # CPU 推理同步调用会阻塞事件循环，放线程池
            _rerank_t0 = time.perf_counter()
            candidate_docs = await asyncio.to_thread(
                rerank_documents, question, candidate_docs, top_k=len(candidate_docs),
            )
            stage_mark("rerank_ms", (time.perf_counter() - _rerank_t0) * 1000)
            print(
                f"[Advanced Retrieval] Cross-Encoder 精排 {len(candidate_docs)} 条"
                f"（池 {candidate_limit} → 注入 {min(top_k, len(candidate_docs))}）"
            )
        except Exception as e:
            print(f"[Advanced Retrieval] 重排序失败，保持 RRF 顺序: {e}")

    retrieved_docs = candidate_docs[:top_k]
    contexts = [doc.page_content for doc in retrieved_docs]

    # 打印最终结果的来源类型分布（便于验证来源加权效果）
    type_counts: dict[str, int] = {}
    for doc in retrieved_docs:
        st = doc.metadata.get("source_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
    print(f"[Advanced Retrieval] 最终返回 {len(retrieved_docs)} 条文档 | 来源分布: {type_counts}")

    return retrieved_docs, contexts
