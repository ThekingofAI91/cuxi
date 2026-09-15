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
import time
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

    # ============================================================
    # 改写与"原始查询检索"重叠执行（教育区延迟优化）
    # ============================================================
    # 旧流程串行：先等改写 LLM（预算 2s）→ 再做向量/BM25。
    # 但改写只服务"变体 + HyDE"这几路查询，原始查询的向量 + BM25 根本不依赖它——
    # 完全可以并行。新流程：
    #   1) 改写任务直接 create_task 挂起（不 await）
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
    bm25_index = None

    try:
        query_results_list.extend(await vec_task)
    except Exception:
        # 向量路失败时取消仍在跑的 BM25 任务，避免"异常从未被读取"告警
        if bm25_task is not None and not bm25_task.done():
            bm25_task.cancel()
        raise

    if bm25_task is not None:
        try:
            bm25_index, first_groups = await bm25_task
            query_results_list.extend(first_groups)
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
    extra_queries: list[str] = []
    extra_queries.extend(query_variants)
    if hyde_doc:
        extra_queries.append(hyde_doc)

    if extra_queries:
        print(f"[Advanced Retrieval] 改写补充 {len(extra_queries)} 路查询:")
        for q in extra_queries:
            print(f"  [改写] {q[:60]}...")

        # 第二轮：变体 + HyDE 的向量检索
        try:
            query_results_list.extend(await _vector_pass(extra_queries))
        except Exception as e:
            print(f"[Advanced Retrieval] 变体向量检索失败，跳过: {e}")

        # 第二轮：变体的 BM25 检索（HyDE 是长文档，不进 BM25；与旧实现一致）
        if bm25_index is not None and query_variants:
            try:
                query_results_list.extend(await _bm25_pass(query_variants, bm25_index))
            except Exception as e:
                print(f"[Advanced Retrieval] 变体 BM25 检索失败，跳过: {e}")

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
    # 重组后块为完整段落（~1000 字符），全候选精排在 CPU 上 15 对要 20s+；
    # 只精排 rerank_candidates（默认 10）对，剩余候选保持 RRF 顺序拼接，质量损失小、耗时减半。
    rerank_n = settings.rerank_candidates
    head = candidate_docs[:rerank_n]
    tail = candidate_docs[rerank_n:]
    if use_rerank and len(head) > 1:
        try:
            from src.retrieval.reranker import rerank_documents
            # CPU 推理（30 对 × 512 token）需数秒，同步调用会阻塞事件循环，放线程池
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
