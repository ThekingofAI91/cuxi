"""
context_compressor.py — 上下文智能压缩（实验性）

把检索到的完整 chunk（~1000 字符/条 × 15 条 ≈ 12K 字符）用一次 LLM 调用压缩成
面向当前问题的"高密度摘要"（每条 ~150 字），替代硬截断（_build_context 的 450 字/条）：
- prefill 省 2/3 以上 → 首字更快
- 摘要聚焦问题 → 忠实度可能提升（去掉了诱导跑偏的无关段落）
- 代价：生成前多一次 LLM 调用（与改写同量级，2-6s），故默认关闭，
  由 CONTEXT_COMPRESSION_ENABLED 控制并配合评估数据决定去留

缓存：与改写缓存同思路——按 (doc 指纹, 问题向量) 做近似命中（余弦 ≥ 0.92），
相似问题直接复用压缩结果，第二轮起零 LLM 成本。
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections import OrderedDict

import numpy as np
from langchain_core.documents import Document

from src.core.config import settings

_COMPRESS_PROMPT = """你是 RAG 上下文压缩器。针对用户问题，从下面的知识库片段中提取"与回答该问题直接相关"的信息。

要求：
- 只保留回答该问题会用到的内容；与问题无关的段落整段丢弃
- 每条片段输出 ≤ 150 字的高密度摘要，保留关键概念、事实、数字、结论
- 保留原文的表述风格要点（如口语/书面的标志性说法），不要改写观点
- 片段与问题完全无关时，该条输出"（无关）"
- 按输入顺序输出，每条一行，格式固定为"片段N: 摘要"

用户问题：{question}

{chunks}"""

_cache: OrderedDict[tuple, tuple[np.ndarray, list[str]]] = OrderedDict()
_CACHE_MAX = 200
_CACHE_LOCK = threading.Lock()

_bg_tasks: set = set()


def _doc_fp(doc: Document) -> str:
    return hashlib.md5((doc.page_content or "")[:200].encode("utf-8")).hexdigest()


def _norm_vec(text: str) -> np.ndarray:
    from src.retrieval.embedder import get_embedder

    v = np.asarray(get_embedder().embed_query(text), dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cache_lookup(qvec: np.ndarray) -> Optional[list[str]]:
    with _CACHE_LOCK:
        best_key, best_sim = None, 0.0
        for key, (vec, _s) in _cache.items():
            sim = float(np.dot(qvec, vec))
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None and best_sim >= 0.92:
            _cache.move_to_end(best_key)
            return _cache[best_key][1]
    return None


def _cache_store(qvec: np.ndarray, doc_fps: tuple, summaries: list[str]) -> None:
    with _CACHE_LOCK:
        # numpy 数组不可哈希 → 键里用 bytes，向量本体存值里供相似度计算
        _cache[(doc_fps, qvec.tobytes())] = (qvec, summaries)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def _parse_summaries(content: str, n: int) -> list[str]:
    """解析"片段N: 摘要"行；缺失/（无关）的置空串"""
    out = [""] * n
    for line in (content or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        head, _, body = line.partition(":")
        body = body.strip()
        digits = "".join(ch for ch in head if ch.isdigit())
        if not digits or not body:
            continue
        idx = int(digits) - 1
        if 0 <= idx < n and body != "（无关）":
            out[idx] = body
    return out


async def compress_docs(question: str, docs: list[Document], llm=None) -> list[Document]:
    """
    压缩检索结果：返回新 Document 列表（page_content 换成压缩摘要，metadata 保留）。
    任何失败都回退为原文档（压缩是优化项，绝不能让检索挂掉）。
    """
    if not docs:
        return docs
    try:
        qvec = await asyncio.to_thread(_norm_vec, question)
        doc_fps = tuple(_doc_fp(d) for d in docs)
        cached = _cache_lookup(qvec)
        if cached is not None and len(cached) == len(docs):
            return [Document(page_content=s or d.page_content, metadata=d.metadata)
                    for s, d in zip(cached, docs)]

        if llm is None:
            from src.core.llm import get_chat_llm
            llm = get_chat_llm(temperature=0.1, max_tokens=800)

        chunks_text = "\n\n".join(
            f"[片段{i+1}]（来源:{(d.metadata or {}).get('source','')} | 章节:{(d.metadata or {}).get('heading','')}）\n{d.page_content[:600]}"
            for i, d in enumerate(docs)
        )
        from src.core.llm import ainvoke_nonempty
        resp = await ainvoke_nonempty(llm, [("user", _COMPRESS_PROMPT.format(
            question=question[:300], chunks=chunks_text))])
        summaries = _parse_summaries(getattr(resp, "content", "") or "", len(docs))
        if not any(summaries):
            return docs  # 全空（解析失败/上游故障）→ 原文兜底

        _cache_store(qvec, doc_fps, summaries)
        return [Document(page_content=s or d.page_content, metadata=d.metadata)
                for s, d in zip(summaries, docs)]
    except Exception as e:
        print(f"[Compressor] ⚠️ 压缩失败（回退原文）: {e}")
        return docs
