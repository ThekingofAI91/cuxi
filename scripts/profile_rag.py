"""RAG 分阶段计时 + 语料质量扫描（诊断脚本）"""
import time
import unicodedata

t0 = time.time()
from framework.supervisor import get_chroma_client
from src.retrieval.embedder import get_embedder
from src.core.config import settings

client = get_chroma_client()
col = client.get_collection("persona_jung")
print(f"[init] chroma client + collection: {time.time()-t0:.2f}s, docs={col.count()}")

t0 = time.time()
emb = get_embedder()
emb.embed_query("预热")
print(f"[init] embedder load+warm: {time.time()-t0:.2f}s")

from src.retrieval.advanced_search import advanced_retrieval, _ensure_bm25_ready
from src.core.llm import get_chat_llm

# ---- 阶段单独计时 ----
t0 = time.time()
docs, idx = _ensure_bm25_ready(col)
print(f"[stage] BM25 index ready: {time.time()-t0:.2f}s")

t0 = time.time()
from src.retrieval.reranker import get_reranker
_ = get_reranker().model
print(f"[stage] reranker model load: {time.time()-t0:.2f}s")

from src.retrieval.reranker import rerank_documents
sample = docs[:10]
t0 = time.time()
rerank_documents("什么是集体无意识", sample, top_k=10)
print(f"[stage] rerank 10 candidates: {time.time()-t0:.2f}s")

llm = get_chat_llm(temperature=0.3, max_tokens=256)
t0 = time.time()
try:
    import asyncio
    from src.retrieval.advanced_search import generate_query_variants_and_hyde
    v, h = asyncio.run(generate_query_variants_and_hyde("什么是集体无意识", llm, 2))
    print(f"[stage] rewrite LLM: {time.time()-t0:.2f}s variants={len(v)}")
except Exception as e:
    print(f"[stage] rewrite LLM FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")

# ---- 端到端 advanced_retrieval ----
import asyncio
for q in ["梦中的蛇象征什么", "荣格如何理解个体化过程"]:
    t0 = time.time()
    got, _ = asyncio.run(advanced_retrieval(q, col, llm, top_k=15, use_multi_query=True, use_hyde=True, use_rerank=True))
    print(f"[e2e] '{q}' 完整检索: {time.time()-t0:.2f}s -> {len(got)} docs")

# ---- 语料质量扫描 ----
print("\n===== 语料质量扫描 persona_jung =====")
_ALLOWED_PUNCT = set(' .,;:!?()[]{}\'"/-_&%$#@*+=<>|~`^\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00b7\u3001\u3002\u300a\u300b\u3010\u3011\uff08\uff09\uff0c\uff1a\uff1b\uff01\uff1f\u201c\u201d')


def garbage_stats(t):
    """返回 (坏字符比例, 首个坏字符说明)"""
    if not t:
        return 0.0, ""
    bad = 0
    first_bad = ""
    for ch in t:
        o = ord(ch)
        is_bad = False
        if ch == "\ufffd":
            is_bad = True
        elif 0xE000 <= o <= 0xF8FF:
            is_bad = True
        elif unicodedata.category(ch) == "Cc" and ch not in "\n\t":
            is_bad = True
        elif 0x4E00 <= o <= 0x9FFF:
            is_bad = False
        elif 0x3000 <= o <= 0x303F:
            is_bad = False
        elif ch.isascii() and (ch.isalnum() or ch in _ALLOWED_PUNCT):
            is_bad = False
        elif unicodedata.category(ch).startswith("N"):
            is_bad = False
        if is_bad:
            bad += 1
            if not first_bad:
                first_bad = f"U+{o:04X}({ch!r})"
    return bad / max(len(t), 1), first_bad


res = col.get(include=["documents", "metadatas"])
texts = res["documents"]
metas = res["metadatas"]

worst = []
flags = 0
for t, m in zip(texts, metas):
    r, first_bad = garbage_stats(t)
    if r > 0.05:
        flags += 1
        worst.append((r, (m or {}).get("source", "?"), first_bad, t[:70]))
worst.sort(reverse=True)
print(f"总块数={len(texts)}，疑似乱码块(坏字符>5%)={flags} ({flags/len(texts)*100:.1f}%)")
print("最严重样例：")
for r, s, fb, t in worst[:8]:
    print(f"  bad={r*100:4.0f}% {fb:20s} src={s} | {t}")

lens = [len(t) for t in texts]
import statistics
print(f"\n块长: min={min(lens)} 中位={statistics.median(lens):.0f} 平均={statistics.mean(lens):.0f} max={max(lens)}")
short = sum(1 for l in lens if l < 50)
print(f"超短块(<50字)={short} ({short/len(lens)*100:.1f}%)")

# BM25 打分耗时（10 万级文档库的痛点）
t0 = time.time()
idx.search("梦中的蛇象征什么", 30)
print(f"\n[stage] BM25 单查询打分: {time.time()-t0:.2f}s")
