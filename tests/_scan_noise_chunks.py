# -*- coding: utf-8 -*-
"""扫描 ChromaDB 中残留的 OCR/版权页/目录噪声 chunk（旧版建库时未清洗）
用法: python tests/_scan_noise_chunks.py [--delete] [--limit N]
日志写入 tests/_noise_scan_out.txt（UTF-8）
"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.retrieval.advanced_search import invalidate_bm25_cache
from scenes.persona_chat.config import persona_chat_config

LOG_PATH = Path(__file__).parent / "_noise_scan_out.txt"
open(LOG_PATH, "w", encoding="utf-8").close()


def log(msg: str = ""):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


do_delete = "--delete" in sys.argv
limit = None
if "--limit" in sys.argv:
    i = sys.argv.index("--limit")
    limit = int(sys.argv[i + 1])

# 版权页/出版信息特征（与 parser.py 清洗规则对齐）
COPYRIGHT_PATTERNS = [
    r"图书在版编目", r"版本图书馆", r"版权信息", r"版权所有",
    r"ISBN\s*[:：]?\s*\d", r"CIP", r"定价[:：]", r"开本[:：]",
    r"印张", r"印次", r"责任编辑", r"封面设计", r"出版发行[:：]",
]

# 孤立数字/日期（页脚页码、版本号）
NUM_ONLY = re.compile(r"^\d{1,6}$")
DATE_LIKE = re.compile(r"^\d{2,4}[./]\d{1,2}([./]\d{1,2})?$")


def looks_noise(text: str) -> tuple[bool, str]:
    """返回 (是否噪声, 原因)"""
    if not text:
        return True, "空文本"
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= 2:
        return True, "过短"
    if NUM_ONLY.match(compact):
        return True, "孤立数字"
    if DATE_LIKE.match(compact):
        return True, "孤立日期/版本号"
    if len(compact) < 60:
        for p in COPYRIGHT_PATTERNS:
            if re.search(p, compact):
                return True, "版权页特征"
    # 目录页启发式：行短 + 以数字结尾
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= 5:
        toc_like = sum(
            1 for l in lines
            if len(re.sub(r"\s+", "", l)) < 40
            and re.search(r"\d+$", l)
            and not re.search(r"[。！？.!?]$", l)
        )
        if toc_like >= max(3, int(len(lines) * 0.4)):
            return True, "疑似目录页"
    return False, ""


client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

total_noise = 0
total_deleted = 0
for character_id, character in persona_chat_config.characters.items():
    collection_name = character.chroma_collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        log(f"无 collection: {collection_name}")
        continue

    count = collection.count()
    log(f"\n{'='*60}")
    log(f"📦 {collection_name}: 共 {count} 个 chunk")

    batch_size = 500
    offset = 0
    noise_ids = []
    noise_samples = []
    while offset < count:
        batch = collection.get(
            offset=offset,
            limit=batch_size,
            include=["documents", "metadatas"],
        )
        ids = batch["ids"]
        docs = batch["documents"] or []
        metas = batch["metadatas"] or []
        for i, doc in enumerate(docs):
            is_noise, reason = looks_noise(doc or "")
            if is_noise:
                source = (metas[i] or {}).get("source", "?")
                noise_ids.append((ids[i], source, reason, (doc or "")[:60].replace("\n", " ")))
        offset += batch_size
        if limit and offset >= limit:
            break

    log(f"  疑似噪声 chunk: {len(noise_ids)}")
    # 按来源聚合统计
    from collections import Counter
    by_source = Counter(src for _, src, _, _ in noise_ids)
    for src, n in by_source.most_common(10):
        log(f"    {src[:50]}: {n}")
    log("  抽样:")
    for item in noise_ids[:12]:
        iid, src, reason, snippet = item
        log(f"    [{iid[:36]}] ({reason}) {snippet} ... | {src[:30]}")

    if do_delete and noise_ids:
        ids_to_delete = [iid for iid, _, _, _ in noise_ids]
        # 分批删除
        for k in range(0, len(ids_to_delete), 500):
            collection.delete(ids=ids_to_delete[k:k + 500])
        # 删除后使 BM25 磁盘缓存失效（下次检索自动重建，避免旧索引包含已删 chunk）
        invalidate_bm25_cache(collection_name)
        log(f"  ✅ 已删除 {len(ids_to_delete)} 个噪声 chunk，BM25 缓存已失效")
        total_deleted += len(ids_to_delete)
    total_noise += len(noise_ids)

log(f"\n{'='*60}")
log(f"扫描完成: 共 {total_noise} 个疑似噪声 chunk" + (f"，已删除 {total_deleted} 个" if do_delete else "（未删除，加 --delete 执行）"))
