# -*- coding: utf-8 -*-
"""热更新 ChromaDB 中语料的 source_type 元数据（按最新规则表重刷，无需重建库）
用法: python tests/_retag_sources.py [--dry-run]
日志写入 tests/_retag_out.txt（UTF-8）
"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

# 输出重定向到文件时 Windows 默认 GBK 编码无法输出 emoji，强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.retrieval.source_profile import classify_source
from scenes.persona_chat.config import persona_chat_config

LOG_PATH = Path(__file__).parent / "_retag_out.txt"


def log(msg: str = ""):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


dry_run = "--dry-run" in sys.argv

# 覆盖旧日志
open(LOG_PATH, "w", encoding="utf-8").close()

client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

total_updated = 0
for character_id, character in persona_chat_config.characters.items():
    collection_name = character.chroma_collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        log(f"⚠️ 无 collection: {collection_name}")
        continue

    count = collection.count()
    log(f"\n📦 {collection_name}: 共 {count} 个 chunk，按最新规则表重刷 source_type ...")

    batch_size = 500
    offset = 0
    updated = 0
    while offset < count:
        batch = collection.get(
            offset=offset,
            limit=batch_size,
            include=["metadatas"],
        )
        ids, metadatas = batch["ids"], batch["metadatas"]
        new_mds = []
        for meta in metadatas or []:
            source = (meta or {}).get("source", "")
            new_type = classify_source(source)
            old_type = (meta or {}).get("source_type")
            if old_type != new_type:
                nm = dict(meta)
                nm["source_type"] = new_type
                new_mds.append(nm)
            else:
                new_mds.append(None)
        changed_ids = [i for i, nm in enumerate(new_mds) if nm is not None]
        if changed_ids and not dry_run:
            collection.update(
                ids=[ids[i] for i in changed_ids],
                metadatas=[new_mds[i] for i in changed_ids],
            )
        for i in changed_ids:
            old = (metadatas[i] or {}).get("source_type")
            log(f"    [{ids[i][:40]}] {old} -> {new_mds[i]['source_type']} | {(metadatas[i] or {}).get('source', '')[:50]}")
        updated += len(changed_ids)
        offset += batch_size

    total_updated += updated
    log(f"  {collection_name}: 更新 {updated} 个 chunk" + ("（dry-run，未写入）" if dry_run else ""))

log(f"\n{'='*50}")
log(f"{'预演（未写入）' if dry_run else '完成'}: 共更新 {total_updated} 个 chunk")
