# -*- coding: utf-8 -*-
"""抽查指定来源的疑似噪声 chunk 内容，确认是否误判"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from scenes.persona_chat.config import persona_chat_config
from tests._scan_noise_chunks import looks_noise

TARGETS = {
    "persona_jung": ["荣格自传：回忆、梦、思考.pdf"],
    "persona_adler": ["超越自卑与洞察人性（阿德勒四大名著合集）.pdf", "走出孤独：阿德勒孤独十五讲.pdf"],
}

client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

for collection_name, keywords in TARGETS.items():
    collection = client.get_collection(collection_name)
    count = collection.count()
    print(f"\n{'='*70}\n{collection_name} 抽查\n{'='*70}")
    for kw in keywords:
        where = {"source": {"$eq": kw}}
        try:
            batch = collection.get(
                limit=2000,
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            print(f"  [{kw}] where 查询失败: {e}")
            continue
        ids, docs, metas = batch["ids"], batch["documents"] or [], batch["metadatas"] or []
        noise = []
        for i, doc in enumerate(docs):
            is_noise, reason = looks_noise(doc or "")
            if is_noise:
                noise.append((ids[i], reason, (doc or "").replace("\n", "⏎")[:80]))
        print(f"\n【{kw}】共取 {len(ids)} 个 chunk，疑似噪声 {len(noise)} 个")
        # 按原因统计
        from collections import Counter
        print("  按原因:", dict(Counter(r for _, r, _ in noise)))
        # 每种原因抽样 3 条
        shown = set()
        for iid, reason, snippet in noise:
            if reason not in shown:
                shown.add(reason)
                print(f"    [{iid[:32]}] ({reason}) {snippet}")
            if len(shown) >= 5:
                break
        # 再随机抽 3 条不同的
        import random
        random.seed(42)
        for iid, reason, snippet in random.sample(noise, min(3, len(noise))):
            print(f"    [{iid[:32]}] ({reason}) {snippet}")
