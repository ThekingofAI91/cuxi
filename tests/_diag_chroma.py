# -*- coding: utf-8 -*-
"""诊断：chroma 版本 + collection 规模 + get() 行为"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

lines = []

try:
    import chromadb
    lines.append(f"chromadb version: {chromadb.__version__}")
except Exception as e:
    lines.append(f"chromadb import failed: {e}")

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from src.core.config import settings
    from framework.supervisor import get_scene_config
    from scenes.persona_chat.config import persona_chat_config

    # 直接设置 persona 场景
    from framework.supervisor import set_scene_config
    set_scene_config(persona_chat_config)
    config = get_scene_config()
    collection_name = getattr(config, 'chroma_collection', 'academic_docs')
    lines.append(f"collection: {collection_name}")

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    count = collection.count()
    lines.append(f"collection count: {count}")

    # 全量 get（复现问题）
    try:
        all_data = collection.get(include=["documents", "metadatas"])
        lines.append(f"full get OK: {len(all_data['documents'])} docs")
    except Exception as e:
        lines.append(f"full get FAIL: {type(e).__name__}: {str(e)[:300]}")

    # 分页 get
    try:
        batch = collection.get(limit=500, offset=0, include=["documents", "metadatas"])
        lines.append(f"paged get(limit=500) OK: {len(batch['documents'])} docs, keys={list(batch.keys())}")
    except Exception as e:
        lines.append(f"paged get FAIL: {type(e).__name__}: {str(e)[:300]}")

    # 用 chroma 原生向量查询测试（不重新 embedding）
    try:
        import numpy as np
        from src.retrieval.embedder import get_embedder
        embedder = get_embedder()
        qvec = embedder.embed_query("什么是人格面具")
        res = collection.query(query_embeddings=[qvec], n_results=5, include=["documents", "metadatas", "distances"])
        lines.append(f"chroma native query OK: {len(res['documents'][0])} results")
    except Exception as e:
        lines.append(f"chroma native query FAIL: {type(e).__name__}: {str(e)[:300]}")

except Exception as e:
    lines.append(f"diag outer FAIL: {type(e).__name__}: {str(e)[:500]}")

report = "\n".join(lines)
print(report)
with open(Path(__file__).parent / "_diag_result.txt", "w", encoding="utf-8") as f:
    f.write(report)
