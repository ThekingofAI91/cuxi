# -*- coding: utf-8 -*-
"""验证阿德勒 persona_adler 检索是否正常"""
import sys
sys.path.insert(0, ".")

from src.retrieval.embedder import get_embedder
import chromadb
from chromadb.config import Settings as ChromaSettings

embedder = get_embedder()
client = chromadb.PersistentClient(
    path="chroma_db",
    settings=ChromaSettings(anonymized_telemetry=False),
)
col = client.get_or_create_collection(
    name="persona_adler",
    metadata={"hnsw:space": "cosine"},
)
print(f"collection count: {col.count()}")

queries = [
    "恋爱中总是忍不住吃醋，怀疑对方，怎么办",
    "和伴侣吵架之后应该怎么和好",
    "失恋了走不出来，觉得自己什么都做不好",
]

for q in queries:
    e = embedder.embed_query(q)
    r = col.query(query_embeddings=[e], n_results=2)
    print("=" * 60)
    print(f"Q: {q}")
    for d in r["documents"][0]:
        print("---")
        print(d[:180].replace("\n", " "))
