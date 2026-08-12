# -*- coding: utf-8 -*-
"""验证来源加权：同一问题在 use_source_weight=False/True 下 top-15 的来源分布对比"""
import asyncio
import sys
from collections import Counter

sys.path.insert(0, r"C:\Users\liu\Desktop\Multi-Agent-RAG-Academic-Assistant")

from src.core.config import settings
from src.retrieval.advanced_search import advanced_retrieval
from src.retrieval.source_profile import classify_source
from langchain_openai import ChatOpenAI
import chromadb
from chromadb.config import Settings as ChromaSettings

QUESTIONS = [
    "积极想象是什么？应该如何练习？",
    "梦的补偿作用是什么意思？",
    "人为什么会感到孤独，该如何面对？",
]


async def run_one(question: str, use_weight: bool):
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,
        max_tokens=256,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="persona_jung",
        metadata={"hnsw:space": "cosine"},
    )
    docs, _ = await advanced_retrieval(
        question=question,
        collection=collection,
        llm=llm,
        top_k=15,
        use_source_weight=use_weight,
    )
    return docs


async def main():
    for q in QUESTIONS:
        print(f"\n{'='*70}\n问题: {q}\n{'='*70}")
        for use_weight in [False, True]:
            docs = await run_one(q, use_weight)
            label = "加权" if use_weight else "未加权"
            counts = Counter()
            lines = []
            for i, d in enumerate(docs):
                src = d.metadata.get("source", "?")
                st = d.metadata.get("source_type", classify_source(src))
                counts[st] += 1
                lines.append(f"    [{i+1:>2}] [{st}] {src[:36]}")
            print(f"\n--- {label} | top-15 来源分布: {dict(counts)} ---")
            print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
