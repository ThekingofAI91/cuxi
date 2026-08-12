"""
Retrieval Agent — 图书管理员 📚
负责文档检索：高级混合检索（Multi-Query + HyDE + BM25 + 向量）+ Cross-Encoder 重排序

与 HTTP 层（/persona/eval_query）共用 src.retrieval.advanced_search.advanced_retrieval，
避免两套检索逻辑不一致。

修复记录（优化十八）：
- 旧实现 collection.get() 全量拉取文档，触发 ChromaDB 内部 SQLite
  "too many SQL variables" 错误（SQLITE_MAX_VARIABLE_NUMBER），
  导致多智能体图检索链路一直失败；
- 旧 vector_search 会对全部文档重新 embedding（本地模型推理，几万条极慢）；
- advanced_search 内部已分页拉取文档，并使用 ChromaDB 原生向量查询（入库时已存向量）。
"""

from __future__ import annotations

from typing import Any

from src.core.config import settings
from src.core.state import AgentState


async def retrieval_agent(state: AgentState) -> dict[str, Any]:
    """
    检索 Agent：根据 query 执行高级混合检索

    流程：
    1. 从 state 读取 query
    2. 获取场景对应的 ChromaDB collection
    3. 执行高级检索（Multi-Query + HyDE + BM25 + 向量 + Cross-Encoder 重排序）
    4. 将 Top-K 结果写入 state["retrieved_docs"]
    """
    query = state["query"]
    history = state.get("history", [])
    print(f"\n[Retrieval Agent] 🔍 正在检索: {query}")
    print(f"[Retrieval Agent] 对话历史轮次: {len(history)}")

    try:
        # ---- 1. 获取场景对应的 collection（与 HTTP 层一致）----
        from framework.supervisor import get_scene_config, get_chroma_client
        config = get_scene_config()
        collection_name = getattr(config, 'chroma_collection', 'persona_jung') if config else 'persona_jung'

        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        count = collection.count()
        if count == 0:
            print("[Retrieval Agent] ⚠️ 文档库为空，请先上传文档")
            return {
                "retrieved_docs": [],
                "route_history": state.get("route_history", []) + ["retrieval_agent"],
            }
        print(f"[Retrieval Agent] 文档库中共 {count} 条文档片段")

        # ---- 2. 高级检索（Multi-Query + HyDE + BM25 + 向量 + 重排序）----
        from src.core.llm import get_chat_llm
        from src.retrieval.advanced_search import advanced_retrieval

        # 初始化 LLM（用于生成查询变体和 HyDE 文档；已合并为一次调用，256 足够）
        retrieval_llm = get_chat_llm(
            temperature=0.3,
            max_tokens=256,
        )

        top_k = 15  # 与 HTTP 层 /persona/eval_query 保持一致

        # ---- 历史感知检索 ----
        # 场景开启 history_aware_retrieval 时（如名人对话），将最近几轮对话拼入检索 query，
        # 解决指代性问题（"那个梦"、"这跟它有什么关系"）检索不到前文上下文的问题。
        question = query
        if config and getattr(config, 'history_aware_retrieval', False) and history:
            recent = history[-2:]  # 最近两轮
            ctx_lines = []
            for q, a in recent:
                if q:
                    ctx_lines.append(f"用户：{q}")
                if a:
                    ctx_lines.append(f"助手：{a[:120]}")
            if ctx_lines:
                question = "以下是最近的对话（用于理解指代和上下文）：\n" + "\n".join(ctx_lines) + "\n\n当前用户问题：\n" + query
                print(f"[Retrieval Agent] 🔁 历史感知检索（拼接最近 {len(recent)} 轮对话）")

        retrieved_docs, _contexts = await advanced_retrieval(
            question=question,
            collection=collection,
            llm=retrieval_llm,
            top_k=top_k,
            use_multi_query=True,
            use_hyde=True,
            num_variants=2,
        )
        print(f"[Retrieval Agent] 高级检索召回 {len(retrieved_docs)} 条")

        # 打印结果摘要
        for i, doc in enumerate(retrieved_docs):
            score = doc.metadata.get("rrf_score", 0.0)
            heading = doc.metadata.get("heading", "未知章节")
            source = doc.metadata.get("source", "未知来源")
            stype = doc.metadata.get("source_type", "unknown")
            print(f"  [{i+1}] rrf={score} | [{stype}] {source} | {heading}")

        return {
            "retrieved_docs": retrieved_docs,
            "route_history": state.get("route_history", []) + ["retrieval_agent"],
        }

    except Exception as e:
        print(f"[Retrieval Agent] ❌ 检索过程中出错: {e}")
        return {
            "retrieved_docs": [],
            "route_history": state.get("route_history", []) + ["retrieval_agent"],
            "error": str(e),
        }
