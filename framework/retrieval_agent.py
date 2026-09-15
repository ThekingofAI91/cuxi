"""
Retrieval Agent — 图书管理员
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

from typing import Any, Optional

from src.core.config import settings
from src.core.state import AgentState
from src.retrieval.knowledge_graph import graph_exists

# bge-small-zh-v1.5 的窗口是 512 token（中文约 1 token/字 ≈ 507 字符）。
# 历史感知检索把最近两轮对话拼进查询，长历史 + 长问题会把"当前用户问题"
# 整个挤出窗口——拼接文本从右侧截断，真实问题根本进不了向量，检索完全失焦。
# 对策：默认装配不变（绝大多数查询 < 460 字符，零行为变化）；
# 超预算时逐级压缩历史长度——历史可以截，当前问题不能丢。
_QUERY_CHAR_BUDGET = 460  # 512 token 留出分词开销安全余量


def build_history_aware_query(query: str, recent: list[tuple[str, str]]) -> str:
    """拼接历史感知检索查询；超预算时逐级压缩历史，保证当前问题完整进窗口"""
    def _assemble(ans_cap: int, q_cap: Optional[int] = None) -> str:
        ctx_lines = []
        for q, a in recent:
            if q:
                ctx_lines.append(f"用户：{q if q_cap is None else q[:q_cap]}")
            if a:
                ctx_lines.append(f"助手：{a[:ans_cap]}")
        return ("以下是最近的对话（用于理解指代和上下文）：\n"
                + "\n".join(ctx_lines) + "\n\n当前用户问题：\n" + query)

    question = _assemble(120)
    if len(question) > _QUERY_CHAR_BUDGET:
        for ans_cap, q_cap in ((80, 80), (50, 50), (30, 30), (0, 0)):
            question = _assemble(ans_cap, q_cap)
            if len(question) <= _QUERY_CHAR_BUDGET:
                break
        print(f"[Retrieval Agent] 历史过长，已压缩至 {len(question)} 字符（保当前问题）")
    return question


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
    print(f"\n[Retrieval Agent] 正在检索: {query}")
    print(f"[Retrieval Agent] 对话历史轮次: {len(history)}")

    # ---- 0. 零检索开关（默认关闭，由 settings.entertainment_light_retrieval 控制）----
    # 实测：娱乐区轻量检索（top_k=3，无改写/无重排/无图谱）中位仅 23ms，
    # 对 8-15s 的回答完全可忽略，故默认保留检索以贴合角色背景资料。
    # 只有把开关置 False（追求极致首字、且角色卡足够完备）时才走这条零检索路径。
    if state.get("skip_retrieval"):
        print("[Retrieval Agent] 零检索（已关闭轻量检索开关），跳过向量库与重排")
        return {
            "retrieved_docs": [],
            "route_history": state.get("route_history", []) + ["retrieval_agent"],
            "graph_used": False,
        }

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
            print("[Retrieval Agent] 文档库为空，请先上传文档")
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

        # 娱乐区（zone=entertainment）：轻量召回 —— 跳过 Multi-Query+HyDE 改写、
        # 跳过 Cross-Encoder 重排、跳过知识图谱，仅用原始问题做向量+BM25 召回 top-3
        # 作为软背景（不引用），换取更快首字与更"像人"的回答。
        light_retrieval = state.get("light_retrieval", False)
        if light_retrieval:
            top_k = 3
            print(f"[Retrieval Agent] 娱乐区轻量召回：跳过改写/重排/图谱，top_k={top_k}")

        # ---- 历史感知检索 ----
        # 场景开启 history_aware_retrieval 时（如名人对话），将最近几轮对话拼入检索 query，
        # 解决指代性问题（"那个梦"、"这跟它有什么关系"）检索不到前文上下文的问题。
        question = query
        if config and getattr(config, 'history_aware_retrieval', False) and history:
            question = build_history_aware_query(query, history[-2:])
            print(f"[Retrieval Agent] 历史感知检索（拼接最近 {min(len(history), 2)} 轮对话，{len(question)} 字符）")

        # 短问题跳过 Multi-Query + HyDE 改写（省 1 次串行 LLM 往返，降低首字延迟）。
        # 原始查询 + BM25 已能覆盖短问题的召回；长/复杂问题仍走完整改写以提升召回。
        # 另有总开关 rewrite_enabled（默认 False）：实测改写延迟 3.5s 超过 2s 等待预算，
        # 新问题必然白等 2s 后被放弃，故默认关闭（忠实度基线 8.8 即在改写无效时测得）。
        use_rewrite = (
            settings.rewrite_enabled
            and len(query.strip()) > settings.skip_rewrite_max_chars
            and not light_retrieval
        )
        if not use_rewrite:
            print(f"[Retrieval Agent] 跳过检索改写（{'总开关关闭' if not settings.rewrite_enabled else '短问题' if not light_retrieval else '娱乐区轻量召回'}），直接向量+BM25")

        retrieved_docs, _contexts = await advanced_retrieval(
            question=question,
            collection=collection,
            llm=retrieval_llm,
            top_k=top_k,
            use_multi_query=use_rewrite,
            use_hyde=use_rewrite,
            use_rerank=not light_retrieval,
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

        # ---- 3. 知识图谱 RAG 增强（GraphRAG，按需调用）----
        # 默认只用文本检索；只有当文本检索质量不达标（召回不足 / top 相关性弱 /
        # 关键实体未被文本覆盖）时才启动图谱增强，避免每轮都多跑一次子图扩展。
        # 图谱未构建或总开关关闭 → 静默降级，不影响原混合检索。
        if settings.graph_rag_enabled and graph_exists(collection.name) and not light_retrieval:
            try:
                from src.retrieval.knowledge_graph import (
                    retrieve_graph_context,
                    should_trigger_graph_retrieval,
                )
                from src.retrieval.advanced_search import _get_doc_id

                trigger, reason = should_trigger_graph_retrieval(
                    question, retrieved_docs, collection.name,
                )
                if trigger:
                    state["graph_used"] = True
                    g_ctx, g_docs = await retrieve_graph_context(
                        question, collection, llm=retrieval_llm,
                    )
                    if g_docs:
                        existing = {_get_doc_id(d.page_content) for d in retrieved_docs}
                        added = 0
                        for d in g_docs:
                            did = _get_doc_id(d.page_content)
                            if did not in existing:
                                retrieved_docs.append(d)
                                existing.add(did)
                                added += 1
                        if added:
                            print(f"[Retrieval Agent] 触发知识图谱增强（{reason}），并入 {added} 条证据")
                    else:
                        print(f"[Retrieval Agent] 图谱已触发（{reason}）但无命中")
                else:
                    print(f"[Retrieval Agent] 文本检索质量达标，跳过知识图谱（{reason}）")
            except Exception as ge:
                print(f"[Retrieval Agent] 知识图谱检索失败，跳过: {ge}")

        return {
            "retrieved_docs": retrieved_docs,
            "route_history": state.get("route_history", []) + ["retrieval_agent"],
            "graph_used": state.get("graph_used", False),
        }

    except Exception as e:
        print(f"[Retrieval Agent] 检索过程中出错: {e}")
        return {
            "retrieved_docs": [],
            "route_history": state.get("route_history", []) + ["retrieval_agent"],
            "error": str(e),
            "graph_used": False,
        }
