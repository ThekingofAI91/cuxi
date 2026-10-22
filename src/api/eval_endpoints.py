"""评测专用端点：一次调用返回「回答 + 注入上下文 + 检索轨迹」。

为什么要单独一个端点，而不是复用 `/persona/query`：
- `/persona/query` 是 SSE 流，逐 token 推事件；评测要的是定稿后的完整文本，
  走流会把 TestClient 缓冲、判分端解析都搞复杂。
- 评测必须能拿到 judge 可见的 contexts（含块尾续接、含图谱证据），而生产 SSE
  不把这些推给前端。
- 检索参数必须与主链路同口径（top_k=6、anchor_k、改写开关跟随 settings），
  否则测的是生产根本不跑的配置——这个坑 2026-10 踩过。

回答直接取 analysis_agent 的产物，不二次生成：二次生成会引入新的幻觉，
且会掩盖检索层的问题。
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from framework.runtime import get_chroma_client, set_scene_config
from scenes.persona_chat.config import persona_chat_config
from scenes.persona_chat.prompt_builder import build_character_prompt
from src.core.config import settings

router = APIRouter()


class EvalQueryRequest(BaseModel):
    """评估查询请求（返回检索上下文）"""
    query: str
    character_id: Optional[str] = "jung"


@router.post("/persona/eval_query")
async def persona_eval_endpoint(request: EvalQueryRequest):
    """
    评估专用端点：返回回答 + 检索到的上下文
    用于 RAG 评估时同时获取回答和检索上下文

    直接用 ChromaDB 向量查询检索相关文档（避免加载全部文档导致 SQLite 溢出）。
    """
    character_id = request.character_id or "jung"
    character = persona_chat_config.characters.get(character_id)
    if not character:
        raise HTTPException(status_code=400, detail=f"角色 '{character_id}' 不存在")

    # 切换到 persona 场景（contextvar 按请求隔离，不影响其他用户）
    from dataclasses import replace as _dc_replace
    set_scene_config(_dc_replace(persona_chat_config, chroma_collection=character.chroma_collection))

    try:
        from langchain_core.documents import Document
        from src.retrieval.embedder import get_embedder
        from src.retrieval.advanced_search import advanced_retrieval
        from src.retrieval.trace import RetrievalTrace
        from src.core.llm import get_chat_llm

        # ---- 1. 高级检索：多查询 + HyDE 组合 ----
        embedder = get_embedder()

        client = get_chroma_client()

        collection_name = character.chroma_collection  # e.g. "persona_jung"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 初始化 LLM（用于生成查询变体和 HyDE 文档；
        # max_tokens 已合并为一次调用，256 足够容纳短变体 + 3-5 句假设答案）
        retrieval_llm = get_chat_llm(
            temperature=0.3,
            max_tokens=256,
        )

        # 改写开关与主链路（retrieve_documents）保持同一配置口径——此前这里硬编码
        # use_multi_query=True，导致评估测的是生产根本不跑的配置（主链路默认关改写）
        use_rewrite = settings.rewrite_enabled

        # 执行高级检索（条数与主链路 retrieve_documents 保持一致：
        # RRF 前 3 直通 + 尾部 7 精排选 3 = 注入 6 条完整段落）
        # trace 只在评测端点启用：把每一路召回名次带出来，供离线脚本做
        # 「这个配置为什么没赚分」的归因；生产路径不传，采集分支全部跳过。
        trace = RetrievalTrace(question=request.query, character=character_id)
        retrieved_docs, contexts = await advanced_retrieval(
            question=request.query,
            collection=collection,
            llm=retrieval_llm,
            top_k=6,
            use_multi_query=use_rewrite,
            use_hyde=use_rewrite,
            num_variants=2,
            anchor_k=settings.rerank_anchor_k,
            trace_out=trace,
        )

        # 与 _build_context 同口径：断句的块在注入时会被补上续接句，判官看到的
        # contexts 必须是同一份文本，否则「上下文精确度」量的是模型没见过的版本。
        from src.retrieval.stitch import with_continuation as _stitch
        contexts = [
            _stitch(d.page_content or "", d.metadata)[0]
            for d in retrieved_docs
        ]

        # ---- 1.5 知识图谱增强（与主链路 retrieve_documents 同口径）----
        # 此前评估端点直接调 advanced_retrieval，完全绕过主链路的图谱门——
        # 荣格图谱建成后的首轮"图谱增强版评测"跑了几题才发现测的还是无图谱基线。
        # 这里复刻 supervisor_agent 的三信号触发 + 证据文档并入（评估无轻量通道，省略 not light_retrieval）
        graph_used = False
        # 图谱侧轨迹：即便没触发也要留结构（triggered=False + 原因），
        # 「为什么这题没走图谱」和「走了图谱但没命中」是两种完全不同的失败。
        graph_trace = {
            "triggered": False, "reason": "", "available": False,
            "matched": 0, "seeds": [], "edges": 0, "evidence": 0, "added": 0,
        }
        if settings.graph_rag_enabled and graph_exists(collection.name):
            try:
                from src.retrieval.knowledge_graph import (
                    retrieve_graph_context,
                    should_trigger_graph_retrieval,
                )
                from src.retrieval.advanced_search import _get_doc_id

                trigger, reason = should_trigger_graph_retrieval(
                    request.query, retrieved_docs, collection.name,
                )
                graph_trace["triggered"] = trigger
                graph_trace["reason"] = reason
                if trigger:
                    graph_used = True
                    g_ctx, g_docs = await retrieve_graph_context(
                        request.query, collection, llm=retrieval_llm,
                        trace_out=graph_trace,
                    )
                    if g_docs:
                        existing = {_get_doc_id(d.page_content) for d in retrieved_docs}
                        added_docs = []
                        for d in g_docs:
                            did = _get_doc_id(d.page_content)
                            if did not in existing:
                                retrieved_docs.append(d)
                                existing.add(did)
                                added_docs.append(d)
                        if added_docs:
                            # 证据文档与主链路一样进生成上下文，也要进 judge 可见的 contexts
                            contexts.extend(d.page_content for d in added_docs)
                            graph_trace["added"] = len(added_docs)
                            print(f"[Eval Query] 触发知识图谱增强（{reason}），并入 {len(added_docs)} 条证据")
                    else:
                        print(f"[Eval Query] 图谱已触发（{reason}）但无命中")
                else:
                    print(f"[Eval Query] 文本检索质量达标，跳过知识图谱（{reason}）")
            except Exception as ge:
                print(f"[Eval Query] 知识图谱检索失败，跳过: {ge}")

        trace.graph = graph_trace

        print(f"[Eval Query] 高级检索到 {len(retrieved_docs)} 条相关文档（图谱{'已用' if graph_used else '未用'}）")

        # ---- 2. 分析 ----
        from framework.analysis_agent import analysis_agent
        from src.core.state import AgentState

        state: AgentState = {
            "query": request.query,
            "session_id": "eval",
            "retrieved_docs": retrieved_docs,
            "retrieval_trace": trace,
            "analysis": "",
            "code_result": "",
            "verification": "",
            "final_answer": "",
            "history": [],
            "route_history": [],
            "next_agent": None,
            "error": None,
            # 评估要测的是线上真实行为，故同样走酒馆式装配（人设+性格+场景+世界书+示例）
            "character_role_prompt": build_character_prompt(character, query=request.query),
            "stream_callback": None,
            "info_gap_questions": None,
        }

        analysis_result = await analysis_agent(state)
        analysis = analysis_result.get("analysis", "")

        # ---- 3. 直接用分析结果作为回答（避免二次生成引入幻觉）----
        # analysis_agent 已有严格 prompt 要求基于参考资料，无需再调 _generate_direct_response
        answer = analysis

        return {
            "answer": answer,
            "contexts": contexts,  # 返回全部上下文，确保 judge 能完整评估
            "analysis": analysis[:500],
            # 过程记录：判官看不到，只给离线归因脚本吃
            "retrieval_trace": trace.to_dict(),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"ERROR: {str(e)}", "contexts": [], "analysis": "", "retrieval_trace": {}}
