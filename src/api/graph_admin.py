"""知识图谱构建接口：提交后台构建任务 + 轮询状态。

构建要对整个 collection 的 chunk 做 LLM 三元组抽取，可能耗时数分钟，
因此用后台任务跑，前端轮询状态；图谱文件已存在且未强制时直接返回既有统计。

拆出来的原因：这套状态（`_graph_jobs` / `_GRAPH_BUILD_FAIL_AT`）跨请求存活，
但只在两个端点里更新；放在 2600 行的 routes 里既难找，也容易被别的端点误改。
注意 `_GRAPH_BUILD_FAIL_AT` 还被对话端点的「自动构建退避」读，routes 从本模块导入。
"""

from __future__ import annotations

import asyncio
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.api.deps import check_rate as _check_rate
from src.api.deps import client_ip as _client_ip
from src.core.config import settings
from scenes.persona_chat.config import persona_chat_config
from framework.runtime import get_chroma_client

router = APIRouter()

# collection_name -> {status, started_at, stats, error}
_graph_jobs: dict[str, dict] = {}
# 构建失败时间戳（退避用）：失败后 1 小时内不自动重试，防每次对话重复触发注定失败的构建
_GRAPH_BUILD_FAIL_AT: dict[str, float] = {}
_graph_job_lock = threading.Lock()


class GraphBuildRequest(BaseModel):
    character_id: str
    force: bool = False  # 已存在时是否强制重建


def resolve_collection(character_id: str):
    """按 character_id 解析出 ChromaDB collection，找不到抛 404"""
    if character_id not in persona_chat_config.characters:
        raise HTTPException(status_code=404, detail=f"角色 '{character_id}' 不存在")
    col_name = persona_chat_config.characters[character_id].chroma_collection
    client = get_chroma_client()
    return client.get_or_create_collection(name=col_name, metadata={"hnsw:space": "cosine"})


async def run_graph_build(collection_name: str, force: bool = False):
    """后台构建任务：成功/失败都更新 _graph_jobs 状态"""
    if force:
        # 彻底重建：清掉旧成品与续传临时文件，从头抽取
        from src.retrieval.knowledge_graph import invalidate_graph, graph_path
        try:
            invalidate_graph(collection_name)
            tp = graph_path(collection_name).with_suffix(".json.tmp")
            if tp.exists():
                tp.unlink()
        except Exception as ex:
            print(f"[GraphBuild] force 清理失败（继续构建）: {ex}")
    with _graph_job_lock:
        _graph_jobs[collection_name] = {
            "status": "building", "started_at": time.time(), "stats": None, "error": None,
        }
    try:
        from src.retrieval.knowledge_graph import build_knowledge_graph
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        graph = await build_knowledge_graph(collection, resume=not force)
        with _graph_job_lock:
            _graph_jobs[collection_name] = {
                "status": "done", "started_at": _graph_jobs.get(collection_name, {}).get("started_at"),
                "stats": graph.get("stats"), "error": None,
            }
        print(f"[GraphBuild] 后台构建完成: {collection_name} -> {graph.get('stats')}")
    except Exception as e:
        print(f"[GraphBuild] 构建失败 {collection_name}: {e}")
        with _graph_job_lock:
            _graph_jobs[collection_name] = {
                "status": "error", "started_at": _graph_jobs.get(collection_name, {}).get("started_at"),
                "stats": None, "error": str(e),
            }
        _GRAPH_BUILD_FAIL_AT[collection_name] = time.time()


def graph_job_building(collection_name: str) -> bool:
    """本进程内该 collection 是否正在构建（对话端点的自动构建用它避重）。"""
    with _graph_job_lock:
        job = _graph_jobs.get(collection_name)
    return bool(job and job.get("status") == "building")


@router.post("/persona/graph/build")
async def build_character_graph(http_request: Request, request: GraphBuildRequest):
    """
    为某角色构建知识图谱（GraphRAG 增强层）。

    全流程异步：提交后立即可用 /persona/graph/status 轮询。
    图谱文件已存在且未 force 时直接返回既有统计，不重复构建。

    限流独立分桶（bucket=graph）：一次构建可能触发几百次 LLM 调用，
    是全站最重的操作，必须严限。
    """
    _check_rate(_client_ip(http_request), per_minute=2, per_day=10, bucket="graph")
    if not settings.graph_rag_enabled:
        raise HTTPException(status_code=400, detail="知识图谱 RAG 当前已关闭（graph_rag_enabled=false）")

    collection = resolve_collection(request.character_id)
    col_name = collection.name

    from src.retrieval.knowledge_graph import graph_exists, load_graph
    if graph_exists(col_name) and not request.force:
        g = load_graph(col_name) or {}
        return {
            "collection": col_name,
            "status": "exists",
            "stats": g.get("stats"),
            "message": "图谱已存在，使用 force=true 可强制重建",
        }

    with _graph_job_lock:
        job = _graph_jobs.get(col_name)
        if job and job.get("status") == "building":
            return {"collection": col_name, "status": "building", "message": "已在构建中，请稍后轮询"}

    try:
        asyncio.create_task(run_graph_build(col_name, force=request.force))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动构建任务失败: {str(e)}")

    return {"collection": col_name, "status": "building", "message": "已启动后台构建，请轮询 /persona/graph/status"}


@router.get("/persona/graph/status")
async def graph_build_status(character_id: str):
    """
    查询某角色知识图谱状态：是否已构建（exists）+ 后台任务进度（building/done/error）。
    """
    if character_id not in persona_chat_config.characters:
        raise HTTPException(status_code=404, detail=f"角色 '{character_id}' 不存在")
    col_name = persona_chat_config.characters[character_id].chroma_collection

    from src.retrieval.knowledge_graph import graph_exists, load_graph, graph_path
    exists = graph_exists(col_name)
    tmp_exists = graph_path(col_name).with_suffix(".json.tmp").exists()
    with _graph_job_lock:
        job = _graph_jobs.get(col_name, {})
    g = load_graph(col_name) or {}

    # 状态以最终文件为准，内存 job 仅表示“本进程内正在构建”
    if exists:
        status = job.get("status") if job.get("status") in ("done", "error") else "done"
    elif job.get("status") == "building":
        status = "building"
    elif tmp_exists:
        # 上次构建中途中断（成品未生成、仅留临时文件），可重跑续传
        status = "interrupted"
    else:
        status = "idle"

    return {
        "character_id": character_id,
        "collection": col_name,
        "exists": exists,
        "status": status,
        "stats": g.get("stats") if exists else (job.get("stats") if status == "done" else None),
        "error": job.get("error"),
    }
