"""
trace.py — 检索链路的结构化轨迹（评测专用）

为什么需要它：评测只拿到最终五维分，等于只知道「结果」，不知道「过程」。
池 10 注入 5 那轮五维几乎不动，但没人能回答「是排序没换对、还是换对了但上下文
变少」——因为 `advanced_retrieval` 只回传 (docs, contexts)，名次、分数、路由归属
全在函数内部被丢掉。

这里把「过程」显式外化成一份可序列化的记录，由调用方主动传入容器收集：

    trace = RetrievalTrace(question=q, character=ch)
    docs, ctx = await advanced_retrieval(..., trace_out=trace)
    ...
    json.dump(trace.to_dict())

刻意不挂 ContextVar：圆桌已经验证过节点 task 会分叉、ContextVar 传不过去
（roundtable 的耗时只能进 state 落 checkpoint）。显式传参最笨，也最不会出错。

生产请求不传 trace_out，所有采集分支都是 `if trace_out is not None` —— 零开销。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RetrievalTrace:
    """一次检索请求的过程记录。

    字段按「回答什么问题」分组，每组独立可空：
      boards  —— 每路召回的原始名次（谁被谁召回、第几名）
      fused   —— RRF 融合后的分数与类内名次（为什么是它进候选池）
      rerank  —— 精排换入换出（池 > 注入 时，精排到底改了什么）
      graph   —— 图谱是否触发、命中哪些实体、扩出多少关系
      truncated —— 注入时被截断的块（原始长 / 保留长 / 是否截在句末）
    """

    question: str = ""
    character: str = ""
    boards: list[dict] = field(default_factory=list)
    # doc_id -> {rrf_rank, score, best_rank{类:名次}, contrib{类:分数}, source, source_type}
    fused: dict[str, dict] = field(default_factory=dict)
    # {enabled, pool_size, top_k, topk_overlap, docs{doc_id: {rrf_rank, rerank_rank, rerank_score, kept}}}
    rerank: dict = field(default_factory=dict)
    # {triggered, reason, available, matched, seeds, edges, evidence, added}
    graph: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
