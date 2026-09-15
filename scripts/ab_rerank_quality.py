"""
ab_rerank_quality.py — 重排模型替换的质量 A/B 验收（v2-m3 vs base）

对每个真实查询跑完整 advanced_retrieval 管线（RRF 候选 → Cross-Encoder 精排），
仅切换重排模型，对比最终进入 LLM prompt 的上下文序列：
  - head5 集合重合：前 5 条（精排区）的文档集合重合度——直接决定 prompt 头部内容
  - head5 顺序一致（Kendall tau）
  - full15 集合 Jaccard：整份上下文的文档集合差异
  - 重排耗时

注意：reranker 的分数缓存键不含模型维度（进程内单模型假设），切换后必须清空。
用法：.venv/Scripts/python.exe scripts/ab_rerank_quality.py
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings as ChromaSettings

import src.retrieval.reranker as RR
from src.retrieval.advanced_search import _get_doc_id, advanced_retrieval

JUNG_Q = [
    "荣格把心灵分为哪三个层次？",
    "什么是人格面具（Persona）？",
    "荣格提出的四种心理功能是什么？",
    "什么是共时性（Synchronicity）？",
    "阴影（Shadow）在荣格心理学中代表什么？它只有负面含义吗？",
    "个性化过程（Individuation）的主要阶段有哪些？",
    "荣格和弗洛伊德对梦的理解有什么不同？",
    "荣格和弗洛伊德是什么时候决裂的？原因是什么？",
    "荣格如何用炼金术象征来描述个性化过程？",
    "什么是积极想象（Active Imagination）？荣格自己是如何使用这个技术的？",
    "阿尼玛和阿尼姆斯分别是什么？它们在心理发展中起什么作用？",
    "荣格认为自性（Self）是什么？它和自我（Ego）有什么区别？",
]
ADLER_Q = [
    "什么是自卑感？它和自卑情结有什么区别？",
    "阿德勒所说的社会兴趣是什么？",
    "生活风格是怎么形成的？",
    "阿德勒如何看待梦？",
    "什么是过度补偿？",
    "阿德勒和弗洛伊德为什么会分道扬镳？",
]
WYYM_Q = [
    "什么是心即理？",
    "知行合一是什么意思？",
    "什么是致良知？",
    "四句教的内容是什么？",
    "王阳明是如何在龙场悟道的？",
    "什么是事上磨练？",
]

QUERY_SETS = [
    ("persona_jung", JUNG_Q),
    ("persona_adler", ADLER_Q),
    ("persona_wangyangming", WYYM_Q),
]

MODELS = ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"]


def kendall_tau_from_orders(a: list[str], b: list[str]) -> float:
    """两个文档序列（同集合）的 Kendall tau；集合不同时按共同部分计算"""
    common = [d for d in a if d in set(b)]
    if len(common) < 2:
        return 1.0
    pos_b = {d: i for i, d in enumerate([x for x in b if x in set(a)])}
    rb = [x for x in b if x in set(a)]
    conc = disc = 0
    seq_a = [d for d in a if d in set(b)]
    for i in range(len(seq_a)):
        for j in range(i + 1, len(seq_a)):
            pi, pj = pos_b[seq_a[i]], pos_b[seq_a[j]]
            if pi < pj:
                conc += 1
            else:
                disc += 1
    return conc / max(conc + disc, 1)


async def run_with_model(model_name: str) -> dict:
    """以指定重排模型跑全部查询，返回 {query_key: {ids, rerank_ms}}"""
    RR._reranker = RR.Reranker(model_name=model_name)
    t0 = time.perf_counter()
    _ = RR._reranker.model  # 触发加载 + 量化
    # 分数缓存键是 (query, doc) 不含模型维度，切换模型必须清空，
    # 否则后跑的模型直接命中前一个模型的分数，A/B 被缓存污染
    RR._SCORE_CACHE.clear()
    print(f"\n[ab] {model_name} 加载+量化 {time.perf_counter() - t0:.1f}s（分数缓存已清空）")

    client = chromadb.PersistentClient(
        path="./chroma_db", settings=ChromaSettings(anonymized_telemetry=False)
    )
    results: dict[str, dict] = {}
    for col_name, questions in QUERY_SETS:
        col = client.get_or_create_collection(col_name, metadata={"hnsw:space": "cosine"})
        for q in questions:
            t0 = time.perf_counter()
            docs, _ctx = await advanced_retrieval(
                q, col, llm=None, top_k=15,
                use_multi_query=False, use_hyde=False, use_rerank=True,
            )
            wall = (time.perf_counter() - t0) * 1000
            key = f"{col_name}::{q}"
            results[key] = {
                "ids": [_get_doc_id(d.page_content) for d in docs],
                "head5_scores": [round(float(d.metadata.get("rerank_score", 0.0)), 4) for d in docs[:5]],
                "wall_ms": round(wall),
            }
            print(f"[ab] {model_name.split('/')[-1]} | {col_name.replace('persona_', '')} | {q[:22]:<22} | {len(docs)} 条 | {wall:.0f} ms")
    return results


def compare(res_a: dict, res_b: dict, name_a: str, name_b: str) -> None:
    print("\n" + "=" * 88)
    print(f"[对比] head5 集合重合 / head5 顺序 tau / full15 Jaccard   A={name_a}  B={name_b}")
    print("=" * 88)
    head_overlaps, full_jaccards, taus = [], [], []
    identical_head5 = identical_head1 = 0
    rows = []
    for key in res_a:
        a, b = res_a[key]["ids"], res_b[key]["ids"]
        head_a, head_b = set(a[:5]), set(b[:5])
        overlap = len(head_a & head_b) / 5
        full_j = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
        tau = kendall_tau_from_orders(a[:5], b[:5])
        head_overlaps.append(overlap)
        full_jaccards.append(full_j)
        taus.append(tau)
        identical_head5 += head_a == head_b
        identical_head1 += a and b and a[0] == b[0]
        q = key.split("::")[1]
        rows.append((key.split("::")[0].replace("persona_", ""), q, overlap, tau, full_j))
    for col, q, ov, tau, fj in rows:
        flag = "  [低]" if ov < 0.6 else ""
        print(f"  {col:<12} {q[:30]:<30} head5={ov:.0%}  tau={tau:.2f}  full15={fj:.0%}{flag}")
    n = len(rows)
    print("-" * 88)
    print(f"  head5 平均重合: {statistics.mean(head_overlaps):.1%} | 完全一致: {identical_head5}/{n}")
    print(f"  top1   完全一致: {identical_head1}/{n}")
    print(f"  head5 顺序 tau 均值: {statistics.mean(taus):.3f}")
    print(f"  full15 Jaccard 均值: {statistics.mean(full_jaccards):.1%}")
    ta = statistics.median(r["wall_ms"] for r in res_a.values())
    tb = statistics.median(r["wall_ms"] for r in res_b.values())
    print(f"  每查询总耗时中位: {name_a} {ta:.0f} ms vs {name_b} {tb:.0f} ms（含检索，重排为主）")


def main() -> None:
    all_res = {}
    for m in MODELS:
        all_res[m] = asyncio.run(run_with_model(m))

    compare(
        all_res[MODELS[0]], all_res[MODELS[1]],
        MODELS[0].split("/")[-1], MODELS[1].split("/")[-1],
    )

    out = Path("output")
    out.mkdir(exist_ok=True)
    path = out / "rerank_ab_results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=1)
    print(f"\n[ab] 明细已存 {path}")


if __name__ == "__main__":
    main()
