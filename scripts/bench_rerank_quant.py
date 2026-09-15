"""
bench_rerank_quant.py — Cross-Encoder int8 动态量化收益实测

模拟生产条件（torch 8 线程、(query, doc) 双侧截断 200 字符、max_length=256、
5/10 候选）对比三种配置的单次 rerank 耗时与排序一致性：
  1. bge-reranker-v2-m3 fp32（现状）
  2. bge-reranker-v2-m3 int8 动态量化（候选方案：torch.ao.quantization.quantize_dynamic）
  3. bge-reranker-base fp32（备选换模型方案）

语料直接取自本地 ChromaDB persona_jung collection，保证文本长度/分词贴近真实。
用法：.venv/Scripts/python.exe scripts/bench_rerank_quant.py
"""

import statistics
import time

import torch

torch.set_num_threads(8)

from sentence_transformers import CrossEncoder

QUERIES = [
    "什么是集体无意识？它和个人无意识有什么区别？",
    "内向和外向性格的人应该如何选择职业方向？",
    "高考志愿填报有哪些常见误区？",
]


def _load_corpus(n: int = 12) -> list[str]:
    """从本地 ChromaDB 取真实 chunk（截到 200 字符，与生产截断一致）"""
    try:
        import chromadb

        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_or_create_collection("persona_jung")
        raw = col.get(include=["documents"], limit=n, offset=0)
        docs = [d for d in (raw.get("documents") or []) if d]
        if len(docs) >= 5:
            print(f"[bench] 使用真实语料 {len(docs)} 条（persona_jung）")
            return [d[:200] for d in docs[:n]]
    except Exception as e:
        print(f"[bench] ChromaDB 取语料失败，改用内置文本: {e}")
    base = (
        "集体无意识是人类心理的一部分，它不同于个体无意识。个体无意识由被压抑的情结构成，"
        "而集体无意识则由原型组成，是从祖先世代经验中沉淀下来的心理倾向。原型包括阴影、"
        "阿尼玛、阿尼姆斯与自性等。阴影是人不愿承认的那部分自己；自性化则是把人格各部分"
        "整合为统一整体的过程，是心理发展的最终目标。"
    )
    return [(base * 3)[:200]] * n


def _bench(model, pairs, n_iter: int = 5) -> tuple[list[float], list[float]]:
    """返回 (每次耗时 ms 列表, 最后一轮分数)"""
    model.predict(pairs[:2], show_progress_bar=False, max_length=256)  # warmup
    times: list[float] = []
    scores: list[float] = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        scores = model.predict(pairs, show_progress_bar=False, max_length=256)
        times.append((time.perf_counter() - t0) * 1000)
    return times, [float(s) for s in scores]


def _kendall_tau(a: list[float], b: list[float]) -> float:
    """两组分数的排序一致度（Kendall tau，1 = 完全同序）"""
    n = len(a)
    if n < 2:
        return 1.0
    ra = sorted(range(n), key=lambda i: -a[i])
    rb = sorted(range(n), key=lambda i: -b[i])
    pos_b = {doc: i for i, doc in enumerate(rb)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = pos_b[ra[i]], pos_b[ra[j]]
            if pi < pj:
                conc += 1
            else:
                disc += 1
    return conc / max(conc + disc, 1)


def main() -> None:
    docs = _load_corpus()
    queries = QUERIES[: max(1, len(docs) // len(QUERIES))]

    # 构造生产同款 (query, doc) 对：doc 按 5/10 候选分组
    all_pairs = []
    for qi, q in enumerate(QUERIES):
        for d in docs:
            all_pairs.append((q, d))

    results: dict[str, dict] = {}

    print("\n[bench] === 1) bge-reranker-v2-m3 fp32（现状） ===")
    m1 = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    for k in (5, 10):
        pairs = all_pairs[:k]
        times, scores = _bench(m1, pairs)
        results[f"v2m3_fp32_{k}"] = {"times": times, "scores": scores}
        print(f"  {k} 对: median {statistics.median(times):.0f} ms  min {min(times):.0f} ms")

    # 注意：只量化 encoder 层。torch 2.13 下整模型量化会破坏 transformers 的输入
    # 解包（embeddings 层把 BatchEncoding 当 tensor 用 → AttributeError），
    # 且 embeddings/分类头只占计算量 ~1%，量化收益趋近于零。
    print("\n[bench] === 2) bge-reranker-v2-m3 int8 动态量化（仅 encoder 层） ===")
    m1.model.roberta.encoder = torch.ao.quantization.quantize_dynamic(
        m1.model.roberta.encoder, {torch.nn.Linear}, dtype=torch.qint8
    )
    for k in (5, 10):
        pairs = all_pairs[:k]
        times, scores = _bench(m1, pairs)
        results[f"v2m3_int8_{k}"] = {"times": times, "scores": scores}
        print(f"  {k} 对: median {statistics.median(times):.0f} ms  min {min(times):.0f} ms")

    print("\n[bench] === 3) bge-reranker-base fp32（参考） ===")
    try:
        m3 = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        for k in (5, 10):
            pairs = all_pairs[:k]
            times, scores = _bench(m3, pairs)
            results[f"base_fp32_{k}"] = {"times": times, "scores": scores}
            print(f"  {k} 对: median {statistics.median(times):.0f} ms  min {min(times):.0f} ms")
    except Exception as e:
        print(f"  base 加载失败: {e}")

    print("\n[bench] ===== 汇总 =====")
    for k in (5, 10):
        fp = results.get(f"v2m3_fp32_{k}")
        i8 = results.get(f"v2m3_int8_{k}")
        bs = results.get(f"base_fp32_{k}")
        if fp and i8:
            sp = statistics.median(fp["times"])
            si = statistics.median(i8["times"])
            tau = _kendall_tau(fp["scores"], i8["scores"])
            print(f"{k} 对: fp32 {sp:.0f}ms → int8 {si:.0f}ms（{sp / si:.2f}x），排序一致度 tau={tau:.3f}")
        if fp and bs:
            sp = statistics.median(fp["times"])
            sb = statistics.median(bs["times"])
            tau = _kendall_tau(fp["scores"], bs["scores"])
            print(f"{k} 对: fp32 {sp:.0f}ms → base {sb:.0f}ms（{sp / sb:.2f}x），排序一致度 tau={tau:.3f}")


if __name__ == "__main__":
    main()
