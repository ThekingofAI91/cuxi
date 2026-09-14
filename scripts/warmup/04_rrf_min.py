"""
热身 04：手写 RRF 融合 —— 把项目里最"高级"的算法拆成 20 行
=========================================================
目的：RRF 听起来很唬人，其实就是给每个文档按名次发分数然后加总。
      手写一遍，面试再问就完全不虚。

运行：
    .venv/Scripts/python.exe scripts/warmup/04_rrf_min.py

对应项目：src/retrieval/advanced_search.py:559
    真实代码：doc_rrf_scores[doc_id] += weight / (rrf_k + rank + 1)   （rrf_k = 60）

为什么需要它（面试必答）：
    向量检索给的是"余弦相似度"（0~1），BM25 给的是"关键词得分"（0~几十），
    两者量纲不同，直接相加没有意义 —— 1.0 的余弦相似度和 15 分的 BM25 谁更重要？
    RRF 的解法：不看分数，只看名次。第 1 名给 1/(60+1)，第 2 名给 1/(60+2)…
    名次是任何检索器都天然有的东西，所以可以跨路合并。
"""

# ============================================================
# 模拟：同一批文档，两路检索各出一个排序结果
# ============================================================
# 场景：用户问"荣格的人格面具是什么"
#   向量路（语义强）：找出了语义相关的段落
#   BM25 路（关键字强）：精确命中了"人格面具"这个词

VECTOR_RANKED = [
    "doc_人格面具_原型章节",   # 第 1 名
    "doc_集体潜意识_总论",     # 第 2 名
    "doc_心理类型_概说",       # 第 3 名
    "doc_人格面具_二手解读",   # 第 4 名
]

BM25_RANKED = [
    "doc_人格面具_二手解读",   # 第 1 名（关键词密度高，但其实是二手评论）
    "doc_人格面具_原型章节",   # 第 2 名
    "doc_梦的解析_片段",       # 第 3 名
]

# 语料来源权重（项目里的 use_source_weight）：
# 原著/口述优先，二手解读降权 —— 避免第三者评价冒充名人原话
SOURCE_WEIGHT = {
    "doc_人格面具_原型章节": 1.0,
    "doc_集体潜意识_总论": 1.0,
    "doc_心理类型_概说": 1.0,
    "doc_人格面具_二手解读": 0.4,   # 二手，降权
    "doc_梦的解析_片段": 1.0,
}

RRF_K = 60  # 项目里的值。K 越大，名次之间的差距被压得越平


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """RRF 融合：按名次倒数和打分。"""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):  # rank 从 1 开始
            weight = SOURCE_WEIGHT.get(doc_id, 1.0)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


def naive_score_average() -> None:
    """反面对照：假装两路分数能直接相加（错误做法）。"""
    print("=" * 60)
    print("对照：如果硬把两路分数相加（错误做法）")
    print("=" * 60)
    print("  向量路分数：doc_人格面具_原型章节=0.91, doc_人格面具_二手解读=0.88")
    print("  BM25 路分数：doc_人格面具_二手解读=18.3, doc_人格面具_原型章节=12.1")
    print()
    print("  相加结果：二手解读 19.18 > 原型章节 13.01")
    print("  → 二手评论被排到原著前面了！")
    print("  根因：0.91 和 18.3 根本不是一个量纲，相加毫无意义。")
    print()


def main() -> None:
    print("=" * 60)
    print("两路检索的原始结果")
    print("=" * 60)
    print("  向量路（语义）：")
    for i, d in enumerate(VECTOR_RANKED, 1):
        print(f"    {i}. {d}")
    print("  BM25 路（关键词）：")
    for i, d in enumerate(BM25_RANKED, 1):
        print(f"    {i}. {d}")
    print()

    print("=" * 60)
    print("RRF 融合（按名次倒数和 + 来源权重）")
    print("=" * 60)
    scores = rrf_fuse([VECTOR_RANKED, BM25_RANKED])

    print(f"  公式：score += weight / (k + rank)，k={RRF_K}")
    print()
    for doc_id, score in sorted(scores.items(), key=lambda x: -x[1]):
        w = SOURCE_WEIGHT.get(doc_id, 1.0)
        print(f"    {score:.6f}  {doc_id}   (来源权重 {w})")

    print()
    print("  最终顺序：")
    for i, (doc_id, _) in enumerate(sorted(scores.items(), key=lambda x: -x[1]), 1):
        print(f"    {i}. {doc_id}")

    print()
    naive_score_average()

    print("=" * 60)
    print("观察到的两件事")
    print("=" * 60)
    print("  1. 两路都排得靠前的文档，RRF 分最高（共识优先）—— 这是融合的意义")
    print("  2. 二手解读虽然 BM25 排第 1，但因为来源权重 0.4 被压下去了")
    print()
    print("跑完想三个问题：")
    print("  1. RRF 为什么不需要归一化分数？")
    print("  2. 如果把 k 从 60 改成 1，排序会怎么变？（提示：名次差距被放大）")
    print("  3. 来源权重压在 RRF 分数上，和在重排阶段压，有什么区别？")


if __name__ == "__main__":
    main()
