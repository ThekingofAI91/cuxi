"""
build_graphs_all.py — GraphRAG 建图战役：批量构建所有教育区角色的知识图谱

背景：graph_rag_enabled 默认开启，但图谱文件从未建成过（自动构建总被上游
空响应/429 打断，1 小时退避又拦住重试）——招牌功能实际处于休眠状态。
本脚本按库逐个构建：节流（批间延迟防 429）+ 断点续传（构建函数自带）+
上游故障退避重试，可反复执行直到全部建成。

用法：
    .venv/Scripts/python.exe scripts/build_graphs_all.py                # 全部缺失的
    .venv/Scripts/python.exe scripts/build_graphs_all.py --force        # 强制重建
    .venv/Scripts/python.exe scripts/build_graphs_all.py --delay 8      # 批间延迟秒数
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.retrieval.knowledge_graph import build_knowledge_graph, graph_exists

# 教育区角色库（娱乐区不走图谱检索，不建）
COLLECTIONS = ["persona_jung", "persona_wangyangming", "persona_adler"]

UPSTREAM_BACKOFF_SEC = 180   # 上游空响应熔断/429 后的等待
MAX_ATTEMPTS_PER_COL = 10    # 单库最大尝试轮数（resume 使每轮都从断点继续）


async def build_one(collection, force: bool, delay: float) -> bool:
    """构建单库图谱，带退避重试；返回是否最终成功"""
    name = collection.name
    if graph_exists(name) and not force:
        print(f"[campaign] {name} 图谱已存在，跳过（--force 可重建）")
        return True

    for attempt in range(1, MAX_ATTEMPTS_PER_COL + 1):
        t0 = time.time()
        try:
            graph = await build_knowledge_graph(
                collection, resume=True, inter_batch_delay=delay,
            )
            stats = graph["stats"]
            print(f"[campaign] {name}: {stats['chunks']} chunk → "
                  f"{stats['entities']} 实体 / {stats['relations']} 关系"
                  f" | 耗时 {time.time() - t0:.0f}s")
            return True
        except RuntimeError as e:
            # 构建函数的空批熔断（上游持续空响应）：等待后续传重试
            print(f"[campaign] {name} 第 {attempt} 轮中止: {e}")
            print(f"[campaign] {UPSTREAM_BACKOFF_SEC}s 后续传重试…")
            await asyncio.sleep(UPSTREAM_BACKOFF_SEC)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "exhausted" in msg.lower():
                wait = UPSTREAM_BACKOFF_SEC
            else:
                wait = 30
            print(f"[campaign] {name} 第 {attempt} 轮异常: {msg[:120]}")
            print(f"[campaign] {wait}s 后续传重试…")
            await asyncio.sleep(wait)

    print(f"[campaign] {name} 达到最大尝试轮数，放弃（进度已存 .tmp，可直接重跑续传）")
    return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG 批量建图战役")
    parser.add_argument("--force", action="store_true", help="已存在也强制重建")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="批间延迟秒数（默认 5s ≈ 防中转 429）")
    parser.add_argument("--only", type=str, default="", help="只构建指定 collection")
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    targets = [args.only] if args.only else COLLECTIONS
    t0 = time.time()
    results = {}
    for name in targets:
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        count = col.count()
        print(f"\n[campaign] === {name}（{count} 条 chunk）===")
        if count == 0:
            print(f"[campaign] {name} 文档库为空，跳过")
            results[name] = "empty"
            continue
        results[name] = await build_one(col, args.force, args.delay)

    print("\n" + "=" * 60)
    print(f"[campaign] 战役结束（总耗时 {(time.time() - t0) / 60:.1f} 分钟）:")
    for name, ok in results.items():
        mark = {"empty": "[空库]", "True": "[完成]", "False": "[失败]"}.get(str(ok), str(ok))
        print(f"  {mark} {name}")


if __name__ == "__main__":
    asyncio.run(main())
