"""
热身 01：async / await 的真实手感
================================
目的：不是"看懂"，是"跑出来看到差别"，把 asyncio 的直觉接回来。

运行：
    .venv/Scripts/python.exe scripts/warmup/01_async_hands.py

对应项目里的位置：
    - 串行 vs 并发  → advanced_search.py:403（改写任务与检索重叠执行）
    - 生产者消费者  → src/api/routes.py:1447（token Queue 喂给 SSE）
    - gather        → 两路检索并行
"""

import asyncio
import time

# ============================================================
# 第 1 部分：串行 vs 并发
# ============================================================
# 核心心智模型：
#   await = "我在这里等，但把 CPU 让给别人"
#   await 在同一个任务里遇到 IO 会挂起自己，事件循环去跑其他就绪任务
#   所以 3 个各睡 1 秒的任务：
#     串行 await → 3 秒（一个等完再等下一个）
#     并发 gather → 1 秒（三个一起等）


async def fake_fetch(name: str, seconds: float) -> str:
    """模拟一次网络/模型调用：等待 seconds 秒后返回。"""
    await asyncio.sleep(seconds)
    return f"{name} 完成"


async def run_serial() -> None:
    t0 = time.perf_counter()
    r1 = await fake_fetch("向量检索", 1.0)
    r2 = await fake_fetch("BM25检索", 1.0)
    r3 = await fake_fetch("改写LLM", 1.0)
    cost = time.perf_counter() - t0
    print(f"  串行：{r1} / {r2} / {r3}")
    print(f"  串行总耗时：{cost:.2f}s")


async def run_concurrent() -> None:
    t0 = time.perf_counter()
    # gather 把多个协程同时丢进事件循环，等它们全部返回
    results = await asyncio.gather(
        fake_fetch("向量检索", 1.0),
        fake_fetch("BM25检索", 1.0),
        fake_fetch("改写LLM", 1.0),
    )
    cost = time.perf_counter() - t0
    print(f"  并发：{' / '.join(results)}")
    print(f"  并发总耗时：{cost:.2f}s")


# ============================================================
# 第 2 部分：create_task —— 先挂起，后收结果
# ============================================================
# 项目里 advanced_search 就是这么做的：
#   1) create_task 把改写丢出去（不 await，先不管它）
#   2) 立刻跑原始查询的向量 + BM25
#   3) 原始路跑完，再回来等改写结果
# 效果：改写那 2 秒被"藏"在检索时间里了。


async def rewrite_query() -> str:
    await asyncio.sleep(2.0)  # 模拟 LLM 改写
    return "改写后的查询变体"


async def original_retrieval() -> str:
    await asyncio.sleep(1.2)  # 模拟原始查询的向量+BM25
    return "原始查询的检索结果"


async def run_overlap() -> None:
    t0 = time.perf_counter()

    # 注意：create_task 后必须留着句柄，否则任务可能被回收
    rewrite_task = asyncio.create_task(rewrite_query())

    # 不等改写，先干原始检索
    got = await original_retrieval()
    print(f"  原始检索回来：{got}（耗时 {time.perf_counter() - t0:.2f}s）")

    # 现在才收改写结果——此时它多半已经跑完了
    rewritten = await rewrite_task
    print(f"  改写回来：{rewritten}（耗时 {time.perf_counter() - t0:.2f}s）")
    print(f"  重叠执行总耗时：{time.perf_counter() - t0:.2f}s")
    print("  对比：如果先 await 改写再检索，要 3.2s")


# ============================================================
# 第 3 部分：asyncio.Queue —— SSE 流式的骨架
# ============================================================
# 项目 routes.py:1447 的结构完全一样：
#   生产者：LLM 每吐一个 token，就 await queue.put(token)
#   消费者：event_stream 循环 await queue.get()，推给浏览器
# 为什么用 Queue 而不是 list？
#   Queue 有"等待"语义：消费者 get() 时队列空会自动挂起，来数据自动唤醒。
#   list 要自己写轮询 + sleep，既费 CPU 又有延迟。


async def llm_stream(queue: asyncio.Queue, tokens: list[str]) -> None:
    """生产者：模拟 LLM 逐 token 输出。"""
    for tok in tokens:
        await asyncio.sleep(0.15)
        await queue.put(tok)
    await queue.put(None)  # 哨兵值，告诉消费者"结束了"


async def sse_pusher(queue: asyncio.Queue) -> str:
    """消费者：模拟把 token 推给前端，拼出完整回答。"""
    pieces = []
    while True:
        tok = await queue.get()
        if tok is None:
            break
        print(f"    → 推送 token：{tok}")
        pieces.append(tok)
    return "".join(pieces)


async def run_stream() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    producer = asyncio.create_task(llm_stream(queue, ["心灵", "分为", "三个", "层次"]))
    answer = await sse_pusher(queue)
    await producer
    print(f"  前端拼出的完整回答：{answer}")


async def main() -> None:
    print("=" * 60)
    print("第 1 部分：串行 vs 并发（同样的工作量，时间差 3 倍）")
    print("=" * 60)
    await run_serial()
    print()
    await run_concurrent()

    print()
    print("=" * 60)
    print("第 2 部分：create_task 重叠执行（把你项目的优化搬过来了）")
    print("=" * 60)
    await run_overlap()

    print()
    print("=" * 60)
    print("第 3 部分：Queue 生产者消费者（SSE 流式骨架）")
    print("=" * 60)
    await run_stream()

    print()
    print("跑完想三个问题：")
    print("  1. 为什么 await fake_fetch(...) 三次是 3 秒，gather 是 1 秒？")
    print("  2. create_task 和直接 await 的区别，用一句话说清？")
    print("  3. Queue 换成 list 会多出什么问题？")


if __name__ == "__main__":
    asyncio.run(main())
