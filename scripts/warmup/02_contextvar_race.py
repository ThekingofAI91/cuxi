"""
热身 02：并发串库 —— 亲手复现你自己的那个 bug
=============================================
目的：这是全项目技术含量最高的一个坑（supervisor.py:36 那 8 行注释）。
      自己把这个 bug 复现一遍，再修一遍，你就永远忘不掉了。

运行：
    .venv/Scripts/python.exe scripts/warmup/02_contextvar_race.py

背景（真实发生过的）：
    场景配置（用哪个角色的语料库、用哪套提示词）原本是进程级全局变量。
    一个用户提问时，代码把它切成"阿德勒"，
    结果另一个正在处理的"荣格"请求醒来，读到的却是阿德勒的配置 → 串库。
    单用户测试永远复现不了，只有并发才暴露。
"""

import asyncio
import contextvars

# ============================================================
# 错误示范：全局变量
# ============================================================
CURRENT_CHARACTER = {"id": "未设置", "collection": "未设置"}

# 每个角色的语料库
CHARACTERS = {
    "jung": {"id": "jung", "collection": "persona_jung"},
    "adler": {"id": "adler", "collection": "persona_adler"},
}


async def handle_request_broken(character_id: str) -> str:
    """有 bug 的实现：用全局变量传请求级配置。"""
    CURRENT_CHARACTER.update(CHARACTERS[character_id])  # 污染全局

    # 关键：这里有一次 await（真实项目里是检索、是 LLM 调用）
    # 一旦 await，事件循环就可能切去跑另一个请求 —— 全局变量被它改写
    await asyncio.sleep(0.05)

    # 醒来后读全局变量 —— 可能已经被别的请求改掉了
    got = CURRENT_CHARACTER["collection"]
    expect = CHARACTERS[character_id]["collection"]
    flag = "错误" if got != expect else "正确"
    return f"请求[{character_id}] 期望 {expect}，实际读到 {got} → {flag}"


# ============================================================
# 正确做法：ContextVar
# ============================================================
# ContextVar 的隔离单位是"上下文"，在 asyncio 里每个任务有独立上下文。
# create_task 创建子任务时会复制父任务的当前上下文 —— 天然请求级隔离。

_current_character = contextvars.ContextVar("current_character", default=None)


async def handle_request_fixed(character_id: str) -> str:
    """修好的实现：用 ContextVar 传请求级配置。"""
    _current_character.set(CHARACTERS[character_id])  # 只影响当前上下文

    await asyncio.sleep(0.05)  # 同样有 await，同样可能被切走

    cfg = _current_character.get()  # 读到的永远是本请求自己的
    got = cfg["collection"]
    expect = CHARACTERS[character_id]["collection"]
    flag = "错误" if got != expect else "正确"
    return f"请求[{character_id}] 期望 {expect}，实际读到 {got} → {flag}"


async def demo(handler, title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    # 同时发三个请求，交错执行 —— 这就是真实并发
    results = await asyncio.gather(
        handler("jung"),
        handler("adler"),
        handler("jung"),
    )
    for r in results:
        print("  " + r)
    print()


async def main() -> None:
    await demo(handle_request_broken, "错误示范：全局变量（会串库）")
    await demo(handle_request_fixed, "正确做法：ContextVar（互不干扰）")

    print("再想一层：为什么 create_task 能自动继承？")
    print("  contextvars 的上下文在任务创建时被复制一份给子任务，")
    print("  所以子任务里 set() 不会影响父任务，也不会影响兄弟任务。")

    # 顺手验证一下这个"继承"结论
    print()
    print("=" * 60)
    print("验证：子任务能否继承父上下文的设置")
    print("=" * 60)

    var = contextvars.ContextVar("demo", default="空")

    async def child() -> str:
        return var.get()

    async def parent() -> None:
        var.set("爸爸设的值")
        print(f"  父任务里读到：{var.get()}")
        # create_task 会复制当前上下文给子任务
        print(f"  子任务里读到：{await asyncio.create_task(child())}")
        print("  → 子任务继承了父任务的上下文，所以读得到")

    await parent()

    print()
    print("跑完想三个问题：")
    print("  1. 为什么 sleep(0.05) 这一行是 bug 暴露的必要条件？去掉会怎样？")
    print("  2. 这个 bug 单用户测试为什么测不出来？")
    print("  3. 还有什么办法能做请求级隔离？（想想 threading.local / 参数传递）")


if __name__ == "__main__":
    asyncio.run(main())
