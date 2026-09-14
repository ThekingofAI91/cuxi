"""
热身 05：LangChain 最小调用 —— 把 langchain_openai 的手感找回来
==============================================================
目的：项目里所有 LLM 调用都走三种写法，认全它们就够读懂 90% 的代码。

运行（会真实调用 LLM，约 10 秒、成本忽略不计）：
    .venv/Scripts/python.exe scripts/warmup/05_llm_call.py

重要背景（面试会问）：
    项目**没有安装 langchain 这个包**，只用 langchain_core（消息/模板/接口）
    和 langchain_openai（ChatOpenAI）。被问"你用了 LangChain 哪些能力"，
    要答 langchain_core 的 Runnable 抽象 + ChatOpenAI + with_fallbacks。

三种写法（认全就行）：
    1. invoke / ainvoke        —— 一次性拿到完整回答
    2. astream                 —— 逐 token 拿到（项目主链路用这个）
    3. with_fallbacks / 结构化输出 —— 工程化能力

另外，请特别注意 max_tokens 的写法：项目里叫 max_tokens，
有些新版本 SDK 叫 max_completion_tokens，报错时改这个。
"""

import asyncio
import sys
from pathlib import Path

# 让脚本能 import 项目里的 src 包（脚本在 scripts/warmup/ 下，根目录要往上三层）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm import get_chat_llm

# 小样本，省钱省时间
SYSTEM = "你是一个简洁的助手，回答不超过 20 个字。"


async def demo_ainvoke() -> None:
    """写法 1：一次性拿完整回答（非流式）。"""
    print("=" * 60)
    print("写法 1：ainvoke（一次拿完整回答）")
    print("=" * 60)

    llm = get_chat_llm(temperature=0.3, max_tokens=64)
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content="用一句话说清什么是 RAG。"),
    ]
    resp = await llm.ainvoke(messages)

    # resp 是 AIMessage，正文在 .content
    print(f"  类型：{type(resp).__name__}")
    print(f"  正文：{resp.content}")
    print("  → 项目对照：analysis_agent.py:161 就是这么调的")


async def demo_astream() -> None:
    """写法 2：流式逐 token（项目主链路）。"""
    print()
    print("=" * 60)
    print("写法 2：astream（逐 token，SSE 的数据源）")
    print("=" * 60)

    llm = get_chat_llm(temperature=0.3, max_tokens=64)
    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content="用一句话说清什么是向量检索。"),
    ]

    print("  逐 token 输出：", end="", flush=True)
    pieces = []
    async for chunk in llm.astream(messages):
        # 注意：chunk.content 在部分模型/版本下可能是空字符串，要判空
        token = getattr(chunk, "content", None) or ""
        if token:
            print(token, end="", flush=True)
            pieces.append(token)
    print()
    print(f"  拼起来就是完整回答：{''.join(pieces)}")
    print("  → 项目对照：astream_nonempty()（llm.py:150）包了一层空响应重试")


async def demo_prompt_template() -> None:
    """写法 3：提示词模板（别用 f-string 硬拼）。"""
    print()
    print("=" * 60)
    print("写法 3：ChatPromptTemplate（提示词模板）")
    print("=" * 60)

    from langchain_core.prompts import ChatPromptTemplate

    # 模板里的 {占位符} 运行时填入，可复用、可测试
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你扮演{character}，只用一句话回答。"),
        ("human", "{question}"),
    ])

    # 先看模板渲染成什么（调试利器，钱都不花）
    rendered = prompt.format_messages(character="荣格", question="什么是原型？")
    print("  渲染结果：")
    for m in rendered:
        print(f"    [{m.type}] {m.content}")

    # 模板 + 模型 = 链（LCEL 的 | 语法，项目里见 prompt_builder.py 的组装思路）
    #
    # 这里故意用 ainvoke_nonempty 而不是裸 ainvoke —— 因为第三方中转会随机返回
    # "HTTP 200 但内容为空"的空壳响应。裸调用会拿到一个空字符串，
    # 在业务里就表现为"用户等了 13 秒，然后什么都没等到"。
    # 空响应重试把它当失败处理，再摇一次骰子（上游随机性下重试命中率很高）。
    from src.core.llm import ainvoke_nonempty

    chain = prompt | get_chat_llm(temperature=0.7, max_tokens=64)
    answer = await ainvoke_nonempty(chain, {"character": "荣格", "question": "什么是原型？"})
    content = (answer.content or "").strip()
    if content:
        print(f"  回答：{content}")
    else:
        print("  回答：（多次重试后上游仍返回空——这就是中转站的随机性，非脚本问题）")
    print("  → 项目对照：人物回答 = 人设 prompt + 检索上下文 + 历史，"
          "按固定顺序装配（scenes/persona_chat/prompt_builder.py）")


async def main() -> None:
    print("三种写法跑完，你就能读懂项目里所有 LLM 调用了。\n")
    await demo_ainvoke()
    await demo_astream()
    await demo_prompt_template()

    print()
    print("=" * 60)
    print("跑完想三个问题：")
    print("  1. ainvoke 和 astream 的返回类型分别是什么？")
    print("  2. 为什么流式要用 astream 而不是把 ainvoke 的结果切片返回？")
    print("  3. 项目里为什么要包一层 astream_nonempty？直接用 astream 会怎样？")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
