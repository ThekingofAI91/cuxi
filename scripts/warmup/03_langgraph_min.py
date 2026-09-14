"""
热身 03：LangGraph 最小可跑版 —— 把你项目的大脑缩小 100 倍
=========================================================
目的：LangGraph 忘了没关系，这个例子 60 行讲完核心。
      跑通它，再回去看 framework/supervisor.py:648 就是同一个东西。

运行：
    .venv/Scripts/python.exe scripts/warmup/03_langgraph_min.py

注意：这个例子**不调用 LLM**（用规则代替判断），所以零成本、秒出结果。
      先把"图怎么跑"搞懂，模型调用是后面的事。

LangGraph 三个核心概念：
    State       —— 一个共享字典，所有节点都读写它
    Node        —— 一个函数，输入 state，返回"要更新的字段"
    Edge        —— 决定下一步走哪个节点（普通边 / 条件边）

项目对照：
    本项目 supervisor.py 的图 = supervisor → retriever/analyzer/verifier → 回到 supervisor
    多一个 verifier 节点、多几条回边，结构完全一样。
"""

import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

# ============================================================
# 1. State：所有节点共享的状态
# ============================================================
# 项目里叫 AgentState，定义在 src/core/state.py，字段更多。
# 关键理解：节点不需要返回完整 state，只返回"我改了哪几个字段"，
#          LangGraph 负责把增量合并进去。


class State(TypedDict):
    query: str
    route: str                 # supervisor 决定的下一步
    retrieved_docs: list[str]
    answer: str
    route_history: list[str]   # 记录走了哪些节点，方便调试
    loops: int                 # 走了几圈，防止死循环


# ============================================================
# 2. Node：每个节点是一个函数
# ============================================================

def supervisor_node(state: State) -> dict[str, Any]:
    """大脑：判断下一步走谁。真实项目里这里是 LLM 调用。"""
    loops = state.get("loops", 0) + 1
    query = state["query"]

    # 用规则代替 LLM：有"闲聊"就直答，否则先检索
    if loops == 1 and "闲聊" not in query:
        route = "retriever"
    elif loops == 1 and "闲聊" in query:
        route = "analyzer"
    else:
        route = "analyzer"  # 检索完 → 去生成

    print(f"    [supervisor] 第 {loops} 次进来 → 决定走 {route}")
    return {"route": route, "loops": loops, "route_history": state.get("route_history", []) + ["supervisor"]}


def retriever_node(state: State) -> dict[str, Any]:
    """检索节点：真实项目里跑混合检索 + 图谱扩展。"""
    print("    [retriever] 开始检索…")
    docs = ["荣格语料片段A：心灵分为意识/个人潜意识/集体潜意识", "荣格语料片段B：人格面具是原型之一"]
    return {"retrieved_docs": docs, "route_history": state["route_history"] + ["retriever"]}


def analyzer_node(state: State) -> dict[str, Any]:
    """生成节点：真实项目里带人设流式生成。"""
    print("    [analyzer] 生成回答…")
    if state.get("retrieved_docs"):
        answer = f"（基于 {len(state['retrieved_docs'])} 条资料回答）这个问题要分三个层次讲…"
    else:
        answer = "（闲聊）你好，我是这里的角色。"
    return {"answer": answer, "route_history": state["route_history"] + ["analyzer"]}


# ============================================================
# 3. 条件边：决定 supervisor 之后往哪走
# ============================================================
# 项目里这个函数读 state["next_agent"]，本项目结构一致。

def route_after_supervisor(state: State) -> str:
    return state["route"]


# ============================================================
# 4. 组装图
# ============================================================

def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)

    workflow.add_edge(START, "supervisor")

    # 条件边：supervisor 的输出决定去 retriever 还是 analyzer
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"retriever": "retriever", "analyzer": "analyzer"},
    )

    # 回边：检索完回到 supervisor 再决策（项目里就是这样）
    workflow.add_edge("retriever", "supervisor")
    workflow.add_edge("analyzer", END)

    return workflow.compile()


# ============================================================
# 5. 跑
# ============================================================

async def run_query(query: str) -> None:
    print(f"\n提问：{query}")
    graph = build_graph()

    initial: State = {
        "query": query,
        "route": "",
        "retrieved_docs": [],
        "answer": "",
        "route_history": [],
        "loops": 0,
    }

    # stream_mode="updates"：每跑完一个节点就吐一次，方便做进度提示
    # 项目 routes.py:1538 用的就是这个模式，把节点完成事件推给前端
    accumulated: dict = {}
    async for update in graph.astream(initial, config={"recursion_limit": 25}):
        for node_name, node_output in update.items():
            print(f"    ← 节点 {node_name} 返回：{list(node_output.keys())}")
            accumulated.update(node_output)

    print(f"  最终回答：{accumulated['answer']}")
    print(f"  走过的节点：{' → '.join(accumulated['route_history'])}")


async def main() -> None:
    print("=" * 60)
    print("跑两个问题，看图的路线怎么变")
    print("=" * 60)
    await run_query("荣格把心灵分为哪三个层次？")
    await run_query("（闲聊）今天天气不错")

    print()
    print("=" * 60)
    print("对照你项目的图（framework/supervisor.py:645 附近）：")
    print("  workflow.add_edge(START, 'supervisor')")
    print("  workflow.add_node('supervisor', supervisor_node)")
    print("  workflow.add_node('retriever', retrieval_agent)")
    print("  workflow.add_node('analyzer', analysis_agent)")
    print("  workflow.add_node('verifier', verification_agent)")
    print("  workflow.add_edge('retriever', 'supervisor')   ← 回边")
    print("  workflow.add_edge('analyzer',  'supervisor')   ← 回边")
    print("  workflow.add_edge('verifier',  'supervisor')   ← 回边")
    print("比你多一个 verifier 节点，其余完全同构。")
    print()
    print("跑完想三个问题：")
    print("  1. 节点之间靠什么传递数据？为什么不直接 return 整个 state？")
    print("  2. 如果删掉 recursion_limit，图会不会死循环？")
    print("  3. 为什么 retriever 之后要回 supervisor，而不是直接连 analyzer？")


if __name__ == "__main__":
    asyncio.run(main())
