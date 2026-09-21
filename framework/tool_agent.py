"""
Tool Agent — 工具化检索（本项目唯一的真 agent 环节）

与其它节点的根本区别（面试要讲清楚）：
supervisor / retriever / analyzer / verifier 全是"代码决定下一步走哪"的 workflow 节点，
LLM 在其中只负责生成文字，输出不改变控制流。本模块是唯一的例外——
把 search_library 工具交给模型，"这一轮要不要查资料、查什么"由模型的输出决定，
代码只负责执行模型的选择并把结果回填。判据来自 Anthropic《Building Effective Agents》：
Workflow = 路径由代码预定义；Agent = LLM 运行时决定流程与工具调用。

流程：
1. 首轮流式生成：模型边生成边决定——直接回答（正文透传给用户），或请求调用 search_library
2. 若请求工具：执行检索 → 把原文作为 ToolMessage 回填 → 再流式生成最终回答
3. 检索到的文档与图谱标记写回 state，供引用出处（citations 事件）与 verifier 复用

开关：settings.tool_retrieval_enabled（默认 False）。
关闭时本模块完全不参与，链路仍走 supervisor 规则路由 → retriever → analyzer。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.llm import get_chat_llm
from src.core.state import AgentState
from framework.retrieval_agent import retrieve_documents


SEARCH_TOOL_NAME = "search_library"

# 工具描述是这套设计的真正载体——模型对工具的理解完全来自这段文字。
# 所以"什么时候用 / 什么时候不用"必须写具体、给正反例、给判据，
# 只写"检索资料库"这种功能描述，模型会把它当成万能工具（连"你好"都去查一次）。
SEARCH_TOOL_DESCRIPTION = """查询这位人物相关的原著与背景资料库，返回可引用的原文片段。

应该用它：
- 用户询问他的思想、主张、观点、著作内容、生平经历、他说过的话
- 回答需要真实凭据：原文原话、具体事实、数字、年份、人名、书名、地名
- 用户在追问某个概念（"你怎么看""你为什么这么讲"），需要落到本人原话上才站得住
- 用户提到"你写过""你书里说过"这类需要核对原文的说法

不该用它：
- 寒暄、问候、道谢、告别、安慰、情绪回应（"你好""谢谢""我最近很难过"）
- 纯粹的语气回应、附和、玩笑、复述（"哈哈""原来如此""是吗""有道理"）
- 只是想继续聊天、拉家常，或是问你现在的心境、感受
- 上一轮刚查过资料，这一轮只是接着刚才的话头往下说

判断标准：这句话要答得可信，是不是非有"凭据"不可？是，就查；只是"说句话"，就别查。
查到资料就用里面的原文与说法支撑回答；没查到就直说想不起来了，不要编造出处。"""


class SearchQuery(BaseModel):
    """search_library 的入参"""

    query: str = Field(
        description="要查的内容，用一句自然语言描述，包含人物名、概念名或关键词"
    )


def build_search_tool(
    sink: dict,
    history: list | None = None,
    zone: str = "education",
) -> StructuredTool:
    """构造 search_library 工具。

    工具是一层闭包，需要感知本轮请求的上下文（对话历史用于指代消解、分区决定
    资料的注入话术），所以按请求现场构造，不做全局单例。

    Args:
        sink: 结果收集容器，检索到的文档写进 sink["docs"]、sink["graph_used"]。
            不把 Document 对象塞进 ToolMessage（那会变成一大坨字符串再被模型复述），
            而是留在进程内交回节点，模型只看渲染后的文本。
    """

    async def _search(query: str) -> str:
        docs, graph_used, error = await retrieve_documents(
            query,
            history=history or [],
            light_retrieval=False,
            skip_retrieval=False,
        )
        sink["docs"] = list(sink.get("docs") or []) + docs
        sink["graph_used"] = bool(sink.get("graph_used")) or graph_used

        if error:
            print(f"[Tool Agent] search_library 检索出错: {error}")
            return "资料库暂时查不到内容。请凭你自己的记忆与判断回答，不要编造具体出处。"
        if not docs:
            print("[Tool Agent] search_library 无命中")
            return "资料库里没有查到相关内容。请凭你自己的记忆与判断回答，不要编造具体出处。"

        print(f"[Tool Agent] search_library 命中 {len(docs)} 条")
        return render_docs_for_tool(docs, zone)

    return StructuredTool.from_function(
        coroutine=_search,
        name=SEARCH_TOOL_NAME,
        description=SEARCH_TOOL_DESCRIPTION,
        args_schema=SearchQuery,
    )


def render_docs_for_tool(docs: list, zone: str = "education") -> str:
    """把检索结果渲染成回填给模型的 ToolMessage 文本。

    教育区必须带上【引用标注】规则：原本这条规则挂在"参考资料"系统提示里，
    工具化之后资料是作为工具返回值进来的，规则不跟着走模型就不会标 [n]，
    前端的引用出处折叠区（判据是正文含 [n]）会整块消失。
    """
    from framework.analysis_agent import _build_context, _build_light_context

    if zone == "entertainment":
        # 娱乐区资料 = 角色自己的记忆，不带任何检索痕迹
        body = _build_light_context(docs)
        return f"【你记得的事】\n{body}" if body else ""

    body = _build_context(docs)
    return f"""以下是检索到的原文片段，编号供正文引用标注使用。

【引用标注】回答中的关键论断、概念解释、事实与数字，请在句末标注所依据资料的编号（如 [2]）；
一个论断依据多条资料时可并列（如 [1][3]）；延伸推理部分不标注。编号与下方资料清单一一对应。

{body}"""


def _collect_tool_calls(chunk: Any, pending: dict) -> None:
    """把流式 chunk 里的 tool_call 增量拼进 pending（按 index 归并）。

    OpenAI 兼容接口的工具调用是增量下发的：函数名、参数 JSON 会分多个 chunk
    陆续到达，必须按 index 累积拼接，不能只看单个 chunk。
    """
    for tc in getattr(chunk, "tool_call_chunks", None) or []:
        idx = tc.get("index")
        if idx is None:
            idx = 0
        slot = pending.setdefault(idx, {"name": "", "args": "", "id": ""})
        if tc.get("name"):
            slot["name"] += tc["name"]
        if tc.get("args"):
            slot["args"] += tc["args"]
        # id 用覆盖而不是拼接：多数上游在首个 chunk 给全量 id，重复追加会拼坏
        if tc.get("id"):
            slot["id"] = tc["id"]


def _finalize_tool_calls(pending: dict) -> list[dict]:
    """把累积结果整理成 langchain 的 tool_calls 格式"""
    calls = []
    for idx in sorted(pending):
        slot = pending[idx]
        name = (slot.get("name") or "").strip()
        if not name:
            continue
        raw_args = (slot.get("args") or "").strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception:
            print(f"[Tool Agent] 工具参数 JSON 解析失败，按空参处理: {raw_args[:80]}")
            args = {}
        calls.append({
            "name": name,
            "args": args,
            "id": slot.get("id") or f"call_{idx}",
            "type": "tool_call",
        })
    return calls


async def _invoke_tool(tool: StructuredTool, call: dict) -> str:
    """执行模型请求的工具调用；任何异常都降级成一句人话回填，不让链路断掉"""
    if call["name"] != SEARCH_TOOL_NAME:
        return f"没有名为 {call['name']} 的工具，请直接回答。"
    try:
        return await tool.ainvoke(call["args"])
    except Exception as e:
        print(f"[Tool Agent] 工具执行失败: {e}")
        return "检索失败，暂时查不到资料。请凭你自己的记忆与判断回答，不要编造具体出处。"


async def tool_agent(state: AgentState) -> dict[str, Any]:
    """
    工具化检索节点：模型自主决定是否检索，代码只执行并回填。

    图里由 supervisor 按 settings.tool_retrieval_enabled 分流到这里，
    结束后回到 supervisor 收尾（supervisor 见 analysis 已就绪即定稿）。
    """
    from framework.supervisor import build_direct_messages

    query = state["query"]
    history = state.get("history", [])
    character_role_prompt = state.get("character_role_prompt", "")
    zone = state.get("zone", "education")
    stream_callback = state.get("stream_callback")
    session_id = state.get("session_id")

    print(f"\n[Tool Agent] 工具化检索轮开始: {query}")

    sink: dict = {"docs": [], "graph_used": False}
    tool = build_search_tool(sink, history=history, zone=zone)

    # with_fallback=False：必须拿裸 ChatOpenAI，with_fallbacks 返回的
    # RunnableWithFallbacks 不提供 bind_tools（见 src/core/llm.py 注释）
    _sampling = state.get("sampling") or {}
    _llm_kwargs: dict[str, Any] = {
        "with_fallback": False,
        "temperature": _sampling.get(
            "temperature", 0.8 if character_role_prompt else 0.6
        ),
        "max_tokens": 300 if zone == "entertainment" else 1600,
    }
    for _k in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
        if _sampling.get(_k) is not None:
            _llm_kwargs[_k] = _sampling[_k]

    try:
        llm = get_chat_llm(**_llm_kwargs)
        llm_with_tools = llm.bind_tools([tool])

        messages = build_direct_messages(
            query,
            history=history,
            character_role_prompt=character_role_prompt,
            session_id=session_id,
            context=None,  # 资料由工具按需带进来，不再预先注入
            zone=zone,
            post_history_directive=state.get("post_history_directive"),
            user_memory=state.get("user_memory"),
        )

        rounds = max(1, int(getattr(settings, "tool_max_rounds", 2)))
        answer_parts: list[str] = []

        # 多跑一轮不带工具的收尾：轮次用满时资料已经拿到，必须逼出答案而不是继续要资料
        for round_idx in range(rounds + 1):
            use_tools = round_idx < rounds
            target = llm_with_tools if use_tools else llm
            pending: dict = {}
            texts: list[str] = []

            async for chunk in target.astream(messages):
                _collect_tool_calls(chunk, pending)
                text = getattr(chunk, "content", None)
                if text:
                    texts.append(text)
                    if stream_callback:
                        await stream_callback(text)

            calls = _finalize_tool_calls(pending)
            answer_parts.append("".join(texts))

            if not calls:
                if not any(p.strip() for p in answer_parts):
                    # 整轮既没正文也没工具请求：上游空壳，交给 supervisor 走直答兜底
                    print("[Tool Agent] 本轮无任何输出，交给 supervisor 兜底")
                break

            print(f"[Tool Agent] 第 {round_idx + 1} 轮：模型请求检索 {len(calls)} 次")
            messages.append(AIMessage(content="".join(texts), tool_calls=calls))
            for call in calls:
                result = await _invoke_tool(tool, call)
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

        answer = "".join(answer_parts).strip()
        docs = sink.get("docs") or []
        print(f"[Tool Agent] 完成：检索 {len(docs)} 条，回答 {len(answer)} 字")

        return {
            "analysis": answer,
            "retrieved_docs": docs,
            "graph_used": bool(sink.get("graph_used")),
            "route_history": state.get("route_history", []) + ["tool_agent"],
        }

    except Exception as e:
        # 工具化失败不能拖垮回答：抛空 analysis，supervisor 会降级为直答
        print(f"[Tool Agent] 工具化检索失败（降级直答）: {e}")
        return {
            "analysis": "",
            "retrieved_docs": [],
            "route_history": state.get("route_history", []) + ["tool_agent"],
            "error": str(e),
        }
