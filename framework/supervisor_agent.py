"""
Supervisor Agent —— 一对一对话的唯一执行体。

架构定性（2026-09-30，取代 tool_agent 两段式）：
    supervisor 不再只是"纯规则路由 + 把检索外包给 tool_agent"，
    而是本项目唯一的真 agent：**检索工具由 supervisor 自己持有**，
    "这轮要不要查、查多深"由模型在生成过程中决定，代码只负责执行并回填原文。
    判据来自 Anthropic《Building Effective Agents》——
    Workflow = 路径由代码预定义；Agent = LLM 运行时决定流程与工具调用。

分档不再按分区走两条路，而是由工具的 strong 参数承担：
    strong=True  → 强检索：改写 + 向量/BM25 混合召回 + Cross-Encoder 精排 +
                   知识图谱增强，15 条，带来源与章节，正文标 [n] 供前端渲染出处。
    strong=False → 轻量检索：只做向量+BM25 召回 top-3，作为"模糊印象"，
                   不带来源标注——说出"根据《xxx》第三章"正是娱乐区要压掉的 AI 味。
    模型按请求性质自己选；工具描述里带本区的倾向提示（教育区偏 true，娱乐区偏 false），
    所以合并路由不会把娱乐区拖进全管线。

成本边界（面试要能报数）：
    模型直答 = 1 次 LLM 往返；模型请求检索 = 2 次（1 次决策带生成 + 1 次依资料作答）。
    轮数上限 = tool_max_rounds + 1，最后一轮不带工具，逼出答案而不是继续要资料。

进度事件：
    on_stage 收到的名字沿用旧节点标识（supervisor / retrieval_agent），
    前端 AGENT_STEP_LABELS（frontend/assets/js/chat.js）已有对应中文文案，前端零改动。

文件内容：
    本文件同时容纳执行体与它调用的检索实现（retrieve_documents，原
    framework/retrieval_agent.py，2026-10-02 并入）——检索只被 search_library
    一个调用方使用，拆文件反而让"工具 → 实现"这条线断在两个文件里。
"""

from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.llm import LLMEmptyResponseError, get_chat_llm
from src.core.logger import stage_mark
from src.core.state import AgentState
from src.retrieval.knowledge_graph import graph_exists


SEARCH_TOOL_NAME = "search_library"

StageCallback = Optional[Callable[[str], Awaitable[None]]]

# 工具描述是这套设计的真正载体——模型对工具的理解完全来自这段文字。
# 所以"什么时候用 / 什么时候不用 / strong 怎么填"必须写具体、给正反例、给判据，
# 只写"检索资料库"这种功能描述，模型会把它当成万能工具（连"你好"都去查一次）。
SEARCH_TOOL_DESCRIPTION = """查询这位人物相关的原著与背景资料库，返回可引用的原文片段。

strong 怎么填：
- true（强检索）：走完整检索管线，返回 15 条带来源与章节的原文，回答需要在句末标 [n]。
  用在：得落到原话、具体事实、数字、年份、书名、人名上才站得住的问题。
- false（轻量检索）：只做一轮向量+关键词召回 top-3，作为模糊印象返回，不标来源。
  用在：只是聊到某个话题、想要一点背景底色，不需要逐句可溯源。

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

# 分区倾向：合并路由之后，靠这段提示把教育区的"要凭据"与娱乐区的"要像真人"分开，
# 否则模型可能给娱乐区开全管线（慢）或给教育区只给模糊印象（不可溯源）。
_ZONE_HINTS = {
    "education": """

【本区倾向】当前角色属教育成长区，回答讲究可溯源：多数实质提问用 strong=true；
只有纯寒暄、闲聊、随口一问才用 strong=false，或者干脆不调本工具。""",
    "entertainment": """

【本区倾向】当前角色属娱乐区，讲究像真人一样自然短答：默认 strong=false
（够用的背景印象即可）；只有用户明确追问具体出处、原话、年份、数字这些硬事实时才用 true。""",
}


# ============================================================
# 检索实现（原 framework/retrieval_agent.py，2026-10-02 并入）
# ============================================================
# 与 HTTP 层（/persona/eval_query）共用 src.retrieval.advanced_search.advanced_retrieval，
# 避免两套检索逻辑不一致。
#
# 修复记录（优化十八，搬迁时保留）：
# - 旧实现 collection.get() 全量拉取文档，触发 ChromaDB 内部 SQLite
#   "too many SQL variables" 错误（SQLITE_MAX_VARIABLE_NUMBER），检索链路一直失败；
# - 旧 vector_search 会对全部文档重新 embedding（本地模型推理，几万条极慢）；
# - advanced_search 内部已分页拉取文档，并使用 ChromaDB 原生向量查询（入库时已存向量）。

# bge-small-zh-v1.5 的窗口是 512 token（中文约 1 token/字 ≈ 507 字符）。
# 历史感知检索把最近两轮对话拼进查询，长历史 + 长问题会把"当前用户问题"
# 整个挤出窗口——拼接文本从右侧截断，真实问题根本进不了向量，检索完全失焦。
# 对策：默认装配不变（绝大多数查询 < 460 字符，零行为变化）；
# 超预算时逐级压缩历史长度——历史可以截，当前问题不能丢。
_QUERY_CHAR_BUDGET = 460  # 512 token 留出分词开销安全余量


def build_history_aware_query(query: str, recent: list[tuple[str, str]]) -> str:
    """拼接历史感知检索查询；超预算时逐级压缩历史，保证当前问题完整进窗口"""
    def _assemble(ans_cap: int, q_cap: Optional[int] = None) -> str:
        ctx_lines = []
        for q, a in recent:
            if q:
                ctx_lines.append(f"用户：{q if q_cap is None else q[:q_cap]}")
            if a:
                ctx_lines.append(f"助手：{a[:ans_cap]}")
        return ("以下是最近的对话（用于理解指代和上下文）：\n"
                + "\n".join(ctx_lines) + "\n\n当前用户问题：\n" + query)

    question = _assemble(120)
    if len(question) > _QUERY_CHAR_BUDGET:
        for ans_cap, q_cap in ((80, 80), (50, 50), (30, 30), (0, 0)):
            question = _assemble(ans_cap, q_cap)
            if len(question) <= _QUERY_CHAR_BUDGET:
                break
        print(f"[Retrieval] 历史过长，已压缩至 {len(question)} 字符（保当前问题）")
    return question


async def retrieve_documents(
    query: str,
    history: list[tuple[str, str]] | None = None,
    light_retrieval: bool = False,
    skip_retrieval: bool = False,
) -> tuple[list, bool, str]:
    """执行高级混合检索，返回 (检索到的文档, 是否并入了知识图谱证据, 错误信息)。

    由本文件的 search_library 工具按需触发，检索力度由工具入参 strong 映射到
    本函数的 light_retrieval：strong=True → 全管线；strong=False → 轻量 top-3。

    异常一律在内部消化（返回空结果 + 错误串），不向上抛：
    检索失败不该炸掉整条回答链路，调用方按"没查到"继续走即可。
    """
    history = history or []

    # ---- 0. 零检索开关 ----
    # 实测：轻量检索（top_k=3，无改写/无重排/无图谱）中位仅 23ms，
    # 对 8-15s 的回答完全可忽略，故默认保留检索以贴合角色背景资料。
    # 只有把开关置 False（追求极致首字、且角色卡足够完备）时才走这条零检索路径。
    if skip_retrieval:
        print("[Retrieval] 零检索（已关闭轻量检索开关），跳过向量库与重排")
        return [], False, ""

    try:
        # ---- 1. 获取场景对应的 collection（与 HTTP 层一致）----
        from framework.runtime import get_scene_config, get_chroma_client
        config = get_scene_config()
        collection_name = getattr(config, 'chroma_collection', 'persona_jung') if config else 'persona_jung'

        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        count = collection.count()
        if count == 0:
            print("[Retrieval] 文档库为空，请先上传文档")
            return [], False, ""

        # ---- 2. 高级检索（Multi-Query + HyDE + BM25 + 向量 + 重排序）----
        from src.core.llm import get_chat_llm
        from src.retrieval.advanced_search import advanced_retrieval

        # 初始化 LLM（用于生成查询变体和 HyDE 文档；已合并为一次调用，256 足够）
        retrieval_llm = get_chat_llm(
            temperature=0.3,
            max_tokens=256,
        )

        top_k = 15  # 与 HTTP 层 /persona/eval_query 保持一致

        # 轻量检索（工具 strong=false）：跳过 Multi-Query+HyDE 改写、
        # 跳过 Cross-Encoder 重排、跳过知识图谱，仅用原始问题做向量+BM25 召回 top-3
        # 作为软背景（不引用），换取更快首字与更"像人"的回答。
        if light_retrieval:
            top_k = 3
            print(f"[Retrieval] 轻量召回（strong=false）：跳过改写/重排/图谱，top_k={top_k}")

        # ---- 历史感知检索 ----
        # 场景开启 history_aware_retrieval 时（如名人对话），将最近几轮对话拼入检索 query，
        # 解决指代性问题（"那个梦"、"这跟它有什么关系"）检索不到前文上下文的问题。
        question = query
        if config and getattr(config, 'history_aware_retrieval', False) and history:
            question = build_history_aware_query(query, history[-2:])
            print(f"[Retrieval] 历史感知检索（拼接最近 {min(len(history), 2)} 轮对话，{len(question)} 字符）")

        # 短问题跳过 Multi-Query + HyDE 改写（省 1 次串行 LLM 往返，降低首字延迟）。
        # 原始查询 + BM25 已能覆盖短问题的召回；长/复杂问题仍走完整改写以提升召回。
        # 另有总开关 rewrite_enabled（默认 False）：实测改写延迟 3.5s 超过 2s 等待预算，
        # 新问题必然白等 2s 后被放弃，故默认关闭（忠实度基线 8.8 即在改写无效时测得）。
        use_rewrite = (
            settings.rewrite_enabled
            and len(query.strip()) > settings.skip_rewrite_max_chars
            and not light_retrieval
        )
        if not use_rewrite:
            print(f"[Retrieval] 跳过检索改写（{'总开关关闭' if not settings.rewrite_enabled else '短问题' if not light_retrieval else '轻量召回'}），直接向量+BM25")

        retrieved_docs, _contexts = await advanced_retrieval(
            question=question,
            collection=collection,
            llm=retrieval_llm,
            top_k=top_k,
            use_multi_query=use_rewrite,
            use_hyde=use_rewrite,
            use_rerank=not light_retrieval,
            num_variants=2,
        )
        print(f"[Retrieval] 高级检索召回 {len(retrieved_docs)} 条")

        # ---- 3. 知识图谱 RAG 增强（GraphRAG，按需调用）----
        # 默认只用文本检索；只有当文本检索质量不达标（召回不足 / top 相关性弱 /
        # 关键实体未被文本覆盖）时才启动图谱增强，避免每轮都多跑一次子图扩展。
        # 图谱未构建或总开关关闭 → 静默降级，不影响原混合检索。
        graph_used = False
        if settings.graph_rag_enabled and graph_exists(collection.name) and not light_retrieval:
            try:
                from src.retrieval.knowledge_graph import (
                    retrieve_graph_context,
                    should_trigger_graph_retrieval,
                )
                from src.retrieval.advanced_search import _get_doc_id

                trigger, reason = should_trigger_graph_retrieval(
                    question, retrieved_docs, collection.name,
                )
                if trigger:
                    graph_used = True
                    g_ctx, g_docs = await retrieve_graph_context(
                        question, collection, llm=retrieval_llm,
                    )
                    if g_docs:
                        existing = {_get_doc_id(d.page_content) for d in retrieved_docs}
                        added = 0
                        for d in g_docs:
                            did = _get_doc_id(d.page_content)
                            if did not in existing:
                                retrieved_docs.append(d)
                                existing.add(did)
                                added += 1
                        if added:
                            print(f"[Retrieval] 触发知识图谱增强（{reason}），并入 {added} 条证据")
                    else:
                        print(f"[Retrieval] 图谱已触发（{reason}）但无命中")
                else:
                    print(f"[Retrieval] 文本检索质量达标，跳过知识图谱（{reason}）")
            except Exception as ge:
                print(f"[Retrieval] 知识图谱检索失败，跳过: {ge}")

        await _log_retrieved(retrieved_docs)
        return retrieved_docs, graph_used, ""

    except Exception as e:
        print(f"[Retrieval] 检索过程中出错: {e}")
        return [], False, str(e)


async def _log_retrieved(docs: list) -> None:
    """打印检索结果摘要（调试用，检索力度的分档由调用方的 strong 参数决定）"""
    for i, doc in enumerate(docs):
        score = doc.metadata.get("rrf_score", 0.0)
        heading = doc.metadata.get("heading", "未知章节")
        source = doc.metadata.get("source", "未知来源")
        stype = doc.metadata.get("source_type", "unknown")
        print(f"  [{i+1}] rrf={score} | [{stype}] {source} | {heading}")


class SearchQuery(BaseModel):
    """search_library 的入参"""

    query: str = Field(
        description="要查的内容，用一句自然语言描述，包含人物名、概念名或关键词"
    )
    strong: bool = Field(
        default=True,
        description="检索力度。true=强检索（全管线 15 条、带来源，正文需标 [n]）；"
        "false=轻量检索（top-3 模糊印象、不标来源）。判断标准见工具说明。",
    )


def build_search_tool(
    sink: dict,
    history: list | None = None,
    zone: str = "education",
    on_retrieval: StageCallback = None,
) -> StructuredTool:
    """构造 search_library 工具。

    工具是一层闭包，需要感知本轮请求的上下文（对话历史用于指代消解、
    分区决定描述里的倾向提示），所以按请求现场构造，不做全局单例。

    Args:
        sink: 结果收集容器，检索到的文档写进 sink["docs"]、sink["graph_used"]。
            不把 Document 对象塞进 ToolMessage（那会变成一大坨字符串再被模型复述），
            而是留在进程内交回调用方，模型只看渲染后的文本。
        on_retrieval: 进度回调——检索真正要跑之前触发一次，前端据此显示"翻查原著"。
    """

    description = SEARCH_TOOL_DESCRIPTION + _ZONE_HINTS.get(zone, "")

    async def _search(query: str, strong: bool = True) -> str:
        if on_retrieval is not None:
            await on_retrieval("retrieval_agent")

        docs, graph_used, error = await retrieve_documents(
            query,
            history=history or [],
            light_retrieval=not strong,
            skip_retrieval=False,
        )
        sink["docs"] = list(sink.get("docs") or []) + docs
        sink["graph_used"] = bool(sink.get("graph_used")) or graph_used

        if error:
            print(f"[Supervisor Agent] search_library 检索出错: {error}")
            return "资料库暂时查不到内容。请凭你自己的记忆与判断回答，不要编造具体出处。"
        if not docs:
            print("[Supervisor Agent] search_library 无命中")
            return "资料库里没有查到相关内容。请凭你自己的记忆与判断回答，不要编造具体出处。"

        print(f"[Supervisor Agent] search_library 命中 {len(docs)} 条（strong={strong}）")
        return render_docs_for_tool(docs, strong=strong)

    return StructuredTool.from_function(
        coroutine=_search,
        name=SEARCH_TOOL_NAME,
        description=description,
        args_schema=SearchQuery,
    )


def render_docs_for_tool(docs: list, strong: bool = True) -> str:
    """把检索结果渲染成回填给模型的 ToolMessage 文本。

    话术跟着**检索力度**走，而不是跟着分区走——力度决定"这批资料有多可信"：
    - 强检索：资料是"参考资料"，必须带【引用标注】规则。这条规则原本挂在
      "参考资料"系统提示里，工具化之后资料是作为工具返回值进来的，
      规则不跟着走模型就不会标 [n]，前端的引用出处折叠区（判据是正文含 [n]）会整块消失。
    - 轻量检索：资料只是"模糊印象"，不能出现来源痕迹——说出"根据《xxx》"
      正是娱乐区要压掉的 AI 味。
    """
    from framework.analysis_agent import _build_context, _build_light_context

    if not strong:
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
            print(f"[Supervisor Agent] 工具参数 JSON 解析失败，按空参处理: {raw_args[:80]}")
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
        print(f"[Supervisor Agent] 工具执行失败: {e}")
        return "检索失败，暂时查不到资料。请凭你自己的记忆与判断回答，不要编造具体出处。"


# 空轮重试次数：与 src/core/llm.py 的空响应容错同一套思路。上游存在
# "HTTP 200 但什么都不吐"的空壳（本项目实测发生率随时段波动），
# 不重试这一轮就会交回空 analysis，把用户拖进下一段降级链路。
_EMPTY_ROUND_ATTEMPTS = 2


async def _stream_round(target, messages: list, stream_callback) -> tuple[str, list[dict]]:
    """跑一轮流式生成，返回 (正文, 模型请求的工具调用)。

    这里不能直接复用 src/core/llm.py 的 astream_nonempty：它把"一个 token 都没产出"
    一律当失败重试，而工具轮天然可能整轮没有正文（模型直接吐一个 tool_call），
    会被误判成空壳，同一轮反复重试到把工具调用丢掉。
    判据因此改成"正文和工具调用都没有"才算空轮；且只在没往用户推过任何 token 时
    才重试——推过就说明上游是活的，重试会让同一段话出现两遍。
    """
    for attempt in range(_EMPTY_ROUND_ATTEMPTS + 1):
        texts: list[str] = []
        pending: dict = {}
        try:
            async for chunk in target.astream(messages):
                _collect_tool_calls(chunk, pending)
                text = getattr(chunk, "content", None)
                if text:
                    texts.append(text)
                    if stream_callback:
                        await stream_callback(text)
        except Exception:
            if texts:
                raise  # 已经推给用户了，重来会重复
            if attempt >= _EMPTY_ROUND_ATTEMPTS:
                raise
            print(f"[Supervisor Agent] 本轮流式异常且无输出（第 {attempt + 1} 次），重试…")
            continue

        calls = _finalize_tool_calls(pending)
        if texts or calls:
            return "".join(texts), calls
        if attempt >= _EMPTY_ROUND_ATTEMPTS:
            return "", []
        print(f"[Supervisor Agent] 上游返回空轮（第 {attempt + 1} 次），重试…")
    return "", []


async def run_supervisor_agent(
    state: AgentState,
    on_stage: StageCallback = None,
) -> dict[str, Any]:
    """一对一对话的完整编排：supervisor 自己决定查不查、查多深，并生成回答。

    Args:
        state: 与图时代同构的输入 state（query / history / zone /
            character_role_prompt / stream_callback / sampling / ...）。
        on_stage: 可选的进度回调，参数是阶段名（supervisor / retrieval_agent）。
            前端据此显示"判断该怎么回应""翻查原著"。不传则静默。

    Returns:
        final_answer / analysis / retrieved_docs / graph_used / route_history

    Raises:
        LLMEmptyResponseError: 上游持续空响应。抛出时保证尚未向用户推送任何 token，
            routes 层会把它转成 type:'error' 事件。
    """
    from framework.runtime import _EMPTY_UPSTREAM_MSG, build_direct_messages

    query = state["query"]
    history = state.get("history", [])
    character_role_prompt = state.get("character_role_prompt", "")
    zone = state.get("zone", "education")
    stream_callback = state.get("stream_callback")
    session_id = state.get("session_id")
    route_history: list[str] = list(state.get("route_history") or [])

    # ---- 首字瀑布埋点：把原先单一的 first_token_ms 拆成四段 ----
    #   prompt_ms   = 进入编排 → 消息装配完成（本地，毫秒级）
    #   decide_ms   = 第 1 轮 LLM（带工具）耗时——模型在这里判断"要不要查、查多深"，通常无正文
    #   retrieve_ms = 工具实际执行（检索）的累计耗时（本地管线）
    #   prefill_ms  = 真正生成正文那一轮的 LLM 调用 → 首个 token 到达（上游 prefill + reasoning）
    # 四段之和 ≈ routes 层的 first_token_ms；若决策轮自己吐了正文，两段会重叠，
    # 那种情况下以 first_token_ms（从请求起点算的总首字）为准。
    _t_enter = time.perf_counter()
    _round_start = _t_enter            # 当前这一轮 LLM 调用的起点（prefill 的基准）
    _round_tokens: list[float] = []    # 本轮首个 token 的到达时刻
    _retrieve_ms = 0.0

    async def _timed_callback(token: str) -> None:
        """记下本轮首个 token 的到达时刻（供 prefill 计算），再原样转发给 routes 的流式回调。"""
        if not _round_tokens:
            _round_tokens.append(time.perf_counter())
        if stream_callback is not None:
            await stream_callback(token)

    async def _stage(name: str) -> None:
        if on_stage is not None:
            await on_stage(name)

    print(f"\n[Supervisor Agent] 收到 query: {query}")
    print(f"[Supervisor Agent] 分区: {zone} | 历史轮次: {len(history)}")

    await _stage("supervisor")

    sink: dict = {"docs": [], "graph_used": False}
    tool = build_search_tool(sink, history=history, zone=zone, on_retrieval=_stage)

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
        stage_mark("prompt_ms", (time.perf_counter() - _t_enter) * 1000)

        rounds = max(1, int(getattr(settings, "tool_max_rounds", 2)))
        answer_parts: list[str] = []
        used_tool = False

        # 多跑一轮不带工具的收尾：轮次用满时资料已经拿到，必须逼出答案而不是继续要资料
        for round_idx in range(rounds + 1):
            use_tools = round_idx < rounds
            target = llm_with_tools if use_tools else llm
            _round_start = time.perf_counter()
            _round_tokens.clear()
            text, calls = await _stream_round(target, messages, _timed_callback)
            answer_parts.append(text)

            if round_idx == 0 and calls:
                # 第一轮就拿到工具调用 = 这一轮是"决策轮"（模型在判断要不要查）
                stage_mark("decide_ms", (time.perf_counter() - _round_start) * 1000)

            if not calls:
                # 这一轮不再要资料 = 就是产出正文的那一轮，此刻才能定性 prefill
                if _round_tokens:
                    stage_mark("prefill_ms", (_round_tokens[0] - _round_start) * 1000)
                if not text.strip() and not answer_parts[:-1]:
                    print("[Supervisor Agent] 本轮重试后仍无任何输出，交给上层兜底")
                break

            used_tool = True
            print(f"[Supervisor Agent] 第 {round_idx + 1} 轮：模型请求检索 {len(calls)} 次")
            messages.append(AIMessage(content=text, tool_calls=calls))
            for call in calls:
                _t_tool = time.perf_counter()
                result = await _invoke_tool(tool, call)
                _retrieve_ms += (time.perf_counter() - _t_tool) * 1000
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        if _retrieve_ms:
            stage_mark("retrieve_ms", _retrieve_ms)

        answer = "".join(answer_parts).strip()
        docs = sink.get("docs") or []
        print(f"[Supervisor Agent] 完成：检索 {len(docs)} 条，回答 {len(answer)} 字")

        # ---- 兜底：模型空手但资料已查到，别让检索白花钱 ----
        # 用同一份装配逻辑再生成一次（资料带进去），而不是走另一个节点，
        # 保证一对一只有一个生成者、一种人设口径。
        if not answer and docs:
            from framework.analysis_agent import _build_context, _build_light_context
            from framework.runtime import _generate_direct_response

            context = (
                _build_light_context(docs)
                if zone == "entertainment"
                else _build_context(docs)
            )
            print("[Supervisor Agent] 模型空手但有资料，带资料重生成一次")
            _round_start = time.perf_counter()
            _round_tokens.clear()
            answer = await _generate_direct_response(
                query,
                history,
                character_role_prompt,
                _timed_callback,
                session_id=session_id,
                context=context,
                max_tokens=300 if zone == "entertainment" else None,
                zone=zone,
                post_history_directive=state.get("post_history_directive"),
                sampling=state.get("sampling"),
                user_memory=state.get("user_memory"),
            )
            answer = (answer or "").strip()
            if _round_tokens:
                stage_mark("prefill_ms", (_round_tokens[0] - _round_start) * 1000)

        if not answer:
            # 上游持续空响应：绝不回退"无资料直答"——那会丢掉本轮全部检索资料，
            # 给出看似有据、实则凭空的回答，比如实报错更糟。此刻尚未推过任何 token，
            # 直接抛是安全的。
            raise LLMEmptyResponseError(_EMPTY_UPSTREAM_MSG)

        route_history.append("supervisor")
        if used_tool:
            route_history.append("retrieval_agent")

        return {
            "final_answer": answer,
            "analysis": answer,
            "retrieved_docs": docs,
            "graph_used": bool(sink.get("graph_used")),
            "route_history": route_history,
        }

    except LLMEmptyResponseError:
        raise
    except Exception as e:
        # 工具化失败不能拖垮回答：sink 里的资料是这一轮已经花掉的检索成本，
        # 异常时不能丢——带着它抛出去，routes 层按错误处理，但资料可用于排查。
        print(f"[Supervisor Agent] 编排失败: {e}")
        route_history.append("supervisor")
        raise LLMEmptyResponseError(_EMPTY_UPSTREAM_MSG) from e
