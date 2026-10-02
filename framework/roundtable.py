"""
圆桌会议（争鸣）—— LangGraph 主持人拓扑

拓扑：主智能体反复决定"下一位谁发言"，子智能体只发言，**子智能体之间没有边**。

              ┌──────────────────────────────┐
              │                              │
    START → host ──(选定某人)──→ speaker ────┘
              │
              └─(收敛 / 配额耗尽 / 超预算)─→ END

- **host 是主智能体**：唯一决定下一位谁发言，并给出**该角色主动请缨的理由**。前端据此
  显示"某某主动请缨"，而不是"主持人点名"——理由由主持人以该角色口吻概括（见 HOST_DIRECTIVE
  的"必须指出靶子"约束）。用户自己当主持人时（picker="user"），host 在决策点 `interrupt()`
  挂起、等前端回填，**用户全权负责**，系统不代他点将。
- **speaker 是子智能体**：每位与会者都用一对一对话那套 supervisor 架构——自持检索工具、
  只检索自己的语料库、逐 token 流式。所有与会者**共用同一个 speaker 节点**，靠状态里的
  `pending_speaker` 区分身份；因此"子智能体之间没有边"是结构性成立的，不靠约定。
- **开场轮是本场唯一的排班**（零 LLM）：先按名单各陈述一次亮明立场，之后才进主持人决策
  的交锋轮。去掉它会让首轮缺少对撞的靶子。
- **收敛靠配额的数学，不靠提示词调参**：每人发言次数有上限，配额耗尽即自然散会；
  主持人只能提前收敛，不能延长。另有全场的 LLM 调用硬上限。

为什么这次值得上图（与一对一对话弃图的判据对照）：
  一对一那条链路每节点每请求最多进一次、可达路径仅 5 条且进入时即确定——路径是代码预定的，
  画成图只是把直筒流程拆成盒子，弃图是对的。而圆桌**真的在图上有环**：host 每轮都要被重新
  进入，走几轮由讨论内容决定；且用户当主持人时**真的需要跨请求挂起**（挂在哪里由运行时决定，
  不是代码写死的分支）。环 + 挂起，正是图的正当用途。

事件流（前端据此渲染）：
  start         {session_id, topic, picker, host, speakers:[{id,name,avatar,theme,quota}]}
  round         {round, total, phase: "opening"|"rebuttal"}
  choice        {character_id, name, by: "host"|"user", reason, candidates, note}
                by=host 时 reason 是该角色的**请缨理由**；by=user 时是用户点将。
  choice_request{round, total, candidates:[{id,name,speeches,quota}]}
                用户主持时推送，会议在此**挂起**，前端需 POST 回填下一位发言人。
  speaker_start {character_id, name, avatar, theme, speeches, quota, reason}
  token         {character_id, content}
  speaker_end   {character_id, content}
  converged     {round, reason: "host"|"quota"|"rounds"}
  budget_exhausted{llm_calls}
  end           {session_id, topic, rounds, speakers, transcript, converge_reason,
                 timings, prompt_chars, answer_chars, meeting_ms}
                —— 权威快照：续跑流不推 start，故 end 必须自带完整记录。
                尾部四个字段是给 monitor 落库的用量统计（段耗时 / 输入输出字符 / 整场耗时）。
  summary_start / summary {data:{clashes,positions,open,speeches,degraded}} / summary_error / summary_end
  error         {content}

  与旧实现的差别：`intent_start` / `intent` 已删除（不再逐人问意愿，改由主持人一次决策）；
  `contention` 被 `choice_request` 取代（用户主持不再靠内存 Event 等待，改为 checkpoint 挂起）。
  纪要仍在 `end` **之后**推送，属非阻塞旁路——纪要失败只是少一段纪要，会议结果已完整交付。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from operator import add
from pathlib import Path
from typing import Annotated, AsyncIterator, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

try:  # 导入位置在 langgraph 版本间搬过家
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    from langgraph.utils.config import get_stream_writer  # type: ignore

from src.core.config import settings
from src.core.llm import ainvoke_nonempty, astream_nonempty, get_chat_llm
from framework.runtime import get_chroma_client
from scenes.persona_chat.config import persona_chat_config
from scenes.persona_chat.prompt_builder import build_character_prompt
from src.retrieval.embedder import get_embedder


# 圆桌人数上限：人太多会摊薄每位的发言篇幅，削弱交锋感。
# 定 3 的依据（P(≥2 人同时有话要说) 随人数上升）：2 人偏静、4 人以上主持人每轮都要
# 在更多人里挑，主席决策的准确率下降，且单场总调用数上升。3 人是有交锋感又不臃肿的上限。
MAX_ROUNDTABLE_CHARS = 3

# 交锋轮数上限：钳制用户输入，避免超预算。
_MAX_ROUNDS = 3

# 发言间隔（秒）：每位发言结束后停顿，既给用户留出阅读上一段的时间，
# 也把 LLM 调用频率压到网关可持续速率（突发即 429）以内。
ROUNDTABLE_SPEAKER_PAUSE = 4.0

# ---- max_tokens 预算：这是推理模型，预算给少了正文会**静默为空** ----
# 实测（deepseek-v4-flash @ token.sensenova.cn，2026-09-21）：
#   该模型是推理模型，reasoning token 与正文**共用** max_tokens 预算。预算不足时
#   上游把预算全烧在 reasoning 上，正文返回空字符串，而且**不抛异常、不打日志**。
#     意愿判断 max_tokens=96/300/700 → 全 reasoning，正文 ''，耗时 3.9/7.6/16.4s
#     意愿判断 max_tokens=1200    → 正文完整，2.3s
#   后果：判断永远解析失败 → 一律按"不发言"处理 → **圆桌再没人说话**，而表面上
#   一切正常（不报错、日志干净、只有角色集体沉默）。这类静默失效比报错难查得多。
# 对策两条：① 预算按 worst case 留（reasoning 实测 98~962 浮动）；② 判 JSON 的调用
# 改**流式**读，凑齐一对花括号就断开，上游还没想完我们已经拿到答案。
_SPEECH_MAX_TOKENS = 1600
_HOST_MAX_TOKENS = 1600
_SUMMARY_MAX_TOKENS = 1600

# ---- 主持人决策的上下文预算 ----
# 决策每轮一次，是自主发言制里调用次数最多的一环，上下文越小越省。
_HOST_CTX_TURNS = 8        # 看最近 8 条发言
_HOST_TURN_CHARS = 160     # 每条截断
_HOST_REASON_CHARS = 48    # 请缨理由长度上限

# ---- 纪要的上下文预算 ----
_SUMMARY_TURN_CHARS = 300
_SUMMARY_TOTAL_CHARS = 6000
_SUMMARY_ATTEMPTS = 2  # 首次 + 命中禁用词后的纠正重试


# ============================================================
# 圆桌辩论专用指令（叠加在角色人设之上，约束"会议"形态）
# ============================================================
ROUNDTABLE_DEBATE_DIRECTIVE = """
你正参加一场圆桌辩论（Roundtable）。多位历史人物齐聚，就同一议题各自陈词、相互交锋。

【你的任务】
1. 始终以第一人称、用你本人的语言习惯与思维方式发言；绝不跳出角色，绝不以"AI / 主持人 / 总结者"口吻说话。
2. 鲜明表达你对该议题的立场，从你自己的学说、经历、时代与价值取向出发；允许直接、但不失礼地反驳其他与会者。
3. 若下方提供了你的思想资料库，请优先援引你本人的原著 / 观点作为论据；资料未覆盖之处，用你视角的合理推演，但不要编造具体书名或伪造引文。
4. 若提供了其他与会者的观点，请针对其中的分歧点做出回应或追问，制造真实的思想交锋；不要泛泛而谈、不要和稀泥。
5. 控制篇幅（开场陈述 150-260 字；后续交锋 120-220 字），像现场辩论一样利落有力。
""".strip()


# ============================================================
# 主持人（主智能体）：只定"下一位谁发言"，不做裁判
# ============================================================
HOST_DIRECTIVE = """
你是这场圆桌会议的主持人（主席）。与会者都是历史人物，各有立场。你的职责只有一件事：
**决定下一位由谁发言**，并替他讲清此刻为什么要站出来说话。

【你不是裁判】
不许评价谁说得对、不许排名次、不许宣布谁更有道理、不许做总结陈词——那是越界。

【选人的唯一依据：谁此刻最有非说不可的话】
· 有人说了与他立场相冲突的话，他最需要站出来反驳——你要能指出对方**具体是哪一句**；
· 或者有人曲解了他的立场，他必须澄清；
· 立场已经讲清、只是想换个说法重复一遍的，不要选他。

【请缨理由的写法】
reason 用**他本人的第一人称口吻**写，20-40 字，说清他要反驳或追问谁的哪句话。
写成像是他自己站起来说的话，而不是你在介绍他。例：
  "适之方才说'多研究些问题'便可不必谈主义，我偏要说：不谈主义的改造，终究是纸上空谈。"

【配额与轮转】
每位与会者的剩余发言额度写在名单里，**额度为 0 的人不能选**。
尽量轮转，别连续两轮点同一个人（除非只剩他还有额度）。

【何时宣布收敛】
当没有任何人的立场被真正触碰到、再发言也只是重复时，就宣布收敛——此时各方立场已充分
展开，讨论自然结束。宁早收，不要靠加轮次凑数。

【输出】只输出一行 JSON，不要任何其他内容：
{"next_speaker_id": "<与会者 id>", "reason": "<20-40字，他本人的第一人称口吻>", "converged": false}
若判定收敛，输出：{"next_speaker_id": "", "reason": "", "converged": true}
""".strip()


# ============================================================
# 书记员（纪要）：只记账，不评分
# ============================================================
CLERK_DIRECTIVE = """
你是这场圆桌会议的**书记员**。你的唯一职责是把讨论如实归档，**不是评价**。

【必须做到】
1. 全程用中立、平实的第三人称陈述。
2. 绝不模仿任何与会者的语气——你不是他们中的任何一位。
3. 全部输出必须是**一个 JSON 对象**，不要任何解释文字、不要 markdown 代码块围栏。

【绝对禁止】以下这类判断一律不写：
· 谁更有说服力 / 谁更有道理 / 谁更站得住脚 / 谁略胜一筹 / 谁赢了；
· 给任何人打分、排名次、宣布胜负；
· "总的来说""综合来看""显然"这类收束性论断。

会议的结局不是由你宣布的。你写完就停笔。

【结构】
{
  "clashes":  [{"a": "甲", "b": "乙", "point": "两人在『某个具体分歧点』上正面撞上"}],
  "positions":[{"name": "甲", "stance": "他这一场的核心立场，一句话"}],
  "open":     ["看完讨论仍然悬而未决的问题，一句话"]
}

要求：clashes 只写**真实发生的**交锋（a / b 必须是会上真正互相回应过的两位），
没有交锋就留空数组；positions 每人一条；open 至多 3 条。
""".strip()


# 命中即视为"越界宣布胜负"，触发纠正重试（详见 build_roundtable_summary）。
# 只拦"宣布胜负"这一类词。「回避了问题」不在此列——它可以是描述交锋的合法事实
# （甲认为乙在回避），书记员如实记录不算越界；系统不该做的只是**下结论**。
_CLERK_BANNED = (
    "更有说服力", "更有道理", "更占理", "更站得住脚", "站不住脚",
    "略胜", "更胜", "胜出", "获胜", "胜者", "谁赢", "赢了",
)


# ============================================================
# 图状态：必须可序列化（要落 checkpoint）
# 不放 asyncio.Event / 角色对象 / Document —— 那些落不了盘，也跨不了请求
# ============================================================
class RoundtableState(TypedDict, total=False):
    """一场圆桌会议的全部可变状态。键名与前端无关，前端只认下面的事件流。"""

    session_id: str
    topic: str
    picker: str                     # "user"=用户当主持人；"agent"=主持人（LLM）自己决定
    max_rounds: int                 # 交锋轮上限
    max_calls: int                  # 全场 LLM 调用硬上限
    quota: int                      # 每人发言次数上限（含开场）

    roster: list[dict]              # 静态名单 [{id,name,avatar,theme,quota}]，只为渲染/取名
    ledger: dict[str, dict]         # {id: {name, speeches, quota}} —— 台账

    transcript: Annotated[list[dict], add]      # 发言记录，累加
    opening_done: Annotated[list[str], add]     # 已完成开场陈述的 id，累加

    round_index: int                # 已完成的交锋轮数
    pending_speaker: str            # host 选定、待 speaker 消费的角色 id
    pending_kind: str               # "opening" | "rebuttal"
    pending_round: int
    pending_reason: str             # 请缨理由

    converged: bool
    converge_reason: str            # "host" | "quota" | "rounds" | "budget"
    llm_calls: int
    ended: bool

    # ---- 用量/耗时统计（供 monitor 落库，随 checkpoint 一起持久化）----
    started_at: float               # 会议起点（epoch 秒）——整场耗时以它为准，跨请求续跑也不丢
    timings: dict                   # {host_ms, speaker_ms} 累积的 LLM 段耗时（毫秒）
    prompt_chars: int               # 全场 LLM 调用的输入字符累计（monitor 估 token 用）
    answer_chars: int               # 全场 LLM 输出的字符累计


def _messages_chars(messages) -> int:
    """统计一批消息的输入字符数（兼容 (role, content) 元组与 BaseMessage）。"""
    total = 0
    for m in messages or []:
        if isinstance(m, (tuple, list)) and len(m) >= 2:
            total += len(str(m[1] or ""))
        else:
            total += len(str(getattr(m, "content", "") or ""))
    return total


def _merge_timings(prev, **delta) -> dict:
    """在已有耗时上累加本次增量（节点级耗时是"多次调用求和"，不是取最大）。"""
    out = {k: float(v) for k, v in (prev or {}).items()}
    for k, v in delta.items():
        out[k] = round(out.get(k, 0.0) + float(v or 0.0), 1)
    return out


def _noop_write(*_args, **_kwargs) -> None:
    return None


def _writer():
    """取本次运行的流写入器。

    用 ainvoke 直接驱动图（如单测）时没有 custom 流，取不到就退化成空写，
    这样节点本身仍可被单独测试，不必为了测试造一条假流。
    """
    try:
        return get_stream_writer()
    except Exception:
        return _noop_write


def _name_of(state: RoundtableState, cid: str) -> str:
    for e in state.get("roster") or []:
        if e.get("id") == cid:
            return str(e.get("name") or cid)
    return cid


def _speeches_of(ledger: dict, cid: str) -> int:
    return int((ledger.get(cid) or {}).get("speeches", 0))


def _fallback_pick(ledger: dict, eligible: list[str]) -> str:
    """规则兜底：在仍有额度的人里选**说得最少**的那位（min 稳定 → 平票按名单顺序）。

    这不是裁判——它只决定下一个谁说话，不对任何内容做评价。模型指认了不合法的人
    （不在名单/额度已满）或用户点将无效时用它，保证会议不会因为一次坏输出空转。
    """
    return min(eligible, key=lambda cid: _speeches_of(ledger, cid))


# ============================================================
# 检索：每位发言者只检索自己的知识库
# ============================================================
async def _retrieve_for_speaker(collection_name: str, query: str, top_k: int = 4) -> str:
    """
    检索某发言者自己知识库中最相关的片段，作为其发言的参考资料。

    用项目统一 Embedder 把 query 向量化，再走 ChromaDB 的 query_embeddings，
    保证与入库时（upload 路径用同一 Embedder）的向量空间一致。
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        embedder = get_embedder()
        emb = embedder.embed_query(query)
        res = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        docs = (res.get("documents") or [[]])[0]
        if not docs:
            return ""
        seen: set[str] = set()
        parts: list[str] = []
        for d in docs:
            if d and d.strip() and d not in seen:
                seen.add(d)
                parts.append(d.strip())
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[Roundtable] 检索失败 col={collection_name}: {e}")
        return ""


def _format_debate_so_far(transcript: list[dict], exclude_id: str | None = None) -> str:
    """把已有发言格式化为『其他与会者观点』，供当前发言者参考（默认排除自己）。

    截断策略：最多保留最近约 1500 字（从最早的发言开始丢弃），
    既保证交锋轮输入的上下文长度稳定（避免上游 "message too long"），
    又让发言者聚焦最新一轮的分歧点。
    """
    if not transcript:
        return ""
    lines = []
    for turn in transcript:
        if exclude_id and turn.get("character_id") == exclude_id:
            continue
        lines.append(f"【{turn.get('name', '发言人')}】{turn.get('content', '')}")
    text = "\n\n".join(lines)
    while len(text) > 1500 and lines:
        lines.pop(0)
        text = "\n\n".join(lines)
    return text


# ============================================================
# 构造单轮发言的 prompt
# ============================================================
async def build_speaker_messages(
    *,
    character,
    topic: str,
    transcript: list[dict],
    round_index: int,
    total_rounds: int,
    is_opening: bool,
) -> list[tuple[str, str]]:
    """拼装某发言者本轮的 LangChain 消息列表（system / user）。检索异步，故为 async。"""
    config = persona_chat_config
    voice = getattr(config, "persona_voice_directive", "") or ""
    # 酒馆式装配：示例对话与世界书让发言者在交锋中也不丢自己的语气；
    # topic 用于世界书关键词命中（开场轮与交锋轮都用议题本身匹配）。
    role_prompt = build_character_prompt(character, query=topic) or getattr(
        config, "direct_response_system_prompt", ""
    )

    # 检索：开场轮用议题本身；交锋轮加入最近他人观点，使其检索更聚焦分歧
    retrieval_query = topic
    if not is_opening:
        others = _format_debate_so_far(transcript, exclude_id=character.id)
        if others:
            retrieval_query = f"{topic}\n\n其他与会者近期观点：\n{others[-700:]}"

    context = await _retrieve_for_speaker(character.chroma_collection, retrieval_query, top_k=4)
    others_view = _format_debate_so_far(transcript, exclude_id=character.id)

    system_parts = [role_prompt, ROUNDTABLE_DEBATE_DIRECTIVE]
    if voice:
        system_parts.append(voice)
    messages: list[tuple[str, str]] = [("system", "\n\n".join(system_parts))]

    if context:
        messages.append(("system",
            "【你的思想资料库（来自你的原著 / 传记，优先援引其中的观点与论据）】\n" + context))

    if others_view:
        messages.append(("system",
            "【其他与会者的发言（请针对其中的分歧点回应或反驳，制造思想交锋）】\n" + others_view))

    if is_opening:
        user_msg = (
            f"议题：{topic}\n\n"
            f"这是圆桌的开场陈述。请首先亮明你对这一议题的核心立场与理由，"
            f"用你本人才有的视角、学说与例证，让听众一眼认出“这就是某某”。"
        )
    else:
        user_msg = (
            f"议题：{topic}\n\n"
            f"这是第 {round_index}/{total_rounds} 轮交锋。请回应其他与会者的观点，"
            f"进一步阐述、修正或捍卫你的立场；若有相左之处，直接而得体地反驳。"
        )
    messages.append(("user", user_msg))
    return messages


# ============================================================
# 构造主持人的决策 prompt
# ============================================================
def build_host_messages(
    *,
    topic: str,
    roster: list[dict],
    ledger: dict,
    transcript: list[dict],
    round_index: int,
    total_rounds: int,
) -> list[tuple[str, str]]:
    """拼装主持人"下一位谁发言"的决策消息。

    刻意压到很小：只看最近若干条发言、每条截断。这一步每轮都要跑一次，
    是主持人制里最贵的固定开销。
    """
    lines = []
    for e in roster:
        cid = e.get("id")
        entry = ledger.get(cid) or {}
        left = max(0, int(entry.get("quota", 0)) - int(entry.get("speeches", 0)))
        lines.append(
            f"- {e.get('name')}（id={cid}）已发言 {int(entry.get('speeches', 0))} 次，剩余额度 {left}"
        )
    roster_text = "\n".join(lines)

    recent = transcript[-_HOST_CTX_TURNS:]
    tlines = []
    for turn in recent:
        body = (turn.get("content") or "").strip().replace("\n", " ")[:_HOST_TURN_CHARS]
        tlines.append(f"【{turn.get('name', '某位')}】{body}")
    transcript_text = "\n".join(tlines) if tlines else "（还没有人发言）"

    user_msg = (
        f"议题：{topic}\n\n"
        f"【与会者名单与发言额度】\n{roster_text}\n\n"
        f"【到目前为止的发言记录】\n{transcript_text}\n\n"
        f"现在进入第 {round_index}/{total_rounds} 轮交锋。"
        f"请决定下一位由谁发言（或宣布收敛），只输出一行 JSON。"
    )
    return [("system", HOST_DIRECTIVE), ("user", user_msg)]


# ============================================================
# 解析
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)

_HOST_ID_RE = re.compile(r"[\"']?next_speaker_id[\"']?\s*[:：]\s*[\"']?([A-Za-z0-9_\-.]+)")
_HOST_CONV_RE = re.compile(r"[\"']?converged[\"']?\s*[:：]\s*(true|false|是|否)", re.I)
_HOST_REASON_RE = re.compile(r"[\"']?reason[\"']?\s*[:：]\s*[\"']([^\"']*)[\"']")


def parse_host_decision(raw: str) -> dict:
    """解析主持人的决策返回。任何解析不出来的情况都返回空决策（由调用方规则兜底）。

    不采用"解析失败就不发言"的保守策略——主持人这一环失败会让整场会议停摆，
    必须能退回"_fallback_pick 按轮转挑一位"。
    """
    out = {"next_speaker_id": "", "reason": "", "converged": False}
    text = (raw or "").strip()
    if not text:
        return out

    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            out["next_speaker_id"] = str(obj.get("next_speaker_id") or "").strip()
            out["reason"] = str(obj.get("reason") or "").strip().replace("\n", " ")
            out["converged"] = bool(obj.get("converged"))
            return out

    # 兜底：模型把 JSON 包在解释里、格式有小出入时，用正则抠关键字段
    mi = _HOST_ID_RE.search(text)
    if mi:
        out["next_speaker_id"] = mi.group(1)
    mc = _HOST_CONV_RE.search(text)
    if mc:
        out["converged"] = mc.group(1).lower() in ("true", "是")
    mr = _HOST_REASON_RE.search(text)
    if mr:
        out["reason"] = mr.group(1).strip().replace("\n", " ")
    return out


def parse_summary(raw: str) -> dict | None:
    """解析书记员的 JSON 返回；拿不到合法结构就返回 None（由调用方降级）。"""
    text = (raw or "").strip()
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def summary_verdict_hit(obj: dict) -> str:
    """检查纪要是否越界宣布胜负，返回命中的词（空串 = 干净）。"""
    try:
        blob = json.dumps(obj, ensure_ascii=False)
    except Exception:
        return ""
    for word in _CLERK_BANNED:
        if word in blob:
            return word
    return ""


def normalize_summary(obj: dict) -> dict:
    """把书记员输出收敛成前端要的四块结构，并剔掉越界条目（不给它混进来的机会）。"""

    def clean_str(v) -> str:
        return str(v or "").strip().replace("\n", " ")

    def ok(v) -> bool:
        s = clean_str(v)
        return bool(s) and not any(w in s for w in _CLERK_BANNED)

    clashes = []
    for item in obj.get("clashes") or []:
        if not isinstance(item, dict):
            continue
        a = clean_str(item.get("a"))
        b = clean_str(item.get("b"))
        point = clean_str(item.get("point"))
        if a and b and ok(point):
            clashes.append({"a": a, "b": b, "point": point})

    positions = []
    for item in obj.get("positions") or []:
        if not isinstance(item, dict):
            continue
        name = clean_str(item.get("name"))
        stance = clean_str(item.get("stance"))
        if name and ok(stance):
            positions.append({"name": name, "stance": stance})

    open_items = [clean_str(x) for x in (obj.get("open") or [])]
    open_items = [x for x in open_items if ok(x)][:3]

    return {"clashes": clashes[:6], "positions": positions, "open": open_items}


def _fallback_positions(transcript: list[dict]) -> list[dict]:
    """零 LLM 的降级版「各方主张」：取每人第一次发言的开头。

    纪要生成失败时用它兜底——至少让用户看到每人立场，而不是一块空白。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for turn in transcript:
        cid = turn.get("character_id") or ""
        if not cid or cid in seen:
            continue
        seen.add(cid)
        stance = (turn.get("content") or "").strip().replace("\n", " ")[:60]
        out.append({"name": turn.get("name") or cid, "stance": stance})
    return out


# ============================================================
# 流式 LLM 调用：429/限流指数退避重试
# ============================================================
async def _stream_llm_with_retry(messages, label: str = "", max_retries: int = 2):
    """流式调用 LLM，逐 token 产出；对 429/限流错误在**未产出任何 token** 时退避重试。

    已在中途产出部分内容后失败，则放弃重试（避免重复内容），由调用方兜底。
    """
    attempts = 0
    while True:
        llm = get_chat_llm(temperature=0.85, max_tokens=_SPEECH_MAX_TOKENS)
        got = ""
        try:
            async for tok in astream_nonempty(llm, messages):
                if tok:
                    got += tok
                    yield tok
            return
        except Exception as e:
            attempts += 1
            err = str(e)
            is_quota = "429" in err or "rpm" in err.lower() or "quota" in err.lower()
            if is_quota and attempts <= max_retries and not got:
                await asyncio.sleep(2 * attempts)  # 2s / 4s 指数退避
                continue
            print(f"[Roundtable] 发言流式失败 {label}（attempts={attempts}）: {e}")
            return


# ============================================================
# 纪要（书记员）：只记账，不评分
# ============================================================
def _build_summary_messages(
    *, topic: str, transcript: list[dict], ledger: dict
) -> list[tuple[str, str]]:
    lines = []
    total = 0
    for turn in transcript:
        body = (turn.get("content") or "").strip().replace("\n", " ")[:_SUMMARY_TURN_CHARS]
        line = f"【{turn.get('name', '某位')}】{body}"
        if total + len(line) > _SUMMARY_TOTAL_CHARS:
            break
        lines.append(line)
        total += len(line)
    transcript_text = "\n\n".join(lines) if lines else "（没有发言记录）"

    counts = "、".join(
        f"{v.get('name', cid)} 发言 {int(v.get('speeches', 0))} 次（上限 {int(v.get('quota', 0))}）"
        for cid, v in ledger.items()
    )
    user_msg = (
        f"议题：{topic}\n\n"
        f"【完整发言记录】\n{transcript_text}\n\n"
        f"【发言次数（仅作事实记录，不参与任何评价）】{counts}\n\n"
        f"请按结构输出 JSON。记住：你是书记员，不是裁判。"
    )
    return [("system", CLERK_DIRECTIVE), ("user", user_msg)]


async def build_roundtable_summary(*, topic: str, transcript: list[dict], ledger: dict) -> dict:
    """生成纪要（分歧地图）。返回 {clashes, positions, open, degraded}。

    两段防线：① 提示词里禁止宣布胜负；② 输出若命中禁用词，纠正重试一次；
    仍越界则走零 LLM 的降级版（只列每人立场）+ 剔掉越界条目。
    这一层是必要的——"总结"是裁判最容易换件衣服回来的地方。
    """
    messages = _build_summary_messages(topic=topic, transcript=transcript, ledger=ledger)
    for _ in range(_SUMMARY_ATTEMPTS):
        try:
            llm = get_chat_llm(temperature=0.2, max_tokens=_SUMMARY_MAX_TOKENS)
            resp = await ainvoke_nonempty(llm, messages, attempts=1)
            obj = parse_summary(getattr(resp, "content", "") or "")
        except Exception as e:
            print(f"[Roundtable] 纪要生成失败: {e}")
            obj = None

        if obj:
            hit = summary_verdict_hit(obj)
            if not hit:
                data = normalize_summary(obj)
                data["degraded"] = False
                return data
            print(f"[Roundtable] 纪要越界（命中「{hit}」），纠正重试")
            messages.append(("assistant", json.dumps(obj, ensure_ascii=False)))
            messages.append((
                "user",
                f"你写了「{hit}」——这是在宣布胜负或做评价，越界了。"
                f"请重新输出 JSON：只如实记录各方立场、真实发生的交锋、悬而未决的问题，"
                f"不评价谁说得更好、不排名次、不宣布结果。",
            ))
        else:
            print("[Roundtable] 纪要未返回合法 JSON")

    data = {"clashes": [], "positions": _fallback_positions(transcript), "open": []}
    data["degraded"] = True
    return data


# ============================================================
# 主智能体节点：host —— 唯一决定"下一位谁发言"
# ============================================================
async def host_node(state: RoundtableState) -> dict:
    """主持人节点。两种模式：

    · 开场阶段：按名单轮转（零 LLM、零决策）——本场唯一的排班，用来保证首轮有靶子。
    · 交锋阶段：用户主持则 `interrupt()` 挂起等回填；否则一次 LLM 决策定人 + 写请缨理由。

    **不要在 interrupt 之前 writer() 推任何事件**：节点在恢复时会从头重跑，
    挂起前推过的事件会被重放一遍，前端就会收到重复的 round/choice。
    """
    write = _writer()
    roster = list(state.get("roster") or [])
    ledger = {k: dict(v) for k, v in (state.get("ledger") or {}).items()}
    transcript = list(state.get("transcript") or [])
    opened = set(state.get("opening_done") or [])
    max_rounds = int(state.get("max_rounds") or 1)
    quota = int(state.get("quota") or 2)
    max_calls = int(state.get("max_calls") or 24)
    llm_calls = int(state.get("llm_calls") or 0)
    round_index = int(state.get("round_index") or 0)
    picker = str(state.get("picker") or "user")
    topic = str(state.get("topic") or "")

    # 本次 host 调用（可能零 LLM：开场排班 / 收敛闸门）的用量与耗时增量
    _host_ms = 0.0
    _host_prompt = 0
    _host_out = 0

    def _tfields() -> dict:
        """把本次 host 调用的耗时/字符增量并入 state（无 LLM 的路径下返回空 dict）。"""
        if not (_host_ms or _host_prompt or _host_out or state.get("timings")):
            return {}
        return {
            "timings": _merge_timings(state.get("timings"), host_ms=_host_ms),
            "prompt_chars": int(state.get("prompt_chars") or 0) + _host_prompt,
            "answer_chars": int(state.get("answer_chars") or 0) + _host_out,
        }

    # ---- 1. 开场轮：固定排班，零 LLM ----
    waiting = [e.get("id") for e in roster if e.get("id") and e.get("id") not in opened]
    if waiting:
        if not opened:
            write({"type": "round", "round": 0, "total": max_rounds, "phase": "opening"})
        return {
            "pending_speaker": str(waiting[0]),
            "pending_kind": "opening",
            "pending_round": 0,
            "pending_reason": "",
            "ended": False,
        }

    # ---- 2. 交锋轮的收敛闸门：数学优先于模型 ----
    def _converge(reason: str) -> dict:
        write({"type": "converged", "round": round_index, "reason": reason})
        if reason == "budget":
            write({"type": "budget_exhausted", "llm_calls": llm_calls})
        out = {
            "converged": True,
            "converge_reason": reason,
            "ended": True,
            "pending_speaker": "",
            "pending_kind": "",
            "pending_reason": "",
        }
        out.update(_tfields())
        return out

    if round_index >= max_rounds:
        return _converge("rounds")

    eligible = [e.get("id") for e in roster if _speeches_of(ledger, e.get("id")) < quota]
    eligible = [cid for cid in eligible if cid]
    if not eligible:
        return _converge("quota")
    # 这一轮要花 1 次主持人决策 + 1 次发言，两次都要留在预算内
    if llm_calls + 2 > max_calls:
        return _converge("budget")

    next_round = round_index + 1
    candidates: list[dict] = []
    by = "host"
    reason = ""
    note = ""
    cid = ""

    # ---- 3. 定人 ----
    if picker == "user":
        # 用户全权负责：挂起等前端回填，系统不代他点将（也不设超时——挂起不占进程，
        # 用户完全可以过一会儿、甚至明天再回来点；checkpoint 已落盘）。
        candidates = [
            {
                "id": c,
                "name": _name_of(state, c),
                "speeches": _speeches_of(ledger, c),
                "quota": int((ledger.get(c) or {}).get("quota", quota)),
            }
            for c in eligible
        ]
        chosen = interrupt({
            "round": next_round,
            "total": max_rounds,
            "topic": topic,
            "candidates": candidates,
        })
        cid = str(chosen or "").strip()
        if cid in eligible:
            by, reason, note = "user", "", ""
        else:
            cid = _fallback_pick(ledger, eligible)
            by, reason, note = "host", "", "未收到有效点将，主持人兜底"
    else:
        messages = build_host_messages(
            topic=topic,
            roster=roster,
            ledger=ledger,
            transcript=transcript,
            round_index=next_round,
            total_rounds=max_rounds,
        )
        llm_calls += 1
        decision = {"next_speaker_id": "", "reason": "", "converged": False}
        _host_prompt = _messages_chars(messages)
        _t_host = time.perf_counter()
        try:
            llm = get_chat_llm(temperature=0.3, max_tokens=_HOST_MAX_TOKENS)
            resp = await ainvoke_nonempty(llm, messages, attempts=1)
            _raw = str(getattr(resp, "content", "") or "")
            _host_out = len(_raw)
            decision = parse_host_decision(_raw)
        except Exception as e:
            print(f"[Roundtable] 主持人决策失败: {e}")
        _host_ms = (time.perf_counter() - _t_host) * 1000

        if decision.get("converged"):
            out = _converge("host")
            out["llm_calls"] = llm_calls
            return out

        cid = str(decision.get("next_speaker_id") or "").strip()
        reason = str(decision.get("reason") or "").strip().replace("\n", " ")[:_HOST_REASON_CHARS]
        if cid not in eligible:
            # 模型指认了不在名单 / 额度已满的人：规则兜底，别让会议空转
            cid = _fallback_pick(ledger, eligible)
            reason = ""
            note = "主持人指认无效，改由规则兜底"

    # ---- 4. 事件（放在定人之后，避免挂起重放时重复）----
    write({"type": "round", "round": next_round, "total": max_rounds, "phase": "rebuttal"})
    write({
        "type": "choice",
        "character_id": cid,
        "name": _name_of(state, cid),
        "by": by,
        "reason": reason,
        "candidates": candidates,
        "note": note,
    })

    return {
        "pending_speaker": cid,
        "pending_kind": "rebuttal",
        "pending_round": next_round,
        "pending_reason": reason,
        "llm_calls": llm_calls,
        "ended": False,
        **_tfields(),
    }


# ============================================================
# 子智能体节点：speaker —— 一对一那套（自持检索、逐 token 流式）
# ============================================================
async def speaker_node(state: RoundtableState) -> dict:
    """让 `pending_speaker` 指定的角色发言，逐 token 推事件，并把这次发言记进台账。

    所有与会者共用这一个节点：身份来自状态而非节点本身，所以"子智能体之间没有边"
    是结构性的——图上根本不存在 speaker→speaker 的连线。
    """
    write = _writer()
    cid = str(state.get("pending_speaker") or "")
    ch = persona_chat_config.characters.get(cid)
    ledger = {k: dict(v) for k, v in (state.get("ledger") or {}).items()}

    if ch is None or cid not in ledger:
        # 角色在会议进行中被删掉（自建角色可删）——提前收场，别把整场拖垮
        write({"type": "error", "content": f"角色 {cid} 已不可用，本场会议提前结束。"})
        return {
            "converged": True,
            "converge_reason": "speaker_missing",
            "ended": True,
            "pending_speaker": "",
        }

    entry = ledger[cid]
    is_opening = str(state.get("pending_kind") or "") == "opening"
    rnd = int(state.get("pending_round") or 0)
    reason = str(state.get("pending_reason") or "")

    write({
        "type": "speaker_start",
        "character_id": ch.id,
        "name": ch.name,
        "avatar": ch.avatar or "🎭",
        "theme": ch.theme or "original",
        "speeches": int(entry.get("speeches", 0)) + 1,
        "quota": int(entry.get("quota", state.get("quota") or 2)),
        "reason": reason,
    })

    messages = await build_speaker_messages(
        character=ch,
        topic=str(state.get("topic") or ""),
        transcript=list(state.get("transcript") or []),
        round_index=rnd,
        total_rounds=int(state.get("max_rounds") or 1),
        is_opening=is_opening,
    )

    full = ""
    label = f"{ch.id} {'开场' if is_opening else f'交锋{rnd}'}"
    _prompt_chars = _messages_chars(messages)
    _t_speech = time.perf_counter()
    async for tok in _stream_llm_with_retry(messages, label=label):
        if not tok:
            continue
        full += tok
        write({"type": "token", "character_id": ch.id, "content": tok})
    _speech_ms = (time.perf_counter() - _t_speech) * 1000

    if not full:
        full = f"（{ch.name} 暂时无法发言。）"
        write({"type": "token", "character_id": ch.id, "content": full})

    from src.core.content_filter import sanitize_output

    full = sanitize_output(full).strip()

    entry["speeches"] = int(entry.get("speeches", 0)) + 1
    ledger[cid] = entry
    write({"type": "speaker_end", "character_id": ch.id, "content": full})

    out: dict = {
        "transcript": [{
            "character_id": ch.id,
            "name": ch.name,
            "content": full,
            "kind": "opening" if is_opening else "rebuttal",
            "round": rnd,
        }],
        "ledger": ledger,
        "llm_calls": int(state.get("llm_calls") or 0) + 1,
        "pending_speaker": "",
        "timings": _merge_timings(state.get("timings"), speaker_ms=_speech_ms),
        "prompt_chars": int(state.get("prompt_chars") or 0) + _prompt_chars,
        "answer_chars": int(state.get("answer_chars") or 0) + len(full),
    }
    if is_opening:
        out["opening_done"] = [cid]
    else:
        out["round_index"] = rnd
    return out


# ============================================================
# 组装图
# ============================================================
def route_after_host(state: RoundtableState) -> str:
    """主持人之后的唯一分叉：还有人要说话就去 speaker，否则收场。"""
    if state.get("ended") or state.get("converged"):
        return END
    if not str(state.get("pending_speaker") or ""):
        return END
    return "speaker"


def build_roundtable_graph(checkpointer=None):
    """编译圆桌图。传入 checkpointer 才能跨请求挂起/恢复。

    拓扑：START→host，host→(speaker | END)，speaker→host。
    注意 speaker 只连回 host —— 子智能体之间没有边，这是刻意的结构约束。
    """
    g = StateGraph(RoundtableState)
    g.add_node("host", host_node)
    g.add_node("speaker", speaker_node)
    g.add_edge(START, "host")
    g.add_conditional_edges("host", route_after_host, {"speaker": "speaker", END: END})
    g.add_edge("speaker", "host")
    return g.compile(checkpointer=checkpointer)


# ============================================================
# 驱动：把图的两股流翻译成前端事件
# ============================================================
class _DriveOutcome:
    """记录本次驱动是"跑完了"还是"挂起等点将"——决定要不要推 end + 纪要。"""

    __slots__ = ("finished",)

    def __init__(self) -> None:
        self.finished = False


def _ckpt_path() -> str:
    p = str(getattr(settings, "roundtable_checkpoint_db_path", "./data/roundtable_checkpoints.db"))
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return p


def _rt_config(session_id: str) -> dict:
    return {
        "configurable": {"thread_id": session_id},
        # 保险丝：host→speaker 交替，实测 3 开场 + 3 交锋约 15 个 super-step。
        # 写 25 是主动收紧——万一路由被改出环，宁可中止也不要无限跑。
        "recursion_limit": 25,
    }


async def _drive_graph(graph, graph_input, config: dict, outcome: _DriveOutcome) -> AsyncIterator[dict]:
    """驱动图并翻译事件。

    · `custom` 流：节点里 writer() 推的就是前端要的事件，逐条透传。
    · `updates` 流：只用来捕获 `__interrupt__`（用户主持时的挂起请求）。
    """
    session_id = config["configurable"]["thread_id"]
    try:
        async for mode, chunk in graph.astream(
            graph_input, config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                if not isinstance(chunk, dict) or not chunk.get("type"):
                    continue
                yield chunk
                if chunk.get("type") == "speaker_end" and ROUNDTABLE_SPEAKER_PAUSE > 0:
                    # 给用户留出读上一段的时间，同时把 LLM 调用压到网关可持续速率内
                    await asyncio.sleep(ROUNDTABLE_SPEAKER_PAUSE)
            else:
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    intrs = chunk.get("__interrupt__") or ()
                    payload = getattr(intrs[0], "value", {}) if intrs else {}
                    if not isinstance(payload, dict):
                        payload = {}
                    yield {"type": "choice_request", "session_id": session_id, **payload}
    except Exception as e:
        print(f"[Roundtable] 图执行异常: {e}")
        yield {"type": "error", "content": str(e)}
        outcome.finished = True
        return

    snap = await graph.aget_state(config)
    outcome.finished = not list(snap.next)


async def _tail_events(graph, config: dict, session_id: str) -> AsyncIterator[dict]:
    """正常收场后的尾巴：end + 非阻塞纪要。

    `end` 里带上**权威的完整记录**（roster / transcript / rounds），而不是让调用方
    去累积前面逐个 SSE 事件——因为用户点将后恢复出来的那条流不会再发 `start`，
    也拿不到挂起前那些发言；只有从图状态里取才是完整的。
    """
    snap = await graph.aget_state(config)
    values = dict(snap.values or {})
    ledger = dict(values.get("ledger") or {})
    transcript = list(values.get("transcript") or [])

    # 整场耗时以 started_at 为准——跨请求续跑出来的那条流也不会把它重置成
    # "这次请求的起点"，否则点将恢复那一场的 latency 会被算成只有最后一段。
    _started = float(values.get("started_at") or 0.0)
    _meeting_ms = int((time.time() - _started) * 1000) if _started else 0

    yield {
        "type": "end",
        "session_id": session_id,
        "topic": str(values.get("topic") or ""),
        "rounds": int(values.get("max_rounds") or 0),
        "speakers": list(values.get("roster") or []),
        "transcript": transcript,
        "converge_reason": str(values.get("converge_reason") or ""),
        # 用量统计（供 monitor 落库）：段耗时 / 输入输出字符 / 整场耗时
        "timings": dict(values.get("timings") or {}),
        "prompt_chars": int(values.get("prompt_chars") or 0),
        "answer_chars": int(values.get("answer_chars") or 0),
        "meeting_ms": _meeting_ms,
    }

    if not getattr(settings, "roundtable_summary_enabled", True):
        return
    yield {"type": "summary_start"}
    try:
        data = await build_roundtable_summary(
            topic=str(values.get("topic") or ""),
            transcript=transcript,
            ledger=ledger,
        )
        data["topic"] = str(values.get("topic") or "")
        data["speeches"] = [
            {
                "id": cid,
                "name": v.get("name") or cid,
                "speeches": int(v.get("speeches", 0)),
                "quota": int(v.get("quota", 0)),
            }
            for cid, v in ledger.items()
        ]
        yield {"type": "summary", "data": data}
    except Exception as e:
        print(f"[Roundtable] 纪要旁路失败: {e}")
        yield {"type": "summary_error", "content": "纪要暂时无法生成，会议内容已完整保留。"}
    yield {"type": "summary_end"}


# ============================================================
# 对外入口
# ============================================================
def _resolve_speakers(character_ids: list[str]) -> list:
    seen: set[str] = set()
    out = []
    for cid in character_ids:
        if cid in seen:
            continue
        ch = persona_chat_config.characters.get(cid)
        if not ch:
            continue
        seen.add(cid)
        out.append(ch)
    return out


async def stream_roundtable(
    topic: str,
    character_ids: list[str],
    rounds: int,
    session_id: str,
    picker: str = "user",
) -> AsyncIterator[dict]:
    """圆桌会议主流程：开场陈述 → 主持人逐轮定人 → 纪要。

    picker: "user"（用户当主持人，遇决策点挂起等前端回填）/ "agent"（主持人自己决定）。
    若图在中途挂起，本生成器**不推 end**——前端收到 choice_request 后调 resume_roundtable。
    """
    speakers = _resolve_speakers(character_ids)
    if len(speakers) < 2:
        yield {"type": "error", "content": "圆桌会议至少需要 2 位有效的对话者。"}
        return
    if len(speakers) > MAX_ROUNDTABLE_CHARS:
        yield {
            "type": "error",
            "content": f"圆桌人数过多（上限 {MAX_ROUNDTABLE_CHARS} 位）。人太多反而难以形成有效交锋，请精简参与人物。",
        }
        return

    rounds = max(1, min(int(rounds), _MAX_ROUNDS))
    picker = "agent" if str(picker).lower() == "agent" else "user"
    max_calls = max(4, int(getattr(settings, "roundtable_max_llm_calls", 24) or 24))
    quota = max(1, int(getattr(settings, "roundtable_max_speeches_per_speaker", 2) or 2))

    roster = [
        {
            "id": ch.id,
            "name": ch.name,
            "avatar": ch.avatar or "🎭",
            "theme": ch.theme or "original",
            "quota": quota,
        }
        for ch in speakers
    ]

    initial: RoundtableState = {
        "session_id": session_id,
        "topic": topic,
        "picker": picker,
        "max_rounds": rounds,
        "max_calls": max_calls,
        "quota": quota,
        "roster": roster,
        "ledger": {
            ch.id: {"name": ch.name, "speeches": 0, "quota": quota} for ch in speakers
        },
        "transcript": [],
        "opening_done": [],
        "round_index": 0,
        "pending_speaker": "",
        "pending_kind": "",
        "pending_round": 0,
        "pending_reason": "",
        "converged": False,
        "converge_reason": "",
        "llm_calls": 0,
        "ended": False,
        "started_at": time.time(),
        "timings": {},
        "prompt_chars": 0,
        "answer_chars": 0,
    }

    config = _rt_config(session_id)
    outcome = _DriveOutcome()
    async with AsyncSqliteSaver.from_conn_string(_ckpt_path()) as saver:
        graph = build_roundtable_graph(checkpointer=saver)

        # 同一 session_id 上若还挂着未完成的会议，不要覆盖它的现场。
        # 这一步放在 `start` 之前——否则前端会先看到舞台搭起来、再收到一条错误。
        snap = await graph.aget_state(config)
        if list(snap.next):
            yield {"type": "error", "content": "这场会议还在等待点将，请先完成它或换个会话。"}
            return
        # 清掉同 id 的历史快照，避免上一场的转写串进这一场
        try:
            if hasattr(saver, "adelete_thread"):
                await saver.adelete_thread(session_id)
        except Exception as e:
            print(f"[Roundtable] 清理历史 checkpoint 失败 {session_id}: {e}")

        yield {
            "type": "start",
            "session_id": session_id,
            "topic": topic,
            "picker": picker,
            "host": "user" if picker == "user" else "agent",
            "speakers": roster,
        }

        async for evt in _drive_graph(graph, initial, config, outcome):
            yield evt
        if outcome.finished:
            async for evt in _tail_events(graph, config, session_id):
                yield evt


async def resume_roundtable(session_id: str, character_id: str) -> AsyncIterator[dict]:
    """用户点将后恢复会议：把选择作为 `Command(resume=...)` 喂回挂起的 host 节点。

    这是跨请求恢复——即便点将的请求被负载均衡打到了另一个 worker，只要连的是同一个
    SQLite，现场就还在。旧的实现用进程内 asyncio.Event，多 worker 下会静默丢选择。
    """
    config = _rt_config(session_id)
    outcome = _DriveOutcome()
    async with AsyncSqliteSaver.from_conn_string(_ckpt_path()) as saver:
        graph = build_roundtable_graph(checkpointer=saver)
        snap = await graph.aget_state(config)
        if not list(snap.next):
            yield {"type": "error", "content": "这场会议已经结束了（或不存在）。"}
            return
        async for evt in _drive_graph(
            graph, Command(resume=str(character_id or "")), config, outcome
        ):
            yield evt
        if outcome.finished:
            async for evt in _tail_events(graph, config, session_id):
                yield evt


async def drop_roundtable_checkpoint(session_id: str) -> None:
    """删除一场会议的图现场（历史记录删除时一并清掉，避免 checkpoint 越积越多）。"""
    try:
        async with AsyncSqliteSaver.from_conn_string(_ckpt_path()) as saver:
            if hasattr(saver, "adelete_thread"):
                await saver.adelete_thread(session_id)
    except Exception as e:
        print(f"[Roundtable] 删除 checkpoint 失败 {session_id}: {e}")


async def roundtable_pending_choice(session_id: str) -> dict | None:
    """查一场会议当前是否挂在"等点将"。返回挂起载荷（含候选人），否则 None。

    前端刷新页面后靠它把点将弹窗找回来——挂起状态在 SQLite 里，不在内存里。
    """
    try:
        config = _rt_config(session_id)
        async with AsyncSqliteSaver.from_conn_string(_ckpt_path()) as saver:
            graph = build_roundtable_graph(checkpointer=saver)
            snap = await graph.aget_state(config)
            if not list(snap.next):
                return None
            for task in snap.tasks or []:
                for intr in getattr(task, "interrupts", ()) or ():
                    val = getattr(intr, "value", None)
                    if isinstance(val, dict):
                        return {"session_id": session_id, **val}
    except Exception as e:
        print(f"[Roundtable] 查询挂起失败 {session_id}: {e}")
    return None
