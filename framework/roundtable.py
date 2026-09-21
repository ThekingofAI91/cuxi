"""
圆桌会议（争鸣）—— 自主发言编排引擎

多名名人角色同桌，就同一议题交锋。**谁说话不由代码排班，由角色自己判断**：
每一轮先并行问每位参会者"此刻有没有非说不可的话"，有人举手才发言；
两人以上同时举手，交给用户点将（用户不主持则由主持人代班裁决）。

设计要点：
- **参会者是平级的**，没有"中心调度"：是否发言由各角色自己决定（这是全项目第二处
  LLM 输出改变控制流的地方，与主链路的确定性编排相对）。
- **意愿判断要有靶子**：判据不是"你想不想说"，而是"你要反驳谁的哪一句"。
  这是压制"过度举牌"的关键——被问要不要发言时，角色扮演倾向会让模型几乎人人说
  要；改成要求指出具体靶子后，附和型发言天然被挡掉，同时解决"和稀泥"。
- **收敛靠配额的数学，不靠提示词调参**：每人发言次数有上限，配额耗尽不再主动
  举牌；无人举手 = 讨论自然收敛，优雅终止（比让主持人判断"该收了"更有说服力）。
- **结尾只有纪要，没有裁判**：书记员只记账不评分（立场 / 论据 / 交锋线 / 悬而未决），
  最后把判断交回用户。理由：给真实人物判"谁说得对"既是合规风险，也等于替用户把
  思考做完，与产品定位自相抵消。
- 每场会议有 LLM 调用次数**硬上限**（见 settings.roundtable_max_llm_calls）。
  自主发言后每轮调用次数不再固定，没有硬上限时超 RPM 会表现为"某位角色静默失声"。

事件流（前端据此渲染）：
  start           {session_id, topic, speakers:[{id,name,avatar,theme,quota}], picker}
  round           {round, total, phase: "opening"|"rebuttal"}
  intent_start    {speakers:[{id,name}]}
  intent          {character_id, name, speak, reason}
  contention      {candidates:[{id,name,reason}], timeout, moderator_default}
  choice          {character_id, by: "auto"|"user"|"moderator", candidates, note}
  converged       {round, reason}
  budget_exhausted{}
  speaker_start   {character_id, name, avatar, theme, speeches, quota}
  token           {character_id, content}
  speaker_end     {character_id, content}
  end             {session_id}
  summary_start   {}
  summary         {data: {clashes, positions, open, speeches, degraded}}
  summary_error   {content}
  summary_end     {}
  error           {content}

  注意：纪要事件在 end **之后**推送，属非阻塞旁路——纪要失败只是少一段纪要，
  会议结果本身已经完整交付（与 verifier / citations 的"先返回、异步补推"同一思路）。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from src.core.config import settings
from src.core.llm import ainvoke_nonempty, astream_nonempty, get_chat_llm
from framework.supervisor import get_chroma_client
from scenes.persona_chat.config import persona_chat_config
from scenes.persona_chat.prompt_builder import build_character_prompt
from src.retrieval.embedder import get_embedder


# 圆桌人数上限：人太多会摊薄每位的发言篇幅，削弱交锋感。
# 定 3 的依据（P(≥2 人举手) = 1-(1-p)^N - N·p·(1-p)^(N-1)，p 取 0.7）：
#   2 人 49% / 3 人 78% / 4 人 92% —— 4 人以上几乎每轮都要用户点鼠标，
#   产品从"观赏"退化成"操作"。3 人是有交锋感又不累人的上限。
MAX_ROUNDTABLE_CHARS = 3

# 交锋轮数上限：钳制用户输入，避免超预算。
_MAX_ROUNDS = 3

# 发言间隔（秒）：每位发言结束后停顿，既给用户留出阅读上一段的时间，
# 也把 LLM 调用频率压到 DeepSeek RPM 限额（默认 20/min）以内，避免 429。
ROUNDTABLE_SPEAKER_PAUSE = 4.0

# ---- max_tokens 预算：这是推理模型，预算给少了正文会**静默为空** ----
# 实测（deepseek-v4-flash @ token.sensenova.cn，2026-09-21）：
#   该模型是推理模型，reasoning token 与正文**共用** max_tokens 预算。预算不足时
#   上游把预算全烧在 reasoning 上，正文返回空字符串，而且**不抛异常、不打日志**：
#     意愿判断 max_tokens=96   → out=96  全 reasoning，正文 ''，3.9s
#     意愿判断 max_tokens=300  → out=300 全 reasoning，正文 ''，7.6s
#     意愿判断 max_tokens=700  → out=700 全 reasoning，正文 ''，16.4s
#     意愿判断 max_tokens=1200 → out=126（reasoning 98），正文完整，2.3s
#   后果：意愿判断永远解析失败 → 一律按"不举手"处理 → **圆桌再也没人主动发言**，
#   而表面上一切正常（不报错、日志干净、只有角色集体沉默）。这类静默失效比报错难查得多。
# 对策两条：
#   ① 预算给足（下面三个常量）；
#   ② **意愿判断改流式**——边收边判、凑齐 JSON 立刻断开，上游还没想完就已经拿到答案
#      （实测流式 300 预算就能拿到完整 JSON，3.0s）。
# 注意：reasoning 长度浮动很大（实测 98 ~ 962），所以预算要按 worst case 留。
_SPEECH_MAX_TOKENS = 1600
_INTENT_MAX_TOKENS = 800
_SUMMARY_MAX_TOKENS = 1600

# ---- 意愿判断的上下文预算 ----
# 上下文本身压到很小：只看最近 3 轮、每人 120 字、人设截 700 字。意愿判断每轮每位
# 都要跑一次，是自主发言制调用次数最多的一环，上下文越小越省。
_INTENT_CTX_TURNS = 3
_INTENT_TURN_CHARS = 120
_INTENT_PERSONA_CHARS = 700
_INTENT_REASON_CHARS = 40

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
# 发言意愿判断：判据是"有没有靶子"，不是"想不想说"
# ============================================================
INTENT_DIRECTIVE = """
现在还没轮到你发言。这一步不是发言，是一次**发言前的内心判断**：此刻你有没有非说不可的话。

【举手的唯一理由：必须有靶子】
· 有人说了与你相冲突的话，你要反驳它——心里要能默念出对方那句原话的关键词；
· 或者有人点名曲解了你的立场，你必须澄清。

【不要举手】以下情况都算没靶子，一律沉默：
· 只是想再补充一点自己的看法；
· 想附和、赞同、顺着别人的话说；
· 你的立场前面已经讲清楚了，只是换个说法重说一遍。

沉默在这里是得体的。宁可不说话，也不要无靶子地插话。
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
# 圆桌台账：每人一格账，参会者写入 / 主持人与书记员只读
# ============================================================
@dataclass
class SpeakerLedger:
    """单个参会者的台账。"""

    character_id: str
    name: str
    quota: int = 2       # 本场发言次数上限（含开场）
    speeches: int = 0    # 已发言次数

    @property
    def remaining(self) -> int:
        """剩余主动举牌额度。配额只约束**主动**举牌，不约束被点名。"""
        return max(0, self.quota - self.speeches)


@dataclass
class RoundtableSession:
    """一场进行中的会议。前端点将端点靠 session_id 找回它。"""

    session_id: str
    topic: str
    ledger: dict[str, SpeakerLedger] = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    touched: float = field(default_factory=time.time)
    picker: str = "user"                       # "user" = 用户主持；"agent" = 主持人代班
    choice_result: Optional[str] = None
    _choice_event: asyncio.Event = field(default_factory=asyncio.Event)

    def note_call(self) -> None:
        """记一次 LLM 调用。调用方在真正发起请求前调用。"""
        self.llm_calls += 1
        self.touched = time.time()

    def speaker(self, character_id: str):
        return persona_chat_config.characters.get(character_id)


# 进行中会议的注册表。只存内存：会议是分钟级交互，进程重启即作废，
# 不必落盘（历史回看走 session_store 的 roundtables 表）。
_RT_SESSIONS: dict[str, RoundtableSession] = {}
_RT_SESSION_TTL = 1800.0  # 半小时无活动即回收，防内存泄漏


def register_roundtable_session(session: RoundtableSession) -> None:
    _gc_roundtable_sessions()
    _RT_SESSIONS[session.session_id] = session


def get_roundtable_session(session_id: str) -> Optional[RoundtableSession]:
    return _RT_SESSIONS.get(session_id)


def drop_roundtable_session(session_id: str) -> None:
    _RT_SESSIONS.pop(session_id, None)


def submit_roundtable_choice(session_id: str, character_id: str) -> bool:
    """用户点将：把选择写进会议并唤醒正在等待的流。返回 False 表示会议已结束/不存在。"""
    session = _RT_SESSIONS.get(session_id)
    if session is None:
        return False
    session.choice_result = character_id
    session.touched = time.time()
    session._choice_event.set()
    return True


def _gc_roundtable_sessions() -> None:
    now = time.time()
    dead = [sid for sid, s in _RT_SESSIONS.items() if now - s.touched > _RT_SESSION_TTL]
    for sid in dead:
        _RT_SESSIONS.pop(sid, None)


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
        # 去重 + 拼接
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


def _format_debate_so_far(transcript: list[dict], exclude_id: Optional[str] = None) -> str:
    """把已有发言格式化为『其他与会者观点』，供当前发言者参考（默认排除自己）。

    截断策略：最多保留最近约 1500 字（从最早的发言开始丢弃），
    既保证交锋轮输入的上下文长度稳定（避免 DeepSeek "message too long"），
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
    """
    拼装某发言者本轮的 LangChain 消息列表（system / user）。
    检索异步进行，故本函数为 async。
    """
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


def build_intent_messages(
    *,
    character,
    topic: str,
    transcript: list[dict],
    ledger: SpeakerLedger,
) -> list[tuple[str, str]]:
    """拼装"要不要发言"的判断消息。

    刻意压到最小：人设截断、只看最近几轮、每轮截断。这一步每轮每位都要跑一次，
    是自主发言制最贵的一环（不压的话大部分 token 会花在"问要不要说话"上）。
    """
    role_prompt = build_character_prompt(character, query=topic) or ""
    if len(role_prompt) > _INTENT_PERSONA_CHARS:
        role_prompt = role_prompt[:_INTENT_PERSONA_CHARS]

    recent = transcript[-_INTENT_CTX_TURNS:]
    lines = []
    for turn in recent:
        body = (turn.get("content") or "").strip().replace("\n", " ")[:_INTENT_TURN_CHARS]
        lines.append(f"【{turn.get('name', '某位')}】{body}")
    recent_text = "\n".join(lines) if lines else "（还没有人发言）"

    system_parts = [role_prompt, INTENT_DIRECTIVE]
    system = "\n\n".join(p for p in system_parts if p)

    user_msg = (
        f"议题：{topic}\n\n"
        f"【最近几位说的话】\n{recent_text}\n\n"
        f"【你的发言次数】已发言 {ledger.speeches} 次（本场上限 {ledger.quota} 次）\n\n"
        f"现在判断：你有没有非说不可的话？\n"
        f"只输出一行 JSON，不要任何其他内容：\n"
        f'{{"speak": true 或 false, "reason": "若举手，20 字内说清你要反驳谁、哪一句"}}'
    )
    return [("system", system), ("user", user_msg)]


# ============================================================
# 解析
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
_SPEAK_FIELD_RE = re.compile(r"[\"']?speak[\"']?\s*[:：]?\s*(true|false|是|否)", re.I)


def parse_intent(raw: str) -> tuple[bool, str]:
    """解析意愿判断的返回。

    解析不出来时一律按"不举手"处理——宁愿少一次发言，也不要无靶子地插话。
    代价可控：无人举手是**合法的终止条件**（讨论收敛），不会把会议卡死。
    """
    text = (raw or "").strip()
    if not text:
        return False, ""
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
        if isinstance(obj, dict) and "speak" in obj:
            speak = bool(obj.get("speak"))
            reason = str(obj.get("reason") or "").strip().replace("\n", " ")
            return speak, (reason[:_INTENT_REASON_CHARS] if speak else "")
    # 兜底：模型把 JSON 包在解释里、或直接答"是/否"
    m2 = _SPEAK_FIELD_RE.search(text)
    if m2:
        val = m2.group(1).lower()
        return val in ("true", "是"), ""
    return False, ""


def parse_summary(raw: str) -> Optional[dict]:
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


def _fallback_positions(session: RoundtableSession) -> list[dict]:
    """零 LLM 的降级版「各方主张」：取每人第一次发言的开头。

    纪要生成失败时用它兜底——至少让用户看到每人立场，而不是一块空白。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for turn in session.transcript:
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
# 发言意愿判断
# ============================================================
async def _read_intent_json(llm, messages) -> str:
    """流式读意愿判断的正文，凑齐一个完整 JSON 对象就立刻断开。

    为什么必须流式（不是图快，是图**拿得到正文**）：见文件头 max_tokens 预算那段的实测——
    非流式时上游要把整段生成完才返回，推理模型的 reasoning 会把预算吃干净并返回空正文；
    流式则能边收边判，上游还没想完我们就已经拿到要的那一行 JSON。
    """
    buf = ""
    async for tok in astream_nonempty(llm, messages):
        buf += tok
        # 我们只要一行 JSON：出现配对的花括号即可收工
        if "{" in buf and "}" in buf:
            break
    return buf


async def poll_speaker_intent(session: RoundtableSession, character) -> dict:
    """问一位参会者：此刻有没有非说不可的话。

    任何异常都降级为"不举手"：单个角色的判断失败不该影响整场会议。
    """
    ledger = session.ledger[character.id]
    result = {"character_id": character.id, "name": character.name, "speak": False, "reason": ""}
    messages = build_intent_messages(
        character=character, topic=session.topic, transcript=session.transcript, ledger=ledger
    )
    try:
        session.note_call()
        llm = get_chat_llm(temperature=0.2, max_tokens=_INTENT_MAX_TOKENS)
        raw = await _read_intent_json(llm, messages)
        speak, reason = parse_intent(raw)
        if speak and ledger.remaining <= 0:
            # 配额是硬闸门，模型说了不算（判据里已告知配额，这里是兜底）
            speak, reason = False, ""
        result["speak"] = speak
        result["reason"] = reason
    except Exception as e:
        print(f"[Roundtable] 意愿判断失败 {character.id}: {e}")
    return result


def moderator_pick(session: RoundtableSession, candidate_ids: list[str]) -> str:
    """主持人代班裁决：多人争抢发言时，让**本场说得最少**的那位先说。

    这不是"裁判"——它只决定下一个谁说话，不对任何内容做评价。
    平票时保持传入顺序（min 稳定排序），也就是发言顺序的原样。
    """
    return min(candidate_ids, key=lambda cid: session.ledger[cid].speeches)


def _prepare_choice_wait(session: RoundtableSession) -> None:
    """在**推送 contention 事件之前**清空选择箱，并进入等待态。

    顺序很关键：如果先 yield 再清空，用户（或测试）在收到事件的那一刻就提交的选择
    会被 clear() 抹掉——点将直接丢失。
    """
    session.choice_result = None
    session._choice_event.clear()


async def _wait_user_choice(session: RoundtableSession, timeout: float) -> Optional[str]:
    """阻塞等待用户点将，超时返回 None（由主持人代班接管）。"""
    try:
        await asyncio.wait_for(session._choice_event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        return None
    return session.choice_result


# ============================================================
# 纪要（书记员）：只记账，不评分
# ============================================================
def _build_summary_messages(session: RoundtableSession) -> list[tuple[str, str]]:
    lines = []
    total = 0
    for turn in session.transcript:
        body = (turn.get("content") or "").strip().replace("\n", " ")[:_SUMMARY_TURN_CHARS]
        line = f"【{turn.get('name', '某位')}】{body}"
        if total + len(line) > _SUMMARY_TOTAL_CHARS:
            break
        lines.append(line)
        total += len(line)
    transcript_text = "\n\n".join(lines) if lines else "（没有发言记录）"

    counts = "、".join(
        f"{e.name} 发言 {e.speeches} 次（上限 {e.quota}）" for e in session.ledger.values()
    )
    user_msg = (
        f"议题：{session.topic}\n\n"
        f"【完整发言记录】\n{transcript_text}\n\n"
        f"【发言次数（仅作事实记录，不参与任何评价）】{counts}\n\n"
        f"请按结构输出 JSON。记住：你是书记员，不是裁判。"
    )
    return [("system", CLERK_DIRECTIVE), ("user", user_msg)]


async def build_roundtable_summary(session: RoundtableSession) -> dict:
    """生成纪要（分歧地图）。返回 {clashes, positions, open, degraded}。

    两段防线：① 提示词里禁止宣布胜负；② 输出若命中禁用词，纠正重试一次；
    仍越界则走零 LLM 的降级版（只列每人立场）+ 剔掉越界条目。
    这一层是必要的——"总结"是裁判最容易换件衣服回来的地方。
    """
    messages = _build_summary_messages(session)
    for _ in range(_SUMMARY_ATTEMPTS):
        try:
            session.note_call()
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

    data = {"clashes": [], "positions": _fallback_positions(session), "open": []}
    data["degraded"] = True
    return data


# ============================================================
# 单人发言
# ============================================================
async def _stream_speaker(
    session: RoundtableSession,
    character,
    *,
    is_opening: bool,
    round_index: int,
    total_rounds: int,
) -> AsyncIterator[dict]:
    """让一位角色发言，逐 token 产出事件。"""
    from src.core.content_filter import sanitize_output

    ledger = session.ledger[character.id]
    yield {
        "type": "speaker_start",
        "character_id": character.id,
        "name": character.name,
        "avatar": character.avatar or "🎭",
        "theme": character.theme or "original",
        "speeches": ledger.speeches + 1,
        "quota": ledger.quota,
    }
    messages = await build_speaker_messages(
        character=character,
        topic=session.topic,
        transcript=session.transcript,
        round_index=round_index,
        total_rounds=total_rounds,
        is_opening=is_opening,
    )
    session.note_call()
    full = ""
    label = f"{character.id} {'开场' if is_opening else f'交锋{round_index}'}"
    async for tok in _stream_llm_with_retry(messages, label=label):
        full += tok
        yield {"type": "token", "character_id": character.id, "content": tok}
    if not full:
        full = f"（{character.name} 暂时无法发言。）"
        yield {"type": "token", "character_id": character.id, "content": full}
    full = sanitize_output(full).strip()
    session.transcript.append({
        "character_id": character.id,
        "name": character.name,
        "content": full,
        "kind": "opening" if is_opening else "rebuttal",
        "round": round_index,
    })
    ledger.speeches += 1
    session.touched = time.time()
    yield {"type": "speaker_end", "character_id": character.id, "content": full}


# ============================================================
# 编排主流程：产出 SSE 事件流
# ============================================================
async def stream_roundtable(
    topic: str,
    character_ids: list[str],
    rounds: int,
    session_id: str,
    picker: str = "user",
) -> AsyncIterator[dict]:
    """圆桌会议主流程：开场陈述 → 自主发言的交锋轮 → 纪要。

    picker: "user"（用户主持，多人争抢时弹窗点将）/ "agent"（主持人代班，直接裁决）。
    """
    # 解析有效发言者（去重、过滤不存在的角色）
    seen_ids: set[str] = set()
    speakers = []
    for cid in character_ids:
        if cid in seen_ids:
            continue
        ch = persona_chat_config.characters.get(cid)
        if not ch:
            continue
        seen_ids.add(cid)
        speakers.append(ch)

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
    pick_timeout = max(5, int(getattr(settings, "roundtable_pick_timeout", 25) or 25))
    quota = max(1, int(getattr(settings, "roundtable_max_speeches_per_speaker", 2) or 2))

    session = RoundtableSession(session_id=session_id, topic=topic, picker=picker)
    for ch in speakers:
        session.ledger[ch.id] = SpeakerLedger(character_id=ch.id, name=ch.name, quota=quota)
    register_roundtable_session(session)

    yield {
        "type": "start",
        "session_id": session_id,
        "topic": topic,
        "picker": picker,
        "speakers": [
            {
                "id": ch.id,
                "name": ch.name,
                "avatar": ch.avatar or "🎭",
                "theme": ch.theme or "original",
                "quota": quota,
            }
            for ch in speakers
        ],
    }

    try:
        # ---- 开场陈述：每位一次，这是本场唯一的"排班"环节 ----
        yield {"type": "round", "round": 0, "total": rounds, "phase": "opening"}
        for ch in speakers:
            async for evt in _stream_speaker(
                session, ch, is_opening=True, round_index=0, total_rounds=rounds
            ):
                yield evt
            await asyncio.sleep(ROUNDTABLE_SPEAKER_PAUSE)

        # ---- 交锋轮：谁说话由角色自己决定 ----
        rebuttal_count = 0
        for r in range(1, rounds + 1):
            if session.llm_calls >= max_calls:
                yield {"type": "budget_exhausted", "llm_calls": session.llm_calls}
                break

            yield {"type": "round", "round": r, "total": rounds, "phase": "rebuttal"}

            eligible = [ch for ch in speakers if session.ledger[ch.id].remaining > 0]
            if not eligible:
                yield {"type": "converged", "round": r, "reason": "quota"}
                break

            yield {
                "type": "intent_start",
                "speakers": [{"id": c.id, "name": c.name} for c in eligible],
            }
            # 并行问：串行等 N 次会把体验拖死（每次都是一个完整 LLM 往返）
            intents = await asyncio.gather(
                *[poll_speaker_intent(session, ch) for ch in eligible]
            )
            for it in intents:
                yield {
                    "type": "intent",
                    "character_id": it["character_id"],
                    "name": it["name"],
                    "speak": it["speak"],
                    "reason": it["reason"],
                }

            raisers = [it for it in intents if it["speak"]]
            chosen_id = ""
            how = ""
            cands: list[dict] = []
            note = ""

            if len(raisers) == 1:
                chosen_id, how = raisers[0]["character_id"], "auto"
            elif len(raisers) > 1:
                cand_ids = [it["character_id"] for it in raisers]
                cands = [
                    {"id": it["character_id"], "name": it["name"], "reason": it["reason"]}
                    for it in raisers
                ]
                default_id = moderator_pick(session, cand_ids)
                if picker == "user":
                    # 推给用户点将。超时不选就由主持人代班，会议不会卡死。
                    # 先进入等待态再推事件，否则用户秒选的结果会被清空（见 _prepare_choice_wait）。
                    _prepare_choice_wait(session)
                    yield {
                        "type": "contention",
                        "candidates": cands,
                        "timeout": pick_timeout,
                        "moderator_default": default_id,
                    }
                    picked = await _wait_user_choice(session, float(pick_timeout))
                    if picked in cand_ids:
                        chosen_id, how = picked, "user"
                    else:
                        chosen_id, how = default_id, "moderator"
                        note = "超时未选，主持人代定"
                else:
                    chosen_id, how = default_id, "moderator"
                    note = "主持人代定"
            else:
                # 无人举手。首轮若一次交锋都没发生，说明开场只是各说各话——
                # 圆桌的意义就在于对撞，这里让主持人指定一次，保证至少有一次交锋。
                if r == 1 and rebuttal_count == 0:
                    chosen_id = moderator_pick(session, [ch.id for ch in speakers])
                    how = "moderator"
                    note = "首轮无人主动交锋，主持人指定"
                else:
                    yield {"type": "converged", "round": r, "reason": "no_raiser"}
                    break

            yield {
                "type": "choice",
                "character_id": chosen_id,
                "by": how,
                "candidates": cands,
                "note": note,
            }

            ch = session.speaker(chosen_id)
            if ch is None:
                break
            async for evt in _stream_speaker(
                session, ch, is_opening=False, round_index=r, total_rounds=rounds
            ):
                yield evt
            rebuttal_count += 1
            await asyncio.sleep(ROUNDTABLE_SPEAKER_PAUSE)

    except Exception as e:
        print(f"[Roundtable] 编排异常: {e}")
        yield {"type": "error", "content": str(e)}

    yield {"type": "end", "session_id": session_id}

    # ---- 纪要：end 之后的非阻塞旁路 ----
    if getattr(settings, "roundtable_summary_enabled", True):
        yield {"type": "summary_start"}
        try:
            data = await build_roundtable_summary(session)
            data["topic"] = topic
            data["speeches"] = [
                {"id": e.character_id, "name": e.name, "speeches": e.speeches, "quota": e.quota}
                for e in session.ledger.values()
            ]
            yield {"type": "summary", "data": data}
        except Exception as e:
            print(f"[Roundtable] 纪要旁路失败: {e}")
            yield {"type": "summary_error", "content": "纪要暂时无法生成，会议内容已完整保留。"}
        yield {"type": "summary_end"}

    drop_roundtable_session(session_id)
