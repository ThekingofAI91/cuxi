"""legend.py — 「传奇」剧情模式的游戏引擎。

它是什么
--------
用户给三段设定（世界观 / 主角 / 配角），然后**自己扮演主角**往下走；
**叙述与所有配角的言行由同一个模型扮演**（单模型全包）。

为什么是单模型而不是每个配角各调一次（设计取舍，写清免得后人改动）
------------------------------------------------------------------
多模型（像圆桌那样每个角色一个调用）能让配角口吻更有区分度，但代价是：
① 模型之间看不到彼此刚说的话，剧情会各说各话；
② 每轮 N+1 次调用，耗时翻倍、成本翻倍；
③ 剧情推进需要「一件事接着一件事」的连贯因果，这正是单一上下文最擅长的。
主流同类产品（AI Dungeon、Character.AI 的 RPG/Adventure）都是单模型全包。
所以这里**刻意不做多智能体**——不是省事，是叙事连贯性优先。

为什么不上 LangGraph
--------------------
控制流是一条直线：用户行动 → 生成一段叙述。没有分支、没有环、没有跨请求挂起，
`state` 就存在存档 JSON 里。上一张图（圆桌）之所以要图，是因为 host 节点每轮重入
且 picker=user 时要 interrupt 挂起；这里一个条件判断都没有，上图纯属给直线包一层壳。

输出格式约定（前端按此渲染，改动需同步 legend.js）
--------------------------------------------------
三类内容分开着色，判据是「这句在写谁」：
- 普通段落                → 环境与背景叙述（默认，不加标记）
- 行内全角括号 `（…）`    → 人物的神情与动作
- `**名字**：台词` 独立行 → 人物说的话（前端高亮发言人）
- 行内引号 `「…」` `“…”`  → 台词本体（与上一行同色系）
- `> 文字`                → 系统提示（如「本局结束」），模型一般不用，
                             留给前端自己插

状态栏（作者定义字段，模型每轮更新，见下）
------------------------------------------
作者在开局前定义两栏字段：`state`（通用，在场每个人物各一份）与
`only`（仅主角）。每轮叙述末尾模型额外吐一段

    <<<STATE
    {"state": {"苏娘": {"好感度": 15}}, "only": {"体力": 87}}
    STATE>>>

这段块由 `StatePatchFilter` 在流式过程中**摘掉**，玩家看不到；
解析出的值经白名单 + 类型钳制后合并进存档。没有定义字段的旧档
`build_state_block` 返回空串，行为与本功能上线前完全一致。

★真链路实测：模型并不总会附这个块（跑三遍命中 1/3、3/3、1/3）。
所以**正文没带块时**会走一次极小的结构化调用 `extract_state_patch`，
只把这一轮的变化从叙述里抽出来。状态栏不能随机停摆，这是兜底的必要性。
"""

from __future__ import annotations

import json
import re
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scenes.persona_chat.legend_store import (
    FIELD_KIND_LABELS,
    LegendSave,
)
from src.core.config import settings
from src.core.llm import ainvoke_nonempty, astream_nonempty, get_chat_llm

# 叙事风格档位 → prompt 里的一句话。三档够用：再多会让用户纠结选哪个，
# 而风格本身在「世界观」里其实已经被暗示了大半。
STYLE_HINTS = {
    "classic": "笔调沉稳，重画面与细节，像一部正经的长篇小说的叙述。",
    "light": "笔调轻快，允许幽默与吐槽，但不要玩梗玩到出戏。",
    "dark": "笔调肃杀，压迫感强，危险真实存在，代价不可轻易抹去。",
}

# 主角行动最多多长。太短（一句话）模型没有可发挥的着力点，
# 太长说明用户在写小说而不是在做选择。
MAX_ACTION_CHARS = 1000

# 状态块的起止标记。选这对尖括号是因为它在中文叙事正文里几乎不可能自然出现，
# 不会把正文误当块头切掉。★改动要同步 legend.js 的常量。
STATE_OPEN = "<<<STATE"
STATE_CLOSE = "STATE>>>"


def _fmt_value(field_dict: dict, value) -> str:
    """把一个状态值渲染成给模型看的短文本。"""
    if field_dict.get("kind") == "number":
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return "?"
    text = "" if value is None else str(value)
    return text if text else "（无）"


def _value_of(fields: list[dict], values, name: str):
    """取一个字段的当前值；没初始化就回落到定义里的初值。

    ★为什么要有这个回落：`state` 只在创建存档时初始化过一次。若存档是手工拼的、
    或字段是后来加的，值就可能缺。缺了还硬渲染就会在 prompt 里写出「体力 ?」——
    模型会当成真的不知道，然后自己发明一个数。
    """
    if isinstance(values, dict) and name in values:
        return values[name]
    for f in fields:
        if f.get("name") == name:
            return f.get("init")
    return None


def _sample_patch(save: LegendSave) -> str:
    """给模型一个**照着填**的具体样例。

    给 `{}` 当样例的提示词实测效果很差——模型会照抄空对象，
    或者在键名上自由发挥。给一个键名正确、值明显是占位符的样例，
    命中率高得多。
    """
    names = save.character_names()
    sample_state: dict = {}
    if save.state_fields and names:
        # 有配角就用配角举例（通用字段在配角身上最能被理解）
        who = names[1] if len(names) > 1 else names[0]
        sample_state[who] = {
            f["name"]: (12 if f["kind"] == "number" else "短词")
            for f in save.state_fields[:2]
        }
    sample_only = {
        f["name"]: (80 if f["kind"] == "number" else "短词")
        for f in save.only_fields[:2]
    }
    return json.dumps({"state": sample_state, "only": sample_only},
                      ensure_ascii=False)


def _field_lines(fields: list[dict]) -> list[str]:
    """字段定义渲染成若干行（build_state_block 与状态抽取共用，防两处漂移）。"""
    out = []
    for f in fields:
        desc = f"：{f['desc']}" if f.get("desc") else ""
        out.append(
            f"- {f['name']}（{FIELD_KIND_LABELS.get(f['kind'], '数值')}，"
            f"初值 {_fmt_value(f, f.get('init'))}）{desc}"
        )
    return out


def _current_value_lines(save: LegendSave) -> list[str]:
    """当前值渲染成若干行（同上，两处共用）。"""
    out = []
    if save.state_fields:
        for who in save.character_names():
            vals = (save.state or {}).get(who) or {}
            body = "、".join(
                f"{f['name']} {_fmt_value(f, _value_of(save.state_fields, vals, f['name']))}"
                for f in save.state_fields
            )
            out.append(f"- {who}：{body}")
    if save.only_fields:
        body = "、".join(
            f"{f['name']} {_fmt_value(f, _value_of(save.only_fields, save.only, f['name']))}"
            for f in save.only_fields
        )
        out.append(f"- {save.protagonist_name}（专属）：{body}")
    return out


def build_state_block(save: LegendSave) -> str:
    """状态栏的 prompt 段：字段定义 + 当前值 + 输出契约。

    ★没有定义任何字段时返回空串——旧档（本功能上线前建的）必须走完全相同的路径，
    否则等于给所有老存档悄悄加了一段它们没有的指令。
    """
    if not save.has_state():
        return ""

    lines: list[str] = ["## 状态栏", ""]
    lines.append("这是本局的数值面板。你每轮都要维护它，并在回复的最后更新。")
    lines.append("")

    # ---- 字段定义 ----
    lines.append("### 字段定义")
    lines.append("")
    if save.state_fields:
        lines.append("通用（在场**每个人物各有一份**）：")
        lines.extend(_field_lines(save.state_fields))
        lines.append("")
    if save.only_fields:
        lines.append(f"仅主角（只有 {save.protagonist_name} 有）：")
        lines.extend(_field_lines(save.only_fields))
        lines.append("")

    # ---- 当前值 ----
    lines.append("### 当前值")
    lines.append("")
    lines.extend(_current_value_lines(save))
    lines.append("")

    # ---- 输出契约 ----
    lines.append("### 每轮必须输出状态块")
    lines.append("")
    lines.append("叙述正文写完之后，**另起一行**输出且只输出这样一段：")
    lines.append("")
    lines.append(STATE_OPEN)
    lines.append(_sample_patch(save))
    lines.append(STATE_CLOSE)
    lines.append("")
    lines.append("规则：")
    lines.append("- 写**这一轮结束时**、本轮叙述里涉及到的字段的值。"
                 "拿不准某个人物有没有变化，就把他一并写上——"
                 "**重复写没有代价，漏写才有代价**。")
    lines.append('- 本轮谁都没被触及，就输出 {"state": {}, "only": {}}。')
    lines.append("- 人物名只能用上面「当前值」里列出的名字；字段名只能用上面定义过的。")
    lines.append("- 数值型给数字（可以带小数）；词条型给一个不超过 12 字的短语。")
    lines.append("- 这个块会被系统取走，**玩家看不到**。所以块里不要写叙述，"
                 "块外也不要解释它。")
    lines.append("- 状态块必须是整段回复的最后内容，之后不要再写任何字。")
    lines.append("- 主角专属的数值字段代表资源与状态。只要这一轮有消耗、收支、"
                 "劳损或恢复，就要如实增减，**不要让它一直停在初值**。")
    lines.append("")
    lines.append("注意：状态栏反映的是**剧情已经发生的结果**，不是你打算发生的事。"
                 "主角的状态由主角的行为决定，你只负责如实记录后果。")
    return "\n".join(lines)


def build_system_prompt(save: LegendSave) -> str:
    """把存档里的三段设定装配成 system prompt。

    顺序有讲究：先立规矩（谁演谁、绝对不能做什么），再给设定。
    反过来的话，模型读完设定已经进入「扮演」状态，后面再讲约束就容易被忽略。
    """
    npcs = save.npc_objects()
    npc_block = "\n".join(f"  - {n.to_card()}" for n in npcs) if npcs else "  - （本局没有配角）"
    style = STYLE_HINTS.get(save.style, STYLE_HINTS["classic"])
    # 状态栏段：是设定也是**输出契约**（没有定义字段时是空串，旧档 prompt 与上线前逐字相同）。
    # ★它**不**排在最后——最后留给了「文面格式」。真链路实测（带状态栏的档）：
    #   状态段排在最末时，模型只顾状态块与叙述，人物的动作与神情一个括号都不用，
    #   三类排版糊成一色。状态块漏了有 extract_state_patch 兜底，文面标记漏了没有兜底，
    #   所以把最近的位置让给文面格式。
    _state = build_state_block(save)
    state_block = f"\n\n{_state}" if _state else ""
    # 状态块是「每次回复都要带」的硬要求，所以也放进铁律（靠前，显著性高）。
    # 没有状态栏时是空串——铁律段与上线前逐字相同。
    state_rule = ""
    if save.has_state():
        state_rule = ("\n4. 每次回复的**最后**必须附上状态块（格式见文末「状态栏」）。"
                      "这是回复的一部分，漏掉即为不完整。")

    return f"""你是一款文字剧情游戏的**叙述者**，同时扮演这个世界里除主角以外的所有人。

## 铁律（违反即出戏）

1. **主角由玩家扮演，你绝不能替主角做决定、说话或行动。**
   你只描写「主角这么做之后，世界发生了什么反应」。
   例如玩家说「我推开门」，你写门后是什么、谁在看过来、空气里有什么味道；
   但**不要写**「我深吸一口气走进房间」这种主角自己的动作与心理。
   你能写的只有：环境、事件、以及**配角**的言行。
2. 不要跳出剧情。禁止出现「作为 AI」「根据设定」「以下是」这类元叙述，
   也不要总结、不要提问、不要提示玩家该做什么。
3. 不引用现实世界的资料，不做考据，不标注任何来源——这是虚构叙事，不是答疑。{state_rule}

## 世界观

{save.world}

## 主角（玩家扮演）

{save.protagonist_name}{f"：{save.protagonist_desc}" if save.protagonist_desc else ""}

## 配角（你来扮演）

{npc_block}

## 叙事要求

- {style}
- 每次回复推进**一小步**剧情，写 150-350 字。不要一次把整场戏演完，
  也不要原地打转。
- 结尾留一个**自然的钩子**（一个新情况、一句追问、一个逼近的危险），
  让玩家知道可以接什么，但**不要直接问玩家要选什么**。
- 玩家给出的信息如果和世界观冲突，以玩家为准，顺势圆过去，不要纠正他。
{state_block}

## 文面格式（决定读者看到的排版，务必遵守）

叙述里有三类内容，写法不同，读者会看到不同的颜色与排版：

1. **人物说的话** → **必须带说话人**，写成 `**配角名**：台词内容`，独立成行。
   名字不能省——读者要一眼看出谁在说话。整行不要再包一层引号。
   例：**苏娘**：城南杜家？这信我可不敢接。
2. **人物的神情与动作** → 用全角括号括起来，例如「（她眼皮都没抬）」。
   括号里只写举止、神态、语气，**不要**在里面写台词，也不要写环境。
   例：（她把声音压得极低，指尖在柜台上一叩）
3. **环境、气氛、背景交代** → 直接写，不加任何标记。

★最容易写错的一种：动作与台词连着写时，**不要**写成「（动作）台词」放在同一行。
那样读者分不清哪句是说出来的、哪句只是做的，排版也分不了色。正确写法是
动作进括号、台词照旧另起一行带名字：

    （她把声音压得极低）
    **苏娘**：城南杜家？这信我可不敢接。

### 照这个例子写（颜色是系统按上面三类自动上的，你只管写对标记）

暮色压着西市的屋脊漫下来，酒肆门帘半垂，长安城最闹的吆喝声被隔在青布外面。

（苏娘正把一只豁口的陶碗翻过来扣在柜台上，目光扫过店里几个零散客人）

**苏娘**：这个时辰，买酒的多是收摊的贩夫，你倒有空来坐？

（她和李寻的目光在灯影里碰了一下，又不动声色地错开）

★这三段分别就是「环境」「动作」「台词」。**开局那一段也必须这样写**——
后面每一轮你都会照着自己上一轮的写法来，开局写平了，整场戏就都平了。

三类可以交替出现。判断标准是「这句在写谁」：写天气、陈设、人群、声响就直接写；
写某个人怎么动、什么神情就加全角括号；某人开口说话就带名字独立成行。"""


def _state_reminder(save: LegendSave) -> str:
    """贴在本轮 user 消息末尾的状态块提醒。

    ★为什么光靠 system prompt 不够（真链路实测）：把输出契约只写在 system 段时，
    3 轮里只有 1 轮真的吐了状态块——推理模型读完长长的设定后，
    注意力全在「写一段好叙述」上，格式要求被稀释掉了。
    把同一句话放到**离生成点最近**的地方（最后一条 user 消息），命中率就上来了。
    没有定义字段时返回空串，旧档的 messages 与上线前逐字相同。
    """
    if not save.has_state():
        return ""
    return ("\n（本轮回复的**最后**必须附上状态块：" + STATE_OPEN
            + ' {"state": {…}, "only": {…}} ' + STATE_CLOSE
            + "；写上本轮涉及到的字段在这一轮结束后的值，拿不准就一并写上）")


def _format_reminder() -> str:
    """贴在本轮 user 消息末尾的文面提醒（三类内容分开写）。

    ★为什么必须要有（真链路实测，两遍对照）：文面契约本来只写在 system 段的
    「叙事要求」里。**没有状态栏**的档跑下来，模型用全角括号标动作很足
    （每轮 4-5 组）；**有状态栏**的档跑下来，几乎一个括号都不用，动作全被
    当成环境叙述写平了。原因跟状态块当初漏块是同一回事：状态栏段排在
    system 的**最后**、还带铁律级强调，把前面的文面契约挤没了。

    状态块漏了有 extract_state_patch 兜底，文面标记漏了**没有兜底**——读者
    看到的就是三类糊成一色。所以把文面提醒也挪到离生成点最近的地方。
    对旧档同样生效：这条契约与有没有状态栏无关。
    """
    return ("\n（文面要求：台词一律带名字、独立成行，写成 `**名字**：台词`；"
            "人物的神情与动作用全角括号括起来，例如（她眼皮都没抬）；"
            "环境与背景直接写、不要加括号；"
            "不要把台词并进括号写成「（动作）台词」）")


def build_messages(save: LegendSave, action: str | None) -> list:
    """装配本轮请求的 messages。

    action 为 None 表示开局（模型自己起头）；否则是玩家的行动。
    历史只取最近的若干条——存档侧已经按轮裁剪过，这里不再二次裁。
    """
    msgs: list = [SystemMessage(content=build_system_prompt(save))]
    for t in save.turns:
        role = t.get("role")
        content = (t.get("content") or "").strip()
        # ★必须用 strip 后的判空：`if not content` 对 "  "（全空白）判为真，
        # 会把空白记录当成一条真实历史注入给模型（测试抓到过）。
        if not content:
            continue
        if role == "user":
            # 给玩家的输入加上角色前缀，强化「这是主角在行动」的信号，
            # 避免模型把玩家的话当成旁白或自己的台词。
            msgs.append(HumanMessage(content=f"（{save.protagonist_name}的行动）{content}"))
        elif role == "narrator":
            msgs.append(AIMessage(content=content))

    if action is None:
        msgs.append(HumanMessage(
            content="（游戏开始。请用一段叙述开场：交代场景与在场的人，"
                    f"并在结尾给 {save.protagonist_name} 一个自然的下手处。"
                    "记住不要替主角做决定。）" + _state_reminder(save)
                    + _format_reminder()
        ))
    else:
        msgs.append(HumanMessage(
            content=f"（{save.protagonist_name}的行动）{action}"
                    + _state_reminder(save) + _format_reminder()
        ))
    return msgs


# 首 token 超时（秒）。
#
# ★必须比 `settings.llm_ttft_timeout`（全局 6s）宽得多，实测踩过：
#   全局那个 6s 的取值依据是 config 注释里写的「正常首 token 实测 1-3s」——
#   但那是 `deepseek-chat`（非推理模型）时代的测定。现在跑的是
#   `deepseek-v4-flash`，**推理模型，思考发生在首 token 之前**，
#   首 token 天然要十几秒。用 6s 的结果是真链路实测里每轮都触发 1-3 次
#   「首 token 超时」重试，第 3 轮更是 3 次全超时、直接返回空白给用户。
#
# 为什么传奇比别的路径更需要放宽：传奇是**纯生成**，前面没有检索/重排占用时间，
# 模型没有「热身」过程，首 token 来得最晚，被 6s 误杀的概率最高。
#
# 取 20s：空壳场景本身约 13s 就返回空内容，会由「整次流零 token」那条判据接住，
# 不会因为放宽而变慢；而真实生成的思考时间（约 10-15s）能安全落进来。
LEGEND_TTFT_TIMEOUT = 20.0


async def stream_turn(save: LegendSave, action: str | None) -> AsyncIterator[str]:
    """生成一轮叙述，逐 token 产出。

    调用方负责把产出拼起来写回 `save.turns` 并落盘——本函数只负责生成，
    不碰存储（这样单测可以只验生成，不落盘）。
    """
    llm = get_chat_llm(with_fallback=True)
    messages = build_messages(save, action)
    async for token in astream_nonempty(
        llm, messages, ttft_timeout=LEGEND_TTFT_TIMEOUT
    ):
        yield token


def legend_model_label() -> str:
    """给前端显示用的模型名（只读提示，不给用户改）。"""
    return getattr(settings, "llm_model", "") or "默认模型"


# ============================================================
# 状态块：流式摘除 + 解析
# ============================================================

def _partial_marker_len(buf: str, marker: str) -> int:
    """buf 末尾有多长是 marker 的前缀（这段必须先扣住不吐，等下一个 token 补齐）。

    不扣住的话，跨 token 边界的 `<<<STATE` 会被当成正文吐出去一半，
    玩家就会在叙述里看到「<<<ST」这种东西。
    """
    for k in range(min(len(marker) - 1, len(buf)), 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0


class StatePatchFilter:
    """把 `<<<STATE … STATE>>>` 从流里摘掉——玩家永远看不到它。

    为什么在**流式**里做、而不是等流完再切：前端是边收边渲染的。
    等流完再切的话，状态块会在界面上闪一下再消失。
    """

    def __init__(self) -> None:
        self._buf = ""          # 还没决定去留的尾巴
        self._in_block = False
        self._cur = ""          # 当前块已吃进的内容
        self.blocks: list[str] = []

    def feed(self, token: str) -> str:
        """吃一个 token，返回其中应展示给玩家的部分（可能为空串）。"""
        self._buf += token
        out: list[str] = []
        while self._buf:
            if self._in_block:
                j = self._buf.find(STATE_CLOSE)
                if j >= 0:
                    self._cur += self._buf[:j]
                    self.blocks.append(self._cur)
                    self._cur = ""
                    self._buf = self._buf[j + len(STATE_CLOSE):]
                    self._in_block = False
                    continue
                hold = _partial_marker_len(self._buf, STATE_CLOSE)
                if hold:
                    self._cur += self._buf[:-hold]
                    self._buf = self._buf[-hold:]
                else:
                    self._cur += self._buf
                    self._buf = ""
                break
            i = self._buf.find(STATE_OPEN)
            if i >= 0:
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(STATE_OPEN):]
                self._in_block = True
                continue
            hold = _partial_marker_len(self._buf, STATE_OPEN)
            if hold:
                out.append(self._buf[:-hold])
                self._buf = self._buf[-hold:]
            else:
                out.append(self._buf)
                self._buf = ""
            break
        return "".join(out)

    def finish(self) -> str:
        """流结束。返回还欠着玩家、应当补发出去的正文。

        ★块没闭合（模型没按格式写）时，把吃进去的内容**原样还给玩家**——
        宁可显示一段带标记的怪文本，也不能因为格式不合规就把模型写的内容吞掉。
        """
        tail = (STATE_OPEN + self._cur + self._buf) if self._in_block else self._buf
        self._buf = ""
        self._cur = ""
        self._in_block = False
        return tail

    def patch(self) -> dict | None:
        """把摘下来的块合成一个补丁。多个块按出现先后覆盖。"""
        merged: dict = {}
        for raw in self.blocks:
            got = parse_state_patch(raw)
            if not isinstance(got, dict):
                continue
            for key in ("state", "only"):
                part = got.get(key)
                if not isinstance(part, dict):
                    continue
                target = merged.setdefault(key, {})
                for k, v in part.items():
                    if isinstance(v, dict) and isinstance(target.get(k), dict):
                        target[k].update(v)
                    else:
                        target[k] = v
        return merged or None


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def parse_state_patch(text: str) -> dict | None:
    """从状态块内容里抠出 JSON。抠不出来返回 None（当作这轮没有更新）。

    模型常见两种走样：① 外面裹了 ``` 代码围栏；② 前后多写了一句解释。
    先直接 loads，失败就取最外层的 {...}，再失败就放弃——
    宁可这轮不更新状态，也不能让脏数据进存档。
    """
    body = (text or "").strip()
    if not body:
        return None
    body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
    body = re.sub(r"\s*```$", "", body).strip()
    try:
        got = json.loads(body)
    except (ValueError, TypeError):
        m = _JSON_OBJ_RE.search(body)
        if not m:
            return None
        try:
            got = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
    return got if isinstance(got, dict) else None


# ============================================================
# 状态抽取（正文没带状态块时的兜底）
# ============================================================
# ★为什么必须有这一道（真链路实测三次）：让模型在**同一段回复**里既写好叙述
# 又附上状态块，命中率只有一半上下——跑三遍分别是 1/3、3/3、1/3。
# 把契约写进铁律、再在最后一条 user 消息里提醒，也只是把命中率抬上来，压不到 1。
# 而状态栏是玩家一直盯着的：漏一轮就「卡住」了，不能靠 prompt 赌运气。
#
# 所以分两条路走：
#   正文里带了块  → 快路径，零额外开销（多数情况下走这条）
#   正文里没带块  → 用一次**极小的结构化调用**，只干「从叙述里抽变化」这一件事
# 抽取这步只有一个目标、且关掉思考（reasoning_effort=none），命中接近确定。
#
# 代价：漏块的那一轮多一次调用（约 5-8s）。换来的是状态栏不再随机停摆。

STATE_EXTRACT_PROMPT = """下面是一段文字剧情的叙述。你要做的是：把「状态栏」更新到**这一段叙述之后**的样子。

## 字段定义
{defs}

## 这一轮之前的值
{values}

## 本轮叙述
{narration}

只输出 JSON，不要任何解释、不要代码围栏。格式：
{{"state": {{"人物名": {{"字段名": 新值}}}}, "only": {{"字段名": 新值}}}}

规则：
- 对**叙述里涉及到**的每个人物，写出他这一轮结束后的字段值（不用判断有没有变，
  直接写现在的值）。叙述里完全没出现、也没被提及的人物不用写。
- 拿不准某个人物有没有被影响到，**宁可写上**——写重复没有代价，漏写才有代价。
- 人物名只能用上面定义里列出的；字段名只能用上面定义过的。
- 数值型给数字；词条型给不超过 12 字的短语。
- 这一段确实谁都没被触及，才输出 {{"state": {{}}, "only": {{}}}}。"""


async def extract_state_patch(save: LegendSave, narration: str) -> dict | None:
    """从一段叙述里抽状态补丁（正文没带状态块的兜底路径）。

    没有定义字段、叙述为空、或抽取失败时一律返回 None——调用方据此保持原值。
    **不抛异常**：状态栏更新失败不该让整轮对话失败。
    """
    if not save.has_state() or not (narration or "").strip():
        return None

    defs: list[str] = []
    if save.state_fields:
        defs.append("通用（在场每个人物各有一份）：")
        defs.extend(_field_lines(save.state_fields))
    if save.only_fields:
        defs.append(f"仅主角（只有 {save.protagonist_name} 有）：")
        defs.extend(_field_lines(save.only_fields))

    prompt = STATE_EXTRACT_PROMPT.format(
        defs="\n".join(defs) or "（无）",
        values="\n".join(_current_value_lines(save)) or "（无）",
        narration=narration.strip()[:4000],
    )
    try:
        # 结构化抽取必须关思考：推理模型的 thinking 会把 JSON 的额度吃掉导致截断
        llm = get_chat_llm(
            with_fallback=True, reasoning_effort="none",
            temperature=0.1, max_tokens=800,
        )
        resp = await ainvoke_nonempty(llm, [HumanMessage(content=prompt)])
    except Exception as e:
        print(f"[legend] 状态抽取调用失败: {e}")
        return None
    return parse_state_patch(getattr(resp, "content", "") or "")
