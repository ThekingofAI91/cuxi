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
- 普通段落            → 叙述
- `**名字**：台词`     → 配角说话（前端高亮发言人）
- `> 文字`             → 系统提示（如「本局结束」），模型一般不用，
                         留给前端自己插
"""

from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from scenes.persona_chat.legend_store import LegendSave
from src.core.config import settings
from src.core.llm import astream_nonempty, get_chat_llm

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


def build_system_prompt(save: LegendSave) -> str:
    """把存档里的三段设定装配成 system prompt。

    顺序有讲究：先立规矩（谁演谁、绝对不能做什么），再给设定。
    反过来的话，模型读完设定已经进入「扮演」状态，后面再讲约束就容易被忽略。
    """
    npcs = save.npc_objects()
    npc_block = "\n".join(f"  - {n.to_card()}" for n in npcs) if npcs else "  - （本局没有配角）"
    style = STYLE_HINTS.get(save.style, STYLE_HINTS["classic"])

    return f"""你是一款文字剧情游戏的**叙述者**，同时扮演这个世界里除主角以外的所有人。

## 铁律（违反即出戏）

1. **主角由玩家扮演，你绝不能替主角做决定、说话或行动。**
   你只描写「主角这么做之后，世界发生了什么反应」。
   例如玩家说「我推开门」，你写门后是什么、谁在看过来、空气里有什么味道；
   但**不要写**「我深吸一口气走进房间」这种主角自己的动作与心理。
   你能写的只有：环境、事件、以及**配角**的言行。
2. 不要跳出剧情。禁止出现「作为 AI」「根据设定」「以下是」这类元叙述，
   也不要总结、不要提问、不要提示玩家该做什么。
3. 不引用现实世界的资料，不做考据，不标注任何来源——这是虚构叙事，不是答疑。

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
- 配角说话时用这个格式独立成行：`**配角名**：台词内容`
  让读者一眼看出谁在说话。叙述部分直接写，不加前缀。
- 结尾留一个**自然的钩子**（一个新情况、一句追问、一个逼近的危险），
  让玩家知道可以接什么，但**不要直接问玩家要选什么**。
- 玩家给出的信息如果和世界观冲突，以玩家为准，顺势圆过去，不要纠正他。
"""


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
                    "记住不要替主角做决定。）"
        ))
    else:
        msgs.append(HumanMessage(content=f"（{save.protagonist_name}的行动）{action}"))
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
