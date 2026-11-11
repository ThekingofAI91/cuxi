"""
prompt_builder.py — 角色提示词装配器（酒馆式：角色卡 + 世界书）

思路借鉴 SillyTavern：让角色"像本人"靠的不是检索更多资料，而是
  ① 结构化的角色卡（性格 / 场景 / 开场白 / 采样手感）
  ② 几轮示例对话（few-shot，模型直接模仿语气与节奏）
  ③ 世界书 Lorebook（关键词命中才注入的设定片段）
  ④ 后历史指令（对话历史之后再钉一次角色，防长对话漂移）

这些构件都是**零检索、零额外 LLM 调用**的：拼字符串而已。
因此它既提升像真度，又不增加延迟 —— 与本项目"砍串行调用"的优化方向一致。

两区用法差异（同一套机制，不同内容）——这是本项目的自有设计，酒馆没有：
- 教育区：世界书 = 核心概念的权威界定（降幻觉、提专业度）；
          示例对话示范"如何严谨作答"的节奏与引用方式；
          后历史指令强调判断先行、术语翻译、动作描写。
- 娱乐区：世界书 = 黑话与老梗（保人味）；示例对话示范口语化短句；
          后历史指令强调口语短句与"零引用痕迹"。

装配顺序经过设计：示例对话放在靠后位置（对生成影响最强），
分区语气指令放最后（最新鲜，压住格式）；
后历史指令单独装配，插在对话历史之后（对生成影响最强的位置）。
"""

from __future__ import annotations

from typing import Any

from scenes.persona_chat.skill_loader import resolve_persona_sections

# 娱乐区专属语气指令：压掉 AI 味，强调查像真人微信聊天
_ENTERTAINMENT_VOICE_DIRECTIVE = """【娱乐区口吻要求】（不要像 AI / 助手）
- 像真人微信聊天：口语化、短句、可带语气词，别写长篇大论
- 可以甩金句、抖机灵、反问、吐槽，但别油滑、别说教
- 严禁任何结构化格式：不要出现"总结""需要注意的是""综上所述""第一/第二/第三"等词
- 严禁 markdown 小标题、列表符号、引用标注、出处说明、置信度
- 严禁以"作为 AI""基于资料"等第三方视角说话，全程第一人称
- 回答尽量短（一到两段就好），说人话，像活人在跟你聊"""

# 世界书抬头：两区措辞不同，教育区强调"以词条为准"，娱乐区强调"按你的说法接"
_LOREBOOK_HEADER_EDU = (
    "【核心概念词条（用户问及下列概念时，以这里的界定为准；"
    "这是你的原意，不要换成通用教科书说法，可自然点明出处）】"
)
_LOREBOOK_HEADER_ENT = (
    "【你的黑话与老梗（用户提到下列词时，按你自己的说法接住，"
    "不要解释成普通话，也不要解释成中立客观的表述）】"
)

_EXAMPLE_HEADER = (
    "【对话示例·严格模仿其中的语气、节奏与句式，"
    "严禁照抄内容、严禁把示例里的事情当成真实发生过的】"
)

_LOREBOOK_MAX_HITS = 4   # 单轮最多注入词条数，防止撑爆 prompt

# 世界书关键词的扫描窗口（轮数）：只扫当前消息时，"我们刚才聊的那个XX"这类
# 指代命不中词条，长对话里设定会静默失效；扫最近几轮才能接住这类引用。
_LOREBOOK_SCAN_TURNS = 3


def compose_lorebook_scan_text(query: str = "", history: list[tuple[str, str]] | None = None) -> str:
    """拼装世界书关键词的扫描文本：最近几轮对话 + 本轮消息。

    只做字符串拼接（零 LLM 调用）；调用方在装配提示词前调用一次，
    把结果传给 build_character_prompt / build_post_history_directive 的 scan_text。
    """
    parts: list[str] = []
    for q, a in (history or [])[-_LOREBOOK_SCAN_TURNS:]:
        if q:
            parts.append(str(q))
        if a:
            parts.append(str(a))
    if query:
        parts.append(query)
    return "\n".join(parts)

# 世界书已简化为「关键词 → 内容」两字段：条目不再带注入位置、常驻开关、
# 排序权重与副关键词。这些能力内置角色从未使用，维护成本却要一直付。
# 自建角色若从旧角色卡导入了这些字段，解析时静默丢弃（见 skill_loader._normalize_lorebook）。


def match_lorebook(
    entries: Any,
    text: str,
    limit: int = _LOREBOOK_MAX_HITS,
) -> list[dict]:
    """挑出本轮该注入的词条：关键词在扫描文本里出现即命中，按原顺序取前 limit 条。

    去重按关键词，先到先得——重复词条不重复占注入名额。
    不做排序：词条的先后就是人设里写的顺序，作者排的就是优先级。
    """
    if not entries or not text:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        keyword = (entry.get("keyword") or "").strip()
        content = (entry.get("content") or "").strip()
        if not keyword or not content or keyword in seen:
            continue
        if keyword not in text:
            continue
        seen.add(keyword)
        out.append({"keyword": keyword, "content": content})
        if limit and len(out) >= limit:
            break
    return out


def _format_lorebook(hits: list[dict], zone: str) -> str:
    header = _LOREBOOK_HEADER_ENT if zone == "entertainment" else _LOREBOOK_HEADER_EDU
    body = "\n".join(f"· {h['keyword']}：{h['content']}" for h in hits)
    return f"{header}\n{body}"


def has_tavern_components(character: Any) -> bool:
    """
    该角色是否已具备酒馆式构件（世界书或示例对话）。

    用于诊断日志：娱乐区但缺构件时只能靠背景检索撑住，属于"弱角色卡"。
    """
    return bool(getattr(character, "lorebook", None) or getattr(character, "mes_example", ""))


def build_character_prompt(
    character: Any,
    query: str = "",
    *,
    scan_text: str = "",
    include_lorebook: bool = True,
    include_examples: bool = True,
) -> str:
    """
    把「人设主体 + 性格 + 场景 + 世界书命中 + 示例对话 + 分区语气」拼成一个提示词。

    query 用于世界书关键词匹配；scan_text 是额外的扫描文本（最近几轮对话），
    让"我们刚才聊的那个XX"类指代也能命中词条。两者都不含词条内容，只是拼接字符串。
    """
    parts: list[str] = []

    # 五块人设统一从技能包取（data/persona_chat/skills/{id}.md 的五个分节），
    # 逐块回退到 CharacterDef 字段——自建角色没有技能包，全程走回退。
    # 见 skill_loader.resolve_persona_sections：人设是内容不是逻辑，外置后可独立改动。
    sections = resolve_persona_sections(character)

    if sections.role_prompt:
        parts.append(sections.role_prompt)

    if sections.personality:
        parts.append("【性格特质】\n" + sections.personality)

    if sections.scenario:
        parts.append("【当前场景】\n" + sections.scenario)

    if include_lorebook:
        scan = "\n".join(t for t in (scan_text, query) if t)
        hits = match_lorebook(sections.lorebook, scan)
        if hits:
            parts.append(_format_lorebook(hits, getattr(character, "zone", "education")))

    if include_examples and sections.mes_example:
        parts.append(f"{_EXAMPLE_HEADER}\n{sections.mes_example}")

    if getattr(character, "zone", "education") == "entertainment":
        parts.append(_ENTERTAINMENT_VOICE_DIRECTIVE)

    return "\n\n".join(parts).strip()


def build_post_history_directive(character: Any, query: str = "", *, scan_text: str = "") -> str:
    """
    后历史指令（对应 SillyTavern 的 Author's Note / post-history instructions）。

    为什么必须存在：模型对越靠后的内容越敏感。人设指令全堆在最前面时，
    对话历史一长就会被稀释，聊十几轮后角色开始漂移、AI 味回来。
    在历史之后再钉一次「你是谁、该怎么说话、什么绝对不能说」，成本只有几十 token。

    两区内容不同（保留本项目的分区特色）：
    - 教育区：判断先行、术语翻译、动作描写、可溯源
    - 娱乐区：口语短句、零引用痕迹、禁结构化
    """
    name = (getattr(character, "name", "") or "").strip() or "这个人物"
    zone = getattr(character, "zone", "education")

    if zone == "entertainment":
        directive = f"""【落笔前再确认一遍】你是{name}本人，正在跟人微信聊天，不是 AI 助手。
- 口语、短句，一到两段就收，别写长篇大论
- 禁止结构化格式：markdown 小标题、列表符号、"第一/第二/第三"、"总结""综上所述"
- 禁止任何引用痕迹：出处、章节名、资料名、"根据……"
- 禁止以第三方视角说话（"作为 AI""基于资料"）"""
    else:
        directive = f"""【落笔前再确认一遍】你是{name}本人，正与来访者当面交谈，不是 AI 助手。
- 先点明你的判断，再用大白话解释，最后给出可落地的做法
- 术语出现当场翻译成白话；引用自己的著作时自然说明出处
- 每次回答穿插 1~2 处 *动作* 描写，用"他"或自己的名字指代，不要用"你"
- 禁止"作为 AI""基于资料""总的来说""综上所述"这类助手腔"""

    parts = [directive]

    # 世界书只在人设之后注入一次（见 build_character_prompt）。
    # 原先还有一档「历史之后」的位置，随世界书简化为两字段一并删除——
    # 内置角色从未用过它，留着只是一条永远不会执行的分支。
    return "\n\n".join(parts).strip()
