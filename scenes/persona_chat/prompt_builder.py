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


def _format_lorebook(hits: list[dict]) -> str:
    body = "\n".join(f"· {h['keyword']}：{h['content']}" for h in hits)
    return f"【核心概念词条】\n{body}"


def has_tavern_components(character: Any) -> bool:
    """
    该角色是否已具备酒馆式构件（世界书或示例对话）。

    用于诊断日志：娱乐区但缺构件时只能靠背景检索撑住，属于"弱角色卡"。
    """
    sections = resolve_persona_sections(character)
    return bool(sections.lorebook or sections.mes_example)


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

    # 内置与自建角色都只从技能包取内容；缺失技能由加载器明确报错。
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
            parts.append(_format_lorebook(hits))

    if include_examples and sections.mes_example:
        parts.append(f"{_EXAMPLE_HEADER}\n{sections.mes_example}")

    if sections.voice_directive:
        parts.append(sections.voice_directive)

    return "\n\n".join(parts).strip()


def build_post_history_directive(character: Any, query: str = "", *, scan_text: str = "") -> str:
    """返回技能中的后历史指令；调用方负责把它放在历史之后。

    query / scan_text 保留调用兼容性；词条仅在 build_character_prompt 中注入。
    """
    return resolve_persona_sections(character).post_history_instructions
