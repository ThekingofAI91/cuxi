"""
skill_loader.py — 角色设定以 SKILL.md 形式外置，代码只负责装配

背景：角色人设原本以 Python 字符串常量写在 `characters/*.py` 里。人设是内容、
不是逻辑，改动它要动代码、要重新部署，也没法让非工程角色参与维护。这里把它搬成
Anthropic Agent Skills 风格的 Markdown 包：

    data/persona_chat/skills/
      jung.md
      adler.md
      wangyangming.md
      fengge.md
      zhangxuefeng.md
      legend-narrator.md

分工（与"Skill 管怎么做、RAG 管说的是否有依据"这条边界一致）：
- skill 正文 = 人设与文面契约（口吻、禁忌、知识边界、输出格式）
- 向量库 = 事实依据（可标 n 回溯到原书章节）

装配优先级：skill 文件存在则用其正文覆盖 `role_prompt`；不存在（如用户自建角色）
则回退到 `CharacterDef.role_prompt`。回退路径保证既有行为零变化。

缓存：按 mtime 失效，改文件不需要重启进程；同一请求内多次读取只有一次磁盘 IO。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 技能包根目录（相对仓库根）。不在 settings 里加开关：目录缺失即等同于"未启用"，
# 全部走回退路径，行为与改造前一致。
SKILL_DIR = Path("data/persona_chat/skills")

# 传奇叙述者不走角色装配，单独按名字取
LEGEND_SKILL_NAME = "legend-narrator"

_lock = threading.Lock()
# {skill_name: (mtime, SkillPack)}
_cache: dict[str, tuple[float, "SkillPack"]] = {}


@dataclass
class SkillPack:
    """一个技能包：frontmatter 元信息 + 正文"""

    name: str
    description: str = ""
    zone: str = ""
    body: str = ""
    path: Optional[Path] = None
    meta: dict = field(default_factory=dict)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """拆出 YAML frontmatter。不引第三方依赖，只认本仓库实际用到的三种写法：

        key: value
        key: |
          多行
        key: >
          多行（折叠成一行）

    解析失败一律返回空 meta + 全文，让调用方走回退，绝不因格式问题抛异常。
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if len(lines) < 2:
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text

    meta: dict = {}
    key: Optional[str] = None
    buf: list[str] = []
    folding = False  # 当前块是否用 > 折叠

    def flush() -> None:
        nonlocal key, buf, folding
        if key is not None:
            if folding:
                meta[key] = " ".join(s.strip() for s in buf if s.strip())
            else:
                meta[key] = "\n".join(buf).strip()
        key, buf, folding = None, [], False

    for raw in lines[1:end]:
        if not raw.strip():
            if key is not None:
                buf.append("")
            continue
        stripped = raw.strip()
        if raw[:1] not in (" ", "\t") and ":" in stripped:
            flush()
            k, _, v = stripped.partition(":")
            k = k.strip()
            v = v.strip()
            if v in ("|", ">"):
                key = k
                folding = v == ">"
                buf = []
            elif v:
                meta[k] = v.strip('"').strip("'")
            else:
                key = k
                buf = []
        elif key is not None:
            buf.append(raw)
    flush()

    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return meta, body


def _read_pack(path: Path) -> Optional[SkillPack]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:  # 编码错误/权限问题都不该炸主链路
        print(f"[Skill] 读取失败 {path}: {e}")
        return None
    meta, body = _split_frontmatter(text)
    if not body.strip():
        return None
    return SkillPack(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        zone=str(meta.get("zone") or ""),
        body=body.strip(),
        path=path,
        meta=meta,
    )


def load_skill(skill_name: str) -> Optional[SkillPack]:
    """按名字加载技能包；不存在或解析失败返回 None。"""
    if not skill_name:
        return None
    path = SKILL_DIR / f"{skill_name}.md"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    with _lock:
        cached = _cache.get(skill_name)
        if cached and cached[0] == mtime:
            return cached[1]

    pack = _read_pack(path)
    if pack is None:
        return None
    with _lock:
        _cache[skill_name] = (mtime, pack)
    return pack


def list_skills() -> list[SkillPack]:
    """列出全部技能包（诊断与文档用），按文件名排序。"""
    if not SKILL_DIR.is_dir():
        return []
    out: list[SkillPack] = []
    for path in sorted(SKILL_DIR.glob("*.md")):
        pack = load_skill(path.stem)
        if pack is not None:
            out.append(pack)
    return out


def resolve_role_prompt(character: Any) -> str:
    """取角色的最终人设文本。

    优先技能包正文；缺失时回退 `CharacterDef.role_prompt`。
    自建角色（is_custom=True）没有技能包，天然走回退。
    """
    char_id = str(getattr(character, "id", "") or "")
    if char_id and not getattr(character, "is_custom", False):
        pack = load_skill(char_id)
        if pack is not None:
            return pack.body
        print(f"[Skill] {char_id} 无技能包，回退内联 role_prompt")
    return (getattr(character, "role_prompt", "") or "").strip()


# ============================================================
# 分节人设：人设 / 性格特质 / 当前场景 / 核心概念词条 / 对话示例
# ============================================================
# 这五块本来分散在 skill 正文与 CharacterDef 的四个字段里，改一句话要动两个文件。
# 现在统一以固定二级标题写在 skill 里，代码只负责按标题取值与补格式标记。
#
# 只认这五个标题：正文里若出现其他 `##`（例如某个人设自带小节），按普通正文保留，
# 不会被误当成下一个分节的开始。

SECTION_ROLE = "人设"
SECTION_PERSONALITY = "性格特质"
SECTION_SCENARIO = "当前场景"
SECTION_LOREBOOK = "核心概念词条"
SECTION_EXAMPLE = "对话示例"

PERSONA_SECTIONS = (
    SECTION_ROLE,
    SECTION_PERSONALITY,
    SECTION_SCENARIO,
    SECTION_LOREBOOK,
    SECTION_EXAMPLE,
)

_SECTION_RE = re.compile(
    r"^##[ \t]+(" + "|".join(PERSONA_SECTIONS) + r")[ \t]*$",
    re.MULTILINE,
)


@dataclass
class PersonaSections:
    """一个角色的五块人设内容。空字符串表示该块缺失，由调用方决定是否跳过。"""

    role_prompt: str = ""
    personality: str = ""
    scenario: str = ""
    lorebook: list = field(default_factory=list)  # [{"keyword": str, "content": str}]
    mes_example: str = ""


def _parse_lorebook_block(block: str) -> list:
    """把「- 关键词：内容」列表解析成词条。

    只按**第一个**冒号切分——内容里常有冒号（「他说：……」），按最后一个切会把关键词吃掉。
    中文全角冒号与英文半角冒号都接受。
    """
    out: list = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line.startswith(("- ", "* ")):
            continue
        body = line[2:].strip()
        for sep in ("：", ":"):
            idx = body.find(sep)
            if idx > 0:
                keyword = body[:idx].strip()
                content = body[idx + 1:].strip()
                if keyword and content:
                    out.append({"keyword": keyword, "content": content})
                break
    return out


def parse_persona_sections(body: str) -> PersonaSections:
    """按五个固定标题把技能包正文拆成结构化人设。

    没有标题的旧格式（整篇都是人设正文）也能用：整篇归入 `role_prompt`。
    """
    if not body or not body.strip():
        return PersonaSections()

    hits = list(_SECTION_RE.finditer(body))
    if not hits:
        return PersonaSections(role_prompt=body.strip())

    sections: dict[str, str] = {}
    for i, m in enumerate(hits):
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        sections[m.group(1)] = body[start:end].strip()

    return PersonaSections(
        role_prompt=sections.get(SECTION_ROLE, ""),
        personality=sections.get(SECTION_PERSONALITY, ""),
        scenario=sections.get(SECTION_SCENARIO, ""),
        lorebook=_parse_lorebook_block(sections.get(SECTION_LOREBOOK, "")),
        mes_example=sections.get(SECTION_EXAMPLE, ""),
    )


def load_persona(character_id: str) -> Optional[PersonaSections]:
    """读技能包并解析成五块；技能包缺失或为空返回 None。"""
    pack = load_skill(character_id)
    if pack is None:
        return None
    parsed = parse_persona_sections(pack.body)
    if not any((parsed.role_prompt, parsed.personality, parsed.scenario,
                parsed.lorebook, parsed.mes_example)):
        return None
    return parsed


def resolve_persona_sections(character: Any) -> PersonaSections:
    """取角色最终的五块人设：优先技能包，缺失的回退 CharacterDef 字段。

    逐块回退而不是整体回退——技能包只写了人设、没写词条时，词条仍从字段取，
    这样迁移可以一个一个角色、一块一块地做，中途不会出现「人设有了但词条没了」。
    """
    char_id = str(getattr(character, "id", "") or "")
    from_skill = None
    if char_id and not getattr(character, "is_custom", False):
        from_skill = load_persona(char_id)

    def pick(skill_value: str, field_name: str) -> str:
        if skill_value and skill_value.strip():
            return skill_value.strip()
        return (getattr(character, field_name, "") or "").strip()

    if from_skill is None:
        return PersonaSections(
            role_prompt=(getattr(character, "role_prompt", "") or "").strip(),
            personality=(getattr(character, "personality", "") or "").strip(),
            scenario=(getattr(character, "scenario", "") or "").strip(),
            lorebook=_normalize_lorebook(getattr(character, "lorebook", None)),
            mes_example=(getattr(character, "mes_example", "") or "").strip(),
        )

    return PersonaSections(
        role_prompt=pick(from_skill.role_prompt, "role_prompt"),
        personality=pick(from_skill.personality, "personality"),
        scenario=pick(from_skill.scenario, "scenario"),
        lorebook=from_skill.lorebook or _normalize_lorebook(getattr(character, "lorebook", None)),
        mes_example=pick(from_skill.mes_example, "mes_example"),
    )


def _normalize_lorebook(raw: Any) -> list:
    """把任意历史格式的词条收敛成 [{"keyword","content"}]。

    旧格式还带 position / constant / order / secondary_keys / enabled，
    这些字段已随世界书简化一并删除；自建角色导入的卡如果还带，这里静默丢弃。
    """
    out: list = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        keyword = str(entry.get("keyword") or "").strip()
        content = str(entry.get("content") or "").strip()
        if keyword and content:
            out.append({"keyword": keyword, "content": content})
    return out


def legend_prompt_template() -> Optional[str]:
    """传奇叙述者的系统提示词模板（含 {world} 等占位符）。

    返回 None 表示技能包缺失，调用方应使用内置模板。
    """
    pack = load_skill(LEGEND_SKILL_NAME)
    return pack.body if pack is not None else None


def invalidate_cache() -> None:
    """清空缓存（测试与热更新用）。"""
    with _lock:
        _cache.clear()
