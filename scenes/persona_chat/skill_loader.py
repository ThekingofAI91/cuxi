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

角色技能是全部人设的唯一来源：人设、性格、场景、世界书、示例、开场白、口吻和
后历史指令都从同一个文件读取。内置角色放在 skills/，自建角色放在 custom/。
旧 JSON 的迁移由 custom_store 负责；这里不回退 Python 或 JSON 中的人设字段。

缓存：按 mtime 失效，改文件不需要重启进程；同一请求内多次读取只有一次磁盘 IO。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 内置技能根目录；自建角色技能与其元数据 JSON 放在同一 custom/ 目录。
SKILL_DIR = Path("data/persona_chat/skills")

# 传奇叙述者不走角色装配，单独按名字取
LEGEND_SKILL_NAME = "legend-narrator"

_lock = threading.Lock()
# 按绝对路径隔离缓存，避免不同目录的同名角色互相污染。
_cache: dict[str, tuple[tuple[int, int], "SkillPack"]] = {}
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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
    if not skill_name or not _SKILL_NAME_RE.fullmatch(skill_name):
        return None
    return _load_pack(SKILL_DIR / f"{skill_name}.md")


def _load_pack(path: Path) -> Optional[SkillPack]:
    key = str(path.resolve())
    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None

    with _lock:
        cached = _cache.get(key)
        if cached and cached[0] == stamp:
            return cached[1]

    pack = _read_pack(path)
    if pack is None:
        return None
    with _lock:
        _cache[key] = (stamp, pack)
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
    """只取技能包中的人设正文，不把其他分节误当成正文。"""
    return resolve_persona_sections(character).role_prompt


# ============================================================
# 分节人设：所有内容从一个文件解析，空节表示作者明确留空。
# ============================================================
SECTION_ROLE = "人设"
SECTION_PERSONALITY = "性格特质"
SECTION_SCENARIO = "当前场景"
SECTION_LOREBOOK = "世界书"
SECTION_EXAMPLE = "对话示例"
SECTION_FIRST_MES = "开场白"
SECTION_VOICE = "口吻要求"
SECTION_POST_HISTORY = "后历史指令"

PERSONA_SECTIONS = (
    SECTION_ROLE,
    SECTION_PERSONALITY,
    SECTION_SCENARIO,
    SECTION_LOREBOOK,
    SECTION_EXAMPLE,
    SECTION_FIRST_MES,
    SECTION_VOICE,
    SECTION_POST_HISTORY,
)

PERSONA_FIELDS = (
    "role_prompt", "personality", "scenario", "lorebook", "mes_example",
    "first_mes", "voice_directive", "post_history_instructions",
)
_SECTION_FIELDS = dict(zip(PERSONA_SECTIONS, PERSONA_FIELDS))
_SECTION_FIELDS["核心概念词条"] = "lorebook"  # 接受旧技能标题
_SECTION_RE = re.compile(r"^##[ \t]+(" + "|".join(_SECTION_FIELDS) + r")[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


@dataclass
class PersonaSections:
    """一个角色的完整人设。空字符串或空列表不会触发旧字段回填。"""

    role_prompt: str = ""
    personality: str = ""
    scenario: str = ""
    lorebook: list = field(default_factory=list)  # [{"keyword": str, "content": str}]
    mes_example: str = ""
    first_mes: str = ""
    voice_directive: str = ""
    post_history_instructions: str = ""


class PersonaSkillError(ValueError):
    """角色技能不可用；调用方应报错，不能悄悄换成通用角色。"""


def _unfence(block: str) -> str:
    lines = block.strip().splitlines()
    if len(lines) >= 2:
        match = _FENCE_RE.match(lines[0])
        if match and lines[-1].strip() == match.group(1):
            return "\n".join(lines[1:-1]).strip()
    return block.strip()


def _parse_lorebook_block(block: str) -> list:
    """把「- 关键词：内容」列表解析成词条。

    只按**第一个**冒号切分——内容里常有冒号（「他说：……」），按最后一个切会把关键词吃掉。
    中文全角冒号与英文半角冒号都接受。
    """
    block = _unfence(block)
    # 角色卡的关键词可带冒号，内容可有多行；JSON 块用于无损保存这些值。
    if block.startswith("["):
        try:
            raw = json.loads(block)
        except json.JSONDecodeError as e:
            raise PersonaSkillError(f"世界书 JSON 格式错误：{e.msg}") from e
        if not isinstance(raw, list):
            raise PersonaSkillError("世界书 JSON 必须是条目数组")
        return _normalize_lorebook(raw)
    out: list = []
    for raw in block.splitlines():
        line = raw.strip()
        match = re.match(r"^[-*]\s+(.+?)[：:](.*)$", line)
        if match:
            keyword, content = (part.strip() for part in match.groups())
            if keyword and content:
                out.append({"keyword": keyword, "content": content})
        elif out and (not line or raw[:1].isspace()):
            out[-1]["content"] += "\n" + line
    for entry in out:
        entry["content"] = entry["content"].strip()
    return out


def parse_persona_sections(body: str) -> PersonaSections:
    """解析固定二级标题；保留未知标题，忽略代码块内看似分节的标题。"""
    if not body or not body.strip():
        return PersonaSections()

    sections: dict[str, list[str]] = {}
    current = "role_prompt"
    found = False
    fence = ""
    for line in body.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence:
            sections.setdefault(current, []).append(line)
            if line.strip() == fence:
                fence = ""
            continue
        if fence_match:
            fence = fence_match.group(1)
        heading = _SECTION_RE.match(line) if not fence else None
        if heading:
            found = True
            current = _SECTION_FIELDS[heading.group(1)]
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    if not found:
        return PersonaSections(role_prompt=body.strip())
    values = {key: _unfence("\n".join(lines)) for key, lines in sections.items()}
    values["lorebook"] = _parse_lorebook_block(values.get("lorebook", ""))
    return PersonaSections(**values)


def load_persona(character_id: str, *, path: Path | None = None) -> Optional[PersonaSections]:
    """读完整角色技能；path 供自建角色使用。"""
    pack = _load_pack(path) if path is not None else load_skill(character_id)
    if pack is None:
        return None
    parsed = parse_persona_sections(pack.body)
    if not any(getattr(parsed, key) for key in PERSONA_FIELDS):
        return None
    return parsed


def resolve_persona_sections(character: Any) -> PersonaSections:
    """取唯一技能源。缺失或无人设正文时明确报错，绝不回退内联字段。"""
    char_id = str(getattr(character, "id", "") or "")
    if not char_id or not _SKILL_NAME_RE.fullmatch(char_id):
        raise PersonaSkillError("角色 id 无效，无法定位人设技能包")
    if getattr(character, "is_custom", False):
        from scenes.persona_chat.custom_store import CUSTOM_DIR
        path = CUSTOM_DIR / f"{char_id}.md"
        sections = load_persona(char_id, path=path)
    else:
        path = SKILL_DIR / f"{char_id}.md"
        sections = load_persona(char_id)
    if sections is None or not sections.role_prompt:
        raise PersonaSkillError(f"角色 {char_id} 的技能包缺失或缺少人设正文：{path}")
    return sections


def persona_from_fields(obj: Any) -> PersonaSections:
    """仅在创建/导入/旧数据迁移边界读取字段，不用于生成时回退。"""
    get = obj.get if isinstance(obj, dict) else lambda key, default: getattr(obj, key, default)
    values = {
        key: str(get(key, "") or "").strip()
        for key in PERSONA_FIELDS if key != "lorebook"
    }
    values["lorebook"] = _normalize_lorebook(get("lorebook", []))
    return PersonaSections(**values)


def default_persona_sections(zone: str, name: str) -> PersonaSections:
    """创建时复制技能模板；之后作者修改个人技能，不再被模板覆盖。"""
    zone = zone if zone in ("education", "entertainment") else "education"
    sections = load_persona(f"_defaults-{zone}")
    if sections is None:
        raise PersonaSkillError(f"缺少自建角色默认技能模板：_defaults-{zone}.md")
    values = {
        key: (getattr(sections, key).replace("{{char}}", name) if key != "lorebook"
              else [dict(entry) for entry in sections.lorebook])
        for key in PERSONA_FIELDS
    }
    return PersonaSections(**values)


def render_persona_skill(character: Any, sections: PersonaSections) -> str:
    """序列化完整技能；围栏保护导入正文里的同名标题，世界书支持多行内容。"""
    def fenced(value: str, language: str = "text") -> str:
        runs = re.findall(r"~+", value)
        fence = "~" * max(3, max((len(run) + 1 for run in runs), default=3))
        return f"{fence}{language}\n{value}\n{fence}"

    cid = str(getattr(character, "id", "") or "")
    name = str(getattr(character, "name", "") or cid).replace("\n", " ")
    zone = str(getattr(character, "zone", "education") or "education")
    header = f"---\nname: {cid}-persona\ndescription: |\n  {name}的完整人设技能\nzone: {zone}\n---"
    blocks = [header]
    for title, key in zip(PERSONA_SECTIONS, PERSONA_FIELDS):
        value = getattr(sections, key)
        if key == "lorebook":
            body = fenced(json.dumps(value, ensure_ascii=False, indent=2), "json")
        elif value:
            body = fenced(value)
        else:
            body = ""
        blocks.append(f"## {title}\n\n{body}".rstrip())
    return "\n\n".join(blocks) + "\n"


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
