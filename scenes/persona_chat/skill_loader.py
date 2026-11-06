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
