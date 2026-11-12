"""
custom_store.py — 用户自建角色持久化

用户在前端提交「人物设定 + 背景」创建的角色，称为"自建角色"。
内置角色的技能包随仓库分发；自建角色的人设写入 custom/<id>.md，
custom/<id>.json 只保存展示信息、检索集合等元数据。

本模块负责：
- load_custom_characters()：启动时把磁盘上的自建角色加载回 CharacterDef 并合并进场景配置
- save_custom_character()：把自建角色序列化落盘
- delete_custom_character_file()：删除落盘文件
- generate_custom_id()：从名字生成一个合法且不冲突的角色 id
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import uuid
from dataclasses import replace
from pathlib import Path

from scenes.persona_chat.models import CharacterDef

# 自建角色落盘目录（相对项目根；与内置角色 data/persona_chat/<id>/ 区分开）
CUSTOM_DIR = Path("data/persona_chat/custom")

# 角色 id 白名单：仅允许小写字母/数字/下划线，避免注入与路径穿越
_ID_PATTERN = re.compile(r"^[a-z0-9_]{2,40}$")

# 写盘锁：多请求并发创建/删除时串行，避免同文件互相覆盖
_LOCK = threading.Lock()


def _safe_slug(name: str) -> str:
    """把任意名字转成 ascii 安全 slug（中文会变成空，调用方需兜底）"""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:24]


def custom_dir() -> Path:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    return CUSTOM_DIR


def is_valid_custom_id(cid: str) -> bool:
    return isinstance(cid, str) and bool(_ID_PATTERN.fullmatch(cid))


def _paths(cid: str) -> tuple[Path, Path, Path]:
    """所有读写都先校验 id 和实际路径，防止穿越或目录外符号链接。"""
    if not is_valid_custom_id(cid):
        raise ValueError("角色 id 仅允许 2-40 位小写字母、数字或下划线")
    directory = custom_dir().resolve()
    paths = tuple(directory / f"{cid}{suffix}" for suffix in (".json", ".md", ".json.bak"))
    if any(path.resolve().parent != directory for path in paths):
        raise ValueError("角色文件必须位于自建角色目录内")
    return paths


def _atomic_write(path: Path, content: str | bytes) -> None:
    """同目录临时文件替换，写入失败时保留已有文件。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content.encode("utf-8") if isinstance(content, str) else content)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _with_default_rules(character: CharacterDef, source):
    from scenes.persona_chat.skill_loader import default_persona_sections, persona_from_fields

    sections = persona_from_fields(source)
    defaults = default_persona_sections(character.zone, character.name)
    return replace(
        sections,
        voice_directive=sections.voice_directive or defaults.voice_directive,
        post_history_instructions=(
            sections.post_history_instructions or defaults.post_history_instructions
        ),
    )


def _metadata(source: dict) -> dict:
    from scenes.persona_chat.skill_loader import PERSONA_FIELDS

    return {key: value for key, value in source.items() if key not in PERSONA_FIELDS}


def _persist(character: CharacterDef, source: dict) -> Path:
    from scenes.persona_chat.skill_loader import (
        PERSONA_FIELDS, render_persona_skill, resolve_persona_sections,
    )

    path, skill_path, backup_path = _paths(character.id)
    # 已存在的技能是唯一内容源。保留它的人工编辑，空 DTO 或旧 JSON 均不覆盖。
    if not skill_path.exists():
        sections = _with_default_rules(character, source)
        if not sections.role_prompt:
            raise ValueError(f"自建角色 {character.id} 缺少技能包和可迁移的人设")
        skill_text = render_persona_skill(character, sections)
    else:
        skill_text = None

    # 先留存原始 JSON，再迁移；备份不参与加载，也不覆盖第一次保存的原文。
    old_data = None
    if path.exists():
        previous = path.read_bytes()
        old_data = json.loads(previous.decode("utf-8"))
        if any(field in old_data for field in PERSONA_FIELDS) and not backup_path.exists():
            _atomic_write(backup_path, previous)

    data = _metadata(source)
    data["is_custom"] = True
    skill_created = False
    try:
        if skill_text is not None:
            _atomic_write(skill_path, skill_text)
            skill_created = True
        # 技能必须可读才提交不含正文的元数据；失败时旧 JSON 仍完整保留。
        resolve_persona_sections(character)
        if old_data != data:
            _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        # 只回滚本次新写的技能；不动已存在的单一内容源或迁移备份。
        if skill_created:
            skill_path.unlink()
        raise
    return path


def load_custom_characters() -> dict[str, CharacterDef]:
    """扫描落盘目录，返回 id -> CharacterDef（强制 is_custom=True）"""
    out: dict[str, CharacterDef] = {}
    d = custom_dir()
    for p in sorted(d.glob("*.json")):
        try:
            with _LOCK:
                if not is_valid_custom_id(p.stem):
                    continue
                safe_path, _, _ = _paths(p.stem)
                data = json.loads(safe_path.read_text(encoding="utf-8"))
                cid = data.get("id")
                if cid != p.stem:
                    print(f"[custom_store] skip mismatched id file: {p.name}")
                    continue
                data["is_custom"] = True
                data.setdefault("created_at", 0.0)
                data.setdefault("zone", "entertainment")
                metadata = _metadata(data)
                character = CharacterDef(**metadata)
                _persist(character, data)
                out[cid] = character
        except Exception as e:
            print(f"[custom_store] skip corrupted file {p.name}: {e}")
    return out


def save_custom_character(char: CharacterDef) -> Path:
    """新角色写入技能与元数据；已有技能始终保留，不从 DTO 反向覆盖。"""
    if not char.is_custom:
        raise ValueError("只能保存自建角色")
    # 仅导出 dataclass 字段，保证 round-trip 干净
    data = {f: getattr(char, f) for f in char.__dataclass_fields__}
    with _LOCK:
        return _persist(char, data)


def delete_custom_character_file(char_id: str) -> bool:
    """删除角色元数据、技能和迁移备份；所有目标均先校验边界。"""
    paths = _paths(char_id)
    deleted = False
    with _LOCK:
        for path in paths:
            if path.exists():
                path.unlink()
                deleted = True
    return deleted


def generate_custom_id(name: str) -> str:
    """从名字生成一个合法且不冲突的角色 id（如 albert_e7f3a2）"""
    base = _safe_slug(name) or "persona"
    suffix = uuid.uuid4().hex[:6]
    return f"{base}_{suffix}"


def collection_name_for(char_id: str) -> str:
    """自建角色使用独立的 ChromaDB collection 命名空间，避免与内置角色冲突"""
    return f"custom_{char_id}"
