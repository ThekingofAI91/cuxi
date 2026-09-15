"""
custom_store.py — 用户自建角色持久化

用户在前端提交「人物设定 + 背景」创建的角色，称为"自建角色"。
与内置角色（写在 characters/*.py）的区别：
- 内置角色：代码写死，随仓库分发，不可删除
- 自建角色：运行时创建，持久化在 data/persona_chat/custom/<id>.json，可删除/重建

本模块负责：
- load_custom_characters()：启动时把磁盘上的自建角色加载回 CharacterDef 并合并进场景配置
- save_custom_character()：把自建角色序列化落盘
- delete_custom_character_file()：删除落盘文件
- generate_custom_id()：从名字生成一个合法且不冲突的角色 id
"""

from __future__ import annotations

import json
import re
import threading
import uuid
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
    return bool(_ID_PATTERN.match(cid or ""))


def load_custom_characters() -> dict[str, CharacterDef]:
    """扫描落盘目录，返回 id -> CharacterDef（强制 is_custom=True）"""
    out: dict[str, CharacterDef] = {}
    d = custom_dir()
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cid = data.get("id")
            if not cid or not is_valid_custom_id(cid):
                print(f"[custom_store] skip invalid id file: {p.name}")
                continue
            data["is_custom"] = True
            data.setdefault("created_at", 0.0)
            # 自建角色默认归娱乐区（重像人/轻检索/极短，不追求专业可溯源）
            data.setdefault("zone", "entertainment")
            out[cid] = CharacterDef(**data)
        except Exception as e:
            print(f"[custom_store] skip corrupted file {p.name}: {e}")
    return out


def save_custom_character(char: CharacterDef) -> Path:
    """把自建角色序列化写入 <id>.json（覆盖写）"""
    d = custom_dir()
    path = d / f"{char.id}.json"
    # 仅导出 dataclass 字段，保证 round-trip 干净
    data = {f: getattr(char, f) for f in char.__dataclass_fields__}
    with _LOCK:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return path


def delete_custom_character_file(char_id: str) -> bool:
    """删除落盘的 <id>.json；成功返回 True。删除失败（权限/回收站不可用）仅告警不抛出。"""
    p = custom_dir() / f"{char_id}.json"
    with _LOCK:
        if p.exists():
            try:
                p.unlink()
                return True
            except OSError as e:
                print(f"[custom_store] 删除文件失败（角色已从内存注销）: {p.name}: {e}")
    return False


def generate_custom_id(name: str) -> str:
    """从名字生成一个合法且不冲突的角色 id（如 albert_e7f3a2）"""
    base = _safe_slug(name) or "persona"
    suffix = uuid.uuid4().hex[:6]
    return f"{base}_{suffix}"


def collection_name_for(char_id: str) -> str:
    """自建角色使用独立的 ChromaDB collection 命名空间，避免与内置角色冲突"""
    return f"custom_{char_id}"
