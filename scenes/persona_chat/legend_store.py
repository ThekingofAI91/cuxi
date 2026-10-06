"""legend_store.py — 「传奇」剧情模式的存档持久化。

传奇是什么：用户先给出**世界观 / 主角 / 配角**三段设定，然后**自己扮演主角**
推进剧情；叙述与所有配角的言行由同一个模型扮演。

与相邻概念的区别（别混）：
- 自建角色（`custom_store.py`）：建一个**对话对象**，进首页「对话对象」列表，
  可以单独找 TA 聊。传奇不是这个。
- 争鸣（圆桌）：选**已有角色**就一个议题交锋，用户是提问者/点将者。
  传奇的角色是用户**现场创作**的，且用户要**下场扮演主角**。

★这三类角色互不流通（用户明确要求）：传奇里的主角与配角**不进**首页对话对象列表，
也不会生成 `custom_*` collection。传奇存档是自包含的一坨 JSON：
世界观 + 角色卡 + 剧情历史全在里面，删档即彻底消失。

落盘位置 `data/persona_chat/legends/<id>.json`。注意这是 `data/persona_chat/`
下的一个**平级新目录**，不在任何角色的 `data_source`（如 `.../wangyangming`）里，
所以不会被 `collect_corpus_files` 当成语料扫进去。
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 存档落盘目录（相对项目根）
LEGEND_DIR = Path("data/persona_chat/legends")

# 存档 id 白名单：仅小写字母/数字/下划线，避免注入与路径穿越
_ID_PATTERN = re.compile(r"^[a-z0-9_]{2,40}$")

# 写盘锁：并发创建/保存时串行，避免同文件互相覆盖
_LOCK = threading.Lock()

# 单局最大配角数。上限不是技术限制而是「叙事质量」限制：
# 一个模型同时演太多人，每人分到的注意力会摊薄，配角就会退化成只有名字的
# 复读机。实测 4 个以内每人还能有稳定口吻。
MAX_NPCS = 4

# 剧情历史最多保留多少轮（一轮 = 用户行动 + 一次的叙述）。
# 超出的旧轮次会被裁掉——不裁的话上下文无限涨，既撞 token 上限又让模型
# 把注意力放在远古剧情上。20 轮约等于一次完整短篇的长度。
MAX_TURNS = 20


@dataclass
class LegendNPC:
    """一个配角。字段刻意少而硬：名字 + 定位 + 设定。"""

    name: str
    role: str = ""      # 身份/定位，如「青楼老板娘」「退隐的老剑客」
    persona: str = ""   # 性格、口吻、与主角的关系、知道什么

    def to_card(self) -> str:
        """渲染成塞进 prompt 的角色卡文本。"""
        bits = [self.name]
        if self.role:
            bits.append(f"（{self.role}）")
        line = "".join(bits)
        if self.persona:
            line += f"：{self.persona}"
        return line


@dataclass
class LegendSave:
    """一局传奇的完整存档。"""

    id: str
    title: str                      # 存档显示名（默认取主角名）
    world: str                      # 世界观设定（必填）
    protagonist_name: str           # 主角名（用户扮演）
    protagonist_desc: str = ""      # 主角身份与设定
    npcs: list[dict] = field(default_factory=list)   # [{name, role, persona}]
    opening: str = ""               # 开场情境（可留空，让模型自己起头）
    style: str = "classic"          # 叙事风格：classic/light/dark
    turns: list[dict] = field(default_factory=list)  # [{role, content, ts}]
    created_at: float = 0.0
    updated_at: float = 0.0

    # ---- 便捷读取 ----
    def npc_objects(self) -> list[LegendNPC]:
        out = []
        for n in self.npcs:
            if isinstance(n, dict) and (n.get("name") or "").strip():
                out.append(LegendNPC(
                    name=str(n.get("name", "")).strip(),
                    role=str(n.get("role", "")).strip(),
                    persona=str(n.get("persona", "")).strip(),
                ))
        return out

    def turn_count(self) -> int:
        """已进行的轮数（只数用户行动，一次行动算一轮）。"""
        return sum(1 for t in self.turns if t.get("role") == "user")

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 目录与 id
# ============================================================

def legend_dir() -> Path:
    LEGEND_DIR.mkdir(parents=True, exist_ok=True)
    return LEGEND_DIR


def is_valid_legend_id(lid: str) -> bool:
    return bool(_ID_PATTERN.match(lid or ""))


def _safe_slug(name: str) -> str:
    """任意名字 → ascii 安全 slug（中文会变空，调用方兜底）"""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")[:20]


def generate_legend_id(protagonist_name: str) -> str:
    """从主角名生成合法且不冲突的存档 id（如 hero_3b6e31）"""
    base = _safe_slug(protagonist_name) or "legend"
    return f"{base}_{uuid.uuid4().hex[:6]}"


# ============================================================
# 读写
# ============================================================

def _path_for(lid: str) -> Path:
    return legend_dir() / f"{lid}.json"


def save_legend(save: LegendSave) -> Path:
    """落盘（覆盖写）。updated_at 由本函数统一刷新。"""
    save.updated_at = time.time()
    if not save.created_at:
        save.created_at = save.updated_at
    # 只保留最后 MAX_TURNS 轮，防止上下文无限增长
    if save.turn_count() > MAX_TURNS:
        save.turns = _trim_turns(save.turns, MAX_TURNS)
    p = _path_for(save.id)
    with _LOCK:
        p.write_text(
            json.dumps(save.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return p


def _trim_turns(turns: list[dict], keep_rounds: int) -> list[dict]:
    """从后往前数 keep_rounds 次用户行动，保留它们及其之后的叙述。

    不能简单按条数切：turns 里用户行动与叙述交替，按条切会把某轮的
    用户行动留下、叙述切掉，模型下一轮就会「忘了自己刚说了什么」。
    """
    seen = 0
    cut = 0
    for i in range(len(turns) - 1, -1, -1):
        cut = i
        if turns[i].get("role") == "user":
            seen += 1
            if seen >= keep_rounds:
                break
    return turns[cut:]


def load_legend(lid: str) -> LegendSave | None:
    """读一个存档；不存在或损坏返回 None。"""
    if not is_valid_legend_id(lid):
        return None
    p = _path_for(lid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[legend_store] 读取失败 {p.name}: {e}")
        return None
    # 兼容旧档：缺字段就补默认值，别让一次格式演进废掉用户存档
    for f, default in (("npcs", []), ("turns", []), ("style", "classic"),
                       ("opening", ""), ("protagonist_desc", "")):
        data.setdefault(f, default)
    try:
        return LegendSave(**data)
    except TypeError as e:
        print(f"[legend_store] 字段不匹配 {p.name}: {e}")
        return None


def list_legends() -> list[dict]:
    """所有存档的摘要（不含剧情全文），按最近更新倒序。"""
    out = []
    for p in legend_dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns = data.get("turns") or []
        out.append({
            "id": data.get("id") or p.stem,
            "title": data.get("title") or data.get("protagonist_name") or p.stem,
            "protagonist": data.get("protagonist_name", ""),
            "npc_count": len(data.get("npcs") or []),
            "turns": sum(1 for t in turns if t.get("role") == "user"),
            "updated_at": data.get("updated_at") or 0.0,
        })
    out.sort(key=lambda d: d["updated_at"], reverse=True)
    return out


def delete_legend(lid: str) -> bool:
    """删除存档文件；成功返回 True。"""
    if not is_valid_legend_id(lid):
        return False
    p = _path_for(lid)
    with _LOCK:
        if p.exists():
            try:
                p.unlink()
                return True
            except OSError as e:
                print(f"[legend_store] 删除失败 {p.name}: {e}")
    return False


# ============================================================
# 入参校验
# ============================================================

def validate_new_save(*, world: str, protagonist_name: str,
                      npcs: list[dict] | None = None) -> tuple[bool, str]:
    """新建存档前的校验。返回 (是否合法, 不合法原因)。

    为什么要在存储层再校验一遍：API 层也会校验，但那层管的是「HTTP 请求格式」，
    这层管的是「能不能构成一局游戏」。两边的失败语义不同，别合并。
    """
    if not (world or "").strip():
        return False, "世界观设定不能为空"
    if not (protagonist_name or "").strip():
        return False, "主角名字不能为空"
    if len(world.strip()) < 10:
        return False, "世界观太短了，至少写 10 个字，否则模型没有可依据的舞台"
    named = [n for n in (npcs or [])
             if isinstance(n, dict) and (n.get("name") or "").strip()]
    if len(named) > MAX_NPCS:
        return False, f"配角最多 {MAX_NPCS} 个（人太多每人的戏份会被摊薄）"
    names = [str(n.get("name")).strip() for n in named]
    if len(names) != len(set(names)):
        return False, "配角名字有重复"
    if protagonist_name.strip() in names:
        return False, "配角名字不能与主角相同"
    return True, ""
