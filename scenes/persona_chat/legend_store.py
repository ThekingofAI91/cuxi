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
世界观 + 角色卡 + 剧情历史 + **状态栏**全在里面，删档即彻底消失。

状态栏怎么存的（细节见下方「状态栏」段）：
- `state_fields` / `only_fields` = **作者在开局前定死的字段定义**（有哪些词条、什么类型、初值）
- `state` / `only` = **每轮被模型改写的运行值**
  前者「通用」——在场每个人物各有一份；后者「仅主角」——只有主角有。
定义与值分开存，是为了模型漏写某个字段时还能按定义补回默认值。

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

# ------------------------------------------------------------
# 状态栏：字段定义（作者写）+ 运行时值（模型每轮更新）
# ------------------------------------------------------------
# 为什么要分成「定义」和「值」两份：定义是**作者在开局前定死的**（有哪些词条、
# 是什么类型、初值多少），值是**每轮变的**。混在一起的话，模型一旦漏写某个
# 字段就再也找不回来了——定义在，才能每轮补齐默认值。
#
# 为什么要分 `state`（通用）和 `only`（仅主角）两段：
#   通用 = 在场**每个人物各有一份**（好感度、态度、身体状态……）
#   仅主角 = 只有主角有（体力、金钱、修为、称号……）
# 这两段正是同类产品（Omnimundia 一类）状态面板的两栏，也是玩家真正会看的东西。

# 字段条数上限。跟 MAX_NPCS 一样是**认知负担**上限而不是技术上限：
# 状态栏是给玩家一眼扫的，条目再多就没人看了，还会挤占 prompt。
MAX_STATE_FIELDS = 8
MAX_ONLY_FIELDS = 6

# 角色/词条名的合法形态。★不能有空白与 JSON 结构字符：字段名会作为 JSON 的 key
# 出现在模型输出的补丁里，一旦带 `"` `:` `,` `{` `}`，补丁解析就会被带歪。
_FIELD_NAME_RE = re.compile(r"^[^\s\"'`:{}\[\],，、]{1,12}$")

# 词条型（tag）值的长度上限；数值型（number）的绝对值上限。
# 两者都是为了「模型胡写也不至于把存档撑爆」——补丁是模型生成的文本，不是可信输入。
MAX_TAG_CHARS = 24
STATE_NUMBER_ABS_MAX = 1e6

FIELD_KINDS = ("number", "tag")
FIELD_KIND_LABELS = {"number": "数值", "tag": "词条"}


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
    # ---- 状态栏 ----
    # 字段定义（作者定，开局前写死）：[{name, kind, init, desc}]，kind ∈ number/tag
    state_fields: list[dict] = field(default_factory=list)   # 通用：在场每个人物各一份
    only_fields: list[dict] = field(default_factory=list)    # 仅主角
    # 运行时值（模型每轮更新，经白名单 + 类型钳制后才落进来）
    state: dict = field(default_factory=dict)   # {人物名: {字段名: 值}}
    only: dict = field(default_factory=dict)    # {字段名: 值}
    created_at: float = 0.0
    updated_at: float = 0.0

    # ---- 便捷读取 ----
    def character_names(self) -> list[str]:
        """在场人物：主角在前，其后是各配角（去重保序）。"""
        return character_names_of(self)

    def has_state(self) -> bool:
        """这局有没有状态栏。没定义字段的旧档走原路径，行为完全不变。"""
        return bool(self.state_fields or self.only_fields)

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
# 状态栏：定义规范化、值初始化、补丁合并
# ============================================================

def character_names_of(save: LegendSave) -> list[str]:
    """主角 + 各配角（去重保序）。状态栏的「通用」那一栏就按这个名单分组。"""
    out: list[str] = []
    p = (save.protagonist_name or "").strip()
    if p:
        out.append(p)
    for n in save.npcs:
        if not isinstance(n, dict):
            continue
        nm = str(n.get("name") or "").strip()
        if nm and nm not in out:
            out.append(nm)
    return out


def _coerce_init(kind: str, value):
    """初值按类型收敛。作者填错就地兜底，不报错——开局前的失败体验很差。"""
    if kind == "number":
        try:
            n = float(value)
        except (TypeError, ValueError):
            return 0.0
        if n != n or n in (float("inf"), float("-inf")):
            return 0.0
        return round(max(-STATE_NUMBER_ABS_MAX, min(STATE_NUMBER_ABS_MAX, n)), 2)
    return ("" if value is None else str(value)).strip().replace("\n", " ")[:MAX_TAG_CHARS]


def _coerce_value(field_dict: dict, value):
    """把模型给的值洗成合法值。返回 None = 这次忽略（保持旧值）。

    ★数值型必须能返回 None：模型可能给「未知」「-」这种非数字，
    直接 float() 会抛，静默转 0 又会把玩家的数值抹掉。忽略才是对的。
    """
    if field_dict.get("kind") == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            n = float(value)
        except (TypeError, ValueError):
            return None
        if n != n or n in (float("inf"), float("-inf")):
            return None
        return round(max(-STATE_NUMBER_ABS_MAX, min(STATE_NUMBER_ABS_MAX, n)), 2)
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")[:MAX_TAG_CHARS]


def normalize_fields(raw) -> list[dict]:
    """把作者填的字段清单洗成规范形态。

    非法条目**直接丢掉而不是报错**：这是创建前的表单数据，一条写错就整局建不成，
    代价太大。名字重复按先到先得。
    """
    out: list[dict] = []
    seen: set[str] = set()
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not _FIELD_NAME_RE.match(name) or name in seen:
            continue
        kind = str(item.get("kind") or "number").strip().lower()
        if kind not in FIELD_KINDS:
            kind = "number"
        seen.add(name)
        out.append({
            "name": name,
            "kind": kind,
            "init": _coerce_init(kind, item.get("init")),
            "desc": str(item.get("desc") or "").strip()[:60],
        })
    return out


def _init_of(fields: list[dict], name: str):
    for f in fields:
        if f.get("name") == name:
            return f.get("init")
    return None


def reset_state(save: LegendSave) -> None:
    """按字段定义把状态值整体重建（创建存档时调用）。"""
    names = character_names_of(save)
    if save.state_fields:
        save.state = {
            n: {f["name"]: f.get("init") for f in save.state_fields}
            for n in names
        }
    else:
        save.state = {}
    save.only = {f["name"]: f.get("init") for f in save.only_fields}


def sync_state(save: LegendSave) -> None:
    """把状态值对齐到「字段定义 + 当前在场人物」。

    用途：① 旧档升级（当年没有状态栏，字段是后加的）；
    ② 定义了新字段但值还没初始化；③ 清掉定义里已经删掉的字段。
    ★不改动已有值，只补默认、删越界——所以可以安全地在每次读档时跑。
    """
    names = character_names_of(save)
    common = [f["name"] for f in save.state_fields]
    only = [f["name"] for f in save.only_fields]

    cur = save.state if isinstance(save.state, dict) else {}
    if common:
        rebuilt = {}
        for n in names:
            old = cur.get(n)
            old = old if isinstance(old, dict) else {}
            rebuilt[n] = {c: old.get(c, _init_of(save.state_fields, c)) for c in common}
        save.state = rebuilt
    else:
        save.state = {}

    cur_only = save.only if isinstance(save.only, dict) else {}
    save.only = {o: cur_only.get(o, _init_of(save.only_fields, o)) for o in only}


def apply_state_patch(save: LegendSave, patch) -> dict:
    """把模型吐出的状态补丁合并进存档，返回真正改动过的值。

    ★两道闸门，缺一不可：
      1. **白名单**：只认定义过的字段 + 在场的人物名。补丁是模型生成的**文本**，
         不设白名单就等于允许模型往存档里写任意键（还能一直涨）。
      2. **类型钳制**：数值型转 float 并限幅，词条型截断长度。
    两道都过不了的条目静默忽略——状态栏宁可少一条，也不能被写坏。
    """
    changed: dict = {"state": {}, "only": {}}
    if not isinstance(patch, dict):
        return changed

    common = {f["name"]: f for f in save.state_fields}
    only = {f["name"]: f for f in save.only_fields}
    names = character_names_of(save)

    raw_state = patch.get("state")
    if isinstance(raw_state, dict) and common and isinstance(save.state, dict):
        # 两种写法都收：按人物分组（推荐），或直接给字段名（默认算作主角的）。
        flat = (raw_state and not any(k in names for k in raw_state)
                and all(k in common for k in raw_state))
        groups = {save.protagonist_name: raw_state} if flat else raw_state
        for who, vals in groups.items():
            if who not in names or not isinstance(vals, dict):
                continue
            bucket = save.state.setdefault(who, {})
            for fname, value in vals.items():
                fd = common.get(fname)
                if fd is None:
                    continue
                nv = _coerce_value(fd, value)
                if nv is None or bucket.get(fname) == nv:
                    continue
                bucket[fname] = nv
                changed["state"].setdefault(who, {})[fname] = nv

    raw_only = patch.get("only")
    if isinstance(raw_only, dict) and only and isinstance(save.only, dict):
        for fname, value in raw_only.items():
            fd = only.get(fname)
            if fd is None:
                continue
            nv = _coerce_value(fd, value)
            if nv is None or save.only.get(fname) == nv:
                continue
            save.only[fname] = nv
            changed["only"][fname] = nv

    return changed


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
                       ("opening", ""), ("protagonist_desc", ""),
                       ("state_fields", []), ("only_fields", []),
                       ("state", {}), ("only", {})):
        data.setdefault(f, default)
    # 字段定义过一遍规范化（防手改存档写进非法名），再把值对齐到定义
    data["state_fields"] = normalize_fields(data.get("state_fields"))[:MAX_STATE_FIELDS]
    data["only_fields"] = normalize_fields(data.get("only_fields"))[:MAX_ONLY_FIELDS]
    try:
        save = LegendSave(**data)
    except TypeError as e:
        print(f"[legend_store] 字段不匹配 {p.name}: {e}")
        return None
    sync_state(save)
    return save


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
                      npcs: list[dict] | None = None,
                      state_fields: list[dict] | None = None,
                      only_fields: list[dict] | None = None) -> tuple[bool, str]:
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

    # 状态栏字段：条数 = 玩家一眼能扫的量；名字必须能作为 JSON 的 key
    for label, raw, limit in (("通用词条", state_fields, MAX_STATE_FIELDS),
                              ("主角专属词条", only_fields, MAX_ONLY_FIELDS)):
        items = [it for it in (raw or [])
                 if isinstance(it, dict) and str(it.get("name") or "").strip()]
        if len(items) > limit:
            return False, f"{label}最多 {limit} 条（状态栏是给一眼扫的，太多就没人看了）"
        seen: set[str] = set()
        for it in items:
            nm = str(it["name"]).strip()
            if not _FIELD_NAME_RE.match(nm):
                return False, f"词条名「{nm}」不能带空格或标点，且不超过 12 个字"
            if nm in seen:
                return False, f"{label}里有重名：{nm}"
            seen.add(nm)
    return True, ""
