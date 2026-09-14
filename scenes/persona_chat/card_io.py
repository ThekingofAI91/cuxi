"""
card_io.py — 「隔空如面」角色卡的导出与导入（原生 JSON 格式）

角色卡 = 一个可对话人物的全部"软设定"：人设 prompt、性格、场景、开场白、
示例对话、世界书（关键词触发的设定片段）、分区与采样偏好。

设计原则：
- 只定义本项目自有的卡片结构，字段名以本项目的概念体系为准
  （分区 zone / 世界书 lorebook / 后历史指令等），不引入其他产品的格式规范。
- 内置角色也可导出（备份/分享设定用），但 background 不随卡分发：
  内置知识库来自语料文件，体积大且未必可再分发；导出的卡仍可正常对话，
  只是导入方没有该角色的向量检索兜底（人设驱动照常工作）。
- 自建角色导出时会从其独立 collection 反查背景全文，实现完整往返；
  导入时若带 background 则自动切块入库，得到一个"完全一样"的自建角色。

卡片示例：
{
  "format": "gkrm-card",
  "version": 1,
  "name": "卡尔·荣格",
  "role_prompt": "你是卡尔·荣格本人……",
  "personality": "...", "scenario": "...",
  "first_mes": "...", "mes_example": "...",
  "lorebook": [{"keyword": "原型", "content": "...", "position": "before_char"}],
  "zone": "education", "theme": "paper", ...
  "background": "（可选）背景知识全文，导入时切块入库"
}
"""

from __future__ import annotations

from typing import Any

from scenes.persona_chat.models import CharacterDef

# 卡片格式标识与版本（导入时校验，向后兼容留出版本号空间）
CARD_FORMAT = "gkrm-card"
CARD_VERSION = 1

# 世界书条目允许写入卡片的字段（其余字段一律丢弃，防脏数据）
_LOREBOOK_FIELDS = (
    "keyword", "content", "secondary_keys", "enabled",
    "constant", "order", "position",
)


def export_card(character: CharacterDef, background: str = "") -> dict:
    """把 CharacterDef 序列化为卡片 dict（不含 format/version 由调用方补）。"""
    return {
        "format": CARD_FORMAT,
        "version": CARD_VERSION,
        "name": character.name or "",
        "description": character.description or "",
        "role_prompt": character.role_prompt or "",
        "personality": character.personality or "",
        "scenario": character.scenario or "",
        "first_mes": character.first_mes or "",
        "mes_example": character.mes_example or "",
        "lorebook": [
            {k: entry[k] for k in _LOREBOOK_FIELDS if k in entry}
            for entry in (character.lorebook or [])
            if isinstance(entry, dict) and entry.get("keyword") and entry.get("content")
        ],
        "zone": character.zone or "education",
        "theme": character.theme or "original",
        "tagline": character.tagline or "",
        "ability": character.ability or "",
        "avatar": character.avatar or "🎭",
        "enable_verification": bool(character.enable_verification),
        "background": background or "",
    }


def parse_card(data: dict) -> dict:
    """
    校验并清洗一张卡片，返回规范化后的字段 dict。

    抛出 ValueError 时消息面向最终用户（直接进 HTTP 400 detail）。
    """
    if not isinstance(data, dict):
        raise ValueError("角色卡格式不正确：需要一个 JSON 对象")
    if data.get("format") not in (None, CARD_FORMAT):
        raise ValueError(f"不支持的角色卡格式：{data.get('format')}（本站格式为 {CARD_FORMAT}）")
    version = data.get("version", CARD_VERSION)
    if not isinstance(version, int) or version > CARD_VERSION:
        raise ValueError(f"角色卡版本过新：{version}（当前支持 ≤ {CARD_VERSION}）")

    name = (data.get("name") or "").strip()
    if not name:
        raise ValueError("角色卡缺少 name 字段")

    role_prompt = (data.get("role_prompt") or "").strip()
    background = (data.get("background") or "").strip()
    if not role_prompt and not background:
        raise ValueError("角色卡既没有 role_prompt 也没有 background，无法成角色")

    # 世界书：逐条清洗，丢掉缺关键词或缺内容的条目
    lorebook = []
    for entry in (data.get("lorebook") or []):
        if not isinstance(entry, dict):
            continue
        kw = (entry.get("keyword") or "").strip()
        content = (entry.get("content") or "").strip()
        if not kw or not content:
            continue
        cleaned = {k: entry[k] for k in _LOREBOOK_FIELDS if k in entry}
        cleaned["keyword"] = kw
        cleaned["content"] = content
        lorebook.append(cleaned)

    zone = (data.get("zone") or "entertainment").strip()
    if zone not in ("education", "entertainment"):
        zone = "entertainment"

    return {
        "name": name,
        "description": (data.get("description") or "").strip() or name,
        "role_prompt": role_prompt,
        "personality": (data.get("personality") or "").strip(),
        "scenario": (data.get("scenario") or "").strip(),
        # 旧版 V1 的开场白字段名为 greeting
        "first_mes": (data.get("first_mes") or data.get("greeting") or "").strip(),
        "mes_example": (data.get("mes_example") or "").strip(),
        "lorebook": lorebook,
        "zone": zone,
        "theme": (data.get("theme") or "original").strip() or "original",
        "tagline": (data.get("tagline") or "").strip(),
        "ability": (data.get("ability") or "").strip(),
        "avatar": (data.get("avatar") or "🎭").strip() or "🎭",
        "enable_verification": bool(data.get("enable_verification", False)),
        "background": background,
    }


def background_from_collection(collection) -> str:
    """从角色的 ChromaDB collection 反查背景全文（自建角色导出用）。

    自建角色的背景切块时按原文顺序写入，这里按存储顺序拼回近似全文，
    足够支持"导出 → 导入 → 重建同等内容知识库"的往返。
    """
    try:
        res = collection.get(include=["documents"])
        docs = [d for d in (res.get("documents") or []) if d and d.strip()]
        return "\n\n".join(docs)
    except Exception:
        return ""


# ============================================================
# 社区通用 PNG 角色卡导入（格式互操作）
# ============================================================
# PNG 角色卡是社区通行的开放做法：角色设定以 base64 JSON 存在 PNG 的
# tEXt/zTXt 文本块中（键名 chara / ccv3）。文件格式本身不受版权保护，
# 本模块自行实现解析（不使用任何第三方项目的代码），仅做格式互操作。
# 合规姿态：只处理用户主动导入的卡片内容，不预装、不分发任何第三方卡。

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_text_chunks(data: bytes) -> dict[str, str]:
    """遍历 PNG 文本块，返回 {keyword: text}（支持 tEXt 与 zTXt 压缩块）。"""
    if data[:8] != _PNG_SIGNATURE:
        raise ValueError("这不是一个有效的 PNG 文件")
    import struct
    import zlib

    out: dict[str, str] = {}
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"tEXt":
            k, _, v = chunk.partition(b"\x00")
            out[k.decode("latin-1", "ignore")] = v.decode("latin-1", "ignore")
        elif ctype == b"zTXt":
            k, _, rest = chunk.partition(b"\x00")
            if rest:  # 首字节为压缩方法（0=zlib），其余为压缩数据
                try:
                    out[k.decode("latin-1", "ignore")] = zlib.decompress(rest[1:]).decode(
                        "latin-1", "ignore"
                    )
                except Exception:
                    pass
        elif ctype == b"IEND":
            break
        pos += 12 + length
    return out


def _st_to_native(card: dict) -> dict:
    """把社区角色卡字段映射为本站原生卡片结构（再经 parse_card 校验清洗）。"""
    # V2/V3 规范把字段包在 data 子对象里；旧版 V1 是扁平结构
    data = card.get("data") if isinstance(card.get("data"), dict) else card
    personality = (data.get("personality") or "").strip()
    scenario = (data.get("scenario") or "").strip()
    description = (data.get("description") or "").strip()

    mes = (data.get("mes_example") or "").strip()
    mes = "\n".join(
        seg.strip() for seg in mes.replace("<START>", "\n").splitlines()
    ).strip()

    lorebook = []
    book = data.get("character_book") or {}
    for e in (book.get("entries") or []):
        if not isinstance(e, dict):
            continue
        keys = [str(k).strip() for k in (e.get("keys") or e.get("key") or []) if str(k).strip()]
        content = (e.get("content") or "").strip()
        if not keys or not content:
            continue
        pos_raw = e.get("position")
        # 社区规范：0/数字=角色设定之前；1/"after_char"=之后。映射到本站两个注入位
        pos = "after_history" if pos_raw in (1, "1", "after_char") else "before_char"
        try:
            order = int(e.get("insertion_order") or 100)
        except (TypeError, ValueError):
            order = 100
        lorebook.append({
            "keyword": keys[0],
            "secondary_keys": keys[1:],
            "content": content,
            "enabled": e.get("enabled", True),
            "constant": bool(e.get("constant")),
            "order": order,
            "position": pos,
        })

    return {
        "name": (data.get("name") or "").strip(),
        "description": description,
        # 优先卡内自带 system_prompt；否则以 description 作为人设主体
        "role_prompt": (data.get("system_prompt") or "").strip() or description,
        "personality": personality,
        "scenario": scenario,
        # 旧版 V1 的开场白字段名为 greeting
        "first_mes": (data.get("first_mes") or data.get("greeting") or "").strip(),
        "mes_example": mes,
        "lorebook": lorebook,
        "zone": "entertainment",
        "theme": "original",
        "enable_verification": False,
        "background": "",
    }


def parse_png_character_card(data: bytes) -> dict:
    """
    解析社区通用 PNG 角色卡，返回**原生格式卡片 dict**（交 parse_card 校验）。

    兼容：chara（V2/V3，base64 JSON，字段在 data 子对象）、
         旧版 V1（扁平 JSON，无 spec 包装）、zTXt 压缩块。
    """
    import base64
    import json

    chunks = _png_text_chunks(data)
    raw = chunks.get("ccv3") or chunks.get("chara")
    if not raw:
        raise ValueError("这张 PNG 里没有角色卡数据（缺少角色卡文本块）")
    try:
        payload = base64.b64decode(raw)
        card = json.loads(payload.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"角色卡数据损坏，无法解析：{e}")

    if not isinstance(card, dict):
        raise ValueError("角色卡格式不正确")
    return _st_to_native(card)
