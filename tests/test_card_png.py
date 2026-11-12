"""
社区通用 PNG 角色卡导入测试：
- PNG 文本块解析（tEXt / zTXt 压缩块）
- V2（chara_card_v2，data 子对象）与旧版 V1（扁平 JSON）字段映射
- 世界书条目映射（keys→keyword/secondary_keys、position 语义）
- 端点：png_base64 导入成功建角、坏图 400、缺文本块 400
合规边界：仅测试格式互操作（自研解析器），不包含任何第三方项目的代码或卡内容。
"""
import base64
import json
import struct
import zlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scenes.persona_chat.card_io import (
    parse_png_character_card,
    parse_card,
    _png_text_chunks,
)


# ============================================================
# 工具：构造最小合法 PNG（签名 + IHDR + tEXt/zTXt + IEND，CRC 正确）
# ============================================================

def _chunk(ctype: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + ctype + data
        + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    )


def build_png(text_chunks: dict[str, str], compressed_keys: set[str] | None = None) -> bytes:
    compressed_keys = compressed_keys or set()
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
    for k, v in text_chunks.items():
        if k in compressed_keys:
            payload = k.encode("latin-1") + b"\x00" + b"\x00" + zlib.compress(v.encode("latin-1"))
            out += _chunk(b"zTXt", payload)
        else:
            out += _chunk(b"tEXt", k.encode("latin-1") + b"\x00" + v.encode("latin-1"))
    out += _chunk(b"IEND", b"")
    return out


def st_v2_card() -> dict:
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "测试剑客",
            "description": "一位浪迹天涯的剑客。",
            "personality": "沉默寡言，重义气",
            "scenario": "客栈相遇",
            "first_mes": "（他抬眼看你）坐。",
            "mes_example": "<START>\n{{user}}: 你好\n{{char}}: 嗯。",
            "voice_directive": "说话惜字如金。",
            "post_history_instructions": "先回答眼前的问题。",
            "character_book": {
                "entries": [
                    {"keys": ["剑", "刀"], "content": "他的剑名为听雨。", "insertion_order": 10,
                     "position": 0, "enabled": True},
                    {"keys": ["往事"], "content": "他曾是一代宗师的弃徒。", "insertion_order": 20,
                     "position": 1, "constant": True},
                    {"keys": [], "content": "缺关键词应被丢弃"},
                ]
            },
        },
    }


# ============================================================
# 解析层
# ============================================================

def test_png_text_chunks_read():
    data = build_png({"chara": "hello", "comment": "x"})
    assert _png_text_chunks(data)["chara"] == "hello"


def test_parse_v2_card_roundtrip():
    card_b64 = base64.b64encode(json.dumps(st_v2_card()).encode("utf-8")).decode("ascii")
    png = build_png({"chara": card_b64})
    fields = parse_card(parse_png_character_card(png))
    assert fields["name"] == "测试剑客"
    assert fields["role_prompt"] == "一位浪迹天涯的剑客。"
    assert fields["first_mes"] == "（他抬眼看你）坐。"
    assert "<START>" not in fields["mes_example"]
    assert fields["voice_directive"] == "说话惜字如金。"
    assert fields["post_history_instructions"] == "先回答眼前的问题。"
    # 世界书已简化为「关键词 + 内容」：keys[0] 作关键词，副关键词与注入位置不再保留
    lb = fields["lorebook"]
    assert len(lb) == 2  # 缺关键词的条目被清洗
    assert lb[0] == {"keyword": "剑", "content": lb[0]["content"]}
    assert "secondary_keys" not in lb[0] and "position" not in lb[0]
    assert lb[1]["keyword"] == "往事"
    assert set(lb[1]) == {"keyword", "content"}


def test_parse_legacy_v1_flat_card():
    v1 = {"name": "老卡人物", "description": "V1 扁平结构", "greeting": "你好呀"}
    # V1 的开场白字段名为 greeting（兼容映射）
    card_b64 = base64.b64encode(json.dumps(v1).encode("utf-8")).decode("ascii")
    png = build_png({"chara": card_b64})
    fields = parse_card(parse_png_character_card(png))
    assert fields["name"] == "老卡人物"


def test_parse_ztxt_compressed_card():
    card_b64 = base64.b64encode(json.dumps(st_v2_card()).encode("utf-8")).decode("ascii")
    png = build_png({"chara": card_b64}, compressed_keys={"chara"})
    fields = parse_card(parse_png_character_card(png))
    assert fields["name"] == "测试剑客"


def test_png_without_card_rejected():
    png = build_png({"comment": "no card here"})
    with pytest.raises(ValueError, match="没有角色卡"):
        parse_png_character_card(png)


def test_non_png_rejected():
    with pytest.raises(ValueError, match="PNG"):
        parse_png_character_card(b"not a png at all")


# ============================================================
# 端点
# ============================================================

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from framework import runtime
    from scenes.persona_chat import custom_store
    from src.api import routes
    monkeypatch.setattr(custom_store, "CUSTOM_DIR", tmp_path)
    monkeypatch.setattr(routes.persona_chat_config, "characters", dict(routes.persona_chat_config.characters))
    monkeypatch.setattr(routes, "_check_rate", lambda *args, **kwargs: None)
    fake_chroma = SimpleNamespace(delete_collection=lambda **kwargs: None)
    monkeypatch.setattr(runtime, "get_chroma_client", lambda: fake_chroma)
    monkeypatch.setattr(routes, "get_chroma_client", lambda: fake_chroma)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_import_png_card_endpoint_creates_character(client, tmp_path):
    card_b64 = base64.b64encode(json.dumps(st_v2_card()).encode("utf-8")).decode("ascii")
    png_b64 = base64.b64encode(build_png({"chara": card_b64})).decode("ascii")
    resp = client.post("/persona/characters/import", json={"png_base64": png_b64})
    assert resp.status_code == 200
    data = resp.json()
    assert data["character"]["name"] == "测试剑客"
    assert data["lorebook_entries"] == 2
    cid = data["character"]["id"]
    stored = json.loads((tmp_path / f"{cid}.json").read_text(encoding="utf-8"))
    assert "role_prompt" not in stored and "first_mes" not in stored
    skill = tmp_path / f"{cid}.md"
    assert skill.is_file()
    skill.write_text(skill.read_text(encoding="utf-8").replace("（他抬眼看你）坐。", "门开着。"), encoding="utf-8")
    exported = client.get(f"/persona/characters/{cid}/export").json()
    assert exported["first_mes"] == "门开着。"
    assert exported["voice_directive"] == "说话惜字如金。"
    listed = client.get("/persona/characters").json()["characters"]
    assert next(char for char in listed if char["id"] == cid)["first_mes"] == "门开着。"
    # 清理：删除自建角色（同时清落盘与向量库）
    assert client.delete(f"/persona/characters/{cid}").status_code == 200
    assert not skill.exists()


def test_import_storage_failure_does_not_register_character(client, monkeypatch):
    from src.api import routes
    before = set(routes.persona_chat_config.characters)

    def fail_save(character):
        raise OSError("disk unavailable")

    monkeypatch.setattr(routes, "save_custom_character", fail_save)
    response = client.post("/persona/characters/import", json={
        "card": {"name": "临时人物", "role_prompt": "你是临时人物。"},
    })
    assert response.status_code == 500
    assert set(routes.persona_chat_config.characters) == before


def test_create_storage_failure_does_not_register_character(client, monkeypatch):
    from src.api import routes
    before = set(routes.persona_chat_config.characters)

    async def build(**kwargs):
        return "你是临时人物。", "背景", [], {}

    async def ingest(*args):
        return 1

    def fail_save(character):
        raise OSError("disk unavailable")

    monkeypatch.setattr(routes, "build_persona", build)
    monkeypatch.setattr(routes, "ingest_texts_async", ingest)
    monkeypatch.setattr(routes, "save_custom_character", fail_save)
    response = client.post("/persona/characters/create", json={
        "name": "临时人物", "role_prompt": "你是临时人物。", "use_web_search": False,
    })
    assert response.status_code == 500
    assert set(routes.persona_chat_config.characters) == before


def test_import_png_without_card_data_400(client):
    png_b64 = base64.b64encode(build_png({"comment": "x"})).decode("ascii")
    resp = client.post("/persona/characters/import", json={"png_base64": png_b64})
    assert resp.status_code == 400


def test_import_non_png_400(client):
    resp = client.post(
        "/persona/characters/import",
        json={"png_base64": base64.b64encode(b"garbage").decode("ascii")},
    )
    assert resp.status_code == 400
