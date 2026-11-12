"""角色技能单源、热更新、角色卡无损序列化及传奇文面契约。"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import framework.legend as legend_mod
from framework.legend import build_system_prompt
from scenes.persona_chat import skill_loader
from scenes.persona_chat.characters.adler import adler_character
from scenes.persona_chat.characters.fengge import fengge_character
from scenes.persona_chat.characters.jung import jung_character
from scenes.persona_chat.characters.wangyangming import wangyangming_character
from scenes.persona_chat.characters.zhangxuefeng import zhangxuefeng_character
from scenes.persona_chat.legend_store import LegendSave
from scenes.persona_chat.skill_loader import (
    load_persona,
    parse_persona_sections,
    invalidate_cache,
    load_skill,
    resolve_role_prompt,
    resolve_persona_sections,
    PersonaSections,
    PersonaSkillError,
    PERSONA_FIELDS,
    render_persona_skill,
)


ALL_CHARACTERS = [
    jung_character,
    adler_character,
    wangyangming_character,
    fengge_character,
    zhangxuefeng_character,
]


@pytest.mark.parametrize("character", ALL_CHARACTERS, ids=lambda c: c.id)
def test_builtin_persona_lives_only_in_skill(character):
    """内置 Python 只留元数据，八块人设均从技能读取。"""
    persona = load_persona(character.id)
    assert persona is not None
    for key in PERSONA_FIELDS:
        assert getattr(persona, key), f"{character.id} 技能缺少 {key}"
        assert not getattr(character, key), f"{character.id} 在 Python 重复维护 {key}"


@pytest.mark.parametrize("character", ALL_CHARACTERS, ids=lambda c: c.id)
def test_frontmatter_parsed(character):
    """frontmatter 至少要解析出 name 与 description，且 name 带角色 id。"""
    pack = load_skill(character.id)
    assert pack is not None
    assert pack.name == f"{character.id}-persona"
    assert character.name in pack.description
    assert pack.zone == character.zone


def test_resolve_prefers_skill(monkeypatch):
    """技能包存在时，装配取的是技能包正文而不是内联字段。"""
    pack = load_skill("jung")
    assert pack is not None
    assert resolve_role_prompt(jung_character) == parse_persona_sections(pack.body).role_prompt


def test_missing_skill_fails_instead_of_using_stale_fields(monkeypatch):
    """缺技能必须明确报错，不能重新启用旧人设。"""
    monkeypatch.setattr(skill_loader, "load_skill", lambda name: None)
    character = SimpleNamespace(id="missing", role_prompt="旧设定不能复活")
    with pytest.raises(PersonaSkillError, match="missing"):
        resolve_role_prompt(character)


def test_custom_character_also_uses_skill(tmp_path, monkeypatch):
    from scenes.persona_chat import custom_store
    monkeypatch.setattr(custom_store, "CUSTOM_DIR", tmp_path)
    custom = type(jung_character)(
        id="persona_deadbe",
        name="自建角色",
        description="测试",
        role_prompt="你是一个测试角色。",
        chroma_collection="persona_custom",
        data_source="data/persona_chat/custom/",
        is_custom=True,
    )
    with pytest.raises(PersonaSkillError):
        resolve_role_prompt(custom)
    (tmp_path / f"{custom.id}.md").write_text("## 人设\n\n技能里的新角色", encoding="utf-8")
    assert resolve_role_prompt(custom) == "技能里的新角色"


def test_empty_skill_sections_never_refill_from_inline(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_loader, "SKILL_DIR", tmp_path)
    (tmp_path / "demo.md").write_text(
        "## 人设\n新身份\n\n## 对话示例\n\n## 世界书\n\n## 后历史指令\n", encoding="utf-8"
    )
    character = SimpleNamespace(
        id="demo", mes_example="旧示例", lorebook=[{"keyword": "旧", "content": "旧词条"}],
        post_history_instructions="旧后历史指令",
    )
    sections = resolve_persona_sections(character)
    assert sections.mes_example == ""
    assert sections.lorebook == []
    assert sections.post_history_instructions == ""


def test_skill_serializer_preserves_embedded_headings_and_multiline_lore():
    character = SimpleNamespace(id="demo", name="演示", zone="education")
    original = PersonaSections(
        role_prompt="这是原文\n\n## 对话示例\n这是原文内的标题\n~~~\n围栏也要保留",
        mes_example="{{char}}: 你好\n## 人设\n这是台词的一部分",
        lorebook=[{"keyword": "HTTP:状态", "content": "他说：第一句\n## 后历史指令\n第二句"}],
        first_mes="坐吧", voice_directive="直接回答", post_history_instructions="保持原来的口吻",
    )
    _, body = skill_loader._split_frontmatter(render_persona_skill(character, original))
    assert parse_persona_sections(body) == original


def test_cache_isolated_between_directories(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "demo.md").write_text("甲的人设", encoding="utf-8")
    (b / "demo.md").write_text("乙的人设", encoding="utf-8")
    import os
    stat = (a / "demo.md").stat()
    os.utime(b / "demo.md", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(skill_loader, "SKILL_DIR", a)
    assert load_skill("demo").body == "甲的人设"
    monkeypatch.setattr(skill_loader, "SKILL_DIR", b)
    assert load_skill("demo").body == "乙的人设"


def test_cache_invalidated_by_mtime(tmp_path, monkeypatch):
    """改文件后重新加载应拿到新内容（缓存按 mtime 失效）。"""
    monkeypatch.setattr(skill_loader, "SKILL_DIR", tmp_path)
    invalidate_cache()
    path = tmp_path / "demo.md"
    path.write_text("---\nname: demo\n---\n\n第一版", encoding="utf-8")
    first = load_skill("demo")
    assert first is not None and first.body == "第一版"

    path.write_text("---\nname: demo\n---\n\n第二版内容", encoding="utf-8")
    import os
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))
    second = load_skill("demo")
    assert second is not None and second.body == "第二版内容"
    invalidate_cache()


def _legend_saves() -> dict[str, LegendSave]:
    return {
        "legacy": LegendSave(
            id="t1",
            title="旧档",
            world="长安西市，暮春。",
            protagonist_name="李寻",
            protagonist_desc="游方书生",
            npcs=[{"name": "苏娘", "role": "酒肆掌柜", "persona": "三十许，眼利心细"}],
        ),
        "styled": LegendSave(
            id="t2",
            title="暗色",
            world="雨夜孤城。",
            protagonist_name="沈砚",
            npcs=[{"name": "黑衣人", "role": "追兵", "persona": "沉默"}],
            style="dark",
        ),
        "stated": LegendSave(
            id="t3",
            title="带状态",
            world="江湖。",
            protagonist_name="陆青",
            npcs=[{"name": "阿七", "role": "同伴", "persona": "话多"}],
            state_fields=[{"name": "体力", "kind": "number", "init": 10, "desc": "0-10"}],
            only_fields=[{"name": "心境", "kind": "tag", "init": "平静", "desc": ""}],
            state={"阿七": {"体力": 8}},
            only={"心境": "警惕"},
        ),
    }


@pytest.mark.parametrize("name", ["legacy", "styled", "stated"])
def test_legend_template_matches_builtin(name, monkeypatch):
    """模板渲染结果必须与内置 f-string 逐字相同（保护文面契约与状态块位置）。"""
    save = _legend_saves()[name]
    from_template = build_system_prompt(save)

    monkeypatch.setattr(legend_mod, "legend_prompt_template", lambda: None)
    builtin = build_system_prompt(save)

    assert hashlib.sha256(from_template.encode()).hexdigest() == hashlib.sha256(
        builtin.encode()
    ).hexdigest()
    assert from_template == builtin


def test_legend_falls_back_without_skill(monkeypatch):
    """技能包缺失时传奇仍能生成完整契约，不抛异常。"""
    monkeypatch.setattr(legend_mod, "legend_prompt_template", lambda: None)
    prompt = build_system_prompt(_legend_saves()["legacy"])
    assert "铁律" in prompt and "文面格式" in prompt


def test_legend_skill_exists():
    """传奇技能包必须存在，否则上面的逐字一致测试等于自证。"""
    pack = load_skill("legend-narrator")
    assert pack is not None
    assert "{{world}}" in pack.body
    assert "{{state_block}}" in pack.body


def test_parse_sections_keeps_unknown_headings():
    """只有五个已知标题切分节，正文里别的 ## 留在原分节里，不会被误切。"""
    body = chr(10).join([
        "## 人设", "", "第一段", "", "## 别的标题", "", "第二段", "", "## 性格特质", "", "温和",
    ])
    sections = parse_persona_sections(body)
    assert sections.role_prompt == chr(10).join(["第一段", "", "## 别的标题", "", "第二段"])
    assert sections.personality == "温和"


def test_lorebook_splits_on_first_colon():
    """词条按第一个冒号切分：内容里的冒号不能被当成关键词分隔符。"""
    body = chr(10).join([
        "## 核心概念词条", "", "- 阴影：他说：这是地下室", "- 原型: 他说：英文冒号也认",
    ])
    sections = parse_persona_sections(body)
    assert sections.lorebook[0] == {"keyword": "阴影", "content": "他说：这是地下室"}
    assert sections.lorebook[1] == {"keyword": "原型", "content": "他说：英文冒号也认"}


def test_legacy_body_without_sections():
    """旧技能正文仍可读，其余设定保持空白。"""
    sections = parse_persona_sections("你是某某，说话直接。")
    assert sections.role_prompt == "你是某某，说话直接。"
    assert sections.personality == "" and sections.lorebook == []
