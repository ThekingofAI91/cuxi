"""
技能包加载与装配的回归测试

守的是三条线：
1. 人设正文逐字一致 —— skill 文件与 characters/*.py 里的 role_prompt 必须完全相同。
   这条防的是「改了 md 没同步上游」或「导出脚本写坏了」，因为 prompt 的措辞直接
   影响生成结果，差一个字就是另一个角色。
2. 回退路径有效 —— 技能包缺失（自建角色、删文件、格式坏）时必须回退到内联字段，
   不能让角色变成没有设定的空白。
3. 传奇文面契约逐字不变 —— 那份 system prompt 决定前端能否把台词/神情/环境分开
   着色，且里面排篇布局是实测调出来的（状态块不能排最后）。把它抽成模板后必须
   与内置模板逐字相同，否则排版会糊成一色。
"""

from __future__ import annotations

import hashlib

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
    invalidate_cache,
    load_skill,
    resolve_role_prompt,
)


ALL_CHARACTERS = [
    jung_character,
    adler_character,
    wangyangming_character,
    fengge_character,
    zhangxuefeng_character,
]


@pytest.mark.parametrize("character", ALL_CHARACTERS, ids=lambda c: c.id)
def test_skill_body_matches_inline_role_prompt(character):
    """技能包正文必须与角色内联 role_prompt 逐字相同。"""
    pack = load_skill(character.id)
    assert pack is not None, f"{character.id} 缺少技能包，请跑 scripts/export_character_skills.py"
    assert pack.body == (character.role_prompt or "").strip()


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
    assert resolve_role_prompt(jung_character) == pack.body


def test_resolve_falls_back_when_skill_missing(monkeypatch):
    """技能包缺失时回退到内联 role_prompt，不返回空串。"""
    monkeypatch.setattr(skill_loader, "load_skill", lambda name: None)
    assert resolve_role_prompt(jung_character) == jung_character.role_prompt.strip()


def test_custom_character_always_uses_inline():
    """自建角色没有技能包，必须走回退。"""
    custom = type(jung_character)(
        id="persona_deadbe",
        name="自建角色",
        description="测试",
        role_prompt="你是一个测试角色。",
        chroma_collection="persona_custom",
        data_source="data/persona_chat/custom/",
        is_custom=True,
    )
    assert resolve_role_prompt(custom) == "你是一个测试角色。"


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
