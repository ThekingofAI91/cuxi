# -*- coding: utf-8 -*-
"""
test_prompt_builder.py — 酒馆式提示词装配器的行为契约

这些用例锁住三件容易悄悄退化的事：
1. 世界书只在关键词命中时注入（否则每轮都塞满词条，prompt 膨胀且跑题）
2. 示例对话与性格/场景按序装配，示例靠后（对生成影响最强）
3. 娱乐区语气指令只给娱乐区（教育区一旦混进去，引用与结构化格式会被禁掉）
"""

import pytest

from scenes.persona_chat.models import CharacterDef
from scenes.persona_chat.prompt_builder import (
    build_character_prompt,
    has_tavern_components,
    match_lorebook,
)


def _char(**kw):
    base = dict(
        id="t",
        name="测试角色",
        description="",
        role_prompt="你是测试角色。",
        chroma_collection="persona_t",
        data_source="",
    )
    base.update(kw)
    return CharacterDef(**base)


LORE = [
    {"keyword": "知行合一", "content": "知而不行，只是未知。"},
    {"keyword": "致良知", "content": "是非之心，人皆有之。"},
]


class TestMatchLorebook:
    def test_hit_returns_only_matched(self):
        hits = match_lorebook(LORE, "请问知行合一怎么做？")
        assert len(hits) == 1
        assert hits[0]["keyword"] == "知行合一"

    def test_miss_returns_empty(self):
        assert match_lorebook(LORE, "今天天气不错") == []

    def test_empty_inputs(self):
        assert match_lorebook([], "知行合一") == []
        assert match_lorebook(LORE, "") == []

    def test_respects_limit(self):
        lore = [{"keyword": f"词{i}", "content": f"内容{i}"} for i in range(10)]
        text = " ".join(f"词{i}" for i in range(10))
        assert len(match_lorebook(lore, text, limit=3)) == 3

    def test_skips_malformed_entries(self):
        lore = [
            {"keyword": "", "content": "空关键词"},
            {"keyword": "无内容", "content": ""},
            "不是字典",
            {"keyword": "正常", "content": "有内容"},
        ]
        hits = match_lorebook(lore, "正常 无内容")
        assert [h["keyword"] for h in hits] == ["正常"]

    def test_no_duplicate_keyword(self):
        lore = [
            {"keyword": "重复", "content": "甲"},
            {"keyword": "重复", "content": "乙"},
        ]
        assert len(match_lorebook(lore, "重复")) == 1


class TestBuildCharacterPrompt:
    def test_includes_all_blocks_in_order(self):
        c = _char(
            personality="- 直率",
            scenario="在书房",
            mes_example="{{user}}: 你好\n{{char}}: 坐。",
            lorebook=LORE,
        )
        p = build_character_prompt(c, query="知行合一是什么？")
        i_personality = p.index("【性格特质】")
        i_scenario = p.index("【当前场景】")
        i_lore = p.index("知行合一")
        i_example = p.index("【对话示例")
        assert i_personality < i_scenario < i_lore < i_example

    def test_lorebook_absent_when_no_hit(self):
        c = _char(lorebook=LORE)
        p = build_character_prompt(c, query="随便聊聊")
        assert "【核心概念词条" not in p
        assert "知而不行" not in p

    def test_entertainment_gets_voice_directive(self):
        c = _char(zone="entertainment")
        p = build_character_prompt(c, query="你好")
        assert "娱乐区口吻要求" in p

    def test_education_has_no_voice_directive(self):
        c = _char(zone="education")
        p = build_character_prompt(c, query="你好")
        assert "娱乐区口吻要求" not in p

    def test_uses_entertainment_lorebook_wording(self):
        c = _char(zone="entertainment", lorebook=LORE)
        p = build_character_prompt(c, query="知行合一")
        assert "【你的黑话与老梗" in p

    def test_bare_character_still_returns_role_prompt(self):
        c = _char()
        assert build_character_prompt(c, query="") == "你是测试角色。"


class TestHasTavernComponents:
    def test_empty_character(self):
        assert has_tavern_components(_char()) is False

    def test_example_only(self):
        assert has_tavern_components(_char(mes_example="{{user}}: a")) is True

    def test_lorebook_only(self):
        assert has_tavern_components(_char(lorebook=LORE)) is True
