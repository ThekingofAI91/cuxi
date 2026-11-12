"""
圆桌发言 prompt 的结构约束

身份在前，记录是 user 数据，技能后历史指令和语言锚在记录之后、本轮任务之前。
这些断言验证装配及单一来源，不把位置契约当成模型抗干扰效果的证明。
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from framework.roundtable import (
    ROUNDTABLE_DEBATE_DIRECTIVE,
    build_speaker_messages,
)
from scenes.persona_chat.characters.jung import jung_character
from scenes.persona_chat import skill_loader
from scenes.persona_chat.skill_loader import PersonaSections, render_persona_skill


async def _fake_retrieve(*_args, **_kwargs) -> str:
    return "【假资料】阴影是被压抑的部分。"


def _transcript():
    return [
        {"character_id": "adler", "name": "阿尔弗雷德·阿德勒", "kind": "opening", "round": 0,
         "content": "人不是被过去推动的，是被目标拉着的。"},
        {"character_id": "jung", "name": "卡尔·荣格", "kind": "opening", "round": 0,
         "content": "阴影不会因为你不看它就消失。"},
    ]


def _build(*, is_opening: bool):
    with patch("framework.roundtable._retrieve_for_speaker", _fake_retrieve):
        return asyncio.run(build_speaker_messages(
            character=jung_character,
            topic="人应当如何面对自己的阴影",
            transcript=[] if is_opening else _transcript(),
            round_index=1,
            total_rounds=2,
            is_opening=is_opening,
        ))


def test_identity_is_first_message():
    """第一条必须是身份：人设正文 + 示例对话。"""
    messages = _build(is_opening=False)
    role, content = messages[0]
    assert role == "system"
    assert "卡尔·古斯塔夫·荣格" in content
    assert "{{user}}" in content  # 示例对话随人设一起在最前


def test_debate_rules_are_last_system_message():
    """辩论规则必须是最后一条 system，不能被排到身份那一条里。"""
    messages = _build(is_opening=False)
    system_indexes = [i for i, (role, _) in enumerate(messages) if role == "system"]
    last_system = system_indexes[-1]
    assert ROUNDTABLE_DEBATE_DIRECTIVE in messages[last_system][1]
    # 规则不能出现在第一条里（那会被人设与记录稀释）
    assert ROUNDTABLE_DEBATE_DIRECTIVE not in messages[0][1]


def test_language_anchor_sits_after_transcript_and_before_task():
    """语言锚必须夹在发言记录之后、本轮任务之前。"""
    messages = _build(is_opening=False)
    transcript = next(i for i, (_, content) in enumerate(messages) if "【完整发言记录" in content)
    anchor = next(i for i, (_, content) in enumerate(messages) if "【回到你自己的语言】" in content)
    task = next(i for i, (_, content) in enumerate(messages) if "这是第 1/2 轮交锋" in content)
    assert transcript < anchor < task
    assert messages[transcript][0] == "user"
    assert messages[anchor][0] == "system"
    assert "{{user}}" in messages[anchor][1]


def test_opening_has_no_anchor_and_no_transcript():
    """独立开场不传任何记录，也就不需要语言锚。"""
    messages = _build(is_opening=True)
    text = "\n".join(content for _, content in messages)
    assert "【完整发言记录" not in text
    assert "回到你自己的语言" not in text
    assert "独立开场陈述" in text


def test_message_roles_order():
    """可信后历史指令实际位于引用记录之后。"""
    messages = _build(is_opening=False)
    roles = [role for role, _ in messages]
    assert roles == ["system", "user", "system", "user"]


def test_skill_edits_update_anchor_and_post_history_without_python_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_loader, "SKILL_DIR", tmp_path)
    character = SimpleNamespace(
        id="skill_only", name="技能人物", zone="education", chroma_collection="persona_skill_only",
        role_prompt="旧 Python 人设", mes_example="旧 Python 示例",
        post_history_instructions="旧 Python 后历史",
    )
    path = tmp_path / "skill_only.md"

    def build_from_skill(example, post):
        sections = PersonaSections(
            role_prompt="技能人物人设", mes_example=example,
            post_history_instructions=post,
        )
        path.write_text(render_persona_skill(character, sections), encoding="utf-8")
        with patch("framework.roundtable._retrieve_for_speaker", _fake_retrieve):
            return asyncio.run(build_speaker_messages(
                character=character, topic="议题", transcript=_transcript(),
                round_index=1, total_rounds=2, is_opening=False,
            ))

    before = build_from_skill("技能示例第一版", "技能后历史第一版")
    after = build_from_skill("技能示例第二版增加内容", "技能后历史第二版增加内容")
    assert "技能示例第一版" in before[0][1] and "技能示例第一版" in before[2][1]
    assert "技能示例第二版增加内容" in after[0][1]
    assert "技能示例第二版增加内容" in after[2][1]
    assert "技能后历史第二版增加内容" in after[2][1]
    assert "技能后历史第二版增加内容" not in after[0][1]
    assert "旧 Python" not in str(after)

    cleared = build_from_skill("", "")
    assert "【回到你自己的语言】" not in str(cleared)
    assert "旧 Python" not in str(cleared)


def test_rule_directive_still_forbids_flattery():
    """改结构不能顺手改掉规则本身：不和稀泥、要反驳这些约束仍在。"""
    assert "不要泛泛而谈" in ROUNDTABLE_DEBATE_DIRECTIVE
    assert "反驳" in ROUNDTABLE_DEBATE_DIRECTIVE
