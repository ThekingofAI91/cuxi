"""
圆桌发言 prompt 的结构约束

守的是「位置」而不是「措辞」：同一段文字放在上下文的不同位置，对生成的影响强度不同。
模型对越靠后的内容越敏感，所以：

- 身份（人设、示例对话）放最前 —— 它定义「我是谁」；
- 辩论规则（篇幅、要反驳、不和稀泥）放最后一条 system —— 它是「这一轮怎么写」，
  排在最前时会被几千字的完整发言记录稀释，最先丢的就是篇幅与交锋性；
- 语言锚必须夹在「别人的发言」之后、「本轮任务」之前 —— 整场辩论里最容易发生的
  不是立场漂移，而是措辞被上一位带跑（立场没变、腔调变了），只有落笔前那一段有效。

这些断言故意绑顺序而不是绑原文：措辞可以改，顺序错了就会退化。
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from framework.roundtable import (
    ROUNDTABLE_DEBATE_DIRECTIVE,
    build_speaker_messages,
)
from scenes.persona_chat.characters.jung import jung_character


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
    user = next(content for role, content in messages if role == "user")

    anchor = user.find("【回到你自己的语言】")
    transcript = user.find("【完整发言记录")
    task = user.find("这是第 1/2 轮交锋")

    assert transcript >= 0 and anchor > transcript
    assert task >= 0 and anchor < task
    assert "{{user}}" in user[anchor:]  # 锚里带的是本人示例


def test_opening_has_no_anchor_and_no_transcript():
    """独立开场不传任何记录，也就不需要语言锚。"""
    messages = _build(is_opening=True)
    user = next(content for role, content in messages if role == "user")
    assert "【完整发言记录" not in user
    assert "回到你自己的语言" not in user
    assert "独立开场陈述" in user


def test_message_roles_order():
    """消息角色顺序固定为 system → system → user，避免把规则塞到 user 之后。"""
    messages = _build(is_opening=False)
    roles = [role for role, _ in messages]
    assert roles == ["system", "system", "user"]


def test_rule_directive_still_forbids_flattery():
    """改结构不能顺手改掉规则本身：不和稀泥、要反驳这些约束仍在。"""
    assert "不要泛泛而谈" in ROUNDTABLE_DEBATE_DIRECTIVE
    assert "反驳" in ROUNDTABLE_DEBATE_DIRECTIVE
