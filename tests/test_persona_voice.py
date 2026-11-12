"""
人物差异化 + 回答完整性相关测试。

覆盖：
- 口吻要求只从人物技能读取，不再使用场景级人设覆盖
- _generate_direct_response 在角色模式下原样注入技能装配结果，不重复叠加口吻
- 角色回答 token 上限已放宽（不再 800，避免尾部被截断）
"""
import asyncio
from unittest.mock import MagicMock, patch

from scenes.persona_chat.config import persona_chat_config
from scenes.persona_chat.prompt_builder import build_character_prompt, build_post_history_directive
from scenes.persona_chat.skill_loader import resolve_persona_sections


def test_persona_voice_is_owned_by_each_skill():
    """人物技能拥有全部口吻内容，场景配置不再定义共享人设。"""
    assert not hasattr(persona_chat_config, "persona_voice_directive")
    for character in persona_chat_config.characters.values():
        sections = resolve_persona_sections(character)
        assert sections.voice_directive.strip()
        assert sections.voice_directive in build_character_prompt(character)
        assert build_post_history_directive(character) == sections.post_history_instructions


def test_generate_direct_response_injects_voice_and_relaxed_tokens():
    """角色模式：role_prompt + 差异化铁律都进系统提示，且 max_tokens 放宽。"""
    from framework import runtime

    captured = {}

    async def fake_astream(messages):
        captured["messages"] = messages
        yield MagicMock(content="你好，我是测试人物。")

    async def fake_callback(token):
        captured.setdefault("tokens", []).append(token)

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream

    character = persona_chat_config.characters["jung"]
    role = build_character_prompt(character)
    voice = resolve_persona_sections(character).voice_directive

    with patch.object(runtime, "get_chat_llm", return_value=fake_llm) as mk:
        asyncio.run(
            runtime._generate_direct_response(
                "你怎么看教育？", [], role, fake_callback,
                session_id="test-session",
            )
        )

    # get_chat_llm 被调用，且 token 上限已放宽（角色模式应 > 800）
    assert mk.called
    _, kwargs = mk.call_args
    assert kwargs.get("max_tokens", 0) >= 1500

    sys_msgs = [m[1] for m in captured["messages"] if m[0] == "system"]
    joined = "\n".join(sys_msgs)
    assert role in joined, "人物 role_prompt 必须出现在系统提示中"
    assert joined == role, "runtime 不应在完整角色技能外再补共享人设"
    assert joined.count(voice) == 1, "技能口吻应只注入一次"


def test_direct_messages_keep_skill_post_history_after_history():
    from framework.runtime import build_direct_messages

    character = persona_chat_config.characters["jung"]
    post = build_post_history_directive(character)
    messages = build_direct_messages(
        "本轮问题", history=[("之前的问题", "之前的回答")],
        character_role_prompt=build_character_prompt(character), post_history_directive=post,
    )
    history_index = next(i for i, (_, text) in enumerate(messages) if "之前的问题" in text)
    post_index = next(i for i, (_, text) in enumerate(messages) if text == post)
    assert history_index < post_index < len(messages) - 1
