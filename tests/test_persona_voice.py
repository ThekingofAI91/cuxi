"""
人物差异化 + 回答完整性相关测试。

覆盖：
- persona_voice_directive 已配置，且 direct_response_system_prompt 不再把角色框成"分析助手"
- _generate_direct_response 在角色模式下，把人物 role_prompt + 差异化铁律注入系统提示
- 角色回答 token 上限已放宽（不再 800，避免尾部被截断）
"""
import asyncio
from unittest.mock import MagicMock, patch

from scenes.persona_chat.config import persona_chat_config


def test_persona_voice_directive_present():
    """差异化铁律必须存在，且基础提示不再稀释人设（旧版"分析助手"框架已移除）。"""
    assert persona_chat_config.persona_voice_directive.strip()
    # 新框架：以"你就是这位历史人物本人"开头，肯定式人设
    assert persona_chat_config.direct_response_system_prompt.startswith("你就是这位历史人物本人")
    # 旧版把角色框成"分析助手"，会稀释身份、导致人物雷同——必须已消失
    assert "你是一个基于历史人物著作的分析助手" not in persona_chat_config.direct_response_system_prompt
    # 铁律里应明确禁止助手口吻与通用套话
    assert "AI 助手" in persona_chat_config.persona_voice_directive
    assert "通用建议" in persona_chat_config.persona_voice_directive


def test_generate_direct_response_injects_voice_and_relaxed_tokens():
    """角色模式：role_prompt + 差异化铁律都进系统提示，且 max_tokens 放宽。"""
    from framework import supervisor

    captured = {}

    async def fake_astream(messages):
        captured["messages"] = messages
        yield MagicMock(content="你好，我是测试人物。")

    async def fake_callback(token):
        captured.setdefault("tokens", []).append(token)

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream

    role = "你是测试人物张三，终生反对一切权威，说话尖刻直接。"

    with patch.object(supervisor, "get_chat_llm", return_value=fake_llm) as mk:
        asyncio.run(
            supervisor._generate_direct_response(
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
    assert persona_chat_config.persona_voice_directive in joined, "差异化铁律必须注入系统提示"
