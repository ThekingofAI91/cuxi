"""
对话操作与角色卡测试。

覆盖：
- 轻聊快速通道：_is_lightweight_turn 整句白名单判定（寒暄直答、真实提问不误判）
- 历史截断 / 弹轮：truncate_conversation_history / pop_last_turn（重新生成与历史编辑的后端基础）
- 世界书扫描窗口：最近几轮对话里的关键词也能命中词条（compose_lorebook_scan_text）
- 角色卡导出/导入：card_io.export_card / parse_card 往返与非法输入
"""
import pytest

from framework import supervisor
from framework.supervisor import (
    _is_lightweight_turn,
    truncate_conversation_history,
    pop_last_turn,
)
from scenes.persona_chat.card_io import export_card, parse_card, CARD_FORMAT
from scenes.persona_chat.models import CharacterDef
from scenes.persona_chat.prompt_builder import (
    compose_lorebook_scan_text,
    build_character_prompt,
    build_post_history_directive,
)


# ============================================================
# 轻聊快速通道
# ============================================================

@pytest.mark.parametrize("q", [
    "你好", "您好！", "哈喽哈喽", "在吗？",
    "你是谁", "你叫什么名字？", "自我介绍一下",
    "谢谢啦", "再见！", "晚安", "辛苦了",
    "真的吗？？", "好的", "嗯嗯", "哈哈哈", "666",
    "  Hello!!  ", "OK。",
])
def test_lightweight_turn_matches_social_phrases(q):
    assert _is_lightweight_turn(q), f"'{q}' 应命中轻聊通道"


@pytest.mark.parametrize("q", [
    "你好，我想问抑郁症怎么治疗",        # 寒暄开头 + 真实提问：绝不能吞进直答通道
    "荣格所说的集体无意识是什么？",
    "你好你好你好，我最近总是梦见蛇，这是什么征兆",
    "", "？？？这个问题我没想清楚，帮我分析下职业选择",
    "我是谁",  # 主语是用户自己，属于需要展开的哲学问题，不是"你是谁"
])
def test_lightweight_turn_rejects_real_questions(q):
    assert not _is_lightweight_turn(q), f"'{q}' 不应命中轻聊通道"


# ============================================================
# 历史截断 / 弹轮
# ============================================================

@pytest.fixture()
def history_session():
    """准备一个 4 轮历史的测试会话，结束后清理。"""
    sid = "test-controls-session"
    supervisor._conversation_history_store[sid] = [
        (f"问题{i}", f"回答{i}") for i in range(1, 5)
    ]
    supervisor._conversation_summaries.pop(sid, None)
    yield sid
    supervisor._conversation_history_store.pop(sid, None)
    supervisor._conversation_summaries.pop(sid, None)


def test_truncate_conversation_history(history_session):
    sid = history_session
    n = truncate_conversation_history(sid, 2)
    assert n == 2
    assert supervisor._conversation_history_store[sid] == [("问题1", "回答1"), ("问题2", "回答2")]
    # 截断到超出当前长度是空操作；不存在的会话返回 -1
    assert truncate_conversation_history(sid, 10) == 2
    assert truncate_conversation_history("no-such-session", 3) == -1


def test_truncate_to_zero_clears(history_session):
    sid = history_session
    assert truncate_conversation_history(sid, 0) == 0
    assert supervisor._conversation_history_store[sid] == []


def test_pop_last_turn(history_session):
    sid = history_session
    assert pop_last_turn(sid, expect_query="问题4") is True
    turns = supervisor._conversation_history_store[sid]
    assert turns[-1] == ("问题3", "回答3")
    # 校验失败时不弹（防止并发误删）
    assert pop_last_turn(sid, expect_query="问题1") is False
    assert len(turns) == 3


def test_pop_last_turn_empty_session():
    sid = "test-pop-empty"
    supervisor._conversation_history_store.pop(sid, None)
    assert pop_last_turn(sid) is False


# ============================================================
# 世界书扫描窗口
# ============================================================

def _char_with_lorebook():
    return CharacterDef(
        id="t", name="测试人物", description="", role_prompt="你是测试人物。",
        chroma_collection="persona_t", data_source="",
        lorebook=[
            {"keyword": " shadow", "content": "阴影是你不愿承认的那部分自己。"},
        ],
    )


def test_lorebook_scan_text_contains_history():
    history = [("我们刚才聊的阴影是什么", "阴影是人格中被压抑的部分。"), ("后来呢", "后来你谈到了整合。")]
    text = compose_lorebook_scan_text("那它要怎么面对", history)
    assert "阴影" in text and "那它要怎么面对" in text
    # 扫描窗口只取最近几轮
    long_history = [(f"q{i}", f"a{i}") for i in range(10)]
    text2 = compose_lorebook_scan_text("q", long_history)
    assert "a0" not in text2 and "a9" in text2


def test_lorebook_triggers_via_recent_history():
    """关键词只出现在最近几轮对话里（不在本轮消息中），词条也应被注入。"""
    char = _char_with_lorebook()
    scan_text = compose_lorebook_scan_text(
        "那它要怎么面对", [("我们刚才聊的 shadow 是什么", "……")]
    )
    prompt = build_character_prompt(char, query="那它要怎么面对", scan_text=scan_text)
    assert "阴影是你不愿承认的那部分自己" in prompt
    # 后历史位置的词条（position=after_history）同样凭扫描窗口命中
    char.lorebook.append({"keyword": "shadow", "content": "后历史位置词条", "position": "after_history"})
    post = build_post_history_directive(char, query="那它要怎么面对", scan_text=scan_text)
    assert "后历史位置词条" in post
    # before_char 词条不进后历史指令（位置语义不混用）
    assert "阴影是你不愿承认的那部分自己" not in post


def test_lorebook_not_triggered_without_history_hit():
    char = _char_with_lorebook()
    prompt = build_character_prompt(char, query="今天天气不错")
    assert "阴影是你不愿承认的那部分自己" not in prompt


# ============================================================
# 角色卡导出 / 导入
# ============================================================

def _full_card_char():
    return CharacterDef(
        id="demo", name="演示人物", description="简介", role_prompt="你是演示人物。",
        chroma_collection="persona_demo", data_source="",
        first_mes="坐。", mes_example="{{user}}: 你好\n{{char}}: 嗯。",
        scenario="书房夜谈", personality="沉稳、毒舌",
        lorebook=[
            {"keyword": "阴影", "content": "……", "position": "before_char"},
            {"keyword": "", "content": "缺关键词，导出时应被丢弃"},
        ],
        zone="education", theme="paper", enable_verification=True,
    )


def test_card_export_parse_roundtrip():
    char = _full_card_char()
    card = export_card(char, background="背景全文")
    assert card["format"] == CARD_FORMAT
    fields = parse_card(card)
    assert fields["name"] == "演示人物"
    assert fields["role_prompt"] == "你是演示人物。"
    assert fields["first_mes"] == "坐。"
    assert fields["zone"] == "education"
    assert fields["enable_verification"] is True
    assert fields["background"] == "背景全文"
    # 缺关键词的词条被清洗掉
    assert len(fields["lorebook"]) == 1
    assert fields["lorebook"][0]["keyword"] == "阴影"


def test_card_rejects_invalid():
    with pytest.raises(ValueError):
        parse_card("not a dict")
    with pytest.raises(ValueError):
        parse_card({"format": CARD_FORMAT, "name": ""})
    # 既无人设也无背景
    with pytest.raises(ValueError):
        parse_card({"format": CARD_FORMAT, "name": "某人"})
    # 不认识的格式 / 过新的版本
    with pytest.raises(ValueError):
        parse_card({"format": "someone-else", "name": "x", "role_prompt": "y"})
    with pytest.raises(ValueError):
        parse_card({"format": CARD_FORMAT, "version": 99, "name": "x", "role_prompt": "y"})


def test_card_importable_without_background_when_role_prompt_present():
    fields = parse_card({"format": CARD_FORMAT, "name": "某人", "role_prompt": "你是某人。"})
    assert fields["role_prompt"] == "你是某人。"
    assert fields["background"] == ""
    assert fields["lorebook"] == []
