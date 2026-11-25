"""
LLM 输出 JSON 提取（src/core/llm_json）的解析容错测试。

这些形态都是真实踩到过的，不是假想：
- 纯 JSON / ```json 代码块 / 前后带解释：一直都要兼容
- list 包 chat 消息包 JSON 字符串：2026-11-25 全量评测时，备用判官经中转站
  返回这种结构，顶层不是 dict，旧逻辑判「无法解析」，那道题五维全 None
- 首元素是道歉、次元素才是 JSON：同一批里出现过，翻遍 list 才拿得到结果
"""

import json

import pytest

from src.core.llm_json import extract_json


PAYLOAD = {
    "faithfulness": {"score": 4, "reason": "论断有依据"},
    "relevancy": {"score": 5, "reason": "正面回应"},
    "context_precision": {"score": 4, "reason": "基本相关"},
    "key_points_coverage": {"score": 3, "reason": "缺一条"},
    "voice_consistency": {"score": 5, "reason": "口吻到位"},
}
PAYLOAD_TEXT = json.dumps(PAYLOAD, ensure_ascii=False)


def _assert_payload(obj):
    assert obj["faithfulness"]["score"] == 4
    assert set(obj) == set(PAYLOAD)


def test_plain_json():
    _assert_payload(extract_json(PAYLOAD_TEXT))


def test_fenced_code_block():
    _assert_payload(extract_json(f"```json\n{PAYLOAD_TEXT}\n```"))


def test_surrounded_by_prose():
    _assert_payload(extract_json(f"评估如下：\n{PAYLOAD_TEXT}\n以上。"))


def test_chat_message_wrapper():
    """中转站把整段回复再包成一条 chat 消息。"""
    raw = json.dumps([{"role": "assistant", "content": PAYLOAD_TEXT}], ensure_ascii=False)
    _assert_payload(extract_json(raw))


def test_chat_message_wrapper_with_fence():
    raw = json.dumps(
        [{"role": "assistant", "content": f"```json\n{PAYLOAD_TEXT}\n```"}],
        ensure_ascii=False,
    )
    _assert_payload(extract_json(raw))


def test_list_skips_unparsable_element():
    """首元素是道歉照样能翻到第二个元素，不能把消息壳本身当结果返回。"""
    raw = json.dumps(
        [
            {"role": "assistant", "content": "抱歉，我无法完成这次评估。"},
            {"role": "assistant", "content": PAYLOAD_TEXT},
        ],
        ensure_ascii=False,
    )
    _assert_payload(extract_json(raw))


def test_business_object_with_content_field_is_not_unwrapped():
    """带业务字段的对象即使有 content 键也不能当消息壳剥掉。"""
    obj = {"content": "某段正文", "score": 4}
    assert extract_json(json.dumps(obj, ensure_ascii=False)) == obj


@pytest.mark.parametrize("bad", ["", "   ", "完全不是 JSON", "[1, 2, 3]"])
def test_unparsable_raises(bad):
    with pytest.raises(ValueError):
        extract_json(bad)
