# -*- coding: utf-8 -*-
"""
test_history_aware_budget.py — 历史感知检索查询的字符预算

锁住三条契约：
1. 短历史：装配格式与旧实现逐字一致（零行为变化）
2. 超长历史：压缩后总长不超预算，且当前问题完整保留
3. 极端情况（压缩到底仍超）：查询至少完整包含当前问题
"""

from framework.retrieval_agent import _QUERY_CHAR_BUDGET, build_history_aware_query


def test_short_history_format_unchanged():
    """短历史走默认装配，格式与旧实现逐字相同"""
    query = "什么是共时性？"
    history = [("你好，我对心理学感兴趣", "你好！很高兴遇到对心灵世界好奇的朋友。")]
    out = build_history_aware_query(query, history)
    expected = (
        "以下是最近的对话（用于理解指代和上下文）：\n"
        "用户：你好，我对心理学感兴趣\n"
        "助手：你好！很高兴遇到对心灵世界好奇的朋友。\n"
        "\n当前用户问题：\n什么是共时性？"
    )
    assert out == expected


def test_long_history_compressed_within_budget():
    """超长历史被压缩到预算内，且当前问题完整保留"""
    query = "我的劣势功能如果是感觉，那在日常生活中会有哪些具体表现？会不会影响考研复习的专注度？"
    history = [
        ("长" * 100, "回" * 130),  # 回答超 120 会被旧逻辑截，此处触发预算压缩
        ("问" * 100, "答" * 130),
    ]
    out = build_history_aware_query(query, history)
    assert len(out) <= _QUERY_CHAR_BUDGET, f"压缩后仍超预算: {len(out)}"
    assert query in out, "当前问题必须完整保留"
    assert out.endswith(query)


def test_extreme_history_drops_but_query_survives():
    """压缩到底仍超（历史问题也巨长）：宁可丢历史，不能丢当前问题"""
    query = "这个问题本身比较长，描述了一个多步骤的具体情境，" * 3 + "最后想要一个答案。"
    history = [(q * 200, a * 200) for q, a in [("问", "答")] * 2]
    out = build_history_aware_query(query, history)
    assert out.endswith(query), "压缩到底时至少要保证当前问题完整"
    # 丢光历史后长度 = 模板 + 问题本身，必然 <= 预算（问题自身超长的极端除外）
    assert len(out) <= _QUERY_CHAR_BUDGET or len(query) > _QUERY_CHAR_BUDGET
