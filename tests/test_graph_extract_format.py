"""抽取结果解析单测：6 列（带类型）与 4 列旧格式兜底。

2026-10-21 给抽取 prompt 加了两个类型列（头类型/尾类型），目的是认出「事件」节点
——事件节点要进闸门，不许往外扩张。解析器因此要同时吃两种格式：新 6 列，以及模型
漏写类型时退回的旧 4 列。这里把两种格式、证据自带竖线、非法类型名都锁住。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import knowledge_graph as kg


class _Resp:
    def __init__(self, content: str):
        self.content = content


def _extract(monkeypatch, content: str):
    async def fake_ainvoke(llm, msgs, fallback_llm=None):
        return _Resp(content)

    monkeypatch.setattr("src.core.llm.ainvoke_nonempty", fake_ainvoke)
    return asyncio.run(kg._llm_extract_triples(object(), "资料文本"))


def test_six_column_line_keeps_types(monkeypatch):
    out = _extract(monkeypatch, "王阳明 | 人物 | 经历 | 龙场悟道 | 事件 | 谪至贵州龙场，夜半大悟")
    assert out == [{
        "head": "王阳明", "head_type": "人物",
        "relation": "经历",
        "tail": "龙场悟道", "tail_type": "事件",
        "evidence": "谪至贵州龙场，夜半大悟",
    }]


def test_four_column_line_still_parses(monkeypatch):
    """模型漏写类型列时退回旧格式，不能因为多要了两列就把整批抽成空。"""
    out = _extract(monkeypatch, "集体潜意识 | 包含 | 原型 | 原型是集体潜意识的内容")
    assert len(out) == 1
    assert out[0]["head"] == "集体潜意识" and out[0]["relation"] == "包含"
    assert out[0]["tail"] == "原型"
    assert out[0]["head_type"] == "" and out[0]["tail_type"] == ""
    assert out[0]["evidence"] == "原型是集体潜意识的内容"


def test_evidence_may_contain_pipes(monkeypatch):
    """证据是最后一列，允许自带竖线；不能因此被切成多行/丢半句。"""
    out = _extract(monkeypatch, "心即理 | 概念 | 出自 | 传习录 | 著作 | 卷上 | 心即理也")
    assert len(out) == 1
    assert out[0]["tail"] == "传习录"
    assert out[0]["evidence"] == "卷上 | 心即理也"


def test_unknown_type_name_is_dropped(monkeypatch):
    """类型名不在枚举里（如「理论」）时保留这条关系，但类型留空——不猜。"""
    out = _extract(monkeypatch, "本我 | 理论 | 对立于 | 超我 | 理论 | 本我与超我的冲突")
    assert len(out) == 1
    assert out[0]["head_type"] == "" and out[0]["tail_type"] == ""


def test_self_loop_and_short_lines_are_skipped(monkeypatch):
    out = _extract(monkeypatch, "王阳明 | 人物 | 是 | 王阳明 | 人物 | 自指\n只有一列\n")
    assert out == []
