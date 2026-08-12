# -*- coding: utf-8 -*-
"""临时复现脚本：with_structured_output 在 DeepSeek 上的 400 错误"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import langchain_openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.config import settings

print("langchain_openai version:", langchain_openai.__version__)
print("model:", settings.llm_model)


class TestDecision(BaseModel):
    """测试结构化输出"""
    should_ask: bool = Field(description="是否需要追问")
    lead_in: str = Field(default="", description="引导语")
    questions: list[str] = Field(default_factory=list, description="问题列表")
    reasoning: str = Field(description="简要分析")


def main():
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,
        max_tokens=800,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )

    # 方式1：默认 method（function_calling）
    print("\n=== 方式1: with_structured_output 默认 ===")
    try:
        structured = llm.with_structured_output(TestDecision)
        result = structured.invoke([("system", "你是一个信息缺口检测器，请判断用户问题是否需要追问。"),
                                    ("user", "什么是人格面具？")])
        print("OK:", result)
    except Exception as e:
        print("FAIL:", type(e).__name__)
        print(str(e)[:2000])

    # 方式2：json_mode
    print("\n=== 方式2: with_structured_output method=json_mode ===")
    try:
        structured = llm.with_structured_output(TestDecision, method="json_mode")
        result = structured.invoke([("system", "你是一个信息缺口检测器，请判断用户问题是否需要追问。"),
                                    ("user", "什么是人格面具？")])
        print("OK:", result)
    except Exception as e:
        print("FAIL:", type(e).__name__)
        print(str(e)[:2000])

    # 方式3：普通 invoke + 手动 JSON 解析
    print("\n=== 方式3: 普通 invoke ===")
    try:
        response = llm.invoke([("system", "你是一个信息缺口检测器。请判断用户问题是否需要追问。"
                                          "请只输出 JSON，格式：{\"should_ask\": false, \"lead_in\": \"\", \"questions\": [], \"reasoning\": \"原因\"}"),
                               ("user", "什么是人格面具？")])
        print("OK:", response.content[:300])
    except Exception as e:
        print("FAIL:", type(e).__name__)
        print(str(e)[:2000])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
