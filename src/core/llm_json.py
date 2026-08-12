"""
llm_json.py — LLM 输出 JSON 提取工具

背景：DeepSeek API（deepseek-v4-flash）不支持 langchain-openai 1.3.5
`with_structured_output` 内部使用的 response_format（json_schema 类型），
会返回 400 错误（"This response_format type is unavailable now"）。

解决方案：改用普通 invoke + prompt 要求输出 JSON + 本工具容错解析，
兼容三种输出形态：纯 JSON、```json 代码块、混在文本中的 JSON。
"""

import json
import re


def extract_json(text: str) -> dict:
    """
    从 LLM 输出文本中提取 JSON 对象（dict）。

    容错顺序：
    1. 整体直接 json.loads
    2. 提取 ```json ... ``` 代码块
    3. 提取首个平衡的 { ... } 大括号片段

    Raises:
        ValueError: 无法解析出合法 JSON 对象时
    """
    if not text or not text.strip():
        raise ValueError("LLM 输出为空")

    text = text.strip()

    # 1. 直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. 代码块提取（```json ... ```）
    block_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if block_match:
        try:
            obj = json.loads(block_match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. 平衡大括号提取（LLM 输出前后带解释文字时兜底）
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"无法从 LLM 输出中解析 JSON 对象: {text[:200]}")
