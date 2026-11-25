"""
llm_json.py — LLM 输出 JSON 提取工具

背景：DeepSeek API（deepseek-v4-flash）不支持 langchain-openai 1.3.5
`with_structured_output` 内部使用的 response_format（json_schema 类型），
会返回 400 错误（"This response_format type is unavailable now"）。

解决方案：改用普通 invoke + prompt 要求输出 JSON + 本工具容错解析，
兼容四种输出形态：纯 JSON、```json 代码块、混在文本中的 JSON、
以及被中转站额外包了一层 chat 消息的 JSON。
"""

import json
import re

_MAX_UNWRAP_DEPTH = 3

# 消息壳里允许出现的字段。命中集合、且 content 是字符串，就当作
# 「把整段回复又包了一层 chat 消息」解包；带别的字段说明是正常业务对象。
_MESSAGE_SHELL_KEYS = {"role", "content", "name", "finish_reason", "index"}


def _unwrap(obj, depth: int = 0):
    """把多余的包装层剥掉，拿到真正的业务对象；剥不出来返回 None。

    2026-11-25 实测：中转站的备用模型偶尔把整段回答再包一层——
        [{"role": "assistant", "content": "{\"faithfulness\": {...}}"}]
    顶层变成 list，`json.loads` 成功但不是 dict，旧逻辑直接判「无法解析」，
    该题五维全 None。这里统一处理三种包装：
      1. 顶层是 list  → 逐个元素试，取第一个能剥出 dict 的
      2. chat 消息壳  → 递归解析 content 字符串
      3. 都不是       → 原样返回，交给调用方判断字段是否齐全
    """
    if depth > _MAX_UNWRAP_DEPTH:
        return None

    if isinstance(obj, list):
        for item in obj:
            got = _unwrap(item, depth + 1)
            if got is not None:
                return got
        return None

    if not isinstance(obj, dict):
        return None

    content = obj.get("content")
    is_shell = isinstance(content, str) and set(obj.keys()) <= _MESSAGE_SHELL_KEYS
    if is_shell:
        # 壳里的 content 解析不出来就返回 None，让上层继续翻 list 的下一个元素；
        # 把壳本身当结果返回会让「首元素是道歉、次元素才是 JSON」的响应白丢
        if content.strip():
            return _extract_from_text(content, depth + 1)
        return None

    return obj


def _extract_from_text(text: str, depth: int = 0):
    """从一段文本里按「直接解析 → 代码块 → 平衡大括号」的顺序找 dict。"""
    if depth > _MAX_UNWRAP_DEPTH or not text or not text.strip():
        return None

    text = text.strip()

    # 1. 直接解析
    try:
        got = _unwrap(json.loads(text), depth)
        if got is not None:
            return got
    except json.JSONDecodeError:
        pass

    # 2. 代码块提取（```json ... ```）
    block_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if block_match:
        try:
            got = _unwrap(json.loads(block_match.group(1).strip()), depth)
            if got is not None:
                return got
        except json.JSONDecodeError:
            pass

    # 3. 平衡大括号提取（LLM 输出前后带解释文字时兜底）
    start = text.find("{")
    if start != -1:
        depth_cnt = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth_cnt += 1
            elif text[i] == "}":
                depth_cnt -= 1
                if depth_cnt == 0:
                    try:
                        got = _unwrap(json.loads(text[start:i + 1]), depth)
                        if got is not None:
                            return got
                    except json.JSONDecodeError:
                        break

    return None


def extract_json(text: str) -> dict:
    """
    从 LLM 输出文本中提取 JSON 对象（dict）。

    容错顺序：
    1. 整体直接 json.loads（若是 list 或消息壳，先剥包装）
    2. 提取 ```json ... ``` 代码块
    3. 提取首个平衡的 { ... } 大括号片段
    4. 包装层里的 content 字符串再走一遍上述流程

    Raises:
        ValueError: 无法解析出合法 JSON 对象时
    """
    if not text or not text.strip():
        raise ValueError("LLM 输出为空")

    obj = _extract_from_text(text)
    if obj is None:
        raise ValueError(f"无法从 LLM 输出中解析 JSON 对象: {text[:200]}")
    return obj
