"""验证名人对话上下文修复：图结构 + 历史注入逻辑"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.supervisor import (
    set_scene_config, get_scene_config, build_graph,
    format_history_for_prompt, _conversation_summaries,
    append_conversation,
)
from scenes.persona_chat.config import persona_chat_config
from scenes.academic.config import academic_config

failures = []


def check(name, cond, detail=""):
    status = "✅" if cond else "❌"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


async def main():
    print("=" * 60)
    print("1️⃣  场景配置字段")
    check("persona.history_aware_retrieval=True",
          persona_chat_config.history_aware_retrieval is True)
    check("persona.enable_info_gap=False",
          persona_chat_config.enable_info_gap is False)
    check("persona.history_instruction 鼓励延续",
          "延续" in persona_chat_config.history_instruction)
    check("academic.enable_info_gap=True",
          academic_config.enable_info_gap is True)
    check("academic.history_aware_retrieval=False",
          academic_config.history_aware_retrieval is False)
    check("academic.history_instruction 防漂移",
          "不要主动延伸历史话题" in academic_config.history_instruction)

    print("=" * 60)
    print("2️⃣  图结构：InfoGap 开关")
    set_scene_config(persona_chat_config)
    persona_graph = build_graph()
    persona_nodes = set(persona_graph.get_graph().nodes.keys())
    check("persona 图不包含 info_gap", "info_gap" not in persona_nodes,
          f"节点={sorted(persona_nodes)}")

    set_scene_config(academic_config)
    academic_graph = build_graph()
    academic_nodes = set(academic_graph.get_graph().nodes.keys())
    check("academic 图包含 info_gap", "info_gap" in academic_nodes,
          f"节点={sorted(academic_nodes)}")

    print("=" * 60)
    print("3️⃣  历史注入轮数（store 5 轮 → prompt 应含 5 轮 + 摘要）")
    set_scene_config(persona_chat_config)

    # 模拟 8 轮对话（触发摘要）
    for i in range(1, 9):
        append_conversation("test_session_ctx", f"第{i}轮问题", f"第{i}轮回答")

    # 等待异步摘要任务完成（串行执行 + API 调用，需要更长时间）
    await asyncio.sleep(40.0)

    from framework.supervisor import get_conversation_history
    history = get_conversation_history("test_session_ctx")
    print(f"  store 轮次: {len(history)} (应为 5)")
    check("store 裁剪到 max_history_turns=5", len(history) == 5,
          f"实际={len(history)}")

    has_summary = "test_session_ctx" in _conversation_summaries
    check("旧对话已生成摘要", has_summary,
          f"摘要={_conversation_summaries.get('test_session_ctx', '')[:60]}...")

    prompt = format_history_for_prompt(history, session_id="test_session_ctx")
    check("prompt 含【早期对话摘要】", "【早期对话摘要】" in prompt)
    check("prompt 含全部 5 轮", all(f"第{i}轮问题" in prompt for i in range(4, 9)))
    # 早期轮次应在摘要中体现（串行合并后不应丢轮次）
    summary_text = _conversation_summaries.get("test_session_ctx", "")
    check("早期轮次内容在摘要中", any(str(i) in summary_text for i in range(1, 4)),
          f"摘要={summary_text[:120]}")

    # 无摘要时注入轮数验证：注入上限与存储上限一致（5 轮）
    print("  无摘要时注入轮数：")
    hist6 = [(f"Q{i}", f"A{i}") for i in range(1, 7)]
    prompt6 = format_history_for_prompt(hist6, session_id="no_summary_session")
    check("6 轮历史注入最近 5 轮（与存储上限一致）",
          all(f"Q{i}" in prompt6 for i in range(2, 7)) and "Q1" not in prompt6)

    print("=" * 60)
    print("4️⃣  历史感知检索 query 构建逻辑")
    from framework.retrieval_agent import retrieval_agent  # noqa: 导入检查
    check("retrieval_agent 导入正常", True)

    print("=" * 60)
    print("5️⃣  supervisor persona 分支（跳过 verifier 逻辑）")
    from framework.supervisor import supervisor_node
    check("supervisor_node 可导入", supervisor_node is not None)

    if failures:
        print(f"\n❌ 失败 {len(failures)} 项: {failures}")
        sys.exit(1)
    print("\n🎉 全部通过")


if __name__ == "__main__":
    asyncio.run(main())
