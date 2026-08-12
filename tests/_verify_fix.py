# -*- coding: utf-8 -*-
"""临时验证脚本：修复后多智能体图全链路跑通测试"""
import asyncio
import sys
from pathlib import Path

# 避免 PowerShell GBK 控制台 emoji 编码崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scenes.persona_chat.config import persona_chat_config
from framework.supervisor import set_scene_config, build_graph
from src.core.state import AgentState


async def run(query: str, session_id: str):
    print(f"\n{'='*60}\n[TEST] 查询: {query}\n{'='*60}")
    set_scene_config(persona_chat_config)
    character = persona_chat_config.characters.get("jung")
    graph = build_graph()

    state: AgentState = {
        "query": query,
        "session_id": session_id,
        "retrieved_docs": [],
        "analysis": "",
        "code_result": "",
        "verification": "",
        "final_answer": "",
        "history": [],
        "route_history": [],
        "next_agent": None,
        "error": None,
        "character_role_prompt": character.role_prompt,
        "stream_callback": None,
        "info_gap_questions": None,
    }

    accumulated = {}
    async for update in graph.astream(state, config={"recursion_limit": 25}, stream_mode="updates"):
        for node_name, node_output in update.items():
            accumulated.update(node_output)
            print(f"  [node] {node_name} -> keys={list(node_output.keys())}")

    print("  [route] ", " -> ".join(accumulated.get("route_history", [])))
    answer = (accumulated.get("final_answer") or "").strip()
    print(f"  [answer] {len(answer)} 字: {answer[:150]}...")
    if accumulated.get("info_gap_questions"):
        print(f"  [info_gap] 追问: {accumulated['info_gap_questions']}")
    return answer


async def main():
    # 1. 事实性问题（应直接走 RAG 链路，不追问）
    a1 = await run("什么是人格面具（Persona）？", "verify-fix-1")

    # 2. 模糊问题（应触发追问或正常回答）
    a2 = await run("我想了解荣格的阴影，该怎么面对它？", "verify-fix-2")

    print("\n[OK] 全链路验证完成")
    print(f"   Q1 回答长度: {len(a1)}")
    print(f"   Q2 回答长度: {len(a2)}")


if __name__ == "__main__":
    asyncio.run(main())
