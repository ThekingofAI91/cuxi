"""
RAGAS 评估脚手架 — 量化名人对话的检索与回答质量

⚠️ 重要：**本脚本不能在项目自己的 venv 里跑通。** ragas 会把 langchain-core 顶到 1.6.x，
   与项目锁定的版本冲突。实际可用的做法是两阶段解耦：
     1) 项目 venv 跑真实流水线，落盘问答与检索上下文（output/_phase1_pipeline.py）
     2) 独立 venv（Python 3.13 + ragas 0.4.3 + langchain-community 0.3.31）只读 JSON 打分
        （output/_phase2_ragas.py）
   本脚本是单机快速自测的参考实现，请在**独立环境**里运行，不要装进项目 venv。

准备（在独立环境里）：
    pip install ragas langchain-openai
    .env 中配置 LLM API（settings.llm_api_key / llm_base_url）

用法：
    python scripts/evaluate_ragas.py --character jung --questions 3

输出：
    1. 每条问题的回答 + 实际检索到的上下文（方便人工核对）
    2. RAGAS 指标：faithfulness（忠实度）/ answer_relevancy（回答相关性）/ context_precision（检索精度）
"""

import argparse
import asyncio
import json

SAMPLE_QUESTIONS = {
    "jung": ["阴影是什么？", "什么是集体潜意识？", "如何理解个体化过程？"],
    "adler": ["自卑与超越是什么关系？", "童年经历对一个人的人格影响有多大？"],
    "wangyangming": ["什么是知行合一？", "如何理解心即理？"],
    # 峰哥 / 张雪峰 已下架（大创参赛仅保留 3 位智者）
    # "fengge": ["失业了该怎么办？", "怎么看待穷这件事？"],
    # "zhangxuefeng": ["孩子想学新闻学，该报吗？", "理工科和文科怎么选？"],
}


async def run_one(character, char_config, question: str) -> dict:
    """执行一次完整流水线，返回回答与检索上下文（供 RAGAS 打分）

    走生产同一条路：framework/supervisor_agent.run_supervisor_agent
    （supervisor 自持 search_library 工具）。2026-09-30 前本脚本跑的是已删除的
    LangGraph 图，故历史 RAGAS 数值与新口径不严格可比，重跑后再对外引用。
    """
    from framework.runtime import set_scene_config
    from framework.supervisor_agent import run_supervisor_agent
    from src.core.state import AgentState

    set_scene_config(char_config)
    state: AgentState = {
        "query": question,
        "session_id": "eval-session",
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
        "enable_verification": character.enable_verification,
        "zone": getattr(character, "zone", "education"),
        "sampling": None,
        "post_history_directive": None,
        "user_memory": None,
        "stream_callback": None,
    }
    final = await run_supervisor_agent(state)
    docs = final.get("retrieved_docs", []) or []
    return {
        "question": question,
        "answer": final.get("final_answer", ""),
        "route_history": final.get("route_history", []),
        "contexts": [d.page_content[:600] for d in docs],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评估脚手架")
    parser.add_argument("--character", default="jung", help="角色 id，如 jung / adler / fengge / zhangxuefeng")
    parser.add_argument("--questions", type=int, default=3, help="最多跑几条问题")
    args = parser.parse_args()

    from dataclasses import replace

    from scenes.persona_chat.config import persona_chat_config

    character = persona_chat_config.characters.get(args.character)
    if not character:
        raise SystemExit(
            f"角色不存在: {args.character}，可用: {list(persona_chat_config.characters.keys())}"
        )

    char_config = replace(persona_chat_config, chroma_collection=character.chroma_collection)

    questions = SAMPLE_QUESTIONS.get(args.character, SAMPLE_QUESTIONS["jung"])[: args.questions]
    results = [await run_one(character, char_config, q) for q in questions]

    # RAGAS 指标（未安装时仅输出流水线结果）
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": r["question"],
                    "response": r["answer"],
                    "retrieved_contexts": r["contexts"],
                }
                for r in results
            ]
        )
        score = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        print("\n===== RAGAS 指标 =====")
        print(score)
    except ImportError:
        print("\n[提示] 未安装 ragas，已跳过指标计算。运行: pip install ragas")
    except Exception as e:  # ragas 版本 API 可能不同，脚手架尽量不阻塞主流程
        print(f"\n[提示] RAGAS 打分失败（版本 API 可能有差异）: {e}")

    print("\n===== 流水线结果 =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
