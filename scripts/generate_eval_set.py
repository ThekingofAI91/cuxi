"""
generate_eval_set.py — 评估集大扩充：从各角色语料自动生成评估题

对教育区 3 个角色库采样 chunk，用 LLM 生成"基于资料"的评估题（问题 + 关键要点 +
难度 + 来源 chunk 摘录），落盘 tests/eval_dataset_expanded.json（人工可审可改）。
与既有 tests/eval_dataset.py 的 12 道荣格题合并后作为全量回归基线。

用法：
    .venv/Scripts/python.exe scripts/generate_eval_set.py             # 生成/续传
    .venv/Scripts/python.exe scripts/generate_eval_set.py --per-char 12

设计：
- 每批 3 条 chunk → 2 题（题目必须能从所给资料中找到答案，防"无据可查"题）
- 增量落盘 + 续传：中断重跑不浪费已生成的题
- 节流：批间 sleep，防中转 429
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings

OUT_PATH = Path("tests/eval_dataset_expanded.json")

# 各教育区角色的目标题量（jung 已有 12 题内置，这里补充覆盖面）
TARGETS = {
    "persona_jung": 14,
    "persona_adler": 14,
    "persona_wangyangming": 14,
}
CHUNKS_PER_BATCH = 3
QUESTIONS_PER_BATCH = 2

GEN_PROMPT = """你是 RAG 评估题出题专家。根据下面的知识库片段，出 {n} 道用户可能会问的评估题。

要求：
- 题目必须是这些片段**能回答**的（答案能在片段中找到依据），不要出片段之外的题
- 模拟真实用户的自然问法（中文，10-40 字），覆盖不同难度：easy（直接问概念）/ medium（问区别、原因）/ hard（综合、应用、象征）
- key_points：3-5 个关键要点（判官会检查回答是否覆盖，用短语）
- ground_truth：依据片段整理的标准答案（1-3 句）

知识库片段：
{chunks}

输出严格 JSON（不要输出 JSON 以外的任何文字）：
{{"questions": [{{"question": "…", "key_points": ["…"], "ground_truth": "…", "difficulty": "easy|medium|hard"}}]}}"""


def _load_done() -> list[dict]:
    if OUT_PATH.exists():
        try:
            return json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save(items: list[dict]) -> None:
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse(content: str) -> list[dict]:
    from src.core.llm_json import extract_json

    try:
        obj = extract_json(content)
        out = []
        for q in (obj or {}).get("questions", []):
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", "")).strip()
            kps = [str(k).strip() for k in (q.get("key_points") or []) if str(k).strip()]
            gt = str(q.get("ground_truth", "")).strip()
            diff = q.get("difficulty", "medium")
            if question and kps and gt and diff in ("easy", "medium", "hard"):
                out.append({"question": question, "key_points": kps,
                            "ground_truth": gt, "difficulty": diff})
        return out
    except Exception:
        return []


async def gen_for_collection(client, col_name: str, want: int, delay: float) -> list[dict]:
    from src.core.llm import get_chat_llm, ainvoke_nonempty

    col = client.get_or_create_collection(col_name, metadata={"hnsw:space": "cosine"})
    total = col.count()
    char_id = col_name.replace("persona_", "")
    items = _load_done()
    done_keys = {it["question"] for it in items}

    llm = get_chat_llm(temperature=0.6, max_tokens=900)
    got = sum(1 for it in items if it.get("character") == char_id)
    offset = 0
    batches = 0
    while got < want and offset < total:
        raw = col.get(include=["documents", "metadatas"], limit=CHUNKS_PER_BATCH, offset=offset)
        docs = [d for d in (raw.get("documents") or []) if d and len(d) > 80]
        offset += CHUNKS_PER_BATCH
        if len(docs) < 2:
            continue
        chunks_text = "\n\n".join(
            f"[片段{j+1}]（来源:{(m or {}).get('source','')} | 章节:{(m or {}).get('heading','')}）\n{t[:600]}"
            for j, (t, m) in enumerate(zip(docs, raw.get("metadatas") or [{}] * len(docs)))
        )
        try:
            resp = await ainvoke_nonempty(llm, [("user", GEN_PROMPT.format(
                n=QUESTIONS_PER_BATCH, chunks=chunks_text))])
            new_items = _parse(getattr(resp, "content", "") or "")
        except Exception as e:
            print(f"  [{col_name}] ⚠️ 生成失败（跳过该批）: {str(e)[:80]}")
            new_items = []
        added = 0
        for q in new_items:
            if q["question"] not in done_keys and got < want:
                items.append({**q, "character": char_id, "collection": col_name})
                done_keys.add(q["question"])
                got += 1
                added += 1
        batches += 1
        _save(items)
        print(f"  [{col_name}] 第 {batches} 批 → +{added} 题（累计 {got}/{want}）")
        if delay > 0:
            await asyncio.sleep(delay)
    return [it for it in items if it.get("character") == char_id]


async def main() -> None:
    parser = argparse.ArgumentParser(description="评估集自动扩充")
    parser.add_argument("--per-char", type=int, default=None, help="覆盖每角色目标题量")
    parser.add_argument("--delay", type=float, default=4.0, help="批间延迟秒数")
    args = parser.parse_args()

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    t0 = time.time()
    for col_name, want in TARGETS.items():
        n = args.per_char or want
        print(f"\n[gen] === {col_name}（目标 {n} 题）===")
        await gen_for_collection(client, col_name, n, args.delay)

    items = _load_done()
    print(f"\n[gen] 完成：共 {len(items)} 题 → {OUT_PATH}"
          f" | 耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print("[gen] 建议人工过一遍题目质量（删除歧义项），再跑全量基线评估")


if __name__ == "__main__":
    asyncio.run(main())
