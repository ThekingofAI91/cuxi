"""回填 `cont_tail`：给断在句中的块补上「下一块的第一句」。

用法：
    .venv/Scripts/python.exe scripts/backfill_cont_tail.py                # dry-run，只看统计
    .venv/Scripts/python.exe scripts/backfill_cont_tail.py --apply        # 真写
    .venv/Scripts/python.exe scripts/backfill_cont_tail.py --only persona_jung --show 10

前提只有两条，都实测过（探针 `output/_stitch_probe.py`、`output/_ov_dist.log`）：
1. chunk id 尾号 = 文本顺序（按 source 分组后严格递增）
2. 相邻块带 100 字重叠——`chunker._add_overlap` 取 `prev[-100:]` 贴到下一块开头，
   全库 5422 对相邻块里 5412 对逐字吻合（99.8%）

所以 `nxt[:100] == prev[-100:]` 成立，就说明这两块文本上真的连着，`nxt[100:]` 的第一句
就是上一块缺的那半句；不成立说明中间删过块（前置页过滤、去重），不该接。

只写 metadata，**不改 documents、不重新 embed**——不需要重建 collection，图谱里指向
chunk id 的证据全部继续有效。
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.stitch import is_markdown, needs_continuation

NUM = re.compile(r"_(\d+)$")
SENT = "。！？…"
OVERLAP = 100         # 与 config.chunk_overlap 一致


def first_sentence(text: str) -> str:
    """取第一句（含句末标点）；整段没有句末标点就整段返回。"""
    text = text.lstrip()
    for i, ch in enumerate(text):
        if ch in SENT:
            return text[: i + 1]
    return text


def next_sentence(prev: str, nxt: str, overlap: int = OVERLAP) -> str:
    """下个块里接着 prev 的那第一句；两块不连续或没有正文则返回空串。"""
    if len(prev) <= overlap or nxt[:overlap] != prev[-overlap:]:
        return ""
    return first_sentence(nxt[overlap:])


def build_updates(ids, docs, metas):
    """算出 [(id, 新 metadata)]，以及统计。

    每个块都写：有续接句就写值，没有就写空串。这样重复跑是幂等的，
    判据改严之后旧值也会被自然覆盖掉。
    """
    by_source = defaultdict(list)
    for cid, doc, meta in zip(ids, docs, metas):
        mt = NUM.search(cid)
        if not mt or not doc:
            continue
        by_source[(meta or {}).get("source", "")].append((int(mt.group(1)), cid, doc, meta))

    updates = []
    stat = {"need": 0, "filled": 0, "no_next": 0, "no_overlap": 0, "skipped_md": 0, "samples": []}

    for items in by_source.values():
        items.sort(key=lambda x: x[0])
        for i, (n, cid, doc, meta) in enumerate(items):
            cont = ""
            nxt_id = ""
            if is_markdown(meta):
                stat["skipped_md"] += 1
            elif needs_continuation(doc):
                stat["need"] += 1
                if i + 1 < len(items):
                    nxt_id, nxt_doc = items[i + 1][1], items[i + 1][2]
                    cont = next_sentence(doc, nxt_doc)
                    stat["filled" if cont else "no_overlap"] += 1
                else:
                    stat["no_next"] += 1

            new_meta = dict(meta or {})
            new_meta["cont_tail"] = cont
            new_meta["cont_from"] = nxt_id
            updates.append((cid, new_meta))

            if cont and len(stat["samples"]) < 8:
                stat["samples"].append((cid, doc[-40:], cont[:50]))

    return updates, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真写库；不加只干跑")
    ap.add_argument("--only", default="", help="只处理某个 collection")
    ap.add_argument("--show", type=int, default=5, help="打印几条样例")
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    from framework.runtime import get_chroma_client

    client = get_chroma_client()
    names = [c.name for c in client.list_collections() if c.name.startswith("persona_")]
    if args.only:
        names = [n for n in names if n == args.only]
    names.sort()

    total_need = total_filled = 0
    for name in names:
        col = client.get_collection(name)
        got = col.get(include=["documents", "metadatas"])
        updates, stat = build_updates(got["ids"], got["documents"], got["metadatas"])
        total_need += stat["need"]
        total_filled += stat["filled"]

        pct = 100 * stat["filled"] / stat["need"] if stat["need"] else 0
        print(f"\n{name}: 全库 {len(got['ids'])} 块 | 需要续接 {stat['need']}"
              f" | 可填 {stat['filled']} ({pct:.1f}%)"
              f" | 尾块无下一条 {stat['no_next']} | 不连续跳过 {stat['no_overlap']}"
              f" | markdown 跳过 {stat['skipped_md']}")
        for cid, tail, cont in stat["samples"][: args.show]:
            print(f"    {cid}: ...{tail!r}")
            print(f"      → {cont!r}")

        if args.apply and updates:
            for s in range(0, len(updates), args.batch):
                chunk = updates[s: s + args.batch]
                col.update(ids=[c[0] for c in chunk], metadatas=[c[1] for c in chunk])
            print(f"    已写入 {len(updates)} 条 metadata（未改 documents，未重新 embed）")

    pct = 100 * total_filled / total_need if total_need else 0
    print(f"\n合计：需要续接 {total_need} 块，可填 {total_filled} ({pct:.1f}%)"
          f"{'，已写入' if args.apply else '，dry-run 未写'}")


if __name__ == "__main__":
    main()
