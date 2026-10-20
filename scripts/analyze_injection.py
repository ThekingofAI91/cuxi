# -*- coding: utf-8 -*-
"""
analyze_injection.py — 把评测结果里的检索过程读成人能看的表

用途：回答「这个配置为什么涨/跌」。判官只给终态五维分，中间过程在
`retrieval_trace` 里（boards / fused / rerank / graph / boundary）。

两种输入都支持：
  1. 新口径（带 retrieval_trace）：逐条读轨迹字段
  2. 旧口径（只有 contexts 字符串）：按长度启发式切正文/证据，只出能算的那几项

用法：
    python scripts/analyze_injection.py
    python scripts/analyze_injection.py --input output/eval_results_110_20261019_inject1500.json
    python scripts/analyze_injection.py --compare output/eval_results_110_old.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "tests" / "eval_dataset_expanded.json"
DEFAULT = ROOT / "tests" / "eval_results.json"

SENTENCE_ENDS = "。！？!?…；;"
CITE_RE = re.compile(r"\[(\d+)\]")
GCITE_RE = re.compile(r"\[G(\d+)\]")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def character_map() -> dict[str, str]:
    """question -> character；内置 12 题不在扩充集里，标 builtin"""
    if not DATASET.exists():
        return {}
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("questions", [])
    return {x["question"]: x.get("character", "builtin") for x in items}


def split_legacy(contexts: list[str]) -> tuple[list[str], list[str]]:
    """旧 JSON 没有 kind 标记：按长度切（正文块 ~1000 字，证据碎片 ~30 字）"""
    body = [c for c in contexts if len(c) >= 300]
    ev = [c for c in contexts if len(c) < 300]
    return body, ev


def per_question(res: dict, cmap: dict[str, str]) -> dict:
    """把一条结果摊平成一行统计"""
    tr = res.get("retrieval_trace") or {}
    contexts = res.get("contexts") or []
    answer = res.get("answer") or ""
    bd = tr.get("boundary") or {}

    if bd:
        body = [v for v in bd.values() if v.get("kind") == "source"]
        ev = [v for v in bd.values() if v.get("kind") == "infer"]
        body_chars = sum(v["chars"] for v in body)
        ev_chars = sum(v["chars"] for v in ev)
        body_n, ev_n = len(body), len(ev)
        ends_ok = sum(1 for v in body if v.get("ends_at_sentence"))
        src = "trace"
    else:
        b, e = split_legacy(contexts)
        body_chars, ev_chars = sum(len(x) for x in b), sum(len(x) for x in e)
        body_n, ev_n = len(b), len(e)
        ends_ok = sum(1 for x in b if x.rstrip()[-1:] in SENTENCE_ENDS) if b else 0
        src = "legacy"

    re_ = tr.get("rerank") or {}
    g = tr.get("graph") or {}
    seeds = g.get("seeds") or []

    cites = [int(x) for x in CITE_RE.findall(answer)]
    gcites = [int(x) for x in GCITE_RE.findall(answer)]

    # 越界判定的分母必须跟编号规则一致：
    #   新口径 [n] 只指正文块（另起 [Gn] 指图谱证据）→ 用 body_n
    #   旧口径 [n] 把正文与证据混编成 1..33 → 只能用注入总条数
    cite_limit = body_n if bd else len(contexts)

    return {
        "question": res.get("question", ""),
        "character": cmap.get(res.get("question", ""), "builtin"),
        "body_n": body_n,
        "ev_n": ev_n,
        "body_chars": body_chars,
        "ev_chars": ev_chars,
        "chars": body_chars + ev_chars,
        "ends_ok": ends_ok,
        "overlap": re_.get("topk_overlap"),
        "pool": re_.get("pool_size"),
        "top_k": re_.get("top_k"),
        "graph_on": g.get("triggered"),
        "edges": g.get("edges"),
        "added": g.get("added"),
        "single_char_seeds": sum(1 for s in seeds if len(s) == 1),
        "seed_n": len(seeds),
        "cites": len(cites),
        "cite_distinct": len(set(cites)),
        "cite_over": (max(cites) > cite_limit) if cites and cite_limit else False,
        "gcites": len(gcites),
        "eval": res.get("evaluation") or {},
        "src": src,
    }


def med(rows: list[dict], key: str):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return st.median(vals) if vals else None


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"


def report(title: str, rows: list[dict]) -> None:
    n = len(rows)
    mode = rows[0]["src"] if rows else "-"
    print(f"\n{'=' * 62}")
    print(f"{title}   n={n}   数据来源={mode}")
    print("=" * 62)

    body_n = sum(r["body_n"] for r in rows)
    ev_n = sum(r["ev_n"] for r in rows)
    body_c = sum(r["body_chars"] for r in rows)
    ev_c = sum(r["ev_chars"] for r in rows)

    print("\n[1] 注入组成（中位）")
    print(f"    正文  {med(rows, 'body_n')} 条 / {med(rows, 'body_chars')} 字")
    print(f"    证据  {med(rows, 'ev_n')} 条 / {med(rows, 'ev_chars')} 字")
    print(f"    证据占条目 {pct(ev_n, ev_n + body_n)}   证据占字符 {pct(ev_c, ev_c + body_c)}")
    print(f"    注入字符中位 {med(rows, 'chars')}")

    print("\n[2] 正文块边界")
    ok = sum(r["ends_ok"] for r in rows)
    print(f"    以句末标点结尾 {ok}/{body_n} = {pct(ok, body_n)}")

    if mode == "trace":
        print("\n[3] 精排换人（池 > 注入 才有效）")
        ov = [r["overlap"] for r in rows if r["overlap"] is not None]
        if ov:
            print(f"    top_k 交集 中位 {st.median(ov):.1f}  分布 {sorted(Counter(ov).items())}")
            print(f"    被换掉的平均条数 {st.median([(r['top_k'] or 5) - r['overlap'] for r in rows if r['overlap'] is not None]):.1f}")

        print("\n[4] 图谱触发")
        gt = [r for r in rows if r["graph_on"] is not None]
        on = sum(1 for r in gt if r["graph_on"])
        print(f"    触发 {on}/{len(gt)} = {pct(on, len(gt))}")
        print(f"    edges 中位 {med(gt, 'edges')}   added 中位 {med(gt, 'added')}")
        sc = sum(r["single_char_seeds"] for r in gt)
        sn = sum(r["seed_n"] for r in gt)
        print(f"    单字实体 {sc}/{sn} = {pct(sc, sn)}（词法对齐没设最小长度）")

    print("\n[5] 引用")
    cited = sum(1 for r in rows if r["cites"])
    print(f"    有 [n] 引用 {cited}/{n} = {pct(cited, n)}")
    print(f"    不同编号中位 {med(rows, 'cite_distinct')}")
    print(f"    编号越界 {sum(1 for r in rows if r['cite_over'])}/{n}")
    if mode == "trace":
        g_cited = sum(1 for r in rows if r["gcites"])
        print(f"    有 [Gn] 图谱引用 {g_cited}/{n} = {pct(g_cited, n)}")

    keys = [
        ("faithfulness", "忠实"),
        ("relevancy", "相关"),
        ("context_precision", "精确"),
        ("key_points_coverage", "覆盖"),
        ("voice_consistency", "口吻"),
    ]
    have = [r["eval"] for r in rows if r["eval"]]
    if have:
        print("\n[6] 五维均分")
        line = []
        for k, label in keys:
            vals = [e[k] for e in have if isinstance(e.get(k), (int, float))]
            line.append(f"{label} {sum(vals) / len(vals):.2f}" if vals else f"{label} -")
        print("    " + "   ".join(line))


def by_character(rows: list[dict]) -> None:
    print("\n[7] 分角色（正文条数 / 正文字符中位 / 句末率）")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["character"], []).append(r)
    for ch, g in sorted(groups.items(), key=lambda x: -len(x[1])):
        bn = sum(x["body_n"] for x in g)
        ok = sum(x["ends_ok"] for x in g)
        print(f"    {ch:<14s} n={len(g):3d}  正文 {med(g, 'body_n')} 条  "
              f"{med(g, 'body_chars')} 字  句末 {pct(ok, bn)}")


def compare(old_rows: list[dict], new_rows: list[dict]) -> None:
    print(f"\n{'=' * 62}\n[对比] 旧 n={len(old_rows)} vs 新 n={len(new_rows)}\n{'=' * 62}")
    keys = [
        ("body_n", "正文条数中位"),
        ("body_chars", "正文字符中位"),
        ("ev_n", "证据条数中位"),
        ("ev_chars", "证据字符中位"),
        ("chars", "注入字符中位"),
    ]
    print(f"    {'指标':<16s}{'旧':>10s}{'新':>10s}")
    for k, label in keys:
        a, b = med(old_rows, k), med(new_rows, k)
        print(f"    {label:<16s}{str(a):>10s}{str(b):>10s}")
    a = sum(x["ends_ok"] for x in old_rows) / max(1, sum(x["body_n"] for x in old_rows))
    b = sum(x["ends_ok"] for x in new_rows) / max(1, sum(x["body_n"] for x in new_rows))
    print(f"    {'正文句末率':<14s}{a * 100:>9.1f}%{b * 100:>9.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT)
    ap.add_argument("--compare", type=Path, help="拿这份当旧基线做配对对比")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"找不到 {args.input}（评测还在跑？）")

    cmap = character_map()
    data = load(args.input)
    rows = [per_question(r, cmap) for r in data.get("results", [])]

    report(args.input.name, rows)
    by_character(rows)

    s = data.get("summary") or {}
    if s:
        print(f"\n    文件内 summary: 忠实 {s.get('avg_faithfulness')} / 相关 {s.get('avg_relevancy')} "
              f"/ 精确 {s.get('avg_context_precision')} / 覆盖 {s.get('avg_key_points_coverage')}")

    if args.compare and args.compare.exists():
        old = [per_question(r, cmap) for r in load(args.compare).get("results", [])]
        compare(old, rows)


if __name__ == "__main__":
    main()
