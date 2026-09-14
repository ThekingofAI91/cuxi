"""
knowledge_graph.py — 知识图谱 RAG（GraphRAG 增强层）

在已有「向量 + BM25 + 重排」混合检索之外，用 **实体-关系知识图谱** 做结构化检索增强。
解决纯 chunk 检索的痛点：跨段落的实体关联、因果/从属关系、概念网络，向量检索很难
一次性召回，而知识图谱能沿着关系链把相关实体与其证据「连」出来，再喂给 LLM。

两条主链路：

1) 构建期 build_knowledge_graph(collection, ...)
   - 分批把入库 chunk 送进 LLM，抽取三元组 (实体 --关系--> 实体 + 证据片段)
   - 实体名归一化去重合并，关系按 (head,rel,tail) 聚合去重、累计权重
   - 证据直接存 chunk 文本片段（带 source/heading），落盘 JSON，查询期无需回查 ChromaDB
   - 落盘路径：{graph_dir}/<collection>.json

2) 查询期 retrieve_graph_context(query, collection, ...)
   - 查询实体用**词法匹配**对齐图谱节点（零 LLM 调用，避免每轮多一次 API 往返）
   - 从对齐节点做 k 跳 BFS 子图扩展，收集关联三元组 + 原始证据
   - 渲染成「知识图谱关联推断」文本块 + 一组 evidence Document（挂**真实出处**并标注
     kg_inferred，并入检索结果供引用核查，但不冒充人物逐字原话）

全程零新依赖。大模型仅用于「构建期抽取三元组」。
图谱文件缺失 / 总开关关闭 / 文本检索已达标时，调用方静默降级，不影响原混合检索。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.core.config import settings
from src.retrieval.source_profile import classify_source


# ============================================================
# 路径与持久化
# ============================================================

# 图谱按 mtime 做进程内缓存：构建/失效时 mtime 变化自动失效，避免每轮检索重复读盘 JSON
_graph_cache: dict[str, tuple[float, Optional[dict]]] = {}
_graph_cache_lock = threading.Lock()


def _graph_cache_clear(collection_name: str) -> None:
    """图谱失效 / 重建后清除缓存条目"""
    with _graph_cache_lock:
        _graph_cache.pop(collection_name, None)


def graph_dir() -> Path:
    p = Path(settings.graph_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def graph_path(collection_name: str) -> Path:
    """图谱 JSON 文件路径（按 collection 名区分）"""
    return graph_dir() / f"{collection_name}.json"


def graph_exists(collection_name: str) -> bool:
    """图谱文件存在**且有实体**才算存在。

    空图谱（entities=0，历史上因上游空响应导致的失败产物）不能算"已构建"——
    否则查询期会永远对着一个空图谱做实体对齐，自动构建也永远不会重新触发。
    """
    p = graph_path(collection_name)
    if not p.exists():
        return False
    g = load_graph(collection_name)
    if not g:
        return False
    stats = g.get("stats") or {}
    entities = g.get("entities") or {}
    return int(stats.get("entities") or len(entities) or 0) > 0


def load_graph(collection_name: str) -> Optional[dict]:
    """加载图谱（不存在或损坏返回 None）。按 mtime 做进程内缓存，避免每轮检索重复读盘。"""
    try:
        p = graph_path(collection_name)
        if not p.exists():
            return None
        mtime = p.stat().st_mtime
        with _graph_cache_lock:
            cached = _graph_cache.get(collection_name)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "relations" not in data:
            return None
        with _graph_cache_lock:
            _graph_cache[collection_name] = (mtime, data)
        return data
    except Exception as e:
        print(f"[KnowledgeGraph] ⚠️ 图谱加载失败 {collection_name}: {e}")
        return None


def invalidate_graph(collection_name: str) -> None:
    """删除图谱文件（文档更新 / 重建时调用）"""
    try:
        p = graph_path(collection_name)
        if p.exists():
            p.unlink()
            print(f"[KnowledgeGraph] 🗑️ 图谱已失效: {p}")
        _graph_cache_clear(collection_name)
    except Exception as e:
        print(f"[KnowledgeGraph] ⚠️ 图谱删除失败: {e}")


# ============================================================
# 文本工具（与 BM25 同思路，避免重复 import 时的循环依赖）
# ============================================================

def _tokenize(text: str) -> list[str]:
    """中英文混合分词：中文按字符、英文/数字按空格；用于查询实体与节点名的词重叠匹配"""
    tokens: list[str] = []
    for part in re.split(r"([\u4e00-\u9fff])", text):
        part = part.strip()
        if not part:
            continue
        if re.match(r"^[\u4e00-\u9fff]+$", part):
            tokens.append(part)
        else:
            tokens.extend(t for t in part.split() if t)
    return [t for t in tokens if t]


def _norm(name: str) -> str:
    """实体名归一化键：小写、去空白与标点（保留 CJK 与字母数字）"""
    name = (name or "").strip().lower()
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"[^\w\u4e00-\u9fff]", "", name)
    return name


# ============================================================
# LLM 抽取（Triple Extraction）
# ============================================================

# 抽取输出用「竖线分隔的行式格式」，不用 JSON。
# 原因（2026-08-29 实测定价）：中转 API 对「要求输出 JSON + 长输入」的组合会静默返回空
# 响应（同一段 7.9KB 文本，JSON 指令 3/3 返回空、自然语言指令正常）——导致图谱构建
# 每批 0 实体、白烧几百次调用。行式输出就是普通文本生成，完全绕开该故障模式。
# 格式：头实体 | 关系 | 尾实体 | 证据片段（一行一条，无关系的行不输出）
_EXTRACT_PROMPT = """你是知识图谱抽取器。下面是某位人物/主题的资料文本（可能包含多段）。
请从中抽取「实体」与「实体之间的关系」，**直接输出结果行**，每行一条，格式（用竖线分隔，共 4 列）：

头实体 | 关系 | 尾实体 | 证据片段

示例：
集体潜意识 | 包含 | 原型 | 原型是集体潜意识中先天的心理倾向
王阳明 | 提出 | 知行合一 | 知是行之始，行是知之成

规则：
- 第一行不要任何标题、解释或 markdown 代码块，直接输出结果行
- 只抽取文本中**明确出现**的关联，绝不臆测；没有就什么都不输出
- 头/尾实体用规范实体名（同一概念用原文主要叫法，如统一用「集体潜意识」）
- 关系用简短中文短语（如「提出」「属于」「对立于」「影响了」「出自」「反对」）
- 证据片段摘自原文，控制在 50 字内
- 实体类型不限：人物、理论、概念、著作、事件、地点都可

资料文本：
{text}"""


_FALLBACK_EXTRACT_LLM = None


def _fallback_extract_llm(primary_llm):
    """备用抽取模型（settings.llm_fallback_model；未配置返回 None）。惰性构建、进程内缓存。"""
    global _FALLBACK_EXTRACT_LLM
    if not settings.llm_fallback_model:
        return None
    if _FALLBACK_EXTRACT_LLM is None:
        # 此前这里引用的 get_chat_llm 是其他函数的局部导入，作用域不可见，
        # 备用模型路径一触发就 NameError——备用模型从未真正生效过
        from src.core.llm import get_chat_llm
        _FALLBACK_EXTRACT_LLM = get_chat_llm(
            model=settings.llm_fallback_model,
            temperature=0.2,
            max_tokens=getattr(primary_llm, "max_tokens", None) or 1200,
        )
    return _FALLBACK_EXTRACT_LLM


async def _llm_extract_triples(llm, text: str) -> list[dict]:
    """调用 LLM 抽取三元组（行式竖线格式；解析失败返回 []）"""
    try:
        from src.core.llm import get_chat_llm, ainvoke_nonempty
        resp = await ainvoke_nonempty(
            llm, [("user", _EXTRACT_PROMPT.format(text=text))],
            fallback_llm=_fallback_extract_llm(llm),
        )
        content = resp.content if isinstance(resp, object) and hasattr(resp, "content") else str(resp)
        content = content.strip()
        # 兼容：个别模型仍输出 ``` 包裹则剥掉
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
        out = []
        for line in content.splitlines():
            line = line.strip().strip("-*·").strip()
            if not line or "|" not in line:
                continue
            cols = [c.strip() for c in line.split("|")]
            if len(cols) < 3:
                continue
            h, r, tl = cols[0], cols[1], cols[2]
            ev = cols[3].strip() if len(cols) >= 4 else ""
            if not h or not r or not tl or h == tl:
                continue
            out.append({
                "head": h,
                "relation": r,
                "tail": tl,
                "evidence": ev,
            })
        return out
    except Exception as ex:
        print(f"[KnowledgeGraph] ⚠️ 三元组抽取失败: {ex}")
        return []


def _doc_id_of(text: str) -> str:
    return hashlib.md5(text[:200].encode("utf-8")).hexdigest()


def _find_evidence_source(batch_docs: list, evidence: str) -> tuple[str, str]:
    """从本批次 chunk 反查证据片段的真实来源 / 章节。

    证据是 LLM 从原文摘录的片段，用子串（归一化后）或词重叠匹配回原文 chunk，
    给图谱关系挂上真实出处，避免「知识图谱」被渲染成一本假书名冒充原著。
    """
    if not evidence:
        return "", ""
    ev_norm = _norm(evidence)
    best: tuple[str, str] = ("", "")
    best_overlap = 0
    for d in batch_docs:
        content = d.page_content or ""
        src = d.metadata.get("source", "") or ""
        hd = d.metadata.get("heading", "") or ""
        if content and ev_norm and (ev_norm in _norm(content) or evidence in content):
            return src, hd
        ov = len(set(_tokenize(content)) & set(_tokenize(evidence)))
        if ov > best_overlap:
            best_overlap = ov
            best = (src, hd)
    return best


# ============================================================
# 构建知识图谱
# ============================================================

def _load_collection_docs(collection, limit: int) -> list[Document]:
    """分页拉取 collection 全部文档（与 advanced_search 同思路，避免 SQLite 变量上限）"""
    docs: list[Document] = []
    offset = 0
    page = 500
    while len(docs) < limit:
        raw = collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
        ids = raw.get("ids") or []
        if not ids:
            break
        for text, meta in zip(raw["documents"], raw["metadatas"]):
            if not text or not text.strip():
                continue
            docs.append(Document(page_content=text, metadata=meta or {}))
            if len(docs) >= limit:
                break
        offset += page
        if len(ids) < page:
            break
    return docs


def _write_graph_tmp(collection_name: str, entities: dict, rel_index: dict, processed: int) -> None:
    """把当前累积的图谱写入 .tmp（每批调用：崩溃可续传，且不污染最终文件）"""
    tmp = graph_path(collection_name).with_suffix(".json.tmp")
    snap = {
        "collection": collection_name,
        "built_at": time.time(),
        "resumable": True,
        "processed": processed,
        "entities": {k: {"name": v["name"], "freq": v["freq"]} for k, v in entities.items()},
        "relations": [
            {"h": r["h"], "r": r["r"], "t": r["t"], "e": r["e"], "w": r["w"], "c": r["c"],
             "s": r.get("s", ""), "hd": r.get("hd", "")}
            for r in rel_index.values()
        ],
    }
    tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")


async def build_knowledge_graph(
    collection,
    llm=None,
    batch_size: Optional[int] = None,
    max_chunks: Optional[int] = None,
    resume: bool = True,
    inter_batch_delay: float = 0.0,
) -> dict:
    """
    从 collection 构建知识图谱并落盘。

    Args:
        collection: ChromaDB collection
        llm: ChatOpenAI 实例（未传则内部创建）
        batch_size: 每批 chunk 数（默认 settings.graph_build_batch）
        max_chunks: 最多抽取 chunk 数（默认 settings.graph_build_max_chunks）
        resume: 中断后从 .tmp 续传（默认开）
        inter_batch_delay: 批间延迟秒数（批量建图时用节流避免触发上游
            rpm/tpm 配额 429；0 = 不延迟，交互式单库构建）

    Returns:
        {"collection", "built_at", "stats":{...}, "entities":{...}, "relations":[...]}
    """
    from src.core.llm import get_chat_llm

    if llm is None:
        llm = get_chat_llm(temperature=0.2, max_tokens=1200)

    batch_size = batch_size or settings.graph_build_batch
    max_chunks = max_chunks or settings.graph_build_max_chunks

    print(f"[KnowledgeGraph] 开始构建图谱: collection={collection.name}")
    t0 = time.time()

    docs = _load_collection_docs(collection, max_chunks)
    print(f"[KnowledgeGraph] 拉取 {len(docs)} 条 chunk（上限 {max_chunks}）")

    # 分批抽取。支持断点续传：若上次构建中途崩溃，临时文件里已有已完成批次的
    # 累积结果；resume=True 时直接恢复并跳过已处理 chunk，避免大库（数千 chunk）
    # 重抽时白烧几百次 API。
    entities: dict[str, dict] = {}   # norm_key -> {name, freq, chunk_ids:set}
    rel_index: dict[tuple, dict] = {}  # (h_key,t_key,rel) -> {h,t,rel,evidence,weight,chunk_id}
    processed = 0
    resumed = False

    tmp_path = graph_path(collection.name).with_suffix(".json.tmp")
    if resume and tmp_path.exists():
        try:
            data = json.loads(tmp_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "relations" in data:
                for k, v in data.get("entities", {}).items():
                    entities[k] = {"name": v.get("name", ""), "freq": v.get("freq", 0), "chunk_ids": set()}
                for r in data.get("relations", []):
                    rk = (_norm(r["h"]), _norm(r["r"]), _norm(r["t"]))
                    rel_index[rk] = {
                        "h": r["h"], "t": r["t"], "r": r["r"],
                        "e": r.get("e", ""), "w": r.get("w", 1), "c": r.get("c", ""),
                        "s": r.get("s", ""), "hd": r.get("hd", ""),
                    }
                processed = int(data.get("processed", 0))
                resumed = True
                print(f"[KnowledgeGraph] 续传恢复: {processed}/{len(docs)} chunk 已完成，跳过")
        except Exception as ex:
            print(f"[KnowledgeGraph] ⚠️ 续传临时文件损坏，从头构建: {ex}")
            entities, rel_index, processed = {}, {}, 0

    empty_streak = 0          # 连续 0 三元组的批数（上游空响应/抽取失败的信号）
    _EMPTY_ABORT_STREAK = 5   # 连续 5 批全空 → 判定上游不可用，熔断（省几百次注定失败的调用）
    for i in range(0, len(docs), batch_size):
        if resumed and i < processed:
            continue  # 跳过上次已抽取的批次
        batch = docs[i:i + batch_size]
        batch_text = "\n\n".join(
            f"[{j + 1}]（来源:{d.metadata.get('source','')} | 章节:{d.metadata.get('heading','')}）\n{d.page_content}"
            for j, d in enumerate(batch)
        )
        triples = await _llm_extract_triples(llm, batch_text)
        processed += len(batch)

        # 空批熔断：正常语料每批都会抽出若干三元组，连续全空说明上游在返回
        # 空响应（实测中转会随机返回 200 空壳），继续跑只是白烧 API + 拖垮延迟
        if not triples:
            empty_streak += 1
            if empty_streak >= _EMPTY_ABORT_STREAK and not entities and not rel_index:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(
                    f"连续 {_EMPTY_ABORT_STREAK} 批抽取结果为空（上游疑似持续返回空响应），"
                    f"已中止构建并清理临时文件；请检查 LLM 服务可用性后重试"
                )
        else:
            empty_streak = 0

        for tr in triples:
            hk, tk = _norm(tr["head"]), _norm(tr["tail"])
            if not hk or not tk:
                continue
            for key, name in ((hk, tr["head"]), (tk, tr["tail"])):
                if key not in entities:
                    entities[key] = {"name": name, "freq": 0, "chunk_ids": set()}
                entities[key]["freq"] += 1
            # 关系聚合（同 head/rel/tail 只留一条，权重累计、保留更长的证据）
            ev = tr["evidence"][:settings.graph_evidence_chars]
            rk = (hk, _norm(tr["relation"]), tk)
            rel = rel_index.get(rk)
            if rel is None:
                src, hd = _find_evidence_source(batch, ev)
                rel_index[rk] = {
                    "h": hk, "t": tk, "r": tr["relation"], "e": ev,
                    "w": 1,
                    "c": _doc_id_of(batch_text),
                    "s": src, "hd": hd,
                }
            else:
                rel["w"] += 1
                if len(ev) > len(rel["e"]):
                    rel["e"] = ev
                    # 证据更长时同步刷新出处（出处跟随最优证据片段）
                    rel["s"], rel["hd"] = _find_evidence_source(batch, ev)

        # 每批增量落盘（写 .tmp；崩溃只留半截临时文件，不污染最终图谱）
        _write_graph_tmp(collection.name, entities, rel_index, processed)
        if (i // batch_size) % 10 == 0:
            print(f"[KnowledgeGraph] 已抽取 {processed}/{len(docs)} chunk | 实体 {len(entities)} | 关系 {len(rel_index)}")
        if inter_batch_delay > 0 and i + batch_size < len(docs):
            await asyncio.sleep(inter_batch_delay)

    relations = [
        {"h": r["h"], "r": r["r"], "t": r["t"], "e": r["e"], "w": r["w"], "c": r["c"],
         "s": r.get("s", ""), "hd": r.get("hd", "")}
        for r in rel_index.values()
    ]

    graph = {
        "collection": collection.name,
        "built_at": time.time(),
        "stats": {
            "chunks": processed,
            "entities": len(entities),
            "relations": len(relations),
        },
        "entities": {k: {"name": v["name"], "freq": v["freq"]} for k, v in entities.items()},
        "relations": relations,
    }

    # 原子落盘：直接 rename 已写入的 .tmp 为最终文件（同级目录 rename 是原子操作，
    # 不会出现“写一半被读”的半截文件；崩溃则 .json 不存在，下次可 resume 续传）
    final_path = graph_path(collection.name)
    if tmp_path.exists():
        tmp_path.replace(final_path)
    else:
        final_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    print(f"[KnowledgeGraph] ✅ 图谱构建完成: {len(entities)} 实体 / {len(relations)} 关系 "
          f"| 耗时 {round(time.time() - t0, 1)}s | 落盘 {final_path}")
    return graph


# ============================================================
# 查询期：实体对齐 + 子图扩展 + 渲染
# ============================================================

def _match_query_entities(query: str, graph: dict, n: int = 8) -> list[tuple[str, float]]:
    """
    查询实体对齐（纯词法匹配，零 LLM 调用）：把查询与图谱节点名做子串 / 词重叠匹配，
    返回 [(实体键, 评分), ...] 按评分降序。取代原来的「每轮 LLM 抽取查询实体」，
    避免每次检索都多一次 LLM 往返，也避免 LLM 发散引入噪声。

    评分：节点名整体作为子串命中查询=3.0；查询作为子串命中节点名=2.5；
    否则按词重叠比例（重叠词数 / 节点词数）给分。
    """
    q_tokens = set(_tokenize(query))
    q_norm = _norm(query)
    scored: list[tuple[str, float]] = []
    for key, info in graph.get("entities", {}).items():
        if not isinstance(info, dict):
            continue
        name = info.get("name", "")
        n_name = _norm(name)
        if not n_name:
            continue
        score = 0.0
        if n_name and n_name in q_norm:
            score = 3.0
        elif q_norm and q_norm in n_name:
            score = 2.5
        elif q_tokens:
            overlap = set(_tokenize(name)) & q_tokens
            if overlap:
                score = len(overlap) / max(1, len(set(_tokenize(name))))
        if score > 0:
            scored.append((key, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def _bfs_subgraph(graph: dict, seed_keys: list[str], hops: int, max_edges: int) -> list[dict]:
    """
    从种子节点做 k 跳 BFS，收集关联三元组（去重、按权重降序截断）。
    返回 [{"h","r","t","e","w","s","hd"}, ...]，s/hd 为证据片段的真实出处/章节。
    """
    entities = graph.get("entities", {})
    relations = graph.get("relations", [])
    if not relations or not seed_keys:
        return []

    # 邻接表：node -> [(other_node, relation_dict)]（无向扩展）
    adj: dict[str, list] = {}
    for r in relations:
        adj.setdefault(r["h"], []).append((r["t"], r))
        adj.setdefault(r["t"], []).append((r["h"], r))  # 无向扩展

    visited = set(seed_keys)
    frontier = list(seed_keys)
    collected: dict[tuple, dict] = {}

    for _ in range(max(1, hops)):
        nxt = []
        for node in frontier:
            for other, r in adj.get(node, []):
                # 关系键用 (min,max,rel) 去重，避免 A->B 与 B->A 重复
                rk = (min(node, other), max(node, other), _norm(r["r"]))
                if rk not in collected:
                    node_info = entities.get(node) or {}
                    other_info = entities.get(other) or {}
                    collected[rk] = {
                        "h": node_info.get("name", node),
                        "t": other_info.get("name", other),
                        "r": r["r"],
                        "e": r.get("e", ""),
                        "w": r.get("w", 1),
                        "s": r.get("s", ""),    # 真实出处（构建期反查原文 chunk）
                        "hd": r.get("hd", ""),  # 真实章节
                    }
                else:
                    collected[rk]["w"] += r.get("w", 1)
                if other not in visited:
                    visited.add(other)
                    nxt.append(other)
        frontier = nxt
        if not frontier:
            break

    edges = sorted(collected.values(), key=lambda x: x["w"], reverse=True)[:max_edges]
    return edges


def _render_graph_context(edges: list[dict]) -> str:
    """把子图三元组渲染成 LLM 可读的结构化补充文本（带证据、限长）"""
    if not edges:
        return ""
    lines = [f"## 知识图谱关联推断（共 {len(edges)} 条关系）",
             "以下关系由知识图谱从资料中归纳得出，属结构化推断、非人物逐字原话，"
             "仅供组织回答时串联概念，引用请以原始资料为准：", ""]
    used = 0
    for i, e in enumerate(edges, 1):
        ev = (e.get("e") or "").strip()
        ev_part = f"（证据：{ev}）" if ev else ""
        line = f"{i}. {e['h']} —{e['r']}→ {e['t']}{ev_part}"
        if used + len(line) > settings.graph_context_chars and i > 1:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


async def retrieve_graph_context(
    query: str,
    collection,
    llm=None,
    top_entities: Optional[int] = None,
    hops: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> tuple[str, list[Document]]:
    """
    查询期知识图谱检索：对齐实体 → 子图扩展 → 渲染上下文 + 证据文档。

    Args:
        query: 用户问题（建议带历史感知，与 advanced_retrieval 一致）
        collection: ChromaDB collection（用于按名定位图谱文件）
        llm: ChatOpenAI（未传内部创建）
        top_entities / hops / max_edges: 覆盖默认配置

    Returns:
        (graph_context_text, graph_evidence_docs)
        - graph_context_text: 渲染好的结构化文本（空串表示无图谱/无命中）
        - graph_evidence_docs: 证据片段 Document 列表（并入 retrieved_docs 供引用核查）
    """
    top_entities = top_entities or settings.graph_top_entities
    hops = hops or settings.graph_hops
    max_edges = max_edges or settings.graph_max_edges

    graph = load_graph(collection.name)
    if not graph or not graph.get("relations"):
        return "", []

    # 1) 查询实体对齐（词法匹配，零 LLM 调用）
    matched = {k: s for k, s in _match_query_entities(query, graph, n=top_entities)}
    if not matched:
        return "", []

    entities = graph["entities"]
    # 种子：对齐得分高 + 出现频率高的节点
    def _entity_freq(k: str) -> int:
        info = entities.get(k) or {}
        return info.get("freq", 0)

    seed_keys = sorted(matched.keys(), key=lambda k: (matched[k], _entity_freq(k)), reverse=True)[:top_entities]
    print(f"[KnowledgeGraph] 命中种子实体 {len(seed_keys)} 个: "
          f"{[(entities.get(k) or {}).get('name', k) for k in seed_keys]}")

    # 2) 子图扩展
    edges = _bfs_subgraph(graph, seed_keys, hops, max_edges)
    if not edges:
        return "", []

    # 3) 渲染 + 生成证据文档（证据挂真实出处，并标注「知识图谱推断」不冒充原著）
    context_text = _render_graph_context(edges)
    ev_docs: list[Document] = []
    for e in edges:
        ev = (e.get("e") or "").strip()
        if not ev:
            continue
        src = e.get("s") or ""
        hd = e.get("hd") or ""
        ev_docs.append(Document(
            page_content=ev,
            metadata={
                "source": src or "知识图谱（推断）",
                "heading": hd or f"{e['h']} {e['r']} {e['t']}",
                "source_type": classify_source(src) if src else "unknown",
                "kg_evidence": True,
                "kg_inferred": True,
                "kg_relation": f"{e['h']} —{e['r']}→ {e['t']}",
            },
        ))

    print(f"[KnowledgeGraph] 子图返回 {len(edges)} 条关系 / {len(ev_docs)} 条证据")
    return context_text, ev_docs


# ============================================================
# 按需触发判断（文本检索质量不达标才启动图谱）
# ============================================================

def should_trigger_graph_retrieval(
    query: str,
    retrieved_docs: list,
    collection_name: str,
) -> tuple[bool, str]:
    """
    决定是否「按需」调用知识图谱：默认只用文本检索；仅当文本检索质量不达标时才启动图谱增强。

    返回 (是否触发, 原因)。触发条件（任一成立即触发）：
    1) 文本召回数量不足（< graph_trigger_min_docs）→ 文本没找到足够材料；
    2) 文本 top 相关性偏弱（rrf_score < graph_trigger_score）→ 最相关片段也很弱；
    3) 查询命中图谱实体，但该实体未被文本结果覆盖 → 文本漏掉了关键概念关联。

    图谱总开关关闭或未构建时直接不触发（静默降级）。
    """
    if not settings.graph_rag_enabled:
        return False, "图谱总开关关闭"
    if not graph_exists(collection_name):
        return False, "图谱未构建"

    # 信号 1：召回数量不足
    if len(retrieved_docs) < settings.graph_trigger_min_docs:
        return True, f"文本召回不足（{len(retrieved_docs)} 条 < 阈值 {settings.graph_trigger_min_docs}）"

    # 信号 2：top 相关性偏弱
    top_score = max(
        ((d.metadata or {}).get("rrf_score", 0.0) for d in retrieved_docs),
        default=0.0,
    )
    if top_score < settings.graph_trigger_score:
        return True, f"文本相关性偏弱（top rrf={top_score:.3f} < 阈值 {settings.graph_trigger_score}）"

    # 信号 3：实体覆盖缺口（图谱里有该实体，但文本结果未提及）
    graph = load_graph(collection_name)
    if graph:
        matched = _match_query_entities(query, graph, n=settings.graph_top_entities)
        if matched:
            text_blob = _norm(" ".join(d.page_content or "" for d in retrieved_docs))
            uncovered = [
                graph["entities"][k]["name"]
                for k, _score in matched
                if _norm(graph["entities"][k]["name"]) not in text_blob
            ]
            if uncovered:
                return True, f"图谱实体未被文本覆盖：{uncovered[:3]}"

    return False, "文本检索质量达标"
