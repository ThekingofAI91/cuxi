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
        print(f"[KnowledgeGraph] 图谱加载失败 {collection_name}: {e}")
        return None


def invalidate_graph(collection_name: str) -> None:
    """删除图谱文件（文档更新 / 重建时调用）"""
    try:
        p = graph_path(collection_name)
        if p.exists():
            p.unlink()
            print(f"[KnowledgeGraph] 图谱已失效: {p}")
        _graph_cache_clear(collection_name)
    except Exception as e:
        print(f"[KnowledgeGraph] 图谱删除失败: {e}")


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

# 实体类型枚举（2026-10-21 加类型列时定）：只有「事件」会改变检索行为——事件节点不参与
# 二跳扩展（见 `_bfs_subgraph` 第 6 条），其余类型只是标注、不进逻辑。所以这里宁可窄，
# 也不放一堆用不上的类型。
_ENTITY_TYPES = ("人物", "概念", "著作", "事件", "地点", "组织")
_ENTITY_TYPES = set(_ENTITY_TYPES)


# 抽取输出用「竖线分隔的行式格式」，不用 JSON。
# 原因（2026-08-29 实测定价）：中转 API 对「要求输出 JSON + 长输入」的组合会静默返回空
# 响应（同一段 7.9KB 文本，JSON 指令 3/3 返回空、自然语言指令正常）——导致图谱构建
# 每批 0 实体、白烧几百次调用。行式输出就是普通文本生成，完全绕开该故障模式。
# 格式：头实体 | 头类型 | 关系 | 尾实体 | 尾类型 | 证据片段（一行一条，无关系的行不输出）
_EXTRACT_PROMPT = """你是知识图谱抽取器。下面是某位人物/主题的资料文本（可能包含多段）。
请从中抽取「实体」与「实体之间的关系」，**直接输出结果行**，每行一条，格式（用竖线分隔，共 6 列）：

头实体 | 头类型 | 关系 | 尾实体 | 尾类型 | 证据片段

示例：
集体潜意识 | 概念 | 包含 | 原型 | 概念 | 原型是集体潜意识中先天的心理倾向
王阳明 | 人物 | 提出 | 知行合一 | 概念 | 知是行之始，行是知之成
王阳明 | 人物 | 经历 | 龙场悟道 | 事件 | 因触怒刘瑾谪至贵州龙场，夜半大悟

规则：
- 第一行不要任何标题、解释或 markdown 代码块，直接输出结果行
- 只抽取文本中**明确出现**的关联，绝不臆测；没有就什么都不输出
- 类型只能填这六个之一：人物、概念、著作、事件、地点、组织
- 「事件」= 一次具体发生的事（龙场悟道、宁王之乱、某某讲学、某场辩论）；
  「概念」= 可以反复讨论的理论/术语（心即理、知行合一、集体潜意识）；
  拿不准是事件还是概念时填「概念」
- 头/尾实体用规范实体名（同一概念用原文主要叫法，如统一用「集体潜意识」）
- 关系用简短中文短语（如「提出」「属于」「对立于」「影响了」「出自」「反对」）
- 证据片段摘自原文，控制在 50 字内
- 证据片段放在最后一列，里面可以出现竖线，前面五列不行

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
            reasoning_effort="none",
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
            # maxsplit=5：证据片段的最后一列允许自带竖线
            cols = [c.strip() for c in line.split("|", 5)]
            if len(cols) < 3:
                continue
            # 新格式 6 列：头 | 头类型 | 关系 | 尾 | 尾类型 | 证据
            # 模型偶尔会漏掉类型列，退回旧格式 4 列：头 | 关系 | 尾 | 证据。
            # 判别看「第 2 列是不是合法类型名」而不是只看列数——旧格式的竖线数也可能
            # 因为证据里带竖线凑够 5 列。
            if (len(cols) >= 5 and cols[1] in _ENTITY_TYPES) or len(cols) >= 6:
                h, h_type, r, tl, t_type = cols[0], cols[1], cols[2], cols[3], cols[4]
                ev = cols[5] if len(cols) > 5 else ""
                h_type = h_type if h_type in _ENTITY_TYPES else ""
                t_type = t_type if t_type in _ENTITY_TYPES else ""
            else:
                h, r, tl = cols[0], cols[1], cols[2]
                ev = cols[3] if len(cols) > 3 else ""
                h_type = t_type = ""
            if not h or not r or not tl or h == tl:
                continue
            out.append({
                "head": h,
                "head_type": h_type,
                "relation": r,
                "tail": tl,
                "tail_type": t_type,
                "evidence": ev.strip(),
            })
        return out
    except Exception as ex:
        print(f"[KnowledgeGraph] 三元组抽取失败: {ex}")
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


def _majority_type(type_counts: dict) -> str:
    """实体类型取出现最多的那个（同一实体在不同批里可能被标成不同类，取多数）。"""
    if not type_counts:
        return ""
    return max(type_counts.items(), key=lambda kv: kv[1])[0]


def _write_graph_tmp(collection_name: str, entities: dict, rel_index: dict, processed: int) -> None:
    """把当前累积的图谱写入 .tmp（每批调用：崩溃可续传，且不污染最终文件）"""
    tmp = graph_path(collection_name).with_suffix(".json.tmp")
    snap = {
        "collection": collection_name,
        "built_at": time.time(),
        "resumable": True,
        "processed": processed,
        "entities": {
            k: {"name": v["name"], "freq": v["freq"], "type": _majority_type(v.get("types") or {})}
            for k, v in entities.items()
        },
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
        # 结构化抽取必须关思考（同 legend.py:615 / 判官）：推理模型的 reasoning_content
        # 会把 max_tokens=1200 的输出预算吃掉 → 大量空响应/截断，实测空响应重试连发
        llm = get_chat_llm(temperature=0.2, max_tokens=1200, reasoning_effort="none")

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
                    # .tmp 里只存了多数派类型（不存计数），续传时按 1 计即可
                    tp = v.get("type") or ""
                    entities[k] = {
                        "name": v.get("name", ""), "freq": v.get("freq", 0),
                        "chunk_ids": set(), "types": {tp: 1} if tp else {},
                    }
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
            print(f"[KnowledgeGraph] 续传临时文件损坏，从头构建: {ex}")
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
            for key, name, tp in (
                (hk, tr["head"], tr.get("head_type", "")),
                (tk, tr["tail"], tr.get("tail_type", "")),
            ):
                if key not in entities:
                    entities[key] = {"name": name, "freq": 0, "chunk_ids": set(), "types": {}}
                entities[key]["freq"] += 1
                if tp:
                    tc = entities[key].setdefault("types", {})
                    tc[tp] = tc.get(tp, 0) + 1
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
        "entities": {
            k: {"name": v["name"], "freq": v["freq"], "type": _majority_type(v.get("types") or {})}
            for k, v in entities.items()
        },
        "relations": relations,
    }

    # 原子落盘：直接 rename 已写入的 .tmp 为最终文件（同级目录 rename 是原子操作，
    # 不会出现“写一半被读”的半截文件；崩溃则 .json 不存在，下次可 resume 续传）
    final_path = graph_path(collection.name)
    if tmp_path.exists():
        tmp_path.replace(final_path)
    else:
        final_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    print(f"[KnowledgeGraph] 图谱构建完成: {len(entities)} 实体 / {len(relations)} 关系 "
          f"| 耗时 {round(time.time() - t0, 1)}s | 落盘 {final_path}")
    return graph


# ============================================================
# 查询期：实体对齐 + 子图扩展 + 渲染
# ============================================================

def _match_query_entities(query: str, graph: dict, n: int = 8) -> list[tuple[str, float]]:
    """
    查询实体对齐（纯词法匹配，零 LLM 调用）：节点名与问题做子串匹配，返回
    [(实体键, 评分), ...] 按评分降序。取代原来的「每轮 LLM 抽取查询实体」，
    避免每次检索都多一次 LLM 往返，也避免 LLM 发散引入噪声。

    只有两档：
    - 3.0：节点名整体出现在问题里（`n_name in q_norm`）
    - 2.5：方向相反，问题整体是节点名的子串（`q_norm in n_name`）。实测这档几乎不产生
      3.0 之外的新实体——要让一整句问题塞进节点名里很难。

    ★第三档「按单字重叠比例给分」已于 2026-10-21 删除。原因是 `_tokenize` 把中文按单字
    切，问题里只要出现「王」字，「宁王」0.50、「王华」0.50、「王通」0.50、「王阳明讲学」
    0.60 全都能拿分，王阳明六道题的种子因此逐字相同。这一档在生产里本来就是死代码
    （`_literal_seeds` 只取 3.0），删它是把语义收窄到「你说的词真的出现了」。

    单字节点名（「三」「心」「梦」「水」）直接不参与对齐：子串匹配下它们在任何含该字的
    查询里都拿 3.0，比真正的长实体还靠前，挤进种子后 2 跳就能撞满 graph_max_edges。
    只挡种子、不删实体——单字实体经别的种子 BFS 走到时仍会带出关系（见 config 注释）。
    """
    q_norm = _norm(query)
    min_len = max(1, int(getattr(settings, "graph_min_seed_len", 1)))
    scored: list[tuple[str, float]] = []
    for key, info in graph.get("entities", {}).items():
        if not isinstance(info, dict):
            continue
        name = info.get("name", "")
        n_name = _norm(name)
        if not n_name or len(n_name) < min_len:
            continue
        if n_name in q_norm:
            scored.append((key, 3.0))
        elif q_norm and q_norm in n_name:
            scored.append((key, 2.5))
    # 同分按节点频次降序：问题里同时出现多个实体时，反复出现的那个信息量更大；
    # 顺带让 `_literal_seeds` 不必「先多取候选、再按频次重排」。
    ents = graph.get("entities", {})
    scored.sort(
        key=lambda kv: (kv[1], (ents.get(kv[0]) or {}).get("freq", 0)),
        reverse=True,
    )
    return scored[:n]


def _literal_seeds(query: str, graph: dict, top_entities: int) -> dict[str, float]:
    """
    取种子实体：只收「节点名整体出现在问题里」的实体（score == 3.0），不补位。

    补位种子已于 2026-10-21 删除，依据是 n=110 的实测：

    - `_match_query_entities` 另外两档低分（查询是节点名的子串 2.5、按词重叠 0.3-0.75）
      描述的是「模糊相关」，拿来凑满名额就是污染。王阳明 24 题里 6 个名额被
      「王阳明 + 王阳明讲学/王阳明弟子 + 宁王/王华/王通」占满，问题真正的主题
      （梦、婚礼、十五岁）一个都进不来 —— 十道题里六道因此拿到逐字相同的边。
    - 补位种子对好图零贡献：荣格阴影题、阿德勒自卑题的 S0（top6 含补位）与只取字面的
      前 8 条证据一字不差，它们本来就排在 30 条之外。
    - 代价是零产出：图里没有「梦」「婚礼」节点时，图谱通道给不出东西。实测王阳明 24 题
      里 12 题零产出、峰哥 2/13、张雪峰 3/13，荣格与阿德勒 0 零产出。这是接受的代价
      —— 宁可不给，也不拿「宁王/王华」冒充问题的实体。

    候选池就是 top_entities 本身：`_match_query_entities` 只返回两档（3.0 字面命中 / 2.5
    反向子串），同分已按节点频次降序，截断后不会漏掉信息量更大的那个。此前「先多取
    8 个候选再按频次重排」的余量是为第三档（单字重叠 0.3-0.75）准备的，那一档删除后
    余量失去意义，一并去掉。
    """
    pool = _match_query_entities(query, graph, n=top_entities)
    return {k: s for k, s in pool if s >= 3.0}


def _resolve_gate_nodes(graph: dict, collection_name: str) -> set[str]:
    """
    找出「闸门节点」的键：本体级枢纽 + 事件节点。

    **本体枢纽**：名字来自 `settings.graph_hub_entities`（collection 名 → 本体名列表），
    不再从度数比例猜。原因：图谱里一个本体常被抽成多个节点（峰哥 = 「峰哥亡命天涯」
    61.1% + 「峰哥」21.2%），比例判据只有其中一个能靠阈值稳定命中，另一个靠余量碰巧够；
    而「谁是主角」本来就是已知信息，没理由去拟合。名字对不上时该库没有枢纽。

    **事件节点**：靠图谱里的 `type == "事件"` 认（抽取期打的标，见 `_EXTRACT_PROMPT`）。
    事件往外连的都是同场的其他人物，放它进 BFS 等于把整场的人都拖进来；而它到本体的
    那条边本身是有用的。所以事件节点与本体同一待遇：只保留「事件 → 问题命中实体」的边，
    不向外扩张。旧图谱没有 type 字段 → 这里一个事件都认不出来 → 行为与加类型前一致。

    返回的是**节点键**（`_norm` 后的键），不是显示名——关系数组里 h/t 存的都是键。
    """
    hubs: set[str] = set()
    names = (getattr(settings, "graph_hub_entities", None) or {}).get(collection_name) or []
    wanted = {_norm(n) for n in names if n}
    if wanted:
        hubs = {
            key for key, info in graph.get("entities", {}).items()
            if isinstance(info, dict) and _norm(info.get("name", "")) in wanted
        }
    events = {
        key for key, info in graph.get("entities", {}).items()
        if isinstance(info, dict) and info.get("type") == "事件"
    }
    return hubs | events


def _bfs_subgraph(
    graph: dict,
    seed_keys: list[str],
    hops: int,
    max_edges: int,
    min_support: Optional[int] = None,
    allowed_endpoints: Optional[set] = None,
    gate_nodes: Optional[set] = None,
) -> list[dict]:
    """
    从种子节点做 k 跳 BFS，收集关联三元组（去重、按优先级截断）。
    返回 [{"h","r","t","e","w","s","hd"}, ...]，s/hd 为证据片段的真实出处/章节。

    选边规则：
    1. 一跳边（直接挂着种子）优先于二跳边。
    2. 二跳边要求「支撑度」达标：支撑度 = 该边另一端在种子一跳邻域里有几个邻居（含跳板
       自己）。支撑度 1 意味着它只是从跳板伸出去的一条孤边——实测全库 79.7% 的二跳边都是
       这种孤边，会顺着一条偶然的邻接把整串无关关系拖进来。典型：题「为什么说价值必须建立
       在社会意义之上」，种子度数只有 1-5、一跳填不满，于是经 `约翰 —仇恨→ 社会` 这条
       w=1 的边，把约翰（度数 48）的 47 条罪犯案例整批拉进来，30 格里占了 20 格。
    3. 两端都在一跳邻域里的边算高支撑（2），它们两侧都贴着问题实体。
    4. 同优先级内按 w 降序。注意全库 w=1 占 94.7-99.5%，w 的实际区分度很弱，排序主要靠
       上面两条结构规则。
    5. **闸门**（2026-10-21）：`gate_nodes` 里的节点（本体枢纽 + 事件节点）参与的边，另一端
       必须在 `allowed_endpoints` 里，否则这条边既不收、也不继续跳。原因是本体一跳边远超
       上限：王阳明 101 条边、张雪峰 363 条，而 w 全是 1、排序退化成关系数组顺序，于是十道
       题里六道拿到逐字相同的 30 条。闸门一上，本体只留「本体 → 问题实体」这几条。
       闸门节点由 `_resolve_gate_nodes` 定位（本体按配置名字、事件按图谱里的 type 字段）。
       不传 `gate_nodes` 时按图谱里的 collection 名现查，所以离线探针只给白名单也一致。
    6. **事件不当跳板**（2026-10-21）：事件节点只贡献「事件 → 问题命中实体」的边。它到
       同场其他人物的边被闸门挡掉，那些人进不了种子邻域，自然也就没有二跳。事件名命中
       问题时它会成为种子，种子本身同样不做二跳出发点（`n1 -= seeds`）。
    """
    entities = graph.get("entities", {})
    relations = graph.get("relations", [])
    if not relations or not seed_keys:
        return []
    if min_support is None:
        min_support = max(2, int(getattr(settings, "graph_min_edge_support", 2)))

    # 邻接表：node -> [(other_node, relation_dict)]（无向扩展）
    adj: dict[str, list] = {}
    for r in relations:
        adj.setdefault(r["h"], []).append((r["t"], r))
        adj.setdefault(r["t"], []).append((r["h"], r))  # 无向扩展

    # 闸门：只认「另一端是问题命中实体」的那几条边。
    # 闸门节点缺省按图谱自带的 collection 名现查（离线探针只传白名单也能拿到正确结果），
    # 没有白名单就没有「允许的另一端」，闸门整体关闭（保持旧行为）。
    if gate_nodes is None and allowed_endpoints is not None:
        gate_nodes = _resolve_gate_nodes(graph, graph.get("collection", "") or "")
    gates: set[str] = set(gate_nodes or ()) if allowed_endpoints is not None else set()

    def _blocked(node: str, other: str) -> bool:
        if not gates:
            return False
        if node in gates and other not in allowed_endpoints:
            return True
        if other in gates and node not in allowed_endpoints:
            return True
        return False

    seeds = set(seed_keys)
    collected: dict[tuple, dict] = {}

    def _record(node: str, other: str, r: dict, hop: int, support: int) -> None:
        rk = (min(node, other), max(node, other), _norm(r["r"]))
        if rk in collected:
            collected[rk]["w"] += r.get("w", 1)
            return
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
            "hop": hop,
            "support": support,
        }

    # 一跳：种子的全部邻居
    n1: set[str] = set()
    for node in seed_keys:
        for other, r in adj.get(node, []):
            if _blocked(node, other):
                continue
            _record(node, other, r, 1, 1)
            n1.add(other)
    n1 -= seeds

    # 二跳：从一跳邻域继续走，只保留有支撑的边
    if hops >= 2:
        for node in n1:
            for other, r in adj.get(node, []):
                if other in seeds:
                    continue  # 一跳已收
                if _blocked(node, other):
                    continue
                if other in n1:
                    _record(node, other, r, 2, 2)  # 两端都贴着种子邻域
                    continue
                support = len({x for x, _ in adj.get(other, [])} & n1)
                if support < min_support:
                    continue
                _record(node, other, r, 2, support)

    edges = sorted(
        collected.values(),
        key=lambda x: (x["hop"] == 1, x["support"], x["w"]),
        reverse=True,
    )[:max_edges]
    for e in edges:
        e.pop("hop", None)
        e.pop("support", None)
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
    trace_out: Optional[dict] = None,
) -> tuple[str, list[Document]]:
    """
    查询期知识图谱检索：对齐实体 → 子图扩展 → 渲染上下文 + 证据文档。

    Args:
        query: 用户问题（建议带历史感知，与 advanced_retrieval 一致）
        collection: ChromaDB collection（用于按名定位图谱文件）
        llm: ChatOpenAI（未传内部创建）
        top_entities / hops / max_edges: 覆盖默认配置
        trace_out: 可选轨迹容器（评测专用），填 available/matched/seeds/edges/evidence

    Returns:
        (graph_context_text, graph_evidence_docs)
        - graph_context_text: 渲染好的结构化文本（空串表示无图谱/无命中）
        - graph_evidence_docs: 证据片段 Document 列表（并入 retrieved_docs 供引用核查）
    """
    top_entities = top_entities or settings.graph_top_entities
    hops = hops or settings.graph_hops
    max_edges = max_edges or settings.graph_max_edges

    graph = load_graph(collection.name)
    if trace_out is not None:
        trace_out["available"] = bool(graph and graph.get("relations"))
    if not graph or not graph.get("relations"):
        return "", []

    # 1) 查询实体对齐（词法匹配，零 LLM 调用）
    #    种子只收「字面命中」：节点名整体出现在问题里（score == 3.0）。补位种子已删除
    #    （2026-10-21，见 `_literal_seeds` 的说明），没有字面命中就零产出、不凑数。
    entities = graph["entities"]
    matched = _literal_seeds(query, graph, top_entities)
    if trace_out is not None:
        trace_out["matched"] = len(matched)
    if not matched:
        print("[KnowledgeGraph] 问题里没有字面命中的实体，本次不产出图谱证据")
        return "", []

    seed_keys = list(matched.keys())
    print(f"[KnowledgeGraph] 命中种子实体 {len(seed_keys)} 个: "
          f"{[(entities.get(k) or {}).get('name', k) for k in seed_keys]}")
    if trace_out is not None:
        trace_out["seeds"] = [(entities.get(k) or {}).get("name", k) for k in seed_keys]

    # 2) 子图扩展：种子集合同时作为「本体/事件跳不过去的白名单」
    #    种子 = 问题里字面命中的实体（含事件名）。事件名进白名单，所以「本体—事件」这条边
    #    收得下；但事件往别处扩张的边会被闸门挡掉，见 `_bfs_subgraph` 第 5、6 条规则。
    gate_nodes = _resolve_gate_nodes(graph, collection.name)
    if trace_out is not None:
        trace_out["gates"] = len(gate_nodes)
    edges = _bfs_subgraph(
        graph,
        seed_keys,
        hops,
        max_edges,
        allowed_endpoints=set(seed_keys),
        gate_nodes=gate_nodes,
    )
    if trace_out is not None:
        trace_out["edges"] = len(edges)
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
    if trace_out is not None:
        trace_out["evidence"] = len(ev_docs)
    return context_text, ev_docs


# ============================================================
# 触发判断（概念覆盖缺口才启动图谱）
# ============================================================

def should_trigger_graph_retrieval(
    query: str,
    retrieved_docs: list,
    collection_name: str,
) -> tuple[bool, str]:
    """
    决定是否调用知识图谱，只做「概念覆盖缺口」这一件事的守门。

    判据只有一条：查询命中的图谱实体，没有被文本检索结果覆盖。

    历史上还有两条阈值信号，都已删除（2026-10-19）：
      · 「top rrf_score < 0.03」—— RRF 分数的大小主要由来源权重决定（同一篇文档
        单类得分 oral 约 0.0246、anchor 约 0.0328），固定阈值切的是来源类型而不是
        相关性，实测只能独立多触发 1/98 题，还把触发原因记成「相关性偏弱」；
      · 「文本召回条数 < 3」—— 注入固定 5 条，正常路径永远走不到，是死兜底。
    删完实测触发 85/98（86.7%），100% 来自实体覆盖缺口。

    图谱总开关关闭或未构建时直接不触发（静默降级）。
    """
    if not settings.graph_rag_enabled:
        return False, "图谱总开关关闭"
    if not graph_exists(collection_name):
        return False, "图谱未构建"

    # 实体覆盖缺口：图谱里有该实体，但文本结果未提及
    graph = load_graph(collection_name)
    if graph:
        # 与取种子同一口径（`_literal_seeds`）：只有字面命中的实体才可能被用来扩子图，
        # 拿模糊命中当触发条件会白跑一趟图谱、最后零产出。
        matched = _literal_seeds(query, graph, settings.graph_top_entities)
        if matched:
            text_blob = _norm(" ".join(d.page_content or "" for d in retrieved_docs))
            uncovered = [
                graph["entities"][k]["name"]
                for k in matched
                if _norm(graph["entities"][k]["name"]) not in text_blob
            ]
            if uncovered:
                return True, f"图谱实体未被文本覆盖：{uncovered[:3]}"

    return False, "查询命中的图谱实体已被文本覆盖"
