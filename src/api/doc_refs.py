"""引用编号与出处清单：把注入的资料整理成前端角标能对上的映射。

两个函数是一对，共同维护「正文角标 ↔ 资料」的映射契约：

- `build_doc_map(docs)`：给每块注入资料编号。正文块 `[1]..[N]`，图谱推断块 `[G1]..[Gn]`。
  编号顺序必须与 `framework.analysis_agent._build_context` 完全一致——那边按
  `retrieved_docs` 的原始顺序依次编号，这边同序生成，前端才能把正文里的 `[3]`
  对上第三条正文块、`[G2]` 对上第二条推断。

- `citations_from_doc_map(doc_map)`：把同一份清单渲染成「引用出处」文本，经 SSE
  `type:'citations'` 事件推给前端折叠区。

两套编号必须分开（2026-10-19 的教训）：旧实现把正文块与图谱推断混编成 1..33，
模型看到 `[31] 来源: 荣格自传` 就当原著引用，把三元组归纳出来的关系说成原话。
"""

from __future__ import annotations

from src.core.utils import strip_doc_ext


def build_doc_map(docs: list) -> list[dict]:
    """把注入的资料转成前端角标映射，两套编号分开：正文 1..N，图谱推断 G1..Gn。

    kind 字段供前端区分样式：source = 原文，infer = 推断（不是原文字句）。
    来源去掉 .pdf/.epub 后缀，界面上只显示著作名。
    """
    doc_map: list[dict] = []
    plain_n = infer_n = 0
    for d in docs or []:
        meta = d.metadata or {}
        if meta.get("kg_inferred"):
            infer_n += 1
            doc_map.append({
                "n": f"G{infer_n}",
                "kind": "infer",
                "source": strip_doc_ext(meta.get("source", "")),
                "heading": meta.get("heading", ""),
            })
        else:
            plain_n += 1
            doc_map.append({
                "n": str(plain_n),
                "kind": "source",
                "source": strip_doc_ext(meta.get("source", "")),
                "heading": meta.get("heading", ""),
            })
    return doc_map


def citations_from_doc_map(doc_map: list[dict]) -> str:
    """把注入清单渲染成「引用出处」文本，编号与正文角标一一对应。

    前端把它塞进 `<details>` 且 `white-space: pre-wrap` 原样展示（不解析 markdown），
    所以只能是纯文本、一行一条——不能出现 `**`、`---`、表格这类需要解析的记号。

    与已删除的 verifier 报告的区别：那边一行是「论断序号. 《出处》 ← 「论断前 40 字」」，
    序号是**论断**的序号，跟正文 [n] 无关，读者容易误以为 [3] 对应第 3 条。
    这里直接用注入块的编号，[3] 就是第三条资料，映射可逆。
    """
    lines: list[str] = []
    for d in doc_map or []:
        n = str(d.get("n") or "").strip()
        if not n:
            continue
        source = (d.get("source") or "").strip() or "未知来源"
        heading = (d.get("heading") or "").strip()
        location = f"{source}｜{heading}" if heading else source
        suffix = "（知识图谱推断）" if d.get("kind") == "infer" else ""
        lines.append(f"{n}. 《{location}》{suffix}")
    return "\n".join(lines)
