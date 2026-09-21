"""
Verification Agent — 引用核查员
负责原著引用核查：检查角色回答中的核心论断是否在检索到的原著资料中有依据，
标注出处与引用可信度，防止回答编造原著中不存在的观点（防幻觉）。

定位：名人对话场景专用 —— 回答应"基于原著"，核查让引用有据可查。
"""

from __future__ import annotations

import re
from typing import Any

from src.core.llm import get_chat_llm

from src.core.config import settings
from src.core.state import AgentState
from src.core.utils import strip_doc_ext


async def verification_agent(state: AgentState) -> dict[str, Any]:
    """
    引用核查 Agent：对角色回答进行原著引用核查

    流程：
    1. 从 state 读取 analysis（角色回答）与 retrieved_docs（原著检索结果）
    2. 提取核心论断
    3. 逐条在原著资料中查找依据
    4. 标注引用可信度与出处
    5. 将核查报告写入 state["verification"]
    """
    analysis = state.get("analysis", "")
    retrieved_docs = state.get("retrieved_docs", [])
    query = state.get("query", "")
    print(f"\n[Verification Agent] 正在核查原著引用...")

    # 收集需要核查的论断
    claims = _extract_claims(analysis)
    if not claims:
        print("[Verification Agent] 没有发现需要核查的论断")
        return {
            "verification": "未发现需要核查的内容。",
            "route_history": state.get("route_history", []) + ["verification_agent"],
        }

    if not retrieved_docs:
        # 无检索资料可对照（如纯闲聊场景），跳过核查，不产出报告
        print("[Verification Agent] 无检索资料可对照，跳过引用核查")
        return {
            "verification": "",
            "route_history": state.get("route_history", []) + ["verification_agent"],
        }

    print(f"[Verification Agent] 提取到 {len(claims)} 条核心论断")

    try:
        verification_parts: list[str] = []
        citation_parts: list[str] = []
        supported_count = 0
        unsupported_count = 0
        partial_count = 0

        for i, claim in enumerate(claims, 1):
            # 在原著检索结果中查找支持证据
            evidence, match_score, inferred = _find_evidence(claim, retrieved_docs)
            verdict, confidence = _evaluate_claim(claim, evidence, match_score)
            print(f"[Verification Agent] 论断{i} 得分={match_score:.2f} 判定={verdict}")
            verification_parts.append(
                _format_verification_item(i, claim, verdict, confidence, evidence)
            )

            if verdict == "原著有据":
                supported_count += 1
            elif verdict == "部分依据":
                partial_count += 1
            else:
                unsupported_count += 1

            # 有依据的论断收集进"引用出处"
            if evidence:
                citation_parts.append(_format_citation_item(i, claim, evidence, inferred))

        # 复杂论断（数量较多时）用 LLM 辅助核查
        llm_verification = ""
        if len(claims) > 3:
            llm_verification = await _llm_verification(query, analysis, retrieved_docs)

        # 计算整体引用可信度
        total = len(claims)
        overall_confidence = (supported_count + 0.5 * partial_count) / max(total, 1)
        overall_confidence = min(overall_confidence, 1.0)

        # 组装核查报告
        report_lines = ["## 原著引用核查", ""]
        report_lines.append("### 核查摘要")
        report_lines.append(f"- 核心论断数: {total}")
        report_lines.append(f"- 原著有据: {supported_count}")
        report_lines.append(f"- 部分依据: {partial_count}")
        report_lines.append(f"- 原著无据: {unsupported_count}")
        report_lines.append(f"- 引用可信度: {overall_confidence:.0%}")

        if citation_parts:
            report_lines.append("")
            report_lines.append("### 引用出处")
            report_lines.extend(citation_parts)

        if llm_verification:
            report_lines.append("")
            report_lines.append(f"### LLM 辅助评估")
            report_lines.append(llm_verification.strip())

        # 可信度阈值检查
        if overall_confidence < settings.verifier_confidence_threshold:
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("### 引用可信度提示")
            report_lines.append(
                f"整体引用可信度 ({overall_confidence:.0%}) 低于阈值 "
                f"({settings.verifier_confidence_threshold:.0%})，回答中部分观点在检索到的原著资料中缺乏直接依据，请注意甄别。"
            )
            print(f"[Verification Agent] 引用可信度 {overall_confidence:.0%} 低于阈值")
        else:
            print(f"[Verification Agent] 引用可信度 {overall_confidence:.0%} 通过阈值检查")

        return {
            "verification": "\n".join(report_lines),
            "route_history": state.get("route_history", []) + ["verification_agent"],
        }

    except Exception as e:
        print(f"[Verification Agent] 核查过程出错: {e}")
        return {
            "verification": "",
            "route_history": state.get("route_history", []) + ["verification_agent"],
            "error": str(e),
        }


def _extract_claims(analysis: str) -> list[str]:
    """
    从回答中提取需要核查的核心论断

    提取策略：
    - 排除动作描写（*动作* 标记）等非论断内容
    - 包含数字/判断性/因果性的陈述
    - 引用性陈述
    """
    claims: list[str] = []
    if not analysis or not analysis.strip():
        return claims

    # 按句子分割
    sentences = re.split(r"(?<=[。！？.!?\n])\s*", analysis)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        if _is_claimable(sentence):
            claims.append(sentence)

    # 去重并限制数量（角色回答论断不宜过多，8 条足够覆盖核心观点）
    seen = set()
    unique_claims = []
    for claim in claims:
        key = claim[:50]
        if key not in seen:
            seen.add(key)
            unique_claims.append(claim)

    return unique_claims[:8]


def _is_claimable(sentence: str) -> bool:
    """判断一个句子是否包含需要核查的论断"""
    # 排除问句和命令句
    if sentence.endswith("？") or sentence.endswith("?"):
        return False

    # 排除动作描写（*动作* 标记整行）与明显的元信息
    skip_patterns = [
        r"^\*.*\*$",  # 动作描写/强调
        r"^【[^】]+】",  # 【动作】标记
        r"^（.*）$",  # （动作）
        r"^---",  # 分隔线
        r"^#",  # 标题
        r"^```",  # 代码块
        r"^\d+\.\s",  # 编号列表
    ]
    for pattern in skip_patterns:
        if re.match(pattern, sentence):
            return False

    # 需要核查的句型
    claim_patterns = [
        r"[是属定义表示]",
        r"\d+[%百分比倍率]",
        r"根据|依据|参考|来源于|著作|书中",
        r"可以|能够|需要|必须|应该",
        r"主要|核心|关键|重要|本质",
        r"原因|导致|因为|由于|因此",
        r"区别|差异|不同|优势|劣势",
        r"第一|第二|首先|其次|最后",
        r"不是|并非|而是",
    ]
    return any(re.search(p, sentence) for p in claim_patterns)


_STOP_CHARS = set(
    "的了在是我有和就都不人也都一一个上也很到说要你去会着"
    "没有看好自己这那吗呢吧他她它我们你们他们因为所以但是"
    "又还或者并且等从为把被对向于让将给以与及或而乃之其么"
    "个中里来去出"
)


def _is_kept_char(ch: str) -> bool:
    """判断字符是否属于保留的有效信息字符（非停用字的中文/字母/数字）"""
    if ch in _STOP_CHARS:
        return False
    if "\u4e00" <= ch <= "\u9fff":
        return True
    return ch.isascii() and (ch.isdigit() or ch.isalpha())


def _char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> set[str]:
    """
    提取文本中的有信息量字符 n-gram（2~4 字滑动窗口）

    中文无空格分词，整句会被正则当成一个 token，导致关键词匹配几乎永远失败。
    改为字符级 n-gram：先剔除停用字，再对剩余连续片段生成滑动窗口，
    与文档原文做集合重合，可容忍同义改写与语序微调。
    """
    fragments: list[str] = []
    current: list[str] = []
    for ch in text:
        if _is_kept_char(ch):
            current.append(ch)
        else:
            if current:
                fragments.append("".join(current))
                current = []
    if current:
        fragments.append("".join(current))

    grams: set[str] = set()
    for frag in fragments:
        if len(frag) < 2:
            continue
        if len(frag) <= max_n:
            grams.add(frag)
        else:
            for n in range(min_n, max_n + 1):
                for i in range(len(frag) - n + 1):
                    grams.add(frag[i : i + n])
    return grams


def _find_evidence(claim: str, docs: list) -> tuple[str, float, bool]:
    """
    在原著检索结果中查找支持证据

    通过字符 n-gram 重合度判断，返回最佳匹配来源及其匹配分。
    长句论断先按标点拆分子句，取子句最高重合度，避免整句 n-gram 被修饰成分稀释。
    """
    if not docs:
        return "", 0.0, False

    # 按标点拆分子句（2-4 字 n-gram 至少需要若干有效字符才有意义）
    sub_sentences = [s.strip() for s in re.split(r"[，。；、！？：…,;!?:\n]", claim)]
    claim_gram_sets = [
        _char_ngrams(s) for s in sub_sentences if len(s) >= 4
    ]
    claim_gram_sets = [g for g in claim_gram_sets if len(g) >= 2]
    if not claim_gram_sets:
        return "", 0.0, False

    best_evidence = ""
    best_score = 0.0
    best_inferred = False

    for doc in docs:
        content = doc.page_content
        content_grams = _char_ngrams(content)
        if not content_grams:
            continue

        # 取所有子句中的最高重合率（论断中任一段有据即算有据）
        score = max(
            len(g & content_grams) / len(g) for g in claim_gram_sets
        )

        if score > best_score:
            best_score = score
            source = doc.metadata.get("source", "未知来源")
            heading = doc.metadata.get("heading", "")
            # 知识图谱证据：关系是图谱归纳的推断，非人物逐字原话
            best_inferred = bool(doc.metadata.get("kg_inferred", False))
            preview = content[:200] + "..." if len(content) > 200 else content

            best_evidence = f"> 来源: {source}"
            if heading:
                best_evidence += f" | 章节: {heading}"
            if best_inferred:
                best_evidence += "\n> 知识图谱推断关系，非人物逐字原话"
            best_evidence += f"\n> {preview}"

    return best_evidence, best_score, best_inferred


def _evaluate_claim(claim: str, evidence: str, match_score: float = 0.0) -> tuple[str, float]:
    """
    评估论断在原著中是否有依据

    依据文档原文关键词覆盖率判定（match_score），避免被修饰词过度稀释。

    Returns:
        (verdict, confidence)
        - "原著有据" / "部分依据" / "原著无据"
    """
    if not evidence:
        return "原著无据", 0.0

    if match_score >= 0.5:
        return "原著有据", float(match_score)
    elif match_score >= 0.25:
        return "部分依据", float(match_score)
    else:
        return "原著无据", float(match_score)


def _format_verification_item(
    index: int,
    claim: str,
    verdict: str,
    confidence: float,
    evidence: str,
) -> str:
    """格式化单条核查结果"""
    item = f"\n#### {index}. {verdict} (置信度: {confidence:.0%})\n"
    item += f"**论断:** {claim}\n"
    if evidence:
        item += f"**依据:**\n{evidence}\n"
    else:
        item += "**依据:** 未在检索资料中找到直接出处\n"
    return item


def _format_citation_item(index: int, claim: str, evidence: str, inferred: bool = False) -> str:
    """格式化"引用出处"条目（紧凑，适合拼到最终回答）"""
    # 从证据中提取来源与章节
    source_match = re.search(r"来源: (.+?)(?:\s*\||\n|$)", evidence)
    heading_match = re.search(r"章节: (.+?)(?:\n|$)", evidence)
    source = source_match.group(1).strip() if source_match else "原著"
    # 界面上只显示著作名，不暴露 .pdf/.epub 这类文件后缀
    source = strip_doc_ext(source)
    heading = heading_match.group(1).strip() if heading_match else ""

    claim_short = claim[:40] + ("…" if len(claim) > 40 else "")
    location = f"{source}｜{heading}" if heading else source
    # 知识图谱证据是推断关系，标注出来，避免被当成人物原话出处
    if inferred:
        location += "（知识图谱推断）"
    return f"{index}. 《{location}》 ← 「{claim_short}」"


async def _llm_verification(query: str, analysis: str, docs: list) -> str:
    """
    使用 LLM 辅助核查：评估回答中的观点是否都能在原著资料中找到依据
    """
    try:
        llm = get_chat_llm(
            temperature=0.1,
            max_tokens=600,
        )

        # 构建文档摘要
        doc_summary = "\n".join([
            f"- [{doc.metadata.get('source', '?')}] {doc.page_content[:200]}"
            for doc in docs[:5]
        ])

        prompt = f"""你是原著引用核查员。用户正在与一位基于著作资料的历史人物对话，回答必须忠实于原著思想。

## 用户问题
{query}

## 角色回答
{analysis[:1500]}

## 检索到的原著资料摘要
{doc_summary}

请评估：
1. 回答中的核心观点是否在原著资料中有依据？
2. 是否有观点在资料中完全找不到出处（疑似编造）？
3. 简要指出哪些观点缺少原著支持（如有）

请给出简短评估，200 字以内。"""

        response = llm.invoke([
            ("system", "你是一个严格的引用核查员，只基于给出的原著资料判断，不引入外部知识。"),
            ("user", prompt),
        ])

        return response.content
    except Exception as e:
        return f"LLM 辅助核查不可用: {e}"
