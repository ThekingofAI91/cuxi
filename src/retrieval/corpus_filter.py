"""corpus_filter.py — 入库前的「非正文块」过滤。

## 为什么需要单独一层

书名页、版权页、目录这三类文本有两个坏性质：

1. **它们把书名和核心概念重复十几遍**，向量相似度和 BM25 都会判它们最相关，
   是典型的「标题堆砌型 hard negative」。实测《原型与集体无意识》的版权页
   在一道问「原型和集体潜意识是什么关系」的题里排到了 `[1]`，而它通篇是
   译者、出版社、ISBN、定价。
2. **对生成零价值**，被注入后还会诱导模型把它当「资料」引用，把一条真材料挤掉。

## 判据刻意保守

宁可漏掉，不能误删正文。三种命中方式，都要求**强特征**：

- 版权页：出现 `ISBN` / `图书在版编目` / `CIP 数据` / `版权所有` / `出版发行`
- 目录：点线 + 页码 的行 ≥ 3 行（"第一章 潜意识……5"）
- 出版信息碎片：极短（< 100 字）且含 `定价 / 开本 / 印张 / 责任编辑` 等

目录页经常和正文被分进同一块（章首页的导读 + 目录），所以目录只**删行**不删块，
删完剩下不足 200 字才整块丢弃——正文里提到"出版社"不会被误伤。

## 量级

全库 5711 块里只有 8 块命中（jung 5 / wangyangming 3），但其中一块正好是
那道题的 top1。所以这层过滤的价值不在占比，在它命中的是**高排位噪声**。
"""

from __future__ import annotations

import re

# 强标记：出现即判版权页
STRONG_MARKS = re.compile(
    r"ISBN|图书在版编目|CIP\s*数据|版权所有|出版发行"
)
# 次要标记：只在「极短块」里作为判据
PUB_MARKS = re.compile(
    r"定价|开本|印张|印次|责任编辑|封面设计|装帧设计|书号|出版时间|出版社"
)
# 版权页专属的「职务栏」用词：一本书的正文里不会同时出现两个以上。
# 为什么要这条：不少版权页/封面页**没有 ISBN**（ISBN 在另一页），光靠强标记
# 抓不到。《原型与集体无意识》的封面+版权+内容简介混编块就是这样漏过去的——
# 它有译者、统筹监制、特约编辑、市场推广、珍藏限量版，却没有 ISBN。
BOILER_MARKS = re.compile(
    r"统筹监制|特约编辑|责任编辑|封面设计|装帧设计|市场推广|限量版|珍藏版"
    r"|丛书主编|出品人|策划编辑|出版公司|出版发行"
)
# 目录行：点线/省略号/中点/连字符 + 行末页码（必须 re.M，否则只有最后一行能匹配）
TOC_LINE = re.compile(r"(\.{4,}|…{2,}|·{4,}|-{6,})[\s\d]*\d{1,4}\s*$", re.M)
# 目录里的「纯标题行」：没有页码但形如「第一讲 xxx」「三、xxx」「目 录」
TOC_HEAD = re.compile(
    r"^\s*(目\s*录|第[〇一二三四五六七八九十百零两\d]{1,4}[讲章节篇部卷回]"
    r"|[一二三四五六七八九十]{1,3}、)\s*\S"
)
_SENT_END_ANY = re.compile(r"[。！？!?；;]")

MIN_USEFUL_CHARS = 200   # 清完剩下的正文少于这个数就整块丢弃
MIN_ANY_CHARS = 20       # 任何不足 20 字的块都是碎片
MIN_BODY_DENSITY = 1.0   # 每 100 字的句末标点数，低于此值判为「没有正文」


def front_matter_kind(text: str) -> str:
    """返回 ""（非前置页）/ "版权页" / "目录" / "出版信息碎片"。"""
    t = (text or "").strip()
    if not t:
        return "碎片"
    if len(t) < MIN_ANY_CHARS:
        return "碎片"
    if STRONG_MARKS.search(t):
        return "版权页"
    # 职务栏用词命中两个以上 → 封面页/版权页。正文里不会同时出现两个。
    if len(set(BOILER_MARKS.findall(t))) >= 2:
        return "版权页"
    if len(TOC_LINE.findall(t)) >= 3:
        return "目录"
    if len(t) < 100 and PUB_MARKS.search(t):
        return "出版信息碎片"
    return ""


def is_front_matter(text: str) -> bool:
    """整块是否属于「入库该丢弃」的非正文块。"""
    return bool(front_matter_kind(text))


def _is_toc_line(line: str) -> bool:
    """目录行判定。只在「整块已被判为目录」时才调用，所以这里可以放宽。

    两类：带页码的点线行；以及不带页码但形如「第一讲 xxx」「三、xxx」的纯标题行。
    纯标题行要求短且没有句末标点——正文句子一定带标点，靠这条把误删挡住。
    """
    s = (line or "").strip()
    if not s:
        return False
    if TOC_LINE.search(s):
        return True
    if len(s) <= 40 and not _SENT_END_ANY.search(s) and TOC_HEAD.match(s):
        return True
    return False


def body_density(text: str) -> float:
    """每 100 字的句末标点数。正文约 2~5，版权页/名单/目录约 0~0.5。

    这是「整块丢」还是「只删几行」的唯一判据，比按 kind 硬判靠谱：
    同一个 kind 里既有纯版权页，也有「正文 + 版权尾」的混编块——
    实测《阴影与自我》有一块 1193 字，前面是荣格生平传记、末尾粘了版权页，
    按 kind 整块丢会连传记一起丢掉。
    """
    if not text:
        return 0.0
    return len(_SENT_END_ANY.findall(text)) / len(text) * 100


def _is_boilerplate_line(line: str) -> bool:
    """版权/出版名单行：出现强标记或职务栏用词即视为该删的行。"""
    s = (line or "").strip()
    if not s:
        return False
    return bool(STRONG_MARKS.search(s) or BOILER_MARKS.search(s))


def strip_front_matter_lines(text: str) -> str:
    """删掉目录行与版权/职务栏行，保留正文。"""
    kept = [
        ln for ln in (text or "").splitlines()
        if not (_is_toc_line(ln) or _is_boilerplate_line(ln))
    ]
    return "\n".join(kept).strip()


def clean_chunk(text: str) -> tuple[str, str]:
    """清洗单块。返回 (清洗后文本, 判定类型)：
    - 文本非空、类型为空 → 原样保留
    - 文本为空、类型非空 → 丢弃
    - 文本非空但已删过行、类型非空 → 用清洗后的文本替换（需要重新 embed）
    """
    kind = front_matter_kind(text)
    if not kind:
        return text, ""
    cleaned = strip_front_matter_lines(text)
    # 删完还剩足够正文（且句读密度像正文）→ 保留清洗版；否则整块丢
    if len(cleaned) < MIN_USEFUL_CHARS or body_density(cleaned) < MIN_BODY_DENSITY:
        return "", kind
    return cleaned, kind


def clean_chunks(chunks: list) -> tuple[list, dict]:
    """批量清洗 Document 列表。

    返回 (保留下来的块, {原因: 丢弃块数})。清洗后文本有变化的块会同步
    更新 `char_length`；向量化发生在这一步之后，所以 embedding 与文本一致。
    """
    kept: list = []
    dropped: dict[str, int] = {}
    for c in chunks:
        text, reason = clean_chunk(c.page_content or "")
        if not text:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        if text != c.page_content:
            c.page_content = text
            if "char_length" in (c.metadata or {}):
                c.metadata["char_length"] = len(text)
        kept.append(c)
    return kept, dropped
