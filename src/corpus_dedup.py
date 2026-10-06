"""语料去重：同一本书只留一份入库载体。

为什么需要这一层
----------------
`collect_corpus_files` 里的 sha256 去重只认**字节完全一致**。但语料目录里
「同一本书的多个载体」有四种形态，sha256 一个都挡不住：

| 形态 | 典型例子 | sha256 能否挡 |
|---|---|---|
| 同名同格式、不同目录 | 根目录 + 「电子书（54本）」子目录各一份 | 尺寸相同才挡得住 |
| 同名跨格式 | 传习录全鉴.pdf（扫描件）+ 传习录全鉴.txt（完整文字） | 完全挡不住 |
| 同一本书的不同作者版本 | 周月亮《王阳明大传》/ 度阴山《知行合一王阳明》/ 冈田武彦《…（全新修订版）》 | 不该挡（三本不同的书） |
| 同一本书的分册 vs 合订本 | 「知行合一的心学智慧」上/中/下（992 页扫描） vs 合订本（983 页文字） | 完全挡不住 |

第 4 种最贵：992 页扫描件要走多模态、约 8 小时，而合订本已有完整文本层。

判据分两层
----------
**第一层：显式规则表**（`KEEP_ONE_PER_BOOK`）。每条写明「保留哪个、排除哪个、依据」，
理由必须落在实测证据上（PDF 元数据 author、页数、文本层覆盖率），不能凭书名印象。
静默排除是事故的温床——所以每条都要能回答「凭什么」。

**第二层：同名跨格式自动判**（`dedupe_cross_format`）。不写死书名，靠特征判：
同名的一组里若同时有 pdf 与 txt，而 pdf 的文本层覆盖率 <50%（即扫描件）、
txt 字符数是 pdf 的 20 倍以上，则只留 txt。理由：扫描件要跑多模态（页均 30s），
而 txt 是完整文字版，留 txt 既省时又不丢内容。

**刻意不做的事**：不按书名相似度跨文件去重。理由是实证教训——
「王阳明最神奇的心学」和「发现心灵的智慧」是同一本网络小说的两个标题
（实测双向覆盖率 25.3% / 18.4%），但归一化后书名完全不同；而
「王阳明大传」（周月亮）和「知行合一王阳明」（度阴山）书名相近却是两本不同的书。
按书名去重会同时犯这两种错，所以只在**有硬证据**（同组跨格式 + 扫描件特征）时才动手。
"""

import re
from pathlib import Path

# 归一化书名：剥掉括号内容、标点、格式尾巴
#
# ★刻意不剥裸的「上/中/下」：`_TAIL` 只认「上册/中册/下册」这类带单位的形式。
# 「知行合一的心学智慧_上」这种分册标记不需要在这里归一——它已经由
# `KEEP_ONE_PER_BOOK` 显式规则挡掉了（依据是 PDF 元数据与页数实测），
# 而按书名猜「上中下是同一本书」恰恰是本模块要避免的那类推断。
# 这里要剥的是纯格式噪声：扩展名残留、扫描版/文字版/修订版等后缀。
_BRACKET = re.compile(r"[（(\[【《].*?[)）\]】》]")
_PUNCT = re.compile(r"[\s_\-—－·。，,、：:；;！!？?\"“”'‘’#]+")
# ★刻意不剥纯数字：`知行合一王阳明(1427-1529)` 里的生卒年是有意义的书名信息，
# 剥掉会让不同年份的同名书撞在一起（第一版把 `\d+` 放进 _TAIL 就是这个错）。
# 格式噪声里的数字一律带单位（「第3版」「上册」），已被 _TAIL 覆盖。
_TAIL = re.compile(
    r"(txt|pdf|text|文字版|电子书|扫描版|高清|全本|完整版|上下册?|上册|中册|下册|"
    r"dk|zb|第\d+版|新版|修订版?|精校版?|无删减|word|doc|epub|mobi)$"
)


def normalize_book_name(stem: str) -> str:
    """把文件名归一成「书名键」，用于把同一本书的不同载体聚到一组。

    入参是 `Path.stem`（已不含扩展名）；这里额外兜一层扩展名残留，
    因为调用方可能直接传文件名。
    """
    s = _PUNCT.sub("", _BRACKET.sub("", stem))
    prev = None
    while prev != s:
        prev = s
        # 每轮都清掉尾点：剥掉「txt」后可能新露出一个「.」（如「书名.txt」）
        s = _TAIL.sub("", s.removesuffix("."))
    return s


# ============================================================
# 显式规则表
# ============================================================
# 每条：(排除文件名子串, 理由)
# 会被 collect_corpus_files 提前挡掉，不进解析流程。
#
# ★这三条的证据都来自 PDF 元数据 / 文本层实测（见 output/_duyin_probe.log、
#   output/_revised_full_check.log），不是按书名猜的。

KEEP_ONE_PER_BOOK: list[tuple[str, str]] = [
    (
        "王阳明为什么倡导人人心中有孔子",
        "不是书：只有 1 页 614x4425pt 的竖版长截图。OCR 实测抽出 4748 字，"
        "内容是**喜马拉雅音频节目的语音转文字稿**（董平讲《王阳明》全集），"
        "不是书页正文。入库后问「王阳明为什么倡导人人心中有孔子」会命中一段播客稿。"
        "★这个坑是踩出来的：`needs_vision` 判它「不是扫描件」→ 掉进常规解析路径 → "
        "走本地 OCR 43.7s 抽出内容 → 切成 5 块入库。"
        "**「needs_vision 判否」不等于「该入库」**——它只回答要不要走多模态",
    ),
    (
        "知行合一的心学智慧  上",
        "冈田武彦《王阳明大传：知行合一的心学智慧》的上册扫描件"
        "（328 页 / 94 MB / creator=Pdg2Pic 纯图片 / 文本层 0）。"
        "同书已有 983 页合订本《王阳明大传：知行合一的心学智慧（全新修订版）.pdf》，"
        "实测文本层覆盖 98.3%、页均 569 字、末尾到附录二完整。"
        "留扫描件等于让 992 页走多模态（约 8 小时）换一份更差的文本。",
    ),
    (
        "知行合一的心学智慧_中",
        "同上，中册扫描件（312 页 / 80 MB / 文本层 0）。",
    ),
    (
        "知行合一的心学智慧_下",
        "同上，下册扫描件（352 页 / 96 MB / 文本层 0）。",
    ),
]

# 刻意**不**排除的文件，写清理由以免后人误加。
#
# 「王阳明大传.pdf」与「知行合一王阳明(1427-1529)-度阴山.pdf」书名相近，
# 但 PDF 元数据实证是两个不同作者的两本书：
#   王阳明大传.pdf          → author=周月亮,   578 页
#   知行合一王阳明-度阴山.pdf → author=度阴山,   982 页
#   知行合一的心学智慧（全新修订版）.pdf → author=冈田武彦, 983 页
# 三本都留：作者视角不同，合订起来才是完整的王阳明传记图景。

KEEP_NOT_EXCLUDED: dict[str, str] = {
    "王阳明大传.pdf": "author=周月亮（PDF 元数据实证），与度阴山/冈田武彦不是同一本书",
    "知行合一王阳明": "author=度阴山（PDF 元数据实证），与周月亮版不是同一本书",
    "王阳明大传：知行合一的心学智慧（全新修订版）":
        "author=冈田武彦（PDF 元数据实证），是分册扫描版的替代载体，必须留",
}


# ============================================================
# 跨格式判据（同名 pdf 与 txt 是同一本书的两个载体）
# ============================================================
# ★踩过的坑：第一版把「pdf 抽出的字少」当成「pdf 是扫描件」，据此只留 txt。
# 实测全量字数后发现判断错了——
#   传习录全鉴-迟双明.pdf  1139 页 → 全量抽出 240563 字（页均 211）
#   传习录全鉴-迟双明.txt            → 244840 字
# 两者字数几乎一样，pdf **不是扫描件**，是「排版版 vs 纯文字版」的同一本书。
# 之前看着像扫描件，是因为只抽了前 30 页、且那几页恰好是版权页/空白页。
#
# 现在的判据改成「同组 pdf 与 txt 都有实质内容 → 留 txt」：
#   - 留 txt 的理由不是「它更全」（字数差不多），而是**它没有排版噪声**：
#     pdf 抽出来的东西混着 ISBN、印张、定价、出版日期、页眉水印、
#     「-c-textilep.com」这类站点字符串（全量比对里两份的差异几乎全在这些）
#   - 判据要求 txt 至少 TEXT_RATIO_MIN 分之一的体量，防止「txt 只是残章」被误留
#
# 为什么不用覆盖率定阈值：实测两份 shingle 双向覆盖率只有 16.4% / 16.8%
# （同一本书、同一内容，只因排版不同就差了这么多）——覆盖率在这个场景下
# 根本不是有效判据，反而会误导。宁可只认「同组且都有内容」这个硬事实。

# pdf 至少有这么多有效字，才算「有实质内容」（低于此视为扫描件/空壳，不参与本判据）
PDF_MIN_CHARS = 100_000
# txt 至少要有 pdf 的这个比例，才敢认定 txt 不是残章
TEXT_RATIO_MIN = 0.5
# 只在 pdf 页数达到这个量级时才做跨格式判据
MIN_PAGES_FOR_CROSSFMT = 100


def _pdf_full_chars(path: Path) -> tuple[int, int]:
    """全本逐页抽文本，返回 (总页数, 有效字符数)。

    必须全量取：抽样会撞上版权页与空白页，把有文本层的书误判成扫描件
    （第一版就是这么错的）。1139 页的逐页 get_text 约 1 秒，可接受。
    """
    try:
        import fitz
    except ImportError:
        return 0, 0
    try:
        chars = 0
        with fitz.open(str(path)) as doc:
            total = len(doc)
            for i in range(total):
                try:
                    chars += len(re.sub(r"\s+", "", doc[i].get_text("text") or ""))
                except Exception:
                    continue
        return total, chars
    except Exception:
        return 0, 0


def _txt_chars(path: Path) -> int:
    """文本文件的有效字符数（沿用多编码兼容顺序）。"""
    for enc in ("utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return len(re.sub(r"\s+", "", path.read_text(encoding=enc)))
        except (UnicodeDecodeError, LookupError):
            continue
    return 0


def dedupe_cross_format(files: list[Path]) -> tuple[list[Path], list[tuple[str, str]]]:
    """同名跨格式去重：同一本书的 pdf（排版版）与 txt（纯文字版）只留 txt。

    返回 (保留的文件, [(被排除的文件名, 理由)])。
    """
    from collections import defaultdict

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        groups[normalize_book_name(f.stem)].append(f)

    drop: dict[Path, str] = {}
    for key, group in groups.items():
        pdfs = [f for f in group if f.suffix.lower() == ".pdf"]
        txts = [f for f in group if f.suffix.lower() in {".txt", ".md"}]
        if len(pdfs) != 1 or len(txts) != 1:
            # 只在「恰好 1 pdf + 1 txt」时判；多份同名交给 sha256 那层
            continue
        pdf, txt = pdfs[0], txts[0]

        pages, pdf_chars = _pdf_full_chars(pdf)
        if pages < MIN_PAGES_FOR_CROSSFMT or pdf_chars < PDF_MIN_CHARS:
            continue  # pdf 本身没实质内容，交给 needs_vision / no_text_extracted 处理

        txt_n = _txt_chars(txt)
        if txt_n < pdf_chars * TEXT_RATIO_MIN:
            continue  # txt 明显残缺，不动

        drop[pdf] = (
            f"与 {txt.name} 同名且同书（pdf 全量 {pdf_chars} 字 / {pages} 页，"
            f"txt {txt_n} 字，体量相当）。两者是同一本书的排版版与纯文字版，"
            f"只留 txt：pdf 抽出的文本混着 ISBN、印张、定价、出版日期与站点水印，"
            f"这些噪声会进向量库"
        )

    kept = [f for f in files if f not in drop]
    dropped = [(p.name, r) for p, r in drop.items()]
    return kept, dropped
