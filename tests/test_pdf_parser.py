"""
PDF 解析 / 分块的回归测试（2026-09-22）

背景：PDFParser 原先逐行 append 元素，并用「短行且无句末标点即标题」的
启发式判标题，实测一本 94 页的书 84% 的行被判成标题，而 AdaptiveChunker
见 heading 就断组，最终碎成 3063 块（块长中位 47 字）。

修复后：
- 标题改由**字号**判定（≥ 正文字号 × 1.12，且该字号行数占比 ≤ 6%，行长 ≤ 45）
- 段落按 block 聚合，不再逐行产出
- min_size 真正参与判断（末尾不足则向上合并）

这些用例全部离线、不依赖真实 PDF，防止上述行为被改回去。
"""

import types

import pytest

from src.document_processing.parser import (
    PDFParser,
    ParsedElement,
    _DECORATED_PAGE_NUM_RE,
    _join_pdf_lines,
    _norm_font_size,
    _pick_heading_sizes,
    _is_ocr_noise_line,
)
from src.retrieval.chunker import AdaptiveChunker


# ---------------------------------------------------------------
# 字号归一化
# ---------------------------------------------------------------

def test_norm_font_size_trims_float_tail():
    """PDF 字号常带浮点尾差，必须归一化，否则同一字号会被当成两个"""
    assert _norm_font_size(30.000000953674316) == _norm_font_size(30.0)
    assert _norm_font_size(15.832467) == 15.8
    assert _norm_font_size(None) == 0.0
    assert _norm_font_size("bad") == 0.0


# ---------------------------------------------------------------
# 标题字号挑选：核心判据
# ---------------------------------------------------------------

def test_pick_heading_sizes_body_is_dominant():
    """正文字号 = 覆盖字符数最多的字号"""
    size_chars = {14.4: 99994, 18.0: 17}
    size_lines = {14.4: 4186, 18.0: 1}
    body, heads = _pick_heading_sizes(size_chars, size_lines)
    assert body == 14.4
    assert heads == {18.0}


def test_pick_heading_sizes_excludes_high_frequency_large_font():
    """用大一号字排引文的书：引文字号行数占比高，不能被当成标题字号

    实测《王阳明心学口诀》19.5 号字占 34.7% 的行（真引文），
    只有 30.0 号那 7 行才是标题。
    """
    size_chars = {15.0: 50000, 19.5: 20000, 30.0: 300}
    size_lines = {15.0: 3000, 19.5: 1600, 30.0: 7}
    body, heads = _pick_heading_sizes(size_chars, size_lines)
    assert body == 15.0
    assert 19.5 not in heads, "高频大字号（引文）被误判成标题字号"
    assert heads == {30.0}


def test_pick_heading_sizes_empty():
    assert _pick_heading_sizes({}, {}) == (0.0, set())


# ---------------------------------------------------------------
# 行拼接
# ---------------------------------------------------------------

def test_join_pdf_lines_cjk_has_no_space():
    """中文之间直接相接（PDF 换行是排版换行，不是语义换行）"""
    assert _join_pdf_lines(["被讨厌的", "勇气"]) == "被讨厌的勇气"


def test_join_pdf_lines_ascii_gets_space():
    assert _join_pdf_lines(["hello", "world"]) == "hello world"


def test_join_pdf_lines_skips_blank():
    assert _join_pdf_lines(["甲", "  ", "乙"]) == "甲乙"


# ---------------------------------------------------------------
# 噪声判据
# ---------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "~ 2 ~",
    "— 12 —",
    "· 7 ·",
    "～ 103 ～",
    "12",
    "2023.4",
    "Playinart 制作",
    "图书在版编目（CIP）数据",
    "   ",
])
def test_noise_lines_are_filtered(line):
    assert _is_ocr_noise_line(line) is True


@pytest.mark.parametrize("line", [
    "被讨厌的勇气",
    "第 2 章　一切烦恼都来自人际关系",
    "2023 年第 2 期正式发行",
    "他说：“人的烦恼皆源于人际关系。”",
    "由团队精心制作",
])
def test_real_content_is_kept(line):
    """正文不能被杀（尤其是含数字/含「制作」的正常句子）"""
    assert _is_ocr_noise_line(line) is False


def test_decorated_page_num_regex_shape():
    assert _DECORATED_PAGE_NUM_RE.fullmatch("~2~")
    assert not _DECORATED_PAGE_NUM_RE.fullmatch("2")
    assert not _DECORATED_PAGE_NUM_RE.fullmatch("第2章")


# ---------------------------------------------------------------
# 一页 → 元素：标题按字号，其余聚成段落
# ---------------------------------------------------------------

def test_page_to_elements_splits_on_heading_size():
    parser = PDFParser()
    lines = [
        (14.4, "正文第一行"),
        (18.0, "第一章"),
        (14.4, "正文第二行"),
    ]
    els = parser._page_to_elements(lines, {18.0}, "t.pdf", 1)

    heads = [e for e in els if e.element_type == "heading"]
    paras = [e for e in els if e.element_type == "paragraph"]

    assert [h.content for h in heads] == ["第一章"]
    assert heads[0].metadata["heading_level"] == 1
    # 标题把正文切成了两段，但段内不再逐行产出
    assert [p.content for p in paras] == ["正文第一行", "正文第二行"]


def test_page_to_elements_long_heading_line_stays_body():
    """字号命中但过长（>45 字）的行不算标题，避免把正文误切"""
    parser = PDFParser()
    long_line = "很长的一行" * 12  # 60 字
    els = parser._page_to_elements([(18.0, long_line)], {18.0}, "t.pdf", 1)
    assert all(e.element_type == "paragraph" for e in els)


def test_page_to_elements_merges_consecutive_body_lines():
    """连续正文行聚成一个段落，而不是每行一个元素（旧 bug 的根因）"""
    parser = PDFParser()
    lines = [(14.4, "第一句。"), (14.4, "第二句。"), (14.4, "第三句。")]
    els = parser._page_to_elements(lines, set(), "t.pdf", 1)
    assert len(els) == 1
    assert els[0].content == "第一句。第二句。第三句。"


# ---------------------------------------------------------------
# 边距判据：页眉页脚要以「页面另有正文」为前提
# ---------------------------------------------------------------

def _fake_line(text: str, size: float, y0: float):
    return {"type": 0, "lines": [
        {"spans": [{"text": text, "size": size}], "bbox": [0, y0, 200, y0 + 10]}
    ]}


class _FakePage:
    """最小 page 替身：只要 get_text('dict') 与 rect.height"""

    def __init__(self, blocks, height=800.0):
        self._blocks = blocks
        self.rect = types.SimpleNamespace(height=height)

    def get_text(self, mode):
        return {"blocks": self._blocks}


def test_margin_line_dropped_when_body_exists():
    """正文存在时，顶部边距内的短行按页眉丢弃"""
    page = _FakePage([
        _fake_line("王阳明大传", 12.0, 10.0),      # 顶部 1.25% → 页眉
        _fake_line("这是一段正经的正文内容，应当保留下来。", 15.0, 400.0),
    ])
    texts = [t for _, t in PDFParser()._extract_page_lines(page)]
    assert texts == ["这是一段正经的正文内容，应当保留下来。"]


def test_margin_line_kept_when_it_is_the_only_text():
    """整页只有一行且落在边距带内 → 不是页眉，是图页配文，必须保留

    实测《王阳明心学口诀》p21/p36/p69 就是这样：整页仅顶部 y0/h=0.02
    一行真内容，此前被边距判据静默丢弃。
    """
    page = _FakePage([_fake_line("同向前，成为圣贤。", 12.0, 8.0)])
    texts = [t for _, t in PDFParser()._extract_page_lines(page)]
    assert texts == ["同向前，成为圣贤。"]


def test_noise_line_dropped_regardless_of_position():
    """噪声（裸页码）即使独占一页也不能留下"""
    page = _FakePage([_fake_line("50", 12.0, 5.0)])
    assert PDFParser()._extract_page_lines(page) == []


# ---------------------------------------------------------------
# min_size：末尾不足则向上合并
# ---------------------------------------------------------------

def _para(text: str) -> ParsedElement:
    return ParsedElement(content=text, element_type="paragraph",
                         metadata={"source": "t.md"})


def test_chunker_merges_undersized_tail_upward():
    """尾部短段应并进本组上一个块，而不是单独产出碎片块"""
    ck = AdaptiveChunker(min_size=500, max_size=1000, overlap=0)
    docs = ck.chunk([_para("甲" * 900), _para("乙" * 60)], source="t.md")
    assert len(docs) == 1
    assert len(docs[0].page_content) == 962


def test_chunker_keeps_single_short_block():
    """组内首块就很小（无上文可并）→ 仍应单独成块，不能丢内容"""
    ck = AdaptiveChunker(min_size=500, max_size=1000, overlap=0)
    docs = ck.chunk([_para("丙" * 60)], source="t.md")
    assert len(docs) == 1
    assert docs[0].page_content == "丙" * 60


def test_chunker_merges_even_when_over_max_size():
    """min_size 与 max_size 冲突时 min_size 优先：章尾碎片必须并上去

    实测一本 94 页的书曾因「上一块已顶到 max_size 就不再合并」留下
    18 个 <200 字的章尾碎片。合并后上限由 min_size 兜住
    （最坏 max_size + min_size - 1）。
    """
    ck = AdaptiveChunker(min_size=500, max_size=1000, overlap=0)
    docs = ck.chunk([_para("丁" * 960), _para("戊" * 80)], source="t.md")
    assert len(docs) == 1
    assert len(docs[0].page_content) == 1042


def test_chunker_merges_tail_of_long_paragraph():
    """长段落被切分/暴力截断后甩出的尾巴也要并回去

    这是碎片块的第二个来源：_split_long_paragraph 有自己的收尾逻辑，
    与 _chunk_paragraphs 的 min_size 判据不是同一处。
    """
    ck = AdaptiveChunker(min_size=500, max_size=1000, overlap=0)
    docs = ck.chunk([_para("甲" * 1100)], source="t.md")
    assert len(docs) == 1, "长段落切分后的尾巴未被合并"
    assert len(docs[0].page_content) >= 1100


# ---------------------------------------------------------------
# OCR 开关：无文本层的 PDF 不该静默返回空
# ---------------------------------------------------------------

def test_allow_ocr_false_returns_placeholder_for_scanned_pdf(tmp_path):
    """无文本层的 PDF 且关掉 OCR 时，应返回带 warning 的占位元素。

    批量入库要用这个 warning 把「需要 OCR 的文件」单独列出来，
    而不是把它当正常块入库（那会污染检索），也不是静默返回空。
    """
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()                      # 空白页：没有文本层
    pdf_path = tmp_path / "scanned.pdf"
    doc.save(str(pdf_path))
    doc.close()

    els = PDFParser().parse(pdf_path, allow_ocr=False)

    assert len(els) == 1
    assert els[0].metadata.get("warning") == "no_text_extracted"


# ---------------------------------------------------------------
# 纯标题组：不立刻发块，由后续正文消费标题链
# ---------------------------------------------------------------

def _heading(text: str, level: int = 1) -> ParsedElement:
    return ParsedElement(content=text, element_type="heading",
                         metadata={"source": "t.md", "heading_level": level})


def test_heading_only_group_does_not_emit_junk_chunk():
    """章首页把「第一章」和正文拆开时，不该为前者单独产出一个 3 字块

    实测一本 94 页的书有 13 个这样的章号页，占全库 13%。标题信息
    应当作为标题链附着到后续正文块上，而不是独立成块。
    """
    ck = AdaptiveChunker(min_size=500, max_size=1000, overlap=0)
    docs = ck.chunk([_heading("第一章"), _para("甲" * 700)], source="t.md")

    assert len(docs) == 1, "纯标题组被单独发了块"
    assert docs[0].metadata["heading"] == "第一章"
    assert docs[0].page_content == "甲" * 700


def test_all_heading_document_keeps_one_consolidated_chunk():
    """通篇只有标题（如纯目录文件）→ 仍要留下一个块，标题不能凭空消失"""
    ck = AdaptiveChunker(min_size=500, max_size=1000, overlap=0)
    docs = ck.chunk([_heading("目 录"), _heading("第二章", level=2)], source="t.md")
    assert len(docs) == 1
    assert "第二章" in docs[0].page_content
