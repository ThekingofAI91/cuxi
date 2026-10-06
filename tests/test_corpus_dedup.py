"""语料去重的单元测试（全部打桩，零网络零大文件）。

覆盖：
  · normalize_book_name：归一化把「上/中/下」「修订版」这类尾巴剥掉
  · KEEP_ONE_PER_BOOK：每条规则都能在真实语料里命中，且被排除的确实是扫描件
  · dedupe_cross_format：只在「1 pdf + 1 txt 且两者都有实质内容」时才动
  · 反向用例：pdf 明显残缺 / txt 只是残章 / 同名多份时，都不误杀
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import corpus_dedup as cd  # noqa: E402


# ============================================================
# normalize_book_name
# ============================================================

@pytest.mark.parametrize("raw,expected_same_as", [
    # 「上/中/下」是分册标记，刻意不归一（由 KEEP_ONE_PER_BOOK 显式规则处理）
    ("传习录全鉴-迟双明", "传习录全鉴 迟双明"),
    ("传习录全鉴-迟双明.txt", "传习录全鉴-迟双明"),
    ("五百年来王阳明-郦波", "五百年来王阳明郦波"),
    # 括号内容（副标题/生卒年）整体剥掉
    ("知行合一王阳明(1427-1529)-度阴山", "知行合一王阳明-度阴山"),
    ("王阳明大传：知行合一的心学智慧（全新修订版）",
     "王阳明大传：知行合一的心学智慧"),
])
def test_normalize_strips_punct_and_tail(raw, expected_same_as):
    assert cd.normalize_book_name(raw) == cd.normalize_book_name(expected_same_as)


def test_normalize_keeps_bare_digits():
    """生卒年这类有意义的数字不能被当页码剥掉（第一版把它写进 _TAIL，是错的）。"""
    assert "1427" in cd.normalize_book_name("王阳明1427年生")


def test_normalize_distinguishes_different_books():
    """不同作者的两本传记书名相近但不能归到一起（实证：周月亮 vs 度阴山）。"""
    assert cd.normalize_book_name("王阳明大传") != cd.normalize_book_name(
        "知行合一王阳明(1427-1529)-度阴山")


# ============================================================
# 规则表自检
# ============================================================

def test_keep_one_rules_actually_hit_real_corpus():
    """每条排除规则都必须在真实语料里命中，否则就是死规则。"""
    base = Path("data/persona_chat/wangyangming")
    if not base.exists():
        pytest.skip("语料目录不存在")
    names = [p.name for p in base.rglob("*") if p.is_file()]
    for pat, reason in cd.KEEP_ONE_PER_BOOK:
        assert any(pat in n for n in names), f"规则「{pat}」没命中任何真实文件"
        assert len(reason) > 30, f"规则「{pat}」的理由太短，等于没写"


def test_keep_not_excluded_has_evidence():
    """刻意保留的三本传记必须写明「为什么不排」。"""
    assert cd.KEEP_NOT_EXCLUDED
    for k, v in cd.KEEP_NOT_EXCLUDED.items():
        assert len(v) > 15, f"{k} 没写清保留理由"


def test_mojibake_copies_are_all_excluded():
    """两份双重乱码副本都必须被 EXCLUDE_PATTERNS 挡住。

    它们是同一本书的两个标题（「王阳明最神奇的心学」=「发现心灵的智慧」），
    根目录两份都是乱码，子目录各有一份正常 GBK 版。
    漏掉任何一条，正常版就仍会与乱码版一起进集合。
    """
    from init_persona_data import EXCLUDE_PATTERNS, collect_corpus_files

    base = Path("data/persona_chat/wangyangming")
    if not base.exists():
        pytest.skip("语料目录不存在")
    files, _, _, excluded = collect_corpus_files(base)
    excluded_names = {Path(r).name for r, _ in excluded}

    assert "《王阳明最神奇的心学》.txt" in excluded_names
    assert "发现心灵的智慧——王阳明人生哲学感悟.txt" in excluded_names

    # 保留集合里不能出现任何一份乱码副本（子目录的正常版可以留）
    kept = {p.name for p in files}
    assert "发现心灵的智慧——王阳明人生哲学感悟.txt" in kept, \
        "子目录的正常 GBK 版应当保留"

    # 每条规则都得有足够长的理由
    for pat, reason in EXCLUDE_PATTERNS:
        assert len(reason) > 30, f"规则「{pat}」的理由太短"


# ============================================================
# dedupe_cross_format
# ============================================================

def _fake_pdf(monkeypatch, pages, chars):
    """打桩 _pdf_full_chars。"""
    monkeypatch.setattr(cd, "_pdf_full_chars", lambda p: (pages, chars))


def _fake_txt(monkeypatch, chars):
    monkeypatch.setattr(cd, "_txt_chars", lambda p: chars)


def test_drops_pdf_when_same_book_has_txt(monkeypatch, tmp_path):
    """同名 pdf 与 txt 都有实质内容 → 只留 txt。"""
    pdf = tmp_path / "传习录全鉴-迟双明.pdf"
    txt = tmp_path / "传习录全鉴-迟双明.txt"
    pdf.write_bytes(b"x")
    txt.write_bytes(b"x")
    _fake_pdf(monkeypatch, 1139, 240563)
    _fake_txt(monkeypatch, 244840)

    kept, dropped = cd.dedupe_cross_format([pdf, txt])
    assert kept == [txt]
    assert len(dropped) == 1
    assert "排版" in dropped[0][1] or "噪声" in dropped[0][1]


def test_keeps_both_when_pdf_is_scanned(monkeypatch, tmp_path):
    """pdf 是扫描件（抽不出字）→ 不属于本判据范围，两份都留，交给 needs_vision。"""
    pdf = tmp_path / "某扫描书.pdf"
    txt = tmp_path / "某扫描书.txt"
    pdf.write_bytes(b"x")
    txt.write_bytes(b"x")
    _fake_pdf(monkeypatch, 466, 0)          # 扫描件：0 字
    _fake_txt(monkeypatch, 200000)

    kept, dropped = cd.dedupe_cross_format([pdf, txt])
    assert set(kept) == {pdf, txt}
    assert dropped == []


def test_keeps_both_when_txt_is_partial(monkeypatch, tmp_path):
    """txt 只是残章（体量远小于 pdf）→ 不动，避免把完整的 pdf 排掉。"""
    pdf = tmp_path / "甲.pdf"
    txt = tmp_path / "甲.txt"
    pdf.write_bytes(b"x")
    txt.write_bytes(b"x")
    _fake_pdf(monkeypatch, 1000, 300000)
    _fake_txt(monkeypatch, 2000)             # 残章

    kept, dropped = cd.dedupe_cross_format([pdf, txt])
    assert set(kept) == {pdf, txt}
    assert dropped == []


def test_skips_small_pdf(monkeypatch, tmp_path):
    """页数太少 → 采样不可靠，不判。"""
    pdf = tmp_path / "小册子.pdf"
    txt = tmp_path / "小册子.txt"
    pdf.write_bytes(b"x")
    txt.write_bytes(b"x")
    _fake_pdf(monkeypatch, 3, 500)
    _fake_txt(monkeypatch, 90000)

    kept, dropped = cd.dedupe_cross_format([pdf, txt])
    assert set(kept) == {pdf, txt}


def test_skips_when_multiple_same_name(monkeypatch, tmp_path):
    """同名多份 → 交给 sha256 那层，不在这里判。"""
    pdf1 = tmp_path / "甲.pdf"
    pdf2 = tmp_path / "sub" / "甲.pdf"
    txt = tmp_path / "甲.txt"
    for p in (pdf1, pdf2, txt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    _fake_pdf(monkeypatch, 1000, 300000)
    _fake_txt(monkeypatch, 300000)

    kept, dropped = cd.dedupe_cross_format([pdf1, pdf2, txt])
    assert len(kept) == 3
    assert dropped == []


def test_real_corpus_cross_format_pairs_are_deduped():
    """真实语料回归：两组同名跨格式必须被识别出来。"""
    base = Path("data/persona_chat/wangyangming")
    if not base.exists():
        pytest.skip("语料目录不存在")
    files = [p for p in base.rglob("*")
             if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}]
    kept, dropped = cd.dedupe_cross_format(files)
    names = {n for n, _ in dropped}
    assert "传习录全鉴-迟双明.pdf" in names
    assert "传习录 一本书读懂阳明心学.pdf" in names
    # 保留的清单里必须还留着两份 txt
    kept_names = {p.name for p in kept}
    assert "传习录全鉴-迟双明.txt" in kept_names
    assert "传习录 一本书读懂阳明心学.txt" in kept_names
