"""
init_persona_data 的语料收集逻辑回归测试（2026-09-22）

覆盖递归扫描带来的三个新行为——它们都是「不做就会静默出错」的那类，
所以必须有测试钉住：
1. 递归扫子目录（此前只扫直接子文件，王阳明 95 本一本没进）
2. 只吃 .pdf/.md/.txt（.epub/.mobi 二进制硬读会产出乱码块）
3. 按内容哈希去重（同一本书同时在根目录与子目录，不去重就双份入库）
4. 排除规则（读客 5 合 1 那本会把另外四位人物的内容混进王阳明的库）
"""

from pathlib import Path

import pytest

from init_persona_data import (
    EXCLUDE_PATTERNS,
    SUPPORTED_EXTENSIONS,
    collect_corpus_files,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_recursive_scan_picks_up_subdirectories(tmp_path):
    """子目录里的语料必须被扫到"""
    _write(tmp_path / "top.pdf", "top")
    _write(tmp_path / "nested" / "deep" / "book.pdf", "deep")
    _write(tmp_path / "nested" / "notes.md", "notes")

    files, skipped, dups, excluded = collect_corpus_files(tmp_path)

    names = sorted(f.name for f in files)
    assert names == ["book.pdf", "notes.md", "top.pdf"]
    assert not skipped and not dups and not excluded


def test_unsupported_extension_reported_not_guessed(tmp_path):
    """epub/mobi 只能被跳过并计数，不能进候选"""
    _write(tmp_path / "good.pdf", "ok")
    _write(tmp_path / "a.epub", "binary-ish")
    _write(tmp_path / "b.mobi", "binary-ish")
    _write(tmp_path / "c.epub", "binary-ish")

    files, skipped, dups, excluded = collect_corpus_files(tmp_path)

    assert [f.name for f in files] == ["good.pdf"]
    assert skipped[".epub"] == 2
    assert skipped[".mobi"] == 1
    for ext in skipped:
        assert ext not in SUPPORTED_EXTENSIONS


def test_dedup_by_content_keeps_shallowest_path(tmp_path):
    """同内容不同路径 → 只留一份，且留路径最浅的（根目录那份是人工挑过的）"""
    _write(tmp_path / "王阳明大传.pdf", "SAME-CONTENT")
    _write(tmp_path / "电子书（54本）" / "王阳明大传（1本）" / "王阳明大传.pdf", "SAME-CONTENT")

    files, _, dups, _ = collect_corpus_files(tmp_path)

    assert [f.name for f in files] == ["王阳明大传.pdf"]
    assert len(files) == 1
    assert len(dups) == 1
    _, paths = dups[0]
    assert len(paths) == 2
    assert paths[0] == "王阳明大传.pdf"          # 保留的是浅路径


def test_same_name_different_content_is_not_deduped(tmp_path):
    """同名不同内容必须都留下——去重看的是内容，不是文件名"""
    _write(tmp_path / "a" / "book.pdf", "AAA")
    _write(tmp_path / "b" / "book.pdf", "BBB1122")

    files, _, dups, _ = collect_corpus_files(tmp_path)

    assert len(files) == 2
    assert not dups


def test_excluded_pattern_reported_with_reason(tmp_path):
    """命中排除规则的文件要被挡掉，且理由是必填的"""
    pattern = EXCLUDE_PATTERNS[0][0]
    _write(tmp_path / "normal.pdf", "ok")
    _write(tmp_path / f"{pattern}：某某.pdf", "5 in 1")

    files, _, _, excluded = collect_corpus_files(tmp_path)

    assert [f.name for f in files] == ["normal.pdf"]
    assert len(excluded) == 1
    rel, reason = excluded[0]
    assert pattern in rel
    assert reason.strip(), "排除理由不能为空——静默排除是事故的温床"


def test_hidden_files_ignored(tmp_path):
    """点开头的隐藏文件不参与"""
    _write(tmp_path / ".DS_Store", "junk")
    _write(tmp_path / ".hidden.pdf", "junk")
    _write(tmp_path / "real.pdf", "ok")

    files, skipped, _, _ = collect_corpus_files(tmp_path)

    assert [f.name for f in files] == ["real.pdf"]
    assert not skipped


def test_no_files_returns_empty_not_error(tmp_path):
    files, skipped, dups, excluded = collect_corpus_files(tmp_path)
    assert files == [] and not skipped and not dups and not excluded


def test_ocr_is_on_by_default():
    """OCR 默认必须是开的。

    线上库里已经有 OCR 来的内容（荣格《人类与象征》339 页无文本层却有 178 块），
    默认关掉会让下一次重建静默丢掉这些书。跳过 OCR 只能靠显式的 --no-ocr。
    """
    import inspect

    import init_persona_data

    sig = inspect.signature(init_persona_data.load_character_data)
    assert sig.parameters["allow_ocr"].default is False, \
        "load_character_data 的 allow_ocr 默认值应跟随 CLI，由 main 显式传入"

    src = inspect.getsource(init_persona_data.main)
    assert '"--no-ocr"' in src, "应当只有 --no-ocr 逃生开关，没有 --ocr 开关"
    assert "not args.no_ocr" in src, "main 应当把 not args.no_ocr 传给 allow_ocr（默认开）"


# ---------------------------------------------------------------
# 文本编码兜底
# ---------------------------------------------------------------

def test_gbk_text_file_is_decoded(tmp_path):
    """GBK 编码的 txt 必须能读出来，不能整本丢掉。

    实测王阳明语料里就有：同一本书的 UTF-8 版解析成功、GBK 版报
    'utf-8' codec can't decode byte 0xb7 直接失败——按内容哈希去重也救不了它，
    因为两种编码的字节不同、哈希不同。
    """
    from src.document_processing.parser import DocumentParser

    path = tmp_path / "gbk.txt"
    path.write_bytes("王阳明说：心即理也。".encode("gbk"))

    elements = DocumentParser().parse(path)

    assert len(elements) == 1
    assert "心即理也" in elements[0].content


def test_utf8_text_file_still_wins(tmp_path):
    """UTF-8 是首选，不能被兜底编码抢走"""
    from src.document_processing.parser import DocumentParser

    path = tmp_path / "utf8.txt"
    path.write_text("知行合一。", encoding="utf-8")

    elements = DocumentParser().parse(path)
    assert elements[0].content.strip() == "知行合一。"
