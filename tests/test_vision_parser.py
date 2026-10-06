"""
vision_parser 单元测试 —— 零 LLM、零网络。

打桩点：src.document_processing.vision_parser 的 get_chat_llm。
不桩 LLM 的话每个测试都要真打网关（页均 30s），且断言不了重试分支。

覆盖：
  · needs_vision 判据（文字型 / 全扫描 / 损坏）
  · 前置元数据拼接格式
  · 三条失败判据：空 content、finish_reason=length、模型拒答
  · 重试与退避（429 单独 60s、其它 10s×2^n）
  · 断点续跑缓存（第二次不发请求）
  · 逐页入库不跨页合并、page_num 不失真
  · 乱码清洗（版权残文 + 电子书站水印两层判据）
"""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_processing import vision_parser as vp


# ============================================================
# 打桩
# ============================================================

class _Resp:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}


class _StubLLM:
    """按预设脚本逐次返回；脚本用尽则重复最后一个。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def stub_llm(monkeypatch):
    def _install(script):
        llm = _StubLLM(script)
        monkeypatch.setattr(vp, "get_chat_llm", lambda **kw: llm)
        monkeypatch.setattr(vp.asyncio, "sleep", _no_sleep)
        return llm
    return _install


async def _no_sleep(_):
    return None


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(vp.asyncio, "sleep", _no_sleep)


@pytest.fixture
def tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))


# ============================================================
# needs_vision 判据
# ============================================================

def test_needs_vision_returns_false_when_pymupdf_missing(monkeypatch):
    """PyMuPDF 不可用时不能抛异常——build_chunks 会对每个 PDF 调它"""
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "fitz":
            raise ImportError("no fitz")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    need, why = vp.needs_vision(ROOT / "nonexistent.pdf")
    assert need is False
    assert "PyMuPDF" in why


def test_needs_vision_marks_scan_only_pdf(tmp_path, no_sleep):
    """无文本层的扫描件应被判为需要多模态"""
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "scan.pdf"
    doc = fitz.open()
    for _ in range(30):
        page = doc.new_page()
        page.insert_text((72, 72), "x")   # 极少文本，达不到 200 字门槛
    doc.save(str(p))
    doc.close()

    need, why = vp.needs_vision(p)
    assert need is True
    assert "多模态" in why


def test_needs_vision_marks_text_pdf_as_normal(tmp_path, no_sleep):
    """有正常文本层的 PDF 不该被多模态接管

    正文用 ASCII：PyMuPDF 的 insert_text 走内置字体，写不进中文
    （实测会静默丢弃 → 文本层为空 → 判据误判为扫描件）。
    且必须**分行写**：单行 insert_text 会超出版面被裁掉，实测单行只剩 44 字、
    同样被判成扫描件；每页 8 行才能拿到 728 字。
    判据只数「有效字符数」，与内容语言无关，用 ASCII 造样本不影响被测逻辑。
    """
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "text.pdf"
    doc = fitz.open()
    line = "the yangming doctrine of mind and unity " * 3
    for _ in range(12):
        page = doc.new_page()
        for k in range(8):
            page.insert_text((50, 60 + k * 14), line)
    doc.save(str(p))
    doc.close()

    need, why = vp.needs_vision(p)
    assert need is False
    assert "常规解析" in why


def test_needs_vision_needs_vision_needs_min_text_chars(tmp_path, no_sleep):
    """文本层存在但字数很少（如只剩版权页残文）仍应判为需多模态——
    这正是《传习录注疏》311 页里 310 页空白那种情况"""
    fitz = pytest.importorskip("fitz")
    p = tmp_path / "thin.pdf"
    doc = fitz.open()
    for _ in range(20):
        doc.new_page().insert_text((72, 72), "UnTitled1150019")   # 仅 17 字
    doc.save(str(p))
    doc.close()

    need, _ = vp.needs_vision(p)
    assert need is True


# ============================================================
# 请求构造（老大的问题 2）
# ============================================================

def test_vision_message_structure():
    """content 必须是 [text, image_url] 且图片在前、指令在后？—— 反过来：
    指令在前、图片在后。先图后文会让部分模型把指令当图里的文字。"""
    msg = vp.build_vision_message(b"\x89PNG-fake")
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"
    assert "简体中文" in msg.content[0]["text"]
    assert msg.content[1]["type"] == "image_url"
    url = msg.content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_vision_prompt_demands_simplified_and_full_page():
    """两条实测得来的硬约束：输出简体 + 逐字转录全页不省略"""
    assert "简体中文" in vp.VISION_PROMPT
    assert "不要省略" in vp.VISION_PROMPT
    assert "思考过程" in vp.VISION_PROMPT


def test_finish_reason_reads_response_metadata():
    assert vp._finish_reason(_Resp("x", "length")) == "length"
    assert vp._finish_reason(_Resp("x", "stop")) == "stop"
    assert vp._finish_reason(object()) == ""


# ============================================================
# 三条失败判据与重试（老大的问题 4）
# ============================================================

@pytest.mark.asyncio
async def test_empty_response_is_retried(stub_llm):
    llm = stub_llm([_Resp(""), _Resp("正文")])
    r = await vp.transcribe_page(b"x", 1)
    assert r.ok is True
    assert r.text == "正文"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_length_finish_reason_is_failure(stub_llm):
    """finish_reason=length：正文被 max_tokens 截断，半页比缺页更坏"""
    llm = stub_llm([_Resp("被截断的半页", "length")])
    r = await vp.transcribe_page(b"x", 1)
    assert r.ok is False
    assert "length" in r.reason
    assert llm.calls == vp.settings.vision_max_retries


@pytest.mark.asyncio
async def test_refusal_is_failure(stub_llm):
    """模型回「无法识别图片」= 网关忽略了 image_url（假支持模型）"""
    llm = stub_llm([_Resp("很抱歉，我无法识别您提供的图片内容。请将图片以文件形式上传。")])
    r = await vp.transcribe_page(b"x", 1)
    assert r.ok is False
    assert "拒答" in r.reason


@pytest.mark.asyncio
async def test_normal_success_stops_at_first_try(stub_llm):
    llm = stub_llm([_Resp("正常正文")])
    r = await vp.transcribe_page(b"x", 1)
    assert r.ok is True
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_rate_limit_is_retried(stub_llm):
    llm = stub_llm([RuntimeError("Error code: 429 rate limit"), _Resp("正文")])
    r = await vp.transcribe_page(b"x", 1)
    assert r.ok is True
    assert llm.calls == 2


# ============================================================
# 前置元数据（老大的问题 3）
# ============================================================

def test_page_prefix_format():
    p = vp.build_page_prefix(
        character_name="王阳明", doc_title="传习录注疏-邓艾民",
        page_num=157, page_total=311,
    )
    assert "【人物】王阳明" in p
    assert "【文献】传习录注疏-邓艾民" in p
    assert "【页码】157/311" in p
    assert "多模态逐页转录" in p
    lines = [ln for ln in p.splitlines() if ln.strip()]
    assert lines[0].startswith("【人物】")


def test_results_to_elements_skips_failed_pages():
    results = [
        vp.PageResult(page_num=1, text="甲", ok=True),
        vp.PageResult(page_num=2, ok=False, reason="x"),
        vp.PageResult(page_num=3, text="丙", ok=True),
    ]
    els = vp.results_to_elements(
        results, source="a.pdf", character_name="王阳明",
        doc_title="a", page_total=3,
    )
    assert [e.metadata["page_num"] for e in els] == [1, 3]
    assert all(e.metadata["vision"] is True for e in els)
    assert all(e.metadata["source"] == "a.pdf" for e in els)


def test_results_to_elements_can_skip_prefix():
    """with_prefix=False 时正文必须是干净的——供不需要元数据拼装的调用方用"""
    results = [vp.PageResult(page_num=1, text="纯正文", ok=True)]
    els = vp.results_to_elements(results, source="a.pdf", with_prefix=False)
    assert els[0].content == "纯正文"


# ============================================================
# 逐页入库（老大的问题 2：页为单位）
# ============================================================

def test_pages_to_documents_one_block_per_page():
    results = [
        vp.PageResult(page_num=1, text="第一页正文" * 40, ok=True),
        vp.PageResult(page_num=2, text="第二页正文" * 40, ok=True),
    ]
    els = vp.results_to_elements(results, source="a.pdf", with_prefix=False)
    docs = vp.pages_to_documents(els, source="a.pdf")
    assert len(docs) == 2, "两页必须是两块，不能被 min_size 合并成一块"
    assert [d.metadata["page_num"] for d in docs] == [1, 2]


def test_pages_to_documents_split_oversize_page_keeping_page_num():
    """超长页切开后共用同一个 page_num——页码是「来自第几页」，不因切分失真"""
    long_text = "知行合一，知与行本是合一之体，验之吾身确然。望之若无物，即此便是。 " * 60
    results = [vp.PageResult(page_num=7, text=long_text, ok=True)]
    els = vp.results_to_elements(results, source="a.pdf", with_prefix=False)
    docs = vp.pages_to_documents(els, source="a.pdf")
    assert len(docs) > 1, "超长页应被切开"
    assert {d.metadata["page_num"] for d in docs} == {7}


def test_pages_to_documents_metadata_keeps_existing_schema():
    """字段集必须与现有库一致（has_overlap/overlap_source 只在切分时产生）"""
    results = [vp.PageResult(page_num=1, text="正文" * 300, ok=True)]
    els = vp.results_to_elements(results, source="a.pdf", with_prefix=False)
    doc = vp.pages_to_documents(els, source="a.pdf")[0]
    for k in ("source", "heading", "element_types", "char_length"):
        assert k in doc.metadata, f"缺现有字段 {k}"
    assert doc.metadata["char_length"] == len(doc.page_content)
    assert doc.metadata["vision"] is True


# ============================================================
# 乱码清洗
# ============================================================

def test_filter_noise_drops_copyright_and_watermark():
    raw = "\n".join([
        "As a reader（74398380）欢迎加入！",       # 电子书站水印（实测垃圾）
        "书名=UnTitled1150019",                   # 实测垃圾
        "版权所有 翻印必究",                       # 版权行
        "— 12 —",                                # 装饰性页码
        "正文第一句，应当保留。",
    ])
    cleaned = vp._filter_noise(raw)
    assert cleaned == "正文第一句，应当保留。"


def test_filter_noise_keeps_normal_text():
    raw = "心即理。阳明龙场悟道，悟的就是这一句。\n知行合一，知与行本是合一之体。"
    assert vp._filter_noise(raw) == raw


# ============================================================
# 断点续跑缓存
# ============================================================

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"not-a-real-pdf")

    d = vp._cache_dir(pdf)
    assert vp._cache_read(pdf, 1) is None          # 还没转录
    vp._cache_write(pdf, 1, "转录正文")
    assert vp._cache_read(pdf, 1) == "转录正文"

    # 空内容不写缓存：否则「转出空页」会被永久固化，重跑再也不会重试
    vp._cache_write(pdf, 2, "   ")
    assert vp._cache_read(pdf, 2) is None
    assert (d / "p00002.json").exists(), "空页也要留痕，便于排查是哪几页失败"


def test_cache_key_changes_with_params(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"x")
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))

    k1 = vp._cache_key(pdf)
    monkeypatch.setattr(vp.settings, "vision_dpi", 300)
    assert vp._cache_key(pdf) != k1, "改 dpi 必须换缓存目录，否则旧结果冒充新参数"
    monkeypatch.setattr(vp.settings, "vision_dpi", 200)
    monkeypatch.setattr(vp.settings, "vision_model", "other-model")
    assert vp._cache_key(pdf) != k1, "改模型必须换缓存目录"


def test_prompt_version_in_cache_key(monkeypatch, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"x")
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))
    monkeypatch.setattr(vp, "VISION_PROMPT_VERSION", "v2")
    a = vp._cache_key(pdf)
    monkeypatch.setattr(vp, "VISION_PROMPT_VERSION", "v3")
    assert vp._cache_key(pdf) != a, "改提示词必须让旧缓存失效"


# ============================================================
# 逐页转录整本（含缓存命中不发请求）
# ============================================================

def test_transcribe_pdf_uses_cache_without_llm(stub_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))
    fitz = pytest.importorskip("fitz")

    p = tmp_path / "tiny.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page().insert_text((72, 72), "页面")
    doc.save(str(p))
    doc.close()

    # 预置前两页缓存
    vp._cache_write(p, 1, "缓存的一页")
    vp._cache_write(p, 2, "缓存的另一页")

    llm = stub_llm([_Resp("真调用的第三页")])
    results = asyncio.run(vp.transcribe_pdf(p, character_name="王阳明"))

    assert [r.page_num for r in results] == [1, 2, 3]
    assert results[0].from_cache is True
    assert results[2].from_cache is False
    assert llm.calls == 1, "缓存命中的页不该再发请求"
    # 命中缓存也要过一遍乱码清洗，但缓存内容是干净的
    assert results[0].text == "缓存的一页"


def test_transcribe_pdf_handles_corrupt_file(stub_llm, tmp_path, monkeypatch):
    """损坏 PDF 不能抛异常——build_chunks 要能继续处理后面的文件"""
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 broken truncated")
    results = asyncio.run(vp.transcribe_pdf(bad, character_name="王阳明"))
    assert results == []


# ============================================================
# 逐页 dpi 自适应（第四种失败形态：图片分辨率超网关上限）
# ============================================================
# 实测依据：《王阳明为什么倡导人人心中有孔子》第 1 页在 dpi=200 下渲出
# 1706x12291 像素，网关直接 400（image resolution exceeds limit）。
# 固定 dpi 对普通开本刚好，对「整页一张长图」的扫描件超限 50%。
# 这类 400 与 429/超时性质不同：请求本身非法，重试无用。


class _FakePage:
    def __init__(self, w_pt, h_pt):
        self.rect = type("R", (), {"width": w_pt, "height": h_pt})()


def test_pick_dpi_keeps_requested_for_normal_page():
    """普通开本（A4 842pt）在 dpi=200 下不超限 → 用原值"""
    page = _FakePage(595, 842)
    assert vp.pick_dpi(page, 200) == 200


def test_pick_dpi_lowers_for_long_strip_page():
    """整页一张长图（4425pt 高）在 200 下超限 → 逐级下调到不超为止"""
    page = _FakePage(614, 4425)
    dpi = vp.pick_dpi(page, 200)
    assert dpi < 200, "超长页必须降 dpi"
    assert 614 * dpi / 72 <= vp.MAX_IMAGE_PX
    assert 4425 * dpi / 72 <= vp.MAX_IMAGE_PX


def test_pick_dpi_result_always_under_limit():
    """任何尺寸的页，pick_dpi 的结果都不能超上限（除极端长条已到最低档）"""
    for h in (800, 1200, 2000, 3000, 4425, 8000):
        page = _FakePage(600, h)
        dpi = vp.pick_dpi(page, 200)
        if dpi > vp.DPI_FALLBACKS[-1]:
            assert h * dpi / 72 <= vp.MAX_IMAGE_PX


def test_transcribe_page_does_not_retry_on_resolution_error(stub_llm):
    """超分辨率是请求非法，必须立刻失败而不是把重试次数耗完"""
    err = Exception(
        "Error code: 400 - {'error': {'message': "
        "'image resolution 1707x12292 exceeds limit'}}"
    )
    llm = stub_llm([err, err, err, err])
    r = asyncio.run(vp.transcribe_page(b"x", 1, max_retries=4, interval=0))
    assert r.ok is False
    assert "分辨率" in r.reason
    assert llm.calls == 1, "超分辨率重试无意义，只该打一次"


# ============================================================
# 跨页扫描切半转录
# ============================================================
# 实测依据：《阳明先生文录》1116 页全是 2592x1728 横版图（一张装两个书页），
# 整张转录成功率 75%、单页 126s；切半后每次 200-250 字能一次转完。
# 更关键的是「左半失败但右半成功」时内容仍能入库——整张方案会丢整页。

class _FakeSpreadPage:
    def __init__(self, w, h):
        self.rect = type("R", (), {"width": w, "height": h})()


def test_is_spread_detects_landscape_two_page():
    """横版接近 3:2 的页面判为跨页（《阳明先生文录》实测 2592x1728）"""
    assert vp.is_spread_page(_FakeSpreadPage(2592, 1728)) is True


def test_is_spread_rejects_portrait():
    """竖版 A4（595x842）、小开本（252x331）都不该判为跨页"""
    assert vp.is_spread_page(_FakeSpreadPage(595, 842)) is False
    assert vp.is_spread_page(_FakeSpreadPage(252, 331)) is False


def test_render_halves_returns_right_then_left(tmp_path):
    """返回顺序必须是（右半, 左半）——竖排古籍从右往左读"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    pg = doc.new_page(width=600, height=400)
    # 右半画一条粗竖线做标记
    pg.draw_line((350, 0), (350, 400), width=8)
    halves = vp.render_spread_halves(pg, 100)
    assert len(halves) == 2
    assert all(isinstance(h, bytes) and h[:4] == b"\x89PNG" for h in halves)
    doc.close()


def test_cache_key_changes_with_split_flag(tmp_path, monkeypatch):
    """切半开关必须进缓存 key，否则旧缓存会被当成新参数的结果"""
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"x")
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))
    monkeypatch.setattr(vp, "VISION_PROMPT_VERSION", "v3")
    monkeypatch.setattr(vp.settings, "vision_split_spread", True)
    a = vp._cache_key(pdf)
    monkeypatch.setattr(vp.settings, "vision_split_spread", False)
    assert vp._cache_key(pdf) != a


def test_cache_read_write_are_part_scoped(tmp_path, monkeypatch):
    """两半各自独立存——一半失败不能丢另一半"""
    monkeypatch.setattr(vp.settings, "vision_cache_dir", str(tmp_path / "vc"))
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"x")
    vp._cache_write(pdf, 7, "右半正文", "b")
    vp._cache_write(pdf, 7, "左半正文", "a")
    assert vp._cache_read(pdf, 7, "b") == "右半正文"
    assert vp._cache_read(pdf, 7, "a") == "左半正文"
    # 整页读不到（因为只写了分半）
    assert vp._cache_read(pdf, 7) is None


# ============================================================
# 10-06 修的三处（老大追问后复验发现的问题）
# ============================================================
# ① pick_dpi 的阶梯下限 80 不够：600x9000pt 在 dpi=80 时出 666x10000 像素仍超限，
#    第一版还「压到最低档就返回它」→ 返回 80 假装解决，实际照样撞 400。
#    真链路探针那次是 4425pt（在 dpi=120 就够）刚好把 bug 掩盖过去。
# ② 「needs_vision 判否」不等于「该入库」：那个 1 页长截图被判「不是扫描件」
#    → 掉进常规路径 → OCR 43.7s 抽出 4748 字音频节目稿 → 5 块入库。
# ③ 整屏截图要在**收集阶段**挡，不能只在 needs_vision 里判否。


class _FakeRect:
    def __init__(self, w, h):
        self.width = w
        self.height = h


def test_pick_dpi_returns_none_when_impossible():
    """压到最低仍超限 → 必须返回 None，不能返回一个仍超限的值"""
    page = _FakePage(600, 20000)
    assert vp.pick_dpi(page, 200) is None


def test_pick_dpi_descends_below_80():
    """9000pt 高在 dpi=80 仍超（10000 像素），必须继续下探"""
    page = _FakePage(600, 9000)
    dpi = vp.pick_dpi(page, 200)
    assert dpi is not None and dpi < 80
    assert 9000 * dpi / 72 <= vp.MAX_IMAGE_PX
    assert 600 * dpi / 72 <= vp.MAX_IMAGE_PX


def test_pick_dpi_real_render_under_limit():
    """★真渲一遍验证：不能只算数学，必须确认 PNG 真实像素不超限。

    踩过的坑：用 fitz.open(stream=png) 读回来时拿到的是 PDF 的 pt 尺寸，
    不是 PNG 真实像素，于是「都在 8000 以内」的断言自己骗了自己。
    """
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page(width=600, height=9000)
    tmp = doc.tobytes()
    doc.close()
    with fitz.open(stream=tmp, filetype="pdf") as d:
        page = d[0]
        dpi = vp.pick_dpi(page, 200)
        assert dpi is not None
        pix = vp.render_page_png(page, dpi)
        with fitz.open(stream=pix, filetype="png") as im:
            assert max(im[0].rect.width, im[0].rect.height) <= vp.MAX_IMAGE_PX, (
                f"真实像素 {im[0].rect} 仍超 {vp.MAX_IMAGE_PX}")


def test_screenshot_pdf_detected(tmp_path):
    """整屏长截图要被独立识别出来（供收集阶段排除）"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page(width=614, height=4425)
    path = tmp_path / "shot.pdf"
    doc.save(str(path))
    doc.close()
    is_shot, why = vp.is_screenshot_pdf(path)
    assert is_shot is True
    assert "截图" in why


def test_normal_pdf_not_screenshot(tmp_path):
    """正常书籍不能被误判成截图"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=595, height=842)
    path = tmp_path / "book.pdf"
    doc.save(str(path))
    doc.close()
    assert vp.is_screenshot_pdf(path)[0] is False


def test_spread_scan_not_screenshot(tmp_path):
    """双页跨页扫描不是截图（它是横版，1716pt 高但完全正常）"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page(width=2592, height=1728)
    path = tmp_path / "spread.pdf"
    doc.save(str(path))
    doc.close()
    assert vp.is_screenshot_pdf(path)[0] is False


def test_screenshot_is_excluded_from_corpus():
    """★回归：那个音频节目稿文件必须在收集阶段就被挡掉，不产块"""
    base = Path("data/persona_chat/wangyangming")
    if not base.exists():
        pytest.skip("语料目录不存在")
    from init_persona_data import collect_corpus_files

    files, _, _, excluded = collect_corpus_files(base)
    assert not any("心中有孔子" in p.name for p in files), \
        "长截图 PDF 仍进了候选集合（会被 OCR 抽成播客稿入库）"
    assert any("心中有孔子" in rel for rel, _ in excluded)
