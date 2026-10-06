"""
vision_parser.py — 扫描件 PDF 的多模态逐页转录

要解决的问题：语料里有一批 PDF **没有可用文本层**，走现有解析链路只会抽出
版权页残文或整页广告。比如实测三本：

    传习录注疏-邓艾民.pdf        311 页只有 1 页有文本层 → 抽出「[General Information]书名=UnTitled1150019」
    阳明学述要 钱穆.pdf          433 页只有 4 页有文本层 → 抽出「As a reader（74398380）欢迎加入！」× 4
    王阳明为什么倡导人人心中有孔子.pdf  文件损坏，pypdf 直接 Stream has ended unexpectedly

它们走不到本地 OCR（`PDFParser` 只要拿到任意元素就不再降级），于是要么灌 1~4 块
垃圾进检索，要么被 `no_text_extracted` 占位挡在门外。

这里的做法：**按页渲成图片 → 交多模态模型转录正文**，不做任何本地 OCR 预处理。

为什么不本地 OCR（RapidOCR）而绕道多模态：
  · CPU OCR 约 3 秒/页，745 页约 37 分钟纯计算，且识别质量随书体差异很大
    （竖排古籍、古刻本订正号），实测「混合型 PDF 静默丢页」这个坑至今没解
  · 多模态对竖排繁体古籍的识别实测可用（见 output/_probe_vision3.py），
    且能靠提示词直接输出简体，不引入 opencc/zhconv 新依赖
  · 代价是要占用网关配额：页均 12.8-48.7s，745 页约 6-8 小时。已做断点续跑缓存。

模型选型（不能照名字挑，必须实测）：
    网关 token.sensenova.cn 当前 9 个模型里**只有 sensenova-6.8-flash-lite 真支持
    视觉**。deepseek-v4-flash 是假支持——默认模式 4000 completion 全被 thinking
    吃掉、finish_reason=length、正文为空；关掉思考后回「无法识别您提供的图片内容」。
    换模型前用 output/_probe_vision2.py 的两条判据复测：
      ① usage.prompt_tokens 是否随图片显著上升（不升 = 网关忽略 image_url）
      ② finish_reason == "length" 且 completion 撞 max_tokens = 思考吃光预算

输出与现有流程的衔接：
    本模块产出的 ParsedElement 字段与 `parser.py` 的 PDF 路径完全一致
    （content / element_type / metadata{source, page_num, ...}），
    可直接喂 `AdaptiveChunker.chunk(elements, source=...)`；
    也提供 `pages_to_documents()` 走「一页一块」的逐页入库路径（见函数文档）。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage

import fitz

from src.core.config import settings
from src.core.llm import get_chat_llm
from src.document_processing.parser import ParsedElement, _is_ocr_noise_line

# ============================================================
# 转录提示词
# ============================================================
# 改这个字符串会让全部断点缓存失效（缓存 key 含 PROMPT_VERSION），重跑要再花几小时，
# 所以措辞改动要慎之又慎。实测有效的一条硬约束是「必须输出简体中文」：
# 不写这句，模型会把竖排繁体原样转出来，embedding 就与库里 5 个简体源分处不同语义位置。
#
# 「逐字转录全页，不要省略任何段落」是后补的：实测 max_tokens=4000 时
# 《传习录注疏》p157 撞 finish_reason=length，只转出 327 字就被切，
# 半页正文入库比缺页更坏（会生成看似完整实则残缺的引用）。
# 同理「不要输出思考过程」——某些网关配置下模型的 thinking 会混进 content。

VISION_PROMPT_VERSION = "v3"

VISION_PROMPT = """你是一个古籍 OCR 引擎。请把图片里的文字逐字转录为纯文本。

规则：
1. 只转录图片中实际印刷出来的文字，不加任何解释、评论、标题或按语。
2. ★必须输出简体中文。原书是竖排繁体，请逐字转换成简体再输出——
   例如「書」→「书」、「學」→「学」、「無」→「无」、「弟輩」→「弟辈」、
   「明覺」→「明觉」、「發」→「发」、「雲」→「云」、「說」→「说」。
   这一点极其重要：输出繁体会让检索时与简体语料落在不同的语义位置。
   如果你不确定某个字怎么写，请选择最常用的那个简体字。
3. 逐字转录全页正文，不要省略、不要概括、不要用「（略）」代替任何段落。
   图片里有两个书页（左页和右页）时，两页的文字都要转录，中间空一行分隔。
4. 段落之间用空行分隔；对话、条目按原样分行。
5. 不要输出思考过程、推理或任何非图片内容。不要在开头或结尾加任何说明。
6. 版心、页眉页脚的页码可以不转录。红圈、红点等圈点符号不需要输出。
7. 若整页没有任何正文文字（纯插图、空白页、纯装饰），输出空字符串，不要解释原因。"""

# 模型明明收不到图片时会回的固定话术。命中即判失败重试 ——
# 防止有人把 vision_model 换成一个「假支持」模型后，
# 整本书入库成一千多页「我无法识别您提供的图片内容」，而日志看起来一片正常。
_REFUSAL_MARKERS = (
    "无法识别",
    "无法查看",
    "无法直接",
    "以文件形式上传",
    "以文字形式",
    "请将图片",
    "请提供图片",
    "无法解析您提供",
    "我看不到",
)

# 前置元数据块：拼在每页正文之前，一份进 page_content（让 embedding 看到出处），
# 一份进 metadata（供溯源与前端引用标注）。字段名与现有向量化流程对齐。
_PREFIX_LABELS = ("人物", "文献", "页码", "来源格式")


# ============================================================
# 单页转录结果
# ============================================================

@dataclass
class PageResult:
    """一页的转录结果（含成功/失败状态，供调用方汇总）"""

    page_num: int
    text: str = ""
    ok: bool = False
    reason: str = ""
    secs: float = 0.0
    from_cache: bool = False


# ============================================================
# 断点续跑缓存
# ============================================================
# 2984 页按页均 9-126s 约 5-40 小时，中间必然会断（网络抖动、机器休眠、手动 Ctrl-C）。
# 缓存按「文件绝对路径 + mtime + 尺寸 + dpi + 模型 + 提示词版本 + 是否切半」哈希分目录，
# 改任何一个参数都会自动换目录，不会拿旧参数的结果冒充新结果。

def _cache_key(pdf: Path) -> str:
    st = pdf.stat()
    raw = "|".join([
        str(pdf.resolve()),
        f"{int(st.st_mtime)}",
        f"{st.st_size}",
        f"dpi{settings.vision_dpi}",
        f"m{settings.vision_model}",
        f"p{VISION_PROMPT_VERSION}",
        f"split{1 if settings.vision_split_spread else 0}",
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_dir(pdf: Path) -> Path:
    d = Path(settings.vision_cache_dir) / _cache_key(pdf)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_read(pdf: Path, page_num: int, part: str = "") -> Optional[str]:
    """读缓存。part 为空表示整页；"a"/"b" 表示跨页切半后的左/右半。

    为什么要按半页存：跨页图切半后两半可能一边成功一边撞 max_tokens。
    如果仍按整页存，一半失败就把整页丢了 —— 而这正是 1116 页古籍的主要失败形态。
    """
    suffix = f"_{part}" if part else ""
    f = _cache_dir(pdf) / f"p{page_num:05d}{suffix}.json"
    if not f.exists():
        return None
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    text = rec.get("text") or ""
    return text if text.strip() else None


def _cache_write(pdf: Path, page_num: int, text: str, part: str = "") -> None:
    suffix = f"_{part}" if part else ""
    f = _cache_dir(pdf) / f"p{page_num:05d}{suffix}.json"
    f.write_text(json.dumps({
        "page": page_num,
        "part": part or "full",
        "chars": len(text),
        "model": settings.vision_model,
        "dpi": settings.vision_dpi,
        "prompt_version": VISION_PROMPT_VERSION,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
    }, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 渲图 + 请求构造
# ============================================================

# 网关对单张图片的像素上限。实测超出会直接 400 拒绝：
#   《王阳明为什么倡导人人心中有孔子》第 1 页在 dpi=200 下渲出 1706x12291，
#   报 `image resolution 1707x12292 exceeds limit`。
# 这是**第四种失败形态**：400 BadRequest，重试再多次也一样（请求本身非法），
# 不像 429/超时那样重试能解决。
MAX_IMAGE_PX = 8000
# 自适应时的降级阶梯（从低到高请求，取第一个不超限的）。
# ★下限必须探到很低：实测 600x9000pt 的页在 dpi=80 时仍出 666x10000 像素（超限），
# 压到 40 才够。第一版阶梯最低 80、且「压到最低档就返回它」，
# 结果函数返回 80 假装解决了，实际照样撞 400 —— 而真链路探针那次
# 恰好是 4425pt 的页（在 dpi=120 就够，7375 像素），把 bug 掩盖过去了。
DPI_FALLBACKS = (200, 150, 120, 100, 80, 60, 50, 40, 30, 20)
# 低于这个 dpi 字就糊了（印刷体 5pt 以下基本认不出），再低不如不转
MIN_USABLE_DPI = 30

# 竖版长截图的判定阈值（高:宽 达到这个值就是「整屏/整篇拼成一张图」）。
# 竖版 A4 是 842:595 ≈ 1.41；双页跨页扫描是 1728:2592（横向，不触发）。
# 实测那个公众号长截图是 4425:614 ≈ 7.2，所以 3.0 留足了余量。
LONG_SHOT_ASPECT = 3.0

# 前 N 页按「易失败页」处理：撞 length 就放弃不重试。
# 依据：实测《王阳明心学口诀》p1 是彩色封面（艺术字 + 肖像），
# dpi=200 跑 106s 后 finish_reason=length、一个字没吐出来；p2 正文页 3.9s 就转出 57 字。
# 封面/扉页/版权页本来就该丢（是排版信息不是正文），不值得为它花两次重试。
FRONT_MATTER_PAGES = 3


def pick_dpi(page, preferred: int) -> int:
    """按页面实际尺寸挑一个不超 MAX_IMAGE_PX 的 dpi。

    为什么必须逐页算：古籍扫描里有一类「整页拼成一张长图」的 PDF，
    页面点高能到 9000pt（正常 A4 只有 842pt）。固定 dpi=200 对普通开本刚好，
    对这类页面直接超限十几倍。

    ★返回 `None` 表示「压到 MIN_USABLE_DPI 仍超限」——调用方必须把它当失败处理，
    不能拿一个仍超限的值去请求（那是白等一次 400）。第一版没做这个区分，
    在 600x9000pt 的页上返回 80（实际 10000 像素）假装解决了。

    实测：600x4425 → 120（1000x7375）；600x9000 → 40（333x5000）；
          600x20000 → None（30dpi 仍出 2500x8333）
    """
    rect = page.rect
    longest_pt = max(rect.width, rect.height) or 1.0
    for dpi in (preferred, *DPI_FALLBACKS):
        if dpi < MIN_USABLE_DPI:
            break
        if longest_pt * dpi / 72 <= MAX_IMAGE_PX:
            return dpi
    return None


def render_page_png(page, dpi: int) -> bytes:
    """PyMuPDF 单页 → PNG 字节。

    为什么不直接送 PDF：多模态接口只收图片，送 PDF 的话既没法控制看哪一页，
    也没法按页重试（失败就得重传整本书）。
    dpi 取 200 而非 150：实测两者失败率都是 1/4，但 200 的识别更完整
    （页均 23-44s vs 13-49s，长尾更稳）。超长页面由 `pick_dpi` 逐页下调。
    """
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def is_spread_page(page) -> bool:
    """判断这一页是不是「双书页跨页扫描」（一张图里装了两个书页）。

    实测依据：《阳明先生文录》全部 1116 页都是 2592x1728pt 的**横版**图，
    打开看是左右两个书页（带档案馆色卡和标尺），约 35 行竖排繁体、
    400-500 字/张。这类页整张转录的成功率只有 75%——正文没写完就撞
    max_tokens，实测 4 页里 p103/p104 两次都失败。

    判据用**宽高比**而不是绝对尺寸：横版（宽 > 高）且接近 3:2 就是跨页扫描。
    竖版 A4（595x842）是 0.71，不会误判；竖版长截图已在 needs_vision 里挡掉。
    """
    rect = page.rect
    return rect.width > rect.height and rect.width / max(rect.height, 1) >= 1.2


def render_spread_halves(page, dpi: int) -> list[bytes]:
    """把一张跨页图切成左半、右半两张分别渲图。

    为什么切：跨页图内容量是普通页的两倍（400-500 字），模型的 thinking
    吃掉的预算随之翻倍，导致「正文没转完就 length」。切成两半后每次
    只处理 200-250 字，max_tokens 就够用了。

    dpi 按**半页尺寸**重算：整页 2592pt 宽在 dpi=200 下已经是 7200 像素，
    切半后每半只有 3600 像素，本可以给更高 dpi 换更清晰的识别。
    这里仍从传入的 dpi 起算并按需下调，保证不超 MAX_IMAGE_PX。

    注意竖排古籍的阅读顺序是**右起**，所以右半在前、左半在后。
    """
    rect = page.rect
    mid = rect.x0 + rect.width / 2
    right = fitz.Rect(mid, rect.y0, rect.x1, rect.y1)
    left = fitz.Rect(rect.x0, rect.y0, mid, rect.y1)

    # 半页最长边决定 dpi 上限。**必须用「从 dpi 起逐步下调」而不是固定乘 0.85**：
    # 实测 600x9000pt 的页在 dpi=80 时出 666x10000 像素（仍超 8000），
    # 而一步降到 0.85×80=68 也还是 941 宽 × 8500 高——照样超。
    half_longest = max(right.width, right.height, left.width, left.height) or 1.0
    safe_dpi = None
    for cand in (dpi, *DPI_FALLBACKS):
        if cand >= MIN_USABLE_DPI and half_longest * cand / 72 <= MAX_IMAGE_PX:
            safe_dpi = cand
            break
    if safe_dpi is None:
        raise ValueError(
            f"半页仍然过长：{int(half_longest)}pt，dpi 降到 {MIN_USABLE_DPI} "
            f"仍超 {MAX_IMAGE_PX} 像素"
        )

    out = []
    for clip in (right, left):
        pix = page.get_pixmap(dpi=safe_dpi, clip=clip)
        out.append(pix.tobytes("png"))
    return out


def build_vision_message(png_bytes: bytes) -> HumanMessage:
    """构造单页转录请求（老大的问题 2 落在这里）。

    图片编码方式：PNG 字节 → base64 → data URL，声明 image/png。
    这是 OpenAI 兼容多模态接口的标准形态，sensenova 网关同样吃这套
    （实测图片贡献约 +300 prompt token，见 _probe_vision2.py）。

    content 是一个列表：第一个块是文本指令，后面的块是图片 ——
    顺序不能反，先图后文会让部分模型把指令当成图里的文字。
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return HumanMessage(content=[
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])


def _finish_reason(resp) -> str:
    """取 finish_reason：LangChain 把它放在 response_metadata 里。"""
    meta = getattr(resp, "response_metadata", None) or {}
    fr = meta.get("finish_reason") or meta.get("stop_reason") or ""
    return str(fr).lower()


def _is_refusal(text: str) -> bool:
    return any(m in text for m in _REFUSAL_MARKERS)


# ============================================================
# 单页转录（含重试与节流）
# ============================================================

async def transcribe_page(
    png_bytes: bytes,
    page_num: int,
    max_retries: Optional[int] = None,
    interval: Optional[float] = None,
) -> PageResult:
    """转录一页，失败按 max_retries 重试。

    本函数**不碰缓存**——读缓存、写缓存、清洗、落统计都在 transcribe_pdf 里。
    分工是刻意的：单页函数保持无状态，方便直接调用做真链路验证
    （output/_vision_live_probe.py 就直接调它打真网关打一次，验证图片有没有
    真进上下文、输出是不是简体）。所以调本函数不会留下缓存文件，
    只有走整本流程才会落盘。

    失败判据有四条（老大的问题 4 落在这里），比「content 为空」更严：

      1. **空 content** —— 上游 200 但返回空壳，本项目已在 llm.py 里
         记录过这种随机故障（约 13s 后回空壳），必须重试。
      2. **finish_reason == "length"** —— 正文被 max_tokens 截断。
         这是推理模型最典型的失败形态：completion 直接撞上限，
         content 里全是 thinking 或者只转出半页。半页正文比缺页更坏，
         因为它看起来是完整的，入库后会生成残缺引用。
      3. **模型回"无法识别图片"** —— 网关忽略了 image_url（假支持）。
         换模型时最容易踩，不拦住就是整本书变成一千多页拒答。
      4. **400 image resolution exceeds limit** —— 图片像素超网关上限。
         这是**请求本身非法**，重试无用（跟 429/超时性质不同）。
         真正的解法在上游：`transcribe_pdf` 用 `pick_dpi` 逐页压 dpi。
         这里必须显式判出来，否则会白白重试完所有次数再报同一个错。

    退避策略：区分「内容问题」和「配额/网络问题」。
      · 前三类（空/截断/拒答）立刻重试 —— 上游随机性下重试命中率很高
      · 400 超分辨率 → 立刻返回失败（重试无意义，让调用方降 dpi 重来）
      · 429 退避 60s（网关可持续 3-6 次/分，刚撞额度就得让一会儿）
      · 其余异常（超时/502/隧道错误）按 10s × 2^n 退避
    """
    attempts = settings.vision_max_retries if max_retries is None else max_retries
    gap = settings.vision_page_interval_sec if interval is None else interval
    # 注意 ChatOpenAI 没有 `timeout` 字段，只有 `request_timeout`（llm.py 的
    # setdefault 也用这个键），传错名字会被 pydantic 静默忽略、退回 60s 默认值。
    # 显式 model 本身就是短路，不会被套上 fallback 链——多模态没有备用模型可选。
    llm = get_chat_llm(
        model=settings.vision_model,
        with_fallback=False,
        temperature=settings.vision_temperature,
        max_tokens=settings.vision_max_tokens,
        request_timeout=settings.vision_page_timeout_sec,
    )
    message = build_vision_message(png_bytes)

    t0 = time.time()
    last = "未知原因"
    for i in range(attempts):
        try:
            resp = await llm.ainvoke([message])
            text = (getattr(resp, "content", None) or "").strip()
            fr = _finish_reason(resp)

            if fr == "length":
                last = "finish_reason=length（正文被 max_tokens 截断）"
            elif not text:
                last = "上游返回空响应"
            elif _is_refusal(text):
                # 拒答内容不入库，但要重试：同一模型下页与页之间偶发失手是可能的
                last = f"模型拒答（疑似未收到图片）: {text[:60]}"
            else:
                return PageResult(
                    page_num=page_num, text=text, ok=True,
                    secs=time.time() - t0, reason="",
                )
        except Exception as e:
            name = type(e).__name__
            msg = str(e)
            if "exceeds limit" in msg or ("resolution" in msg and "400" in msg):
                # 请求本身非法，重试一万次也是同样结果
                last = f"图片分辨率超网关上限（{msg[:100]}）"
                return PageResult(page_num=page_num, ok=False, reason=last,
                                  secs=time.time() - t0)
            if "429" in msg or "rate" in msg.lower():
                last = f"429 限流: {msg[:80]}"
                await asyncio.sleep(60)
            elif "timeout" in name.lower() or "Timeout" in msg:
                last = f"超时（>{settings.vision_page_timeout_sec:.0f}s）"
                await asyncio.sleep(10)
            else:
                # 502 / Tunnel connection failed 这类网关瞬时故障
                last = f"{name}: {msg[:80]}"
                await asyncio.sleep(10 * (2 ** i))
        if i < attempts - 1:
            print(f"      [p{page_num}] 第 {i + 1}/{attempts} 次失败（{last}），重试…")
    return PageResult(page_num=page_num, ok=False, reason=last, secs=time.time() - t0)


# ============================================================
# 整本逐页转录
# ============================================================

# 多模态转录特有的补充噪声判据。
#
# 为什么不直接改 parser.py 的 _OCR_NOISE_PATTERNS：那套判据已经被 5 个库、
# 上万条既有 chunk 依赖，往里加规则会同时改变线上检索结果（属于改口径），
# 而这里只需要救回扫描件。
#
# 下面这几条全部来自实测三本薄 PDF 抽出的真实垃圾，逐条对应：
#   「As a reader（74398380）欢迎加入！」 ← 阳明学述要 钱穆.pdf 433 页里 4 页全是它
#   「书名=UnTitled1150019」              ← 传习录注疏-邓艾民.pdf 唯一那页文本层
#   「UnTitled」/「各类PDF电子书」        ← 盗版电子书站的水印与文件名占位
_VISION_NOISE_PATTERNS = [
    r"欢迎加入",
    r"一个分享阅读体验和求书找书的平台",
    r"As a reader",
    r"UnTitled\d*",
    r"书名\s*[=＝]",
    r"各类\s*PDF\s*电子书",
    r"电子书\s*下载",
    r"最新电子书",
    r"免费分享社",
    r"免费分享\s*[:：]?",
    r"仅供.{0,4}(试读|学习|交流)",
    r"请支持正版",
    r"侵权.{0,4}联系",
    r"本文档由.{0,10}(整理|制作|扫描)",
    r"扫描全能版",
    r"^\s*第?\s*\d+\s*页?\s*$",
]


def _is_vision_noise_line(line: str) -> bool:
    """补充判据：电子书站水印 / 文件名占位 / 孤立页码。"""
    compact = re.sub(r"\s+", "", line)
    if not compact:
        return True
    for pattern in _VISION_NOISE_PATTERNS:
        if re.search(pattern, compact):
            return True
    return False


def _filter_noise(text: str) -> str:
    """丢掉转录结果里的出版信息 / 水印 / 孤立页码。

    两层判据叠加：
      ① `_is_ocr_noise_line` —— parser.py 里那套版权页/装饰页码判据，
         复用它免得两条解析路径的清洗规则各说各话
      ② `_is_vision_noise_line` —— 上面那批多模态特有的水印判据
    清洗后整页为空 = 这页实质没有正文，调用方按「无内容」处理，不入库不写缓存。
    """
    kept = [ln for ln in text.splitlines()
            if not _is_ocr_noise_line(ln) and not _is_vision_noise_line(ln)]
    return "\n".join(kept).strip()


def build_page_prefix(
    *,
    character_name: str,
    doc_title: str,
    page_num: int,
    page_total: int,
) -> str:
    """前置元数据块（老大的问题 3 落在这里）。

    拼在 page_content 正文之前，格式固定为四行标签 + 分隔线：

        【人物】王阳明
        【文献】传习录注疏-邓艾民
        【页码】157/311
        【来源格式】扫描件·多模态逐页转录
        ————————————————————————
        （正文）

    为什么要有这一段：向量库里每块的 page_content 就是唯一被 embedding 的文本。
    只放正文的话，「这一段出自哪本书第几页」这个信息只存在于 metadata，
    检索时 BM25 拿不到、embedding 也看不到；而古籍里同一句话在不同版本里
    措辞差异极大，出处本身就是判别信号。拼进正文后，检索命中「传习录 页码」
    这类 query 也能召回。
    """
    lines = [
        f"【{_PREFIX_LABELS[0]}】{character_name}",
        f"【{_PREFIX_LABELS[1]}】{doc_title}",
        f"【{_PREFIX_LABELS[2]}】{page_num}/{page_total}",
        f"【{_PREFIX_LABELS[3]}】扫描件·多模态逐页转录",
    ]
    return "\n".join(lines) + "\n" + "—" * 24


async def transcribe_pdf(
    pdf_path: Path,
    *,
    character_name: str = "",
    on_page: Optional[Callable[[PageResult], None]] = None,
) -> list[PageResult]:
    """整本 PDF 逐页转录，返回每页结果（含失败页，便于调用方点名报告）。

    跨页扫描（一张图装两个书页，如《阳明先生文录》1116 页）在
    `vision_split_spread` 开启时**切成右半/左半分别转录**再拼接。
    理由实测：整张转录成功率 75%、单页 126s；切半后成功率接近 100%、
    每半 200-250 字能一次转完。竖排古籍右起阅读，所以右半在前。

    节流：每页之间 sleep settings.vision_page_interval_sec（默认 30s）。
    页均 9.3-126.2s 实测跨度很大（单开本 vs 跨页高清），但都在网关可持续
    区间（3-6 次/分）内，批量跑时仍需节流——429 是突发式的。
    """
    pdf_path = Path(pdf_path)
    try:
        import fitz
    except ImportError:
        print("[VisionParser] PyMuPDF 未安装，无法渲图")
        return []

    results: list[PageResult] = []
    t_start = time.time()
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"[VisionParser] 打开失败 {pdf_path.name}: {e}")
        return []

    try:
        total = len(doc)
        # 先数一遍跨页页，好提前告知调用方耗时口径
        spreads = sum(1 for i in range(total) if is_spread_page(doc[i]))
        print(f"  [VisionParser] {pdf_path.name}: {total} 页，dpi={settings.vision_dpi}，"
              f"模型={settings.vision_model}，节流 {settings.vision_page_interval_sec:.0f}s/页")
        if spreads:
            print(f"    跨页扫描 {spreads} 页"
                  + (f"，将切半转录（{2 * spreads} 次请求）"
                     if settings.vision_split_spread else "，未开切半（整张转录）"))
        for idx in range(total):
            page_num = idx + 1
            spread = is_spread_page(doc[idx])
            split = spread and settings.vision_split_spread

            # 断点续跑：已转录且非空的页直接取缓存，不发请求。
            # 跨页切半时两半各自独立读缓存（_cache_read 带 part 参数），
            # 一半命中一半没命中就只补缺的那半。
            if split:
                parts = []
                any_cached = False
                for part in ("b", "a"):        # b=右半（先读，竖排从右往左）
                    c = _cache_read(pdf_path, page_num, part)
                    if c is not None:
                        parts.append(c)
                        any_cached = True
                if any_cached:
                    r = PageResult(page_num=page_num, text="\n".join(parts),
                                   ok=True, from_cache=True, reason="cache")
                    results.append(r)
                    if on_page:
                        on_page(r)
                    continue
            else:
                cached = _cache_read(pdf_path, page_num)
                if cached is not None:
                    r = PageResult(page_num=page_num, text=cached, ok=True,
                                   from_cache=True, reason="cache")
                    results.append(r)
                    if on_page:
                        on_page(r)
                    continue

            try:
                # 逐页挑 dpi：超长页面（整页一张长图）压 dpi 才不超网关上限。
                # None = 压到最低仍超限 → 该页直接放弃，别浪费一次 400 请求。
                page_dpi = pick_dpi(doc[idx], settings.vision_dpi)
                if page_dpi is None:
                    rect = doc[idx].rect
                    longest_pt = int(max(rect.width, rect.height))
                    r = PageResult(
                        page_num=page_num, ok=False,
                        reason=(f"页面过长无法转录：{longest_pt}pt 高，"
                                f"压到 dpi={MIN_USABLE_DPI} 仍超 {MAX_IMAGE_PX} 像素"),
                    )
                    results.append(r)
                    if on_page:
                        on_page(r)
                    continue
                # 切半路径下整页 PNG 不会用到，跳过这次渲图（大图渲一次要好几秒）
                png = None if split else render_page_png(doc[idx], page_dpi)
            except Exception as e:
                r = PageResult(page_num=page_num, ok=False, reason=f"渲图失败: {e}")
                results.append(r)
                if on_page:
                    on_page(r)
                continue

            # 前 N 页按「易失败页」处理：撞 length 就直接放弃，不重试。
            #
            # 为什么不靠图像特征判封面（试过三条都走不通）：
            #   ① 彩色占比 —— 《王阳明心学口诀》封面 62.5% vs 正文 0-1.5% 判得很准，
            #      但《阳明先生文录》是档案馆实体书扫描（带色卡和标尺），
            #      **1116 页正文全是 31-57% 偏色**，用占比会整本跳过
            #   ② 色数 —— 文录正文 16843-30459 色、封面 5911-16815 色，区间重叠
            #   ③ 灰度对比度 —— 文录正文 36-108、封面 63，重叠
            # 图像特征在高质量古籍扫描上失效，而那恰恰是最该入库的部分。
            # 封面本来就只有前 1-3 页，撞了就丢，不值得为它加判据。
            is_front_matter = page_num <= FRONT_MATTER_PAGES

            # ---- 跨页切半路径 ----
            if split:
                halves = render_spread_halves(doc[idx], page_dpi)
                # b=右半（竖排从右往左读，先右后左）
                labels = ("b", "a")
                texts: list[str] = []
                bad_parts: list[str] = []
                for png_half, part in zip(halves, labels):
                    cached_half = _cache_read(pdf_path, page_num, part)
                    if cached_half is not None:
                        texts.append(cached_half)
                        continue
                    hr = await transcribe_page(
                        png_half, page_num,
                        max_retries=1 if is_front_matter else None,
                    )
                    if hr.ok:
                        cleaned = _filter_noise(hr.text)
                        if cleaned.strip():
                            _cache_write(pdf_path, page_num, cleaned, part)
                            texts.append(cleaned)
                        else:
                            bad_parts.append(f"{part}(清洗后为空)")
                    else:
                        bad_parts.append(f"{part}({hr.reason[:24]})")
                    if part != labels[-1]:
                        await asyncio.sleep(settings.vision_page_interval_sec)

                r = PageResult(
                    page_num=page_num,
                    text="\n".join(texts),
                    ok=bool(texts),
                    reason="；".join(bad_parts) if bad_parts else "",
                )
                results.append(r)
                if on_page:
                    on_page(r)
                if page_num < total:
                    await asyncio.sleep(settings.vision_page_interval_sec)
                continue

            # ---- 整页路径 ----
            r = await transcribe_page(
                png, page_num,
                max_retries=1 if is_front_matter else None,
            )
            # pick_dpi 是按 MAX_IMAGE_PX=8000 算的，但网关的真实上限可能更严。
            # 命中「分辨率超限」就逐级降 dpi 再试，这是唯一有效的解法
            # （transcribe_page 内部已判出这条失败并跳过重试）。
            tries = 0
            while (not r.ok and "分辨率" in r.reason
                   and page_dpi > DPI_FALLBACKS[-1] and tries < 3):
                tries += 1
                page_dpi = next(d for d in DPI_FALLBACKS
                                if d < page_dpi)   # 往下一级
                print(f"      [p{page_num}] 分辨率超限，降 dpi 到 {page_dpi} 重试…")
                png = render_page_png(doc[idx], page_dpi)
                r = await transcribe_page(png, page_num)

            if r.ok:
                r.text = _filter_noise(r.text)
                if r.text.strip():
                    _cache_write(pdf_path, page_num, r.text)
                else:
                    # 全是噪声行 = 这页实质没有正文，不入库也不写缓存
                    r.ok = False
                    r.reason = "转录结果清洗后为空（版权页/纯页码）"
            results.append(r)
            if on_page:
                on_page(r)

            if page_num < total:
                await asyncio.sleep(settings.vision_page_interval_sec)
    finally:
        doc.close()

    ok = sum(1 for r in results if r.ok)
    elapsed = time.time() - t_start
    print(f"  [VisionParser] {pdf_path.name} 完成: {ok}/{total} 页成功，"
          f"耗时 {elapsed / 60:.1f} 分钟")
    if ok < total:
        bad = [r for r in results if not r.ok]
        print(f"  [VisionParser] 失败 {len(bad)} 页，明细: "
              + ", ".join(f"p{r.page_num}({r.reason[:30]})" for r in bad[:8])
              + ("…" if len(bad) > 8 else ""))
    return results


def transcribe_pdf_sync(pdf_path: Path, *, character_name: str = "") -> list[PageResult]:
    """同步包装（init_persona_data.py 是同步脚本）。"""
    return asyncio.run(transcribe_pdf(pdf_path, character_name=character_name))


# ============================================================
# 出口：ParsedElement（一页一个） / Document（一页一块）
# ============================================================

def results_to_elements(
    results: list[PageResult],
    *,
    source: str,
    character_name: str = "",
    doc_title: str = "",
    page_total: int = 0,
    with_prefix: bool = True,
) -> list[ParsedElement]:
    """转录结果 → ParsedElement（一页一个）。

    字段与 parser.py 的 PDF 路径完全对齐：
      content / element_type="paragraph" / metadata{source, page_num}
    额外带两个标记位：vision=True（这条是模型转录来的，不是文本层抽取）、
    doc_title（文献名，供 pages_to_documents 组 heading）。

    with_prefix=True 时把前置元数据拼进 content —— 这样它会随 element
    进入 AdaptiveChunker，最终落在 page_content 里被 embedding。
    """
    elements: list[ParsedElement] = []
    for r in results:
        if not r.ok or not r.text.strip():
            continue
        content = r.text
        if with_prefix:
            content = build_page_prefix(
                character_name=character_name,
                doc_title=doc_title or source,
                page_num=r.page_num,
                page_total=page_total or r.page_num,
            ) + "\n" + content
        elements.append(ParsedElement(
            content=content,
            element_type="paragraph",
            metadata={
                "source": source,
                "page_num": r.page_num,
                "vision": True,
                "doc_title": doc_title or source,
            },
        ))
    return elements


def pages_to_documents(
    elements: list[ParsedElement],
    *,
    source: str,
) -> list:
    """逐页入库路径：一页一块 Document（不再跨页合并）。

    与「喂 AdaptiveChunker」的区别就是老大的第 2 条要求——
    「以页为单位做 embedding，逐页写入向量库」：
      · 喂 chunker 时 min_size=500 会把相邻的稀疏页并成一块，页码就丢了
        （chunker 只透传 source/heading/element_types/char_length 四个字段）
      · 这里每页独立成块，metadata 带 page_num / page_total，可溯源到页
    超长页（一页正文超过 chunk_size，古籍大字排版会出现）按句边界切开，
    切出的几块共用同一个 page_num —— 页码是「这一块来自第几页」，
    不因为切分而失真。

    metadata 字段集与现有流程对齐：source / heading / element_types /
    char_length 照旧，额外加 page_num / page_total / vision / doc_title。
    rel_path 与 source_type 由 init_persona_data.py 统一补（与所有来源一致）。
    """
    from langchain_core.documents import Document

    from src.retrieval.chunker import AdaptiveChunker

    max_size = settings.chunk_size
    chunker = AdaptiveChunker(
        min_size=max_size // 2,
        max_size=max_size,
        overlap=settings.chunk_overlap,
    )
    docs: list[Document] = []
    for el in elements:
        content = el.content.strip()
        if not content:
            continue
        page_num = el.metadata.get("page_num", 0)
        doc_title = el.metadata.get("doc_title", "")

        # 未超长 → 一页一块，直接产出。
        # 超长（古籍大字排版一页能到 2000+ 字）→ 把**单个元素**交给 chunker.chunk()。
        # 逐页调用而不是整本一次性调用：每次只传一个元素，chunker 看不到
        # 别的页，就绝不会把相邻页并进同一块，页码因此不失真。
        # chunker 返回的 metadata 里带 has_overlap / overlap_source（切分时才产生），
        # 透传下去以保持字段集与现有库一致。
        extra: dict = {}
        if len(content) > max_size:
            pieces_docs = chunker.chunk([el], source=source)
            pieces = [d.page_content for d in pieces_docs]
            if len(pieces_docs) > 1:
                extra = {
                    "has_overlap": True,
                    "overlap_source": doc_title or source,
                }
        else:
            pieces = [content]

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            docs.append(Document(
                page_content=piece,
                metadata={
                    "source": source,
                    # 无标题结构（多模态按页转录拿不到字号信息），heading 留空
                    # 与 OCR 路径的产出保持一致，不伪造标题链
                    "heading": "",
                    "element_types": "paragraph",
                    "char_length": len(piece),
                    "page_num": int(page_num),
                    "doc_title": doc_title,
                    "vision": True,
                    **extra,
                },
            ))
    return docs


# ============================================================
# 判定：这个 PDF 该不该走多模态
# ============================================================

def is_screenshot_pdf(pdf_path: Path) -> tuple[bool, str]:
    """判断这个 PDF 是不是「整屏/整篇截图」而不是书。

    返回 (是?, 理由)。与 `needs_vision` 分开的原因是：
    **`needs_vision` 判否不等于这个文件该入库。** 它只回答「要不要走多模态」，
    判否之后文件会掉进常规解析路径——而那里有本地 OCR，会把截图里的内容抽出来入库。

    实踩过：《王阳明为什么倡导人人心中有孔子》只有 1 页 614x4425pt，
    `needs_vision` 判它「不是扫描件」→ 掉进常规路径 → OCR 43.7s 抽出 4748 字 →
    切成 5 块入库。而那 4748 字是**喜马拉雅音频节目的语音转文字稿**
    （董平讲《王阳明》全集），不是书。入库后问「王阳明为什么倡导人人心中有孔子」
    会命中一段播客稿。

    判据用**高宽比**（全页竖版且 高/宽 >= LONG_SHOT_ASPECT）：
    竖版 A4 是 1.41、双页跨页扫描是横版 0.67，只有「整屏截图」会到 7.2。
    不用点高：横版双页扫描点高 1728pt 也会超标，按点高会误伤 1116 页最重要的原著。
    """
    pdf_path = Path(pdf_path)
    try:
        import fitz
    except ImportError:
        return False, "PyMuPDF 未安装"
    try:
        with fitz.open(str(pdf_path)) as doc:
            total = len(doc)
            if total == 0:
                return False, "空文档"
            shots = 0
            for i in range(total):
                rect = doc[i].rect
                longest = max(rect.width, rect.height) or 1.0
                shortest = min(rect.width, rect.height) or 1.0
                if rect.height > rect.width and longest / shortest >= LONG_SHOT_ASPECT:
                    shots += 1
            if shots == total:
                return True, (
                    f"全部 {total} 页都是竖版长截图（高宽比 ≥ {LONG_SHOT_ASPECT}）——"
                    f"网页/公众号整屏截图或音频节目图文稿，不是书页；"
                    f"转录会撞 max_tokens，OCR 会抽出一堆非正文内容"
                )
    except Exception as e:
        return False, f"打开失败: {e}"
    return False, "非长截图"


def needs_vision(pdf_path: Path, *, min_text_chars: int = 200) -> tuple[bool, str]:
    """判断 PDF 是否需要多模态救回，返回 (需要?, 理由)。

    判据按「有效文本层覆盖率」而不是「有没有元素」——现有解析链路只要拿到
    任意一个元素就不再降级，所以《传习录注疏》311 页里 310 页空白却仍然
    「解析成功」，只抽出 1 页版权页残文。必须自己数。

    抽样：最多看 12 页（首尾 + 均匀分布），命中任意一页有足够文本即认为
    「文字型」。12 页的判断足够把「310/311 页空白」和「正常文字型」分开，
    又不必为 1116 页的书扫全本。

    min_text_chars=200 不是拍的，是实测两类 PDF 单页有效字符数后定的
    （output/_wy_threshold.log，抽 12 页）：

        文字型单页字符数   《五百年来王阳明》中位 314 / 最低 64
                          《王阳明全集》中位 488 / 最低 254
        扫描件单页字符数   《传习录注疏》全 12 页里只有 1 页 38 字
                          《阳明学述要》非零页 40 与 205（205 那页是版权残文）
                          《传习录详注集评》全 12 页为 0

    50 会被扫描件的 205 字残文骗过、300 会被文字型的 64 字插页判死，
    **200 是唯一把两类完全分开的取值**，落在两者的空档里。
    """
    pdf_path = Path(pdf_path)
    try:
        import fitz
    except ImportError:
        return False, "PyMuPDF 未安装"

    # 整屏截图一律不走多模态（判据与 is_screenshot_pdf 共用，避免两处漂移）
    is_shot, shot_why = is_screenshot_pdf(pdf_path)
    if is_shot:
        return False, shot_why

    try:
        with fitz.open(str(pdf_path)) as doc:
            total = len(doc)
            if total == 0:
                return False, "空文档"

            sample_idxs = sorted({
                0,
                total - 1,
                *[round(i * (total - 1) / 11) for i in range(12)],
            })
            with_text = 0
            text_chars = 0
            for i in sample_idxs:
                try:
                    t = doc[i].get_text("text") or ""
                except Exception:
                    t = ""
                if len(re.sub(r"\s+", "", t)) >= min_text_chars:
                    with_text += 1
                    text_chars += len(t)
    except Exception as e:
        return False, f"打开失败: {e}"

    ratio = with_text / len(sample_idxs)
    if ratio < 0.5:
        return True, (f"抽样 {len(sample_idxs)} 页仅 {with_text} 页有有效文本层"
                      f"（{ratio:.0%}），正文需多模态转录")
    return False, f"抽样 {len(sample_idxs)} 页中 {with_text} 页有文本层（{ratio:.0%}），走常规解析"
