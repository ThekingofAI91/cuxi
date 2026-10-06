"""
初始化名人对话场景的数据

把 data/persona_chat/<角色>/ 下的语料解析、分块、向量化后存入 ChromaDB。

用法：
    python init_persona_data.py --dry-run                 # 只统计文件与块数，不向量化、不写库
    python init_persona_data.py --character adler         # 只重建指定角色
    python init_persona_data.py                           # 重建全部角色（含 OCR，慢）
    python init_persona_data.py --no-ocr                  # 跳过扫描件 OCR，只为快速迭代
    python init_persona_data.py --character wangyangming --no-ocr --vision
                                                       # 薄 PDF 走多模态逐页转录（替代本地 OCR）
    python init_persona_data.py --retag-source-type       # 只重算 source_type 元数据（不重算向量）

四条 2026-09-22 定下的规矩（都是踩过才加的）：
1. **递归扫子目录**。此前只扫 data_source 的直接子文件，王阳明那 95 本
   藏在 电子书（54本）/ 下的书一本都没进库。
2. **只吃 .pdf / .md / .txt**。不再对未知扩展名做"当文本读"的兜底猜测——
   .epub/.mobi 是二进制，按 UTF-8 硬读会产出乱码块污染检索。跳过清单会在
   结尾按扩展名打印，不静默丢。
3. **按内容哈希去重**。递归扫描会撞上同一本书既在根目录又在子目录里的情况
   （实测王阳明有 4 组），不去重会双份入库、重复命中。保留路径最浅的那份。
4. **OCR 默认开启**（`--no-ocr` 可跳过），且需要 OCR 的文件会单独列出。
   实测语料里有整本扫描件：荣格《人类与象征》339 页、陈荣捷《传习录详注集评》
   466 页，CPU 上分别约 17 / 25 分钟；线上库里已经有《人类与象征》的 178 块，
   说明历史入库是开着 OCR 的，默认关掉会把它弄丢——所以默认必须是开。
    《传习录注疏》311 页里 310 页无文本层、只剩 1 页有字，这类"混合型"PDF
    走不到 OCR（有元素就不触发），目前**无法自动救回**，只能在报告里点名。
    2026-10-02 起这类文件多了第四条出路：`--vision` 把无有效文本层的 PDF
    逐页渲成图交多模态模型转录（见 src/document_processing/vision_parser.py），
    代价是页均 12.8-48.7s 的网关占用，故做成显式开关而非默认路径。
"""

import argparse
import hashlib
import os
import sys
import time
from collections import Counter
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import settings
from src.corpus_dedup import KEEP_ONE_PER_BOOK, dedupe_cross_format
from src.document_processing.parser import DocumentParser
from src.retrieval.chunker import AdaptiveChunker
from src.retrieval.embedder import get_embedder
from src.retrieval.source_profile import classify_source
from scenes.persona_chat.config import persona_chat_config

# 能真正解析出干净文本的扩展名。其余一律不猜、只报告。
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

# 不收的文件（按文件名子串匹配），每条都要写清理由——静默排除是事故的温床。
EXCLUDE_PATTERNS: list[tuple[str, str]] = [
    ("中国古代大人物系列",
     "读客 5 合 1：一本里同时有王阳明/曾国藩/刘伯温/成吉思汗/张居正，"
     "整本入王阳明的库会把另外四人的内容混进去（实测这一本就有 2123 块）"),
]

# 双重乱码副本：**只挡根目录那一份**，不能按文件名挡。
#
# 为什么必须走单独一张表：子目录「其它单本书（10本）」下有两份**同名**的正常 GBK 版
# （《王阳明最神奇的心学》149KB、发现心灵的智慧 111KB），它们是要入库的。
# 而根目录那两份同名的（123KB / 255KB）是双重乱码——UTF-8 字节被当 GBK 读过一次、
# 又以 UTF-8 写回磁盘。它们能正常解码、不触发 no_text_extracted，
# 只看「解析成功与否」发现不了，逆转也不可逆（encode('gb18030')→decode('utf-8')
# 中途抛 invalid continuation；errors='ignore' 会吃字，实测产出「智慄17」）。
# 两份之间实测双向字符覆盖率 25.3% / 18.4%：它们其实是同一本书的两个标题。
#
# 匹配方式：**相对 data_dir 的路径**恰好等于表中值（不含目录名），
# 而 EXCLUDE_PATTERNS 是文件名子串匹配。两者混用会误杀正常版——踩过一次。
MOJIBAKE_COPIES: dict[str, str] = {
    "《王阳明最神奇的心学》.txt":
        "根目录的乱码副本（123k 字）。正常 GBK 版在「电子书（54本）/其它单本书（10本）」下",
    "发现心灵的智慧——王阳明人生哲学感悟.txt":
        "根目录的乱码副本（255KB）。正常 GBK 版在「电子书（54本）/其它单本书（10本）」下",
}

# 乱码探测：正文里 mojibake 特征字占比超过该值就整份丢弃。
# 为什么需要它：双重乱码的文件**能**被正常解码（utf-8 不报错），所以既不会被
# 扩展名白名单挡掉、也不会触发 no_text_extracted，只看「解析成功与否」发现不了。
# 特征字取自实测：这些字在正常中文文本里几乎不出现，出现在开头 3000 字里即断定。
MOJIBAKE_MARKS = "锛嬨晲鐨勬櫤閲戝殑鑷绔鑻鏄庣瓧绗鍙戠幇"
MOJIBAKE_RATIO_THRESHOLD = 0.02   # 前 3000 字里特征字占比 ≥ 2% 即判定乱码
MOJIBAKE_SNIFF_CHARS = 3000

EMBED_BATCH = 256    # 向量化 + 入库的批大小（控内存、给进度）


# ============================================================
# 语料收集
# ============================================================

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_corpus_files(data_dir: Path) -> tuple[list[Path], Counter, list[tuple[str, list[str]]], list[tuple[str, str]]]:
    """递归收集语料文件。

    Returns:
        (files, skipped_by_ext, dup_groups, excluded)
        - skipped_by_ext: 被扩展名白名单挡掉的文件计数
        - dup_groups: 内容重复的组 [(哈希前12位, [路径...]), ...]
        - excluded: 被 EXCLUDE_PATTERNS / KEEP_ONE_PER_BOOK / 跨格式判据挡掉的
          文件 [(相对路径, 理由), ...]
    """
    candidates: list[Path] = []
    skipped: Counter = Counter()
    excluded: list[tuple[str, str]] = []

    # 两层显式排除：先「五合一时序」（会混进另外四个人物的内容），
    # 再「同一本书只留一份」（分册扫描件已有合订本文字版）。
    # 后者有显式证据（PDF 元数据 author + 页数 + 文本层覆盖率），不需要猜。
    exclude_rules: list[tuple[str, str]] = (
        list(EXCLUDE_PATTERNS) + list(KEEP_ONE_PER_BOOK)
    )

    for f in sorted(data_dir.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        ext = f.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped[ext] += 1
            continue
        rel = str(f.relative_to(data_dir))
        # 乱码副本只按「相对路径恰好等于文件名」匹配（即根目录那份），
        # 不能按文件名子串——子目录有同名正常版会被误杀。
        if rel in MOJIBAKE_COPIES:
            excluded.append((rel, f"双重乱码副本：{MOJIBAKE_COPIES[rel]}"))
            continue
        hit = next((reason for pat, reason in exclude_rules if pat in f.name), None)
        if hit:
            excluded.append((rel, hit))
            continue
        candidates.append(f)

    # 同尺寸才去算哈希，省 IO
    by_size: dict[int, list[Path]] = {}
    for f in candidates:
        by_size.setdefault(f.stat().st_size, []).append(f)

    kept: list[Path] = []
    dup_groups: list[tuple[str, list[str]]] = []

    for size, group in by_size.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        by_hash: dict[str, list[Path]] = {}
        for f in group:
            by_hash.setdefault(_sha256(f), []).append(f)
        for h, same in by_hash.items():
            if len(same) == 1:
                kept.append(same[0])
                continue
            # 路径最浅的优先（根目录那份是人工挑过的），其次字典序
            same.sort(key=lambda p: (len(p.relative_to(data_dir).parts), str(p)))
            kept.append(same[0])
            dup_groups.append((h[:12], [str(p.relative_to(data_dir)) for p in same]))

    # sha256 之后仍可能剩「同书多载体」：同名的一组里 pdf 是扫描件、
    # txt 是完整文字版。sha256 挡不住这种（字节本来就不同），所以单独一层。
    kept, fmt_dropped = dedupe_cross_format(kept)
    for path_str, reason in fmt_dropped:
        excluded.append((path_str, reason))

    kept.sort(key=lambda p: str(p.relative_to(data_dir)))
    return kept, skipped, dup_groups, excluded


# ============================================================
# 解析 + 分块
# ============================================================

def _rel(path: Path, base: Path) -> str:
    """相对路径；越出 base（如父目录的人物档案 md）时退回文件名"""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


def _pdf_page_count(path: Path) -> int:
    """PDF 页数（估 OCR 用时用）。拿不到就返回 0。"""
    if path.suffix.lower() != ".pdf":
        return 0
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            return len(doc)
    except Exception:
        return 0


def _tag_chunks(chunks: list, rel: str) -> None:
    """给一批块补上 source_type 与 rel_path（所有来源统一走这里）"""
    for c in chunks:
        # 语料来源类型打标（original/oral/secondary/artificial/anchor）：
        # 检索阶段按类型加权（口述体优先、二手解读降权），提前写入元数据。
        # 注意：classify_source 是按**文件名**子串匹配的，这里只传 name；
        # 相对路径另存 rel_path，仅作溯源，不参与分类。
        c.metadata.setdefault("source_type", classify_source(c.metadata.get("source", "")))
        c.metadata["rel_path"] = rel


def build_vision_chunks(
    character,
    files: list[Path],
    vision_targets: list[tuple[Path, str]],
) -> tuple[list, dict]:
    """薄 PDF 走多模态逐页转录，其余文件仍走常规解析。

    分工（老大的第 5 问，1A 与 3A 的处理差异）：

    | 来源类型 | 走哪条路 | 原因 |
    |---|---|---|
    | `.md`（3A 新建的 wangyangming_core_ideas.md） | MarkdownParser | 纯文本，有标题层级，chunker 能建标题链；且 classify_source 认 `*_core_ideas` → anchor（权重 2.0） |
    | `.txt`（如《传习录 一本书读懂阳明心学.txt》） | 文本路径 | 同上，GBK/UTF-8 多编码兼容已有 |
    | 文字型 `.pdf` | PDFParser（PyMuPDF 字号感知） | 有文本层，拿到标题结构比多模稿准确得多 |
    | **薄 PDF（无有效文本层）** | **vision_parser 逐页转录** | 文本层里只剩版权残文/广告，常规路径会灌 1~4 块垃圾 |

    所以 --vision 只改变**薄 PDF 那一类**，其余路径字节级不变。

    逐页入库（老大的第 2 问）：一页一块 Document，不跨页合并，
    page_num 不会被 chunker 的 min_size 合并冲掉。
    """
    import asyncio

    from src.document_processing.vision_parser import (
        pages_to_documents,
        results_to_elements,
        transcribe_pdf,
    )

    data_dir = Path(character.data_source)
    vision_paths = {p for p, _ in vision_targets}

    all_chunks: list = []
    stats = {"vision_files": 0, "vision_pages_ok": 0, "vision_pages_fail": 0,
             "secs": 0.0, "elements": 0, "ok": 0, "failed": 0}

    for file_path in files:
        if file_path not in vision_paths:
            # 非薄 PDF 不在本函数里处理——已由 build_chunks 的常规循环负责，
            # 在这里再跑一遍会把 stats["ok"] 双记，报告出来的文件数会对不上。
            continue

        rel = _rel(file_path, data_dir)
        t0 = time.time()

        # ---- 薄 PDF：逐页渲图 → 多模态转录 ----
        doc_title = file_path.stem
        try:
            results = asyncio.run(transcribe_pdf(
                file_path,
                character_name=character.name,
            ))
        except KeyboardInterrupt:
            # 断点续跑：已转录的页都写在缓存里，中断后重跑会跳过
            print(f"    {rel[:70]:<72} 被中断，已转录页已缓存，重跑可续")
            raise
        except Exception as e:
            stats["failed"] += 1
            print(f"    {rel[:70]:<72} 多模态转录失败: {e}")
            continue

        ok_pages = [r for r in results if r.ok and r.text.strip()]
        fail_pages = [r for r in results if not r.ok]
        elements = results_to_elements(
            results,
            source=file_path.name,
            character_name=character.name,
            doc_title=doc_title,
            page_total=len(results),
        )
        chunks = pages_to_documents(elements, source=file_path.name)
        _tag_chunks(chunks, rel)

        # 同样闻一遍乱码：模型理论上只会吐简体正常字，但网关返回异常内容时
        # 不拦住就等于把 745 页的垃圾全灌进库里
        v_moji = _sniff_mojibake(elements)
        if v_moji >= MOJIBAKE_RATIO_THRESHOLD:
            print(f"    {rel[:70]:<72} 转录结果疑似乱码（{v_moji:.1%}），丢弃不入库")
            stats["mojibake"] = stats.get("mojibake", 0) + 1
            continue

        stats["vision_files"] += 1
        stats["vision_pages_ok"] += len(ok_pages)
        stats["vision_pages_fail"] += len(fail_pages)
        stats["elements"] += len(elements)
        stats["secs"] += time.time() - t0
        all_chunks.extend(chunks)
        print(f"    {rel[:70]:<72} 多模态 {len(ok_pages)}/{len(results)} 页 -> "
              f"{len(chunks):5d} 块  {time.time() - t0:5.1f}s"
              + (f"（失败 {len(fail_pages)} 页）" if fail_pages else ""))

    return all_chunks, stats


def _sniff_mojibake(elements: list) -> float:
    """闻出双重乱码：返回前若干字里 mojibake 特征字的占比，正常文本返回 0.0。

    为什么需要单独一道：双重乱码（UTF-8 字节被当 GBK 读过一次、又以 UTF-8
    写回磁盘）的文件**能**被正常解码——parser 不会报错、也不触发
    no_text_extracted，只看「解析成功与否」发现不了。而一旦入库，
    12 万字的乱码块会挤占 head 5 里的位置，把真正相关的 chunk 顶出去。
    代价极低：只看前 3000 字，纯 Python 字符统计，微秒级。
    """
    head = "".join(e.content for e in elements)[:MOJIBAKE_SNIFF_CHARS]
    if not head:
        return 0.0
    hits = sum(head.count(c) for c in MOJIBAKE_MARKS)
    return hits / len(head)


def build_chunks(character, files: list[Path], allow_ocr: bool = False,
                 use_vision: bool = False,
                 vision_only: bool = False) -> tuple[list, dict, list[tuple[str, int]]]:
    """把语料文件解析分块。返回 (chunks, 统计, [(需要 OCR 的文件, 页数)])

    vision_only=True 时只跑多模态转录（落缓存），跳过常规解析与后续入库。
    为什么需要这个开关：全量入库要 33 小时，而 `delete_collection` 是在
    **全部解析完成之后**才执行的（第 519 行），这意味着
      · 中断在转录阶段 → 旧库还活着，安全，重跑靠缓存续
      · 中断在「删库 → 向量入库」那几分钟窗口 → 只剩半截，库废了
    用 --vision-only 先把 33 小时的转录单独跑完（纯网络 IO，随时可断可续），
    再跑一次不带 --vision-only 的完整入库——那时转录全部命中缓存，
    整个进程几分钟就过完，删库窗口从几小时压到几分钟。
    """
    parser = DocumentParser()
    chunker = AdaptiveChunker(
        min_size=settings.chunk_size // 2,
        max_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    data_dir = Path(character.data_source)

    # --vision：先把无有效文本层的 PDF 挑出来，它们改走多模态逐页转录
    vision_targets: list[tuple[Path, str]] = []
    if use_vision or vision_only:
        from src.document_processing.vision_parser import needs_vision

        for f in files:
            if f.suffix.lower() != ".pdf":
                continue
            need, why = needs_vision(f)
            if need:
                vision_targets.append((f, why))
        if vision_targets:
            total_pages = 0
            print(f"  多模态转录目标 {len(vision_targets)} 个:")
            for f, why in vision_targets:
                pages = _pdf_page_count(f)
                total_pages += pages
                print(f"    {f.name[:60]:<62} {pages:>5} 页  {why}")
            # 页均按 40 秒估（实测 13-71s，取中位再留余量），含页间 30s 节流
            est_h = total_pages * 70 / 3600
            print(f"    合计 {total_pages} 页；已转录的页会命中缓存直接跳过")
            print(f"    按页均最坏 70s 估，转录完约需 {est_h:.1f} 小时"
                  f"（可随时 Ctrl-C，重跑自动续）")
        else:
            print("  多模态转录: 未发现无有效文本层的 PDF")

    all_chunks: list = []
    needs_ocr: list[tuple[str, int]] = []
    stats = {"ok": 0, "failed": 0, "elements": 0, "secs": 0.0}

    rest = list(files)
    if vision_targets:
        v_chunks, v_stats = build_vision_chunks(character, files, vision_targets)
        all_chunks.extend(v_chunks)
        stats["vision"] = v_stats
        # ok 单独记：多模态目标文件解析成功与否已由 v_stats["vision_files"] 表达，
        # 再累加到 ok 会让「成功 N 个文件」的口径与实际文件数对不上
        stats["failed"] += v_stats["failed"]
        stats["secs"] += v_stats["secs"]
        rest = [f for f in files if f not in {p for p, _ in vision_targets}]

    if vision_only:
        # 只转录不入库：把转录结果留在缓存里就返回，常规解析也不做（那是纯 CPU 时间）
        v = stats.get("vision")
        if v:
            print(f"\n  [vision-only] 转录完成并已落缓存："
                  f"{v['vision_pages_ok']} 页成功 / {v['vision_pages_fail']} 页失败，"
                  f"覆盖 {v['vision_files']} 本")
        else:
            print("\n  [vision-only] 没有需要转录的目标")
        print(f"  [vision-only] 已跳过常规解析与入库（未触碰 collection）")
        return all_chunks, stats, needs_ocr

    for file_path in rest:
        rel = _rel(file_path, data_dir)
        try:
            t0 = time.time()
            elements = parser.parse(file_path, allow_ocr=allow_ocr)

            # 取不到文本层时 parser 会返回一个占位元素。这种块入库只会污染检索，
            # 单独记录、不入库。要救它有两条路：--ocr（本地 RapidOCR，约 3 秒/页）
            # 或 --vision（多模态逐页转录，页均 12.8-48.7s 但对竖排古籍更准）。
            if len(elements) == 1 and elements[0].metadata.get("warning") == "no_text_extracted":
                stats["secs"] += time.time() - t0
                pages = _pdf_page_count(file_path)
                needs_ocr.append((rel, pages))
                print(f"    {rel[:70]:<72} 无文本层，需 OCR 或 --vision（{pages} 页）")
                continue

            # 乱码探测：双重乱码的文件能正常解码、也能"解析成功"，
            # 上面那道 no_text_extracted 判据抓不到它，只能自己闻。
            mojibake = _sniff_mojibake(elements)
            if mojibake:
                stats["secs"] += time.time() - t0
                stats["mojibake"] = stats.get("mojibake", 0) + 1
                print(f"    {rel[:70]:<72} 疑似乱码（mojibake 特征字 {mojibake:.1%}），丢弃不入库")
                continue

            chunks = chunker.chunk(elements, source=file_path.name)
            _tag_chunks(chunks, rel)
            dt = time.time() - t0
            stats["ok"] += 1
            stats["elements"] += len(elements)
            stats["secs"] += dt
            print(f"    {rel[:70]:<72} {len(elements):5d} 元素 -> {len(chunks):5d} 块  {dt:5.1f}s")
            all_chunks.extend(chunks)
        except Exception as e:
            stats["failed"] += 1
            print(f"    {rel[:70]:<72} 处理失败: {e}")

    return all_chunks, stats, needs_ocr


# ============================================================
# 入库
# ============================================================

def _client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def load_character_data(character_id: str, dry_run: bool = False, allow_ocr: bool = False,
                        use_vision: bool = False, vision_only: bool = False) -> None:
    """加载指定角色的数据到 ChromaDB"""

    character = persona_chat_config.characters.get(character_id)
    if not character:
        print(f"角色 '{character_id}' 不存在")
        return

    data_dir = Path(character.data_source)
    collection_name = character.chroma_collection

    print(f"\n{'=' * 72}")
    print(f"角色: {character.name}  |  collection: {collection_name}")
    print(f"数据目录: {data_dir}")
    print(f"{'=' * 72}")

    if not data_dir.exists():
        # 内置角色目录缺失时明确报错；自建角色可能没有目录，跳过即可。
        print(f"  数据目录不存在，跳过（不会删除已有 collection）: {data_dir}")
        return

    files, skipped, dup_groups, excluded = collect_corpus_files(data_dir)
    # 父目录的人物档案 md（core_ideas）仍要带上——它是 anchor 类型，权重最高
    parent_dir = data_dir.parent
    if parent_dir != data_dir and parent_dir.exists():
        for f in sorted(parent_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".md" and not f.name.startswith(".") \
                    and f not in files:
                files.append(f)

    if not files:
        print(f"  没找到可解析的语料文件，跳过（不会删除已有 collection）")
        return

    print(f"  可解析文件 {len(files)} 个（递归扫描）")
    if skipped:
        detail = ", ".join(f"{ext or '(无扩展名)'} x{n}" for ext, n in skipped.most_common())
        print(f"  按扩展名跳过 {sum(skipped.values())} 个: {detail}")
        print(f"  （白名单 {sorted(SUPPORTED_EXTENSIONS)}；epub/mobi 需另做解析器，目前不猜）")
    if excluded:
        print(f"  按排除规则挡掉 {len(excluded)} 个:")
        for path, reason in excluded:
            print(f"    跳过 {path}")
            print(f"         理由: {reason}")
    if dup_groups:
        print(f"  内容重复已去重 {len(dup_groups)} 组:")
        for h, paths in dup_groups[:10]:
            print(f"    [{h}] 保留 {paths[0]}")
            for p in paths[1:]:
                print(f"           跳过 {p}")

    print(f"\n  解析 + 分块中...")
    all_chunks, stats, needs_ocr = build_chunks(
        character, files, allow_ocr=allow_ocr, use_vision=use_vision,
        vision_only=vision_only)
    v_stats = stats.get("vision")
    print(f"  完成: 常规成功 {stats['ok']} / 失败 {stats['failed']}"
          + (f" / 多模态成功 {v_stats['vision_files']} 本" if v_stats else "")
          + f"，共 {len(all_chunks)} 块，解析耗时 {stats['secs']:.1f}s")
    if v_stats:
        print(f"  多模态转录: {v_stats['vision_pages_ok']} 页成功 / "
              f"{v_stats['vision_pages_fail']} 页失败")
    if stats.get("mojibake"):
        print(f"  乱码丢弃 {stats['mojibake']} 个文件（双重乱码，能解码但内容全是"
              f" mojibake，入库会挤占 head 位置）")
    if needs_ocr:
        pages = sum(p for _, p in needs_ocr)
        est_min = pages * 3 / 60          # 实测 CPU 上约 3 秒/页
        print(f"  需要 OCR 的文件 {len(needs_ocr)} 个（合计 {pages} 页，"
              f"估计约 {est_min:.0f} 分钟）:")
        for rel, p in needs_ocr:
            print(f"    {p:>5d} 页  {rel}")
        if not allow_ocr:
            print(f"  ↑ 本次是 --no-ocr，这些文件**没有入库**")

    if vision_only:
        # 转录结果已落缓存，本次到此为止——绝不能往下走 delete_collection
        print(f"\n  [vision-only] 全流程结束，未删除也未写入 collection。"
              f"转录结果在 {settings.vision_cache_dir}")
        print(f"  [vision-only] 下一步：确认转录无缺失后，跑一次不带 --vision-only 的"
              f"完整入库（转录会全部命中缓存，几分钟完成）")
        return

    if not all_chunks:
        print("  没有生成任何文档块，跳过（不会删除已有 collection）")
        return

    type_dist = Counter(c.metadata.get("source_type", "unknown") for c in all_chunks)
    print(f"  来源类型分布: {dict(type_dist.most_common())}")
    # 全落 unknown 不是 bug，是 source_profile 的规则表里没有这个人物的条目 ——
    # 那时 RRF 融合会按 1.0 等权累加，本人原著与二手解读同权竞争。
    if type_dist.get("unknown", 0) == len(all_chunks):
        print(f"  ↑ 全部落 unknown：source_profile._SOURCE_TYPE_RULES 里没有该人物的规则，"
              f"检索时按 1.0 等权累加。补规则后跑 --retag-source-type 可免重算向量。")

    if dry_run:
        # 估一下向量化时间：bge-small 在 CPU 上约 3~6ms/块
        print(f"\n  [dry-run] 未向量化、未写库。估算向量化耗时约 "
              f"{len(all_chunks) * 0.005:.0f}~{len(all_chunks) * 0.005 * 2:.0f}s")
        return

    embedder = get_embedder()
    client = _client()

    # 先清旧数据，避免新旧混在一起（id 复用）。失败就此打住，不要带上半截状态。
    try:
        client.delete_collection(name=collection_name)
        print(f"\n  已清除旧 collection: {collection_name}")
    except Exception:
        print(f"\n  无旧 collection，继续...")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"  向量化 + 入库（{len(all_chunks)} 块，批大小 {EMBED_BATCH}）...")
    t0 = time.time()
    done = 0
    for start in range(0, len(all_chunks), EMBED_BATCH):
        end = min(start + EMBED_BATCH, len(all_chunks))
        batch = all_chunks[start:end]
        vectors = embedder.embed_documents([c.page_content for c in batch])
        collection.add(
            ids=[f"{character_id}_{j}" for j in range(start, end)],
            documents=[c.page_content for c in batch],
            embeddings=vectors,
            metadatas=[c.metadata for c in batch],
        )
        done = end
        pct = done / len(all_chunks) * 100
        print(f"    {done:6d}/{len(all_chunks)} ({pct:5.1f}%)  "
              f"累计 {time.time() - t0:6.1f}s", end="\r" if pct < 100 else "\n")
    print(f"  入库完成: {collection.count()} 块，耗时 {time.time() - t0:.1f}s")


# ============================================================
# 免重算向量的 source_type 重标
# ============================================================

def retag_source_type(character_id: str) -> None:
    """按已存的 source 元数据重算 source_type 并更新，不重算向量。

    用途：`source_profile.py` 的规则表是随语料演进的，新增规则后不该为了
    改一个元数据字段把整个 collection 重新嵌入一遍。
    """
    character = persona_chat_config.characters.get(character_id)
    if not character:
        print(f"角色 '{character_id}' 不存在")
        return

    collection_name = character.chroma_collection
    client = _client()
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"  取不到 collection {collection_name}: {e}")
        return

    total = collection.count()
    print(f"\n[{character.name}] {collection_name}: {total} 块，重算 source_type...")

    changed = 0
    dist_before: Counter = Counter()
    dist_after: Counter = Counter()
    offset = 0
    while offset < total:
        got = collection.get(include=["metadatas"], limit=1000, offset=offset)
        ids, metas = got["ids"], got["metadatas"]
        if not ids:
            break
        new_metas = []
        batch_changed = False
        for m in metas:
            old = m.get("source_type", "unknown")
            new = classify_source(m.get("source", ""))
            dist_before[old] += 1
            dist_after[new] += 1
            if new != old:
                changed += 1
                batch_changed = True
            m["source_type"] = new
            new_metas.append(m)
        if batch_changed:
            collection.update(ids=ids, metadatas=new_metas)
        offset += len(ids)

    print(f"  已更新 {changed} 块的 source_type（共 {total} 块）")
    print(f"  前: {dict(dist_before.most_common())}")
    print(f"  后: {dict(dist_after.most_common())}")


# ============================================================
# 入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="名人对话场景 — 数据初始化")
    ap.add_argument("--character", action="append", default=None,
                    help="只处理指定角色，可重复（默认全部）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计文件与块数，不向量化、不写库")
    ap.add_argument("--no-ocr", action="store_true",
                    help="跳过对无文本层 PDF 的 OCR。默认**开启** OCR：线上库里已经有 "
                         "OCR 来的内容（荣格《人类与象征》178 块就是 OCR 产物），"
                         "默认关掉会把它弄丢。OCR 在 CPU 上约 3 秒/页，很慢，"
                         "只想快速迭代时才加这个开关")
    ap.add_argument("--vision", action="store_true",
                    help="让无有效文本层的薄 PDF 走多模态逐页转录（渲图→"
                         "sensenova-6.8-flash-lite→逐页一块入库），而不是丢弃。"
                         "与 --no-ocr 搭配使用。代价：页均 13-71s 的网关占用"
                         "（3000 页约 30-40 小时），但对竖排古籍比本地 OCR 准，"
                         "且能靠提示词直接输出简体。已转录页落在 "
                         f"{settings.vision_cache_dir}，中断重跑会自动续")
    ap.add_argument("--vision-only", action="store_true",
                    help="只做多模态转录，不解析不入库。用于把 33 小时的转录"
                         "与「删库→向量入库」这几分钟的危险窗口分开："
                         "先跑这个（纯网络 IO，随时可断可断可续），"
                         "转录齐了再跑一次完整入库，那时全部命中缓存。"
                         "本开关下绝不触碰 collection")
    ap.add_argument("--retag-source-type", action="store_true",
                    help="只按现有 source 重算 source_type，不重算向量")
    args = ap.parse_args()

    targets = args.character or list(persona_chat_config.characters.keys())

    print("名人对话场景 — 数据初始化")
    print(f"  内置 + 自建角色: {list(persona_chat_config.characters.keys())}")
    print(f"  本次处理: {targets}")
    print(f"  分块参数: min_size={settings.chunk_size // 2} "
          f"max_size={settings.chunk_size} overlap={settings.chunk_overlap}")
    if args.vision or args.vision_only:
        print(f"  多模态转录: 开（模型={settings.vision_model} dpi={settings.vision_dpi} "
              f"节流={settings.vision_page_interval_sec:.0f}s/页 "
              f"重试={settings.vision_max_retries} 缓存={settings.vision_cache_dir}）")

    if args.retag_source_type:
        for cid in targets:
            retag_source_type(cid)
        print(f"\n{'=' * 72}")
        print("source_type 重标完成（未动向量）")
        return

    if not args.dry_run and not args.vision_only:
        print("  注意：会先删除同名 collection 再重建，请确认服务已停止"
              "（ChromaDB 不支持多进程同时写同一持久化目录）")
    elif args.vision_only:
        print("  本次是 --vision-only：不会删除也不会写入任何 collection，可放心跑")

    if (args.vision or args.vision_only) and not os.environ.get("LLM_API_KEY") \
            and not settings.llm_api_key:
        print("\n  提示：多模态走同一个网关，需要已配置 LLM_API_KEY，"
              "未配置会在转录阶段逐页失败")

    for character_id in targets:
        load_character_data(character_id, dry_run=args.dry_run,
                            allow_ocr=not args.no_ocr, use_vision=args.vision,
                            vision_only=args.vision_only)

    print(f"\n{'=' * 72}")
    if args.dry_run:
        print("dry-run 统计完成!")
    elif args.vision_only:
        print("多模态转录完成（未入库）！下一步跑不带 --vision-only 的完整入库")
    else:
        print("所有角色数据初始化完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
