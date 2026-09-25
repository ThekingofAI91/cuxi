"""
初始化名人对话场景的数据

把 data/persona_chat/<角色>/ 下的语料解析、分块、向量化后存入 ChromaDB。

用法：
    python init_persona_data.py --dry-run                 # 只统计文件与块数，不向量化、不写库
    python init_persona_data.py --character adler         # 只重建指定角色
    python init_persona_data.py                           # 重建全部角色（含 OCR，慢）
    python init_persona_data.py --no-ocr                  # 跳过扫描件 OCR，只为快速迭代
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
"""

import argparse
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import settings
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
        - excluded: 被 EXCLUDE_PATTERNS 挡掉的文件 [(相对路径, 理由), ...]
    """
    candidates: list[Path] = []
    skipped: Counter = Counter()
    excluded: list[tuple[str, str]] = []

    for f in sorted(data_dir.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        ext = f.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped[ext] += 1
            continue
        hit = next((reason for pat, reason in EXCLUDE_PATTERNS if pat in f.name), None)
        if hit:
            excluded.append((str(f.relative_to(data_dir)), hit))
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


def build_chunks(character, files: list[Path], allow_ocr: bool = False) -> tuple[list, dict, list[tuple[str, int]]]:
    """把语料文件解析分块。返回 (chunks, 统计, [(需要 OCR 的文件, 页数)])"""
    parser = DocumentParser()
    chunker = AdaptiveChunker(
        min_size=settings.chunk_size // 2,
        max_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    data_dir = Path(character.data_source)

    all_chunks: list = []
    needs_ocr: list[tuple[str, int]] = []
    stats = {"ok": 0, "failed": 0, "elements": 0, "secs": 0.0}

    for file_path in files:
        rel = _rel(file_path, data_dir)
        try:
            t0 = time.time()
            elements = parser.parse(file_path, allow_ocr=allow_ocr)

            # 取不到文本层时 parser 会返回一个占位元素。这种块入库只会污染检索，
            # 单独记录、不入库——要么开 OCR，要么走逐页 OCR（见待办）。
            if len(elements) == 1 and elements[0].metadata.get("warning") == "no_text_extracted":
                stats["secs"] += time.time() - t0
                pages = _pdf_page_count(file_path)
                needs_ocr.append((rel, pages))
                print(f"    {rel[:70]:<72} 无文本层，需 OCR（{pages} 页）")
                continue

            chunks = chunker.chunk(elements, source=file_path.name)
            # 语料来源类型打标（original/oral/secondary/artificial/anchor）：
            # 检索阶段按类型加权（口述体优先、二手解读降权），提前写入元数据。
            # 注意：classify_source 是按**文件名**子串匹配的，这里只传 name；
            # 相对路径另存 rel_path，仅作溯源，不参与分类。
            for c in chunks:
                c.metadata.setdefault("source_type", classify_source(c.metadata.get("source", "")))
                c.metadata["rel_path"] = rel
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


def load_character_data(character_id: str, dry_run: bool = False, allow_ocr: bool = False) -> None:
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
    all_chunks, stats, needs_ocr = build_chunks(character, files, allow_ocr=allow_ocr)
    print(f"  完成: 成功 {stats['ok']} / 失败 {stats['failed']}，"
          f"共 {len(all_chunks)} 块，解析耗时 {stats['secs']:.1f}s")
    if needs_ocr:
        pages = sum(p for _, p in needs_ocr)
        est_min = pages * 3 / 60          # 实测 CPU 上约 3 秒/页
        print(f"  需要 OCR 的文件 {len(needs_ocr)} 个（合计 {pages} 页，"
              f"估计约 {est_min:.0f} 分钟）:")
        for rel, p in needs_ocr:
            print(f"    {p:>5d} 页  {rel}")
        if not allow_ocr:
            print(f"  ↑ 本次是 --no-ocr，这些文件**没有入库**")

    if not all_chunks:
        print("  没有生成任何文档块，跳过（不会删除已有 collection）")
        return

    type_dist = Counter(c.metadata.get("source_type", "unknown") for c in all_chunks)
    print(f"  来源类型分布: {dict(type_dist.most_common())}")

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
    ap.add_argument("--retag-source-type", action="store_true",
                    help="只按现有 source 重算 source_type，不重算向量")
    args = ap.parse_args()

    targets = args.character or list(persona_chat_config.characters.keys())

    print("名人对话场景 — 数据初始化")
    print(f"  内置 + 自建角色: {list(persona_chat_config.characters.keys())}")
    print(f"  本次处理: {targets}")
    print(f"  分块参数: min_size={settings.chunk_size // 2} "
          f"max_size={settings.chunk_size} overlap={settings.chunk_overlap}")

    if args.retag_source_type:
        for cid in targets:
            retag_source_type(cid)
        print(f"\n{'=' * 72}")
        print("source_type 重标完成（未动向量）")
        return

    if not args.dry_run:
        print("  注意：会先删除同名 collection 再重建，请确认服务已停止"
              "（ChromaDB 不支持多进程同时写同一持久化目录）")

    for character_id in targets:
        load_character_data(character_id, dry_run=args.dry_run, allow_ocr=not args.no_ocr)

    print(f"\n{'=' * 72}")
    print("dry-run 统计完成!" if args.dry_run else "所有角色数据初始化完成!")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
