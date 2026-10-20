"""块尾续接：块尾断在句中时，把下一块的第一句接上。

注入是整块、不裁剪的，所以模型读到的最后一句就是块尾。分块器按页装箱
（parser 一页 = 一个段落元素），而书翻页常断在句子中间，于是有大量块以
「…我们恐」这种半句收尾。模型不会停下来问，它会顺着补。

「下一块的第一句」离线算好写进 metadata 的 `cont_tail`（见
`scripts/backfill_cont_tail.py`），这里只负责运行期读出来拼上去，零额外耗时。
只改 metadata、不改 documents，所以不需要重建 collection，图谱证据链也全保留。
"""

SENTENCE_ENDS = "。！？!?…"

# markdown 语料的块一行就是一条记录（列表项、表格行），一行结束不代表句子没说完，
# 所以整类不参与续接。
_MARKDOWN = (".md", ".markdown")


def is_markdown(metadata: dict | None) -> bool:
    return (metadata or {}).get("source", "").lower().endswith(_MARKDOWN)


def needs_continuation(text: str) -> bool:
    """块尾是不是一个被切断的句子。

    两个条件：结尾不是句末标点；末 8 字里至少有一个实义字符（纯标点的块尾不接）。
    """
    tail = (text or "").rstrip()
    if not tail or tail[-1] in SENTENCE_ENDS:
        return False
    return any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in tail[-8:])


def with_continuation(text: str, metadata: dict | None) -> tuple[str, str]:
    """按需把 `cont_tail` 接到块尾。

    Returns:
        (最终文本, 实际接上的那半句)；没接则返回 (原文, "")。
    """
    if is_markdown(metadata):
        return text, ""

    cont = ((metadata or {}).get("cont_tail") or "").strip()
    if not cont or not needs_continuation(text):
        return text, ""

    return f"{text.rstrip()}\n{cont}", cont
