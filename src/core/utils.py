"""
通用小工具
界面上展示用的文本处理辅助，放在这里供多处复用。
"""

from __future__ import annotations

import re

# 展示资料出处时要去掉的扩展名（读作：这是文件，不是书名）
_DOC_EXT_RE = re.compile(
    r"\.(pdf|epub|mobi|azw3|djvu|txt|md|doc|docx|wps|rtf)$",
    re.IGNORECASE,
)


def strip_doc_ext(name: str) -> str:
    """去掉资料文件名的扩展名，用于在界面上展示来源著作名。

    "王阳明大传.pdf" -> "王阳明大传"

    非文件名的字符串（如"未知来源"）原样返回。
    """
    if not name:
        return name
    return _DOC_EXT_RE.sub("", name).strip()
