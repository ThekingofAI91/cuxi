"""
轻量内容安全过滤：输入拦截 + 输出兜底。

只拦截真正的高危/违法类别（武器、毒品、器官买卖、洗钱、网赌、
开盒人肉、诈骗话术、恐怖袭击、邪教、传销拉人）。

【明确不拦截】两性 / 性教育 / 亲密关系 / 性爱等正常咨询问答——
此类话题属于正常对话内容，一律放行，不做任何限制。

思路：宁可多拦少漏，拦截时给一个礼貌的兜底回复，不暴露过滤规则。
"""

_BANNED_PATTERNS = [
    "制作炸弹",
    "枪支",
    "冰毒",
    "海洛因",
    "买卖器官",
    "洗钱",
    "赌博网站",
    "开盒",
    "人肉搜索",
    "诈骗话术",
    "恐怖袭击",
    "邪教",
    "传销拉人",
]

_REFUSAL = "这个话题我不太方便聊，咱们换个话题吧。"


def contains_sensitive(text: str) -> bool:
    """输入侧：是否命中敏感词"""
    if not text:
        return False
    for p in _BANNED_PATTERNS:
        if p in text:
            return True
    return False


def sanitize_output(text: str) -> str:
    """输出侧：回答命中敏感词时整体替换为兜底话术"""
    return _REFUSAL if contains_sensitive(text) else text


def refusal_message() -> str:
    return _REFUSAL
