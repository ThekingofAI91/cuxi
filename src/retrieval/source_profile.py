"""
source_profile.py — 语料来源类型档案

将知识库中的每个来源（按文件名）分类，并在检索 RRF 融合阶段按类型加权：

- original:   名人本人著作 —— 观点支撑，权威性最高
- oral:       口述 / 讲座 / 回忆录 —— 对话风格参考（学"怎么说话"）
- artificial: 后人虚构创作（如《被讨厌的勇气》）—— 风格可参考，观点不能当原话
- secondary:  二手解读 / 编译（别人写名人的书）—— 强降权，避免"第三者评价"冒充名人原话
- anchor:     人工整理的事实锚点（core_ideas.md）—— 生平/著作/关键语录，防幻觉关键
- unknown:    未匹配的来源 —— 保持中性权重（行为与未加权时一致）

使用方式：
    classify_source("红书.pdf")          -> "original"
    get_source_weight("红书.pdf")        -> 1.2
    检索代码在 RRF 累加时乘以 get_source_weight(source) 即可。
"""

# ============================================================
# 来源类型权重
# ============================================================

SOURCE_TYPE_WEIGHTS: dict[str, float] = {
    "anchor": 2.0,       # 事实锚点：关键事实唯一出处，最高优先
    "oral": 1.5,         # 口述/讲座/回忆录：对话风格语料，回答"像不像"优先召回
    "original": 1.2,     # 原著：观点支撑，高于默认
    "artificial": 0.6,   # 后人虚构创作：风格参考，观点不可靠
    "secondary": 0.25,   # 二手解读：强降权（不彻底过滤，保证冷门问题仍有兜底召回）
    "unknown": 1.0,      # 未标记来源：与未加权行为一致
}

# ============================================================
# 文件名子串 -> 类型 规则表（按列表顺序匹配，命中即返回）
# ============================================================

_SOURCE_TYPE_RULES: list[tuple[str, str]] = [
    # ---- 荣格（data/persona_chat/jung/荣格/）----
    ("荣格自传", "oral"),                    # 口述回忆录（回忆、梦、思考）
    ("荣格分析心理学导论", "oral"),          # 1925 苏黎世讲座记录（沙姆达萨尼编，荣格原话）
    ("大师思想集萃", "oral"),               # CIP：荣格著/高适编译 —— 荣格本人第一人称回忆选编，非二手解读
    ("荣格心理学", "secondary"),             # 他人编著的解读本
    ("荣格性格哲学", "secondary"),           # 他人编著的解读本
    ("阴影与自我", "secondary"),             # 解读本
    ("心理类型", "original"),
    ("原型与集体无意识", "original"),
    ("自我与自性", "original"),
    ("人类与象征", "original"),
    ("移情心理学", "original"),
    ("炼金术之梦", "original"),
    ("未发现的自我", "original"),
    ("红书", "original"),
    ("哲学树", "original"),
    ("金花的秘密", "original"),
    ("寻求灵魂的现代人", "original"),
    ("现代人的心灵问题", "original"),
    ("jung_core_ideas", "anchor"),           # 人工整理的核心思想/生平/语录
    # ---- 阿德勒（data/persona_chat/adler/阿德勒/）----
    ("走出孤独", "oral"),                    # 演讲/讲座整理
    ("被讨厌的勇气", "artificial"),          # 岸见一郎/古贺史健虚构的哲人对话
    ("阿德勒的情绪整理术", "secondary"),     # 后世解读/普及读物
    ("超越自卑与洞察人性", "original"),
    ("生命重建课", "original"),              # CIP 确认：阿德勒本人著作
    ("adler_core_ideas", "anchor"),          # 人工整理的核心思想/生平/语录
    # ---- 峰哥（data/persona_chat/fengge/）----
    ("interviews-2026", "oral"),             # 2024 媒体专访实录（腾讯新闻/全媒派/36氪）
    ("fengge_skill", "oral"),                # 峰哥公开视频/专访/百科多渠道调研（口述为主）
    # ---- 张雪峰（data/persona_chat/zhangxuefeng/）----
    ("quotes-classified", "oral"),           # 公开语录分类辑录（直播/讲座/媒体引述）
    ("zhangxuefeng_skill", "oral"),          # 著作+采访+语录多渠道调研（口述/原著混合）
    # ---- 通用兜底（置于角色规则之后，避免抢占角色专属匹配）----
    ("interview", "oral"),                   # 访谈实录
    ("quotes", "oral"),                      # 语录辑录
]


def classify_source(source: str) -> str:
    """
    按文件名子串匹配来源类型。

    Args:
        source: 文档来源标识（通常为文件名，如 "红书.pdf"）

    Returns:
        source_type: "original" / "oral" / "artificial" / "secondary" / "anchor" / "unknown"
    """
    if not source:
        return "unknown"
    for keyword, source_type in _SOURCE_TYPE_RULES:
        if keyword in source:
            return source_type
    return "unknown"


def get_source_weight(source: str) -> float:
    """返回来源类型对应的检索权重（RRF 融合阶段使用）"""
    return SOURCE_TYPE_WEIGHTS.get(classify_source(source), SOURCE_TYPE_WEIGHTS["unknown"])
