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
    # ---- 王阳明（data/persona_chat/wangyangming/）----
    # 顺序要紧：注疏/解译类必须排在「传习录」本体规则之前，否则会被先命中。
    # 归类依据是各文件首页/版权页实证的作者归属（output/_wy_author.log），
    # 不是按书名印象——实测「王阳明大传」这条线下有三本不同的书：
    #   周月亮著 578 页（王阳明大传.pdf）
    #   度阴山著 982 页（知行合一王阳明(1427-1529).pdf）
    #   冈田武彦著 983 页（知行合一的心学智慧 全新修订版.pdf）
    # 而《王阳明：全三册》与《王阳明的六次突围》同为许葆云著。
    # ★冈田武彦那本原有「上/中/下」三个分册扫描件（992 页合计、文本层 0），
    #   已由 KEEP_ONE_PER_BOOK 排除（合订本文字层覆盖 98.3%、页均 569 字，更全），
    #   所以下面「知行合一的心学智慧」这条现在只命中合订本。
    ("wangyangming_core_ideas", "anchor"),   # 人工整理的核心思想/生平/语录
    # 传习录的注疏/解译本：主体虽是王阳明语录，但解译者的话占相当体量，
    # 按「别人写名人的书」处理更安全（宁可降权也别把注解当原话）。
    # 代价要说清：secondary 权重只有 0.25，问「传习录怎么讲」时原话可能召不回来，
    # 只能靠 BM25 兜底。若评测显示传习录细节题召回不足，这里是第一个该调的口子。
    ("传习录全鉴", "secondary"),            # （明）王阳明著 + 迟双明解译，中国纺织出版社
    ("传习录注疏", "secondary"),            # 邓艾民注疏
    ("传习录详注集评", "secondary"),         # 陈荣捷详注集评
    ("一本书读懂阳明心学", "secondary"),      # 现代解读
    ("传习录", "original"),                  # 纯语录本（当前语料里只有上面几种注疏本）
    # 本人著作：《王阳明全集》各辑本（含讹字「王明阳全集」）与《阳明先生文录》
    ("王阳明全集", "original"),
    ("王明阳全集", "original"),
    ("阳明先生文录", "original"),
    # 二手解读/传记/通俗读物（作者见 output/_wy_author.log）
    ("五百年来王阳明", "secondary"),          # 郦波著，上海人民出版社
    ("此心光明", "secondary"),               # 杨东标《王阳明传》
    ("中外名人传记百部", "secondary"),        # 王旭编著
    ("有无之境", "secondary"),               # 陈来《王阳明哲学的精神》
    ("阳明学述要", "secondary"),             # 钱穆著
    ("明朝一哥", "secondary"),               # 吕峥著
    ("王阳明大传", "secondary"),             # 周月亮著
    ("知行合一的心学智慧", "secondary"),      # 冈田武彦著；分册扫描件已排除，只剩合订本
    ("王阳明的六次突围", "secondary"),        # 许葆云著
    ("王阳明：全三册", "secondary"),          # 许葆云著
    ("让良知自由", "secondary"),             # 赵柏田著
    ("知行合一王阳明", "secondary"),          # 度阴山著
    ("修炼强大内心", "secondary"),           # 同上系列的改名重印
    ("心学口诀", "secondary"),               # 他人编写的浓缩口诀，不是原话
    ("心中有孔子", "secondary"),             # 通俗讲述（该文件本身已损坏）
    # 两个 .txt 的实测结论（output/_wy_txt_fix.log）：
    #   《王阳明最神奇的心学》.txt —— 双重乱码已烙进文件，逆转不可逆
    #     （encode('gb18030')→decode('utf-8') 会在中途抛 invalid continuation，
    #      errors='ignore' 会吃字产出「智慄17」这种更坏的结果）。救不回，只能标 secondary。
    #   发现心灵的智慧——王阳明人生哲学感悟.txt —— 文件本身是 GBK，_read_text 能正确解开，
    #     属可用语料。
    ("王阳明最神奇的心学", "secondary"),
    ("发现心灵的智慧", "secondary"),
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
