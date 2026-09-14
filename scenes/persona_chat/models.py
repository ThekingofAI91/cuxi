"""
名人对话场景数据模型
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CharacterDef:
    """一位名人的定义"""
    id: str                        # 唯一标识，如 "jung"
    name: str                      # 显示名称（全名），如 "卡尔·荣格"
    description: str               # 简介
    role_prompt: str               # 角色人设prompt（决定说话风格、立场、知识范围）
    chroma_collection: str         # 向量库collection名
    data_source: str               # 数据目录
    avatar: str = "🎭"             # 头像（emoji，空状态中央展示）
    tagline: str = ""              # 个性化标语（首页问候语，按角色展示各自擅长领域）
    ability: str = ""              # 能力标签（首页人物卡片，一句话定位）
    review: str = ""               # 评语（首页人物卡片，一句点评）
    theme: str = ""                # 前端主题: original / paper / noir（留空则由前端按角色id兜底）
    location: Optional[dict] = None  # 经典地点：{"name", "lat", "lon", "note"}，首页地球标记用
    enable_verification: bool = True  # 是否启用引用核查（沉浸型人设如峰哥/张雪峰关闭）
    is_custom: bool = False           # 是否为用户自建角色（内置角色为 False；自建角色可删除/重建知识库）
    created_at: float = 0.0           # 自建角色创建时间戳（Unix 秒）；内置角色为 0.0
    zone: str = "education"           # 分区："education"（教育成长区，重专业/可溯源）| "entertainment"（娱乐区，重像人/轻检索/极短）

    # ---- 酒馆式角色卡构件（思路借鉴 SillyTavern：结构化角色卡 + 世界书）----
    # 两区共用。示例对话与世界书决定"像不像这个人"，但用法不同：
    #   教育区 —— 词条规定核心概念（降幻觉、提专业度），示例对话示范严谨作答的节奏；
    #   娱乐区 —— 词条记黑话与口头禅（保人味），并取代向量检索。
    first_mes: str = ""               # 开场白：进入对话时角色先说的一句
    mes_example: str = ""             # 示例对话（few-shot），用 {{user}} / {{char}} 标记
    scenario: str = ""                # 场景设定：角色与来访者所处的情境
    personality: str = ""             # 性格特质（结构化补充，不与人设主体重复）
    # 世界书：[{"keyword": "触发词", "content": "背景片段"}]，用户消息命中关键词才注入
    # 可扩展字段（均可省略，老格式只有 keyword/content 也能正常工作）：
    #   secondary_keys: 副关键词列表    enabled: 单条开关（默认 True）
    #   constant: 常驻条目，不命中也注入  order: 同批排序，小的在前（默认 100）
    #   position: "before_char"（默认，人设后历史前）| "after_history"（历史后）
    lorebook: list = field(default_factory=list)

    # ---- 采样手感（可选；None = 沿用代码默认值）----
    # 不同角色想要的手感不同：峰哥要"稳、短、有梗"，张雪峰要"快、密、有压迫感"。
    # 留空则完全走默认（temperature 0.8），既有角色不受影响。
    # 仅支持 OpenAI 兼容参数（temperature / top_p / frequency_penalty / presence_penalty）；
    # repetition_penalty 已弃用——DeepSeek API 会拒收该字段（历史遗留字段，勿再配置）。
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None  # 已弃用：仅保留字段兼容旧落盘 JSON，不再透传
