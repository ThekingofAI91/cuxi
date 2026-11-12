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
    chroma_collection: str         # 向量库collection名
    data_source: str               # 数据目录
    # 以下人设字段仅承接创建/导入请求与旧 JSON 迁移，不作为运行时回退。
    # 内置定义不填写正文；生成、导出与开场展示统一读取 Markdown 技能包。
    role_prompt: str = ""
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

    # ---- 创建/导入时的人设输入，保存后以技能包为唯一来源 ----
    first_mes: str = ""               # 开场白：进入对话时角色先说的一句
    mes_example: str = ""             # 示例对话（few-shot），用 {{user}} / {{char}} 标记
    scenario: str = ""                # 场景设定：角色与来访者所处的情境
    personality: str = ""             # 性格特质（结构化补充，不与人设主体重复）
    # 世界书：[{"keyword": "触发词", "content": "背景片段"}]，用户消息命中关键词才注入
    lorebook: list = field(default_factory=list)
    voice_directive: str = ""         # 口吻要求
    post_history_instructions: str = ""  # 在历史之后注入的角色指令

    # ---- 采样手感（可选；None = 沿用代码默认值）----
    # 不同角色想要的手感不同：峰哥要"稳、短、有梗"，张雪峰要"快、密、有压迫感"。
    # 留空则完全走默认（temperature 0.8），既有角色不受影响。
    # 仅支持 OpenAI 兼容参数（temperature / top_p / frequency_penalty / presence_penalty）；
    # repetition_penalty 已弃用——DeepSeek API 会拒收该字段（历史遗留字段，勿再配置）。
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None  # 已弃用：仅保留字段兼容旧落盘 JSON，不再透传
