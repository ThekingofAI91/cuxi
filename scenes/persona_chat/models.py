"""
名人对话场景数据模型
"""

from dataclasses import dataclass, field


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
    theme: str = ""                # 前端主题: original / paper / noir（留空则由前端按角色id兜底）
    enable_verification: bool = True  # 是否启用引用核查（沉浸型人设如峰哥/张雪峰关闭）
