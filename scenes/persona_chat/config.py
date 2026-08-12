"""
名人对话场景配置
不同名人共用同一套框架，只是角色prompt + 数据源不同
"""

from dataclasses import dataclass, field
from typing import Optional

from scenes.persona_chat.models import CharacterDef
from scenes.persona_chat.characters.jung import jung_character
from scenes.persona_chat.characters.adler import adler_character
from scenes.persona_chat.characters.fengge import fengge_character
from scenes.persona_chat.characters.zhangxuefeng import zhangxuefeng_character
from scenes.persona_chat.characters.wangyangming import wangyangming_character


@dataclass
class PersonaChatSceneConfig:
    """名人对话场景配置"""

    scene_name: str = "persona_chat"
    display_name: str = "与历史名人对话"
    description: str = "与荣格、阿德勒、峰哥、爱因斯坦、马克思等名人对话，从他们的视角获得见解"

    # ---- 可用名人列表 ----
    characters: dict[str, CharacterDef] = field(default_factory=lambda: {
        "jung": jung_character,
        "adler": adler_character,
        "fengge": fengge_character,
        "zhangxuefeng": zhangxuefeng_character,
        "wangyangming": wangyangming_character,
    })

    # ---- 默认路由配置 ----
    chroma_collection: str = "persona_jung"
    default_character: str = "jung"

    # ---- Agent 系统提示词 ----
    supervisor_system_prompt: str = field(default_factory=lambda: """
你是一个智能任务路由器。

可用的 Agent：
- retriever: 需要查人物著作/资料时调用
- analyzer: 需要分析、回答问题时调用

决策规则：
1. 大部分问题走 analyzer
2. 需要引用名人原著的，先调 retriever 再调 analyzer
根据用户问题，输出路由决策。
""".strip())

    direct_response_system_prompt: str = field(default_factory=lambda: """
你是一个基于历史人物著作的分析助手。

直接回答用户的问题，但请使用当前角色的人设口吻。
如果无法回答，诚实说明你的知识范围。
""".strip())

    analysis_system_prompt: str = field(default_factory=lambda: """
你是一个基于人物原著的分析专家。根据参考资料进行分析输出。
""".strip())

    analysis_fallback_prompt: str = field(default_factory=lambda: """
你是一个友好、知识丰富的助手。请根据当前角色设定回答用户问题。
""".strip())

    routing_keywords: dict = field(default_factory=lambda: {
        "analyzer": ["分析", "理解", "为什么", "怎么看", "如何看待", "你觉得", "你认为"],
    })

    default_agent: str = "analyzer"

    # ---- 上下文管理配置 ----
    # 对话历史在 prompt 中的使用指令（名人对话需要延续性，与学术场景不同）
    history_instruction: str = field(default_factory=lambda: """
以下是你们之前的对话历史，请仔细阅读并记住其中的内容：
1. 回答时必须自然地延续之前的讨论，像老朋友一样记得对方说过的话
2. 如果用户的问题与之前轮次相关（指代、追问、延续），必须结合之前的内容回答
3. 可以引用你之前说过的观点，保持前后一致，不要自相矛盾
4. 避免重复之前已经完整回答过的内容，在此基础上进一步深入
5. 如果用户完全开启新话题，直接回答新话题即可
""".strip())

    # 是否启用 InfoGap 追问（名人对话是闲聊/咨询式，开启会打断连贯性，故关闭）
    enable_info_gap: bool = False

    # 检索时是否结合最近对话历史（解决指代性问题如"那个梦""这跟它有什么关系"）
    history_aware_retrieval: bool = True


# 全局实例
persona_chat_config = PersonaChatSceneConfig()
