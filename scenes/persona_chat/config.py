"""
名人对话场景配置
不同名人共用同一套框架，只是角色prompt + 数据源不同
"""

from dataclasses import dataclass, field
from typing import Optional

from scenes.persona_chat.models import CharacterDef
from scenes.persona_chat.characters.jung import jung_character
from scenes.persona_chat.characters.adler import adler_character
# 峰哥 / 张雪峰 归入「娱乐区」（轻量检索 + 极短回答 + 关引用核查），与大创保留的 3 位智者（教育成长区）分区展示
from scenes.persona_chat.characters.fengge import fengge_character
from scenes.persona_chat.characters.zhangxuefeng import zhangxuefeng_character
from scenes.persona_chat.characters.wangyangming import wangyangming_character


@dataclass
class PersonaChatSceneConfig:
    """名人对话场景配置"""

    scene_name: str = "persona_chat"
    display_name: str = "与历史名人对话"
    description: str = "与荣格、阿德勒、王阳明等历史智者对话，从他们的视角获得见解"

    # ---- 可用名人列表 ----
    # 分区：教育成长区（jung/adler/wangyangming）重专业与可溯源；
    #       娱乐区（fengge/zhangxuefeng）重像人、轻检索、极短回答、关引用核查
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
你就是这位历史人物本人，正在与来访者当面交谈。

请以第一人称、用这个人物真实的语言习惯与思维方式直接回应，不要以"分析助手""AI"之类的第三方视角作答；不要给出放之四海而皆准的通用回答，要从你这个人的独特立场、学说与经历出发。
""".strip())

    # 角色一致性 / 差异化铁律：注入到角色化生成的最前方，压制"千篇一律的助手腔"
    persona_voice_directive: str = field(default_factory=lambda: """
【角色一致性铁律】
你现在就是这位人物本人，正与来访者当面交谈。务必做到：

1. 绝对是你，不是 AI 助手：禁止出现"作为 AI""基于资料分析如下""总的来说""需要注意的是""综上所述"等第三方/助手口吻；全程第一人称，像活人一样说话。
2. 差异化优先：同一个问题，你与孔子、弗洛伊德、阿德勒、李白等人的回答必须判若两人——立场、价值排序、措辞、举例、比喻、情绪基调都应鲜明体现"你这个人的特质"，而不是给出任何人都能说的通用建议。
3. 禁止套话与模板：不要硬套"先说观点、再解释、最后给建议"的千篇一律结构（你的人设若如此要求，那只是最低线，优先按你真实的表达习惯来组织）；允许你先讲一个故事、先反问、或劈头给出判断。
4. 用你自己的语言：多用你这个人的口头禅、惯用比喻、独特句式；少用教科书式排比与工整的"第一/第二/第三"。
5. 忠于你的世界观：从你本人的学说、经历、时代局限出发回应；涉及你不可能知道的现代事物，用你的视角去"翻译"，而不是跳出角色做科普。
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


# 显式标注分区（覆盖各角色文件可能的默认值，集中管理）
jung_character.zone = "education"
adler_character.zone = "education"
wangyangming_character.zone = "education"
fengge_character.zone = "entertainment"
zhangxuefeng_character.zone = "entertainment"

# 全局实例
persona_chat_config = PersonaChatSceneConfig()

# 启动时合并用户自建角色（持久化在 data/persona_chat/custom/<id>.json）。
# 自建角色也进入 characters 字典，因此 /persona/characters、预热 BM25、
# 对话检索都会自动覆盖它们，与内置角色完全一致。
try:
    from scenes.persona_chat.custom_store import load_custom_characters
    _custom_chars = load_custom_characters()
    if _custom_chars:
        # 防御：自建 id 不应覆盖同名内置角色
        _dup = set(_custom_chars) & set(persona_chat_config.characters)
        for _dup_id in _dup:
            print(f"[Config] ⚠️ 自建角色 id '{_dup_id}' 与内置角色冲突，已跳过加载")
            _custom_chars.pop(_dup_id, None)
        persona_chat_config.characters.update(_custom_chars)
        print(f"[Config] ✅ 已加载 {len(_custom_chars)} 个自建角色")
except Exception as e:
    print(f"[Config] ⚠️ 加载自建角色失败: {e}")
