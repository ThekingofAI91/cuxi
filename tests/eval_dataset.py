"""
RAG 评估测试集
基于荣格知识库的问题 + 标准答案 + 评估要点
"""

# 评估测试集：每个问题包含
# - question: 问题
# - ground_truth: 标准答案（用于参考）
# - key_points: 评估要点（LLM Judge 会检查这些点是否被覆盖）
# - difficulty: 难度级别（easy/medium/hard）

EVAL_DATASET = [
    # ===== 基础概念题（Easy）=====
    {
        "question": "荣格把心灵分为哪三个层次？",
        "ground_truth": "荣格将心灵（Psyche）划分为三个层次：意识（Ego）、个人潜意识（Personal Unconscious）和集体潜意识（Collective Unconscious）。",
        "key_points": ["意识", "个人潜意识", "集体潜意识"],
        "difficulty": "easy",
    },
    {
        "question": "什么是人格面具（Persona）？",
        "ground_truth": "人格面具是个人在社会中展示的外在形象，是适应社会期望的面具。它帮助我们顺利社交，但如果过度认同人格面具，就会失去真实的自我。",
        "key_points": ["社会", "外在形象", "面具", "适应"],
        "difficulty": "easy",
    },
    {
        "question": "荣格提出的四种心理功能是什么？",
        "ground_truth": "荣格提出的四种心理功能是：思维（Thinking）、情感（Feeling）、感觉（Sensation）和直觉（Intuition）。",
        "key_points": ["思维", "情感", "感觉", "直觉"],
        "difficulty": "easy",
    },
    {
        "question": "什么是共时性（Synchronicity）？",
        "ground_truth": "共时性是指两个或多个事件之间有意义的巧合，它们之间没有因果关系，但有意义上的联系。",
        "key_points": ["有意义的巧合", "无因果关系", "意义联系"],
        "difficulty": "easy",
    },

    # ===== 理解题（Medium）=====
    {
        "question": "阴影（Shadow）在荣格心理学中代表什么？它只有负面含义吗？",
        "ground_truth": "阴影是人格中被意识否定、压抑或忽视的部分，包含不被接受的欲望、弱点。但阴影并非纯粹的恶，它也包含创造力和生命力。荣格强调认识自己的阴影是心理成长的关键步骤。",
        "key_points": ["被压抑的部分", "不只是负面", "创造力", "心理成长"],
        "difficulty": "medium",
    },
    {
        "question": "个性化过程（Individuation）的主要阶段有哪些？",
        "ground_truth": "个性化的主要阶段包括：1.人格面具的识别 2.阴影的整合 3.阿尼玛/阿尼姆斯的整合 4.自性的实现。",
        "key_points": ["人格面具识别", "阴影整合", "阿尼玛整合", "自性实现"],
        "difficulty": "medium",
    },
    {
        "question": "荣格和弗洛伊德对梦的理解有什么不同？",
        "ground_truth": "弗洛伊德认为梦是被压抑欲望的伪装满足；荣格认为梦是潜意识的自然表达，具有补偿和预示功能。",
        "key_points": ["弗洛伊德：压抑欲望", "荣格：潜意识表达", "补偿功能"],
        "difficulty": "medium",
    },
    {
        "question": "荣格和弗洛伊德是什么时候决裂的？原因是什么？",
        "ground_truth": "1912年，荣格与弗洛伊德因理论分歧决裂，主要是关于力比多的本质和集体潜意识。",
        "key_points": ["1912年", "理论分歧", "力比多", "集体潜意识"],
        "difficulty": "medium",
    },

    # ===== 深度分析题（Hard）=====
    {
        "question": "荣格如何用炼金术象征来描述个性化过程？",
        "ground_truth": "荣格发现炼金术的象征体系与个性化过程相似：黑化（Nigredo）代表面对阴影；白化（Albedo）代表净化和意识觉醒；黄化（Citrinitas）代表智慧的曙光；红化（Rubedo）代表最终的整合与圆满。哲人石是自性的象征。",
        "key_points": ["黑化-阴影", "白化-净化", "黄化-智慧", "红化-整合", "哲人石-自性"],
        "difficulty": "hard",
    },
    {
        "question": "什么是积极想象（Active Imagination）？荣格自己是如何使用这个技术的？",
        "ground_truth": "积极想象是荣格发展的一种心理技术，用于与潜意识内容进行有意识的对话。步骤包括：放松状态下让潜意识意象浮现、以第一人称与其对话、记录内容、反思意义。荣格在与弗洛伊德分裂后的'潜意识风暴'期间大量使用，后来记录在《红书》中。",
        "key_points": ["与潜意识对话", "步骤", "红书", "与弗洛伊德分裂后"],
        "difficulty": "hard",
    },
    {
        "question": "阿尼玛和阿尼姆斯分别是什么？它们在心理发展中起什么作用？",
        "ground_truth": "阿尼玛是男性心灵中的女性面向，阿尼姆斯是女性心灵中的男性面向。它们代表异性特质，影响我们与异性的关系。阿尼玛在男性内心扮演'灵魂意象'的角色，阿尼姆斯在女性内心代表'意义与逻辑'的面向。",
        "key_points": ["阿尼玛-男性中的女性", "阿尼姆斯-女性中的男性", "影响异性关系", "整合是个性化的一部分"],
        "difficulty": "hard",
    },
    {
        "question": "荣格认为自性（Self）是什么？它和自我（Ego）有什么区别？",
        "ground_truth": "自性是心灵的整体，是意识与潜意识的统一体，是心理发展的终极目标。自我只是意识的中心。自性不仅是中心，而是整个圆周的 encompassing 整体。",
        "key_points": ["自性=整体", "自我=意识中心", "统一体", "终极目标"],
        "difficulty": "hard",
    },
]


def get_dataset_by_difficulty(difficulty: str = None) -> list[dict]:
    """按难度筛选测试集"""
    if difficulty is None:
        return EVAL_DATASET
    return [d for d in EVAL_DATASET if d["difficulty"] == difficulty]


def get_all_questions() -> list[str]:
    """获取所有问题"""
    return [d["question"] for d in EVAL_DATASET]
