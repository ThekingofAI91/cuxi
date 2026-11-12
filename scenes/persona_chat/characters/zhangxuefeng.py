"""
张雪峰（本名张子彪）角色定义
定位：考研名师 · 高考志愿填报专家 · 现实主义升学顾问（"社会筛子论"视角）
人设素材来源：alchaincyf/zhangxuefeng-skill（5本著作 + 15+篇深度采访 + 30+条语录调研，2026-04）
"""
# 人设内容统一维护在 data/persona_chat/skills/zhangxuefeng.md。

from scenes.persona_chat.models import CharacterDef

zhangxuefeng_character = CharacterDef(
    id="zhangxuefeng",
    name="张雪峰",
    description="长跑之王，择业导航",
    tagline="专业怎么选，城市怎么挑，张雪峰一聊，全都明了。",
    ability="升学规划师",
    review="普通家庭的孩子，选对路比努力更重要。",
    chroma_collection="persona_zhangxuefeng",
    data_source="data/persona_chat/zhangxuefeng/",
    avatar="",
    theme="live",
    location={
        "name": "郑州大学",
        "lat": 34.8176,
        "lon": 113.5384,
        "note": "给排水专业毕业 · 生涯起点",
    },
    enable_verification=False,
    zone="entertainment",
)
