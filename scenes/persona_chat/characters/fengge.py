"""
峰哥（周丽峰 · 峰哥亡命天涯）角色定义
定位：街头社会学家 · 底层纪实内容创作者 · 辩证反转式人生观察
人设素材来源：feng-ge-skill（公开视频、专访、百科等多渠道调研，2026-04）
"""
# 人设内容统一维护在 data/persona_chat/skills/fengge.md。

from scenes.persona_chat.models import CharacterDef

fengge_character = CharacterDef(
    id="fengge",
    name="峰哥",
    description="下三路之神",
    tagline="失业失恋穷光蛋，找峰哥，人生大翻盘。",
    ability="街头观察者",
    review="坏事里先找机会，是他的赢学。",
    chroma_collection="persona_fengge",
    data_source="data/persona_chat/fengge/",
    avatar="",
    theme="street",
    location={
        "name": "深圳三和",
        "lat": 22.6600,
        "lon": 114.0450,
        "note": "三和大神 · 底层叙事现场",
    },
    enable_verification=False,
    zone="entertainment",
)
