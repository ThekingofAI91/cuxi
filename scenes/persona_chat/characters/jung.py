"""
荣格（Carl Gustav Jung）角色定义
"""
# 人设内容统一维护在 data/persona_chat/skills/jung.md。

from scenes.persona_chat.models import CharacterDef

jung_character = CharacterDef(
    id="jung",
    name="卡尔·荣格",
    description="梦里乾坤，荣格解魂",
    tagline="梦见老宅别乱猜，荣格开口梦全开。",
    ability="解梦师",
    review="把梦当作信使，陪你在潜意识里照见自己。",
    chroma_collection="persona_jung",
    data_source="data/persona_chat/jung/荣格/",
    avatar="",
    theme="original",
    location={
        "name": "波林根",
        "lat": 47.2043,
        "lon": 8.8994,
        "note": "湖畔石塔 · 荣格自建隐居处",
    },
    zone="education",
)
