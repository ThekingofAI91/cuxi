"""
阿德勒（Alfred Adler）角色定义
定位：感情问题咨询师——恋爱、婚姻、失恋、嫉妒、争吵等亲密关系问题
"""
# 人设内容统一维护在 data/persona_chat/skills/adler.md。

from scenes.persona_chat.models import CharacterDef

adler_character = CharacterDef(
    id="adler",
    name="阿尔弗雷德·阿德勒",
    description="爱里迷路，阿德勒指路",
    tagline="情场失意别发怵，阿德勒一聊心里有数。",
    ability="恋爱顾问",
    review="把爱看作合作，把自卑变成起点。",
    chroma_collection="persona_adler",
    data_source="data/persona_chat/adler/阿德勒/",
    avatar="",
    theme="noir",
    location={
        "name": "维也纳",
        "lat": 48.2082,
        "lon": 16.3738,
        "note": "个体心理学发源地",
    },
    zone="education",
)
