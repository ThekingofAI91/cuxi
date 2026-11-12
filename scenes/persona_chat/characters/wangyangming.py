"""
王阳明（王守仁 · 阳明先生）角色定义
定位：明代思想家 · 心学集大成者 · 知行合一
人格层先行：资源层（data/persona_chat/wangyangming/）待补充后入库
"""
# 人设内容统一维护在 data/persona_chat/skills/wangyangming.md。

from scenes.persona_chat.models import CharacterDef

wangyangming_character = CharacterDef(
    id="wangyangming",
    name="王阳明",
    description="知行合一，专破心机",
    tagline="心里有贼别发愁，知行合一跟我走。",
    ability="心学宗师",
    review="破山中贼易，破心中贼难，专治想多做少。",
    chroma_collection="persona_wangyangming",
    data_source="data/persona_chat/wangyangming/",
    avatar="",
    theme="paper",
    location={
        "name": "龙场",
        "lat": 26.8470,
        "lon": 106.5900,
        "note": "龙场悟道",
    },
    zone="education",
)
