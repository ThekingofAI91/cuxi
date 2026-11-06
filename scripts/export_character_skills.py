"""
export_character_skills.py — 把内置角色的 role_prompt 导出成 skill 包

人设正文在 `scenes/persona_chat/characters/*.py` 里以字符串常量维护，这个脚本把它
逐字搬到 `data/persona_chat/skills/{id}.md`，前面补一段 frontmatter（名称、定位、分区、
对应 collection），后面保持原文不动。

为什么要脚本而不是手抄：人设是一千多字的精细措辞，手抄必然改动字句，而 prompt 的
字句直接影响生成结果。脚本保证「搬过去的内容与搬之前逐字相同」，行为零变化。

幂等：重复执行覆盖同名文件；已存在但名字不在内置角色列表里的文件不动
（例如手写的 legend-narrator.md）。

用法：
    .venv/Scripts/python.exe scripts/export_character_skills.py
    .venv/Scripts/python.exe scripts/export_character_skills.py --check   # 只比对不写盘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SKILL_DIR = ROOT / "data" / "persona_chat" / "skills"


def _frontmatter(char) -> str:
    """生成 frontmatter。description 用首页卡片已有的文案，避免另写一份说法。"""
    desc_bits = [b for b in (char.ability, char.review) if b]
    desc = "；".join(desc_bits) or char.description
    lines = [
        "---",
        f"name: {char.id}-persona",
        "description: |",
        f"  {char.name}角色扮演设定：{desc}",
        f"  分区：{char.zone}　｜　语料库：{char.chroma_collection}",
        "  正文为逐字迁移的原始人设，改动请只改正文、不要动 frontmatter。",
        f"zone: {char.zone}",
        f"collection: {char.chroma_collection}",
        "---",
        "",
    ]
    return "\n".join(lines)


def collect_characters() -> list:
    from scenes.persona_chat.characters.adler import adler_character
    from scenes.persona_chat.characters.fengge import fengge_character
    from scenes.persona_chat.characters.jung import jung_character
    from scenes.persona_chat.characters.wangyangming import wangyangming_character
    from scenes.persona_chat.characters.zhangxuefeng import zhangxuefeng_character

    return [
        jung_character,
        adler_character,
        wangyangming_character,
        fengge_character,
        zhangxuefeng_character,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="只比对差异，不写盘")
    args = parser.parse_args()

    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    chars = collect_characters()
    changed = 0

    for char in chars:
        path = SKILL_DIR / f"{char.id}.md"
        content = _frontmatter(char) + (char.role_prompt or "").strip() + "\n"

        if args.check:
            old = path.read_text(encoding="utf-8") if path.exists() else ""
            status = "一致" if old == content else "有差异"
            if old != content:
                changed += 1
            print(f"{status:>4}　{char.id:<14} {len(char.role_prompt or ''):>5} 字")
            continue

        path.write_text(content, encoding="utf-8")
        print(f"写入　{char.id:<14} {len(char.role_prompt or ''):>5} 字　{path}")

    if args.check:
        print(f"\n共 {len(chars)} 个角色，{changed} 个与磁盘不一致")
        return 1 if changed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
