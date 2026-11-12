"""校验内置角色技能包；保留旧命令入口，不再从 Python 导出或覆盖人设。

技能包是人设内容的唯一来源。此命令只检查技能结构、必需人设和身份元数据，
不启动应用，不读取自建角色，也不写任何技能文件。

用法：
    .venv/Scripts/python.exe scripts/export_character_skills.py --check
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "data" / "persona_chat" / "skills"
CHARACTER_IDS = ("jung", "adler", "wangyangming", "fengge", "zhangxuefeng")
PERSONA_HEADINGS = (
    "人设", "性格特质", "当前场景", "世界书", "对话示例", "开场白", "口吻要求", "后历史指令",
)


def _load_parser() -> ModuleType:
    """直接加载无应用副作用的解析器，避免包入口启动 config 和自建角色存储。"""
    name = "_persona_skill_validation_loader"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scenes" / "persona_chat" / "skill_loader.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载技能解析器")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.SKILL_DIR = SKILL_DIR
    return module


def _metadata(character_id: str) -> dict[str, str]:
    """只读取 Python 中的身份元数据，不实例化角色，也不把正文当作校验基线。"""
    path = ROOT / "scenes" / "persona_chat" / "characters" / f"{character_id}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    call = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "CharacterDef"
    )
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in call.keywords
        if keyword.arg in {"id", "zone", "chroma_collection"}
    }


def _headings(body: str) -> set[str]:
    headings = set(re.findall(r"^##[ \t]+([^\r\n]+?)[ \t]*$", body, flags=re.MULTILINE))
    # 旧标题继续可读，作者可以逐份改名，无需同时重写所有技能。
    if "核心概念词条" in headings:
        headings.add("世界书")
    return headings


def validate() -> list[str]:
    parser = _load_parser()
    errors: list[str] = []
    for character_id in CHARACTER_IDS:
        filename = f"{character_id}.md"
        pack = parser.load_skill(character_id)
        if pack is None:
            errors.append(f"{filename}: 技能包缺失或为空")
            continue
        missing = set(PERSONA_HEADINGS) - _headings(pack.body)
        if missing:
            errors.append(f"{filename}: 缺少分节 {', '.join(sorted(missing))}")
        sections = parser.parse_persona_sections(pack.body)
        if not sections.role_prompt:
            errors.append(f"{filename}: 人设不能为空")
        metadata = _metadata(character_id)
        if pack.name != f"{character_id}-persona":
            errors.append(f"{filename}: name 与角色 id 不一致")
        if pack.zone != metadata.get("zone", "education"):
            errors.append(f"{filename}: zone 与角色元数据不一致")
        if pack.meta.get("collection") != metadata["chroma_collection"]:
            errors.append(f"{filename}: collection 与角色元数据不一致")

    for zone in ("education", "entertainment"):
        filename = f"_defaults-{zone}.md"
        pack = parser.load_skill(f"_defaults-{zone}")
        if pack is None:
            errors.append(f"{filename}: 默认模板缺失或为空")
            continue
        if _headings(pack.body) != {"口吻要求", "后历史指令"}:
            errors.append(f"{filename}: 默认模板只应包含口吻要求与后历史指令")
        if pack.zone != zone:
            errors.append(f"{filename}: zone 与模板名称不一致")
        sections = parser.parse_persona_sections(pack.body)
        if not sections.voice_directive or not sections.post_history_instructions:
            errors.append(f"{filename}: 默认口吻与后历史指令不能为空")
        if "{{char}}" not in sections.post_history_instructions:
            errors.append(f"{filename}: 后历史指令应使用 {{{{char}}}} 角色名占位符")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="只读校验技能包，不再从 Python 导出人设")
    parser.add_argument("--check", action="store_true", help="兼容旧命令；所有运行均只读校验")
    parser.parse_args()
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("技能校验通过：5 个内置角色、2 个自建角色默认模板。未写入任何文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
