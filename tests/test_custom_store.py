"""自建角色技能源、旧 JSON 迁移与失败恢复；只读写 tmp_path。"""

import json
from dataclasses import asdict

import pytest

from scenes.persona_chat import custom_store as store
from scenes.persona_chat.card_io import export_card
from scenes.persona_chat.models import CharacterDef
from scenes.persona_chat.skill_loader import (
    PERSONA_FIELDS, PersonaSkillError, default_persona_sections, resolve_persona_sections,
)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CUSTOM_DIR", tmp_path)


def character(**overrides):
    fields = dict(
        id="test_persona", name="试作人物", description="展示信息",
        chroma_collection="custom_test_persona", data_source="", is_custom=True,
        zone="entertainment", role_prompt="你是试作人物。", first_mes="进来吧。",
        personality="直率", scenario="站在门边", mes_example="{{char}}: 说吧。",
        lorebook=[{"keyword": "门", "content": "门外：下着雨。\n雨已下了三天。"}],
    )
    fields.update(overrides)
    return CharacterDef(**fields)


def test_new_character_persists_complete_skill_and_metadata_only(tmp_path):
    original = character()
    path = store.save_custom_character(original)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert not (set(PERSONA_FIELDS) & metadata.keys())
    assert metadata["description"] == "展示信息"
    loaded = store.load_custom_characters()[original.id]
    assert not loaded.role_prompt and not loaded.first_mes and not loaded.lorebook
    sections = resolve_persona_sections(loaded)
    assert sections.role_prompt == original.role_prompt
    assert sections.first_mes == "进来吧。"
    assert sections.lorebook == original.lorebook
    defaults = default_persona_sections(original.zone, original.name)
    assert sections.voice_directive == defaults.voice_directive
    assert sections.post_history_instructions == defaults.post_history_instructions
    assert "{{char}}" not in sections.post_history_instructions


def test_legacy_json_is_backed_up_before_migrating(tmp_path):
    original = character()
    path = tmp_path / f"{original.id}.json"
    old_text = json.dumps(asdict(original), ensure_ascii=False, indent=2)
    path.write_text(old_text, encoding="utf-8")
    loaded = store.load_custom_characters()[original.id]
    assert (tmp_path / f"{original.id}.json.bak").read_text(encoding="utf-8") == old_text
    assert not (set(PERSONA_FIELDS) & json.loads(path.read_text(encoding="utf-8")).keys())
    assert resolve_persona_sections(loaded).mes_example == original.mes_example
    # 第二次启动既不改正文，也不覆写首次备份。
    skill = tmp_path / f"{original.id}.md"
    previous = (skill.read_bytes(), skill.stat().st_mtime_ns, path.stat().st_mtime_ns)
    assert original.id in store.load_custom_characters()
    assert (skill.read_bytes(), skill.stat().st_mtime_ns, path.stat().st_mtime_ns) == previous
    assert (tmp_path / f"{original.id}.json.bak").read_text(encoding="utf-8") == old_text


def test_existing_skill_wins_over_legacy_json_and_empty_dto(tmp_path):
    original = character()
    store.save_custom_character(original)
    skill = tmp_path / f"{original.id}.md"
    skill.write_text(skill.read_text(encoding="utf-8").replace("进来吧。", "先敲门。"), encoding="utf-8")
    edited = skill.read_bytes()
    (tmp_path / f"{original.id}.json").write_text(json.dumps(asdict(original)), encoding="utf-8")
    loaded = store.load_custom_characters()[original.id]
    loaded.description = "新展示信息"
    store.save_custom_character(loaded)
    assert skill.read_bytes() == edited
    assert export_card(loaded)["first_mes"] == "先敲门。"
    assert json.loads((tmp_path / f"{original.id}.json").read_text(encoding="utf-8"))["description"] == "新展示信息"


def test_migration_failure_preserves_legacy_json_and_backup(tmp_path, monkeypatch):
    original = character()
    path = tmp_path / f"{original.id}.json"
    old_bytes = json.dumps(asdict(original)).encode("utf-8")
    path.write_bytes(old_bytes)
    write = store._atomic_write

    def fail_metadata(target, content):
        if target.suffix == ".json":
            raise OSError("disk full")
        return write(target, content)

    monkeypatch.setattr(store, "_atomic_write", fail_metadata)
    assert store.load_custom_characters() == {}
    assert path.read_bytes() == old_bytes
    assert (tmp_path / f"{original.id}.json.bak").read_bytes() == old_bytes
    assert not (tmp_path / f"{original.id}.md").exists()


def test_missing_skill_has_no_python_or_backup_fallback(tmp_path):
    original = character()
    path = store.save_custom_character(original)
    (tmp_path / f"{original.id}.md").unlink()
    assert store.load_custom_characters() == {}
    assert path.is_file()
    with pytest.raises(PersonaSkillError):
        resolve_persona_sections(original)


@pytest.mark.parametrize("invalid", ["../escape", "AA", "bad\n", "x", "a" * 41])
def test_invalid_paths_are_rejected_before_writing(tmp_path, invalid):
    assert not store.is_valid_custom_id(invalid)
    with pytest.raises(ValueError):
        store.save_custom_character(character(id=invalid))
    with pytest.raises(ValueError):
        store.delete_custom_character_file(invalid)
    assert list(tmp_path.iterdir()) == []


def test_mismatched_filename_is_not_migrated(tmp_path):
    path = tmp_path / "other_name.json"
    path.write_text(json.dumps(asdict(character())), encoding="utf-8")
    assert store.load_custom_characters() == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other_name.json"]


def test_delete_removes_metadata_skill_and_migration_backup(tmp_path):
    original = character()
    (tmp_path / f"{original.id}.json").write_text(json.dumps(asdict(original)), encoding="utf-8")
    assert original.id in store.load_custom_characters()
    assert store.delete_custom_character_file(original.id)
    assert list(tmp_path.iterdir()) == []
