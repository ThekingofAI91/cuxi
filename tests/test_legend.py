"""「传奇」剧情模式的测试（全部打桩，零网络）。

覆盖三层：
  1. 存储层 legend_store：读写/删除/列表/轮次裁剪/字段兼容/校验
  2. 引擎 framework.legend：system prompt 装配、messages 轮次映射、开局与行动的差别
  3. API：创建校验、读档、删档、以及 act 的 SSE 事件序列（打桩 LLM，不打真网关）

★重点验「不替主角做决定」这条铁律真的写进 prompt 了。
传奇的定义性特征就是「用户扮演主角」，模型一旦替主角说话/做决定，
这个玩法就塌了——它比任何功能点都该被测试锁住。
"""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework import legend as lg  # noqa: E402
from scenes.persona_chat import legend_store as ls  # noqa: E402


# ============================================================
# 存储层
# ============================================================

@pytest.fixture
def store(tmp_path, monkeypatch):
    """把存档目录指到临时目录，避免污染真实 data/。"""
    monkeypatch.setattr(ls, "LEGEND_DIR", tmp_path / "legends")
    return tmp_path / "legends"


def _new_save(**kw):
    base = dict(
        id="hero_abc123",
        title="李寻",
        world="大唐末年，长安城暗流涌动，江湖与庙堂的界限早已模糊。",
        protagonist_name="李寻",
        protagonist_desc="一个靠替人送信为生的落魄游侠",
        npcs=[{"name": "苏娘", "role": "酒肆老板娘", "persona": "消息灵通"}],
        style="classic",
    )
    base.update(kw)
    return ls.LegendSave(**base)


def test_save_and_load_roundtrip(store):
    save = _new_save()
    ls.save_legend(save)
    got = ls.load_legend("hero_abc123")
    assert got is not None
    assert got.world == save.world
    assert got.protagonist_name == "李寻"
    assert got.npcs[0]["name"] == "苏娘"
    assert got.created_at > 0 and got.updated_at > 0


def test_load_missing_returns_none(store):
    assert ls.load_legend("nope_000000") is None


def test_load_rejects_bad_id(store):
    """非法 id 直接拒（防路径穿越）"""
    assert ls.load_legend("../etc/passwd") is None
    assert ls.load_legend("UPPER") is None
    assert ls.load_legend("") is None


def test_delete(store):
    ls.save_legend(_new_save())
    assert ls.delete_legend("hero_abc123") is True
    assert ls.load_legend("hero_abc123") is None
    assert ls.delete_legend("hero_abc123") is False


def test_list_sorted_by_updated(store):
    a = _new_save(id="a_111111", title="甲")
    b = _new_save(id="b_222222", title="乙")
    ls.save_legend(a)
    ls.save_legend(b)
    # 再动一次 a，让它成为最近更新的
    a.turns.append({"role": "user", "content": "推门", "ts": 1.0})
    ls.save_legend(a)
    items = ls.list_legends()
    assert items[0]["id"] == "a_111111"
    assert items[0]["turns"] == 1
    assert items[1]["id"] == "b_222222"


def test_turns_trimmed_by_round_not_by_entry(store):
    """★裁剪必须按「轮」而不是「条」：切一半会让模型忘记自己刚说了什么。"""
    save = _new_save()
    for i in range(ls.MAX_TURNS + 5):
        save.turns.append({"role": "user", "content": f"行动{i}", "ts": float(i)})
        save.turns.append({"role": "narrator", "content": f"叙述{i}", "ts": float(i)})
    ls.save_legend(save)
    got = ls.load_legend("hero_abc123")

    assert got.turn_count() == ls.MAX_TURNS
    # 每轮必须成对：最后一条一定是叙述，第一条一定是用户行动
    assert got.turns[-1]["role"] == "narrator"
    assert got.turns[0]["role"] == "user"
    # 且保留的是「最近的」那些轮
    assert got.turns[-2]["content"] == f"行动{ls.MAX_TURNS + 4}"


def test_load_tolerates_missing_fields(store):
    """旧档缺字段要能补默认值，不能因为一次格式演进废掉用户存档。"""
    import json
    d = ls.legend_dir()
    (d / "old_999999.json").write_text(json.dumps({
        "id": "old_999999", "title": "旧档", "world": "旧世界",
        "protagonist_name": "旧人",
    }, ensure_ascii=False), encoding="utf-8")
    got = ls.load_legend("old_999999")
    assert got is not None
    assert got.npcs == [] and got.turns == []
    assert got.style == "classic"


def test_generate_id_is_valid_and_unique(store):
    a = ls.generate_legend_id("李寻")
    b = ls.generate_legend_id("李寻")
    assert ls.is_valid_legend_id(a)
    assert a != b          # 中文名会退化成 legend_ 前缀，靠 uuid 保证唯一


# ---- 校验 ----

def test_validate_rejects_empty_world():
    ok, why = ls.validate_new_save(world="", protagonist_name="甲")
    assert not ok and "世界观" in why


def test_validate_rejects_short_world():
    ok, why = ls.validate_new_save(world="很短", protagonist_name="甲")
    assert not ok and "10 个字" in why


def test_validate_rejects_empty_protagonist():
    ok, why = ls.validate_new_save(world="这是一个足够长的世界观设定文本。", protagonist_name="  ")
    assert not ok and "主角" in why


def test_validate_rejects_too_many_npcs():
    world = "这是一个足够长的世界观设定文本内容。"
    npcs = [{"name": f"配{i}"} for i in range(ls.MAX_NPCS + 1)]
    ok, why = ls.validate_new_save(world=world, protagonist_name="甲", npcs=npcs)
    assert not ok and str(ls.MAX_NPCS) in why


def test_validate_rejects_duplicate_npc_names():
    world = "这是一个足够长的世界观设定文本内容。"
    npcs = [{"name": "苏娘"}, {"name": "苏娘"}]
    ok, why = ls.validate_new_save(world=world, protagonist_name="甲", npcs=npcs)
    assert not ok and "重复" in why


def test_validate_rejects_npc_same_as_protagonist():
    world = "这是一个足够长的世界观设定文本内容。"
    ok, why = ls.validate_new_save(world=world, protagonist_name="李寻",
                                   npcs=[{"name": "李寻"}])
    assert not ok and "不能与主角相同" in why


def test_validate_accepts_good_input():
    ok, why = ls.validate_new_save(
        world="这是一个足够长的世界观设定文本内容。",
        protagonist_name="李寻",
        npcs=[{"name": "苏娘"}],
    )
    assert ok and why == ""


# ============================================================
# 引擎
# ============================================================

def test_system_prompt_has_all_sections():
    save = _new_save()
    p = lg.build_system_prompt(save)
    assert save.world in p
    assert save.protagonist_name in p
    assert "苏娘" in p
    assert "酒肆老板娘" in p


def test_system_prompt_forbids_acting_for_protagonist():
    """★铁律：不能替主角做决定/说话。这是传奇玩法成立的前提。"""
    p = lg.build_system_prompt(_new_save())
    assert "绝不能替主角做决定" in p
    # 也要明确禁止「跳出剧情」的元叙述
    assert "作为 AI" in p


def test_system_prompt_states_no_retrieval():
    """传奇是虚构叙事，不能标来源——这条要写死在 prompt 里。"""
    p = lg.build_system_prompt(_new_save())
    assert "不标注任何来源" in p


def test_system_prompt_handles_no_npcs():
    save = _new_save(npcs=[])
    p = lg.build_system_prompt(save)
    assert "本局没有配角" in p


def test_style_hint_applied():
    a = lg.build_system_prompt(_new_save(style="dark"))
    b = lg.build_system_prompt(_new_save(style="light"))
    assert a != b
    assert lg.STYLE_HINTS["dark"] in a


def test_messages_map_history_roles():
    save = _new_save()
    save.turns = [
        {"role": "user", "content": "推开门", "ts": 1.0},
        {"role": "narrator", "content": "门后是一片黑暗。", "ts": 2.0},
    ]
    msgs = lg.build_messages(save, "点亮火折子")
    # 1 system + 2 history + 1 本轮
    assert len(msgs) == 4
    assert msgs[0].__class__.__name__ == "SystemMessage"
    # 玩家的历史输入要带主角名前缀，避免模型把它当旁白
    assert "李寻" in msgs[1].content and "推开门" in msgs[1].content
    assert msgs[2].content == "门后是一片黑暗。"
    assert "点亮火折子" in msgs[3].content


def test_messages_opening_has_no_action():
    save = _new_save()
    msgs = lg.build_messages(save, None)
    assert len(msgs) == 2      # system + 开场指令
    assert "游戏开始" in msgs[-1].content
    assert "不要替主角做决定" in msgs[-1].content


def test_messages_skip_empty_history_entries():
    save = _new_save()
    save.turns = [
        {"role": "user", "content": "  ", "ts": 1.0},
        {"role": "narrator", "content": "", "ts": 2.0},
        {"role": "unknown", "content": "不该出现", "ts": 3.0},
    ]
    msgs = lg.build_messages(save, "走")
    assert len(msgs) == 2      # 只有 system + 本轮


# ============================================================
# API
# ============================================================

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "LEGEND_DIR", tmp_path / "legends")
    # ★放开限流：本文件要连发十几个 create/act，而默认是 10 次/分钟。
    # 不放开的话，靠后的用例会莫名其妙地吃 429——表现成「状态补丁没生效」这种
    # 极难排查的假故障（真踩过）。限流本身另有专门用例覆盖，这里不需要它。
    import src.api.routes as routes_mod
    from src.core import config as cfg_mod
    routes_mod._RATE_STORE.clear()
    monkeypatch.setattr(cfg_mod.settings, "rate_limit_per_minute", 1000)
    monkeypatch.setattr(cfg_mod.settings, "rate_limit_per_day", 100000)
    # ★兜底抽取会打真网关，单测必须打掉。不打掉的话，「模型漏了状态块」的用例
    # 会真的发一次网络请求——慢、可能 429、且不可复现。
    async def _no_extract(save, narration):
        return None
    monkeypatch.setattr(routes_mod, "legend_extract_state_patch", _no_extract, raising=False)
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


GOOD = {
    "world": "大唐末年，长安城暗流涌动，江湖与庙堂的界限早已模糊。",
    "protagonist_name": "李寻",
    "protagonist_desc": "落魄游侠",
    "npcs": [{"name": "苏娘", "role": "酒肆老板娘", "persona": "消息灵通"}],
    "style": "classic",
}


def test_api_create_and_get_and_delete(client):
    r = client.post("/persona/legend/create", json=GOOD)
    assert r.status_code == 200, r.text
    lid = r.json()["id"]

    g = client.get(f"/persona/legend/{lid}")
    assert g.status_code == 200
    body = g.json()
    assert body["protagonist_name"] == "李寻"
    assert body["npcs"][0]["name"] == "苏娘"
    assert body["turns"] == []

    lst = client.get("/persona/legends").json()
    assert any(x["id"] == lid for x in lst["legends"])
    assert lst["max_npcs"] == ls.MAX_NPCS

    d = client.delete(f"/persona/legend/{lid}")
    assert d.status_code == 200
    assert client.get(f"/persona/legend/{lid}").status_code == 404


def test_api_create_rejects_short_world(client):
    bad = dict(GOOD, world="短")
    r = client.post("/persona/legend/create", json=bad)
    assert r.status_code == 400
    assert "10 个字" in r.json()["detail"]


def test_api_get_missing_404(client):
    assert client.get("/persona/legend/nope_123456").status_code == 404


def test_api_delete_missing_404(client):
    assert client.delete("/persona/legend/nope_123456").status_code == 404


def test_api_act_streams_events(client, monkeypatch):
    """act 必须依次推 start → token… → end，并把叙述写回存档。"""
    r = client.post("/persona/legend/create", json=GOOD)
    lid = r.json()["id"]

    async def fake_stream(save, action):
        for piece in ["门开了。", "**苏娘**：", "你来了。"]:
            yield piece

    monkeypatch.setattr(lg, "stream_turn", fake_stream)
    monkeypatch.setattr("src.api.routes.legend_stream_turn", fake_stream, raising=False)

    with client.stream("POST", f"/persona/legend/{lid}/act",
                       json={"action": "推开门"}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())

    assert '"type": "start"' in text
    assert '"type": "token"' in text
    assert '"type": "end"' in text
    assert "门开了。" in text

    # 存档里应有一条用户行动 + 一条叙述
    g = client.get(f"/persona/legend/{lid}").json()
    roles = [t["role"] for t in g["turns"]]
    assert roles == ["user", "narrator"]
    assert g["turns"][1]["content"].startswith("门开了。")
    assert g["turn_count"] == 1


def test_api_opening_does_not_count_as_turn(client, monkeypatch):
    """开局没有玩家行动，不该被计成一轮。"""
    r = client.post("/persona/legend/create", json=GOOD)
    lid = r.json()["id"]

    async def fake_stream(save, action):
        yield "长安城下起了雨。"

    monkeypatch.setattr("src.api.routes.legend_stream_turn", fake_stream, raising=False)

    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": ""}) as resp:
        text = "".join(resp.iter_text())
    assert '"opening": true' in text

    g = client.get(f"/persona/legend/{lid}").json()
    assert [t["role"] for t in g["turns"]] == ["narrator"]
    assert g["turn_count"] == 0


def test_api_act_rejects_overlong_action(client):
    r = client.post("/persona/legend/create", json=GOOD)
    lid = r.json()["id"]
    too_long = "啊" * (lg.MAX_ACTION_CHARS + 10)
    resp = client.post(f"/persona/legend/{lid}/act", json={"action": too_long})
    assert resp.status_code == 400
    assert "最多" in resp.json()["detail"]


def test_api_act_missing_save_404(client):
    resp = client.post("/persona/legend/nope_123456/act", json={"action": "走"})
    assert resp.status_code == 404


# ============================================================
# 首 token 超时（真链路实测踩出来的坑）
# ============================================================
# 全局 `llm_ttft_timeout=6.0` 的取值依据是 config 注释里写的「正常首 token 实测 1-3s」——
# 但那是 `deepseek-chat`（非推理模型）时代的测定。现在跑的是 `deepseek-v4-flash`，
# **推理模型，思考发生在首 token 之前**，首 token 天然要十几秒。
#
# 真链路实测（output/_legend_live.log）：
#   用默认 6s → 每轮触发 1-3 次「首 token 超时」重试，第 3 轮 3 次全超时返回空白
#   改成 20s → 首 token 超时 0 次，三轮全成功（8.9s / 15.8s / 16.6s）
#
# 传奇比别的路径更需要放宽：它是**纯生成**，前面没有检索/重排占用时间，
# 模型没有「热身」过程，首 token 来得最晚，被 6s 误杀的概率最高。


def test_stream_turn_passes_extended_ttft_timeout(monkeypatch):
    """★必须显式传放宽后的 ttft_timeout，不能吃全局默认 6s。"""
    captured = {}

    async def fake_astream(llm, messages, **kw):
        captured.update(kw)
        yield "内容"

    monkeypatch.setattr(lg, "astream_nonempty", fake_astream)
    monkeypatch.setattr(lg, "get_chat_llm", lambda **kw: object())

    async def run():
        out = []
        async for t in lg.stream_turn(_new_save(), "走"):
            out.append(t)
        return out

    import asyncio
    got = asyncio.run(run())
    assert got == ["内容"]
    assert "ttft_timeout" in captured, "没传 ttft_timeout 就会吃全局 6s 默认值"
    assert captured["ttft_timeout"] == lg.LEGEND_TTFT_TIMEOUT
    assert lg.LEGEND_TTFT_TIMEOUT >= 15.0, \
        "推理模型的思考时间在 10-15s 量级，给低于 15s 会重现「每轮都重试」"


# ============================================================
# 状态栏：存储层
# ============================================================
# 结构：state_fields/only_fields = 作者定死的字段定义；state/only = 模型每轮改的值。
# state（通用）在场每个人物各一份；only（仅主角）只有主角有。

STATE_FIELDS = [
    {"name": "好感度", "kind": "number", "init": 0, "desc": "对主角的信任"},
    {"name": "态度", "kind": "tag", "init": "中立"},
]
ONLY_FIELDS = [{"name": "体力", "kind": "number", "init": 100}]


def _state_save(**kw):
    kw.setdefault("state_fields", [dict(f) for f in STATE_FIELDS])
    kw.setdefault("only_fields", [dict(f) for f in ONLY_FIELDS])
    return _new_save(**kw)


def test_reset_state_gives_every_character_a_copy(store):
    """★通用字段是「每个人物各一份」——主角和配角都要有自己的一份初值。"""
    save = _state_save()
    ls.reset_state(save)
    assert set(save.state) == {"李寻", "苏娘"}
    assert save.state["苏娘"] == {"好感度": 0, "态度": "中立"}
    assert save.state["李寻"] == {"好感度": 0, "态度": "中立"}
    # 仅主角的字段只有一份，不按人物分
    assert save.only == {"体力": 100}
    assert save.has_state() is True


def test_no_fields_means_no_state(store):
    """没定义字段的档（含老档）必须完全走原路径，state 保持空。"""
    save = _new_save()
    ls.reset_state(save)
    assert save.state == {} and save.only == {}
    assert save.has_state() is False


def test_patch_whitelist_drops_unknown_character_and_field(store):
    """★补丁是模型生成的文本，必须有白名单：不在场的人、没定义的字段一律不落。"""
    save = _state_save()
    ls.reset_state(save)
    changed = ls.apply_state_patch(save, {
        "state": {"苏娘": {"好感度": 15, "心情": "很好"},   # 心情 未定义
                  "龙王": {"好感度": 99}},                  # 龙王 不在场
        "only": {"体力": 87, "修为": 3},                    # 修为 未定义
    })
    assert save.state["苏娘"]["好感度"] == 15
    assert "心情" not in save.state["苏娘"]
    assert "龙王" not in save.state
    assert save.only == {"体力": 87}
    assert changed["state"] == {"苏娘": {"好感度": 15}}
    assert changed["only"] == {"体力": 87}


def test_patch_clamps_and_ignores_garbage(store):
    """类型钳制：数值限幅、词条截断、非法数值忽略（不能把玩家数值抹成 0）。"""
    save = _state_save()
    ls.reset_state(save)
    save.state["苏娘"]["好感度"] = 50
    ls.apply_state_patch(save, {
        "state": {"苏娘": {"好感度": "未知", "态度": "x" * 80}},
        "only": {"体力": 1e12},
    })
    assert save.state["苏娘"]["好感度"] == 50, "非数字应被忽略而不是清零"
    assert len(save.state["苏娘"]["态度"]) == ls.MAX_TAG_CHARS
    assert save.only["体力"] == ls.STATE_NUMBER_ABS_MAX


def test_patch_accepts_flat_form_as_protagonist(store):
    """容错：模型偷懒直接给字段名（不按人物分组）时，算作主角的。"""
    save = _state_save()
    ls.reset_state(save)
    ls.apply_state_patch(save, {"state": {"好感度": 7}})
    assert save.state["李寻"]["好感度"] == 7
    assert save.state["苏娘"]["好感度"] == 0


def test_sync_state_backfills_and_drops(store):
    """字段/人物变了要能对齐：补齐缺失、丢掉已不存在的。"""
    save = _state_save()
    ls.reset_state(save)
    save.state["苏娘"]["好感度"] = 30
    save.state["已删除的人"] = {"好感度": 1}
    save.state["苏娘"]["旧字段"] = "值"
    save.only["已删除字段"] = 1
    ls.sync_state(save)
    assert set(save.state) == {"李寻", "苏娘"}
    assert save.state["苏娘"]["好感度"] == 30, "已有值不能被覆盖"
    assert "旧字段" not in save.state["苏娘"]
    assert set(save.only) == {"体力"}


def test_old_save_without_state_fields_loads_clean(store):
    """老档升级：文件里没有状态栏字段时，读出来是空而不是崩。"""
    import json
    d = ls.legend_dir()
    (d / "old_888888.json").write_text(json.dumps({
        "id": "old_888888", "title": "旧档", "world": "旧世界",
        "protagonist_name": "旧人",
    }, ensure_ascii=False), encoding="utf-8")
    got = ls.load_legend("old_888888")
    assert got is not None
    assert got.state_fields == [] and got.only_fields == []
    assert got.state == {} and got.only == {}
    assert got.has_state() is False


def test_normalize_fields_drops_illegal_names(store):
    """字段名带空格/标点会把 JSON 补丁带歪，规范化时直接丢掉。"""
    out = ls.normalize_fields([
        {"name": "好感度", "kind": "number", "init": "3"},
        {"name": "带 空格"},
        {"name": "带:冒号"},
        {"name": "好感度"},           # 重名
        {"name": "怪类型", "kind": "乱写"},
    ])
    assert [f["name"] for f in out] == ["好感度", "怪类型"]
    assert out[0]["init"] == 3.0
    assert out[1]["kind"] == "number", "非法 kind 要回落到 number"


def test_validate_checks_state_fields():
    world = "这是一个足够长的世界观设定文本内容。"
    ok, why = ls.validate_new_save(world=world, protagonist_name="甲",
                                   state_fields=[{"name": "带 空格"}])
    assert not ok and "空格" in why

    ok, why = ls.validate_new_save(
        world=world, protagonist_name="甲",
        state_fields=[{"name": f"字段{i}"} for i in range(ls.MAX_STATE_FIELDS + 1)])
    assert not ok and str(ls.MAX_STATE_FIELDS) in why

    ok, why = ls.validate_new_save(world=world, protagonist_name="甲",
                                   state_fields=[{"name": "好感度"}, {"name": "好感度"}])
    assert not ok and "重名" in why


# ============================================================
# 状态栏：引擎层（prompt + 流过滤）
# ============================================================

def test_system_prompt_contains_state_contract():
    p = lg.build_system_prompt(_state_save())
    assert "## 状态栏" in p
    assert "好感度" in p and "态度" in p and "体力" in p
    assert lg.STATE_OPEN in p and lg.STATE_CLOSE in p
    # 定义、当前值、输出契约三块都要在
    assert "每个人物各有一份" in p
    assert "当前值" in p
    # ★契约措辞：要「写涉及到的字段的值」而不是「只写变化」——
    # 前者宽容、可重复写（合并时按相等去重），后者会让模型保守到什么都不写。
    assert "重复写没有代价，漏写才有代价" in p
    # 样例里必须出现真实人物名，模型才有得照抄
    assert "苏娘" in p


def test_system_prompt_has_no_state_section_without_fields():
    """★没有定义字段时，prompt 必须与上线前一致——不能给老档悄悄加指令。"""
    p = lg.build_system_prompt(_new_save())
    assert "## 状态栏" not in p
    assert lg.STATE_OPEN not in p
    assert lg.build_state_block(_new_save()) == ""
    # 铁律里的第 4 条（状态块）也不能出现
    assert "漏掉即为不完整" not in p


def test_state_rule_added_to_iron_rules():
    p = lg.build_system_prompt(_state_save())
    assert "漏掉即为不完整" in p
    assert "4. 每次回复的**最后**必须附上状态块" in p


def test_state_reminder_reaches_every_turn():
    """★真链路实测：契约只写在 system 段时，3 轮只有 1 轮吐块。
    所以最后一轮完整轮次里必须再提醒一次——离生成点越近越管用。"""
    save = _state_save()
    for milestone in (None, "推门进去"):
        msgs = lg.build_messages(save, milestone)
        assert lg.STATE_OPEN in msgs[-1].content, "本轮消息里没有状态块提醒"
        assert lg.STATE_CLOSE in msgs[-1].content
    # 没有状态栏的档不能多出这段
    assert lg.STATE_OPEN not in lg.build_messages(_new_save(), "走")[-1].content


def test_state_block_lists_only_protagonist_fields_separately():
    p = lg.build_system_prompt(_state_save())
    assert "仅主角（只有 李寻 有）" in p
    assert "李寻（专属）：体力 100" in p


def test_state_filter_strips_block_across_chunk_boundaries():
    """★块标记会跨 token 边界，必须扣住不吐——否则玩家会看到半截标记。"""
    f = lg.StatePatchFilter()
    full = ('门开了。\n**苏娘**：你来了。\n'
            + lg.STATE_OPEN + '\n{"state": {"苏娘": {"好感度": 15}}, "only": {"体力": 87}}\n'
            + lg.STATE_CLOSE)
    visible = ""
    for i in range(0, len(full), 5):      # 故意用 5 字一段切碎
        visible += f.feed(full[i:i + 5])
    visible += f.finish()
    assert visible == "门开了。\n**苏娘**：你来了。\n"
    assert "STATE" not in visible
    assert f.patch() == {"state": {"苏娘": {"好感度": 15}}, "only": {"体力": 87}}


def test_state_filter_restores_unclosed_block():
    """★块没闭合（模型没按格式写）时，吃进去的内容要原样还给玩家，不能吞掉。"""
    f = lg.StatePatchFilter()
    visible = f.feed("他说。" + lg.STATE_OPEN + ' {"x": 1')
    visible += f.finish()
    assert visible == "他说。" + lg.STATE_OPEN + ' {"x": 1'
    assert f.blocks == []


def test_state_filter_swallows_block_in_the_middle():
    """块写在中间也要能摘掉，并且块之后的正文继续吐出来。"""
    f = lg.StatePatchFilter()
    visible = f.feed("前段。" + lg.STATE_OPEN + '{"only": {"体力": 50}}' + lg.STATE_CLOSE + "后段。")
    visible += f.finish()
    assert visible == "前段。后段。"
    assert f.patch() == {"only": {"体力": 50}}


def test_parse_state_patch_tolerates_fences_and_noise():
    assert lg.parse_state_patch('```json\n{"state": {}}\n```') == {"state": {}}
    assert lg.parse_state_patch('变化如下：{"only": {"体力": 80}} 以上。') == {"only": {"体力": 80}}
    assert lg.parse_state_patch("完全不是 JSON") is None
    assert lg.parse_state_patch("") is None
    assert lg.parse_state_patch("[1, 2]") is None


def test_stream_filter_plain_text_passes_through_unchanged():
    f = lg.StatePatchFilter()
    out = f.feed("苏娘愣了一下，") + f.feed("把酒盏推了过来。") + f.finish()
    assert out == "苏娘愣了一下，把酒盏推了过来。"
    assert f.patch() is None


# ---- 兜底抽取（正文没带状态块时） ----
# ★真链路实测模型不会每轮都吐块（跑三遍 1/3、3/3、1/3），状态栏不能靠 prompt 赌，
# 所以正文没带块时要补一次极小的结构化调用。

def test_extract_state_patch_disables_reasoning(monkeypatch):
    """★结构化抽取必须关思考：推理模型的 thinking 会把 JSON 的额度吃掉导致截断。"""
    captured = {}

    def fake_get_chat_llm(**kw):
        captured.update(kw)
        return object()

    async def fake_ainvoke(llm, messages, **kw):
        class R:
            content = '{"state": {"苏娘": {"好感度": 3}}}'
        return R()

    monkeypatch.setattr(lg, "get_chat_llm", fake_get_chat_llm)
    monkeypatch.setattr(lg, "ainvoke_nonempty", fake_ainvoke)

    got = asyncio.run(lg.extract_state_patch(_state_save(), "苏娘点了点头。"))
    assert got == {"state": {"苏娘": {"好感度": 3}}}
    assert captured.get("reasoning_effort") == "none"
    assert captured.get("with_fallback") is True


def test_extract_state_patch_skips_when_nothing_to_do(monkeypatch):
    """没有字段 / 叙述为空时不该发起调用。"""
    def boom(**kw):
        raise AssertionError("这种情况不该调模型")

    monkeypatch.setattr(lg, "get_chat_llm", boom)
    assert asyncio.run(lg.extract_state_patch(_new_save(), "有叙述")) is None
    assert asyncio.run(lg.extract_state_patch(_state_save(), "   ")) is None


def test_extract_state_patch_swallows_llm_failure(monkeypatch):
    """抽取失败不能把异常抛给调用方——状态不更新而已，这轮对话不该失败。"""
    def boom(**kw):
        raise RuntimeError("没配 key")

    monkeypatch.setattr(lg, "get_chat_llm", boom)
    assert asyncio.run(lg.extract_state_patch(_state_save(), "叙述")) is None


# ============================================================
# 状态栏：API 层
# ============================================================

GOOD_STATE = dict(
    GOOD,
    state_fields=STATE_FIELDS,
    only_fields=ONLY_FIELDS,
)


def test_api_create_initializes_state_per_character(client):
    r = client.post("/persona/legend/create", json=GOOD_STATE)
    assert r.status_code == 200, r.text
    lid = r.json()["id"]

    body = client.get(f"/persona/legend/{lid}").json()
    assert [f["name"] for f in body["state_fields"]] == ["好感度", "态度"]
    assert [f["name"] for f in body["only_fields"]] == ["体力"]
    assert body["state"]["李寻"] == {"好感度": 0, "态度": "中立"}
    assert body["state"]["苏娘"] == {"好感度": 0, "态度": "中立"}
    assert body["only"] == {"体力": 100}


def test_api_list_exposes_field_limits(client):
    d = client.get("/persona/legends").json()
    assert d["max_state_fields"] == ls.MAX_STATE_FIELDS
    assert d["max_only_fields"] == ls.MAX_ONLY_FIELDS


def test_api_create_rejects_too_many_state_fields(client):
    bad = dict(GOOD, state_fields=[{"name": f"字段{i}"} for i in range(ls.MAX_STATE_FIELDS + 1)])
    r = client.post("/persona/legend/create", json=bad)
    assert r.status_code == 400
    assert str(ls.MAX_STATE_FIELDS) in r.json()["detail"]


def test_api_act_applies_state_patch_and_hides_block(client, monkeypatch):
    """★一轮走完：叙述里不能残留状态块，值要落进存档，end 事件要带回新状态。"""
    lid = client.post("/persona/legend/create", json=GOOD_STATE).json()["id"]

    async def fake_stream(save, action):
        yield "苏娘的眼神变了。\n"
        yield lg.STATE_OPEN + "\n"
        yield '{"state": {"苏娘": {"好感度": 15}}, "only": {"体力": 73}}'
        yield "\n" + lg.STATE_CLOSE

    monkeypatch.setattr("src.api.routes.legend_stream_turn", fake_stream, raising=False)

    with client.stream("POST", f"/persona/legend/{lid}/act",
                       json={"action": "我把信推给她"}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())

    assert "STATE" not in text, "状态块泄漏到了用户可见的流里"
    assert "苏娘的眼神变了。" in text
    assert '"type": "end"' in text

    body = client.get(f"/persona/legend/{lid}").json()
    assert body["state"]["苏娘"]["好感度"] == 15
    assert body["state"]["李寻"]["好感度"] == 0, "没提到的人不该被动过"
    assert body["only"]["体力"] == 73
    # 写进历史的叙述里也不能有块
    assert "STATE" not in body["turns"][-1]["content"]


def test_api_act_without_state_block_keeps_values(client, monkeypatch):
    """模型这轮漏了状态块 → 值原样保留，不能清零也不能报错。"""
    lid = client.post("/persona/legend/create", json=GOOD_STATE).json()["id"]

    async def t1(save, action):
        yield "第一次。\n" + lg.STATE_OPEN + '{"only": {"体力": 60}}' + lg.STATE_CLOSE

    monkeypatch.setattr("src.api.routes.legend_stream_turn", t1, raising=False)
    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": "走"}) as resp:
        "".join(resp.iter_text())

    async def t2(save, action):
        yield "第二次，什么都没有发生。"

    monkeypatch.setattr("src.api.routes.legend_stream_turn", t2, raising=False)
    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": "再看"}) as resp:
        "".join(resp.iter_text())

    body = client.get(f"/persona/legend/{lid}").json()
    assert body["only"]["体力"] == 60
    assert body["turn_count"] == 2


def test_api_act_drops_out_of_whitelist_patch(client, monkeypatch):
    """模型胡写人名/字段名，一个字都不该进存档。"""
    lid = client.post("/persona/legend/create", json=GOOD_STATE).json()["id"]

    async def fake_stream(save, action):
        yield "叙述。\n" + lg.STATE_OPEN + \
            '{"state": {"查无此人": {"好感度": 99}}, "only": {"体力": 1e12, "乱写": 5}}' + \
            lg.STATE_CLOSE

    monkeypatch.setattr("src.api.routes.legend_stream_turn", fake_stream, raising=False)
    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": "走"}) as resp:
        "".join(resp.iter_text())

    body = client.get(f"/persona/legend/{lid}").json()
    assert set(body["state"]) == {"李寻", "苏娘"}
    assert body["only"] == {"体力": ls.STATE_NUMBER_ABS_MAX}


def test_api_act_falls_back_to_extraction_when_block_missing(client, monkeypatch):
    """★模型这轮没吐块 → 走兜底抽取，状态栏不能就这么停摆。"""
    import src.api.routes as routes_mod
    lid = client.post("/persona/legend/create", json=GOOD_STATE).json()["id"]

    async def no_block(save, action):
        yield "苏娘笑了一下，把酒推过来。"

    seen = {}

    async def fake_extract(save, narration):
        seen["narration"] = narration
        return {"state": {"苏娘": {"好感度": 9}}, "only": {"体力": 88}}

    monkeypatch.setattr("src.api.routes.legend_stream_turn", no_block, raising=False)
    monkeypatch.setattr(routes_mod, "legend_extract_state_patch", fake_extract)

    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": "坐下"}) as resp:
        text = "".join(resp.iter_text())

    assert '"state_source": "extract"' in text
    assert "苏娘笑了一下" in seen.get("narration", ""), "应把清洗后的叙述喂给抽取"
    body = client.get(f"/persona/legend/{lid}").json()
    assert body["state"]["苏娘"]["好感度"] == 9
    assert body["only"]["体力"] == 88


def test_api_act_marks_block_as_source_when_present(client, monkeypatch):
    """正文带了块就走快路径，不该再打一次抽取。"""
    import src.api.routes as routes_mod
    lid = client.post("/persona/legend/create", json=GOOD_STATE).json()["id"]

    async def with_block(save, action):
        yield "苏娘点头。\n" + lg.STATE_OPEN + '{"only": {"体力": 70}}' + lg.STATE_CLOSE

    async def should_not_run(save, narration):
        raise AssertionError("带了块就不该再抽一次")

    monkeypatch.setattr("src.api.routes.legend_stream_turn", with_block, raising=False)
    monkeypatch.setattr(routes_mod, "legend_extract_state_patch", should_not_run)

    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": "走"}) as resp:
        text = "".join(resp.iter_text())
    assert '"state_source": "block"' in text
    assert client.get(f"/persona/legend/{lid}").json()["only"]["体力"] == 70


def test_api_act_extraction_failure_does_not_break_turn(client, monkeypatch):
    """兜底抽取炸了也不能让这一轮失败——状态不动而已。"""
    import src.api.routes as routes_mod
    lid = client.post("/persona/legend/create", json=GOOD_STATE).json()["id"]

    async def no_block(save, action):
        yield "什么都没发生。"

    async def boom(save, narration):
        raise RuntimeError("上游炸了")

    monkeypatch.setattr("src.api.routes.legend_stream_turn", no_block, raising=False)
    monkeypatch.setattr(routes_mod, "legend_extract_state_patch", boom)

    with client.stream("POST", f"/persona/legend/{lid}/act", json={"action": "等"}) as resp:
        text = "".join(resp.iter_text())

    assert '"type": "end"' in text
    body = client.get(f"/persona/legend/{lid}").json()
    assert body["turn_count"] == 1
    assert body["only"]["体力"] == 100, "抽取失败应保持原值"
