"""「传奇」剧情模式的测试（全部打桩，零网络）。

覆盖三层：
  1. 存储层 legend_store：读写/删除/列表/轮次裁剪/字段兼容/校验
  2. 引擎 framework.legend：system prompt 装配、messages 轮次映射、开局与行动的差别
  3. API：创建校验、读档、删档、以及 act 的 SSE 事件序列（打桩 LLM，不打真网关）

★重点验「不替主角做决定」这条铁律真的写进 prompt 了。
传奇的定义性特征就是「用户扮演主角」，模型一旦替主角说话/做决定，
这个玩法就塌了——它比任何功能点都该被测试锁住。
"""

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
