"""
管理后台 + 意见反馈 测试：
- FeedbackStore 存储层（record / list / set_status / count_by_status）
- 后台 API（反馈提交免 token / 读取需 token / dashboard 聚合 / /admin 页面）
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.core.feedback as fb_mod
from src.core import config as cfg


@pytest.fixture
def client(tmp_path, monkeypatch):
    # 反馈库指向临时文件，避免污染项目 data/
    monkeypatch.setattr(cfg.settings, "feedback_db_path", str(tmp_path / "feedback.db"))
    monkeypatch.setattr(cfg.settings, "monitor_token", "secret")
    # 重置懒加载单例，确保下次调用读取新的路径配置
    fb_mod._store = None

    from src.api.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ============ 存储层 ============

def test_feedback_store_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.settings, "feedback_db_path", str(tmp_path / "fb.db"))
    fb_mod._store = None
    store = fb_mod.get_feedback_store()

    # 空内容拒绝
    with pytest.raises(ValueError):
        store.record("")

    r1 = store.record("不好用", character="jung", contact="qq123", page="/")
    assert r1["id"] >= 1 and r1["status"] == "pending"

    r2 = store.record("很赞", character="adler")
    assert r2["id"] != r1["id"]

    # 列表倒序
    items = store.list()
    assert items[0]["id"] == r2["id"]

    # 状态过滤
    pending = store.list(status="pending")
    assert len(pending) == 2

    # 标记已回复
    updated = store.set_status(r1["id"], "replied", reply="已修复")
    assert updated["status"] == "replied" and updated["reply"] == "已修复"

    # 非法状态
    with pytest.raises(ValueError):
        store.set_status(r1["id"], "bogus")

    # 不存在的 id
    assert store.set_status(999999, "resolved") is None

    # 计数
    counts = store.count_by_status()
    assert counts.get("replied") == 1 and counts.get("pending") == 1


# ============ 后台 API ============

def test_submit_feedback_no_token(client):
    """提交反馈不需要 token"""
    resp = client.post("/admin/feedback", json={"content": "界面卡顿"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["id"] >= 1


def test_submit_feedback_empty_rejected(client):
    resp = client.post("/admin/feedback", json={"content": "   "})
    assert resp.status_code == 400


def test_list_feedback_requires_token(client):
    client.post("/admin/feedback", json={"content": "意见1"})
    # 无 token -> 403
    assert client.get("/admin/feedback").status_code == 403
    # 错误 token -> 403
    assert client.get("/admin/feedback", params={"token": "wrong"}).status_code == 403
    # 正确 token -> 200 且含刚提交
    resp = client.get("/admin/feedback", params={"token": "secret"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(i["content"] == "意见1" for i in items)
    assert resp.json()["counts"].get("pending", 0) >= 1


def test_set_feedback_status(client):
    r = client.post("/admin/feedback", json={"content": "建议", "contact": "wx9"})
    fid = r.json()["id"]
    resp = client.post(
        f"/admin/feedback/{fid}/status",
        params={"token": "secret"},
        json={"status": "resolved"},
    )
    assert resp.status_code == 200
    assert resp.json()["feedback"]["status"] == "resolved"

    # 不存在
    assert client.post(
        "/admin/feedback/999999/status", params={"token": "secret"}, json={"status": "resolved"}
    ).status_code == 404


def test_dashboard_aggregates(client):
    client.post("/admin/feedback", json={"content": "聚合测试"})
    # 无 token -> 403
    assert client.get("/admin/dashboard").status_code == 403
    resp = client.get("/admin/dashboard", params={"token": "secret"})
    assert resp.status_code == 200
    d = resp.json()
    # 核心聚合字段齐全
    for key in ("usage", "sessions", "custom_characters", "graphs", "feedback"):
        assert key in d, f"dashboard 缺少 {key}"
    # usage 含今日/本周/总计
    assert set(d["usage"].keys()) >= {"today", "week", "all"}
    # feedback 概览
    assert "counts" in d["feedback"] and "recent" in d["feedback"]


def test_admin_page_served(client):
    """管理后台页面返回 HTML（需 token 才能进，但页面本身可直接访问）"""
    resp = client.get("/admin", params={"token": "secret"})
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "管理后台" in resp.text
