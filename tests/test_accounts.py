"""
账号系统测试：
- 存储层：pbkdf2 密码哈希/校验、账号与密码校验规则、注册/登录/会话生命周期
- API：注册自动登录（Cookie）/ 登录 / me / 登出 / 重复注册 / 错误密码 / 未登录 me 为空
- 限流主体：登录用户按账号限流（auth 相关端点独立分桶）
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.core.accounts as accounts
from src.core import config as cfg


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    """账号库指向临时文件 + 重置连接单例，避免污染项目 data/。"""
    monkeypatch.setattr(cfg.settings, "accounts_db_path", str(tmp_path / "accounts.db"))
    accounts._CONN = None
    yield
    accounts._CONN = None


@pytest.fixture
def client(auth_env, monkeypatch):
    from src.api.routes import router
    from src.api import routes as routes_mod
    from src.core import config as cfg_mod

    # 限流是模块级全局桶，跨测试累积会误伤；默认放大限额，
    # 只有专门测限流的用例自己去覆盖 settings
    routes_mod._RATE_STORE.clear()
    monkeypatch.setattr(cfg_mod.settings, "rate_limit_per_minute", 1000)
    monkeypatch.setattr(cfg_mod.settings, "rate_limit_per_day", 1000)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ============ 存储层 ============

def test_password_hash_roundtrip():
    h = accounts.hash_password("s3cret-pw!")
    assert h.startswith("pbkdf2$")
    assert accounts.verify_password("s3cret-pw!", h)
    assert not accounts.verify_password("wrong", h)


def test_password_hash_salts_unique():
    assert accounts.hash_password("same") != accounts.hash_password("same")


def test_validate_account_rules():
    assert accounts.validate_account("user@x.com") is None
    assert accounts.validate_account("ab_cd-12") is None
    assert accounts.validate_account("abc") is not None        # 太短
    assert accounts.validate_account("") is not None
    assert accounts.validate_account("有中文字符账号哦") is not None


def test_validate_password_rules():
    assert accounts.validate_password("12345678") is None
    assert accounts.validate_password("1234567") is not None   # 太短
    assert accounts.validate_password("") is not None


def test_create_and_authenticate(auth_env):
    accounts.create_user("Alice@x.com", "password1", "爱丽丝")
    user = accounts.get_user_by_account("alice@x.com")  # 大小写归一
    assert user and user["display_name"] == "爱丽丝"
    assert accounts.authenticate("Alice@x.com", "password1")["id"] == user["id"]
    with pytest.raises(ValueError):
        accounts.authenticate("alice@x.com", "wrong-password")
    with pytest.raises(ValueError):
        accounts.authenticate("nobody@x.com", "password1")     # 不泄露账号是否存在


def test_duplicate_account_rejected(auth_env):
    accounts.create_user("dup@x.com", "password1")
    with pytest.raises(ValueError, match="已被注册"):
        accounts.create_user("dup@x.com", "password2")


def test_session_lifecycle(auth_env):
    user = accounts.create_user("sess@x.com", "password1")
    token = accounts.create_session(user["id"])
    got = accounts.get_user_by_session(token)
    assert got and got["account"] == "sess@x.com"
    accounts.delete_session(token)
    assert accounts.get_user_by_session(token) is None
    assert accounts.get_user_by_session("forged-token") is None


# ============ API ============

def test_register_login_me_logout_flow(client):
    r = client.post("/auth/register", json={"account": "boot@x.com", "password": "password1", "display_name": "小明"})
    assert r.status_code == 200
    assert "gkrm_session" in r.cookies
    assert r.json()["user"]["display_name"] == "小明"

    r = client.get("/auth/me")
    assert r.status_code == 200 and r.json()["user"]["account"] == "boot@x.com"

    # 登出后再查应为空
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").json()["user"] is None

    # 重新登录
    r = client.post("/auth/login", json={"account": "boot@x.com", "password": "password1"})
    assert r.status_code == 200
    assert client.get("/auth/me").json()["user"] is not None


def test_register_validation_and_duplicate(client):
    assert client.post("/auth/register", json={"account": "x", "password": "password1"}).status_code == 400
    assert client.post("/auth/register", json={"account": "ok-account", "password": "short"}).status_code == 400
    assert client.post("/auth/register", json={"account": "ok-account", "password": "password1"}).status_code == 200
    r = client.post("/auth/register", json={"account": "ok-account", "password": "password1"})
    assert r.status_code == 400 and "已被注册" in r.json()["detail"]


def test_login_wrong_password_401(client):
    client.post("/auth/register", json={"account": "pw@x.com", "password": "password1"})
    r = client.post("/auth/login", json={"account": "pw@x.com", "password": "wrong-pass"})
    assert r.status_code == 401


def test_anonymous_me_is_null(client):
    assert client.get("/auth/me").json()["user"] is None


# ============ 账号注销（隐私政策"删除权"） ============

def test_delete_account_removes_user_and_sessions(auth_env):
    user = accounts.create_user("del@x.com", "password1")
    token = accounts.create_session(user["id"])
    assert accounts.delete_account(user["id"]) is True
    # 账号与会话一并消失，未登录态可重新注册同一账号
    assert accounts.get_user_by_account("del@x.com") is None
    assert accounts.get_user_by_session(token) is None
    assert accounts.delete_account(user["id"]) is False  # 幂等


def test_delete_account_endpoint_requires_login(client):
    assert client.delete("/auth/account").status_code == 401


def test_delete_account_endpoint_flow(client):
    client.post("/auth/register", json={"account": "bye@x.com", "password": "password1"})
    r = client.delete("/auth/account")
    assert r.status_code == 200 and r.json()["deleted"] is True
    # Cookie 里的会话已被连带删除
    assert client.get("/auth/me").json()["user"] is None
    # 同一账号可再次注册（数据已删干净）
    assert client.post("/auth/register", json={"account": "bye@x.com", "password": "password1"}).status_code == 200


# ============ 法务页面 ============

def test_terms_and_privacy_pages_served(client):
    r = client.get("/terms")
    assert r.status_code == 200 and "用户协议" in r.text
    r = client.get("/privacy")
    assert r.status_code == 200 and "隐私政策" in r.text
    assert "AI 生成" in client.get("/terms").text  # 内容性质提示

def test_query_rate_limit_uses_account_bucket(client, monkeypatch):
    """登录用户达到限额时按账号拒绝；匿名（另一 IP）不受该账号额度影响。"""
    from src.api import routes as routes_mod
    from src.core import config as cfg_mod

    # 限到极低便于触发
    monkeypatch.setattr(cfg_mod.settings, "rate_limit_per_minute", 1)
    routes_mod._RATE_STORE.clear()

    client.post("/auth/register", json={"account": "quota@x.com", "password": "password1"})
    headers = {"X-Forwarded-For": "10.9.9.9"}  # 固定 IP，确认按账号而非 IP 限流
    r1 = client.post("/persona/query", json={"query": "你好", "character_id": "jung"}, headers=headers)
    r2 = client.post("/persona/query", json={"query": "你好", "character_id": "jung"}, headers=headers)
    assert r1.status_code == 200
    assert r2.status_code == 429

    # 同一 IP 的匿名用户不受该账号额度影响（账号桶与 IP 桶隔离）
    client.cookies.delete("gkrm_session")
    r3 = client.post("/persona/query", json={"query": "你好", "character_id": "jung"}, headers=headers)
    assert r3.status_code == 200
