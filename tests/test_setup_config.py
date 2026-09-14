"""
设置页（开源版：用户自己填 LLM API Key）测试：

- runtime_config 存储层：优先级、空值不覆盖、原子写入、损坏 JSON 容错、脱敏
- setup API：状态查询 / 保存 / 连通性测试 / 写入鉴权 / 不泄露明文 Key
- LLM 工厂：未配置时抛人话异常而不是 401

落盘路径全部指向 tmp_path，不碰真实的 data/user_config.json。
连通性测试用假的 ChatOpenAI，不发真实网络请求。
"""
import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.core import runtime_config as rc
from src.core import config as cfg


@pytest.fixture
def client(tmp_path, monkeypatch):
    """只挂 setup 路由的最小应用，避免触发 main 的重型 lifespan 预热。"""
    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "user_config.json")
    rc.load(force=True)

    from src.api.setup import router as setup_router

    app = FastAPI()
    app.include_router(setup_router)
    return TestClient(app)


@pytest.fixture
def no_env_key(monkeypatch):
    """模拟"克隆下来什么都没配"的状态。"""
    monkeypatch.setattr(cfg.settings, "llm_api_key", "")


@pytest.fixture
def with_env_key(monkeypatch):
    monkeypatch.setattr(cfg.settings, "llm_api_key", "sk-env-key-1234567890")


# ============ 存储层 ============

def test_user_config_overrides_env(tmp_path, monkeypatch, with_env_key):
    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "uc.json")
    rc.load(force=True)

    cfg_before = rc.resolve(cfg.settings)
    assert cfg_before["source"] == "env"

    rc.save({"base_url": "https://api.deepseek.com", "api_key": "sk-user-key-abcdef", "model": "deepseek-chat"})
    cfg_after = rc.resolve(cfg.settings)
    assert cfg_after["source"] == "user"
    assert cfg_after["api_key"] == "sk-user-key-abcdef"
    assert cfg_after["model"] == "deepseek-chat"


def test_empty_value_does_not_wipe_existing(tmp_path, monkeypatch, with_env_key):
    """只改模型名时 Key 必须保留——否则用户每次改模型都要重新粘贴 Key。"""
    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "uc.json")
    rc.load(force=True)
    rc.save({"api_key": "sk-keep-me-123456", "model": "deepseek-chat"})

    rc.save({"model": "deepseek-reasoner"})
    cfg_now = rc.resolve(cfg.settings)
    assert cfg_now["api_key"] == "sk-keep-me-123456"
    assert cfg_now["model"] == "deepseek-reasoner"


def test_corrupt_json_is_treated_as_unconfigured(tmp_path, monkeypatch, no_env_key):
    """用户手改坏 JSON 是开源项目常态，配置读取绝不该让服务起不来。"""
    path = tmp_path / "uc.json"
    path.write_text("{ 这不是合法 json", encoding="utf-8")
    monkeypatch.setattr(rc, "CONFIG_PATH", path)
    rc.load(force=True)

    assert rc.load() == {}
    assert rc.is_configured(cfg.settings) is False


def test_save_creates_parent_dir_and_is_atomic(tmp_path, monkeypatch):
    nested = tmp_path / "data" / "uc.json"
    monkeypatch.setattr(rc, "CONFIG_PATH", nested)
    rc.load(force=True)

    rc.save({"api_key": "sk-nested-12345678"})
    assert nested.exists()
    # 原子写入的临时文件不应残留
    assert not (nested.parent / (nested.name + ".tmp")).exists()
    assert json.loads(nested.read_text(encoding="utf-8"))["api_key"] == "sk-nested-12345678"


def test_mask_hides_middle():
    assert rc.mask("") == ""
    # 注意：这里只允许用合成 Key，禁止贴真实 Key（否则 mask 测试会变成泄漏点）
    assert rc.mask("sk-fake0123456789abcdefghijklmnop") == "sk-fak********mnop"
    # 短 Key 只留前两位，避免"脱敏后反而暴露全部"
    assert "*" in rc.mask("sk-123")


# ============ 状态接口 ============

def test_status_unconfigured(client, no_env_key):
    body = client.get("/setup/status").json()
    assert body["configured"] is False
    assert body["source"] == "none"
    assert body["api_key_masked"] == ""
    assert len(body["presets"]) >= 3


def test_status_reports_env_source(client, with_env_key):
    body = client.get("/setup/status").json()
    assert body["configured"] is True
    assert body["source"] == "env"


def test_status_never_leaks_plaintext_key(client, with_env_key):
    raw = cfg.settings.llm_api_key
    resp = client.get("/setup/status")
    assert raw not in resp.text
    assert "sk-env-key-1234567890" not in resp.text


# ============ 保存接口 ============

def test_save_then_status_roundtrip(client, no_env_key):
    assert client.get("/setup/status").json()["configured"] is False

    resp = client.post("/setup/llm", json={
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-user-filled-987654",
        "model": "deepseek-chat",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is True
    assert body["source"] == "user"
    assert body["model"] == "deepseek-chat"
    # 回传的必须是脱敏串
    assert body["api_key_masked"] == "sk-use********7654"
    assert "sk-user-filled-987654" not in resp.text


def test_save_rejects_bad_base_url(client, no_env_key):
    resp = client.post("/setup/llm", json={"base_url": "不是网址", "api_key": "sk-1234567890"})
    assert resp.status_code == 400
    assert "Base URL" in resp.json()["detail"]


def test_save_rejects_too_short_key(client, no_env_key):
    resp = client.post("/setup/llm", json={"api_key": "sk-1"})
    assert resp.status_code == 400
    assert "API Key" in resp.json()["detail"]


# ============ 写入鉴权 ============

class _StubRequest:
    """够 _guard_write 用的最小请求替身。"""

    def __init__(self, host: str, token: str = ""):
        self.client = type("C", (), {"host": host})()
        self.headers = {"X-Admin-Token": token} if token else {}
        self.query_params = {}


def test_guard_allows_first_time_config_from_anywhere(tmp_path, monkeypatch, no_env_key):
    """首次配置必须放行——否则服务器部署后没法从浏览器完成初始化。"""
    from src.api.setup import _guard_write

    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "uc.json")
    rc.load(force=True)
    _guard_write(_StubRequest("203.0.113.9"))  # 不抛异常即通过


def test_guard_blocks_remote_overwrite_after_configured(tmp_path, monkeypatch, with_env_key):
    from src.api.setup import _guard_write

    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "uc.json")
    rc.load(force=True)
    monkeypatch.setattr(cfg.settings, "monitor_token", "secret-token")

    # 公网 + 无令牌 → 拒绝（换 Key 等于把调用成本转嫁给部署者）
    with pytest.raises(HTTPException) as e:
        _guard_write(_StubRequest("203.0.113.9"))
    assert e.value.status_code == 403

    # 公网 + 错误令牌 → 拒绝
    with pytest.raises(HTTPException):
        _guard_write(_StubRequest("203.0.113.9", token="wrong"))

    # 公网 + 正确令牌 → 放行
    _guard_write(_StubRequest("203.0.113.9", token="secret-token"))

    # 本机 → 放行
    _guard_write(_StubRequest("127.0.0.1"))


# ============ 连通性测试接口 ============

def test_test_endpoint_requires_key(client, no_env_key):
    resp = client.post("/setup/test", json={})
    assert resp.status_code == 400
    assert "API Key" in resp.json()["detail"]


def test_test_endpoint_translates_auth_error(client, no_env_key, monkeypatch):
    """上游 401 必须翻译成人话，而不是把英文 JSON 甩给用户。"""

    class _AuthError(Exception):
        status_code = 401

    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            raise _AuthError("Unauthorized")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeLLM)

    resp = client.post("/setup/test", json={"api_key": "sk-some-key-123456"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "Key 无效" in body["message"]


def test_test_endpoint_translates_timeout(client, no_env_key, monkeypatch):
    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            raise TimeoutError("request timed out")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeLLM)

    body = client.post("/setup/test", json={"api_key": "sk-some-key-123456"}).json()
    assert body["ok"] is False
    assert "超时" in body["message"]


def test_test_endpoint_success(client, no_env_key, monkeypatch):
    class _FakeLLM:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            return type("R", (), {"content": "hi"})()

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _FakeLLM)

    body = client.post("/setup/test", json={"api_key": "sk-some-key-123456", "model": "deepseek-chat"}).json()
    assert body["ok"] is True
    assert "deepseek-chat" in body["message"]
    assert body["latency_ms"] >= 0


# ============ LLM 工厂 ============

def test_get_chat_llm_raises_friendly_error_when_unconfigured(tmp_path, monkeypatch, no_env_key):
    from src.core.llm import LLMNotConfiguredError, get_chat_llm

    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "uc.json")
    rc.load(force=True)

    with pytest.raises(LLMNotConfiguredError) as e:
        get_chat_llm()
    # 提示必须指向可操作的动作，而不是只说"出错了"
    assert "API Key" in str(e.value)


def test_get_chat_llm_respects_explicit_api_key(tmp_path, monkeypatch, no_env_key):
    """调用方显式传了 Key（如测试注入）就应放行，不受全局未配置影响。"""
    from src.core.llm import get_chat_llm

    monkeypatch.setattr(rc, "CONFIG_PATH", tmp_path / "uc.json")
    rc.load(force=True)

    llm = get_chat_llm(api_key="sk-explicit-123456", model="deepseek-chat")
    assert llm is not None
