"""收集测试之前隔离自建角色存储，避免应用导入迁移真实用户数据。"""

from tempfile import TemporaryDirectory
from pathlib import Path

from scenes.persona_chat import custom_store


_custom_storage = TemporaryDirectory(prefix="persona-tests-")
custom_store.CUSTOM_DIR = Path(_custom_storage.name)


def pytest_sessionfinish(session, exitstatus):
    _custom_storage.cleanup()
