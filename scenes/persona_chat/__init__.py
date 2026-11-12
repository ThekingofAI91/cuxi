"""名人对话场景包"""

__all__ = ["PersonaChatSceneConfig"]


def __getattr__(name):
    # 读取模型、技能与存储工具不应顺带初始化应用或迁移用户数据。
    if name == "PersonaChatSceneConfig":
        from scenes.persona_chat.config import PersonaChatSceneConfig
        return PersonaChatSceneConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
