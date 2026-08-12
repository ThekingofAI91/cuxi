"""
全局配置管理
使用 pydantic-settings 从环境变量和 .env 文件读取配置
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，从 .env 文件和环境变量自动加载"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,# 环境变量不区分大小写
    )

    # ---- LLM 配置 ----
    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    # 稳定性：单次 LLM 请求超时（秒）+ 失败重试次数（默认重试 2 次）
    llm_request_timeout: float = 60
    llm_max_retries: int = 2

    # ---- Embedding 配置 ----
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    embedding_device: str = "cpu"

    # ---- 检索配置 ----
    chunk_size: int = 1000
    chunk_overlap: int = 100
    retrieval_top_k: int = 50
    rerank_top_k: int = 5
    bm25_weight: float = 0.5

    # ---- 存储配置 ----
    chroma_persist_dir: str = "./chroma_db"
    log_dir: str = "./logs"

    # ---- Agent 配置 ----
    max_history_turns: int = 5
    verifier_confidence_threshold: float = 0.7

    # ---- API 配置 ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # CORS 允许来源，逗号分隔；上线后务必填具体域名（如 https://example.com）
    cors_origins: str = "*"
    # 开发调试时设置 APP_RELOAD=1 启用热重载；默认关闭，避免文件变动触发服务重启
    app_reload: bool = False

    # ---- 上线安全：限流 / 配额 / 上传限制 / 会话持久化 ----
    rate_limit_per_minute: int = 10      # 每人每分钟最多提问次数
    rate_limit_per_day: int = 300        # 每人每天最多提问次数
    upload_limit_per_day: int = 20       # 每人每天最多上传文档数
    upload_max_mb: int = 10              # 单文件大小上限（MB）
    session_db_path: str = "./data/sessions.db"   # 会话持久化 SQLite 路径
    session_ttl_days: int = 7            # 会话保留天数（过期自动清理）

    # ---- 成本 / 监控 ----
    monitor_db_path: str = "./data/usage.db"     # 用量记录 SQLite 路径
    monitor_token: str = ""              # 访问 /admin/stats 的令牌（留空则任何人可看，上线务必设置）
    cost_input_per_1m: float = 1.0       # 元 / 百万输入 token（DeepSeek-chat 约 1 元）
    cost_output_per_1m: float = 2.0      # 元 / 百万输出 token（DeepSeek-chat 约 2 元）

    def get_chroma_path(self) -> Path:
        """返回 ChromaDB 持久化路径"""
        return Path(self.chroma_persist_dir)

    def get_log_path(self) -> Path:
        """返回日志目录路径"""
        path = Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# 全局单例
settings = Settings()
