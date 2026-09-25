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
    # 开源版：Key 允许为空——用户可以在浏览器「设置」页里填（落盘 data/user_config.json），
    # 也可以通过 .env 填。两者都为空时服务照常启动，只是调用 LLM 前会给出明确提示，
    # 而不是在 import 阶段抛一个看不懂的 pydantic 校验错误。
    # 优先级：用户填写 > .env > 此处默认值（见 src/core/runtime_config.py）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    # 备用模型：主模型额度用完/不可用时自动切换（留空则不启用 fallback）
    llm_fallback_model: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    # 稳定性：单次 LLM 请求超时（秒）+ 失败重试次数（默认重试 2 次）
    llm_request_timeout: float = 60
    llm_max_retries: int = 2
    # LLM HTTP 连接池跨请求复用（按事件循环隔离的共享 httpx.AsyncClient）。
    # 默认每个 ChatOpenAI 实例各建一个连接池，而本项目每请求都新建实例 →
    # 每次 LLM 调用都要重付 TCP+TLS 握手（走第三方中转实测 0.1-0.5s/次）。
    # 开启后全部异步调用复用同一连接池。单次请求超时仍由 llm_request_timeout
    # 在 openai SDK 层生效，不受此客户端影响。
    llm_shared_http_client: bool = True

    # ---- 并发 / 性能 ----
    # 重排序模型：默认 bge-reranker-base（278M，1.1GB）。
    # 实测（优化七十二，2026-08-30）：24 个真实查询 × 3 个教育区库的检索级 A/B，
    # base 与 v2-m3（2.27GB/560M）进入 prompt 的 top-15 文档集合 24/24 完全一致
    # （head5 集合亦 24/24 一致，仅内部顺序小幅换位 tau=0.908）；
    # 12 题端到端成对评估两者得分持平（忠实度 5.8 vs 5.5，噪声内）。
    # 重排耗时 1400ms → 370ms（~3.9x），模型加载也快一倍。
    # 如需 v2-m3 的极致精度：改 .env 的 RERANK_MODEL=BAAI/bge-reranker-v2-m3 即可。
    rerank_model: str = "BAAI/bge-reranker-base"
    # 重排序与向量化并发上限（CPU 推理有界并行：全串行浪费多核，全放开会互相抢占）
    rerank_max_concurrent: int = 2
    embedding_max_concurrent: int = 4
    # 精排候选数：RRF 前 N 条交给 Cross-Encoder（其余按 RRF 顺序兜底），
    # 候选越多 CPU 越慢（15 对 ≈ 20s+），5 对 ≈ 3-4s，排序质量损失很小
    rerank_candidates: int = 5
    # 重排模型 int8 动态量化（只量化 encoder 层 Linear）：
    # 实测（bge-reranker-v2-m3，CPU 8 线程，真实语料）5 对 1925→1223ms（1.57x），
    # 排序一致度 0.8-0.98（翻转集中在分差毫厘之间的近并列对，不影响头部选择）。
    # 转换一次性 ~2.3s，在启动预热线程里完成。torch 2.13 下整模型量化有兼容问题，
    # 必须只量化 encoder 层；量化失败自动回退 fp32。
    rerank_int8_quantize: bool = True
    # torch 推理线程数：建议 = 物理核数 // rerank_max_concurrent（16 核 / 2 并发 = 8），
    # 单请求也够快（8 线程 ≈ 16 线程的 95%），并发时不互相抢占
    torch_num_threads: int = 8

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
    # 短问题（字符数 <= 该值）跳过 Multi-Query + HyDE 改写：省 1 次串行 LLM 往返，
    # 直接以原始查询做向量 + BM25 检索，显著降低首字延迟；长/复杂问题仍走完整改写。
    # 原为 15，放宽到 30：实测改写的收益集中在长难问题（需要换角度召回），
    # 30 字以内的中短问题用原始查询 + BM25 已足够，不值得为它多等 4-8 秒。
    skip_rewrite_max_chars: int = 30
    # 首字为改写预留的等待上限（秒）。
    # 严格意义上"改写与首字并行"做不到——召回必须全部就绪才能起生成。
    # 务实做法：首字最多为改写等这么多秒，超时就放弃改写、直接用原始查询检索；
    # 改写任务不取消，继续在后台跑完并写入缓存，下次问相似问题即可零等待命中。
    rewrite_deadline_sec: float = 2.0
    # 检索改写（Multi-Query + HyDE）单次调用的超时上限（秒）。
    # 实测（deepseek-v4-flash，走第三方中转）：成功 3.5-6.7s，但约 20% 概率
    # 触发重试/超时，实测最坏 24.8s —— 整轮首字延迟就是这么被拖垮的。
    # 超时即降级为"只用原始查询检索"（召回略降但不卡死），绝不让用户干等。
    rewrite_timeout_sec: float = 10.0

    # 检索查询改写（Multi-Query + HyDE，一次 LLM 调用）总开关，默认关闭。
    # 实测（2026-08-29）：改写真实延迟 3.5s+，超过等待预算（rewrite_deadline_sec=2s）
    # 后被放弃——即每个新问题都白等 2s 且改写从未生效；而忠实度基线 8.8 恰是在
    # "改写无效"状态下测得的，关闭不损失基线质量。重复问题靠语义缓存秒回。
    # 打开场景：语料召回明显不足、且愿意为变体检索多付最长 2s 时再开。
    rewrite_enabled: bool = False

    # 娱乐区是否保留轻量检索（top_k=3，无改写 / 无重排 / 无图谱）。
    # 实测（persona_fengge 102 条语料，bge-small-zh CPU）：中位仅 23ms。
    # 对比参照：Cross-Encoder 重排 10421ms —— 真正贵的是重排与改写（LLM 往返），
    # 不是检索本身。23ms 放在 8-15s 的回答里占比约 0.2%，肉眼不可见，
    # 却能换回"回答贴合角色背景资料"，故默认开启。若将来觉得慢，置 False 即回到零检索。
    entertainment_light_retrieval: bool = True
    # 娱乐区单条上下文字符上限：3 条 × 150 = 450 字，够唤起细节又不撑大 prefill
    entertainment_context_chars: int = 150

    # 轻聊快速通道：寒暄/道谢/语气回应等"整句白名单"消息跳过检索管线，
    # 一次 LLM 直出（首字 ~2s，对比全管线 ~20s）。判定刻意保守（全等匹配），
    # 真实提问永远走检索；置 False 恢复全管线。
    light_chat_enabled: bool = True

    # 检索策略按角色分区（CharacterDef.zone）决定，不再有运行期开关
    # （此前的 tool_retrieval_enabled 已删除，避免"配置默认值 ≠ 声明行为"）：
    #   教育区 → 工具化检索（framework/tool_agent）：模型自己判断该不该查原书，
    #            这是本项目唯一的真 agent 环节（LLM 输出决定控制流）。
    #   娱乐区 → 轻量召回（retriever 节点，top-3 向量+BM25），见 entertainment_light_retrieval。
    # 判据函数：framework/supervisor.py 的 resolve_retrieval_strategy()。
    # 为什么这么分：教育区要凭据、可溯源，值一次额外 LLM 往返（3-8s）换"不该查时
    # 一次都不查"；娱乐区检索是 23ms 的软背景、靠角色卡撑人设，加一次往返得不偿失。
    # 工具轮次上限：模型最多请求几轮检索（下限被夹在 1，见 tool_agent 里的 max(1, ...)）。
    # 1 = 一轮带工具做决策 + 一轮不带工具作答，最多 2 次 LLM 往返（模型直答时只 1 次）。
    # 2 会多插一个带工具的中间轮，模型常常忍不住再查一遍（多一次 3-5s 检索 + 一次往返），
    # 实测收益不抵延迟。需要多个检索角度的场景靠模型在**同一轮里并列发多个工具调用**覆盖，
    # 不必靠加轮次。
    tool_max_rounds: int = 1

    # ---- 圆桌会议 / 争鸣（自主发言编排）----
    # 每场会议中，单个角色的发言次数上限（含开场陈述）。作用只有一个：防止某位角色
    # 霸占全场。它**不参与任何胜负判定**（圆桌已去掉裁判），所以这个值只影响节奏、
    # 不影响结果，调大调小都无所谓。
    # 注意配额是"举牌资格"闸门而非"发言许可"：配额耗尽只是不再**主动**举手，
    # 不会出现"举了手却被拒"的尴尬；因此也不存在"配额刚用完恰逢被严重曲解、
    # 只能沉默"这种破坏真实感的情况——那种情况由用户点将来兜（见 roundtable.py）。
    roundtable_max_speeches_per_speaker: int = 2
    # 多人同时举牌时，等待用户点将的秒数。超时由主持人代班决定（相当于"全权委托"）。
    # 这个超时是"用户点选疲劳"的兜底：用户走开或懒得点时，会议不会卡死。
    roundtable_pick_timeout: int = 25
    # 单场会议的 LLM 调用次数硬上限（开场 + 意愿判断 + 发言 + 纪要全部计入）。
    # 起因：自主发言后每轮调用次数不再固定，而 DeepSeek 默认 20 RPM。没有硬上限时，
    # 超限会以 429 的形式暴露，用户看到的是"某位角色静默失声"——比报错更难排查。
    # 默认 24 是按 3 人 × 3 轮算出的上界（3 开场 + 3×4 交锋 + 1 纪要 = 16）再留余量。
    roundtable_max_llm_calls: int = 24
    # 是否在会议结束后生成纪要（分歧地图）。关掉则只留发言记录，省一次调用。
    roundtable_summary_enabled: bool = True

    # 自建角色总量上限（全站）：无账号体系阶段防"脚本批量造角"撑爆磁盘与向量库。
    # 上线账号体系后可按用户单独限额，届时调大或取消全局上限。
    max_custom_characters: int = 200

    # ---- 存储配置 ----
    chroma_persist_dir: str = "./chroma_db"
    log_dir: str = "./logs"

    # ---- Agent 配置 ----
    max_history_turns: int = 5
    verifier_confidence_threshold: float = 0.7
    # 异步引用核查（回答定稿后补跑）的等待预算（秒）。
    # verifier 在 SSE 里位于 citations 事件与 end 事件之间：核查不完，
    # 前端的"完成"状态就一直挂着。超时即放弃本次引用出处推送，按时发 end。
    verify_deadline_sec: float = 15.0

    # ---- API 配置 ----
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # CORS 允许来源，逗号分隔；上线后务必填具体域名（如 https://example.com）
    cors_origins: str = "*"
    # uvicorn worker 数：BM25 检索等纯 Python 计算吃 GIL，单 worker 下并发会串行排队；
    # 多 worker 可线性提升并发吞吐（代价：每个 worker 独立加载模型 ~2.5GB 内存）。
    # 8GB 服务器建议 2，16GB 建议 4；限流/答案缓存为进程内实现，多 worker 时各算各的。
    app_workers: int = 1
    # 开发调试时设置 APP_RELOAD=1 启用热重载；默认关闭，避免文件变动触发服务重启
    app_reload: bool = False

    # ---- 上线安全：限流 / 配额 / 上传限制 / 会话持久化 ----
    rate_limit_per_minute: int = 10      # 每人每分钟最多提问次数
    rate_limit_per_day: int = 300        # 每人每天最多提问次数
    upload_limit_per_day: int = 20       # 每人每天最多上传文档数
    upload_max_mb: int = 10              # 单文件大小上限（MB）
    session_db_path: str = "./data/sessions.db"   # 会话持久化 SQLite 路径
    session_ttl_days: int = 7            # 会话保留天数（过期自动清理）

    # ---- 网络防护（上线前请核对） ----
    security_headers_enabled: bool = True     # 安全响应头（CSP / nosniff / X-Frame-Options 等）
    security_hsts_enabled: bool = False       # HSTS：仅在 HTTPS（nginx 反代 + 证书）下开启
    global_rate_limit_per_minute: int = 120   # 全局兜底限流：单 IP 每分钟请求上限（静态资源不计）
    max_request_body_mb: int = 8              # 单请求体大小上限（MB），超出直接 413

    # ---- 账号系统 ----
    accounts_db_path: str = "./data/accounts.db"  # 用户/登录会话 SQLite 路径
    auth_cookie_name: str = "gkrm_session"        # 登录会话 Cookie 名（HttpOnly）
    auth_cookie_secure: bool = False              # HTTPS 部署时设 True（Cookie 仅经加密连接发送）

    # 上下文智能压缩（实验性）：生成前用 1 次 LLM 调用把检索 chunk 压成面向问题的
    # 摘要（~150 字/条），替代 450 字硬截断——prefill 省 2/3、聚焦问题；代价是
    # 生成前多 2-6s（相似问题命中缓存后为 0）。默认关闭，按 A/B 评估数据决定。
    context_compression_enabled: bool = False

    # ---- 用户长期记忆 ----
    # 跨会话记住登录用户的背景/偏好/关注主题（匿名用户不建记忆，无跨会话身份）。
    # 提取：每轮回答返回后异步 1 次 LLM 调用（≤400 token）；注入：profile 常驻
    # + 主题记忆按查询向量 top-3 命中。账号注销时全量删除（隐私承诺）。
    memory_enabled: bool = True
    memory_db_path: str = "./data/memory.db"

    # ---- 成本 / 监控 ----
    monitor_db_path: str = "./data/usage.db"     # 用量记录 SQLite 路径
    monitor_token: str = ""              # 访问 /admin/stats 的令牌（留空则任何人可看，上线务必设置）
    cost_input_per_1m: float = 1.0       # 元 / 百万输入 token（DeepSeek-chat 约 1 元）
    cost_output_per_1m: float = 2.0      # 元 / 百万输出 token（DeepSeek-chat 约 2 元）

    # ---- 意见反馈 / 管理后台 ----
    # 用户反馈（意见/bug/建议）落盘 SQLite，管理后台 /admin 直接查看与标记处理。
    feedback_db_path: str = "./data/feedback.db"  # 用户反馈 SQLite 路径
    # 访问管理后台（/admin 页面 与 /admin/dashboard、/admin/feedback 读取接口）的令牌，
    # 复用 monitor_token；留空则任何人可看（上线务必设置），提交反馈接口本身不需要 token。

    # ---- 自建角色：网络搜索辅助创建 ----
    # 创建热门人物时，用网络搜索收集公开资料辅助生成人设与背景知识库。
    # 默认开启；若部署环境无外网或希望纯手工录入，设为 false。
    # provider: auto 依次尝试 bing/baidu/ddg；也可指定单一引擎 bing/baidu/ddg
    web_search_enabled: bool = True
    web_search_provider: str = "auto"
    web_search_max_results: int = 8      # 收集多少个搜索结果片段
    web_search_timeout: float = 8.0      # 单次抓取超时（秒）

    # ---- 知识图谱 RAG（GraphRAG 增强层）----
    # 在向量/BM25 检索之外，额外用「实体-关系」知识图谱做结构化检索增强：
    # 构建期：用 LLM 从入库 chunk 批量抽取三元组（实体--关系-->实体 + 证据片段），落盘 JSON；
    # 查询期：对齐查询实体 → k 跳子图扩展 → 把结构化关联 + 原始证据补进检索结果。
    graph_rag_enabled: bool = True       # 总开关；图谱文件不存在时自动降级（不影响原检索）
    graph_auto_build: bool = True        # 角色首次被对话且图谱缺失时，自动后台构建（常驻可用，无需用户点按钮）
    graph_dir: str = "./data/persona_chat/graphs"  # 图谱 JSON 持久化目录（按 collection 名）
    graph_build_batch: int = 6           # 构建时每批送入 LLM 的 chunk 数（太大降质量、太小题太多）
    graph_build_max_chunks: int = 4000  # 单 collection 最多抽取多少 chunk（防爆 token，超大库截断）
    graph_hops: int = 2                  # 查询期子图扩展跳数
    graph_top_entities: int = 6         # 匹配上的种子实体最多扩展多少个
    graph_max_edges: int = 30           # 子图返回的关系条数上限（控制上下文长度）
    graph_evidence_chars: int = 260     # 单条证据片段截断字符数
    graph_context_chars: int = 2200     # 图谱上下文总字符上限

    # ---- 知识图谱「按需触发」阈值（默认只在文本检索不达标时才启动图谱增强）----
    # 文本召回条数少于该值 → 触发；top rrf_score 低于该值 → 触发。
    # 二者都不触发、且查询命中的图谱实体均已被文本结果覆盖 → 跳过图谱（纯文本检索）。
    graph_trigger_min_docs: int = 3     # 文本召回不足阈值（条）
    graph_trigger_score: float = 0.03   # 文本 top 相关性偏弱阈值（rrf_score）

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
