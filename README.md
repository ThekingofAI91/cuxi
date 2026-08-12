# 名人对话 AI 助手（Multi-Agent RAG Persona Chat）

一个让你和历史名人、当代网红"面对面"聊天的 AI 应用：荣格、阿德勒、王阳明、峰哥、张雪峰，五位人物各有一套专属人设、知识库和界面主题。

后端用 LangGraph 多智能体协作 + 混合检索 RAG，前端是零构建的单文件页面，手机优先、桌面为尊。

---

## ✨ 功能亮点

- **5 位可对话人物**：首页轮播选择 → 进入各自专属 UI；对话页只显示当前人物，可一键退回首页换人
- **每人一套视觉主题**：荣格=纸墨、阿德勒=印刷、王阳明=夜航、峰哥=街头、张雪峰=直播数据台；电脑端横排轮播，手机端堆叠卡片
- **沉浸模式**：峰哥/张雪峰关闭"要点/来源"展示，保持对话沉浸感；荣格/阿德勒/王阳明保留专业溯源
- **多智能体流水线**：Supervisor 规则路由（不烧 LLM）→ Retriever / Analyzer / Verifier / Summarizer
- **混合检索 + 来源加权**：向量 + BM25 双通道，Cross-Encoder 重排；语料按 original/oral/anchor/secondary 分级加权，二手解读强降权，防止"第三者评价"冒充名人原话
- **防幻觉验证**：专业型角色回答前对引用做核查与置信度标注
- **工程化打磨**：限流、上传限制、会话 SQLite 持久化（重启不丢）、答案内存缓存、成本监控看板

---

## 🧱 技术栈

| 层 | 技术 |
|----|------|
| 后端 | FastAPI + LangGraph + DeepSeek API |
| 向量检索 | ChromaDB + BAAI/bge-small-zh-v1.5（512 维，CPU 可跑） |
| 关键词检索 | rank-bm25（磁盘缓存，重启复用） |
| 重排序 | BAAI/bge-reranker（Cross-Encoder） |
| 前端 | 原生 HTML/CSS/JS 单文件（`frontend/index.html`，无构建步骤） |
| 持久化 | SQLite（会话 / 用量统计）+ ChromaDB（向量库） |
| 评估 | RAGAS（scripts/evaluate_ragas.py）+ pytest（20 个用例） |

---

## 🚀 快速开始

环境要求：Python 3.10+

```bash
# 1. 安装依赖
uv sync --extra dev          # 或 pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env         # 填入 DEEPSEEK API Key（LLM_API_KEY）

# 3. 初始化语料（解析 md/pdf → 分块 → 向量化 → 写入 ChromaDB）
python init_persona_data.py  # 只重灌某个人物见下方"数据维护"

# 4. 启动服务
python main.py               # 默认 http://localhost:8000
```

启动后访问：

- 对话页面：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 成本统计：http://localhost:8000/admin/stats（上线前务必在 `.env` 设置 `MONITOR_TOKEN`）

---

## 🗂️ 项目结构

```
.
├── main.py                        # FastAPI 入口（启动预热 + 会话恢复）
├── init_persona_data.py           # 语料入库脚本（可指定单角色）
├── framework/                     # LangGraph 智能体
│   ├── supervisor.py              #   主管：规则路由 + 最终答案编译
│   ├── retrieval_agent.py         #   检索智能体（混合检索）
│   ├── analysis_agent.py          #   分析智能体
│   ├── verification_agent.py      #   验证智能体（引用核查）
│   └── summarizer.py              #   历史压缩智能体
├── scenes/persona_chat/           # 名人对话场景
│   ├── config.py                  #   场景与角色注册
│   └── characters/                #   每位人物的角色定义（人设/主题/开关）
├── src/
│   ├── api/routes.py              # REST + SSE 接口
│   ├── core/                      # 配置 / 会话存储 / 成本监控
│   ├── document_processing/       # 文档解析
│   └── retrieval/                 # 嵌入 / 分块 / 混合检索 / 重排 / 来源加权
├── data/persona_chat/             # 各人物语料（md/pdf/txt）
├── frontend/index.html            # 单文件前端（全部主题与交互）
├── scripts/
│   ├── cost_report.py             # 成本日报
│   └── evaluate_ragas.py          # RAGAS 评测
└── tests/                         # 智能体 + API 集成测试
```

---

## 🧑‍🤝‍🧑 人物与知识库

| 人物 | 定位 | ChromaDB collection | 语料构成 |
|------|------|---------------------|----------|
| 荣格 | 解梦大师 · 潜意识捕手 | `persona_jung` | 原著（红书/心理类型/原型与集体无意识…）+ 演讲与回忆录 |
| 阿德勒 | 感情急救员 · 自卑超越教练 | `persona_adler` | 原著 + 演讲 + 《被讨厌的勇气》（后人虚构，降权） |
| 王阳明 | 心学宗师 · 破心中贼专家 | `persona_wangyangming` | 54 本电子书精选（传习录/大传/讲稿）+ 扫描件 |
| 峰哥 | 下三路之神 · 街头社会学家 | `persona_fengge` | 2024 媒体专访实录 + 公开视频/直播语料 |
| 张雪峰 | 长跑之王 · 升学指路 | `persona_zhangxuefeng` | 5 本著作 + 深度采访 + 语录分类辑录 |

> 说明：张雪峰语料信息截止 **2026-03-24**（其离世日），仅收录生前公开言论，不掺悼念内容。

---

## 🧠 多智能体架构

```
用户提问
   │
   ▼
Supervisor（规则路由，不调 LLM）
   │
   ├─ 需要查资料? ──► Retriever（Multi-Query + HyDE + 向量/BM25 + 重排）
   │                        │
   │                        ▼
   ├─ 分析回答?  ────► Analyzer（基于检索资料组织答案）
   │                        │
   │                        ▼
   ├─ 有引用且角色开启验证? ► Verifier（逐条核查出处 + 置信度）
   │
   └─ Summarizer（多轮历史超阈值时自动压缩）
   │
   ▼
最终答案（SSE 流式返回）
```

要点：

- **规则优先路由**：Supervisor 按关键词规则分流，跳过 LLM 路由调用，省成本降延迟；路由循环有上限兜底（GraphRecursionError 防护）
- **验证按人设开关**：`enable_verification` 默认开启；峰哥/张雪峰（沉浸角色）关闭，避免"要点/来源"破坏代入感
- **历史感知检索**：结合最近对话轮次解决"那个梦""这跟它有什么关系"这类指代问题

---

## 🔍 检索管线

1. **查询增强**：一次 LLM 调用同时生成查询变体（Multi-Query）与假设文档（HyDE）
2. **双通道召回**：变体查询批量向量检索 + 原问 BM25 关键词检索（精确命中专有名词）
3. **RRF 融合 + 来源加权**：`source_profile.py` 按文件名把语料分为 original/oral/anchor/artificial/secondary，RRF 累加时乘以权重（secondary 0.25 强降权）
4. **Cross-Encoder 重排**：bge-reranker 对候选深度精排，取 top-K

### 数据维护

```bash
# 只重灌某个人物（避免全量重建其他角色的大库）
python -c "import init_persona_data; init_persona_data.load_character_data('fengge')"

# 重灌后必须失效该角色的 BM25 磁盘缓存（磁盘缓存优先路径不校验数量）
python -c "from src.retrieval.advanced_search import invalidate_bm25_cache; invalidate_bm25_cache('persona_fengge')"
```

新增语料文件后，建议在 `src/retrieval/source_profile.py` 补文件名规则（如访谈→`oral`），否则新文件会落入 `unknown`（权重 1.0）。

---

## 📡 API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/persona/characters` | 人物列表（id/名称/主题/标语） |
| POST | `/persona/query` | 对话提问（SSE 流式返回；命中缓存直接回放） |
| POST | `/persona/upload` | 上传文档（每用户每日限额 + 10MB 上限） |
| DELETE | `/conversation/{session_id}` | 删除会话 |
| GET | `/conversation/pending/{session_id}` | 恢复未完成的流式回答 |
| GET | `/conversation/{session_id}/meta` | 会话元信息 |
| GET | `/health` | 健康检查 |
| GET | `/admin/stats` | 今日/本周/按角色用量与成本（需令牌） |

---

## 🧪 测试与评估

```bash
pytest                              # 20 个用例：智能体行为 + API 集成（SSE/限流/缓存/敏感过滤/会话持久化/上传限制）
python scripts/evaluate_ragas.py     # 跑真实流水线，输出 faithfulness / answer_relevancy / context_precision
python scripts/cost_report.py        # 成本日报（基于 SQLite 用量日志）
```

---

## 📝 备注

- 前端为单文件 `frontend/index.html`（约 169KB），五套主题通过 CSS 变量切换，无需构建工具
- 本项目的定位是**展示型个人项目**：演示多智能体协作、混合检索、来源可信度设计与工程化细节；LLM 与向量模型均可在 CPU 上低成本运行
