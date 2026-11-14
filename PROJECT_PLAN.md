# 电商政策智能问答系统 (Policy-RAG) 执行计划与架构规范

> **定位**：面向大厂实习/校招求职、可独立运行、可开源上线 GitHub 的高可靠“电商政策智能问答系统”。
> **核心特性**：多路混合检索（BM25 + BGE 向量 + RRF + BGE Cross-Encoder 精排） + LangGraph 自适应受控补检状态机（Self-RAG 闭环防假充分、防死循环） + 事实忠实度自检 + 纯内存极速启动。

---

## 目录
1. [系统目标与边界原则](#1-系统目标与边界原则)
2. [关键数据契约与接口约定](#2-关键数据契约与接口约定)
3. [重点改造机制：受控 Self-RAG 与防假充分闭环](#3-重点改造机制受控-self-rag-与防假充分闭环)
4. [项目目录结构与模块划分](#4-项目目录结构与模块划分)
5. [分阶段实施路线图与验收标准](#5-分阶段实施路线图与验收标准)
6. [评测体系与消融实验口径](#6-评测体系与消融实验口径)
7. [求职亮点与面试关键考点映射](#7-求职亮点与面试关键考点映射)
8. [求职简历标准草稿（实测对齐版）](#8-求职简历标准草稿实测对齐版)

---

## 1. 系统目标与边界原则

### 1.1 核心建设范围
- **政策文档解析与分块**：对 Markdown 格式的 38 篇电商真实售后/物流/服务政策进行清洗、标题层级分块、生成确定性持久化元数据。
- **内存级四路混合检索体系**：
  - 词法检索：基于 `jieba` 分词的 BM25Okapi（捕捉专有名词与精确数字）；
  - 语义检索：基于本地已缓存的 `BAAI/bge-small-zh-v1.5`（或 bge-base）密集向量召回；
  - 排名融合：基于 Cormack RRF（Reciprocal Rank Fusion，$k=60$）；
  - 交叉编码精排：基于本地 `BAAI/bge-reranker-base` Cross-Encoder 交互重排，附带词法保底降级。
- **LangGraph 自适应受控状态机**：
  - 路由分流：`DIRECT`（打招呼/无检索直答）、`SINGLE`（单轮混合检索）、`MULTI`（受控多轮补检）、`REFUSE`（越界/黑灰产/业务写操作拦截）；
  - 事实忠实度检查（Faithfulness）：裁判模型校验生成回答是否忠实于证据，检测事实幻觉并执行自纠错（Self-Correction）。
- **工程交付物**：FastAPI RESTful 接口、全链路请求耗时分解、确定性离线评测套件、可复现 README 与简历草稿。

### 1.2 边界约束与原则（坚决不做的冗余）
- **绝不引入业务写操作**：不处理真实订单取消、修改地址等写操作，不引入 SQL 数据库（Alembic/PostgreSQL）、不引入 Redis 缓存与分布式锁。
- **绝不引入复杂 Agent 工具框架**：不引入 MCP 协议客户端、ToolRegistry、动态代码执行、多实例分布式集群部署。
- **检索纯内存自包含**：第一版检索索引全部常驻内存，`git clone` 后无需启动任何独立向量数据库（如 Qdrant、Milvus）即可单机自愈运行。
- **参考仓库只读原则**：参考仓库 `Adaptive-rag-main` 与 `all-in-rag` 严格保持只读，绝不直接在原目录修改。

---

## 2. 关键数据契约与接口约定

系统内部所有模块间传递的数据结构均通过 Pydantic 严格校验，防止字段漂移与类型错误。

### 2.1 文档片段契约 (`RetrievedChunk`)
```python
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    document_id: str                      # 政策文档唯一标识（如 return_policy_a1b2c3d4）
    chunk_id: str                         # 片段稳定 ID（格式：{document_id}_{chunk_index:02d}）
    content: str                          # 清洗后的切块文本
    title: str                            # 文档标题（如 "7天无理由退换货规则"）
    source: str = "knowledge"             # 数据源来源标识（ingest/bm25/vector/rrf/rerank）
    section: str | None = None            # 所在 Markdown 章节标题（如 "退货条件及运费承担"）
    score: float = 0.0                    # 检索/融合/精排打分
    rank: int | None = None               # 在候选列表中的排名（从 1 开始）
    chunk_index: int = 0                  # 块在原文档中的物理顺序序号
    metadata: dict[str, str] = Field(default_factory=dict)
```

### 2.2 补检状态与退出枚举
```python
from enum import Enum

class GradeStatus(str, Enum):
    SUFFICIENT = "sufficient"               # 证据明确、充分，足以支撑严谨回答
    INSUFFICIENT = "insufficient"           # 证据不足、关键条件缺失
    GRADER_UNAVAILABLE = "grader_unavailable" # 评审模型调用失败/超时/返回非法 JSON

class ExitReason(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence" # 证据已充分，正常提前终止
    NO_NEW_EVIDENCE = "no_new_evidence"         # 该轮未检索到任何去重增量证据，提前终止
    MAX_HOPS_REACHED = "max_hops_reached"       # 达到最大配置轮数（默认 2 轮），终止
    SINGLE_HOP_DONE = "single_hop_done"         # 单轮检索策略执行完毕
    DIRECT_ANSWER = "direct_answer"             # 策略判定无需检索直接回答
    GUARDRAIL_REFUSED = "guardrail_refused"     # 安全护栏拦截或业务操作越界拒答
```

### 2.3 单轮补检详细追踪 (`HopTrace`)
```python
class HopTrace(BaseModel):
    hop_index: int                         # 当前轮次（1, 2...）
    query: str                             # 触发该轮的输入 Query
    rewritten_queries: list[str]           # 改写模型生成的分支检索词
    retrieved_chunk_ids: list[str]         # 该轮召回的 Chunk ID 列表
    new_chunk_ids: list[str]               # 相比历史轮次累积集合的新增 Chunk ID
    grade_status: GradeStatus              # 该轮评审结论
    missing_aspect: str = ""               # 若不足，缺失的条件或维度说明
    rationale: str = ""                    # 评审逻辑说明
```

### 2.4 全生命周期请求状态与 API 输出 (`ChatResponse`)
```python
class Citation(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    section: str | None = None

class LatencyBreakdown(BaseModel):
    guardrail_ms: float = 0.0
    routing_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    faithfulness_ms: float = 0.0
    total_ms: float = 0.0

class ChatResponse(BaseModel):
    answer: str                            # 最终回答
    strategy: str                          # 采用的策略 (direct/single/multi/refuse)
    user_query: str                        # 原始用户提问
    rewritten_queries: list[str] = Field(default_factory=list) # 所有参与检索的改写词
    hop_count: int = 0                     # 实际执行的检索/补检轮次
    hop_traces: list[HopTrace] = Field(default_factory=list)   # 每轮详细追踪记录
    final_grade_status: GradeStatus = GradeStatus.SUFFICIENT
    exit_reason: ExitReason = ExitReason.SINGLE_HOP_DONE
    evidence: list[RetrievedChunk] = Field(default_factory=list) # 最终生效的证据列表
    citations: list[Citation] = Field(default_factory=list)       # 答案引用的证据
    refused: bool = False                  # 是否拒答
    error: str | None = None               # 异常说明（区分空检索结果与底层异常）
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
```

---

## 3. 重点改造机制：受控 Self-RAG 与防假充分闭环

针对常见 RAG 项目中“检出即假定充分”、“异常时误判为充分”、“补检过程死循环”、“生成层无视证据不足”等严重缺陷，建立如下控制逻辑：

```mermaid
flowchart TD
    Start([进入受控补检 Loop]) --> InitAccum[初始化累积证据池 accumulated = {}]
    InitAccum --> LoopHop{hop < MAX_HOPS ?}
    
    LoopHop -- 超过上限 --> ExitMax[退出: MAX_HOPS_REACHED]
    LoopHop -- 继续本轮 --> Rewrite[LLM 根据 missing_aspect 改写检索词]
    Rewrite --> HybridSearch[执行 Hybrid 混合检索 + Cross-Encoder 精排]
    HybridSearch --> Dedupe[与已有 accumulated 计算差集 new_chunks]
    
    Dedupe --> CheckProgress{len(new_chunks) > 0 ?}
    CheckProgress -- 否: 无新证据 --> ExitNoProg[退出: NO_NEW_EVIDENCE]
    CheckProgress -- 是: 获得新证据 --> UpdateAccum[更新 accumulated 证据池]
    
    UpdateAccum --> GradeLLM[LLM 评审证据充分性]
    GradeLLM -- 抛出异常/超时/JSON损坏 --> GradeDegrade[状态 = GRADER_UNAVAILABLE]
    GradeDegrade --> ExitUnavail[降级退出: 终止补检并保留当前证据]
    
    GradeLLM -- 返回合规 JSON --> CheckSuff{sufficient == true ?}
    CheckSuff -- 是 --> ExitSuff[退出: SUFFICIENT_EVIDENCE]
    CheckSuff -- 否 --> SetMissing[记录 missing_aspect, 轮次 hop + 1]
    SetMissing --> LoopHop

    ExitMax --> GenLayer[生成层 generate_node]
    ExitNoProg --> GenLayer
    ExitUnavail --> GenLayer
    ExitSuff --> GenLayer

    GenLayer --> FaithCheck[忠实度自检与引用对齐]
```

### 3.1 核心防护守则
1. **绝对禁止“检索到了文档 = 证据充分”**：即使召回了相关段落，若段落内未回答用户涉及的核心约束（如“定制类商品”、“特殊偏远地区”），评审结果必须置为 `INSUFFICIENT`。
2. **严密的异常降级**：当模型网络抖动、超时或返回非标准 JSON 时，捕获异常并将该轮状态置为 `GRADER_UNAVAILABLE`，**严禁将其强制判定为充分**，安全终止补检并把当前有限证据转交生成层。
3. **无增量提前终止（Anti-Thrashing）**：若本轮多路召回重排后经 `chunk_id` 去重计算发现新片段数为 0，说明改写词未能在知识库中挖掘出任何新知识，立即触发 `NO_NEW_EVIDENCE` 退出，避免白白浪费 LLM 算力与推理时间。
4. **生成层安全保底（Prompt Guard）**：
   - 当退出原因为 `MAX_HOPS_REACHED`、`NO_NEW_EVIDENCE` 且证据仍为 `INSUFFICIENT` 时，系统在生成提示词中注入强约束：“*当前已知政策中未明确包含 {missing_aspect}，请向用户如实阐述已知条款，并明确声明此条件未找到规定，切忌捏造政策细节*”。
5. **引用严格对应**：生成层提取的 `citations` 必须在当前最终的 `evidence` 中完全存在，若模型引用了非本轮召回的虚假 `chunk_id`，后置逻辑予以自动过滤，杜绝悬空伪造引用。

---

## 4. 项目目录结构与模块划分

```text
D:\PythonProject\Application\policy-rag\
├── README.md                      # 项目主文档 (架构图、安装复现、设计理念、简历总结)
├── PROJECT_PLAN.md                # [本文件] 项目顶层执行计划与契约标准
├── pyproject.toml                 # 现代化依赖配置 (FastAPI, LangGraph, BGE, BM25, Pytest)
├── .env.example                   # 环境变量无密钥模版
├── .gitignore                     # Git 忽略配置 (忽略虚拟环境、权重缓存、.env、评测报告)
├── LICENSE                        # 开源协议 (Apache-2.0 / MIT)
│
├── knowledge/                     # 政策 Markdown 知识库 (38 篇真实电商标准政策)
│   ├── return_policy.md           # 7天无理由退货、破损、定制条款
│   ├── shipping.md                # 包邮门槛、偏远地区加价、时效
│   ├── warranty.md                # 数码/家电/配饰保修期与人为损坏条款
│   ├── coupons.md                 # 满减规则与券叠加使用政策
│   └── ...
│
├── policy_rag/                    # 核心代码包
│   ├── __init__.py
│   ├── config.py                  # Pydantic 强类型系统配置
│   ├── schemas/                   # 跨模块数据契约
│   │   ├── __init__.py
│   │   ├── retrieval.py           # RetrievedChunk, Citation, HopTrace, GradeStatus
│   │   └── chat.py                # ChatRequest, ChatResponse, LatencyBreakdown
│   ├── indexing/                  # 知识库离线/启动时处理管道
│   │   ├── __init__.py
│   │   ├── cleaning.py            # 文本清洗、NFKC、CJK 换行修整、SimHash 去重
│   │   ├── chunking.py            # Markdown 标题结构感知分块与稳定 ID 生成
│   │   └── pipeline.py            # 知识库扫描加载与常驻内存管道
│   ├── retrieval/                 # 检索核心实现
│   │   ├── __init__.py
│   │   ├── tokenizer.py           # Jieba 中文分词与停用词处理
│   │   ├── bm25.py                # 词法 BM25Okapi 索引
│   │   ├── vector.py              # 纯内存向量数据库 (余弦相似度极速计算)
│   │   ├── fusion.py              # Reciprocal Rank Fusion (RRF k=60)
│   │   └── reranker.py            # 交叉编码重排器 (Cross-Encoder / Lexical 降级)
│   ├── providers/                 # 模型抽象与驱动实现
│   │   ├── __init__.py
│   │   ├── base.py                # LLMProvider, EmbeddingProvider, RerankProvider 协议
│   │   ├── deepseek.py            # DeepSeek-V3 异步客户端 (结构化输出与容错)
│   │   ├── embeddings.py          # BGE 本地模型加载器 (与 Hash 降级)
│   │   ├── cross_encoder.py       # 本地 BGE-Reranker-Base 加载器
│   │   └── fake.py                # 单元测试专用 Mock 模型桩
│   ├── guardrails/                # 安全与业务边界检查
│   │   ├── __init__.py
│   │   ├── injection.py           # Prompt 注入过滤
│   │   └── boundary.py            # 政策问答范围检查与“订单写操作”拦截引导
│   ├── agents/                    # LangGraph 自适应状态机
│   │   ├── __init__.py
│   │   ├── state.py               # 强类型 AgentState
│   │   ├── prompts.py             # 核心 Prompt (意图分类、改写、充分性评审、生成、忠实度)
│   │   ├── loop.py                # 受控 Self-RAG 补检循环核心逻辑
│   │   ├── nodes.py               # LangGraph 独立功能节点集合
│   │   └── graph.py               # StateGraph 组装、条件边路由与编译
│   ├── service.py                 # 容器装配 (PolicyContainer) 与业务入口
│   └── api/                       # HTTP 接口层
│       ├── __init__.py
│       ├── main.py                # FastAPI 实例生命周期管理
│       └── routes.py              # /api/v1/chat, /health, /metrics 路由
│
├── eval/                          # 科学评测与可复现验证套件
│   ├── dataset/
│   │   ├── rag_eval.jsonl         # 政策检索金标测试集 (Query + Relevant Doc IDs)
│   │   └── edge_cases.jsonl       # 边界与异常测试集 (无答案/越界写操作/注入攻击)
│   ├── metrics.py                 # P@1, R@3, Hit@1, Hit@3, MRR, Faithfulness 计算
│   └── runner.py                  # 自动化评测主程序 (对比 BM25, Vector, Hybrid, Rerank)
│
└── tests/                         # 自动化测试回归套件
    ├── conftest.py                # 测试固件与隔离环境初始化
    ├── test_cleaning.py           # 文本清洗与去重单元测试
    ├── test_chunking.py           # Markdown 分块与 ID 稳定性测试
    ├── test_retrieval_fusion.py   # BM25 + 向量检索 + RRF 融合测试
    ├── test_reranker.py           # Cross-Encoder 精排与降级测试
    ├── test_self_rag_loop.py      # 受控补检状态机、防假充分与提前退出测试
    ├── test_guardrails.py         # 安全注入与业务操作拦截测试
    ├── test_faithfulness.py       # 忠实度自检与自纠错测试
    └── test_e2e_api.py            # FastAPI 端到端问答与字段契约测试
```

---

## 5. 分阶段实施路线图与验收标准

系统建设划分为 6 个阶段，严格遵循“先完成单轮可运行基线，再推进数据链路、混合检索、受控补检和评测”的次序：

### 阶段 1：独立目录单轮最小可运行基线 (Minimal Runnable Baseline)
- **解决问题**：在目标目录建立独立工程结构，打通配置、数据契约、虚拟模型驱动和端到端调用，保证从第一行代码起可运行、可测试。
- **输入输出**：输入为模拟问答调用；输出为符合完整 Schema 的结构化 `ChatResponse`。
- **参考来源**：`Adaptive-rag-main/app/main.py` 及基本配置系统。
- **修改与取舍**：剥离所有数据库、Redis、SQLAlchemy，依赖内存注入；采用 `FakeLLMProvider` 和 `HashEmbeddingProvider`，确保本地零网络即可完整跑通单测。
- **验证手段**：运行 `pytest tests/test_e2e_api.py`，断言基础链路打通。
- **掌握知识**：Pydantic Settings v2 强类型配置、FastAPI 依赖注入（Container 模式）、测试驱动的假模型隔离。

### 阶段 2：数据链路（Markdown 政策清洗、分块与元数据管道）
- **解决问题**：精准提取 38 篇电商政策，按章节逻辑结构切块，生成稳定唯一的 `document_id` 和 `chunk_id`。
- **输入输出**：输入为 `knowledge/*.md`；输出为结构化的 `list[RetrievedChunk]`。
- **参考来源**：`Adaptive-rag-main/app/retrieval/chunking.py`, `cleaning.py`。
- **修改与取舍**：引入基于标题（Heading #/##/###）的内容切分，在段落过长时使用带有 Overlap（60字符）的滑动窗口，保留章节路径上下文（`section`），避免跨条款破坏语义。
- **验证手段**：运行 `pytest tests/test_chunking.py`，断言切块数量合理、无空字段、ID 在重复运行时具备确定性不变性。
- **掌握知识**：RAG 场景中文本分块的边界效应（Boundary Effect）、SimHash 去重算法、CJK 语言换行规范化。

### 阶段 3：内存混合检索系统（BM25 + 向量召回 + RRF + BGE Cross-Encoder）
- **解决问题**：克服单纯向量检索在精确数字、专有条款上的泛化丢失，以及单纯关键词检索在语义表达上的偏狭。
- **输入输出**：输入为查询 Query 与 `top_k`；输出为带有归一化分数的重排候选片段。
- **参考来源**：`Adaptive-rag-main/app/retrieval/` 下的 `bm25.py`, `vector.py`, `fusion.py`, `reranker.py`。
- **修改与取舍**：接入本地 HuggingFace 缓存中的真实 BGE 权重（`bge-small-zh-v1.5` 与 `bge-reranker-base`）；底层向量库采用内存点积/余弦计算，省去外部数据库中间件开销。
- **验证手段**：运行 `pytest tests/test_retrieval_fusion.py` 与 `test_reranker.py`，验证四种模式均能产生合理排序，并测试缺失权重时的词法平滑降级。
- **掌握知识**：双塔召回（Bi-Encoder）与交叉编码精排（Cross-Encoder）的计算复杂度权衡；RRF 算法在异构打分场景下的数学优势。

### 阶段 4：LangGraph 自适应受控补检状态机（重点攻坚）
- **解决问题**：实现复杂政策的自适应多轮检索，彻底解决“假充分”漏洞，严密跟踪每轮增量，提供无新证据与达到轮数上限的退出保护，完成事实忠实度检查。
- **输入输出**：输入为用户 Query；输出为填满 `hop_traces`、明确 `exit_reason`、带有真实证据引用的完整状态。
- **参考来源**：`Adaptive-rag-main/app/retrieval/self_rag.py`, `app/agents/nodes/`。
- **修改与取舍**：重写 `SelfRAGLoop` 为显式状态机，新增 `GradeStatus` 与 `ExitReason` 枚举，限制最多 2 轮补检；针对业务写操作指令（如取消订单）直接在路由拦截，返回专业导流说明。
- **验证手段**：运行 `pytest tests/test_self_rag_loop.py`，覆盖全部 6 种退出分支场景及降级逻辑。
- **掌握知识**：LangGraph 条件状态机工作原理、Self-RAG 迭代反思机制、大模型推理防死循环机制。

### 阶段 5：FastAPI 接口、执行记录与离线科学评测
- **解决问题**：将系统封装为生产级 RESTful Web 接口，构建量化评测体系，在固定测试集上消融对比四种检索方案。
- **输入输出**：提供 `/api/v1/chat` 接口；评测脚本输出 Precision@1, Recall@3, Hit@1, Hit@3, MRR 评测报表。
- **参考来源**：`Adaptive-rag-main/eval/runner.py`, `metrics.py`。
- **修改与取舍**：文档级（Doc Stem）去重评测口径；新增包含复杂条款、缺失证据与业务写操作的边界回归集；记录全链路耗时明细。
- **验证手段**：运行 `python -m eval.runner` 生成客观评测数据与失败样本诊断报告。
- **掌握知识**：信息检索（IR）工业评测口径计算方法、全链路延迟（P50/P95）调优与链路瓶颈排查。

### 阶段 6：GitHub 开源交付物与求职简历输出
- **解决问题**：将工程提炼为可以直接展示给面试官的代码资产，产出高标准的 README、真实 Trace 样例及求职简历话术。
- **输入输出**：完整无密钥的工程、架构图、中英文 README、真实追踪案例日志、简历草稿。
- **参考来源**：`Adaptive-rag-main/eval/MEASURED.md` 与一线大厂开源工程规范。
- **修改与取舍**：简历内容覆盖知识库构建、混合检索、自适应补检、工程交付四大模块，未实测的数字坚决不编造，标明待测项。
- **验证手段**：全新虚拟环境模拟用户 `git clone` $\to$ `pip install` $\to$ 运行通过。
- **掌握知识**：如何向技术面试官清晰拆解 RAG 项目难点、如何阐述业务架构权衡（Trade-offs）。

---

## 6. 评测体系与消融实验口径

为确保系统效果客观可信，杜绝数字造假，评测脚本执行严格消融实验：

### 6.1 检索消融方案对比（固定 65 条政策金标测试集）
1. **纯词法检索 (BM25)**：评估对专有名词、时间限制、金额门槛的召回表现；
2. **纯密集向量 (Dense Vector)**：评估基于 `bge-small-zh-v1.5` 的语义联想与泛化能力；
3. **混合检索 (Hybrid: BM25 + Dense + RRF)**：验证倒数排名融合能否互补词法与语义盲区；
4. **混合检索 + 重排 (Hybrid + BGE Cross-Encoder)**：验证交互式重排对 Top-1 精度的提振效果。

### 6.2 统计口径与去重规则
- **去重方式**：文档级去重（提取 `document_id` 的主体前缀 Stem，同一文档多片段召回仅算一次命中）。
- **指标定义**：
  - **Precision@1**：Top-1 检索结果是否包含标准相关文档；
  - **Recall@3**：Top-3 召回集合覆盖标准相关文档的比例；
  - **Hit@1 / Hit@3**：前 1 / 3 项中命中至少一篇相关文档的样本比例；
  - **MRR (Mean Reciprocal Rank)**：首个相关文档在检索结果中的倒数排名均值。
- **成本与耗时核算**：全面统计意图分类、改写查询、证据评审、最终生成、忠实度自检的调用次数与 Token 开销，不将单一生成阶段混淆为全系统成本。

---

## 7. 求职亮点与面试关键考点映射

本系统在技术面试中可作为突出个人系统工程能力的核心亮点：

| 简历模块 | 核心技术亮点 | 面试常见追问与应答要点 |
| :--- | :--- | :--- |
| **知识库构建** | Markdown 结构感知切块 + 稳定指纹生成 + 纯内存轻量索引 | **追问**：为什么不直接按固定 500 字切块？<br>**应答**：固定切块会割裂政策的“前提条件”与“责任方说明”。通过 Heading 语义解析，将条款小标题作为上下文元数据注入切块，检索召回更精准。 |
| **混合检索工程** | BM25 + BGE 向量召回 + RRF 融合 + Cross-Encoder 交互重排 | **追问**：既然有了强语义向量模型，为什么还要保留 BM25？<br>**应答**：电商政策中存在大量“7天”、“59元”、“极速退款”等精确数字与专有缩写，纯向量易因语义平滑出现漏检。BM25 保障精确词命中，RRF 解决异构分值不可比的问题。 |
| **自适应受控补检** | LangGraph 状态机 + 防假充分判定 + 无新证据早停 + 降级容错 | **追问**：Self-RAG 循环如何防止死循环和幻觉扩散？<br>**应答**：建立了三态判定（充分/不充分/评审不可用），每轮比对新增 Chunk 集合；若无增量则直接早停，防止盲目改写空耗算力；模型异常时不误判为充分。 |
| **生产级可靠性** | 忠实度自检 (Faithfulness) + 证据引用对齐 + 边界拦截 | **追问**：用户如果要求“帮我取消订单”，系统会怎么处理？<br>**应答**：系统定位为政策问答，护栏在路由阶段判定写操作越界，优雅导流至订单中心，严格防止 Agent 越权操作或产生不可控行为。 |

---

## 8. 求职简历标准草稿（实测对齐升级版）

### 8.1 简历项目经历陈述（已回填 `report_20261113_082135.json` 实测数据）

> **电商售后政策智能问答系统｜Adaptive & Graph-Enhanced RAG**  
> **技术栈**：Python、FastAPI、LangGraph、DeepSeek、BGE、BM25、Cross-Encoder、Qdrant、NetworkX、SSE
> 
> **项目概述**：面向电商退换货、运费责任与保修咨询场景，构建融合条款拓扑图谱、版本时效过滤与自适应多跳补检的 RAG 系统，解决通用总则与类目特例混淆、跨文档隐式关联漏检及过期规则误引等痛点。
> - **结构化解析与双擎索引**：针对含跨页免责条款与折旧表格的政策文档，实现 Markdown 结构感知父子分块与表格完整性保护；提取生效区间与品类元数据持久化至 Qdrant 支持标量前置过滤，并基于 NetworkX 构建【下位细则-上位总则-争议举证】条款拓扑依赖图（32 节点、26 条有向边）。
> - **图谱增强混合检索与重排**：采用 BM25 + BGE 稠密向量双路召回与 RRF 倒数排名融合，引入 1-Hop 条款拓扑扩展与 Cross-Encoder 门控精排，并按 Lex Specialis 效力优先级（特例优于总则）回填父文档；在 **145** 条基准集上，复杂问题完整证据覆盖率（ComplexComplete@5）由 **73.83%** 提升至 **93.46%**（无图扩展基线为 **89.72%**，全集 Complete@5 由 **79.31%** 提升至 **95.17%**），Recall@5 达 **97.70%**（Precision@1 稳定保持 **93.10%**）。
> - **自适应路由与多跳补检**：基于 LangGraph 构建 Adaptive-RAG 路由（直答/单跳/多跳/澄清/拒答），对复合问题执行子查询拆解、证据充分性评估与无增量早停补检，结合多轮关键条件澄清，历史与现行版本匹配准确率达 **100%**。
> - **幻觉抑制与流式工程交付**：引入条款级引用锚点校验与 Faithfulness 忠实度自纠错闭环，无据拒答与越界拦截 F1 达 **100%**；基于 FastAPI + SSE 构建实时节点流式服务、`/capabilities` 降级探针与全链路阶段耗时监控，检索重排（含图谱扩展）P95 延迟 **0.89** 秒（离线端到端 P95 延迟 **0.91** 秒）。

### 8.2 实测指标与来源映射（严格严谨，溯源至 `eval/reports/report_20261113_082135.json`）
- `[N] = 145`：冻结基准测试集样本数（`eval/dataset/jd_eval_benchmark.jsonl` 共 145 条单轮/跨版本/多源复合评测题 + 31 条多轮追问轮次，总计 176 轮评测；另有 `jd_eval_guardrail_and_correction.jsonl` 共 35 条越界拒答/无据拒答/纠偏专项题）；
- `[A]% -> [B]%`：
  - 复杂多源/跨版本子集（`N_complex = 107`）完整证据覆盖率 `ComplexComplete@5`：纯 BM25 基线 **73.83%** $\to$ 无图精排基线 **89.72%** $\to$ 完整 ClauseGraph 流水线 **93.46%**；
  - 全量基准集（`N = 145`）完整证据覆盖率 `Complete@5`：纯 BM25 基线 **79.31%** $\to$ 无图精排基线 **92.41%** $\to$ 完整 ClauseGraph 流水线 **95.17%**；
  - 文档级 `Precision@1`：纯 BM25 基线 **74.48%** $\to$ 完整流水线 **93.10%**；
- `[C]% = 97.70%`（Recall@5）：完整 ClauseGraph + Cross-Encoder 门控重排流水线在 Top-5 的文档级 `Recall@5`（`MRR = 0.9591`）；
- `[V]% = 100.0%`：历史与现行版本匹配准确率（`version_isolation_rate = 100.0%` 且 `version_contrast_both_rate = 100.0%`）；
- `[F]% = 100.0%`：无据拒答与越界拦截 F1（越界拒答准确率 `11/11 = 100.0%`，无据拒答防胡编率 `13/13 = 100.0%`，纠偏检索准确率 `11/11 = 100.0%`，145 条正常样本误拒答数 `0`）；
- `[T] = 0.89 秒（888.21ms，检索重排含图谱扩展 P95）/ 0.91 秒（910.15ms，离线全链路端到端 P95）`：全链路阶段耗时监控实测 P95。

---
*注：本执行计划为 Policy-RAG 项目唯一的顶层设计与验收指南。后续任何子任务推进均须符合上述数据契约与架构约定。*
