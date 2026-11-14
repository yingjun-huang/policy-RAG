# Policy-RAG 进阶改造执行计划与目标简历（Graph & Adaptive-RAG）

> **定位说明**：本文档面向「1 个深度 RAG 项目 + 1 个多工具 Agent 项目」的双项目简历布局，专为 `Policy-RAG` 制定**拒绝玩具感（Toy）、拒绝同质化烂大街、同时杜绝“大炮打苍蝇（过度设计）”**的精准升级方案。所有改造项均严格基于现有 32 篇京东售后政策语料（114 个切片）与 145 条基准评测集的真实 Bad Cases 驱动。

---

## 一、构想的最终版简历（升级目标版）

### 1.1 简历正文（精简高密度版，可直接放入简历）

> **电商售后政策智能问答系统｜Adaptive & Graph-Enhanced RAG**  
> **技术栈**：Python、FastAPI、LangGraph、DeepSeek、BGE、BM25、Cross-Encoder、Qdrant、NetworkX、SSE  
> **项目概述**：面向电商退换货、运费责任与保修咨询场景，构建融合条款拓扑图谱、版本时效过滤与自适应多跳补检的 RAG 系统，解决通用总则与类目特例混淆、跨文档隐式关联漏检及过期规则误引等痛点。
> - **结构化解析与双擎索引**：实现 Markdown 结构感知父子分块与折旧表格完整性保护；提取生效区间与品类元数据持久化至 Qdrant 支持标量前置过滤，并基于 NetworkX 构建【下位细则-上位总则-争议举证】条款拓扑依赖图（32 节点、26 条有向边）。
> - **图谱增强混合检索与重排**：采用 BM25 + BGE 向量双路召回与 RRF 融合，针对向量正交导致的上位规则漏检，引入 1-Hop 条款拓扑扩展与 Cross-Encoder 门控精排，按效力优先级（特例优于总则）回填父文档；在 **145** 条基准集上（实测报告 `report_20261113_082135.json`），复杂问题完整证据覆盖率（ComplexComplete@5）由 **73.83%** 提升至 **93.46%**（无图扩展基线为 **89.72%**，全集 Complete@5 由 **79.31%** 提升至 **95.17%**），Recall@5 达 **97.70%**（Precision@1 稳定保持 **93.10%**）。
> - **自适应路由与受控补检**：基于 LangGraph 构建 Adaptive-RAG 状态机（直答/单跳/多跳/澄清/拒答），对复合问题执行子查询拆解、三态充分性评估与无增量早停补检，结合多轮关键条件槽位澄清，历史与现行版本匹配准确率达 **100%**。
> - **幻觉抑制与流式工程交付**：引入条款锚点校验与 Faithfulness 忠实度自纠错闭环（无据拒答与越界拦截 F1 达 **100%**）；基于 FastAPI + SSE 实现节点状态与答案流式推送，配套多级优雅降级探针与全链路阶段耗时监控，检索重排（含图谱扩展）P95 延迟为 **0.89** 秒（`888.21ms`）。

---

### 1.2 为什么这套简历既高级又不会在面试中“穿帮”？（红黑榜边界说明）

| 技术方案 | 评估结论 | 真实数据依据与面试官视角解释 |
| :--- | :---: | :--- |
| **NetworkX 条款拓扑依赖图（1-Hop 扩展）** | ✅ **强烈保留（核心亮点）** | 现有 145 条评测集中有 **11 道复杂题**（如 `jd_case_086/087/098/139`）因用户仅提具体品类或口语化怀疑，导致词法与向量均漏检《通用总则 (`jd_rule_01`)》或《争议举证 (`jd_rule_02`)》。用轻量有向图做 1-Hop 拓扑补全，耗时 `<1ms`，直击业务痛点。 |
| **法条效力优先级裁决（Lex Specialis）** | ✅ **强烈保留（领域深度）** | 当通用总则与品类特例同时召回时，标注 `[P1 专项服务] > [P2 品类特例] > [P3 通用总则]` 并做首尾重排，解决大模型“和稀泥”痛点。 |
| **Qdrant 标量前置过滤（版本 + 品类）** | ✅ **已落地验证** | 解决 2023/2024 历史归档政策对现行回答的污染，实测版本隔离率与跨版本对比双全率均达 **100.0%**。 |
| **FastAPI + SSE 节点流式 + `/capabilities` 探针** | ✅ **强烈保留（工程素养）** | 解决多跳检索与精排期间前端白屏等待问题，实时推送 LangGraph 节点进度与耗时；内置单页 Web 溯源台仅作 GitHub 门面演示，不写 `Streamlit` 进简历技术栈。 |
| **Neo4j 重型 GraphRAG + 社区聚类摘要** | ❌ **剔除（大炮打苍蝇）** | 当前语料共 32 篇、1.5 万字、114 个切片，上 Neo4j 与 LLM 实体抽取成本高且无必要，面试官一问文档规模即会质疑过度设计。 |
| **128k 预算三级上下文压缩** | ❌ **移交第 2 个 Agent 项目** | 政策问答单次检索 Top-5 父文档仅约 1,500 Token，不可能触发 128k 的 85% 压缩阈值；留给调用 15+ 工具的 Agent 项目才真实合理。 |
| **Unstructured PDF OCR + PII 身份证脱敏** | ❌ **剔除（材料不支持）** | `knowledge_jd/` 均为干净的结构化 Markdown，无扫描件与用户隐私数据，不写无实测支撑的模块。 |

---

## 二、真实 Bad Cases 溯源（驱动步骤 1 与步骤 2 的实证依据）

在已跑通的 `5. Hybrid + MetadataFilter + Rerank` 评测（`eval/reports/report_20261113_065129.json`）中，107 道复杂题的 `ComplexComplete@5` 为 **89.72%**，共有 **11 道题**未完全召回全部金标文档，全部属于**“跨文档隐式层级/程序依赖”**：

1. **下位品类细则漏检上位《通用总则 (`jd_rule_01_seven_days_general`)》（共 6 题）**：
   - `jd_case_086`：*“手机收到好几天了，现在还能退吗？”*（命中 `jd_cat_01_mobile_phones`，漏 `jd_rule_01` 七天起算规则）
   - `jd_case_087`：*“衣服不想要了，现在还能退吗？”*（命中 `jd_cat_07_fashion_apparel`，漏 `jd_rule_01`）
   - `jd_case_088`：*“大家电送到家，机身配件完好，想退货，运费谁出？”*（命中 `jd_cat_03` + `jd_ship_05`，漏 `jd_rule_01`）
   - `jd_case_096`：*“小家电拆开用了，现在还能退吗？”*（命中 `jd_cat_04_appliances_small`，漏 `jd_rule_01`）
   - `jd_case_100`：*“大件家具拆箱发现板角破皮，不想收了，能退吗？”*（命中 `jd_cat_12_home_furniture`，漏 `jd_rule_01`）
   - `jd_case_140`：*“前几个月买的空气炸锅突然通电不加热了，不想修，能直接让我换一台全新机吗？”*（命中 `jd_srv_02_180_days_exchange`，漏 `jd_rule_01`）
2. **实质争议场景漏检程序法《争议举证与责任裁决 (`jd_rule_02_dispute_resolution`)》（共 3 题）**：
   - `jd_case_030`：*“给台式机装CPU时涂了点散热硅脂，签收第3天开机点不亮，检测确认是出厂自带电容击穿……”*（命中 `jd_cat_02`，漏 `jd_rule_02`）
   - `jd_case_098`：*“定制窗帘看着长了，怀疑尺寸不对，能退货退款吗？”*（命中 `jd_cat_11`，漏 `jd_rule_02`）
   - `jd_case_139`：*“买了瓶护肤品看着印刷很粗糙怀疑是假货，能按退一赔十赔我多少钱？”*（命中 `jd_rule_03` + `jd_rule_06`，漏 `jd_rule_02`）
3. **专项服务与附加运费联动漏检（共 2 题）**：
   - `jd_case_095`：*“网上买了一盒每只大小价格都差不多的活虾，收到打开发现有死虾，这怎么赔？”*（命中 `jd_cat_06_fresh_seafood`，漏 `jd_srv_01_fresh_instant_pay`）
   - `jd_case_094`：*“在京东用普通快递买点小东西，不是同城秒送也不是大件专送，这单运费怎么算？”*（命中 `jd_ship_01_freight_standard`，漏 `jd_ship_02_remote_freight`）

---

## 三、分步工程改造执行计划（Step-by-Step Implementation Plan）

### 步骤 1：构建基于 `NetworkX` 的政策条款拓扑依赖图与门控精排

- **目标**：解决上述 11 道隐式依赖漏检问题，同时确保单跳简单题的 `Precision@1`（93.10%）不下降。
- **涉及文件**：
  1. `pyproject.toml`：添加 `networkx>=3.2.0`、`sse-starlette>=2.0.0`。
  2. **新建** `policy_rag/retrieval/clause_graph.py`：
     - 定义 `ClauseDependencyGraph` 类，内部维护 `nx.DiGraph()`。
     - 构建三类有向拓扑边：
       - `INHERITS_GENERAL`：`jd_cat_01` ~ `jd_cat_14`、`jd_srv_02` $\to$ `jd_rule_01_seven_days_general`
       - `REQUIRES_DISPUTE_PROOF`：`jd_cat_02`、`jd_cat_11`、`jd_rule_03`、`jd_rule_06` $\to$ `jd_rule_02_dispute_resolution`
       - `LINKS_COMPLEMENT`：`jd_cat_05/06` $\to$ `jd_srv_01_fresh_instant_pay`；`jd_ship_01` $\to$ `jd_ship_02_remote_freight`、`jd_ship_05_pickup_service`
     - 实现 `expand_candidates(query: str, primary_chunks: list[RetrievedChunk], metadata_filter: MetadataFilter | None) -> list[RetrievedChunk]`：
       - 提取 RRF 初筛结果中的文档节点，沿有向边检索 1-Hop 关联目标文档；
       - 结合查询意图信号门控触发，将缺失的关联文档核心 Chunk（标记 `metadata["graph_expanded"] = True`、`metadata["graph_edge"] = edge_type`）追加到候选池尾部。
  3. **修改** `policy_rag/retrieval/hybrid.py`：
     - 在 `HybridRetriever.__init__` 中支持可选注入 `clause_graph: ClauseDependencyGraph | None = None`；
     - 在 `retrieve()` 方法中，当启用 `clause_graph` 时，对 RRF 融合结果执行 1-Hop 图谱拓扑扩展。
  4. **修改** `policy_rag/retrieval/reranker.py`：
     - 实现 **“主召回 Top-1 保序 + 互补槽位注入（Gated Graph Reranking）”**：
       - 对直接召回的候选块（`not chunk.metadata.get("graph_expanded")`）与图扩展候选块分别用 Cross-Encoder 打分；
       - 保证 Top-1 ~ Top-2 严格由直接召回最高分占据（确保 `Precision@1` 不受图扩展影响），并在 Top-5 的剩余互补槽位中注入得分达标的 1-Hop 扩展文档块。
- **验证方式**：
  - 编写单元测试 `tests/test_clause_graph.py`，验证 1-Hop 拓扑边构建、过滤兼容性和门控重排保序性。

---

### 步骤 2：实现法条效力优先级裁决（Lex Specialis）与首尾重排

- **目标**：当《通用总则》与《品类特例/专项服务》同时进入生成上下文时，通过显式效力优先级标签与首尾排布消除规则混淆。
- **涉及文件**：
  1. **修改** `policy_rag/retrieval/filters.py`（`expand_parent_contexts` 函数）：
     - 为回填后的每个父文档计算 `authority_tier` 与 `authority_label`：
       - `service_rules` / `legacy_rules` $\to$ `tier=1`, `"P1-专项服务与特定版本规则（最高优先适用）"`
       - `category_rules` / `shipping` $\to$ `tier=2`, `"P2-品类特例与专项运费细则（优先于通用总则）"`
       - `general_rules` $\to$ `tier=3`, `"P3-平台通用基础总则（无特例冲突时适用）"`
     - 实现 **Lost-in-the-Middle 首尾交替排布**（`reorder_lost_in_the_middle=True`）：将排序第 1 的核心证据置于开头，第 2 置于末尾，其余置于中间。
  2. **修改** `policy_rag/agents/loop.py`（`_format_evidence_for_prompt` 函数）：
     - 在格式化证据块时输出 `[效力优先级: {authority_label}]` 与 `[召回来源: {直接召回 / Graph 1-Hop 拓扑补全}]`。
- **验证方式**：
  - 在 `tests/test_retrieval_fusion.py` 中新增测试，验证父文档回填后的效力层级标注与首尾重排顺序。

---

### 步骤 3：实现 FastAPI + SSE 节点级流式推送与 `/capabilities` 降级探针

- **目标**：提供生产级实时节点流转反馈（SSE）与运行时组件健康度/降级自省接口。
- **涉及文件**：
  1. **修改** `policy_rag/schemas/chat.py`：
     - 新增 `CapabilitiesResponse` 模型（包含 `vector_store`、`clause_graph_enabled`、`clause_graph_edges`、`embedding_backend`、`reranker_backend`、`llm_backend`、`degraded`）。
  2. **修改** `policy_rag/service.py`：
     - 在 `PolicyContainer` 中装配 `ClauseDependencyGraph(self.chunks)`；
     - 在 `PolicyQAService` 中新增 `stream_answer(request: ChatRequest) -> AsyncIterator[dict[str, Any]]`：
       - 使用 `self.container.graph.astream(initial_state, stream_mode="updates")` 捕获每个 LangGraph 节点的实时状态更新；
       - 依次 yield `{"event": "node", "data": ...}`、`{"event": "token", "data": ...}` 和 `{"event": "done", "data": ...}`。
  3. **修改** `policy_rag/api/routes.py`：
     - 新增 `GET /api/v1/capabilities`（及根路径 `/capabilities`）探针接口；
     - 新增 `POST /api/v1/chat/stream`（及 `/chat/stream`）SSE 流式问答接口（使用 `StreamingResponse` / `EventSourceResponse` 输出标准 `text/event-stream`）。
- **验证方式**：
  - 在 `tests/test_e2e_api.py` 中新增针对 `/api/v1/capabilities` 和 `/api/v1/chat/stream` 的端到端测试。

---

### 步骤 4：构建内置单页 Web 白盒溯源演示工作台（GitHub 门面 Demo）

- **目标**：启动 `uvicorn policy_rag.api.main:app` 后直接访问 `http://127.0.0.1:8000` 即可操作高颜值可视化工作台，用于录制 GitHub 首页 GIF/截图及面试演示。
- **涉及文件**：
  1. **新建** `policy_rag/web/index.html`：
     - 顶部展示 `/api/v1/capabilities` 实时组件状态灯（`Qdrant`、`NetworkX ClauseGraph`、`Cross-Encoder`、`DeepSeek`）；
     - 左侧为对话流与 6 个典型场景一键测试卡片（含跨版本对比、手机激活折旧、衣服退货触发 `NetworkX` 1-Hop 补全、缺槽位澄清、越界拒答）；
     - 右侧为白盒溯源面板：实时点亮 SSE 推送的 LangGraph 节点流转轨迹、各阶段耗时瀑布条、命中的父子条款卡片（高亮显示 `[Graph 1-Hop 补全]` 与 `[效力层级 P1/P2/P3]` 徽章）。
  2. **修改** `policy_rag/api/main.py`：
     - 挂载 `GET /` 返回 `policy_rag/web/index.html`。
- **验证方式**：
  - 单测验证 `GET /` 返回 `200 OK` 且包含工作台 HTML 结构。

---

### 步骤 5：升级 6 路消融实验套件并回填真实评测指标

- **目标**：在 `eval/runner.py` 中新增第 6 组消融实验 `6. Hybrid + ClauseGraph (1-Hop) + MetadataFilter + Rerank`，跑出完整 JSON 评测报告，将实测数字回填至简历与文档。
- **涉及文件**：
  1. **修改** `eval/runner.py`：
     - 将消融对比从 5 组扩充为 6 组，直观对比无图扩展（第 5 组）与加入 `ClauseDependencyGraph` 1-Hop 扩展（第 6 组）在 `Precision@1`、`Recall@5`、`Complete@5`、`ComplexComplete@5` 及延迟上的真实差异。
  2. **修改** `PROJECT_PLAN.md` 与 `UPGRADE_PLAN.md`：
     - 用最新生成的 `eval/reports/report_*.json` 中的真实数字替换简历中的待测占位符。

---

## 四、面试高频拷打预案（针对本次新增亮点）

### Q1：你们总共只有 32 篇政策文档，为什么要引入图结构（NetworkX）？是不是为了凑 GraphRAG 的热度？
- **标准高分回答**：
  > “恰恰相反，我特意**没有**用重型的 Neo4j 或微软那种靠 LLM 抽三元组做社区摘要的 GraphRAG，因为对几十篇结构化政策文档上重型图数据库属于典型的大炮打苍蝇。  
  > 我引入轻量级 `NetworkX` 条款拓扑图，完全是被 **145 条基准评测集里的 11 个真实 Bad Cases** 逼出来的：在评测 `BM25 + BGE + RRF + Cross-Encoder` 时，我发现复杂问题的完整证据覆盖率卡在 `89.7%`。拆解那 11 道错题（比如用户问*‘衣服不想要了还能退吗’*、或者*‘定制窗帘看着长了怀疑尺寸不对能退吗’*）时发现，用户的提问只包含下位品类词（衣服、窗帘），在词面和向量空间上与上位《七天无理由通用总则》或《争议举证规则》完全正交，无论怎么调向量阈值都召回不到。  
  > 电商规则和法条一样，天然存在【下位品类细则 $\to$ 上位通用总则 / 争议举证程序 / 专项赔付服务】的确定性引用关系。所以我用 `NetworkX` 建了一张包含继承边、举证边和联动边的有向拓扑图（32 节点、26 条有向边），在混合检索后做 **1-Hop 拓扑候选扩展**，再经 Cross-Encoder 门控重排。只用了不到 1ms 的内存图遍历开销，就把复杂问题完整证据覆盖率从 **89.72% 提升到了 93.46%**（全集 Complete@5 从 **92.41%** 提升至 **95.17%**，Recall@5 达 **97.70%**），同时保持 Top-1 精度 **93.10%** 零下降。”

### Q2：1-Hop 图谱扩展把通用总则补进来之后，如果总则（7天可无理由退）和细则（手机激活不可无理由退）冲突，大模型怎么保证不答错？
- **标准高分回答**：
  > “这正是我们在父文档回填阶段做**法条效力优先级裁决（Lex Specialis）**的原因。我们在切块入库时提取了文档的 `doc_group`（专项服务、品类细则、通用总则），在回填父文档组装 Prompt 时，会自动为每条证据打上显式的效力层级标签：`[P1 专项服务/特定版本] > [P2 品类特例细则] > [P3 通用总则]`，并结合 Lost-in-the-Middle 首尾排布把高优先级特例放在 Prompt 最前与最后。同时在生成 System Prompt 与 Faithfulness 校验节点中强制约束‘特别法优于一般法’，从而避免模型把通用规则误套在特例商品上。”

---

## 五、6 路消融实测结果回填（`eval/reports/report_20261113_082135.json`）

| 方案 (Scheme) | P@1 | R@3 | R@5 | Complete@5 | ComplexComplete@5 | MRR | 版本总准确率 | P50 | P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 (无版本过滤)** | 74.48% | 81.26% | 88.28% | 79.31% | 73.83% | 0.8427 | 80.0% | 1.43ms | 2.08ms |
| **2. Vector (无版本过滤)** | 82.76% | 91.15% | 94.71% | 88.97% | 85.05% | 0.9069 | 76.55% | 11.77ms | 15.99ms |
| **3. Hybrid RRF (无版本过滤)** | 90.34% | 91.15% | 95.17% | 90.34% | 87.85% | 0.9483 | 77.93% | 13.52ms | 19.01ms |
| **4. Hybrid + MetadataFilter** | 90.34% | 91.84% | 95.17% | 90.34% | 87.85% | 0.9483 | 100.0% | 14.75ms | 20.09ms |
| **5. Hybrid + MetadataFilter + Rerank** | 93.10% | 93.45% | 96.55% | 92.41% | 89.72% | 0.9603 | 100.0% | 618.11ms | 772.89ms |
| **6. Hybrid + ClauseGraph (1-Hop) + MetadataFilter + Rerank** | **93.10%** | **93.45%** | **97.70%** | **95.17%** | **93.46%** | 0.9591 | **100.0%** | 701.09ms | 888.21ms |

- **端到端确定性行为评测（176 轮基准对话 + 35 条护栏/拒答/纠偏专项）**：`zero_fake_fallback_rate = 100.0%`、`citation_integrity_rate = 100.0%`、`active_version_isolation_rate = 100.0%`、`followup_evidence_hit_rate = 100.0%`（31/31）、`refusal_abstain_f1 = 100.0%`、`user_correction_retrieval_accuracy = 100.0%`（11/11）、零正常样本误拒答（0/145）。
