# Policy-RAG：电商售后政策问答系统

面向退换货、运费核算与保修咨询的仿真政策RAG项目。基于32份Markdown文档，支持版本过滤、规则依赖补全、受控补检、澄清与无据留白。

**技术栈**：Python · FastAPI · LangGraph · DeepSeek · BGE · BM25 · Cross-Encoder · Qdrant · NetworkX · SSE

![Web对话与引用界面](docs/assets/workbench-demo.png)

## 当前正式版本

2026-11-19最终应用评测：**189个场景、51个追问、240轮问答**。随后仅为保存答案补充Reasoner接口评分，原答案和主裁成绩保持冻结。统一数据见[正式评测集](eval/dataset/jd_eval_final.jsonl)。

| 指标 | 实测结果 |
|---|---:|
| Chat接口主裁加权 / 严格正确率 | 86.46% / 85.42% |
| Reasoner接口复核加权 / 严格正确率 | 82.50% / 78.75% |
| 两次评审判定一致率 | 91.25%（219/240） |
| 策略路由准确率 | 82.58%（109/132） |
| BM25 → 完整检索P@1 / Recall@5 | 74.29% → 89.71% / 88.86% → 97.52% |
| 无图混合精排 → 图补全复杂Complete@5 | 90.18% → 93.75%（112首问） |

请求使用deepseek-chat与deepseek-reasoner；当前服务端均返回deepseek-flash，接口检查显示思考模式不同，因此属于**接口双模式复核**，不宣称两个独立模型或人工验证。完整口径见[评测说明](docs/EVALUATION.md)、[精简结果](eval/reports/RESULTS.md)，[补充评分证据](eval/reports/final_20261119/cross_deepseek_reasoner_20261119/CROSS_RESULTS.md)单独保留，原主裁报告不改写。

## 系统流程

~~~mermaid
flowchart LR
    U[用户] --> API[FastAPI JSON / SSE]
    API --> G[注入与交易意图拦截]
    G --> R[LangGraph四路路由]
    R --> D[直答 / 拒答]
    R --> S[单跳混合检索]
    R --> M[子查询拆解与最多两轮补检]
    S --> E[版本过滤 / 1-Hop依赖扩展 / CE门控精排]
    M --> E
    E --> A[回答 / 澄清 / 无据留白]
    A --> V[引用ID校验与风险门控忠实度核验]
    V --> O[SSE节点状态与校验后答案]
    D --> O
~~~

- **解析**：标题感知父子分块，超长表格按行切分并重复表头。
- **检索**：BM25+BGE双路召回、RRF融合、Qdrant元数据过滤及生效期校验。
- **补全**：config/policy_graph.yaml定义依赖，NetworkX进行1-Hop扩展，Cross-Encoder评分筛选补充证据。
- **流程**：LangGraph编排直答、单跳、多跳、拒答；补检最多两轮，无新增证据早停。
- **输出**：SSE先推送节点进度，生成与校验后分段发送答案。引用ID存在不等同于逐句支持。

## 安装与运行

需要Python 3.11+；本次评测环境为Python 3.12.7，依赖版本见冻结清单。

~~~bash
pip install -e ".[ml,dev]"
cp .env.example .env
~~~

PowerShell可用Copy-Item创建尚不存在的.env。在.env填写自己的DEEPSEEK_API_KEY；也可配置LLM_API_KEY、LLM_BASE_URL、LLM_MODEL。

~~~bash
uvicorn policy_rag.api.main:app --host 127.0.0.1 --port 8000
~~~

打开http://127.0.0.1:8000；API文档/docs；能力自检/api/v1/capabilities；问答POST /api/v1/chat；SSE问答POST /api/v1/chat/stream。

示例默认CROSS_ENCODER_USE_ONNX=false、风险门控开启；设置CROSS_ENCODER_USE_ONNX=true启用量化后端。正式评测脚本显式固定ONNX INT8。Qdrant默认内存嵌入式运行。

密钥与模型权重不提交。首次本地推理可能下载模型；完整冻结评测要求预先准备ONNX文件，见[复现说明](docs/EVALUATION.md)。

## 验证与评测

~~~bash
# 离线测试，不调用真实模型API
python -m pytest -q

# 只读校验来源分片与兼容数据
python -m eval.build_dataset --check
~~~

最终应用评测入口为scripts/run_closure_final_20261119.py；补充评分复用scripts/cross_model_judge.py的评分函数，并逐条保存结果。已发布结果无需重跑，历史入口和独立实验不替代当前240轮口径。实际模型调用须明确授权，复现约束见评测说明。

## 目录与边界

- policy_rag/：应用代码、API与Web。
- knowledge_jd/：32份仿真政策，包含有业务用途的历史版本。
- config/：规则依赖图。
- eval/dataset/：正式数据与来源/回归夹具；eval/judges/：评分协议。
- eval/reports/RESULTS.md、metrics.json：当前精简结果；final_20261119/：冻结主裁和补充评分证据。
- scripts/、tests/：评测工具与离线测试。
- docs/：[评测说明](docs/EVALUATION.md)、[审查结论](docs/PROJECT_REVIEW.md)、[最终简历](docs/RESUME.md)。

旧命名数据、来源分片及policy_rag/retrieval/legacy仍被测试、校验或消融引用，因此保留。它们是当前依赖和基线，正式完整评测只读统一数据。历史补丁、候选、讨论和旧成绩存于独立本地归档，不随本目录提交。

本项目为功能展示与自建评测，不能视为独立盲测或线上分布。意图、年份与风险识别包含业务启发式；未发现题号/gold绑定回答的实现。当前仍有模型策略误判、过度澄清及计算占位符协议错误；已执行算术一致不保证公式选择或最终答案正确。会话存于内存，不执行真实订单操作。政策为项目仿真，不构成真实平台承诺。

LICENSE：MIT。
