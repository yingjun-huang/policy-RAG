# 最终评测结果：2026-11-19

本轮是用户授权的最终版本评测，源码、知识库、数据、模型配置和评分协议均冻结。189场景、51追问、240轮全部完成且主裁判响应有效。生成与主裁均为deepseek-chat；不重复交叉裁判、检索消融和性能实验。每条返回答案及有效低分均保留，无按分数重写、调参或重跑。

## 核心指标

|指标|最终结果|口径|
|---|---|---|
|回答严格正确率|85.42%（205/240）|correct计正确|
|回答加权正确率|86.46%|correct/partial/incorrect分别计1/0.5/0；205正确、5部分、30错误|
|路由准确率|82.58%（109/132）|有明确策略标签的首问；single与table_lookup映射single，multi_hop与version_contrast映射multi，办理越界映射refuse|
|计算题严格/加权正确率|80.85% / 80.85%|冻结47个gold_calculation目标的最终回答评分，包含规则/公式选择和输出协议执行|
|计算器使用|32/47|最终目标答案包含执行轨迹；没有执行轨迹不算使用|
|混合检索文档级P@1|74.29%→89.71%|175标注首问、BM25与完整检索方案对照，本轮复用已完成的当前版本本地消融|
|混合检索文档级Recall@5|88.86%→97.52%|逻辑文档去重后Top-5召回|
|图补全复杂Complete@5|90.18%→93.75%|112复杂首问、无图混合精排与图补全方案对照|

## 自适应检索与剩余表现

实际multi轮次61轮，其中46轮第一轮评审充分。实际进行至少二轮且有gold文档的5轮，首轮完整证据覆盖4/5（80.0%），最终5/5（100.0%），新增补齐1轮。这是同一批补检轮次的累积文档覆盖，不是Top-5指标，不是答案改善对照，不代表所有问题必须二轮。

程序返回错误标签的答案共15轮：{"generation_invalid_response": 15}。全部保留计分。计算表达式执行一致性检查70/70；这只验证已执行表达式的结果，不能替代模型输出协议遵从、公式选择或计算题答案正确率。具体错误与部分正确样本见FINAL_ERROR_DETAILS.json，不据此追加修改或费用。

新版交叉裁判未测量。旧版90.42%的交叉加权成绩仍是旧版材料，不混入这套新分数。引用合法率、核验触发率、小样本纠错改善和CPU性能不作为本轮简历核心成果。自建功能评测集不代表外部通用基准。

本轮API使用：{"returned_generate_calls": {"pipeline": 815, "primary_judge": 240}, "reported_total_tokens": {"pipeline": 1335149, "primary_judge": 3360741}, "failed_generate_calls": {}, "note": "Provider usage for this invocation; transport retries and supplier billing may differ. Journal retains stage responses."}。这是调用返回的token统计，不是供应商账单；API阶段原始响应保存在api_journal文件中。

## 文件

- 报告：report_20261119_053743_398780.json
- 全部预测：predictions_20261119_053743_398780.jsonl
- 冻结清单：snapshot.json
- 独立复核：VERIFIED_FINAL_RESULTS.json
- 错误明细：FINAL_ERROR_DETAILS.json
- 简历对应文案：RESUME_PROJECT_FINAL.md
