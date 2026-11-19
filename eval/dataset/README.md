# 评测数据入口

- jd_eval_final.jsonl：正式统一集，189场景、51追问、240轮。
- jd_eval_final_manifest.json：公开副本运行清单，仅README哈希随发布说明更新。
- jd_eval_benchmark.jsonl、jd_eval_guardrail_and_correction.jsonl、jd_eval_supplement_v1.jsonl、shards/：来源与回归夹具。测试、校验及独立实验仍引用，因此保留；不分别充当当前完整入口。

只读检查：python -m eval.build_dataset --check。不要无参数重建覆盖数据。
