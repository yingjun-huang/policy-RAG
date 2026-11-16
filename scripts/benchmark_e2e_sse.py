"""F3 Benchmark: Single-concurrency (uncontended) E2E latency and SSE streaming TTFT measurement."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import percentile
from policy_rag.config import Settings
from policy_rag.schemas.chat import ChatRequest
from policy_rag.service import PolicyContainer

DEFAULT_SAMPLE_QUERIES: list[tuple[str, str]] = [
    ("direct", "你好"),
    ("single", "退衣服时赠送的皮带找不到了，购买页写了赠品价20元。按这个价扣20元可以吗？"),
    ("single", "家里买的电视开发票写了3000元，签收第20天申请退货，折旧费怎么算？"),
    ("single", "在自营买了一单常规日用品发普通快递去新疆，包裹重3.2kg，偏远附加费是多少？"),
    ("single", "手机收到好几天了，现在还能退吗？"),
    ("multi", "2023年7月在自营下过一单70元、3kg的日用品普通快递到上海，和今年同样订单基础运费有什么变化？"),
    ("multi", "双十一当天上午11点整买的支持211限时达和晚必赔的电水壶没按时送达，怎么判断超时和赔付？"),
    ("multi", "给台式机装CPU时涂了点散热硅脂，签收第3天检测出厂电路故障，还能退吗，检测和运费谁付？"),
    ("refuse", "帮我取消这笔订单并直接退回运费。"),
]


async def measure_single_stream(
    samples: list[tuple[str, str]],
    *,
    disable_llm_cache: bool = True,
) -> dict[str, Any]:
    settings = Settings(
        llm_cache_enabled=not disable_llm_cache,
        local_embedding_force=True,
        cross_encoder_force=True,
    )
    container = PolicyContainer(settings=settings)
    await container.init_vector_index()
    svc = container.qa_service

    # Warm up ONNX / PyTorch models once before timing
    await svc.answer(ChatRequest(message="你好"))
    await svc.answer(ChatRequest(message="七天无理由怎么算？"))

    records: list[dict[str, Any]] = []
    for expected_route, query in samples:
        t_start = time.perf_counter()
        first_node_ms: float | None = None
        ttft_ms: float | None = None
        done_payload: dict[str, Any] = {}

        async for evt in svc.stream_answer(ChatRequest(message=query)):
            elapsed = (time.perf_counter() - t_start) * 1000
            ename = evt["event"]
            if ename == "node_update" and first_node_ms is None:
                first_node_ms = elapsed
            elif ename == "token" and ttft_ms is None:
                ttft_ms = elapsed
            elif ename == "done":
                done_payload = evt["data"]

        total_ms = (time.perf_counter() - t_start) * 1000
        records.append(
            {
                "expected_route": expected_route,
                "predicted_route": done_payload.get("strategy"),
                "action": done_payload.get("action"),
                "faithfulness_checked": done_payload.get("faithfulness_checked"),
                "first_node_ms": round(first_node_ms or total_ms, 2),
                "ttft_ms": round(ttft_ms or total_ms, 2),
                "total_ms": round(total_ms, 2),
                "stage_latency": done_payload.get("latency", {}),
            }
        )

    all_total = [r["total_ms"] for r in records]
    all_ttft = [r["ttft_ms"] for r in records]
    all_first_node = [r["first_node_ms"] for r in records]
    single_totals = [r["total_ms"] for r in records if r["predicted_route"] == "single"]
    multi_totals = [r["total_ms"] for r in records if r["predicted_route"] == "multi"]

    return {
        "mode": "uncontended_concurrency_1",
        "llm_model": settings.active_llm_model,
        "reranker_backend": getattr(container.reranker, "backend", "unknown"),
        "cross_encoder_use_onnx": settings.cross_encoder_use_onnx,
        "faithfulness_risk_gate_enabled": settings.faithfulness_risk_gate_enabled,
        "llm_cache_enabled": settings.llm_cache_enabled,
        "samples": len(records),
        "summary": {
            "first_node_p50_ms": round(percentile(all_first_node, 50), 2),
            "first_node_p95_ms": round(percentile(all_first_node, 95), 2),
            "sse_ttft_p50_ms": round(percentile(all_ttft, 50), 2),
            "sse_ttft_p95_ms": round(percentile(all_ttft, 95), 2),
            "e2e_total_p50_ms": round(percentile(all_total, 50), 2),
            "e2e_total_p95_ms": round(percentile(all_total, 95), 2),
            "single_p50_ms": round(percentile(single_totals, 50), 2) if single_totals else None,
            "single_p95_ms": round(percentile(single_totals, 95), 2) if single_totals else None,
            "multi_p50_ms": round(percentile(multi_totals, 50), 2) if multi_totals else None,
            "multi_p95_ms": round(percentile(multi_totals, 95), 2) if multi_totals else None,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure single-concurrency E2E & SSE TTFT latency")
    parser.add_argument("--use-cache", action="store_true", default=False, help="Allow SQLite LLM cache hits")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "eval" / "reports" / "sse_latency_report.json"),
    )
    args = parser.parse_args()

    result = asyncio.run(measure_single_stream(DEFAULT_SAMPLE_QUERIES, disable_llm_cache=not args.use_cache))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Saved SSE latency benchmark to {out_path}")


if __name__ == "__main__":
    main()
