"""End-to-end and schema verification tests for Stage 1 baseline."""

import pytest
from httpx import ASGITransport, AsyncClient

from policy_rag.api.main import create_app
from policy_rag.config import Settings
from policy_rag.providers.fake import FakeLLMProvider
from policy_rag.schemas.chat import CapabilitiesResponse, ChatResponse
from policy_rag.schemas.retrieval import ExitReason, GradeStatus, RetrievalStrategy
from policy_rag.service import PolicyContainer


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["app_env"] == "testing"


@pytest.mark.asyncio
async def test_chat_baseline_contract(client: AsyncClient):
    payload = {
        "message": "普通商品退货的运费由谁承担？",
        "user_id": "u_test_01",
    }
    resp = await client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    # Validate against strict Pydantic model
    validated = ChatResponse.model_validate(data)

    assert validated.user_query == payload["message"]
    assert validated.strategy == RetrievalStrategy.SINGLE
    assert validated.final_grade_status == GradeStatus.SUFFICIENT
    assert validated.exit_reason == ExitReason.SINGLE_HOP_DONE
    assert len(validated.evidence) > 0
    assert validated.citations == []  # Offline model gave no citation IDs; reference evidence remains visible.
    # Strict grounding check: citation chunk_id must exist in evidence
    evidence_ids = {c.chunk_id for c in validated.evidence}
    for cite in validated.citations:
        assert cite.chunk_id in evidence_ids

    # Latency tracking breakdown (full-pipeline stage breakdown)
    assert validated.latency.total_ms > 0
    assert validated.latency.retrieval_ms >= 0
    assert validated.latency.rerank_ms >= 0
    assert validated.latency.guardrail_ms >= 0
    assert len(validated.hop_traces) == 1
    assert validated.hop_traces[0].hop_index == 1


@pytest.mark.asyncio
async def test_chat_validation_empty_query(client: AsyncClient):
    resp = await client.post("/api/v1/chat", json={"message": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_validation_oversized_query(client: AsyncClient):
    resp = await client.post("/api/v1/chat", json={"message": "a" * 4001})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_custom_container_override():
    canned_answer = "定制商品若存在严重外观瑕疵或功能质量问题，自签收起15天内均可免费退货并包邮。"
    import json
    custom_llm = FakeLLMProvider(canned={"已有政策证据":'{"sufficient":true}',"定制": json.dumps({"action": "answer", "answer": canned_answer, "citation_chunk_ids": []}, ensure_ascii=False)})
    container = PolicyContainer(
        settings=Settings(app_env="testing"),
        llm=custom_llm,
    )
    app = create_app(container)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as custom_client:
        resp = await custom_client.post("/api/v1/chat", json={"message": "定制商品能不能退"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == canned_answer


@pytest.mark.asyncio
async def test_capabilities_endpoint(client: AsyncClient):
    resp = await client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    caps = CapabilitiesResponse.model_validate(resp.json())
    assert caps.clause_graph_enabled is True
    assert caps.clause_graph_nodes >= 32
    assert caps.clause_graph_edges >= 10
    assert caps.chunk_count > 0
    assert caps.vector_store == "qdrant"


@pytest.mark.asyncio
async def test_chat_stream_sse_events(client: AsyncClient):
    import json

    payload = {
        "message": "普通商品退货的运费由谁承担？",
        "user_id": "u_stream_01",
    }
    events: list[tuple[str, dict]] = []
    async with client.stream("POST", "/api/v1/chat/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        current_event = "message"
        data_lines: list[str] = []
        async for raw_line in resp.aiter_lines():
            line = raw_line.rstrip("\r\n")
            if line == "":
                if data_lines:
                    events.append((current_event, json.loads("\n".join(data_lines))))
                    current_event = "message"
                    data_lines = []
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            events.append((current_event, json.loads("\n".join(data_lines))))

    event_names = [e[0] for e in events]
    assert "status" in event_names
    assert "node_update" in event_names
    assert "token" in event_names
    assert "done" in event_names

    done_payload = next(data for name, data in events if name == "done")
    validated = ChatResponse.model_validate(done_payload)
    assert validated.user_query == payload["message"]
    assert len(validated.evidence) > 0
    streamed_answer = "".join(data["delta"] for name, data in events if name == "token")
    assert streamed_answer == validated.answer


@pytest.mark.asyncio
async def test_workbench_root_html(client: AsyncClient):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Policy-RAG" in resp.text
    assert "/api/v1/chat/stream" in resp.text
    assert "/api/v1/capabilities" in resp.text
