"""FastAPI route definitions."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from policy_rag.schemas.chat import CapabilitiesResponse, ChatRequest, ChatResponse
from policy_rag.service import PolicyContainer

router = APIRouter()


def get_container(request: Request) -> PolicyContainer:
    return request.app.state.container


@router.get("/health", tags=["system"])
async def health_check(container: PolicyContainer = Depends(get_container)) -> dict[str, str]:
    return {
        "status": "healthy",
        "app_name": container.settings.app_name,
        "app_env": container.settings.app_env,
        "version": "0.1.0",
    }


@router.get("/capabilities", response_model=CapabilitiesResponse, tags=["system"])
async def capabilities_check(
    container: PolicyContainer = Depends(get_container),
) -> CapabilitiesResponse:
    """Expose runtime retrieval, ClauseGraph, embedding, reranker, and LLM capability status."""
    return container.get_capabilities()


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(
    payload: ChatRequest,
    container: PolicyContainer = Depends(get_container),
) -> ChatResponse:
    """Answer an e-commerce policy question and return structured verification trace."""
    return await container.qa_service.answer(payload)


@router.post("/chat/stream", tags=["chat"])
async def chat_stream_endpoint(
    payload: ChatRequest,
    container: PolicyContainer = Depends(get_container),
) -> EventSourceResponse:
    """Stream LangGraph node progress, answer token slices, and full verification payload via SSE."""

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            async for item in container.qa_service.stream_answer(payload):
                yield {
                    "event": item["event"],
                    "data": json.dumps(item["data"], ensure_ascii=False),
                }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
