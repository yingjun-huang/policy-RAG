"""Unit tests for DocumentReranker and fallback mechanism."""

import pytest

from policy_rag.config import Settings
from policy_rag.providers.cross_encoder import CrossEncoderRerankProvider
from policy_rag.providers.fake import LexicalRerankProvider
from policy_rag.retrieval.reranker import DocumentReranker
from policy_rag.schemas.retrieval import RetrievedChunk


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=f"doc_{cid}",
        chunk_id=cid,
        title="政策",
        section="条款",
        content=text,
    )


@pytest.mark.asyncio
async def test_lexical_reranker():
    provider = LexicalRerankProvider()
    reranker = DocumentReranker(provider)

    chunks = [
        _chunk("c1", "关于保修期的说明，数码产品保修12个月。"),
        _chunk("c2", "关于发票申请流程，支付后即刻开具电子发票。"),
    ]

    reranked = await reranker.rerank("数码产品保修多长时间", chunks, top_n=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].source == "rerank"
    assert reranked[0].rank == 1


@pytest.mark.asyncio
async def test_cross_encoder_provider_graceful_fallback():
    # In testing mode without CROSS_ENCODER_FORCE, it should use lexical fallback safely
    settings = Settings(app_env="testing", cross_encoder_force=False)
    provider = CrossEncoderRerankProvider(settings)

    assert provider.backend == "lexical"

    scored = await provider.rerank("退货", ["7天无理由退货细则", "偏远地区不包邮"], top_n=2)
    assert len(scored) == 2
    assert scored[0][0] == 0  # index 0 matches "退货"
