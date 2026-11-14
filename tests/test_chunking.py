"""Unit tests for markdown chunking and knowledge ingestion pipeline."""

from pathlib import Path

from policy_rag.indexing.chunking import chunk_markdown, document_id_from_path, stable_chunk_id
from policy_rag.indexing.pipeline import load_and_chunk_knowledge


def test_chunk_markdown_heading_extraction():
    md_content = """# 电商平台配送与运费政策

政策版本：2024-Q4

## 包邮门槛与时效

普通订单满59元包邮，发货后48小时内送达。

## 偏远地区附加费

新疆、西藏等偏远地区不享受满包邮政策，每单加收15元远程运费。
"""
    chunks = chunk_markdown(md_content, document_id="shipping_rule", default_title="配送政策")

    assert len(chunks) >= 2
    assert chunks[0].title == "电商平台配送与运费政策"
    assert chunks[0].section in {"包邮门槛与时效", "概述"}
    assert "59元" in chunks[0].content or "59元" in chunks[1].content
    assert chunks[0].document_id == "shipping_rule"
    assert chunks[0].chunk_id == "shipping_rule_00"


def test_sliding_window_overlap():
    long_body = "退款条款规定。" * 100  # 7 chars x 100 = 700 chars, exceeds MAX_CHUNK_CHARS (500)
    md_content = f"# 退款细则\n\n## 复杂流程\n\n{long_body}"

    chunks = chunk_markdown(md_content, document_id="refund_doc")
    assert len(chunks) > 1

    # Chunk IDs should be incrementing deterministically
    assert chunks[0].chunk_id == "refund_doc_00"
    assert chunks[1].chunk_id == "refund_doc_01"


def test_document_id_stability():
    p1 = Path("knowledge/return_policy.md")
    p2 = Path("knowledge/return_policy.md")
    assert document_id_from_path(p1) == document_id_from_path(p2)
    assert stable_chunk_id("doc_123", 5) == "doc_123_05"


def test_ingest_all_real_policy_documents():
    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge_jd"
    assert knowledge_dir.exists()

    chunks = load_and_chunk_knowledge(knowledge_dir)

    # 32 JD policy files should produce structured chunks cleanly
    assert len(chunks) >= 90

    for chunk in chunks:
        # Every single chunk must satisfy the mandatory contract
        assert chunk.document_id, "Missing document_id"
        assert chunk.chunk_id, "Missing chunk_id"
        assert chunk.content, "Missing content"
        assert chunk.title, "Missing title"
        assert chunk.source == "markdown_ingest"
        assert chunk.section, "Missing section"
        assert chunk.chunk_id.startswith(chunk.document_id)
