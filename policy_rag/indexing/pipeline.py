"""Knowledge ingestion pipeline for scanning and batch chunking policy documents."""

from __future__ import annotations

import logging
from pathlib import Path

from policy_rag.indexing.chunking import (
    MAX_CHUNK_CHARS,
    MAX_PARENT_CHARS,
    OVERLAP_CHARS,
    chunk_markdown_file,
)
from policy_rag.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def load_and_chunk_knowledge(
    knowledge_dir: Path | str,
    *,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    max_parent_chars: int = MAX_PARENT_CHARS,
) -> list[RetrievedChunk]:
    """Scans knowledge directory for Markdown files and builds structured chunks."""
    dir_path = Path(knowledge_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {dir_path.resolve()}")

    md_files = sorted(dir_path.glob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found in {dir_path.resolve()}")
        return []

    all_chunks: list[RetrievedChunk] = []
    for file_path in md_files:
        try:
            chunks = chunk_markdown_file(
                file_path,
                root=dir_path,
                max_chunk_chars=max_chunk_chars,
                overlap_chars=overlap_chars,
                max_parent_chars=max_parent_chars,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path.name}: {e}")
            raise

    logger.info(
        f"Knowledge ingestion completed: {len(md_files)} documents parsed into {len(all_chunks)} chunks."
    )
    return all_chunks
