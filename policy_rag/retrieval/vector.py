"""Qdrant and In-memory vector stores for dense retrieval with scalar payload pre-filtering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import Any
import uuid

from policy_rag.providers.base import EmbeddingProvider
from policy_rag.retrieval.filters import matches_metadata_filter
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk

logger = logging.getLogger(__name__)


def _dot_product(a: list[float], b: list[float]) -> float:
    """Computes dot product between two float vectors (equivalent to cosine when normalized)."""
    return sum(x * y for x, y in zip(a, b, strict=False))


def stable_point_id(chunk_id: str) -> str:
    """Derives a deterministic UUID string from chunk_id for idempotent Qdrant upserts."""
    return str(uuid.UUID(hashlib.md5(chunk_id.encode("utf-8"), usedforsecurity=False).hexdigest()))


@dataclass
class VectorRecord:
    chunk: RetrievedChunk
    embedding: list[float]


class InMemoryVectorStore:
    """Pure in-memory vector store maintaining dense vector representations."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        existing = {rec.chunk.chunk_id: idx for idx, rec in enumerate(self._records)}
        for rec in records:
            if rec.chunk.chunk_id in existing:
                self._records[existing[rec.chunk.chunk_id]] = rec
            else:
                self._records.append(rec)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        if not self._records:
            return []

        scored: list[tuple[float, VectorRecord]] = [
            (_dot_product(query_vector, rec.embedding), rec)
            for rec in self._records
            if matches_metadata_filter(rec.chunk, metadata_filter)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (score, rec) in enumerate(scored[:top_k], start=1):
            chunk = rec.chunk.model_copy(deep=True)
            chunk.score = float(score)
            chunk.rank = rank
            chunk.source = "vector"
            results.append(chunk)

        return results

    def __len__(self) -> int:
        return len(self._records)


class QdrantVectorStore:
    """Qdrant-backed vector store supporting scalar payload indexing and pre-filtering."""

    def __init__(
        self,
        *,
        collection: str = "policy_chunks",
        dim: int = 512,
        url: str = "",
        path: str = "",
    ) -> None:
        self._collection = collection
        self._dim = dim
        self._url = url.strip()
        self._path = path.strip()
        self._client: Any = None
        self._count: int = 0

    def _ensure_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            if self._url:
                self._client = QdrantClient(url=self._url, timeout=10.0)
            elif self._path:
                self._client = QdrantClient(path=self._path)
            else:
                self._client = QdrantClient(location=":memory:")
        return self._client

    def ensure_collection(self, dim: int | None = None) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        target_dim = dim or self._dim
        client = self._ensure_client()
        existing = {c.name for c in client.get_collections().collections}
        if self._collection not in existing:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=target_dim, distance=Distance.COSINE),
            )
            if self._url:
                for field_name in ("document_status", "doc_group", "policy_family", "logical_doc_id"):
                    try:
                        client.create_payload_index(
                            collection_name=self._collection,
                            field_name=field_name,
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        from qdrant_client.models import PointStruct

        actual_dim = len(records[0].embedding) if records[0].embedding else self._dim
        self._dim = actual_dim
        client = self._ensure_client()
        self.ensure_collection(dim=actual_dim)

        points: list[PointStruct] = []
        for rec in records:
            chunk = rec.chunk
            meta = dict(chunk.metadata or {})
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "content": chunk.content,
                "section": chunk.section,
                "parent_id": chunk.parent_id,
                "chunk_index": chunk.chunk_index,
                "logical_doc_id": meta.get("logical_doc_id", chunk.document_id),
                "doc_group": meta.get("doc_group", "general"),
                "policy_family": meta.get("policy_family"),
                "document_status": str(meta.get("document_status", "current")).lower(),
                "version": meta.get("version"),
                "effective_from": meta.get("effective_from"),
                "effective_to": meta.get("effective_to"),
                "effective_to_exclusive": bool(meta.get("effective_to_exclusive", True)),
                "merchant_types": meta.get("merchant_types"),
                "metadata": meta,
            }
            points.append(
                PointStruct(
                    id=stable_point_id(chunk.chunk_id),
                    vector=rec.embedding,
                    payload=payload,
                )
            )

        client.upsert(collection_name=self._collection, points=points)
        try:
            self._count = int(client.count(collection_name=self._collection, exact=True).count)
        except Exception:
            self._count = len(points)

    @staticmethod
    def _build_qdrant_filter(metadata_filter: MetadataFilter | None) -> Any:
        if metadata_filter is None:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        must: list[Any] = []
        must_not: list[Any] = []

        if not metadata_filter.include_archived:
            must_not.append(
                FieldCondition(
                    key="document_status",
                    match=MatchValue(value="archived"),
                )
            )
        if metadata_filter.doc_groups:
            must.append(
                FieldCondition(
                    key="doc_group",
                    match=MatchAny(any=list(metadata_filter.doc_groups)),
                )
            )
        if metadata_filter.policy_families:
            must.append(
                FieldCondition(
                    key="policy_family",
                    match=MatchAny(any=list(metadata_filter.policy_families)),
                )
            )
        if metadata_filter.merchant_types:
            must.append(
                FieldCondition(
                    key="merchant_types",
                    match=MatchAny(any=list(metadata_filter.merchant_types)),
                )
            )

        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        if len(self) == 0:
            return []

        client = self._ensure_client()
        qdrant_filter = self._build_qdrant_filter(metadata_filter)
        fetch_limit = max(top_k * 2, 10) if metadata_filter and metadata_filter.as_of else top_k

        resp = client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=fetch_limit,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []
        for point in resp.points:
            payload = point.payload or {}
            meta = dict(payload.get("metadata") or {})
            chunk = RetrievedChunk(
                document_id=str(payload.get("document_id", "")),
                chunk_id=str(payload.get("chunk_id", point.id)),
                content=str(payload.get("content", "")),
                title=str(payload.get("title", "")),
                source="vector",
                section=payload.get("section"),
                parent_id=payload.get("parent_id"),
                score=float(point.score),
                rank=None,
                chunk_index=int(payload.get("chunk_index") or 0),
                metadata=meta,
            )
            if not matches_metadata_filter(chunk, metadata_filter):
                continue
            chunk.rank = len(results) + 1
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results

    def __len__(self) -> int:
        return self._count


def build_vector_store(
    *,
    backend: str = "qdrant",
    collection: str = "policy_chunks",
    dim: int = 512,
    url: str = "",
    path: str = "",
) -> InMemoryVectorStore | QdrantVectorStore:
    """Factory creating QdrantVectorStore (default) or InMemoryVectorStore fallback."""
    if backend.strip().lower() == "memory":
        return InMemoryVectorStore()
    try:
        store = QdrantVectorStore(collection=collection, dim=dim, url=url, path=path)
        store._ensure_client()
        return store
    except Exception as exc:
        logger.warning(f"Qdrant initialization failed ({exc}), falling back to InMemoryVectorStore")
        return InMemoryVectorStore()


class VectorRetriever:
    """Dense retriever combining vector store (Qdrant or InMemory) and embedding provider."""

    def __init__(self, store: InMemoryVectorStore | QdrantVectorStore, embedder: EmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        query_vectors = await self.embedder.embed([query])
        if not query_vectors:
            return []
        return self.store.search(query_vectors[0], top_k=top_k, metadata_filter=metadata_filter)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Async alias for consistent retriever interface."""
        return await self.search(query, top_k=top_k, metadata_filter=metadata_filter)
