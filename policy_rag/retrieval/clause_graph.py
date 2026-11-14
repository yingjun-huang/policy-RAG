"""Configuration-driven NetworkX clause dependency graph for 1-Hop retrieval expansion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from policy_rag.retrieval.filters import matches_metadata_filter
from policy_rag.retrieval.tokenizer import tokenize
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk

logger = logging.getLogger(__name__)

EDGE_INHERITS_GENERAL = "INHERITS_GENERAL"
EDGE_REQUIRES_DISPUTE_PROOF = "REQUIRES_DISPUTE_PROOF"
EDGE_LINKS_COMPLEMENT = "LINKS_COMPLEMENT"
ALLOWED_EDGE_TYPES = frozenset(
    {
        EDGE_INHERITS_GENERAL,
        EDGE_REQUIRES_DISPUTE_PROOF,
        EDGE_LINKS_COMPLEMENT,
    }
)

_DEFAULT_GRAPH_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "policy_graph.yaml"


def logical_doc_id_of(chunk: RetrievedChunk) -> str:
    """Extracts the canonical logical document ID from a chunk."""
    meta_id = (chunk.metadata or {}).get("logical_doc_id")
    if meta_id:
        return str(meta_id)
    return chunk.document_id.rsplit("_", 1)[0]


def load_and_validate_graph_config(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Loads and validates policy rule-graph edge specs from YAML."""
    path = Path(config_path) if config_path else _DEFAULT_GRAPH_CONFIG
    if not path.is_absolute() and not path.exists():
        candidate = _DEFAULT_GRAPH_CONFIG.parent.parent / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Policy graph config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("edges"), list):
        raise ValueError(f"Invalid policy graph config format in {path}: expected mapping with 'edges' list")

    validated: list[dict[str, Any]] = []
    for idx, item in enumerate(raw["edges"]):
        if not isinstance(item, dict):
            raise ValueError(f"Edge spec #{idx} must be a mapping")
        has_src = bool(str(item.get("src") or "").strip())
        has_group = bool(str(item.get("src_group") or "").strip())
        if has_src == has_group:
            raise ValueError(f"Edge spec #{idx} must specify exactly one of 'src' or 'src_group'")

        tgt = str(item.get("tgt") or "").strip()
        if not tgt:
            raise ValueError(f"Edge spec #{idx} missing 'tgt'")

        edge_type = str(item.get("type") or "").strip()
        if edge_type not in ALLOWED_EDGE_TYPES:
            raise ValueError(
                f"Edge spec #{idx} has unsupported type '{edge_type}'; expected one of {sorted(ALLOWED_EDGE_TYPES)}"
            )

        priority_raw = item.get("priority", 1)
        if not isinstance(priority_raw, int) or isinstance(priority_raw, bool) or priority_raw < 1:
            raise ValueError(f"Edge spec #{idx} priority must be an integer >= 1, got {priority_raw!r}")

        if has_src and str(item["src"]).strip() == tgt:
            raise ValueError(f"Edge spec #{idx} cannot be a self-loop on '{tgt}'")

        validated.append(
            {
                "src": str(item["src"]).strip() if has_src else None,
                "src_group": str(item["src_group"]).strip() if has_group else None,
                "tgt": tgt,
                "type": edge_type,
                "priority": priority_raw,
                "note": str(item.get("note") or ""),
            }
        )

    return validated


class ClauseDependencyGraph:
    """Directed clause dependency graph built on NetworkX from YAML domain priors."""

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        *,
        config_path: str | Path | None = None,
        edge_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.doc_chunks: dict[str, list[RetrievedChunk]] = {}

        for chunk in chunks:
            doc_id = logical_doc_id_of(chunk)
            self.doc_chunks.setdefault(doc_id, []).append(chunk)

        for doc_id, group in self.doc_chunks.items():
            group.sort(key=lambda c: c.chunk_index)
            first = group[0]
            self.graph.add_node(
                doc_id,
                title=first.title,
                doc_group=str((first.metadata or {}).get("doc_group") or ""),
            )

        self.edge_specs = (
            edge_specs if edge_specs is not None else load_and_validate_graph_config(config_path)
        )
        self._build_edges()
        logger.info(
            "ClauseDependencyGraph initialized: %d nodes, %d edges",
            self.node_count,
            self.edge_count,
        )

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def _add_typed_edge(
        self,
        src: str,
        tgt: str,
        edge_type: str,
        priority: int = 1,
    ) -> None:
        if src == tgt:
            return
        if src in self.doc_chunks and tgt in self.doc_chunks:
            self.graph.add_edge(
                src,
                tgt,
                edge_type=edge_type,
                priority=priority,
            )

    def _build_edges(self) -> None:
        group_to_docs: dict[str, list[str]] = {}
        for doc_id, data in self.graph.nodes(data=True):
            grp = str(data.get("doc_group") or "").strip()
            if grp:
                group_to_docs.setdefault(grp, []).append(doc_id)

        for spec in self.edge_specs:
            tgt = spec["tgt"]
            edge_type = spec["type"]
            priority = int(spec.get("priority", 1))
            if spec.get("src"):
                self._add_typed_edge(spec["src"], tgt, edge_type, priority=priority)
            elif spec.get("src_group"):
                for src_doc in sorted(group_to_docs.get(spec["src_group"], [])):
                    self._add_typed_edge(src_doc, tgt, edge_type, priority=priority)

    def _select_best_chunk(
        self,
        doc_id: str,
        query: str,
        metadata_filter: MetadataFilter | None,
    ) -> RetrievedChunk | None:
        candidates = [
            c
            for c in self.doc_chunks.get(doc_id, [])
            if matches_metadata_filter(c, metadata_filter)
        ]
        if not candidates:
            return None

        q_tokens = set(tokenize(query))
        if not q_tokens:
            return candidates[0]

        best_chunk = candidates[0]
        best_score = -1
        for chunk in candidates:
            text = f"{chunk.section or ''} {chunk.content}"
            c_tokens = set(tokenize(text))
            overlap = len(q_tokens & c_tokens)
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        return best_chunk

    def expand_candidates(
        self,
        query: str,
        primary_chunks: list[RetrievedChunk],
        metadata_filter: MetadataFilter | None = None,
        *,
        seed_top_docs: int = 4,
        max_expand: int = 3,
    ) -> list[RetrievedChunk]:
        """Expands RRF candidates with 1-Hop linked documents; Cross-Encoder threshold gates final inclusion."""
        if not primary_chunks or not self.graph.number_of_edges():
            return list(primary_chunks)

        existing_docs: set[str] = set()
        seed_docs: list[str] = []
        for chunk in primary_chunks:
            doc_id = logical_doc_id_of(chunk)
            existing_docs.add(doc_id)
            if doc_id not in seed_docs and len(seed_docs) < seed_top_docs:
                seed_docs.append(doc_id)

        expanded_candidates: list[tuple[int, str, str, str, RetrievedChunk]] = []
        added_targets: set[str] = set()

        for src_doc in seed_docs:
            if src_doc not in self.graph:
                continue
            for _, tgt_doc, edge_data in self.graph.out_edges(src_doc, data=True):
                if tgt_doc in existing_docs or tgt_doc in added_targets:
                    continue

                rep_chunk = self._select_best_chunk(tgt_doc, query, metadata_filter)
                if rep_chunk is None:
                    continue

                added_targets.add(tgt_doc)
                expanded_candidates.append(
                    (
                        int(edge_data.get("priority", 1)),
                        src_doc,
                        tgt_doc,
                        str(edge_data.get("edge_type", EDGE_LINKS_COMPLEMENT)),
                        rep_chunk,
                    )
                )

        expanded_candidates.sort(key=lambda item: item[0])

        out = list(primary_chunks)
        for _, src_doc, tgt_doc, edge_type, rep_chunk in expanded_candidates[:max_expand]:
            copy_chunk = rep_chunk.model_copy(deep=True)
            copy_chunk.source = "clause_graph"
            copy_chunk.metadata = dict(copy_chunk.metadata or {})
            copy_chunk.metadata["graph_expanded"] = True
            copy_chunk.metadata["graph_edge"] = edge_type
            copy_chunk.metadata["graph_source_doc"] = src_doc
            copy_chunk.metadata["graph_target_doc"] = tgt_doc
            out.append(copy_chunk)

        return out
