"""Lightweight NetworkX clause topology dependency graph for 1-Hop retrieval expansion."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from policy_rag.retrieval.filters import matches_metadata_filter
from policy_rag.retrieval.tokenizer import tokenize
from policy_rag.schemas.retrieval import MetadataFilter, RetrievedChunk

logger = logging.getLogger(__name__)

EDGE_INHERITS_GENERAL = "INHERITS_GENERAL"
EDGE_REQUIRES_DISPUTE_PROOF = "REQUIRES_DISPUTE_PROOF"
EDGE_LINKS_COMPLEMENT = "LINKS_COMPLEMENT"

_RETURN_GENERAL_CUES = (
    "退",
    "换",
    "无理由",
    "七天",
    "7天",
    "几天",
    "收到",
    "签收",
    "能吗",
    "可以吗",
    "完好",
    "赠品",
    "不想要",
    "运费",
)

_DISPUTE_PROOF_CUES = (
    "假货",
    "假一赔",
    "退一赔",
    "赔偿",
    "赔我",
    "检测",
    "鉴定",
    "证明",
    "举证",
    "怀疑",
    "责任",
    "仲裁",
    "误差",
    "尺寸",
    "故障",
    "争议",
    "纠纷",
    "坏了",
    "寄回",
    "粗糙",
)

_FRESH_INSTANT_PAY_CUES = (
    "赔",
    "退",
    "死",
    "坏",
    "烂",
    "腐",
    "变质",
    "优鲜赔",
    "寄回",
)

_REMOTE_FREIGHT_CUES = (
    "运费",
    "怎么算",
    "收多少",
    "快递",
    "偏远",
    "新疆",
    "西藏",
    "包邮",
)

_PICKUP_SERVICE_CUES = (
    "取件",
    "上门",
    "退货运费",
    "寄回",
)

_EXCHANGE_180_CUES = (
    "只换不修",
    "180",
    "换新",
    "换机",
    "不想修",
)

_CATEGORY_DOCS = [
    "jd_cat_01_mobile_phones",
    "jd_cat_02_computer_office",
    "jd_cat_03_appliances_large",
    "jd_cat_04_appliances_small",
    "jd_cat_05_fresh_fruits",
    "jd_cat_06_fresh_seafood",
    "jd_cat_07_fashion_apparel",
    "jd_cat_08_beauty_cosmetics",
    "jd_cat_09_jewelry_watches",
    "jd_cat_10_maternal_infant",
    "jd_cat_11_custom_products",
    "jd_cat_12_home_furniture",
    "jd_cat_13_virtual_cards",
    "jd_cat_14_cross_border",
]


def logical_doc_id_of(chunk: RetrievedChunk) -> str:
    """Extracts the canonical logical document ID from a chunk."""
    meta_id = (chunk.metadata or {}).get("logical_doc_id")
    if meta_id:
        return str(meta_id)
    return chunk.document_id.rsplit("_", 1)[0]


class ClauseDependencyGraph:
    """Directed clause dependency graph built on NetworkX for deterministic 1-Hop expansion."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
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
        cues: tuple[str, ...],
        priority: int = 1,
    ) -> None:
        if src in self.doc_chunks and tgt in self.doc_chunks:
            self.graph.add_edge(
                src,
                tgt,
                edge_type=edge_type,
                cues=cues,
                priority=priority,
            )

    def _build_edges(self) -> None:
        # 1. INHERITS_GENERAL: Sub-category & 180-day exchange rules -> 7-day general return policy
        general_doc = "jd_rule_01_seven_days_general"
        for cat_doc in _CATEGORY_DOCS:
            self._add_typed_edge(
                cat_doc,
                general_doc,
                EDGE_INHERITS_GENERAL,
                _RETURN_GENERAL_CUES,
                priority=1,
            )
        self._add_typed_edge(
            "jd_srv_02_180_days_exchange",
            general_doc,
            EDGE_INHERITS_GENERAL,
            _RETURN_GENERAL_CUES,
            priority=1,
        )

        # 2. REQUIRES_DISPUTE_PROOF: Substantive dispute & fake rules -> general dispute arbitration & burden of proof
        dispute_doc = "jd_rule_02_dispute_resolution"
        for src in (
            "jd_cat_01_mobile_phones",
            "jd_cat_02_computer_office",
            "jd_cat_08_beauty_cosmetics",
            "jd_cat_11_custom_products",
            "jd_rule_03_fake_penalty",
            "jd_rule_06_direct_refund",
        ):
            self._add_typed_edge(
                src,
                dispute_doc,
                EDGE_REQUIRES_DISPUTE_PROOF,
                _DISPUTE_PROOF_CUES,
                priority=1,
            )

        # 3. LINKS_COMPLEMENT: Complementary rules across fresh food, shipping, and appliances
        for fresh_doc in ("jd_cat_05_fresh_fruits", "jd_cat_06_fresh_seafood"):
            self._add_typed_edge(
                fresh_doc,
                "jd_srv_01_fresh_instant_pay",
                EDGE_LINKS_COMPLEMENT,
                _FRESH_INSTANT_PAY_CUES,
                priority=1,
            )

        self._add_typed_edge(
            "jd_ship_01_freight_standard",
            "jd_ship_02_remote_freight",
            EDGE_LINKS_COMPLEMENT,
            _REMOTE_FREIGHT_CUES,
            priority=1,
        )
        self._add_typed_edge(
            "jd_ship_01_freight_standard",
            "jd_ship_05_pickup_service",
            EDGE_LINKS_COMPLEMENT,
            _PICKUP_SERVICE_CUES,
            priority=2,
        )
        self._add_typed_edge(
            "jd_cat_03_appliances_large",
            "jd_srv_02_180_days_exchange",
            EDGE_LINKS_COMPLEMENT,
            _EXCHANGE_180_CUES,
            priority=2,
        )

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
        for idx, chunk in enumerate(candidates):
            text = f"{chunk.section or ''} {chunk.content}"
            c_tokens = set(tokenize(text))
            overlap = len(q_tokens & c_tokens)
            # Prefer earlier foundational chunk on ties
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
        """Expands RRF candidates with 1-Hop topologically linked documents gated by query intent."""
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
                cues: tuple[str, ...] = edge_data.get("cues", ())
                if cues and not any(cue in query for cue in cues):
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

        # Sort by edge priority (1 first), preserve discovery order on ties
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
