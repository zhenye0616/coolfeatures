"""Hybrid search primitive — parallel dense + sparse retrieval fused via Reciprocal Rank Fusion.

Implements the `search_corpus` tool from the Context-1 agent harness:
  1. Issue dense (embedding) and sparse (BM25) queries in parallel
  2. Fuse results with RRF
  3. Deduplicate against previously-seen chunk IDs
  4. Return top-k fused candidates
"""

from __future__ import annotations

from config import ScoredDocument, SearchConfig
from storage.storage import DocumentStore


class HybridSearcher:
    """Executes parallel dense + sparse retrieval and fuses with RRF."""

    def __init__(self, store: DocumentStore, config: SearchConfig) -> None:
        self._store = store
        self._config = config

    def search(
        self,
        query: str,
        seen_ids: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[ScoredDocument]:
        """Run hybrid search: dense + sparse → RRF fusion → dedup → top-k."""
        exclude = seen_ids or set()
        k = top_k or self._config.top_k_retrieval

        # Both retrievals run against the same store (would be parallel in async)
        dense_results = self._store.query_dense(query, top_k=k, exclude_ids=exclude)
        sparse_results = self._store.query_sparse(query, top_k=k, exclude_ids=exclude)

        # Fuse with reciprocal rank fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:k]

    def _reciprocal_rank_fusion(
        self,
        dense: list[ScoredDocument],
        sparse: list[ScoredDocument],
    ) -> list[ScoredDocument]:
        """Combine dense and sparse result lists using RRF.

        score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))

        Documents appearing in only one list get a single-term RRF score.
        """
        k = self._config.rrf_k
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, ScoredDocument] = {}

        # Score from dense results
        for rank, sd in enumerate(dense, start=1):
            doc_id = sd.document.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = sd

        # Score from sparse results
        for rank, sd in enumerate(sparse, start=1):
            doc_id = sd.document.doc_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = sd

        # Sort by fused score descending
        sorted_ids = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)
        return [
            ScoredDocument(
                document=doc_map[did].document,
                score=rrf_scores[did],
                source="fused",
            )
            for did in sorted_ids
        ]
