"""Tier 3 — Storage layer: ChromaDB (dense embeddings) + BM25 (sparse keyword index).

Provides a unified DocumentStore that maintains both indexes over the same corpus
and supports querying each independently.
"""

from __future__ import annotations

import re
from typing import Sequence

import chromadb
from rank_bm25 import BM25Okapi

from .config import Document, ScoredDocument, SearchConfig


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class DocumentStore:
    """Wraps ChromaDB (dense) and an in-memory BM25 index (sparse) over the same corpus."""

    def __init__(self, config: SearchConfig, persist_dir: str | None = None) -> None:
        self._config = config
        # ChromaDB client — ephemeral unless persist_dir given
        if persist_dir:
            self._chroma = chromadb.PersistentClient(path=persist_dir)
        else:
            self._chroma = chromadb.EphemeralClient()
        self._collection = self._chroma.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # In-memory BM25 state
        self._docs: list[Document] = []
        self._doc_index: dict[str, Document] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_documents(self, docs: Sequence[Document]) -> None:
        """Add documents to both dense and sparse indexes."""
        if not docs:
            return

        ids = [d.doc_id for d in docs]
        texts = [d.text for d in docs]
        metadatas = [d.metadata for d in docs]

        # Upsert into ChromaDB (handles embedding automatically)
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        # Update in-memory stores
        for doc in docs:
            if doc.doc_id not in self._doc_index:
                self._docs.append(doc)
            self._doc_index[doc.doc_id] = doc

        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from all stored documents."""
        self._bm25_corpus = [_tokenize(d.text) for d in self._docs]
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_dense(
        self, query: str, top_k: int | None = None, exclude_ids: set[str] | None = None
    ) -> list[ScoredDocument]:
        """Semantic similarity search via ChromaDB embeddings."""
        k = top_k or self._config.top_k_retrieval
        # ChromaDB doesn't support exclude filters natively on IDs in all versions,
        # so we over-fetch and filter client-side.
        fetch_k = k + (len(exclude_ids) if exclude_ids else 0)
        results = self._collection.query(query_texts=[query], n_results=min(fetch_k, self._collection.count()))
        if not results["ids"] or not results["ids"][0]:
            return []

        scored: list[ScoredDocument] = []
        for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
            if exclude_ids and doc_id in exclude_ids:
                continue
            doc = self._doc_index.get(doc_id)
            if doc is None:
                continue
            # ChromaDB returns distances; convert cosine distance → similarity
            similarity = 1.0 - distance
            scored.append(ScoredDocument(document=doc, score=similarity, source="dense"))
            if len(scored) >= k:
                break
        return scored

    def query_sparse(
        self, query: str, top_k: int | None = None, exclude_ids: set[str] | None = None
    ) -> list[ScoredDocument]:
        """Keyword search via BM25."""
        if self._bm25 is None or not self._docs:
            return []
        k = top_k or self._config.top_k_retrieval
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # Pair with docs, sort descending
        paired = sorted(zip(self._docs, scores), key=lambda x: x[1], reverse=True)

        scored: list[ScoredDocument] = []
        for doc, score in paired:
            if exclude_ids and doc.doc_id in exclude_ids:
                continue
            if score <= 0:
                break
            scored.append(ScoredDocument(document=doc, score=score, source="sparse"))
            if len(scored) >= k:
                break
        return scored

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, doc_id: str) -> Document | None:
        return self._doc_index.get(doc_id)

    @property
    def count(self) -> int:
        return len(self._docs)
