"""Reranking primitive — scores fused candidates and selects top results within a token budget.

After hybrid search returns RRF-fused candidates, the reranker:
  1. Asks the LLM to score each candidate's relevance to the query (0-10)
  2. Sorts by relevance score
  3. Greedily packs documents into the per-call token budget
"""

from __future__ import annotations

import anthropic

from .config import LLMConfig, ScoredDocument, SearchConfig, extract_json

_RERANK_PROMPT = """\
You are a relevance judge. Given a search query and a list of document chunks,
score each chunk's relevance to the query on a scale of 0-10.

Each chunk is shown with an "id:" field. Return ONLY a JSON array of objects
where each object has "id" (the EXACT id value shown for that chunk) and "score"
(integer 0-10). Example: [{{"id": "doc_1#chunk_0", "score": 8}}, ...]

Query: {query}

Chunks:
{chunks}
"""


class Reranker:
    """Scores fused candidates via LLM and selects top results within a token budget."""

    def __init__(self, config: SearchConfig, llm_config: LLMConfig) -> None:
        self._config = config
        self._llm_config = llm_config
        self._client = anthropic.Anthropic(api_key=llm_config.api_key)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredDocument],
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> list[ScoredDocument]:
        """Score candidates for relevance, then select top-k within token budget."""
        if not candidates:
            return []

        k = top_k or self._config.top_k_rerank
        budget = token_budget or self._config.per_call_token_budget

        scored = self._score_relevance(query, candidates)
        # Sort by LLM relevance score descending
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return self._select_within_budget(scored, budget, k)

    def _score_relevance(
        self, query: str, candidates: list[ScoredDocument]
    ) -> list[tuple[ScoredDocument, float]]:
        """Single LLM call to score all candidates."""
        chunks_text = "\n\n".join(
            f"id: {sd.document.doc_id}\ntext: {sd.document.text[:800]}"
            for sd in candidates
        )
        prompt = _RERANK_PROMPT.format(query=query, chunks=chunks_text)

        try:
            response = self._client.messages.create(
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            scores_list = extract_json(content, kind="array")
            if not scores_list:
                return [(sd, sd.score) for sd in candidates]
            score_map = {item["id"]: float(item["score"]) for item in scores_list}
        except Exception:
            # On any failure, fall back to RRF scores
            return [(sd, sd.score) for sd in candidates]

        return [
            (sd, score_map.get(sd.document.doc_id, sd.score))
            for sd in candidates
        ]

    def _select_within_budget(
        self,
        scored: list[tuple[ScoredDocument, float]],
        budget: int,
        top_k: int,
    ) -> list[ScoredDocument]:
        """Greedily pack highest-scored docs into the token budget."""
        selected: list[ScoredDocument] = []
        tokens_used = 0
        for sd, relevance in scored:
            est = sd.document.token_estimate
            if tokens_used + est > budget and selected:
                break
            selected.append(
                ScoredDocument(
                    document=sd.document,
                    score=relevance,
                    source="reranked",
                )
            )
            tokens_used += est
            if len(selected) >= top_k:
                break
        return selected
