"""Tier 2 — Answer generation: synthesizes a final answer from curated context.

The SearchAgent (Tier 1) retrieves and curates context.  The AnswerGenerator
(Tier 2) takes that context and produces a coherent answer with citations.

This separation matches the Context-1 architecture where Context-1 finds
documents and a frontier reasoning model produces the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from config import AgentResult, ContextEntry, LLMConfig

_GENERATE_SYSTEM = """\
You are a knowledgeable assistant.  You are given a user question and a set of
retrieved document chunks gathered by a search agent.

Your job:
1. Synthesize the information from the chunks into a clear, accurate answer.
2. Cite your sources using [doc_id] notation inline.
3. If the evidence is insufficient, say so honestly.
4. Be concise but thorough.
"""


@dataclass
class GeneratedAnswer:
    answer: str
    context_used: list[ContextEntry]


class AnswerGenerator:
    """Produces the final answer from retrieved context using a frontier LLM."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm_config = llm_config
        self._client = anthropic.Anthropic(api_key=llm_config.api_key)

    def generate(
        self,
        query: str,
        agent_result: AgentResult,
        context_token_budget: int = 16_000,
    ) -> GeneratedAnswer:
        """Generate a final answer from the agent's curated context.

        Entries are sorted by relevance and greedily packed within
        *context_token_budget* to avoid exceeding the model's context window.
        """
        context = agent_result.context_snapshot
        if not context:
            return GeneratedAnswer(
                answer="I was unable to find relevant information to answer this question.",
                context_used=[],
            )

        # Prioritize highest-relevance entries and enforce a token budget
        sorted_context = sorted(
            context, key=lambda e: e.relevance_score, reverse=True
        )
        selected: list[ContextEntry] = []
        tokens_used = 0
        for entry in sorted_context:
            est = entry.token_estimate
            if tokens_used + est > context_token_budget and selected:
                break
            selected.append(entry)
            tokens_used += est

        chunks_text = "\n\n---\n\n".join(
            f"[{e.doc_id}] (relevance={e.relevance_score:.2f})\n{e.text}"
            for e in selected
        )

        prompt = (
            f"Question: {query}\n\n"
            f"Retrieved context:\n{chunks_text}\n\n"
            f"Please answer the question based on the context above."
        )

        response = self._client.messages.create(
            model=self._llm_config.model,
            max_tokens=self._llm_config.max_tokens,
            system=_GENERATE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = response.content[0].text
        return GeneratedAnswer(answer=answer_text, context_used=selected)
