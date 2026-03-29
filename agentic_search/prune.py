"""Self-editing context — the agent selectively prunes its accumulated context.

This is the key innovation from Context-1: when the agent's context window
approaches a soft limit, it reviews accumulated chunks and discards irrelevant
ones, freeing capacity for further exploration and combating context rot.

The pruner:
  1. Checks if accumulated context exceeds the soft limit
  2. Asks the LLM which entries are no longer relevant to the evolving query
  3. Removes stale entries, keeping the most relevant material
"""

from __future__ import annotations

import anthropic

from .config import ContextEntry, LLMConfig, SearchConfig, extract_json

_PRUNE_SYSTEM = """\
You are a context editor for a search agent.  The agent has accumulated
document chunks during a multi-step search.  The context window is getting
full and we need to free space for further exploration.

Given the current query and the accumulated context entries, decide which
entries to KEEP and which to DISCARD.

Rules:
- Keep entries that are directly relevant to answering the query.
- Discard entries that are tangential, redundant, or superseded by better evidence.
- When in doubt, keep the entry.
- You MUST keep at least 2 entries.

Return ONLY a JSON object:
{
  "keep": ["entry_id_1", "entry_id_2", ...],
  "discard": ["entry_id_3", ...],
  "reasoning": "brief explanation"
}
"""


class ContextPruner:
    """Agent-driven context editing to combat context rot."""

    def __init__(self, llm_config: LLMConfig, search_config: SearchConfig) -> None:
        self._llm_config = llm_config
        self._config = search_config
        self._client = anthropic.Anthropic(api_key=llm_config.api_key)

    def should_prune(self, context: list[ContextEntry]) -> bool:
        """Check if the accumulated context exceeds the soft limit."""
        total_tokens = sum(e.token_estimate for e in context)
        return total_tokens > self._config.context_soft_limit

    def prune(
        self, context: list[ContextEntry], query: str
    ) -> list[ContextEntry]:
        """Ask the LLM which context entries to keep vs discard.

        Returns the pruned list of entries (only those kept).
        """
        if len(context) <= 2:
            return context

        entries_desc = "\n\n".join(
            f"[{e.entry_id}] (doc={e.doc_id}, step={e.step_added}, "
            f"relevance={e.relevance_score:.2f})\n{e.text[:400]}"
            for e in context
        )
        prompt = f"Query: {query}\n\nContext entries:\n{entries_desc}"

        try:
            response = self._client.messages.create(
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                system=_PRUNE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            parsed = extract_json(content, kind="object")
            if not parsed:
                return context
            keep_ids = set(parsed.get("keep", []))

            if not keep_ids:
                return context

            pruned = [e for e in context if e.entry_id in keep_ids]
            # Safety: always keep at least 2
            if len(pruned) < 2:
                pruned = sorted(context, key=lambda e: e.relevance_score, reverse=True)[:2]
            return pruned

        except Exception:
            # On failure, do a simple heuristic prune: keep top half by relevance
            sorted_ctx = sorted(context, key=lambda e: e.relevance_score, reverse=True)
            return sorted_ctx[: max(2, len(sorted_ctx) // 2)]

    def context_token_count(self, context: list[ContextEntry]) -> int:
        return sum(e.token_estimate for e in context)
