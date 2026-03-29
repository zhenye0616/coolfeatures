"""Executor — solves a single PlanStep by running a tool-calling loop.

The Executor implements the observe → reason → act loop for one step:
  1. Send the step query + accumulated context to the LLM with tool definitions
  2. If the LLM returns tool_use blocks, execute them (search_corpus)
  3. Append tool results, loop back
  4. When the LLM returns a text response (no tool calls), produce a StepOutcome
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import anthropic

from .config import (
    ContextEntry,
    LLMConfig,
    PlanStep,
    SearchConfig,
    ScoredDocument,
    StepOutcome,
    StepStatus,
)
from .rerank import Reranker
from .search import HybridSearcher

# ---------------------------------------------------------------------------
# Tool definitions exposed to the LLM (mirrors Context-1 harness)
# ---------------------------------------------------------------------------

SEARCH_CORPUS_TOOL: dict[str, Any] = {
    "name": "search_corpus",
    "description": (
        "Search the document corpus using hybrid dense + sparse retrieval "
        "with reciprocal rank fusion. Returns the top reranked chunks. "
        "Use this to find information relevant to the current sub-query. "
        "Each call automatically deduplicates against previously seen chunks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query — be specific and targeted.",
            },
        },
        "required": ["query"],
    },
}

FINISH_STEP_TOOL: dict[str, Any] = {
    "name": "finish_step",
    "description": (
        "Call this when you have gathered enough evidence to answer the "
        "current sub-query, or when further searching is unlikely to help. "
        "Provide a concise summary of what you found and any candidate answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Summary of findings for this step.",
            },
            "candidate_answer": {
                "type": "string",
                "description": "Best candidate answer extracted from retrieved evidence, if any.",
            },
        },
        "required": ["summary"],
    },
}

EXECUTOR_TOOLS = [SEARCH_CORPUS_TOOL, FINISH_STEP_TOOL]

_EXECUTOR_SYSTEM = """\
You are a search agent executing a single retrieval step.  Your job is to
find documents that answer the sub-query below.  You have access to these tools:

- search_corpus(query): Searches the corpus. Returns top chunks. Use targeted,
  specific queries.  You may call this tool multiple times with different queries.
- finish_step(summary, candidate_answer): Call when done gathering evidence.

Strategy:
- Start with the most specific query you can formulate.
- If results are insufficient, reformulate and search again.
- Do NOT repeat the same query — each search automatically excludes previously
  seen chunks.
- Limit yourself to at most 4 search calls per step.
"""


class Executor:
    """Solves a single PlanStep via a tool-calling loop."""

    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        llm_config: LLMConfig,
        search_config: SearchConfig,
    ) -> None:
        self._searcher = searcher
        self._reranker = reranker
        self._llm_config = llm_config
        self._search_config = search_config
        self._client = anthropic.Anthropic(api_key=llm_config.api_key)

    def execute(
        self,
        step: PlanStep,
        seen_ids: set[str],
        existing_context: list[ContextEntry],
    ) -> tuple[StepOutcome, list[ContextEntry], set[str]]:
        """Run the tool-calling loop for *step*.

        Returns (outcome, new_context_entries, updated_seen_ids).
        """
        new_entries: list[ContextEntry] = []
        step_num = step.step_id
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Sub-query: {step.query}\nRationale: {step.rationale}"}
        ]

        max_iterations = 6  # safety cap
        for _ in range(max_iterations):
            response = self._client.messages.create(
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                system=_EXECUTOR_SYSTEM,
                messages=messages,
                tools=EXECUTOR_TOOLS,
            )

            # Check if the model wants to use tools
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_use_blocks:
                # Model responded with text only — treat as implicit finish
                text = text_blocks[0].text if text_blocks else ""
                return (
                    StepOutcome(
                        step_id=step.step_id,
                        status=StepStatus.SUCCESS,
                        retrieved_docs=[],
                        candidate_answer=text,
                        evidence=text,
                    ),
                    new_entries,
                    seen_ids,
                )

            # Build assistant message with all content blocks
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call
            tool_results = []
            for block in tool_use_blocks:
                if block.name == "search_corpus":
                    query = block.input.get("query", step.query)
                    result_text, new_docs = self._execute_search(query, seen_ids, step_num)
                    new_entries.extend(new_docs)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
                elif block.name == "finish_step":
                    summary = block.input.get("summary", "")
                    candidate = block.input.get("candidate_answer", "")
                    all_retrieved = [
                        ScoredDocument(
                            document=e.document if hasattr(e, "document") else type("D", (), {"doc_id": e.doc_id, "text": e.text, "metadata": {}, "token_estimate": e.token_estimate})(),
                            score=e.relevance_score,
                            source="context",
                        )
                        for e in new_entries
                    ]
                    return (
                        StepOutcome(
                            step_id=step.step_id,
                            status=StepStatus.SUCCESS,
                            retrieved_docs=all_retrieved,
                            candidate_answer=candidate,
                            evidence=summary,
                        ),
                        new_entries,
                        seen_ids,
                    )
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

        # Exhausted iterations — return what we have
        return (
            StepOutcome(
                step_id=step.step_id,
                status=StepStatus.SUCCESS,
                candidate_answer="",
                evidence="Max iterations reached.",
            ),
            new_entries,
            seen_ids,
        )

    def _execute_search(
        self, query: str, seen_ids: set[str], step_num: int
    ) -> tuple[str, list[ContextEntry]]:
        """Run search_corpus: hybrid search → rerank → format results."""
        raw = self._searcher.search(query, seen_ids=seen_ids)
        reranked = self._reranker.rerank(query, raw)

        entries: list[ContextEntry] = []
        lines: list[str] = []
        for sd in reranked:
            doc = sd.document
            seen_ids.add(doc.doc_id)
            entry = ContextEntry(
                entry_id=str(uuid.uuid4()),
                doc_id=doc.doc_id,
                text=doc.text,
                relevance_score=sd.score,
                step_added=step_num,
            )
            entries.append(entry)
            lines.append(f"[{doc.doc_id}] (score={sd.score:.3f})\n{doc.text[:600]}")

        if not lines:
            return "No results found.", []
        return "\n\n---\n\n".join(lines), entries
