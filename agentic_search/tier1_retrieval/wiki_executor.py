"""Wikipedia Executor — tool-calling loop with Wikipedia search and read tools.

Same observe → reason → act loop as executor.py, but backed by the Wikipedia
API instead of a local DocumentStore. Used for the FRAMES benchmark evaluation.
"""

from __future__ import annotations

import uuid
from typing import Any

import anthropic

from config import (
    ContextEntry,
    Document,
    LLMConfig,
    PlanStep,
    SearchConfig,
    ScoredDocument,
    StepOutcome,
    StepStatus,
)
from rerank import Reranker
from wikipedia import chunk_page, fetch_wikipedia, search_wikipedia

# ---------------------------------------------------------------------------
# Tool definitions for Wikipedia search
# ---------------------------------------------------------------------------

SEARCH_WIKIPEDIA_TOOL: dict[str, Any] = {
    "name": "search_wikipedia",
    "description": (
        "Search Wikipedia for pages matching a query. Returns page titles and "
        "snippets. Use specific, targeted queries. After finding relevant pages, "
        "use read_wikipedia to get the full content."
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

READ_WIKIPEDIA_TOOL: dict[str, Any] = {
    "name": "read_wikipedia",
    "description": (
        "Fetch and read the full content of a Wikipedia page by its exact title. "
        "The content is chunked and reranked for relevance. Use this after "
        "search_wikipedia finds a promising page."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The exact Wikipedia page title to read.",
            },
        },
        "required": ["title"],
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
                "description": "Best candidate answer extracted from evidence.",
            },
        },
        "required": ["summary"],
    },
}

WIKI_EXECUTOR_TOOLS = [SEARCH_WIKIPEDIA_TOOL, READ_WIKIPEDIA_TOOL, FINISH_STEP_TOOL]

_WIKI_EXECUTOR_SYSTEM = """\
You are a search agent executing a single retrieval step. Your job is to
find Wikipedia articles that answer the sub-query below. You have these tools:

- search_wikipedia(query): Search Wikipedia for matching pages. Returns titles
  and snippets. Use targeted, specific queries.
- read_wikipedia(title): Fetch the full content of a Wikipedia page by its
  exact title. Returns the most relevant chunks.
- finish_step(summary, candidate_answer): Call when done gathering evidence.

Strategy:
- Start with a specific search query.
- When you find a promising page title, read it to get the full content.
- Reformulate and search again if initial results are insufficient.
- Do NOT repeat the same query — vary your search terms.
- Limit yourself to at most 6 tool calls per step.
"""


class WikiExecutor:
    """Solves a single PlanStep via Wikipedia search + read tools."""

    def __init__(
        self,
        reranker: Reranker,
        llm_config: LLMConfig,
        search_config: SearchConfig,
    ) -> None:
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

        max_iterations = 8  # slightly higher than local executor — web calls
        for _ in range(max_iterations):
            response = self._client.messages.create(
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                system=_WIKI_EXECUTOR_SYSTEM,
                messages=messages,
                tools=WIKI_EXECUTOR_TOOLS,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if not tool_use_blocks:
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

            # Build assistant message
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
                if block.name == "search_wikipedia":
                    query = block.input.get("query", step.query)
                    try:
                        results = search_wikipedia(query, limit=10)
                        if results:
                            result_text = "\n".join(
                                f"- {r['title']}: {r['snippet'][:200]}"
                                for r in results
                            )
                        else:
                            result_text = "No Wikipedia results found."
                    except Exception as e:
                        result_text = f"Search failed: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                elif block.name == "read_wikipedia":
                    title = block.input.get("title", "")
                    try:
                        page_text = fetch_wikipedia(title)
                        if page_text is None:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Page '{title}' not found on Wikipedia.",
                            })
                            continue

                        chunks = chunk_page(title, page_text)
                        if not chunks:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Page '{title}' has no extractable content.",
                            })
                            continue

                        # Convert chunks to ScoredDocuments for reranking
                        scored_chunks = [
                            ScoredDocument(document=doc, score=1.0, source="wikipedia")
                            for doc in chunks
                            if doc.doc_id not in seen_ids
                        ]

                        # Rerank chunks against the step query
                        if scored_chunks:
                            reranked = self._reranker.rerank(step.query, scored_chunks)
                        else:
                            reranked = []

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
                            new_entries.append(entry)
                            lines.append(
                                f"[{doc.doc_id}] (score={sd.score:.3f})\n{doc.text[:600]}"
                            )

                        result_text = "\n\n---\n\n".join(lines) if lines else "No new relevant chunks found (all previously seen)."
                    except Exception as e:
                        result_text = f"Failed to read page: {e}"

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
                            document=Document(doc_id=e.doc_id, text=e.text),
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

        # Exhausted iterations
        return (
            StepOutcome(
                step_id=step.step_id,
                status=StepStatus.FAILURE,
                candidate_answer="",
                evidence="Max iterations reached without completion.",
            ),
            new_entries,
            seen_ids,
        )
