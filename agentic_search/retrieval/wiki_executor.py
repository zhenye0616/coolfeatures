"""Wikipedia Executor — tool-calling loop with Wikipedia search and read tools.

Matches Context-1 harness design:
  - Agent controls its own pruning via prune_context tool
  - Token usage visibility after every turn
  - Soft threshold: inject message suggesting pruning
  - Hard cutoff: reject all tools except prune_context
"""

from __future__ import annotations

import uuid
from typing import Any

from client import create_client
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
from retrieval.rerank import Reranker
from retrieval.wikipedia import chunk_page, fetch_wikipedia, search_wikipedia

# ---------------------------------------------------------------------------
# Tool definitions
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

PRUNE_CONTEXT_TOOL: dict[str, Any] = {
    "name": "prune_context",
    "description": (
        "Remove irrelevant chunks from your context to free up space. "
        "Pass the chunk IDs (shown in [brackets] in tool results) that "
        "you want to discard. Keep chunks that are directly relevant to "
        "the query; discard tangential, redundant, or low-quality ones."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of chunk doc_ids to remove from context.",
            },
        },
        "required": ["chunk_ids"],
    },
}

FINISH_STEP_TOOL: dict[str, Any] = {
    "name": "finish_step",
    "description": (
        "Call this when you have retrieved sufficient relevant documents, "
        "or when further searching is unlikely to find new relevant pages. "
        "Summarize which documents were retrieved and why they are relevant."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Summary of which documents were retrieved and their relevance to the sub-query.",
            },
        },
        "required": ["summary"],
    },
}

WIKI_EXECUTOR_TOOLS = [
    SEARCH_WIKIPEDIA_TOOL,
    READ_WIKIPEDIA_TOOL,
    PRUNE_CONTEXT_TOOL,
    FINISH_STEP_TOOL,
]

# Restricted tool set when past hard cutoff — only pruning and finishing allowed
WIKI_EXECUTOR_TOOLS_RESTRICTED = [PRUNE_CONTEXT_TOOL, FINISH_STEP_TOOL]

_WIKI_EXECUTOR_SYSTEM = """\
You are a document retrieval agent in a multi-agent system. Your specific role \
is to identify and retrieve the most relevant Wikipedia pages for a downstream \
agent that will answer questions. You do NOT answer questions yourself — you \
only find and retrieve relevant documents.

Tools:
- search_wikipedia(query): Search Wikipedia for matching pages. Returns titles \
  and snippets. Use targeted, specific queries.
- read_wikipedia(title): Fetch the full content of a Wikipedia page by its \
  exact title. You MUST read every page that looks relevant — do not rely on \
  search snippets alone.
- prune_context(chunk_ids): Remove irrelevant chunks from your context to free \
  up space. Pass chunk doc_ids you want to discard. Use this when your token \
  usage is high and you need room for more searches.
- finish_step(summary): Call when you have retrieved sufficient documents, or \
  when further searching is unlikely to help.

Process:
1. Decompose the sub-query into key concepts and information needs.
2. Plan multiple distinct, non-overlapping search strategies approaching the \
   question from different angles.
3. Execute searches, and READ every promising page — the downstream agent \
   needs the full document content, not just snippets.
4. After each round, evaluate:
   - What do I know? List the key facts your retrieved documents cover.
   - What should I search for next? What approaches haven't you tried?
   - What should I prune? If token usage is high, identify low-value chunks \
     to discard and free space for more useful searches.
   - Do I have enough? Are there critical gaps in the evidence?
5. Avoid getting stuck on a single search strategy — if one approach isn't \
   yielding results, try different approaches.
6. Follow explicit textual evidence rather than speculation.
7. Monitor your token usage shown after each tool result. When approaching \
   the limit, prune aggressively or finish.
"""


def _token_count(entries: list[ContextEntry]) -> int:
    """Sum token estimates across all context entries."""
    return sum(e.token_estimate for e in entries)


class WikiExecutor:
    """Solves a single PlanStep via Wikipedia search + read tools.

    Implements Context-1-style agent-controlled pruning:
      - Token usage visibility after every turn
      - Soft threshold: inject message suggesting pruning
      - Hard cutoff: reject all tools except prune_context
    """

    def __init__(
        self,
        reranker: Reranker,
        llm_config: LLMConfig,
        search_config: SearchConfig,
    ) -> None:
        self._reranker = reranker
        self._llm_config = llm_config
        self._search_config = search_config
        self._client = create_client(llm_config)

    def execute(
        self,
        step: PlanStep,
        seen_ids: set[str],
        existing_context: list[ContextEntry],
        trace=None,
    ) -> tuple[StepOutcome, list[ContextEntry], set[str]]:
        """Run the tool-calling loop for *step*.

        Returns (outcome, updated_full_context, updated_seen_ids).
        Unlike the local Executor, this returns the FULL context (existing +
        new - pruned) because the agent controls pruning via the prune_context tool.
        """
        context = list(existing_context)
        new_entries: list[ContextEntry] = []
        step_num = step.step_id
        budget = self._search_config.context_token_budget
        soft_limit = budget // 2
        hard_limit = int(budget * 0.85)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": (
                f"Sub-query: {step.query}\n"
                f"Rationale: {step.rationale}\n\n"
                f"[Token usage: {_token_count(context)}/{budget}]"
            )}
        ]

        max_iterations = 10
        for iteration in range(max_iterations):
            current_tokens = _token_count(context)
            past_hard = current_tokens > hard_limit
            tools = WIKI_EXECUTOR_TOOLS_RESTRICTED if past_hard else WIKI_EXECUTOR_TOOLS

            response = self._client.messages.create(
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                system=_WIKI_EXECUTOR_SYSTEM,
                messages=messages,
                tools=tools,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Log model reasoning
            if trace and text_blocks:
                for tb in text_blocks:
                    trace.log("model_text", step_id=step.step_id, iteration=iteration, text=tb.text)

            if not tool_use_blocks:
                text = text_blocks[0].text if text_blocks else ""
                if trace:
                    trace.log("step_finish_implicit", step_id=step.step_id, iteration=iteration, text=text[:500])
                return (
                    StepOutcome(
                        step_id=step.step_id,
                        status=StepStatus.SUCCESS,
                        retrieved_docs=[],
                        candidate_answer="",
                        evidence=text,
                    ),
                    context,
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
                if trace:
                    trace.log("tool_call", step_id=step.step_id, iteration=iteration,
                              tool=block.name, input=block.input)

                if block.name == "search_wikipedia":
                    result_text = self._exec_search(block, step)
                    if trace:
                        trace.log("tool_result", step_id=step.step_id, tool="search_wikipedia",
                                  result=result_text[:500])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                elif block.name == "read_wikipedia":
                    result_text, read_entries = self._exec_read(
                        block, step, seen_ids, step_num
                    )
                    new_entries.extend(read_entries)
                    context.extend(read_entries)
                    if trace:
                        trace.log("tool_result", step_id=step.step_id, tool="read_wikipedia",
                                  title=block.input.get("title", ""),
                                  chunks_added=len(read_entries),
                                  result=result_text[:500])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                elif block.name == "prune_context":
                    result_text, context = self._exec_prune(
                        block, context, trace, step.step_id, iteration
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                elif block.name == "finish_step":
                    summary = block.input.get("summary", "")
                    if trace:
                        trace.log("tool_call_finish", step_id=step.step_id, iteration=iteration,
                                  summary=summary[:500])
                    all_retrieved = [
                        ScoredDocument(
                            document=Document(doc_id=e.doc_id, text=e.text),
                            score=e.relevance_score,
                            source="context",
                        )
                        for e in context
                    ]
                    return (
                        StepOutcome(
                            step_id=step.step_id,
                            status=StepStatus.SUCCESS,
                            retrieved_docs=all_retrieved,
                            candidate_answer="",
                            evidence=summary,
                        ),
                        context,
                        seen_ids,
                    )
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Unknown tool: {block.name}",
                        "is_error": True,
                    })

            # Append token usage + threshold warnings to the last tool result
            current_tokens = _token_count(context)
            usage_str = f"\n\n[Token usage: {current_tokens}/{budget}]"

            if current_tokens > hard_limit:
                usage_str += (
                    f"\n\n🛑 Context budget nearly exhausted ({current_tokens}/{budget}). "
                    f"Only prune_context and finish_step are available. "
                    f"Prune chunks to free space or finish."
                )
            elif current_tokens > soft_limit:
                usage_str += (
                    f"\n\n⚠ Context is {current_tokens * 100 // budget}% full. "
                    f"Consider pruning low-value chunks with prune_context to free "
                    f"space for more searches, or call finish_step if you have "
                    f"enough documents."
                )

            if tool_results:
                tool_results[-1]["content"] += usage_str

            messages.append({"role": "user", "content": tool_results})

        # Exhausted iterations
        return (
            StepOutcome(
                step_id=step.step_id,
                status=StepStatus.FAILURE,
                candidate_answer="",
                evidence="Max iterations reached without completion.",
            ),
            context,
            seen_ids,
        )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _exec_search(self, block, step: PlanStep) -> str:
        query = block.input.get("query", step.query)
        try:
            results = search_wikipedia(query, limit=10)
            if results:
                return "\n".join(
                    f"- {r['title']}: {r['snippet'][:200]}" for r in results
                )
            return "No Wikipedia results found."
        except Exception as e:
            return f"Search failed: {e}"

    def _exec_read(
        self, block, step: PlanStep, seen_ids: set[str], step_num: int
    ) -> tuple[str, list[ContextEntry]]:
        title = block.input.get("title", "")
        entries: list[ContextEntry] = []
        try:
            page_text = fetch_wikipedia(title)
            if page_text is None:
                return f"Page '{title}' not found on Wikipedia.", []

            chunks = chunk_page(title, page_text)
            if not chunks:
                return f"Page '{title}' has no extractable content.", []

            scored_chunks = [
                ScoredDocument(document=doc, score=1.0, source="wikipedia")
                for doc in chunks
                if doc.doc_id not in seen_ids
            ]

            reranked = self._reranker.rerank(step.query, scored_chunks) if scored_chunks else []

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

            result_text = "\n\n---\n\n".join(lines) if lines else "No new relevant chunks found (all previously seen)."
            return result_text, entries
        except Exception as e:
            return f"Failed to read page: {e}", []

    def _exec_prune(
        self,
        block,
        context: list[ContextEntry],
        trace,
        step_id: int,
        iteration: int,
    ) -> tuple[str, list[ContextEntry]]:
        chunk_ids = set(block.input.get("chunk_ids", []))
        before = len(context)
        before_tokens = _token_count(context)
        pruned = [e for e in context if e.doc_id not in chunk_ids]
        removed = before - len(pruned)
        after_tokens = _token_count(pruned)

        if trace:
            trace.log("agent_prune", step_id=step_id, iteration=iteration,
                      requested=list(chunk_ids)[:20],
                      removed=removed, before=before, after=len(pruned),
                      tokens_before=before_tokens, tokens_after=after_tokens)

        result = (
            f"Pruned {removed} chunk(s). "
            f"Context: {before} → {len(pruned)} entries, "
            f"{before_tokens} → {after_tokens} tokens."
        )
        return result, pruned

