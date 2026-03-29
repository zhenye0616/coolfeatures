# FRAMES Benchmark Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the FRAMES benchmark (824 multi-hop Wikipedia questions) through our agent harness with Claude Sonnet, producing metrics directly comparable to Chroma Context-1's published numbers.

**Architecture:** Three new modules — `wikipedia.py` (Wikipedia API client), `wiki_executor.py` (executor with Wikipedia tools), and `frames_eval.py` (benchmark runner + metrics). A new `WikiSearchAgent` subclass of `BaseAgent` wires Wikipedia tools into the existing planner/evaluator/pruner pipeline. No changes to existing modules.

**Tech Stack:** `httpx` for Wikipedia API calls, `datasets` (HuggingFace) for loading FRAMES, `anthropic` SDK (existing), all existing agent primitives.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `wikipedia.py` (create) | Wikipedia search API + page fetch + chunking into `Document` objects |
| `wiki_executor.py` (create) | Executor variant with `search_wikipedia` and `read_wikipedia` tools |
| `wiki_agent.py` (create) | `WikiSearchAgent(BaseAgent)` that wires Wikipedia tools with existing planner/evaluator/pruner |
| `frames_eval.py` (create) | CLI script: loads FRAMES dataset, runs agent per question, computes metrics, writes results |
| `pyproject.toml` (modify) | Add `httpx` and `datasets` dependencies |

Reused unchanged: `config.py`, `planner.py`, `evaluator.py`, `prune.py`, `generate.py`, `rerank.py`, `ingest.py` (chunking function only).

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add httpx and datasets to pyproject.toml**

In `pyproject.toml`, replace the dependencies list:

```toml
dependencies = [
    "anthropic>=0.49.0",
    "chromadb>=0.6.3",
    "datasets>=3.0.0",
    "httpx>=0.27.0",
    "rank-bm25>=0.2.2",
    "streamlit>=1.41.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: clean install, no errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add httpx and datasets for FRAMES evaluation"
```

---

### Task 2: Wikipedia API client

**Files:**
- Create: `wikipedia.py`
- Test: manual verification via `uv run python -c "..."`

- [ ] **Step 1: Create wikipedia.py**

```python
"""Wikipedia API client — search + fetch + chunk for the FRAMES evaluation harness.

Uses the public MediaWiki API (no API key required):
  - opensearch endpoint for title search
  - parse endpoint for full page text extraction
"""

from __future__ import annotations

import re

import httpx

from config import Document

_API_URL = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 30.0


def search_wikipedia(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search Wikipedia and return a list of {title, snippet} dicts.

    Uses the MediaWiki opensearch API for title matching, then srsearch
    for snippet-based full-text search as fallback.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("query", {}).get("search", []):
        # Strip HTML tags from snippet
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        results.append({"title": item["title"], "snippet": snippet})
    return results


def fetch_wikipedia(title: str) -> str | None:
    """Fetch the plain-text content of a Wikipedia page by title.

    Uses the MediaWiki parse API with wikitext → plain text extraction.
    Returns None if the page doesn't exist.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "utf8": "1",
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        return page.get("extract", "")
    return None


def chunk_page(
    title: str, text: str, chunk_size: int = 512, chunk_overlap: int = 50
) -> list[Document]:
    """Split a Wikipedia page into overlapping chunks.

    Each chunk gets a doc_id like "wiki:PageTitle#chunk_0" and metadata
    including the Wikipedia URL for retrieval F1 scoring.
    """
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[Document] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_tokens = len(para) // 4
        if current_len + para_tokens > chunk_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(
                Document(
                    doc_id=f"wiki:{title}#chunk_{len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "source": f"wiki:{title}",
                        "wiki_url": url,
                        "wiki_title": title,
                        "chunk_index": len(chunks),
                    },
                )
            )
            # Overlap: keep trailing paragraphs within budget
            overlap_paras: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                p_len = len(p) // 4
                if overlap_len + p_len > chunk_overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_len += p_len
            current = overlap_paras
            current_len = overlap_len

        current.append(para)
        current_len += para_tokens

    if current:
        chunk_text = "\n\n".join(current)
        chunks.append(
            Document(
                doc_id=f"wiki:{title}#chunk_{len(chunks)}",
                text=chunk_text,
                metadata={
                    "source": f"wiki:{title}",
                    "wiki_url": url,
                    "wiki_title": title,
                    "chunk_index": len(chunks),
                },
            )
        )
    return chunks
```

- [ ] **Step 2: Verify Wikipedia API works**

Run:
```bash
uv run python -c "
from wikipedia import search_wikipedia, fetch_wikipedia, chunk_page
results = search_wikipedia('Albert Einstein physics', limit=3)
print(f'Search returned {len(results)} results')
for r in results:
    print(f'  - {r[\"title\"]}: {r[\"snippet\"][:80]}')
text = fetch_wikipedia(results[0]['title'])
print(f'Page text length: {len(text)} chars')
chunks = chunk_page(results[0]['title'], text)
print(f'Chunked into {len(chunks)} chunks')
print(f'First chunk doc_id: {chunks[0].doc_id}')
print(f'First chunk wiki_url: {chunks[0].metadata[\"wiki_url\"]}')
"
```

Expected: 3 search results, page text fetched, chunks created with `wiki:` prefixed doc_ids and `wiki_url` in metadata.

- [ ] **Step 3: Commit**

```bash
git add wikipedia.py
git commit -m "feat: add Wikipedia API client for FRAMES evaluation"
```

---

### Task 3: Wikipedia executor

**Files:**
- Create: `wiki_executor.py`

This is a variant of `executor.py` that replaces `search_corpus` with `search_wikipedia` and `read_wikipedia` tools. The tool-calling loop structure is identical.

- [ ] **Step 1: Create wiki_executor.py**

```python
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
        # Track which Wikipedia URLs we've retrieved (for metrics)
        retrieved_urls: set[str] = set()
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

                        # Track the URL for retrieval metrics
                        wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        retrieved_urls.add(wiki_url)

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
```

Note: The `Document` import is needed in the `finish_step` branch. Add it to the imports:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add wiki_executor.py
git commit -m "feat: add Wikipedia executor with search/read/finish tools"
```

---

### Task 4: WikiSearchAgent

**Files:**
- Create: `wiki_agent.py`

- [ ] **Step 1: Create wiki_agent.py**

```python
"""WikiSearchAgent — agent harness wired to Wikipedia for FRAMES evaluation.

Subclass of BaseAgent that replaces the local DocumentStore/HybridSearcher
with live Wikipedia API tools. Reuses the existing QueryPlanner, Evaluator,
and ContextPruner unchanged.
"""

from __future__ import annotations

from config import (
    AgentResult,
    ContextEntry,
    Document,
    LLMConfig,
    SearchConfig,
)
from agent import BaseAgent, AgentTrace
from evaluator import Evaluator
from planner import QueryPlanner
from prune import ContextPruner
from rerank import Reranker
from wiki_executor import WikiExecutor


class WikiSearchAgent(BaseAgent):
    """Concrete agent wired to Wikipedia via live API calls.

    Used for FRAMES benchmark evaluation. Same plan → execute → evaluate → prune
    loop as SearchAgent, but the executor calls Wikipedia instead of a local store.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        search_config: SearchConfig | None = None,
    ) -> None:
        cfg = search_config or SearchConfig()
        reranker = Reranker(cfg, llm_config)
        planner = QueryPlanner(llm_config)
        executor = WikiExecutor(reranker, llm_config, cfg)
        evaluator = Evaluator(llm_config)
        pruner = ContextPruner(llm_config, cfg)
        super().__init__(planner, executor, evaluator, pruner, cfg)
        self._llm_config = llm_config

    def search(self, query: str) -> AgentResult:
        """Run the full agentic search over Wikipedia and return results."""
        context, trace = self.run(query)

        # Collect unique source wiki URLs from context entries
        seen_urls: set[str] = set()
        sources: list[Document] = []
        for entry in context:
            url = entry.doc_id.split("#")[0]  # "wiki:Title" part
            if url not in seen_urls:
                sources.append(
                    Document(
                        doc_id=entry.doc_id,
                        text=entry.text,
                        metadata={"source": url},
                    )
                )
                seen_urls.add(url)

        return AgentResult(
            answer="",
            sources=sources,
            steps_taken=len([s for s in trace.steps if s["event"] == "execute_done"]),
            context_snapshot=context,
        )
```

- [ ] **Step 2: Verify the agent wires up correctly**

Run:
```bash
uv run python -c "
import os
from config import LLMConfig, SearchConfig
from wiki_agent import WikiSearchAgent

llm = LLMConfig(api_key=os.environ['ANTHROPIC_API_KEY'])
cfg = SearchConfig(max_agent_steps=3)
agent = WikiSearchAgent(llm, cfg)
print('WikiSearchAgent created successfully')
print(f'  planner: {type(agent._planner).__name__}')
print(f'  executor: {type(agent._executor).__name__}')
print(f'  evaluator: {type(agent._evaluator).__name__}')
print(f'  pruner: {type(agent._pruner).__name__}')
"
```

Expected: prints `WikiSearchAgent created successfully` with correct component types.

- [ ] **Step 3: Commit**

```bash
git add wiki_agent.py
git commit -m "feat: add WikiSearchAgent for FRAMES evaluation"
```

---

### Task 5: FRAMES evaluation runner and metrics

**Files:**
- Create: `frames_eval.py`

This is the main CLI script. It loads the FRAMES dataset, runs the agent on each question, computes metrics matching the Context-1 paper, and writes results to JSON.

- [ ] **Step 1: Create frames_eval.py**

```python
"""FRAMES benchmark evaluation — runs WikiSearchAgent on FRAMES questions.

Usage:
    uv run python frames_eval.py [--limit N] [--output results.json]

Metrics (matching Context-1 paper):
  - Final Answer Found: LLM judge checks if retrieved context contains the answer
  - Retrieval F1: precision/recall of retrieved Wikipedia URLs vs gold URLs
  - Answer Accuracy: LLM judge on generated answer vs gold answer (FRAMES paper metric)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
from datasets import load_dataset

from config import LLMConfig, SearchConfig
from generate import AnswerGenerator
from wiki_agent import WikiSearchAgent


# ---------------------------------------------------------------------------
# Data models for evaluation results
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_idx: int
    question: str
    gold_answer: str
    gold_urls: list[str]
    retrieved_urls: list[str]
    generated_answer: str
    final_answer_found: bool
    answer_accurate: bool
    precision: float
    recall: float
    f1: float
    steps_taken: int
    context_entries: int
    elapsed_seconds: float
    error: str | None = None


@dataclass
class EvalSummary:
    total_questions: int
    final_answer_found_rate: float
    answer_accuracy_rate: float
    mean_retrieval_precision: float
    mean_retrieval_recall: float
    mean_retrieval_f1: float
    mean_steps: float
    mean_elapsed: float


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_gold_urls(row: dict) -> list[str]:
    """Extract gold Wikipedia URLs from a FRAMES dataset row."""
    urls = []
    # Try the wiki_links JSON field first
    wiki_links = row.get("wiki_links", "")
    if wiki_links:
        try:
            parsed = json.loads(wiki_links) if isinstance(wiki_links, str) else wiki_links
            if isinstance(parsed, list):
                urls.extend(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    # Also check individual link columns
    if not urls:
        for i in range(1, 15):
            col = f"wikipedia_link_{i}"
            if col in row and row[col]:
                urls.append(row[col])

    # Normalize URLs: strip trailing slashes, ensure consistent format
    normalized = []
    for url in urls:
        if isinstance(url, str) and url.strip():
            normalized.append(url.strip().rstrip("/"))
    return normalized


def extract_retrieved_urls(context_snapshot: list) -> list[str]:
    """Extract unique Wikipedia URLs from agent context entries."""
    urls: set[str] = set()
    for entry in context_snapshot:
        # doc_id format: "wiki:PageTitle#chunk_N"
        doc_id = entry.doc_id if hasattr(entry, "doc_id") else entry.get("doc_id", "")
        if doc_id.startswith("wiki:"):
            title = doc_id.split("#")[0].replace("wiki:", "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            urls.add(url)
    return list(urls)


def normalize_wiki_url(url: str) -> str:
    """Normalize a Wikipedia URL to a canonical form for comparison."""
    url = url.strip().rstrip("/")
    # Handle both /wiki/ and full URLs
    if "/wiki/" in url:
        title = url.split("/wiki/")[-1]
        # Decode percent encoding, normalize spaces/underscores
        from urllib.parse import unquote
        title = unquote(title).replace("_", " ").strip().lower()
        return title
    return url.lower()


def retrieval_f1(gold_urls: list[str], retrieved_urls: list[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 between gold and retrieved Wikipedia URLs."""
    if not gold_urls:
        return (1.0, 1.0, 1.0) if not retrieved_urls else (0.0, 1.0, 0.0)
    if not retrieved_urls:
        return (0.0, 0.0, 0.0)

    gold_normalized = {normalize_wiki_url(u) for u in gold_urls}
    retrieved_normalized = {normalize_wiki_url(u) for u in retrieved_urls}

    true_positives = len(gold_normalized & retrieved_normalized)
    precision = true_positives / len(retrieved_normalized) if retrieved_normalized else 0.0
    recall = true_positives / len(gold_normalized) if gold_normalized else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def judge_final_answer_found(
    context_texts: list[str],
    gold_answer: str,
    client: anthropic.Anthropic,
    model: str,
) -> bool:
    """LLM judge: does the retrieved context contain the answer?"""
    context_combined = "\n\n---\n\n".join(context_texts[:20])  # cap for token budget
    prompt = (
        f"You are an evaluation judge. Determine whether the following retrieved "
        f"context contains sufficient information to answer the question.\n\n"
        f"Gold answer: {gold_answer}\n\n"
        f"Retrieved context:\n{context_combined}\n\n"
        f"Does the context contain the information needed to produce the gold answer? "
        f"Reply with ONLY 'YES' or 'NO'."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return False


def judge_answer_accuracy(
    generated_answer: str,
    gold_answer: str,
    question: str,
    client: anthropic.Anthropic,
    model: str,
) -> bool:
    """LLM judge: does the generated answer match the gold answer?

    Uses the FRAMES paper's evaluation approach: checks if the meaning and
    vital facts of the gold answer are present in the generated answer.
    """
    prompt = (
        f"You are an evaluation judge. Determine whether the predicted answer "
        f"contains the same meaning and vital facts as the gold answer.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Predicted answer: {generated_answer}\n\n"
        f"Does the predicted answer contain the essential meaning and facts of "
        f"the gold answer? Reply with ONLY 'YES' or 'NO'."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    llm_config: LLMConfig,
    search_config: SearchConfig,
    limit: int | None = None,
    output_path: str = "frames_results.json",
) -> EvalSummary:
    """Run FRAMES evaluation end-to-end."""
    print("Loading FRAMES dataset from HuggingFace...")
    ds = load_dataset("google/frames-benchmark", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"Evaluating {len(ds)} questions")

    agent = WikiSearchAgent(llm_config, search_config)
    generator = AnswerGenerator(llm_config)
    judge_client = anthropic.Anthropic(api_key=llm_config.api_key)

    results: list[QuestionResult] = []

    for idx, row in enumerate(ds):
        question = row["Prompt"]
        gold_answer = row["Answer"]
        gold_urls = extract_gold_urls(row)

        print(f"\n[{idx + 1}/{len(ds)}] {question[:100]}...")

        start = time.time()
        try:
            # Tier 1: Agent search
            agent_result = agent.search(question)

            # Tier 2: Generate answer
            gen_result = generator.generate(question, agent_result)

            elapsed = time.time() - start

            # Extract retrieved URLs from context
            retrieved_urls = extract_retrieved_urls(agent_result.context_snapshot)

            # Context texts for judging
            context_texts = [e.text for e in agent_result.context_snapshot]

            # Metrics
            faf = judge_final_answer_found(
                context_texts, gold_answer, judge_client, llm_config.model
            )
            accuracy = judge_answer_accuracy(
                gen_result.answer, gold_answer, question, judge_client, llm_config.model
            )
            prec, rec, f1 = retrieval_f1(gold_urls, retrieved_urls)

            result = QuestionResult(
                question_idx=idx,
                question=question,
                gold_answer=gold_answer,
                gold_urls=gold_urls,
                retrieved_urls=retrieved_urls,
                generated_answer=gen_result.answer,
                final_answer_found=faf,
                answer_accurate=accuracy,
                precision=prec,
                recall=rec,
                f1=f1,
                steps_taken=agent_result.steps_taken,
                context_entries=len(agent_result.context_snapshot),
                elapsed_seconds=elapsed,
            )

            print(f"  FAF={faf} | Acc={accuracy} | F1={f1:.2f} | "
                  f"Steps={agent_result.steps_taken} | {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            result = QuestionResult(
                question_idx=idx,
                question=question,
                gold_answer=gold_answer,
                gold_urls=gold_urls,
                retrieved_urls=[],
                generated_answer="",
                final_answer_found=False,
                answer_accurate=False,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                steps_taken=0,
                context_entries=0,
                elapsed_seconds=elapsed,
                error=str(e),
            )
            print(f"  ERROR: {e}")

        results.append(result)

        # Write intermediate results after each question
        _save_results(results, output_path)

    # Compute summary
    n = len(results)
    summary = EvalSummary(
        total_questions=n,
        final_answer_found_rate=sum(r.final_answer_found for r in results) / n if n else 0,
        answer_accuracy_rate=sum(r.answer_accurate for r in results) / n if n else 0,
        mean_retrieval_precision=sum(r.precision for r in results) / n if n else 0,
        mean_retrieval_recall=sum(r.recall for r in results) / n if n else 0,
        mean_retrieval_f1=sum(r.f1 for r in results) / n if n else 0,
        mean_steps=sum(r.steps_taken for r in results) / n if n else 0,
        mean_elapsed=sum(r.elapsed_seconds for r in results) / n if n else 0,
    )

    _save_results(results, output_path, summary)
    _print_summary(summary)
    return summary


def _save_results(
    results: list[QuestionResult],
    path: str,
    summary: EvalSummary | None = None,
) -> None:
    """Write results to JSON file."""
    data = {
        "results": [asdict(r) for r in results],
    }
    if summary:
        data["summary"] = asdict(summary)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def _print_summary(summary: EvalSummary) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print("FRAMES Evaluation Results")
    print("=" * 60)
    print(f"  Questions evaluated:     {summary.total_questions}")
    print(f"  Final Answer Found:      {summary.final_answer_found_rate:.3f}")
    print(f"  Answer Accuracy:         {summary.answer_accuracy_rate:.3f}")
    print(f"  Retrieval Precision:     {summary.mean_retrieval_precision:.3f}")
    print(f"  Retrieval Recall:        {summary.mean_retrieval_recall:.3f}")
    print(f"  Retrieval F1:            {summary.mean_retrieval_f1:.3f}")
    print(f"  Mean Steps/Question:     {summary.mean_steps:.1f}")
    print(f"  Mean Time/Question:      {summary.mean_elapsed:.1f}s")
    print("=" * 60)

    # Context-1 paper comparison
    print("\nContext-1 Paper Comparison (FRAMES):")
    print(f"  {'Metric':<25} {'Ours':>8} {'Sonnet-4.5':>12} {'C1 (1x)':>10} {'C1 (4x)':>10}")
    print(f"  {'Final Answer Found':<25} {summary.final_answer_found_rate:>8.3f} {'0.960':>12} {'0.870':>10} {'0.960':>10}")
    print(f"  {'Retrieval F1':<25} {summary.mean_retrieval_f1:>8.3f} {'0.820':>12} {'0.650':>10} {'0.790':>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FRAMES benchmark evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate (default: all 824)")
    parser.add_argument("--output", type=str, default="frames_results.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--max-steps", type=int, default=12, help="Max agent steps per question")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    llm_config = LLMConfig(api_key=api_key, model=args.model)
    search_config = SearchConfig(max_agent_steps=args.max_steps)

    run_evaluation(llm_config, search_config, limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify FRAMES dataset loads**

Run:
```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('google/frames-benchmark', split='test')
print(f'FRAMES dataset: {len(ds)} questions')
print(f'Columns: {ds.column_names}')
row = ds[0]
print(f'Sample question: {row[\"Prompt\"][:100]}...')
print(f'Sample answer: {row[\"Answer\"]}')
"
```

Expected: 824 questions loaded, column names printed, sample displayed.

- [ ] **Step 3: Commit**

```bash
git add frames_eval.py
git commit -m "feat: add FRAMES benchmark evaluation runner with metrics"
```

---

### Task 6: End-to-end smoke test

**Files:** None (testing only)

- [ ] **Step 1: Run evaluation on 2 questions**

Run:
```bash
uv run python frames_eval.py --limit 2 --output frames_smoke_test.json
```

Expected: Both questions run through the full pipeline (plan → execute with Wikipedia → evaluate → generate → judge). Output shows FAF, accuracy, F1 per question plus a summary.

- [ ] **Step 2: Inspect the output JSON**

Run:
```bash
uv run python -c "
import json
data = json.loads(open('frames_smoke_test.json').read())
for r in data['results']:
    print(f'Q{r[\"question_idx\"]}: FAF={r[\"final_answer_found\"]} Acc={r[\"answer_accurate\"]} F1={r[\"f1\"]:.2f}')
    print(f'  Retrieved URLs: {r[\"retrieved_urls\"]}')
    print(f'  Gold URLs: {r[\"gold_urls\"]}')
    print()
"
```

Expected: JSON has correct structure with all fields populated.

- [ ] **Step 3: Fix any issues found during smoke test**

If errors occur, debug and fix. Common issues:
- Wikipedia API rate limiting → add retry logic to `wikipedia.py`
- FRAMES column name mismatch → check actual dataset schema vs code
- Anthropic API errors → check rate limits, token counts

- [ ] **Step 4: Clean up smoke test file and commit**

```bash
rm frames_smoke_test.json
git add -A
git commit -m "test: verify FRAMES evaluation pipeline end-to-end"
```

---

### Task 7: Add .gitignore entries and update CLAUDE.md

**Files:**
- Modify: `.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update .gitignore**

Append to `.gitignore`:

```
.env
frames_results.json
frames_smoke_test.json
*.json
!package.json
```

Wait — that's too broad. Instead:

```
.env
frames_results*.json
```

- [ ] **Step 2: Update CLAUDE.md commands section**

Add to the Commands section in CLAUDE.md:

```markdown
# Run FRAMES benchmark evaluation
uv run python frames_eval.py --limit 10 --output frames_results.json

# Full evaluation (824 questions — costs ~$50-100 in API calls)
uv run python frames_eval.py --output frames_results.json
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore CLAUDE.md
git commit -m "docs: update CLAUDE.md with FRAMES eval commands, update gitignore"
```
