# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic search system with hybrid retrieval, self-editing context, and multi-hop reasoning — inspired by Chroma Context-1. Three-tier architecture: SearchAgent (retrieval) → AnswerGenerator (generation) → DocumentStore (storage).

## Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit demo UI
uv run streamlit run app.py

# Run a single module directly
uv run python -m agentic_search.<module>

# Run FRAMES benchmark evaluation (small sample)
uv run python frames_eval.py --limit 10 --output frames_results.json

# Full FRAMES evaluation (824 questions — costs ~$50-100 in API calls)
uv run python frames_eval.py --output frames_results.json
```

Requires `ANTHROPIC_API_KEY` environment variable. No test suite or linter is configured.

## Architecture

### Three-Tier Design

- **Tier 1 — SearchAgent** (`agent.py`): Orchestrates the retrieve-reason-act loop. `QueryPlanner` decomposes queries into dependency-aware `PlanStep`s, `Executor` runs tool-calling loops per step, `Evaluator` decides whether to continue/stop/add steps, and `ContextPruner` discards low-relevance entries when context exceeds the soft token limit.
- **Tier 2 — AnswerGenerator** (`generate.py`): Takes curated context from Tier 1, sorts by relevance, packs within a token budget, and produces a cited answer.
- **Tier 3 — DocumentStore** (`storage.py`): Wraps ChromaDB (dense embeddings) + BM25 (sparse keyword index) over the same corpus.

### Data Flow

```
Query → QueryPlanner → [PlanSteps with dependencies]
  → Executor (per step): HybridSearcher → RRF fusion → Reranker → ContextEntries
  → ContextPruner (if over soft limit)
  → Evaluator (sufficient / continue / add_steps)
  → AnswerGenerator → final answer with citations
```

### Key Modules

- `config.py` — Shared data models (`Document`, `ContextEntry`, `PlanStep`, `AgentResult`), `LLMConfig`/`SearchConfig` defaults, and `extract_json()` helper for robust JSON parsing from LLM output.
- `search.py` — Parallel dense + sparse retrieval fused via Reciprocal Rank Fusion (RRF). Tracks seen chunk IDs for deduplication across searches.
- `rerank.py` — Single LLM call scores candidates 0-10; greedy packing within `per_call_token_budget`.
- `prune.py` — LLM-driven context pruning at `context_soft_limit` to combat context rot.
- `planner.py` — Query decomposition into sub-queries with dependency tracking for multi-hop.
- `executor.py` — Tool-calling loop: LLM iteratively calls `search_corpus` or `finish_step`.
- `evaluator.py` — Post-batch evaluation deciding next action.
- `ingest.py` — Reads `.md` files, splits on paragraph boundaries into overlapping chunks, loads into DocumentStore.
- `app.py` — Streamlit UI with configurable parameters (sliders for all SearchConfig values).
- `wikipedia.py` — Wikipedia API client (search + fetch + chunk) for FRAMES evaluation.
- `wiki_executor.py` — Executor variant with `search_wikipedia`/`read_wikipedia` tools.
- `wiki_agent.py` — `WikiSearchAgent(BaseAgent)` wired to Wikipedia for benchmark evaluation.
- `frames_eval.py` — CLI evaluation runner for FRAMES benchmark with FAF, Retrieval F1, and Answer Accuracy metrics.

### Token Budget Hierarchy

Three budget levels control context management:
- `context_token_budget` (24k) — total agent context window
- `context_soft_limit` (18k) — triggers pruning
- `per_call_token_budget` (4k) — max tokens per individual search call

### Default LLM

`claude-sonnet-4-20250514` with 4096 max_tokens (configured in `LLMConfig`).
