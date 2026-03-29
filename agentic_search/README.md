# Agentic Search

An agentic search system with hybrid retrieval, self-editing context, and multi-hop reasoning — inspired by [Chroma Context-1](https://www.trychroma.com/research/context-1).

## Architecture

Three-tier design matching the Context-1 agent harness:

| Tier | Component | Role |
|------|-----------|------|
| 1 | **SearchAgent** | Plans, retrieves, prunes, curates context |
| 2 | **AnswerGenerator** | Synthesizes final answer from curated context |
| 3 | **DocumentStore** | ChromaDB (dense) + BM25 (sparse) storage |

## Core Primitives

### 1. Hybrid Search (`search.py`)
Parallel dense (embedding) + sparse (BM25) retrieval, fused via **Reciprocal Rank Fusion** (RRF). Deduplication tracks seen chunk IDs across all search calls.

### 2. Reranking (`rerank.py`)
LLM-based relevance scoring of fused candidates. Greedily selects top results within a per-call token budget.

### 3. Self-Editing Context (`prune.py`)
When accumulated context exceeds a soft limit, the agent reviews entries and prunes irrelevant ones — freeing capacity and combating **context rot**.

### 4. Query Decomposition (`planner.py`)
Breaks complex queries into targeted sub-queries with dependency tracking. Enables multi-hop retrieval.

### 5. Agent Harness (`agent.py`, `executor.py`, `evaluator.py`)
The observe → reason → act loop:
- **QueryPlanner** decomposes the query into PlanSteps
- **Executor** runs a tool-calling loop per step (`search_corpus`, `finish_step`)
- **Evaluator** decides: continue, sufficient, or add more steps
- **BaseAgent** orchestrates the full loop with pruning at soft limits

## Quick Start

```bash
cd agentic_search
uv sync
export ANTHROPIC_API_KEY="your-key"
uv run streamlit run app.py
```

## File Map

```
config.py       — Data models (Document, ContextEntry, PlanStep, etc.)
storage.py      — Tier 3: ChromaDB + BM25 wrapper
search.py       — Hybrid search with RRF fusion
rerank.py       — LLM-based reranking with token budget
prune.py        — Self-editing context (soft limit pruning)
planner.py      — Query decomposition into PlanSteps
executor.py     — Tool-calling loop (search_corpus, finish_step)
evaluator.py    — Plan evaluation and re-planning
agent.py        — BaseAgent + SearchAgent orchestration
generate.py     — Tier 2: Answer generation with citations
ingest.py       — Markdown chunking and corpus loading
app.py          — Streamlit demo UI
demo_corpus/    — 10 markdown docs on AI/search topics
```
