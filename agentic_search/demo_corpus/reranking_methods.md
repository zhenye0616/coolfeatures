# Reranking in Information Retrieval

Reranking is a second-stage retrieval process that rescores an initial set of candidate documents to improve precision. It bridges the gap between fast but approximate first-stage retrieval (BM25, ANN search) and the quality needed for downstream tasks.

## Why Reranking Matters

First-stage retrieval methods optimize for recall — casting a wide net to avoid missing relevant documents. However, many retrieved documents may be only tangentially related. Reranking applies a more expensive but more accurate model to distinguish the truly relevant documents from the noise.

In agentic search systems like Chroma's Context-1, the reranker operates after reciprocal rank fusion (RRF) combines dense and sparse retrieval results. The top 50 fused candidates are scored by the reranker, which selects results within a per-call token budget.

## Cross-Encoder Reranking

Cross-encoders process the query and document together as a single input, allowing deep interaction between query and document tokens. This produces more accurate relevance scores than bi-encoder (embedding-based) approaches, which encode query and document independently.

Popular cross-encoder models:
- **ms-marco-MiniLM-L-6-v2**: A lightweight cross-encoder trained on the MS MARCO passage ranking dataset.
- **BGE-reranker-v2-m3**: A multilingual reranker from BAAI that supports 100+ languages.
- **Cohere Rerank**: A commercial reranking API with strong zero-shot performance.

## LLM-Based Reranking

Recent approaches use large language models as rerankers. The LLM is given the query and a batch of candidates and asked to score each for relevance. Advantages include richer reasoning about relevance and the ability to handle complex queries. The main trade-off is cost and latency.

Context-1's agent harness uses this approach: after RRF fusion produces candidates, the reranker scores them and greedily selects documents within a per-call token budget. This ensures the agent's context window is used efficiently.

## Token Budget Selection

A key practical concern is the token budget — how many tokens of retrieved content to pass to the LLM. Too little context risks missing relevant information; too much risks context rot (degrading LLM performance with irrelevant material).

The typical approach is greedy selection: sort candidates by reranking score, then pack them into the budget top-down until the limit is reached. Context-1 uses a per-call token budget for each search_corpus invocation, and a separate soft limit for the agent's total accumulated context.

## Reranking in the Three-Tier Architecture

In the three-tier architecture used by Context-1:
- **Tier 3 (Storage)**: Vector DB + BM25 produce raw candidates
- **Reranking**: Sits between Tier 3 and Tier 1, filtering candidates
- **Tier 1 (Search Agent)**: Receives reranked results, manages context
- **Tier 2 (Generation)**: Produces the final answer from curated context
