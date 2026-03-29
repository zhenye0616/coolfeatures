# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) outputs by grounding them in external knowledge retrieved at inference time. The approach was first formalized by Lewis et al. in their 2020 paper at Facebook AI Research.

## The Problem RAG Solves

LLMs are trained on static datasets and have a knowledge cutoff. They can hallucinate — generating fluent but factually incorrect text. RAG addresses both issues by retrieving relevant documents from an external knowledge base before generating a response.

## How RAG Works

A standard RAG pipeline has three stages:

1. **Indexing**: Documents are split into chunks, embedded into vector representations using an embedding model, and stored in a vector database.

2. **Retrieval**: When a user submits a query, the query is embedded using the same model, and the most similar document chunks are retrieved via approximate nearest neighbor search.

3. **Generation**: The retrieved chunks are prepended to the query as context, and the LLM generates a response grounded in this evidence.

## Limitations of Single-Pass RAG

Single-pass RAG has well-documented limitations:

- **Query-document mismatch**: The user's query may not lexically match the relevant documents, leading to poor retrieval.
- **Single-hop constraint**: Complex questions that require combining information from multiple documents cannot be answered in a single retrieval step.
- **No reasoning over results**: The retrieval step is a static lookup — there is no mechanism to evaluate, filter, or iterate on the results.
- **Context window pollution**: Irrelevant retrieved documents occupy context space, potentially degrading the LLM's generation quality.

These limitations motivate agentic search, where an LLM actively plans, retrieves, evaluates, and iterates.

## Advanced RAG Techniques

Several techniques have been developed to improve RAG quality:

- **Hybrid retrieval**: Combining dense (embedding-based) and sparse (BM25/keyword) retrieval for better coverage.
- **Reranking**: Using a cross-encoder or LLM to rescore retrieved documents before passing them to the generator.
- **Query rewriting**: Reformulating the user's query to improve retrieval recall.
- **Chunk optimization**: Experimenting with chunk sizes, overlap, and hierarchical chunking strategies.

## RAG vs Fine-Tuning

RAG and fine-tuning serve complementary purposes. Fine-tuning adapts the model's weights to a specific domain or task. RAG provides dynamic access to external, up-to-date knowledge without modifying the model. In practice, many production systems combine both approaches.
