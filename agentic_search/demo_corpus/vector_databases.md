# Vector Databases

Vector databases are purpose-built storage systems optimized for storing, indexing, and querying high-dimensional vector embeddings. They are a critical component of modern AI applications, particularly retrieval-augmented generation (RAG) systems.

## What Are Embeddings?

Embeddings are dense numerical representations of data (text, images, audio) in a continuous vector space. Similar items are mapped to nearby points in this space. Text embedding models like OpenAI's text-embedding-3, Cohere's embed-v3, and open-source models like BGE and E5 convert text into vectors of 384 to 3072 dimensions.

## Key Vector Database Systems

### Chroma
Chroma is an open-source embedding database designed for AI applications. It provides a simple API for storing embeddings with metadata, querying by similarity, and filtering results. Chroma supports both ephemeral (in-memory) and persistent storage modes. It uses HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor search.

### Pinecone
Pinecone is a managed vector database service. It handles infrastructure, scaling, and index management automatically. Pinecone supports metadata filtering, namespaces for multi-tenancy, and hybrid search combining dense and sparse vectors.

### Weaviate
Weaviate is an open-source vector database that supports multiple vectorization modules. It offers a GraphQL API and supports hybrid search natively. Weaviate includes built-in generative search capabilities.

### Qdrant
Qdrant is an open-source vector similarity search engine written in Rust. It supports filtering, payload storage, and distributed deployment. Qdrant offers both gRPC and REST APIs.

## Approximate Nearest Neighbor (ANN) Search

Exact nearest neighbor search in high-dimensional spaces is computationally expensive (O(n) per query). Vector databases use ANN algorithms to trade small amounts of accuracy for orders-of-magnitude speedups:

- **HNSW**: Builds a multi-layer graph where each node is connected to its approximate nearest neighbors. Queries navigate this graph from coarse to fine layers. Offers excellent query performance with tunable recall.
- **IVF (Inverted File Index)**: Partitions the vector space into clusters. At query time, only the nearest clusters are searched. Reduces search space but requires a training step.
- **Product Quantization (PQ)**: Compresses vectors by splitting them into sub-vectors and quantizing each independently. Reduces memory usage at the cost of some accuracy.

## Distance Metrics

Common distance metrics for vector similarity:
- **Cosine similarity**: Measures the angle between vectors. Commonly used for text embeddings.
- **Euclidean distance (L2)**: Measures straight-line distance. Sensitive to vector magnitude.
- **Dot product**: Equivalent to cosine similarity when vectors are normalized.

## The Role of Vector Databases in Agentic Search

In agentic search systems, vector databases serve as Tier 3 — the storage and indexing layer. The search agent issues queries against the vector database, retrieves candidate documents, and iteratively refines its search based on the results. The combination of dense vector search with sparse keyword search (hybrid retrieval) provides the best coverage for diverse query types.
