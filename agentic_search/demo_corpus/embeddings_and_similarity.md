# Text Embeddings and Semantic Similarity

Text embeddings are dense vector representations that capture the semantic meaning of text. They are the foundation of dense retrieval systems and enable semantic search — finding documents that are conceptually similar to a query, even without shared keywords.

## How Text Embeddings Work

Embedding models are typically transformer-based encoder models trained on large text corpora. During training, the model learns to map semantically similar texts to nearby points in a high-dimensional vector space.

Common training objectives include:
- **Contrastive learning**: Pulling positive pairs (semantically similar texts) together while pushing negative pairs apart.
- **Knowledge distillation**: Training a smaller model to replicate the embeddings of a larger teacher model.
- **Masked language modeling**: Pre-training on predicting masked tokens, then fine-tuning for similarity tasks.

## Popular Embedding Models

- **all-MiniLM-L6-v2**: A lightweight 384-dimensional model from Sentence Transformers. Fast inference, good quality for general-purpose similarity. Used as ChromaDB's default embedding function.
- **BGE (BAAI General Embedding)**: A family of models from the Beijing Academy of AI. BGE-large-en-v1.5 produces 1024-dimensional embeddings and ranks highly on the MTEB benchmark.
- **text-embedding-3-small/large**: OpenAI's embedding models. The large variant produces 3072-dimensional vectors with state-of-the-art performance.
- **Cohere embed-v3**: Supports multiple languages and produces embeddings optimized for search, classification, and clustering.
- **E5 (EmbEddings from bidirEctional Encoder rEpresentations)**: Microsoft's embedding models, trained with a "query: " and "passage: " prefix convention.

## Embedding Dimensions and Trade-offs

Higher-dimensional embeddings can capture more nuanced semantic information but require more storage and compute. Typical dimensions range from 384 (MiniLM) to 3072 (OpenAI large). Recent work on Matryoshka Representation Learning allows truncating embeddings to lower dimensions with graceful degradation.

## Semantic vs Lexical Search

Semantic search (using embeddings) and lexical search (using BM25) have complementary strengths:

| Aspect | Semantic (Dense) | Lexical (Sparse) |
|--------|-----------------|-------------------|
| Synonyms | Handles well | Misses |
| Exact terms | May miss | Handles well |
| Speed | Requires ANN index | Very fast |
| Training | Needs embedding model | No training needed |
| Out-of-domain | May degrade | Generally robust |

The best retrieval systems combine both approaches through hybrid search, using reciprocal rank fusion to merge the results.
