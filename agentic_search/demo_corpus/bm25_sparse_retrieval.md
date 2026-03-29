# BM25 and Sparse Retrieval

BM25 (Best Matching 25) is a ranking function used in information retrieval to estimate the relevance of documents to a given search query. It is the most widely used sparse retrieval method and remains competitive with modern dense retrieval approaches for many tasks.

## How BM25 Works

BM25 is a bag-of-words retrieval function that ranks documents based on the query terms appearing in each document. The score for a document D given a query Q is:

    score(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))

Where:
- f(qi, D) is the term frequency of query term qi in document D
- |D| is the document length
- avgdl is the average document length in the collection
- k1 and b are free parameters (typically k1 ∈ [1.2, 2.0], b = 0.75)
- IDF(qi) is the inverse document frequency of the query term

## Strengths of BM25

- **Exact keyword matching**: BM25 excels when the query contains specific terms (names, codes, identifiers) that must appear verbatim in relevant documents.
- **No training required**: Unlike dense retrieval models, BM25 requires no training data or GPU resources.
- **Interpretable**: The scoring function is fully transparent — you can trace exactly why a document was ranked highly.
- **Fast indexing**: Building a BM25 index is fast and memory-efficient compared to generating embeddings for every document.

## Limitations

- **Vocabulary mismatch**: BM25 cannot match semantically similar terms with different surface forms (e.g., "car" vs "automobile").
- **No semantic understanding**: The ranking is purely based on term overlap, with no understanding of meaning.
- **Query length sensitivity**: Very short queries (1-2 words) or very long queries can produce suboptimal rankings.

## BM25 in Hybrid Retrieval

Modern retrieval systems combine BM25 with dense retrieval to get the best of both worlds. The typical approach:

1. Run both BM25 (sparse) and embedding-based (dense) retrieval in parallel.
2. Fuse the results using Reciprocal Rank Fusion (RRF) or other fusion methods.
3. Rerank the fused results using a cross-encoder or LLM.

This hybrid approach is used by Chroma's Context-1 agent, which issues both sparse and dense queries through the `search_corpus` tool and fuses them via RRF.

## Reciprocal Rank Fusion (RRF)

RRF is a simple, effective method for combining ranked lists from multiple retrieval systems:

    RRF_score(d) = Σ 1 / (k + rank_i(d))

Where k is a constant (typically 60) and rank_i(d) is the rank of document d in the i-th ranked list. RRF is parameter-free beyond k, requires no training, and handles different score distributions gracefully.
