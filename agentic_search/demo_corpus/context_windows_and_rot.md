# Context Windows and Context Rot

## What Is a Context Window?

A context window is the maximum amount of text (measured in tokens) that a language model can process in a single forward pass. Modern LLMs have context windows ranging from 4K tokens (older models) to 200K tokens (Claude 3.5) and even 1M+ tokens (Gemini 1.5 Pro).

The context window determines how much information can be provided to the model at once — including the system prompt, conversation history, retrieved documents, and the user's query.

## Context Rot

Context rot is the phenomenon where LLM performance degrades as irrelevant or redundant material accumulates in the context window. Even if the total content fits within the model's context limit, adding tangential documents can:

- **Dilute attention**: The model may attend to irrelevant passages instead of the key evidence.
- **Increase hallucination**: Conflicting or noisy context can lead the model to generate incorrect information.
- **Degrade instruction following**: Long contexts make it harder for the model to follow the original instructions.

Chroma's research on context rot (published at research.trychroma.com) found that increasing input tokens beyond a certain threshold causes measurable performance degradation, even when the relevant information is present in the context.

## Context Rot in Agentic Search

Agentic search is especially vulnerable to context rot because the agent accumulates documents over multiple retrieval steps. After 5-10 search iterations, the context may be packed with dozens of document chunks — many of which were relevant to earlier sub-queries but not to the current focus.

This is the core motivation for Context-1's self-editing context mechanism. Rather than passively accumulating all retrieved documents, the agent actively decides which entries to keep and which to discard.

## Self-Editing Context

Context-1 introduces a self-editing context strategy:

1. A **soft limit** is defined (e.g., 18K tokens out of a 32K budget).
2. When the accumulated context exceeds this soft limit, the agent reviews all entries.
3. Entries deemed irrelevant to the current query focus are pruned.
4. This frees capacity for fresh retrieval results.

A concrete trajectory:
- Search 1 → context at 11K/32K tokens
- Search 2 → context at 24K/32K (exceeds soft limit)
- **Prune**: Agent reviews and removes 5 entries → context drops to 14K
- Search 3 → context at 21K/32K
- Search 4 → context at 28K/32K (exceeds soft limit again)
- **Prune**: Agent removes 3 entries → context drops to 20K
- Finish: Agent returns curated context with high-relevance entries

Context-1 achieves 0.94 prune accuracy — meaning 94% of its pruning decisions correctly identify and remove irrelevant entries while retaining relevant ones.

## Strategies for Managing Context

Beyond self-editing, other context management strategies include:

- **Summarization**: Compress older context entries into shorter summaries.
- **Sliding window**: Only keep the most recent N entries.
- **Relevance scoring**: Continuously re-score entries and drop the lowest.
- **Hierarchical context**: Maintain summaries at different levels of detail.

The self-editing approach is the most flexible because the agent can make nuanced decisions about what to keep based on the evolving query context.
