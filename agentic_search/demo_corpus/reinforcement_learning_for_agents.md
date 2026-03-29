# Reinforcement Learning for Training Search Agents

Reinforcement learning (RL) is increasingly used to train LLM-based agents for specific tasks like search, coding, and tool use. Unlike supervised fine-tuning (SFT), which teaches the model to imitate demonstrations, RL optimizes the model's behavior directly against a reward signal tied to task outcomes.

## Why RL for Search Agents?

SFT can teach a model the basic patterns of tool use — when to call search, how to format queries. But it cannot teach the model to optimize its search strategy for the best overall outcome. RL fills this gap by:

- Rewarding the model for finding relevant documents (outcome reward).
- Penalizing inefficient strategies like redundant searches or excessive pruning.
- Encouraging exploration of diverse search queries.

## Context-1's Training Approach

Chroma trained Context-1 using a combination of SFT and RL, starting from the gpt-oss-20B base model.

### Supervised Fine-Tuning Phase
First, the model was trained on demonstrations of successful search trajectories — showing it how to decompose queries, call search_corpus, and manage context.

### Reinforcement Learning Phase
Then, RL was applied using a reward function with multiple components:

**Outcome Reward (F-beta score)**:
- Measured by F-beta with beta set high (recall weighted 16x over precision).
- Rationale: Missing a relevant document is worse than including an irrelevant one, because the downstream generation model can filter but cannot recover what was never retrieved.

**Process Reward (Trajectory Recall)**:
- Credits the agent for encountering relevant documents during search, even if they were later pruned.
- Without this, the agent converges to issuing one broad search and quitting.

**Penalties**:
- **Repeated pruning penalty**: Discourages pruning one entry at a time in long streaks.
- **Turn count penalty**: Discourages diminishing-return search loops where additional turns add little value.

### Training in the Production Harness
A critical detail: RL rollouts run through the same tools, prompts, and execution environments that the model encounters in production. This ensures the model learns strategies that work in the real harness, not in a simplified simulation.

## Synthetic Task Generation

Real-world agentic search tasks are hard to collect at scale. Chroma built a synthetic generation pipeline across four domains:

1. **Web**: General knowledge questions requiring multi-document retrieval.
2. **Finance (SEC filings)**: Questions about company financials requiring specific data extraction.
3. **Legal (USPTO patents)**: Prior art search requiring technical document retrieval.
4. **Email**: Information retrieval across email threads and attachments.

The pipeline:
1. Gather supporting documents with unique facts.
2. Generate obfuscated clues (indirect references to facts) and a question.
3. Verify by extracting verbatim quotes and checking they appear in source text.
4. Use an LLM judge to assess task quality, minimizing need for human annotation.

Over 8,000 synthetic tasks were generated for training.

## Results

Context-1 achieves retrieval performance comparable to frontier-scale LLMs while being 10x faster and 25x cheaper. The key insight is that purpose-built, RL-trained subagents optimized for specific capabilities outperform general-purpose LLMs repurposed for the same task.
