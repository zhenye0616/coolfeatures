# Agentic Search Systems

Agentic search is a paradigm where an LLM-powered agent actively plans, retrieves, evaluates, and iterates to answer complex queries. Unlike single-pass RAG, which performs one retrieval step, agentic search uses a loop of LLM calls with search tools to progressively build up the evidence needed for an answer.

## Why Agentic Search?

Many real-world queries require multi-hop retrieval — where the answer depends on combining information from multiple sources, and the output of one search informs the next. Examples:

- "Compare the safety profiles of Drug A and Drug B for patients with condition C" → requires retrieving info about each drug and the condition separately.
- "What were the revenue impacts of the 2023 policy change across the top 3 affected divisions?" → requires identifying the divisions, then finding revenue data for each.

Single-pass retrieval cannot handle these queries because it issues one query and gets one set of results. Agentic search decomposes the question and iteratively builds up context.

## The Agent Harness

An agent harness is the complete system scaffold within which a search agent operates. It includes:

- **System prompt**: Frames the agent's role and available tools.
- **Tool definitions**: Names, descriptions, and input schemas for each tool.
- **Loop structure**: How tool outputs are fed back to the model.
- **Pre/post-processing**: Applied to inputs and outputs at each step.
- **Stopping criteria**: When to terminate the search loop.

The harness is critical because the model learns to expect specific tool names, output formats, and loop structures during training. Running a trained agent outside its harness will not reproduce its performance.

## Key Components (Chroma Docs Pattern)

From Chroma's agentic search documentation:

### QueryPlanner
Generates a query plan — a list of PlanSteps, each tracking its status (Pending, Success, Failure, Cancelled) and dependencies on other steps. The planner is an iterator that yields batches of ready steps.

### Executor
Solves a single PlanStep via a tool-calling loop. Calls search tools, evaluates results, and produces a StepOutcome with candidate answers and supporting evidence.

### Evaluator
Considers the plan and outcome history to decide: continue with remaining steps, declare sufficient evidence, or add new steps.

### SearchAgent
Extends BaseAgent with specific tools (search_corpus) and prompts tailored to the retrieval task.

## Context-1: A Specialized Search Agent

Chroma's Context-1 is a 20B parameter model trained specifically for agentic search using RL + SFT. Key features:

- **Parallel tool calling**: Averages 2.56 tool calls per turn.
- **Self-editing context**: Prunes irrelevant entries when context exceeds a soft limit (0.94 accuracy).
- **Deduplication**: The harness tracks seen chunk IDs and excludes them from subsequent searches.
- **Designed as a subagent**: Retrieves and curates documents; a separate frontier model generates the answer.

## Three-Tier Architecture

- **Tier 1 (Search Agent)**: Plans, searches, prunes, curates context.
- **Tier 2 (Frontier LLM)**: Generates the final answer from curated context.
- **Tier 3 (Storage)**: Vector database + BM25 index — stores and retrieves documents.

This separation allows using a smaller, specialized model for retrieval (10x faster, 25x cheaper) while reserving the expensive frontier model only for the final generation step.
