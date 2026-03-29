# Tool Use and Function Calling in LLMs

Tool use (also called function calling) is a capability that allows LLMs to interact with external systems by generating structured tool invocations. Instead of only producing text, the model can request that specific functions be executed, receive the results, and continue reasoning.

## How Tool Use Works

1. **Tool definitions** are provided to the LLM as part of the system prompt or API call. Each tool has a name, description, and an input schema (typically JSON Schema).

2. The LLM generates a response that includes **tool_use blocks** — structured requests to invoke specific tools with specific arguments.

3. The **harness** (the system hosting the LLM) executes the tool calls and returns the results as **tool_result blocks**.

4. The LLM processes the results and either makes additional tool calls or generates a final text response.

## Tool Use in Agentic Search

In agentic search, the primary tool is `search_corpus`:

```json
{
  "name": "search_corpus",
  "description": "Search the document corpus using hybrid retrieval. Returns top reranked chunks.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query — be specific and targeted."
      }
    },
    "required": ["query"]
  }
}
```

The agent also typically has a `finish` tool to signal completion:

```json
{
  "name": "finish",
  "description": "Signal that search is complete. Provide the final curated context.",
  "input_schema": {
    "type": "object",
    "properties": {
      "summary": {"type": "string"}
    },
    "required": ["summary"]
  }
}
```

## Parallel Tool Calling

Modern LLMs can generate multiple tool calls in a single response. This is critical for agentic search:

- **Independent sub-queries** can be searched simultaneously.
- Context-1 averages 2.56 tool calls per turn, reducing total turns and latency.
- The harness executes parallel calls concurrently and returns all results together.

## The Observe-Reason-Act Loop

The tool-calling loop follows the observe-reason-act pattern:

1. **Observe**: The model receives the current state (accumulated context, tool results).
2. **Reason**: The model decides what to do next — search again, prune context, or finish.
3. **Act**: The model emits tool calls or a final response.

This loop continues until the model calls `finish` or a maximum iteration count is reached.

## Best Practices for Tool Definitions

- Use clear, specific descriptions that tell the model when and how to use each tool.
- Keep input schemas simple — complex nested schemas lead to more errors.
- Provide examples in the system prompt showing expected tool usage patterns.
- Track tool call history to prevent the model from repeating identical calls.
