"""LLM backends for wiki tool loops.

Each backend wraps a specific LLM provider (Anthropic, OpenAI) and exposes
a unified ``run_tool_loop`` interface that the engine calls with an
arbitrary tools_spec, allowing different tool sets for ingest, lint, and
query operations.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable


def _retry_on_rate_limit(fn, max_retries=5):
    """Call *fn()*, retrying with backoff on 429 rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            error_str = str(e)
            if "429" not in error_str or attempt == max_retries:
                raise
            # Extract wait time from error message if available
            match = re.search(r"try again in (\d+\.?\d*)(ms|s)", error_str, re.IGNORECASE)
            if match:
                wait = float(match.group(1))
                if match.group(2) == "ms":
                    wait /= 1000
            else:
                wait = 2 ** attempt
            wait = max(wait, 1.0)  # at least 1s
            print(f"    Rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)


class AnthropicBackend:
    """Anthropic Claude API backend."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str | None = None):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model or self.DEFAULT_MODEL

    def run_tool_loop(
        self,
        system: str,
        user_message: str,
        tools_spec: list[dict],
        execute_tool: Callable[[str, dict], str],
        max_iterations: int = 30,
    ) -> str:
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools_spec
        ]
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            response = _retry_on_rate_limit(lambda: self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
            ))

            text_parts = []
            tool_calls = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            if not tool_calls:
                return "\n".join(text_parts) if text_parts else "Done."

            tool_results = []
            done_summary = None
            for tc in tool_calls:
                result = execute_tool(tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })
                if tc.name == "done":
                    done_summary = tc.input.get("summary", "Done.")

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if done_summary is not None:
                return done_summary

        return "Warning: reached maximum iterations."


class OpenAIBackend:
    """OpenAI API backend."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or self.DEFAULT_MODEL

    def run_tool_loop(
        self,
        system: str,
        user_message: str,
        tools_spec: list[dict],
        execute_tool: Callable[[str, dict], str],
        max_iterations: int = 30,
    ) -> str:
        tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools_spec
        ]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_iterations):
            response = _retry_on_rate_limit(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            ))

            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                return message.content or "Done."

            # Append the assistant message (with tool_calls) to history
            messages.append(message)

            done_summary = None
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                if tc.function.name == "done":
                    done_summary = args.get("summary", "Done.")

            if done_summary is not None:
                return done_summary

        return "Warning: reached maximum iterations."


BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
}
