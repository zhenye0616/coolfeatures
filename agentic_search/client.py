"""LLM client factory — supports Anthropic API and OpenRouter.

OpenRouter uses the OpenAI chat completions format, so this module provides
a thin adapter that presents the same interface as anthropic.Anthropic().messages
while translating requests/responses under the hood.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx

from config import LLMConfig

_OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
_TIMEOUT = 300.0


def create_client(llm_config: LLMConfig):
    """Create an LLM client based on the provider in llm_config."""
    if llm_config.provider == "openrouter":
        return OpenRouterClient(llm_config)
    return anthropic.Anthropic(api_key=llm_config.api_key)


# ---------------------------------------------------------------------------
# Shims that match the Anthropic SDK response shape
# ---------------------------------------------------------------------------

@dataclass
class _TextBlock:
    type: str
    text: str


@dataclass
class _ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict


@dataclass
class _MessagesResponse:
    content: list[_TextBlock | _ToolUseBlock]
    model: str
    stop_reason: str | None


# ---------------------------------------------------------------------------
# Message / tool format translation
# ---------------------------------------------------------------------------

def _translate_messages(messages: list[dict], system: str | None) -> list[dict]:
    """Convert Anthropic message format to OpenAI chat format."""
    oai: list[dict] = []

    if system:
        oai.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                oai.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # tool_result blocks → separate "tool" role messages
                tool_results = [
                    b for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        oai.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": tr.get("content", ""),
                        })
                else:
                    # Plain text blocks
                    parts = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b["text"])
                        elif isinstance(b, str):
                            parts.append(b)
                    oai.append({"role": "user", "content": "\n".join(parts)})

        elif role == "assistant":
            if isinstance(content, str):
                oai.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict] = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            text_parts.append(b["text"])
                        elif b.get("type") == "tool_use":
                            tool_calls.append({
                                "id": b["id"],
                                "type": "function",
                                "function": {
                                    "name": b["name"],
                                    "arguments": json.dumps(b["input"]),
                                },
                            })
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                oai.append(entry)

    return oai


def _translate_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def _translate_response(data: dict) -> _MessagesResponse:
    """Convert OpenAI chat completion JSON to Anthropic-like response."""
    choice = data["choices"][0]
    message = choice["message"]

    blocks: list[_TextBlock | _ToolUseBlock] = []

    if message.get("content"):
        blocks.append(_TextBlock(type="text", text=message["content"]))

    for tc in message.get("tool_calls") or []:
        func = tc["function"]
        try:
            input_data = json.loads(func["arguments"])
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        blocks.append(_ToolUseBlock(
            type="tool_use",
            id=tc["id"],
            name=func["name"],
            input=input_data,
        ))

    if not blocks:
        blocks.append(_TextBlock(type="text", text=""))

    stop = "end_turn"
    fr = choice.get("finish_reason", "")
    if fr == "tool_calls":
        stop = "tool_use"
    elif fr == "length":
        stop = "max_tokens"

    return _MessagesResponse(
        content=blocks,
        model=data.get("model", ""),
        stop_reason=stop,
    )


# ---------------------------------------------------------------------------
# OpenRouter client (Anthropic-compatible facade over OpenAI format)
# ---------------------------------------------------------------------------

class _MessagesAPI:
    """Drop-in replacement for ``anthropic.Anthropic().messages``."""

    def __init__(self, client: OpenRouterClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> _MessagesResponse:
        oai_messages = _translate_messages(messages, system)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }
        if tools:
            body["tools"] = _translate_tools(tools)

        resp = self._client._http.post(
            _OPENROUTER_BASE,
            headers={
                "Authorization": f"Bearer {self._client._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return _translate_response(resp.json())


class OpenRouterClient:
    """Wraps httpx pointed at OpenRouter, presenting an Anthropic-like interface."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._api_key = llm_config.api_key
        self._http = httpx.Client()
        self.messages = _MessagesAPI(self)
