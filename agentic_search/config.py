"""Configuration and shared data models for the agentic search system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# JSON extraction helper — used by planner, reranker, evaluator, pruner
# ---------------------------------------------------------------------------

def extract_json(text: str, kind: str = "object") -> Any | None:
    """Robustly extract a JSON object or array from LLM free-text output.

    Uses bracket-counting to find the matching closing delimiter, which
    handles nested structures correctly (unlike greedy/non-greedy regex).

    Args:
        text: Raw LLM output that may contain surrounding prose.
        kind: "object" to find the first ``{...}`` or "array" for ``[...]``.

    Returns:
        The parsed Python object, or ``None`` if no valid JSON was found.
    """
    open_ch, close_ch = ("{", "}") if kind == "object" else ("[", "]")
    start = text.find(open_ch)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    provider: str = "anthropic"  # "anthropic" or "openrouter"


# ---------------------------------------------------------------------------
# Search / retrieval configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    collection_name: str = "default"
    # RRF parameters
    rrf_k: int = 60
    # How many raw candidates each retrieval method returns
    top_k_retrieval: int = 50
    # How many results survive reranking
    top_k_rerank: int = 10
    # Token budget for the agent's accumulated context window (matches Context-1: 32k)
    context_token_budget: int = 32_000
    # Soft limit — kept for external pruner (local corpus); wiki executor uses budget/2
    context_soft_limit: int = 16_000
    # Maximum agent loop iterations
    max_agent_steps: int = 12
    # Per-search-call token budget for returned chunks
    per_call_token_budget: int = 4_000
    # Chunk settings for ingestion
    chunk_size: int = 500
    chunk_overlap: int = 50


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.text) // 4


@dataclass
class ScoredDocument:
    document: Document
    score: float
    source: str  # "dense" | "sparse" | "fused"


# ---------------------------------------------------------------------------
# Context entry — a document chunk held in the agent's working context
# ---------------------------------------------------------------------------

@dataclass
class ContextEntry:
    entry_id: str
    doc_id: str
    text: str
    relevance_score: float
    step_added: int

    @property
    def token_estimate(self) -> int:
        return len(self.text) // 4


# ---------------------------------------------------------------------------
# Plan models (QueryPlanner / Executor / Evaluator)
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    step_id: int
    query: str
    rationale: str
    depends_on: list[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING


@dataclass
class StepOutcome:
    step_id: int
    status: StepStatus
    retrieved_docs: list[ScoredDocument] = field(default_factory=list)
    candidate_answer: str = ""
    evidence: str = ""


@dataclass
class AgentResult:
    answer: str
    sources: list[Document]
    steps_taken: int
    context_snapshot: list[ContextEntry]
    trace: list[dict[str, Any]] = field(default_factory=list)
