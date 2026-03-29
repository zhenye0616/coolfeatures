"""Configuration and shared data models for the agentic search system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096


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
    # Token budget for the agent's accumulated context window
    context_token_budget: int = 24_000
    # Soft limit — when exceeded, the agent should prune before searching more
    context_soft_limit: int = 18_000
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
