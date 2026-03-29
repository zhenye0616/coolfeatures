"""SearchAgent — the complete agent harness that orchestrates all primitives.

Architecture (mirrors Chroma docs + Context-1 harness):

  BaseAgent
    ├── QueryPlanner  — decomposes query into PlanSteps
    ├── Executor      — tool-calling loop per step (search_corpus, finish_step)
    ├── Evaluator     — decides continue / sufficient / add_steps
    └── ContextPruner — self-editing context at soft limit

  SearchAgent(BaseAgent)
    └── wires in HybridSearcher, Reranker, DocumentStore

The main loop:
  1. Plan  — decompose user query into sub-queries
  2. For each batch of ready steps:
     a. Execute step (Executor runs tool-calling loop)
     b. Check context budget → prune if over soft limit
     c. Evaluate progress → decide next action
  3. When sufficient or plan exhausted → return context to generation tier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    AgentResult,
    ContextEntry,
    Document,
    LLMConfig,
    SearchConfig,
    StepOutcome,
    StepStatus,
)
from .evaluator import Evaluator
from .executor import Executor
from .planner import QueryPlanner
from .prune import ContextPruner
from .rerank import Reranker
from .search import HybridSearcher
from .storage import DocumentStore


@dataclass
class AgentTrace:
    """Records the agent's execution trace for observability."""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def log(self, event: str, **kwargs: Any) -> None:
        self.steps.append({"event": event, **kwargs})


class BaseAgent:
    """Base orchestrator — plan → execute → evaluate loop."""

    def __init__(
        self,
        planner: QueryPlanner,
        executor: Executor,
        evaluator: Evaluator,
        pruner: ContextPruner,
        search_config: SearchConfig,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._evaluator = evaluator
        self._pruner = pruner
        self._config = search_config

    def run(self, query: str) -> tuple[list[ContextEntry], AgentTrace]:
        """Execute the full agentic search loop.

        Returns (curated_context, trace).
        """
        trace = AgentTrace()

        # ------ Phase 1: Plan ------
        plan_steps = self._planner.plan(query)
        trace.log("plan", steps=[{"id": s.step_id, "query": s.query} for s in plan_steps])

        context: list[ContextEntry] = []
        seen_ids: set[str] = set()
        outcomes: list[StepOutcome] = []
        total_steps = 0

        # ------ Phase 2: Execute + Evaluate loop ------
        for batch in self._planner:
            if total_steps >= self._config.max_agent_steps:
                trace.log("max_steps_reached", total=total_steps)
                break

            for step in batch:
                if total_steps >= self._config.max_agent_steps:
                    break

                self._planner.update_status(step.step_id, StepStatus.IN_PROGRESS)
                trace.log("execute_start", step_id=step.step_id, query=step.query)

                outcome, new_entries, seen_ids = self._executor.execute(
                    step, seen_ids, context
                )
                context.extend(new_entries)
                outcomes.append(outcome)
                self._planner.update_status(step.step_id, outcome.status)
                total_steps += 1

                trace.log(
                    "execute_done",
                    step_id=step.step_id,
                    new_entries=len(new_entries),
                    context_tokens=self._pruner.context_token_count(context),
                )

                # ------ Self-editing context: prune at soft limit ------
                if self._pruner.should_prune(context):
                    before = len(context)
                    context = self._pruner.prune(context, query)
                    trace.log(
                        "prune",
                        before=before,
                        after=len(context),
                        tokens_after=self._pruner.context_token_count(context),
                    )

            # ------ Phase 3: Evaluate ------
            decision, new_steps = self._evaluator.evaluate(
                query, self._planner.steps, outcomes
            )
            trace.log("evaluate", decision=decision, new_steps=len(new_steps))

            if decision == "sufficient":
                break
            elif decision == "add_steps":
                for s in new_steps:
                    self._planner.steps.append(s)

        return context, trace


class SearchAgent(BaseAgent):
    """Concrete agent wired to a DocumentStore with hybrid search + reranking.

    This is the Tier-1 retrieval agent in the three-tier architecture:
      Tier 1: SearchAgent (retrieval)  ← this
      Tier 2: AnswerGenerator (generation)
      Tier 3: DocumentStore (storage)
    """

    def __init__(
        self,
        store: DocumentStore,
        llm_config: LLMConfig,
        search_config: SearchConfig | None = None,
    ) -> None:
        cfg = search_config or SearchConfig()
        searcher = HybridSearcher(store, cfg)
        reranker = Reranker(cfg, llm_config)
        planner = QueryPlanner(llm_config)
        executor = Executor(searcher, reranker, llm_config, cfg)
        evaluator = Evaluator(llm_config)
        pruner = ContextPruner(llm_config, cfg)
        super().__init__(planner, executor, evaluator, pruner, cfg)
        self._store = store
        self._llm_config = llm_config

    def search(self, query: str) -> AgentResult:
        """Run the full agentic search and return structured results."""
        context, trace = self.run(query)

        # Collect unique source documents
        seen: set[str] = set()
        sources: list[Document] = []
        for entry in context:
            if entry.doc_id not in seen:
                doc = self._store.get_by_id(entry.doc_id)
                if doc:
                    sources.append(doc)
                    seen.add(entry.doc_id)

        return AgentResult(
            answer="",  # Tier 2 (AnswerGenerator) fills this in
            sources=sources,
            steps_taken=len([s for s in trace.steps if s["event"] == "execute_done"]),
            context_snapshot=context,
        )
