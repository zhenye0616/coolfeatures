"""QueryPlanner — decomposes a complex query into a plan of PlanSteps.

The planner asks the LLM to break a user query into targeted sub-queries,
each with a rationale and dependency information.  It acts as an iterator
that yields the next batch of steps whose dependencies are satisfied.
"""

from __future__ import annotations

from typing import Iterator

from client import create_client
from config import LLMConfig, PlanStep, StepStatus, extract_json

_PLANNER_SYSTEM = """\
You are a search query planner.  Given a user question, decompose it into a
sequence of focused sub-queries that an information retrieval system should
execute to gather all the evidence needed to answer the question.

Rules:
- Each sub-query should target a single, specific piece of information.
- If one sub-query depends on the result of another, specify the dependency.
- Return between 1 and 5 sub-queries.
- Return ONLY a JSON array. Each element must have:
  "step_id" (int, starting at 0), "query" (str), "rationale" (str),
  "depends_on" (list[int] — IDs of steps that must complete first, empty if independent).

Example:
[
  {"step_id": 0, "query": "What is RLHF?", "rationale": "Define the core concept", "depends_on": []},
  {"step_id": 1, "query": "RLHF training procedure steps", "rationale": "Get procedural detail", "depends_on": [0]}
]
"""


class QueryPlanner:
    """Generates and manages a plan of PlanSteps for a user query."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm_config = llm_config
        self._client = create_client(llm_config)
        self.steps: list[PlanStep] = []

    def plan(self, query: str) -> list[PlanStep]:
        """Ask the LLM to decompose *query* into PlanSteps."""
        response = self._client.messages.create(
            model=self._llm_config.model,
            max_tokens=self._llm_config.max_tokens,
            system=_PLANNER_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        content = response.content[0].text
        raw = extract_json(content, kind="array")
        if not raw:
            # Fallback: single-step plan with the original query
            self.steps = [PlanStep(step_id=0, query=query, rationale="Direct search")]
            return self.steps
        self.steps = [
            PlanStep(
                step_id=s["step_id"],
                query=s["query"],
                rationale=s["rationale"],
                depends_on=s.get("depends_on", []),
            )
            for s in raw
        ]
        return self.steps

    def ready_steps(self) -> list[PlanStep]:
        """Return pending steps whose dependencies have all succeeded."""
        completed = {s.step_id for s in self.steps if s.status == StepStatus.SUCCESS}
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed for dep in s.depends_on)
        ]

    def update_status(self, step_id: int, status: StepStatus) -> None:
        for s in self.steps:
            if s.step_id == step_id:
                s.status = status
                return

    def override_plan(self, new_steps: list[PlanStep]) -> None:
        """Replace the current plan (used by the Evaluator to re-plan)."""
        self.steps = new_steps

    @property
    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.SUCCESS, StepStatus.FAILURE, StepStatus.CANCELLED)
            for s in self.steps
        )

    def __iter__(self) -> Iterator[list[PlanStep]]:
        """Yield batches of ready steps until the plan is done."""
        while not self.is_complete:
            batch = self.ready_steps()
            if not batch:
                break
            yield batch
