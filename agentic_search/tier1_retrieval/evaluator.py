"""Evaluator — decides how to proceed after each batch of step executions.

The Evaluator considers the plan and the history of outcomes to decide:
  - Are we done?  → produce final answer
  - Do we need more steps?  → add new PlanSteps
  - Should we re-plan?  → override the plan entirely
"""

from __future__ import annotations

import anthropic

from config import LLMConfig, PlanStep, StepOutcome, StepStatus, extract_json

_EVALUATOR_SYSTEM = """\
You are an evaluation agent.  You are given:
1. The original user query.
2. A query plan with step statuses.
3. The outcomes (evidence, candidate answers) from completed steps.

Decide ONE of:
- "sufficient" — we have enough evidence to answer the query.
- "continue" — let the remaining plan steps execute as-is.
- "add_steps" — we need additional sub-queries (provide them).

Return ONLY a JSON object:
{
  "decision": "sufficient" | "continue" | "add_steps",
  "reasoning": "...",
  "new_steps": [...]   // only if decision == "add_steps"; same format as planner output
}
"""


class Evaluator:
    """Evaluates plan progress and decides the next action."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm_config = llm_config
        self._client = anthropic.Anthropic(api_key=llm_config.api_key)

    def evaluate(
        self,
        query: str,
        steps: list[PlanStep],
        outcomes: list[StepOutcome],
    ) -> tuple[str, list[PlanStep]]:
        """Return (decision, new_steps).

        decision is one of "sufficient", "continue", "add_steps".
        new_steps is non-empty only when decision == "add_steps".
        """
        plan_desc = "\n".join(
            f"  Step {s.step_id} [{s.status.value}]: {s.query}"
            for s in steps
        )
        outcomes_desc = "\n".join(
            f"  Step {o.step_id}: answer={o.candidate_answer!r}, evidence={o.evidence[:200]!r}"
            for o in outcomes
        )

        prompt = (
            f"User query: {query}\n\n"
            f"Plan:\n{plan_desc}\n\n"
            f"Outcomes:\n{outcomes_desc}"
        )

        response = self._client.messages.create(
            model=self._llm_config.model,
            max_tokens=self._llm_config.max_tokens,
            system=_EVALUATOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text

        try:
            parsed = extract_json(content, kind="object")
            if not parsed:
                return ("continue", [])
            decision = parsed.get("decision", "continue")

            new_steps: list[PlanStep] = []
            if decision == "add_steps" and parsed.get("new_steps"):
                max_id = max(s.step_id for s in steps) if steps else -1
                for i, s in enumerate(parsed["new_steps"]):
                    new_steps.append(
                        PlanStep(
                            step_id=max_id + 1 + i,
                            query=s.get("query", ""),
                            rationale=s.get("rationale", ""),
                            depends_on=s.get("depends_on", []),
                        )
                    )
            return (decision, new_steps)

        except (KeyError, TypeError):
            return ("continue", [])
