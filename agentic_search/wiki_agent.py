"""WikiSearchAgent — agent harness wired to Wikipedia for FRAMES evaluation.

Subclass of BaseAgent that replaces the local DocumentStore/HybridSearcher
with live Wikipedia API tools. Reuses the existing QueryPlanner, Evaluator,
and ContextPruner unchanged.
"""

from __future__ import annotations

from config import (
    AgentResult,
    ContextEntry,
    Document,
    LLMConfig,
    SearchConfig,
)
from agent import BaseAgent, AgentTrace
from evaluator import Evaluator
from planner import QueryPlanner
from prune import ContextPruner
from rerank import Reranker
from wiki_executor import WikiExecutor


class WikiSearchAgent(BaseAgent):
    """Concrete agent wired to Wikipedia via live API calls.

    Used for FRAMES benchmark evaluation. Same plan → execute → evaluate → prune
    loop as SearchAgent, but the executor calls Wikipedia instead of a local store.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        search_config: SearchConfig | None = None,
    ) -> None:
        cfg = search_config or SearchConfig()
        reranker = Reranker(cfg, llm_config)
        planner = QueryPlanner(llm_config)
        executor = WikiExecutor(reranker, llm_config, cfg)
        evaluator = Evaluator(llm_config)
        pruner = ContextPruner(llm_config, cfg)
        super().__init__(planner, executor, evaluator, pruner, cfg)
        self._llm_config = llm_config

    def search(self, query: str) -> AgentResult:
        """Run the full agentic search over Wikipedia and return results."""
        context, trace = self.run(query)

        # Collect unique source wiki URLs from context entries
        seen_urls: set[str] = set()
        sources: list[Document] = []
        for entry in context:
            url = entry.doc_id.split("#")[0]  # "wiki:Title" part
            if url not in seen_urls:
                sources.append(
                    Document(
                        doc_id=entry.doc_id,
                        text=entry.text,
                        metadata={"source": url},
                    )
                )
                seen_urls.add(url)

        return AgentResult(
            answer="",
            sources=sources,
            steps_taken=len([s for s in trace.steps if s["event"] == "execute_done"]),
            context_snapshot=context,
        )
