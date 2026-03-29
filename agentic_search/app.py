"""Streamlit demo app for the agentic search system.

Provides a UI to:
  - Configure API key and model
  - Ingest the demo corpus into ChromaDB + BM25
  - Submit queries and watch the agent harness execute
  - View the agent trace, retrieved context, and generated answer
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from .agent import SearchAgent
from .config import AgentResult, LLMConfig, SearchConfig
from .generate import AnswerGenerator
from .ingest import load_corpus
from .storage import DocumentStore

DEMO_CORPUS_DIR = str(Path(__file__).parent / "demo_corpus")


def _init_store(config: SearchConfig) -> DocumentStore:
    """Initialize DocumentStore and load demo corpus (cached in session)."""
    if "store" not in st.session_state:
        store = DocumentStore(config)
        n = load_corpus(DEMO_CORPUS_DIR, store, config)
        st.session_state["store"] = store
        st.session_state["corpus_chunks"] = n
    return st.session_state["store"]


def main() -> None:
    st.set_page_config(page_title="Agentic Search", page_icon="🔍", layout="wide")
    st.title("Agentic Search System")
    st.caption(
        "Hybrid retrieval · RRF fusion · Self-editing context · "
        "Multi-hop reasoning — inspired by Chroma Context-1"
    )

    # ---- Sidebar: configuration ----
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
        )
        model = st.selectbox(
            "Model",
            ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
            index=0,
        )

        st.divider()
        st.subheader("Search Config")
        top_k_rerank = st.slider("Top-K after rerank", 3, 20, 10)
        context_budget = st.slider("Context token budget", 8_000, 48_000, 24_000, step=2_000)
        soft_limit = st.slider("Soft limit (triggers prune)", 4_000, 40_000, 18_000, step=2_000)
        max_steps = st.slider("Max agent steps", 2, 20, 12)

        st.divider()
        st.subheader("Corpus")
        search_config = SearchConfig(
            top_k_rerank=top_k_rerank,
            context_token_budget=context_budget,
            context_soft_limit=soft_limit,
            max_agent_steps=max_steps,
        )
        store = _init_store(search_config)
        st.success(f"Corpus loaded: {st.session_state.get('corpus_chunks', 0)} chunks")

    # ---- Main area ----
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar.")
        return

    llm_config = LLMConfig(api_key=api_key, model=model)
    query = st.text_area("Enter your query", height=80, placeholder="e.g. How does self-editing context in agentic search combat context rot?")

    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("Search", type="primary", use_container_width=True)

    if run_btn and query.strip():
        agent = SearchAgent(store, llm_config, search_config)
        generator = AnswerGenerator(llm_config)

        # ---- Phase 1: Agentic retrieval ----
        with st.status("Running agentic search...", expanded=True) as status:
            st.write("**Planning** — decomposing query into sub-queries...")
            agent_result: AgentResult = agent.search(query)
            st.write(f"**Done** — {agent_result.steps_taken} steps, {len(agent_result.context_snapshot)} context entries")
            status.update(label="Retrieval complete", state="complete")

        # ---- Phase 2: Answer generation ----
        with st.status("Generating answer...", expanded=True) as status:
            generated = generator.generate(query, agent_result)
            status.update(label="Answer ready", state="complete")

        # ---- Display results ----
        st.markdown("---")
        st.subheader("Answer")
        st.markdown(generated.answer)

        st.markdown("---")
        st.subheader("Retrieved Sources")
        for i, doc in enumerate(agent_result.sources, 1):
            with st.expander(f"Source {i}: {doc.doc_id}"):
                st.text(doc.text[:1000])

        st.markdown("---")
        st.subheader("Agent Context Snapshot")
        for entry in agent_result.context_snapshot:
            with st.expander(
                f"[Step {entry.step_added}] {entry.doc_id} "
                f"(relevance={entry.relevance_score:.2f}, ~{entry.token_estimate} tokens)"
            ):
                st.text(entry.text[:600])


if __name__ == "__main__":
    main()
