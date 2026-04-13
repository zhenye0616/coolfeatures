"""Streamlit chat interface for LLM Wiki."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.backends import BACKENDS
from core.engine import WikiEngine, QUERY_TOOLS

st.set_page_config(page_title="LLM Wiki", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("LLM Wiki")
    wiki_dir = st.text_input("Wiki directory", value="my_wiki")
    backend_name = st.selectbox("Backend", ["openai", "anthropic"], index=0)
    model = st.text_input("Model", value="gpt-4.1-nano")

    st.divider()

    # Initialize engine (cached so it persists across reruns)
    @st.cache_resource
    def get_engine(_wiki_dir, _backend_name, _model):
        backend = BACKENDS[_backend_name](model=_model or None)
        return WikiEngine(_wiki_dir, backend=backend)

    try:
        engine = get_engine(wiki_dir, backend_name, model)
        pages = engine.manifest.list_pages()
        topics = engine.manifest.list_topics()

        st.metric("Pages", len(pages))
        st.metric("Topics", len(topics))

        if topics:
            st.caption("**Topics**")
            for topic, subs in topics.items():
                st.markdown(f"- **{topic}** ({len(subs)} subtopic{'s' if len(subs) != 1 else ''})")
    except Exception as e:
        st.error(f"Failed to load wiki: {e}")
        st.stop()

# ── Chat ─────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources consulted ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.code(src, language=None)

if prompt := st.chat_input("Ask your wiki..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching wiki..."):
            # Track which pages the LLM reads during the query
            pages_read = []
            original_execute = engine.execute_tool

            def tracking_execute(name, inputs):
                if name == "read_page":
                    pages_read.append(inputs["path"])
                return original_execute(name, inputs)

            schema = engine._read_schema()
            system = f"""\
You are a research assistant powered by a personal wiki. Answer questions using the wiki as your knowledge base.

## Wiki Schema
{schema}

## Instructions

1. Call `list_pages` to see what exists, then read relevant pages.
2. Synthesize a clear answer, citing wiki pages as sources.
3. If your answer is a valuable synthesis, save it as a new page in `analyses/`.
4. Do NOT write to index.md or log.md -- the system manages those automatically.
5. Call `done` with your full answer as the summary."""

            answer = engine.backend.run_tool_loop(
                system,
                f"Question: {prompt}",
                QUERY_TOOLS,
                tracking_execute,
            )
            engine.rebuild()

        st.markdown(answer)
        if pages_read:
            with st.expander(f"Sources consulted ({len(pages_read)})"):
                for src in pages_read:
                    st.code(src, language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": pages_read,
    })
