# coolfeatures

Collection of reusable small features. Pick and combine for new projects.

## Features

| Feature | Description | Launch |
|---------|-------------|--------|
| [docforge](docforge/) | Extract format from any .docx/.pdf, create reusable templates, fill with new data via LLM | `cd docforge && uv run streamlit run app.py` |
| [agentic_search](agentic_search/) | Agentic search with hybrid retrieval, self-editing context, and multi-hop reasoning — inspired by Chroma Context-1 | `cd agentic_search && uv run streamlit run app.py` |
| [llm_wiki](llm_wiki/) | Persistent, LLM-maintained knowledge base with incremental ingestion, wikilinks, and Obsidian integration — inspired by Karpathy's LLM Wiki pattern | `cd llm_wiki && uv run python cli.py --help` |

## Adding a new feature

1. Create a new directory: `my-feature/`
2. Add a `pyproject.toml` with `uv` deps
3. Add a one-liner to the table above
4. Each feature is self-contained — its own deps, venv, and entry point
