# CLAUDE.md

## Repository Structure

This is a monorepo of independent Python projects. Each subdirectory is its own self-contained project with its own `pyproject.toml` and virtual environment.

- `agentic_search/` — Agentic search system with hybrid retrieval and multi-hop reasoning
- `docforge/` — Extract format from .docx files, create reusable templates, fill with new data via LLM
- `llm_wiki/` — Persistent, LLM-maintained knowledge base (Karpathy's LLM Wiki pattern)

## Package Management

**Always use `uv` for all package management.** Never use `pip`, `pip install`, or `pip3 install` directly.

```bash
# Install dependencies for a subproject
cd <subdir> && uv sync

# Add a dependency
cd <subdir> && uv add <package>

# Run a script
cd <subdir> && uv run python <script.py>

# Run a module
cd <subdir> && uv run python -m <module>
```

Each subproject must have:
- Its own `pyproject.toml` with dependencies declared
- Its own `.venv` managed by `uv` (created automatically by `uv sync`)
- No shared virtual environments between subprojects
