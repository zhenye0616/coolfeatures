# coolfeatures

Collection of reusable small features. Pick and combine for new projects.

## Features

| Feature | Description | Launch |
|---------|-------------|--------|
| [docforge](docforge/) | Extract format from any .docx/.pdf, create reusable templates, fill with new data via LLM | `cd docforge && uv run streamlit run app.py` |

## Adding a new feature

1. Create a new directory: `my-feature/`
2. Add a `pyproject.toml` with `uv` deps
3. Add a one-liner to the table above
4. Each feature is self-contained — its own deps, venv, and entry point
