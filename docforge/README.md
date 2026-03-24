# DocForge

Extract format from any .docx or .pdf, create reusable templates, fill with new data via LLM structured output.

## Quick start

```bash
cd docforge
uv sync
uv run streamlit run app.py
```

Set your API key in the UI sidebar, or create a `.env` file:
```
OPENROUTER_API_KEY=your-key-here
```

## How it works

1. Upload any formatted document (.docx or .pdf)
2. LLM analyzes and identifies all variable fields (names, dates, addresses, amounts...)
3. A reusable template is created — all original formatting preserved
4. Fill the template with new data to generate new documents

## Files

- `docx_template_pipeline.py` — core module (importable API)
- `app.py` — Streamlit UI
- `templates/` — 16 pre-built industry templates
