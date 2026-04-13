# LLM Wiki

A persistent, LLM-maintained knowledge base. Instead of RAG (re-deriving knowledge from scratch on every query), the LLM incrementally builds and maintains a structured wiki of interlinked markdown files that compounds over time.

Inspired by [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Setup

```bash
uv sync
cp .env.example .env  # add your API keys
```

## Usage

```bash
# Initialize a new wiki
uv run python cli.py --dir my_wiki init

# Ingest a source document
uv run python cli.py --dir my_wiki ingest path/to/article.md

# Query the wiki
uv run python cli.py --dir my_wiki query "What are the key themes?"

# Audit wiki health (fix broken links, missing cross-references, etc.)
uv run python cli.py --dir my_wiki lint
```

### Backend Selection

Supports both Anthropic and OpenAI. Defaults to Anthropic.

```bash
# Anthropic (default, uses claude-sonnet)
uv run python cli.py --dir my_wiki ingest article.md

# OpenAI (uses gpt-4o)
uv run python cli.py --dir my_wiki --backend openai ingest article.md

# Custom model
uv run python cli.py --dir my_wiki --backend openai --model gpt-4o-mini query "summarize everything"
```

Set API keys in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## How It Works

### Three Layers

- **`raw/`** — Immutable archive of source documents. The LLM reads from these but never modifies them.
- **`wiki/`** — LLM-generated markdown pages. Summaries, entity pages, concept pages, analyses — all interlinked with `[[wikilinks]]`. The LLM owns this layer entirely.
- **`schema.md`** — Conventions the LLM follows: page structure, cross-referencing rules, log format. Edit this to customize behavior.

### Four Commands

| Command | What happens |
|---------|-------------|
| `init` | Creates the directory structure and default schema |
| `ingest` | LLM reads the source, creates/updates entity + concept + source pages, maintains cross-references, updates index and log |
| `query` | LLM searches the wiki via index, reads relevant pages, synthesizes an answer — saves valuable analyses back as new pages |
| `lint` | LLM reads every page, fixes contradictions, broken links, missing cross-references |

### Compounding Knowledge

The key insight: query results get saved back into the wiki as analysis pages. Every interaction makes the wiki richer. The cross-references are already there. The contradictions have already been flagged. Nothing is re-derived from scratch.

### No Embeddings Needed

At modest scale (~100 sources), `index.md` acts as the search layer — the LLM reads it to find relevant pages. No vector DB, no chunking, no embedding infrastructure. You only need proper search as the wiki grows large.

## Obsidian Integration

The wiki is valid Obsidian markdown with `[[wikilinks]]`. Point Obsidian at the `wiki/` directory to browse pages, follow links, and view the graph.
