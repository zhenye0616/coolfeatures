# LLM Wiki

A persistent, LLM-maintained knowledge base. Instead of RAG (re-deriving knowledge from scratch on every query), the LLM incrementally builds and maintains a structured wiki of interlinked markdown files that compounds over time.

Inspired by [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Setup

```bash
uv sync
cp .env.example .env  # add your API keys
```

Set API keys in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Initialize a new wiki
uv run python cli.py --dir my_wiki init

# Ingest a source document or directory
uv run python cli.py --dir my_wiki ingest path/to/article.pdf
uv run python cli.py --dir my_wiki ingest path/to/papers/

# Parallel ingestion (recommended for directories)
uv run python cli.py --dir my_wiki --workers 4 ingest path/to/papers/

# Query the wiki
uv run python cli.py --dir my_wiki query "What are the key themes?"

# Incremental lint (only changed pages)
uv run python cli.py --dir my_wiki lint

# Full lint (all pages)
uv run python cli.py --dir my_wiki lint --full

# Rebuild index.md and log.md from manifest (no LLM calls)
uv run python cli.py --dir my_wiki rebuild

# Migrate an existing wiki into the manifest (one-time)
uv run python cli.py --dir my_wiki migrate
```

### Backend Selection

Supports both Anthropic and OpenAI. Defaults to Anthropic.

```bash
# Anthropic (default, uses claude-sonnet)
uv run python cli.py --dir my_wiki ingest article.md

# OpenAI (uses gpt-4o)
uv run python cli.py --dir my_wiki --backend openai ingest article.md

# Custom model (gpt-4.1-nano: fastest/cheapest, gpt-4.1: best quality with 1M context)
uv run python cli.py --dir my_wiki --backend openai --model gpt-4.1 ingest large-textbook.pdf
```

### Obsidian Topic Colors

Auto-generates color-coded topics for Obsidian's graph view and note styling.

```bash
# Generate/refresh the Obsidian CSS snippet and graph colors
uv run python cli.py --dir my_wiki theme
# Then in Obsidian: Settings → Appearance → CSS snippets → toggle "topic-colors"
```

The LLM assigns `topic`, `subtopic`, `cssclasses`, and `tags` in each page's YAML frontmatter during ingestion. The `theme` command reads all topics from the manifest, assigns each a unique HSL hue (with lightness variations per subtopic), and writes:
- `wiki/.obsidian/snippets/topic-colors.css` — colored left borders and background gradients per note
- `wiki/.obsidian/graph.json` — color groups for the graph view

### Streamlit Chat

```bash
uv run streamlit run chat.py
```

Browser-based chat interface for querying the wiki. Shows which pages the LLM consulted for each answer.

### Factory Reset

```bash
uv run python reset.py --dir my_wiki
```

Wipes all pages, manifest data, and raw archives. Preserves `schema.md` and `.obsidian/` settings.

## How It Works

### Architecture

```
llm_wiki/                # Project
  core/                  # Core library
    backends.py          # LLM providers (Anthropic, OpenAI)
    engine.py            # Wiki engine, tool loop, ingest/query/lint
    manifest.py          # SQLite manifest
    theme.py             # Obsidian CSS + graph color generation
  cli.py                 # CLI entry point
  chat.py                # Streamlit chat interface
  reset.py               # Factory reset utility
  tests/

my_wiki/                 # Wiki data (created by init)
  raw/                   # Immutable archive of source documents
  wiki/                  # LLM-generated markdown pages with [[wikilinks]]
    sources/             # One summary per ingested document
    entities/            # People, orgs, products, etc.
    concepts/            # Ideas, themes, technologies
    analyses/            # Query results and syntheses
    index.md             # System-generated master index
    log.md               # System-generated operation log
  schema.md              # Conventions the LLM follows (edit to customize)
  manifest.db            # SQLite manifest tracking pages, links, ingestion status
```

### Commands

| Command | What happens |
|---------|-------------|
| `init` | Creates the directory structure, default schema, and manifest |
| `ingest` | **Phase 1:** LLM creates source/entity/concept pages in parallel. **Phase 2:** System rebuilds index and log from manifest. |
| `query` | LLM searches the wiki via manifest, reads relevant pages, synthesizes an answer — saves valuable analyses back as new pages |
| `lint` | Per-page LLM audit — reads each changed page + its neighbors, fixes issues. Parallelizable with `--workers`. |
| `rebuild` | Regenerates `index.md` and `log.md` from the manifest. No LLM calls. |
| `migrate` | One-time scan of existing wiki pages into the manifest. |
| `theme` | Generates Obsidian CSS snippet and graph color groups from manifest topics. |

### Two-Phase Ingest

Ingestion is split into two phases for scalability:

1. **Parallel LLM phase** — Each worker processes a source document independently, creating wiki pages. Workers never touch `index.md` or `log.md`, so there are no conflicts.
2. **Consolidation phase** — The system deterministically rebuilds `index.md` and `log.md` from the manifest. No LLM calls needed.

Resume support: if ingestion is interrupted, re-run the same command. Already-ingested files are skipped, interrupted files are retried.

### Incremental Lint

Instead of reading every page in one massive LLM call, lint processes pages individually:

- Only pages changed since the last lint are processed (use `--full` to lint everything)
- Each page is checked alongside its linked neighbors for broken links, missing cross-references, and contradictions
- Parallelizable with `--workers` — each LLM call holds only a few pages in context

### SQLite Manifest

A `manifest.db` file tracks page metadata (title, type, summary), wikilink relationships, and ingestion status. This eliminates filesystem scanning, prevents index clobbers during parallel ingestion, and enables incremental lint. The manifest can always be rebuilt from the wiki files using `rebuild`.

### Compounding Knowledge

Query results get saved back into the wiki as analysis pages. Every interaction makes the wiki richer. The cross-references are already there. The contradictions have already been flagged. Nothing is re-derived from scratch.

## Obsidian Integration

The wiki is valid Obsidian markdown with `[[wikilinks]]`. Point Obsidian at the `wiki/` directory to browse pages, follow links, and view the color-coded graph.

Each page's frontmatter includes `topic`, `subtopic`, `cssclasses`, and `tags` — assigned automatically by the LLM during ingestion. Run `theme` to generate the CSS snippet and graph color groups.

## Known Gaps

See [KNOWN_GAPS.md](KNOWN_GAPS.md) for a detailed analysis of architectural limitations and improvement directions covering ingestion, retrieval, knowledge representation, query reasoning, maintenance, and scaling.

## Development

```bash
# Run tests
uv run python -m pytest tests/ -v
```
