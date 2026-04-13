# Scalable Wiki Engine Design

## Problem

The current wiki engine has three scaling bottlenecks:

1. **Index clobbers during parallel ingest** — multiple workers read `index.md`, add their entries, and write it back. Later writes overwrite earlier ones.
2. **Ingest cost grows with wiki size** — each worker reads the full index and related pages to cross-reference, so the LLM context (and token cost) increases with every page added.
3. **Lint reads the entire wiki** — a single LLM call reads every page, which blows past context limits and max iterations at ~30+ pages.

## Target Scale

Thousands of documents, lab/team-scale, multiple topics. Optimize for both token cost and wiki quality.

## Design

### 1. SQLite Manifest

A `manifest.db` file in the wiki base directory with three tables:

```sql
pages (
    path        TEXT PRIMARY KEY,   -- relative to wiki/, e.g. "entities/CLIP.md"
    title       TEXT NOT NULL,
    type        TEXT NOT NULL,      -- source | entity | concept | analysis
    summary     TEXT,               -- one-line description
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    last_linted TEXT                 -- NULL if never linted
)

links (
    from_path   TEXT NOT NULL,
    to_path     TEXT NOT NULL,
    PRIMARY KEY (from_path, to_path)
)

ingestions (
    filename    TEXT PRIMARY KEY,
    status      TEXT NOT NULL,      -- ingesting | done | failed
    started_at  TEXT NOT NULL,
    finished_at TEXT
)
```

**Key behaviors:**
- Updated by `_execute_tool` on every `write_page` call — parses `[[wikilinks]]` from content and upserts into `pages` and `links`.
- `list_pages` queries the manifest instead of scanning the filesystem.
- `index.md` and `log.md` become generated outputs rebuilt deterministically from the manifest. The LLM never writes to them directly.
- Replaces `.status.json` for ingestion tracking.
- SQLite handles concurrent writes natively (WAL mode), so no application-level file lock needed for manifest access.

### 2. Two-Phase Ingest

**Phase 1: Parallel content extraction** (LLM, parallelizable with `--workers`)

Each worker:
- Reads the source document.
- Gets a slim context from the manifest: list of existing entity/concept page titles (names only, not content) via a new `list_existing_entities` tool. This lets the LLM know what exists without reading those pages.
- Creates source/entity/concept pages via `write_page` tool calls.
- Does NOT touch `index.md` or `log.md`.
- Manifest is updated on each `write_page` call.

**Phase 2: Consolidation** (no LLM, deterministic)

After all workers finish:
- Query manifest for all pages, rebuild `index.md` grouped by type (Sources, Entities, Concepts, Analyses).
- Append ingest entries to `log.md` from the `ingestions` table.
- Print summary: N new pages, N updated, N failed.

**Benefits:**
- No index clobbers — `index.md` is never written by parallel workers.
- Cost per ingest stays flat — workers only get a page title list, not full page contents.
- Resume works via `ingestions` table status.

### 3. Incremental Lint

Instead of reading every page in one LLM call:

**Per-page lint** (LLM, parallelizable with `--workers`)

- Query manifest for pages where `last_linted IS NULL OR last_linted < updated_at` (only changed pages).
- For each page, read it + read its linked neighbors (from the `links` table).
- LLM checks: broken wikilinks, missing cross-references, stale content, contradictions with neighbors.
- LLM fixes the page if needed, manifest updates `last_linted`.

**Index rebuild** (no LLM, runs after per-page lint)

- Same deterministic rebuild as ingest Phase 2.

**Full audit mode** (`lint --full`)

- Ignores `last_linted`, processes all pages. Still parallelized per-page, just not incremental.
- Catches global contradictions between unlinked pages that incremental lint would miss.

**Benefits:**
- First lint after migration touches every page, subsequent lints only process changes.
- Each LLM call holds ~3-10 pages in context instead of the entire wiki.
- Scales to thousands of pages — cost proportional to changes, not total size.

### 4. Tool & Prompt Changes

**`_execute_tool` changes:**
- `write_page` — after writing the file, parse `[[wikilinks]]` from content, upsert into `pages` and `links` tables.
- `list_pages` — query manifest instead of scanning filesystem. Returns path, title, type, summary.
- New tool: `list_existing_entities` — returns just titles of existing entity/concept pages so the LLM can decide whether to create or update.
- `index.md` and `log.md` are removed from writable paths — they are system-managed.

**Ingest prompt changes:**
- Remove steps 5-6 (update index, append to log).
- Add: "Use `list_existing_entities` to check what exists before creating duplicates."
- Keep cross-referencing between pages the worker creates.

**Lint prompt changes:**
- Scoped to a single page + its neighbors instead of the whole wiki.
- Prompt: "Here is a page and its linked pages. Check for issues and fix them."

**New CLI commands:**
- `rebuild` — force-regenerate `index.md` and `log.md` from manifest (useful standalone).
- `lint --full` — ignore `last_linted`, process all pages.

### 5. Migration

For existing wikis (like the current `my_wiki`):
- On first run after upgrade, if `manifest.db` doesn't exist, scan all existing wiki pages to populate the manifest.
- Parse each page for title (first `# heading`), type (from directory), and `[[wikilinks]]`.
- Import `.status.json` entries into `ingestions` table, then delete `.status.json`.
- Rebuild `index.md` and `log.md` from manifest.

### 6. Backward Compatibility

- Wiki markdown files remain the source of truth for content. The manifest is a derived index that can be rebuilt from files at any time (`rebuild` command).
- Obsidian compatibility is preserved — all `[[wikilinks]]` and markdown structure unchanged.
- `--workers 1` (default) still works sequentially, same as before.
