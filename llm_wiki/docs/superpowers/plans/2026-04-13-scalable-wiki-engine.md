# Scalable Wiki Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the wiki engine to use a SQLite manifest, two-phase ingest, and incremental lint so it scales to thousands of documents without index clobbers or context blowup.

**Architecture:** Split the monolithic `cli.py` into focused modules: `manifest.py` (SQLite layer), `backends.py` (LLM backends), `engine.py` (wiki operations), `cli.py` (arg parsing + entry point). The manifest becomes the single source of truth for page metadata and relationships; `index.md` and `log.md` become generated outputs.

**Tech Stack:** Python 3.10+, SQLite (stdlib), anthropic SDK, openai SDK, pymupdf, python-dotenv

**Spec:** `docs/superpowers/specs/2026-04-13-scalable-wiki-engine-design.md`

---

### Task 1: Create the manifest module with SQLite schema

**Files:**
- Create: `manifest.py`
- Create: `tests/test_manifest.py`

- [ ] **Step 1: Write failing tests for Manifest class**

```python
# tests/test_manifest.py
import sqlite3
from pathlib import Path
from manifest import Manifest


def test_init_creates_tables(tmp_path):
    db_path = tmp_path / "manifest.db"
    m = Manifest(db_path)
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "pages" in tables
    assert "links" in tables
    assert "ingestions" in tables


def test_upsert_page_and_query(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert_page("entities/CLIP.md", title="CLIP", type="entity", summary="Contrastive language-image model")
    page = m.get_page("entities/CLIP.md")
    assert page["title"] == "CLIP"
    assert page["type"] == "entity"
    assert page["summary"] == "Contrastive language-image model"
    assert page["created_at"] is not None
    assert page["updated_at"] is not None
    assert page["last_linted"] is None


def test_upsert_page_updates_existing(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert_page("entities/CLIP.md", title="CLIP", type="entity", summary="v1")
    old = m.get_page("entities/CLIP.md")
    m.upsert_page("entities/CLIP.md", title="CLIP", type="entity", summary="v2")
    new = m.get_page("entities/CLIP.md")
    assert new["summary"] == "v2"
    assert new["created_at"] == old["created_at"]  # preserved
    assert new["updated_at"] >= old["updated_at"]


def test_set_links_replaces_old(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert_page("a.md", title="A", type="concept", summary="")
    m.set_links("a.md", ["b.md", "c.md"])
    assert m.get_outgoing_links("a.md") == ["b.md", "c.md"]
    m.set_links("a.md", ["d.md"])
    assert m.get_outgoing_links("a.md") == ["d.md"]


def test_get_neighbors(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert_page("a.md", title="A", type="concept", summary="")
    m.upsert_page("b.md", title="B", type="concept", summary="")
    m.set_links("a.md", ["b.md"])
    m.set_links("b.md", ["a.md", "c.md"])
    neighbors = m.get_neighbors("a.md")
    assert "b.md" in neighbors  # outgoing
    # b.md links to a.md, so b.md is also incoming neighbor — already covered


def test_list_pages_by_type(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert_page("sources/a.md", title="A", type="source", summary="s1")
    m.upsert_page("entities/b.md", title="B", type="entity", summary="s2")
    m.upsert_page("entities/c.md", title="C", type="entity", summary="s3")
    assert len(m.list_pages(type="entity")) == 2
    assert len(m.list_pages()) == 3


def test_ingestion_status(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.set_ingestion_status("paper.pdf", "ingesting")
    assert m.get_ingestion_status("paper.pdf") == "ingesting"
    m.set_ingestion_status("paper.pdf", "done")
    assert m.get_ingestion_status("paper.pdf") == "done"


def test_pages_needing_lint(tmp_path):
    m = Manifest(tmp_path / "manifest.db")
    m.upsert_page("a.md", title="A", type="concept", summary="")
    m.upsert_page("b.md", title="B", type="concept", summary="")
    m.mark_linted("a.md")
    needing = m.pages_needing_lint()
    paths = [p["path"] for p in needing]
    assert "b.md" in paths
    assert "a.md" not in paths


def test_parse_wikilinks():
    from manifest import parse_wikilinks
    content = "See [[CLIP]] and [[Vision Transformer]] for details. Also [[CLIP]] again."
    links = parse_wikilinks(content)
    assert "CLIP" in links
    assert "Vision Transformer" in links
    assert len(links) == 2  # deduplicated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_manifest.py -v`
Expected: FAIL — `manifest` module does not exist

- [ ] **Step 3: Implement Manifest class**

```python
# manifest.py
"""SQLite manifest for wiki page metadata, links, and ingestion tracking."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path


def parse_wikilinks(content: str) -> set[str]:
    """Extract unique wikilink targets from markdown content."""
    return set(re.findall(r"\[\[([^\]]+)\]\]", content))


def _resolve_wikilink(link: str, all_paths: list[str]) -> str | None:
    """Resolve a wikilink name to an actual page path.

    Matches against the filename stem (case-insensitive).
    Returns the path if found, None otherwise.
    """
    link_lower = link.lower()
    for path in all_paths:
        stem = Path(path).stem.lower()
        if stem == link_lower:
            return path
    return None


class Manifest:
    """SQLite-backed manifest tracking wiki pages, links, and ingestions."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
                path        TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                type        TEXT NOT NULL,
                summary     TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                last_linted TEXT
            );
            CREATE TABLE IF NOT EXISTS links (
                from_path   TEXT NOT NULL,
                to_path     TEXT NOT NULL,
                PRIMARY KEY (from_path, to_path)
            );
            CREATE TABLE IF NOT EXISTS ingestions (
                filename    TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                finished_at TEXT
            );
        """)
        conn.commit()
        conn.close()

    def upsert_page(self, path: str, *, title: str, type: str, summary: str):
        now = datetime.now().isoformat()
        conn = self._connect()
        existing = conn.execute("SELECT created_at FROM pages WHERE path = ?", (path,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE pages SET title=?, type=?, summary=?, updated_at=? WHERE path=?",
                (title, type, summary, now, path),
            )
        else:
            conn.execute(
                "INSERT INTO pages (path, title, type, summary, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (path, title, type, summary, now, now),
            )
        conn.commit()
        conn.close()

    def get_page(self, path: str) -> dict | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM pages WHERE path = ?", (path,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_pages(self, *, type: str | None = None) -> list[dict]:
        conn = self._connect()
        if type:
            rows = conn.execute("SELECT * FROM pages WHERE type = ? ORDER BY path", (type,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM pages ORDER BY path").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def set_links(self, from_path: str, to_paths: list[str]):
        conn = self._connect()
        conn.execute("DELETE FROM links WHERE from_path = ?", (from_path,))
        for tp in to_paths:
            conn.execute("INSERT OR IGNORE INTO links (from_path, to_path) VALUES (?, ?)", (from_path, tp))
        conn.commit()
        conn.close()

    def get_outgoing_links(self, path: str) -> list[str]:
        conn = self._connect()
        rows = conn.execute("SELECT to_path FROM links WHERE from_path = ? ORDER BY to_path", (path,)).fetchall()
        conn.close()
        return [r["to_path"] for r in rows]

    def get_neighbors(self, path: str) -> set[str]:
        """Get all pages linked to or from this page."""
        conn = self._connect()
        outgoing = conn.execute("SELECT to_path FROM links WHERE from_path = ?", (path,)).fetchall()
        incoming = conn.execute("SELECT from_path FROM links WHERE to_path = ?", (path,)).fetchall()
        conn.close()
        result = {r["to_path"] for r in outgoing} | {r["from_path"] for r in incoming}
        result.discard(path)
        return result

    def set_ingestion_status(self, filename: str, status: str):
        now = datetime.now().isoformat()
        conn = self._connect()
        existing = conn.execute("SELECT * FROM ingestions WHERE filename = ?", (filename,)).fetchone()
        if existing:
            if status == "done":
                conn.execute("UPDATE ingestions SET status=?, finished_at=? WHERE filename=?", (status, now, filename))
            else:
                conn.execute("UPDATE ingestions SET status=?, started_at=? WHERE filename=?", (status, now, filename))
        else:
            conn.execute("INSERT INTO ingestions (filename, status, started_at) VALUES (?,?,?)", (filename, status, now))
        conn.commit()
        conn.close()

    def get_ingestion_status(self, filename: str) -> str | None:
        conn = self._connect()
        row = conn.execute("SELECT status FROM ingestions WHERE filename = ?", (filename,)).fetchone()
        conn.close()
        return row["status"] if row else None

    def pages_needing_lint(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM pages WHERE last_linted IS NULL OR last_linted < updated_at ORDER BY path"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def mark_linted(self, path: str):
        now = datetime.now().isoformat()
        conn = self._connect()
        conn.execute("UPDATE pages SET last_linted = ? WHERE path = ?", (now, path))
        conn.commit()
        conn.close()

    def all_paths(self) -> list[str]:
        """Return all page paths (for wikilink resolution)."""
        conn = self._connect()
        rows = conn.execute("SELECT path FROM pages ORDER BY path").fetchall()
        conn.close()
        return [r["path"] for r in rows]

    def recent_ingestions(self, status: str = "done") -> list[dict]:
        """Return ingestions with the given status, ordered by start time."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM ingestions WHERE status = ? ORDER BY started_at DESC", (status,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_manifest.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add manifest.py tests/test_manifest.py
git commit -m "feat: add SQLite manifest module with page, link, and ingestion tracking"
```

---

### Task 2: Extract LLM backends into their own module

**Files:**
- Create: `backends.py`
- Modify: `cli.py` — remove backend classes, import from `backends.py`

- [ ] **Step 1: Create backends.py by extracting from cli.py**

Move `_retry_on_rate_limit`, `TOOLS_SPEC`, `AnthropicBackend`, `OpenAIBackend`, and `BACKENDS` dict into `backends.py`. The `TOOLS_SPEC` will be passed in rather than hardcoded — this lets the engine customize available tools.

```python
# backends.py
"""LLM backend implementations for wiki tool loops."""

from __future__ import annotations

import json
import re
import time
from typing import Callable


def _retry_on_rate_limit(fn, max_retries=5):
    """Call fn(), retrying with backoff on 429 rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            error_str = str(e)
            if "429" not in error_str or attempt == max_retries:
                raise
            match = re.search(r"try again in (\d+\.?\d*)(ms|s)", error_str, re.IGNORECASE)
            if match:
                wait = float(match.group(1))
                if match.group(2) == "ms":
                    wait /= 1000
            else:
                wait = 2 ** attempt
            wait = max(wait, 1.0)
            print(f"    Rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)


class AnthropicBackend:
    """Anthropic Claude API backend."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str | None = None):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model or self.DEFAULT_MODEL

    def run_tool_loop(
        self,
        system: str,
        user_message: str,
        tools_spec: list[dict],
        execute_tool: Callable[[str, dict], str],
        max_iterations: int = 30,
    ) -> str:
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools_spec
        ]
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            response = _retry_on_rate_limit(lambda: self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
            ))

            text_parts = []
            tool_calls = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            if not tool_calls:
                return "\n".join(text_parts) if text_parts else "Done."

            tool_results = []
            done_summary = None
            for tc in tool_calls:
                result = execute_tool(tc.name, tc.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })
                if tc.name == "done":
                    done_summary = tc.input.get("summary", "Done.")

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if done_summary is not None:
                return done_summary

        return "Warning: reached maximum iterations."


class OpenAIBackend:
    """OpenAI API backend."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or self.DEFAULT_MODEL

    def run_tool_loop(
        self,
        system: str,
        user_message: str,
        tools_spec: list[dict],
        execute_tool: Callable[[str, dict], str],
        max_iterations: int = 30,
    ) -> str:
        tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools_spec
        ]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_iterations):
            response = _retry_on_rate_limit(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            ))

            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                return message.content or "Done."

            messages.append(message)

            done_summary = None
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                if tc.function.name == "done":
                    done_summary = args.get("summary", "Done.")

            if done_summary is not None:
                return done_summary

        return "Warning: reached maximum iterations."


BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
}
```

Note: The key change is `run_tool_loop` now takes `tools_spec` as a parameter instead of using a module-level global. This lets the engine pass different tool sets for ingest vs lint vs query.

- [ ] **Step 2: Verify backends.py imports cleanly**

Run: `uv run python -c "from backends import BACKENDS; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backends.py
git commit -m "feat: extract LLM backends into backends.py with parameterized tools_spec"
```

---

### Task 3: Create the wiki engine module with manifest integration

**Files:**
- Create: `engine.py`
- Create: `tests/test_engine.py`

This is the core refactor. The engine uses the manifest for all metadata, tool execution updates the manifest on writes, and `index.md`/`log.md` are system-generated.

- [ ] **Step 1: Write failing tests for engine consolidation and tool execution**

```python
# tests/test_engine.py
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock
from manifest import Manifest
from engine import WikiEngine, INGEST_TOOLS, LINT_TOOLS, QUERY_TOOLS


def _make_engine(tmp_path, backend=None):
    """Create an engine with a mock backend for testing tool execution."""
    base = tmp_path / "wiki_test"
    base.mkdir()
    if backend is None:
        backend = MagicMock()
    engine = WikiEngine(str(base), backend=backend, workers=1)
    engine.init()
    return engine


def test_init_creates_structure(tmp_path):
    engine = _make_engine(tmp_path)
    assert (engine.base_dir / "raw").is_dir()
    assert (engine.base_dir / "wiki").is_dir()
    assert (engine.base_dir / "schema.md").exists()
    assert engine.manifest.db_path == str(engine.base_dir / "manifest.db")


def test_write_page_updates_manifest(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_tool("write_page", {
        "path": "entities/CLIP.md",
        "content": "# CLIP\n\nA model by [[OpenAI]]. Related to [[Vision Transformer]].\n",
    })
    assert "Wrote" in result
    # Check manifest was updated
    page = engine.manifest.get_page("entities/CLIP.md")
    assert page is not None
    assert page["title"] == "CLIP"
    assert page["type"] == "entity"
    # Check links were parsed
    links = engine.manifest.get_outgoing_links("entities/CLIP.md")
    # Links are resolved against known pages — these don't exist yet, so links list may be empty
    # But the file should exist on disk
    assert (engine.wiki_dir / "entities/CLIP.md").exists()


def test_write_page_blocks_index_and_log(tmp_path):
    engine = _make_engine(tmp_path)
    result = engine.execute_tool("write_page", {"path": "index.md", "content": "# Hacked"})
    assert "system-managed" in result.lower() or "cannot write" in result.lower()
    result = engine.execute_tool("write_page", {"path": "log.md", "content": "# Hacked"})
    assert "system-managed" in result.lower() or "cannot write" in result.lower()


def test_list_pages_uses_manifest(tmp_path):
    engine = _make_engine(tmp_path)
    engine.execute_tool("write_page", {
        "path": "sources/paper.md",
        "content": "# Paper\n\nSummary of paper.\n",
    })
    result = engine.execute_tool("list_pages", {})
    assert "paper.md" in result
    assert "source" in result.lower()


def test_list_existing_entities(tmp_path):
    engine = _make_engine(tmp_path)
    engine.execute_tool("write_page", {
        "path": "entities/CLIP.md",
        "content": "# CLIP\n\nModel.\n",
    })
    engine.execute_tool("write_page", {
        "path": "concepts/Attention.md",
        "content": "# Attention\n\nMechanism.\n",
    })
    result = engine.execute_tool("list_existing_entities", {})
    assert "CLIP" in result
    assert "Attention" in result


def test_rebuild_index(tmp_path):
    engine = _make_engine(tmp_path)
    engine.execute_tool("write_page", {
        "path": "sources/paper.md",
        "content": "# My Paper\n\nContent.\n",
    })
    engine.execute_tool("write_page", {
        "path": "entities/CLIP.md",
        "content": "# CLIP\n\nModel.\n",
    })
    engine.rebuild()
    index_content = (engine.wiki_dir / "index.md").read_text()
    assert "My Paper" in index_content
    assert "CLIP" in index_content
    assert "Sources" in index_content
    assert "Entities" in index_content


def test_rebuild_log_from_ingestions(tmp_path):
    engine = _make_engine(tmp_path)
    engine.manifest.set_ingestion_status("paper.pdf", "done")
    engine.rebuild()
    log_content = (engine.wiki_dir / "log.md").read_text()
    assert "paper.pdf" in log_content


def test_ingest_tools_exclude_index_log():
    """Verify ingest tool spec doesn't include write access to index/log."""
    tool_names = [t["name"] for t in INGEST_TOOLS]
    assert "list_pages" in tool_names
    assert "write_page" in tool_names
    assert "list_existing_entities" in tool_names
    assert "done" in tool_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_engine.py -v`
Expected: FAIL — `engine` module does not exist

- [ ] **Step 3: Implement engine.py**

```python
# engine.py
"""Wiki engine with manifest-backed operations."""

from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from manifest import Manifest, parse_wikilinks, _resolve_wikilink

# ---------------------------------------------------------------------------
# Tool specs — different tool sets for different operations
# ---------------------------------------------------------------------------

_BASE_TOOLS = [
    {
        "name": "list_pages",
        "description": "List all wiki pages with path, title, type, and summary.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_page",
        "description": "Read the full content of a wiki page.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within wiki/ (e.g. 'entities/openai.md')",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_page",
        "description": "Create or overwrite a wiki page. Parent directories are created automatically. Cannot write to index.md or log.md (system-managed).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path within wiki/ for the page.",
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content for the page.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "done",
        "description": "Signal that the operation is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished.",
                },
            },
            "required": ["summary"],
        },
    },
]

_LIST_EXISTING_ENTITIES_TOOL = {
    "name": "list_existing_entities",
    "description": "List titles of all existing entity and concept pages. Use this to check what already exists before creating new pages.",
    "parameters": {"type": "object", "properties": {}},
}

INGEST_TOOLS = _BASE_TOOLS + [_LIST_EXISTING_ENTITIES_TOOL]
LINT_TOOLS = _BASE_TOOLS[:]  # list_pages, read_page, write_page, done
QUERY_TOOLS = _BASE_TOOLS[:]

# ---------------------------------------------------------------------------
# Default schema
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA = """\
# Wiki Schema

## Structure

- `index.md` — Master index (system-generated, do not write to it).
- `log.md` — Operation log (system-generated, do not write to it).
- `sources/` — One summary page per ingested source document.
- `entities/` — Pages for people, organizations, places, products, etc.
- `concepts/` — Pages for ideas, themes, technologies, methodologies, etc.
- `analyses/` — Comparison tables, syntheses, and query results worth preserving.

## Page Conventions

- Each page starts with a `# Title` heading.
- Use `[[wikilinks]]` for cross-references between pages.
- Entity/concept pages include: brief definition, key facts, and a **References** section linking back to source pages.
- Source pages include: title, author/date if known, key takeaways, entities mentioned, concepts covered.

## Cross-Referencing

When creating or updating a page, add links to related existing pages. Ensure both sides link to each other where appropriate.
"""

# ---------------------------------------------------------------------------
# Type inference from path
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "sources": "source",
    "entities": "entity",
    "concepts": "concept",
    "analyses": "analysis",
}


def _infer_type(path: str) -> str:
    """Infer page type from its directory."""
    parts = path.split("/")
    if len(parts) > 1 and parts[0] in _TYPE_MAP:
        return _TYPE_MAP[parts[0]]
    return "other"


def _extract_title(content: str) -> str:
    """Extract the first # heading from markdown content."""
    match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else Path(content).stem


def _extract_summary(content: str) -> str:
    """Extract first non-heading, non-empty line as summary."""
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:200]
    return ""


# ---------------------------------------------------------------------------
# Wiki Engine
# ---------------------------------------------------------------------------

class WikiEngine:
    """Core engine for LLM-maintained wiki operations."""

    def __init__(self, base_dir: str, backend, workers: int = 1):
        self.base_dir = Path(base_dir).resolve()
        self.raw_dir = self.base_dir / "raw"
        self.wiki_dir = self.base_dir / "wiki"
        self.backend = backend
        self.workers = workers
        self.manifest = Manifest(self.base_dir / "manifest.db")

    # -- Public commands ----------------------------------------------------

    def init(self):
        """Initialize a new wiki directory structure."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        schema_path = self.base_dir / "schema.md"
        if not schema_path.exists():
            schema_path.write_text(DEFAULT_SCHEMA)

        self.rebuild()

        print(f"Initialized wiki at {self.base_dir}")
        print(f"  raw/       — drop source documents here")
        print(f"  wiki/      — LLM-generated wiki pages")
        print(f"  schema.md  — wiki conventions (edit to customize)")

    def ingest(self, source_path: str):
        """Ingest source document(s) — Phase 1: parallel LLM, Phase 2: consolidation."""
        source = Path(source_path).resolve()
        if not source.exists():
            print(f"Error: {source_path} not found", file=sys.stderr)
            sys.exit(1)

        if source.is_dir():
            files = sorted(f for f in source.iterdir() if f.is_file() and not f.name.startswith("."))
            if not files:
                print(f"No files found in {source_path}", file=sys.stderr)
                sys.exit(1)
            print(f"Found {len(files)} file(s) in {source.name}/\n")
        else:
            files = [source]

        # Phase 1: parallel ingestion
        if self.workers > 1 and len(files) > 1:
            print(f"Using {self.workers} parallel workers\n")
            self._ingest_parallel(files)
        else:
            for f in files:
                self._ingest_file(f)
                print()

        # Phase 2: consolidation (no LLM)
        print("Rebuilding index and log...")
        self.rebuild()
        print("Done.")

    def query(self, question: str):
        """Query the wiki and synthesize an answer."""
        system = f"""\
You are a research assistant powered by a personal wiki. Answer questions using the wiki as your knowledge base.

## Wiki Schema
{self._read_schema()}

## Instructions

1. Call `list_pages` to find relevant pages.
2. Read relevant pages to gather information.
3. Synthesize a clear answer, citing wiki pages as sources.
4. If your answer is a valuable synthesis, save it as a new page in `analyses/`.
5. Call `done` with your full answer as the summary.

If the wiki lacks information to answer fully, say so and suggest what sources could fill the gap."""

        user_msg = f"Question: {question}"
        print("Searching wiki...")
        answer = self.backend.run_tool_loop(system, user_msg, QUERY_TOOLS, self.execute_tool)
        self.rebuild()
        print(f"\n{answer}")

    def lint(self, full: bool = False):
        """Incremental lint — only process changed pages (or all with --full)."""
        if full:
            pages = self.manifest.list_pages()
        else:
            pages = self.manifest.pages_needing_lint()

        if not pages:
            print("Nothing to lint — all pages up to date.")
            return

        print(f"Linting {len(pages)} page(s)...\n")

        if self.workers > 1 and len(pages) > 1:
            self._lint_parallel(pages)
        else:
            for page in pages:
                self._lint_page(page)

        print("\nRebuilding index and log...")
        self.rebuild()
        print("Done.")

    def rebuild(self):
        """Deterministically regenerate index.md and log.md from the manifest."""
        self._rebuild_index()
        self._rebuild_log()

    def migrate(self):
        """Migrate an existing wiki by scanning files into the manifest."""
        print("Scanning existing wiki pages into manifest...")
        count = 0
        for f in sorted(self.wiki_dir.rglob("*.md")):
            rel = str(f.relative_to(self.wiki_dir))
            if rel in ("index.md", "log.md"):
                continue
            content = f.read_text()
            title = _extract_title(content)
            page_type = _infer_type(rel)
            summary = _extract_summary(content)
            self.manifest.upsert_page(rel, title=title, type=page_type, summary=summary)
            # Parse and resolve wikilinks
            wikilink_names = parse_wikilinks(content)
            all_paths = self.manifest.all_paths()
            resolved = []
            for name in wikilink_names:
                target = _resolve_wikilink(name, all_paths)
                if target:
                    resolved.append(target)
            self.manifest.set_links(rel, resolved)
            count += 1

        # Import .status.json if it exists
        status_json = self.raw_dir / ".status.json"
        if status_json.exists():
            import json
            old_status = json.loads(status_json.read_text())
            for filename, status in old_status.items():
                self.manifest.set_ingestion_status(filename, status)
            status_json.unlink()
            print(f"  Migrated .status.json ({len(old_status)} entries)")

        print(f"  Indexed {count} pages into manifest")
        self.rebuild()
        print("Migration complete.")

    # -- Tool execution -----------------------------------------------------

    def execute_tool(self, name: str, inputs: dict) -> str:
        """Execute a wiki tool call, updating the manifest as needed."""
        if name == "list_pages":
            pages = self.manifest.list_pages()
            if not pages:
                return "(no pages yet)"
            lines = []
            for p in pages:
                lines.append(f"{p['path']} [{p['type']}] — {p.get('summary', '')}")
            return "\n".join(lines)

        if name == "list_existing_entities":
            entities = self.manifest.list_pages(type="entity")
            concepts = self.manifest.list_pages(type="concept")
            lines = []
            if entities:
                lines.append("## Entities")
                for p in entities:
                    lines.append(f"- {p['title']} ({p['path']})")
            if concepts:
                lines.append("## Concepts")
                for p in concepts:
                    lines.append(f"- {p['title']} ({p['path']})")
            return "\n".join(lines) if lines else "(no entities or concepts yet)"

        if name == "read_page":
            path = self.wiki_dir / inputs["path"]
            if not path.exists():
                return f"Error: '{inputs['path']}' does not exist."
            return path.read_text()

        if name == "write_page":
            rel_path = inputs["path"]
            if rel_path in ("index.md", "log.md"):
                return f"Cannot write to {rel_path} — system-managed. Create content pages instead."
            path = self.wiki_dir / rel_path
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            content = inputs["content"]
            path.write_text(content)

            # Update manifest
            title = _extract_title(content)
            page_type = _infer_type(rel_path)
            summary = _extract_summary(content)
            self.manifest.upsert_page(rel_path, title=title, type=page_type, summary=summary)

            # Parse and resolve wikilinks
            wikilink_names = parse_wikilinks(content)
            all_paths = self.manifest.all_paths()
            resolved = []
            for link_name in wikilink_names:
                target = _resolve_wikilink(link_name, all_paths)
                if target:
                    resolved.append(target)
            self.manifest.set_links(rel_path, resolved)

            label = "Updated" if existed else "Created"
            print(f"  {label}: wiki/{rel_path}")
            return f"Wrote {rel_path}"

        if name == "done":
            return inputs.get("summary", "Done.")

        return f"Unknown tool: {name}"

    # -- Private: ingest ----------------------------------------------------

    def _ingest_parallel(self, files: list[Path]):
        failed = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._ingest_file, f): f for f in files}
            for future in as_completed(futures):
                source_file = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((source_file.name, e))
                    print(f"  ERROR ingesting {source_file.name}: {e}")
        if failed:
            print(f"\n{len(failed)} file(s) failed:")
            for name, err in failed:
                print(f"  - {name}: {err}")

    def _ingest_file(self, source: Path):
        status = self.manifest.get_ingestion_status(source.name)

        if status == "done":
            print(f"  Skipping {source.name} (already ingested)")
            return

        content = self._read_source(source)
        if content is None:
            print(f"  Skipping {source.name} (unsupported format)")
            return

        if status == "ingesting":
            print(f"  Retrying {source.name} (previously interrupted)")

        # Archive source to raw/
        dest = self.raw_dir / source.name
        if not dest.exists():
            shutil.copy2(source, dest)
            print(f"  Archived: raw/{source.name}")

        self.manifest.set_ingestion_status(source.name, "ingesting")

        system = f"""\
You are a wiki maintainer. Integrate new source documents into an existing wiki.

## Wiki Schema
{self._read_schema()}

## Instructions

A new source document is provided below. Your task:

1. Call `list_existing_entities` to see what entity and concept pages already exist.
2. Call `list_pages` to see all pages, then `read_page` on any relevant existing pages.
3. Create a summary page for this source in `sources/`.
4. Create or update entity pages in `entities/` for notable people, orgs, products, etc.
5. Create or update concept pages in `concepts/` for key ideas, themes, technologies.
6. Add cross-reference links between related pages.
7. Call `done` with a brief summary.

Do NOT write to index.md or log.md — the system manages those automatically.
Be thorough but concise. Extract the most important information."""

        user_msg = f"Ingest this source:\n\n**Filename:** {source.name}\n\n---\n\n{content}"

        print(f"Ingesting {source.name}...")
        summary = self.backend.run_tool_loop(system, user_msg, INGEST_TOOLS, self.execute_tool)
        self.manifest.set_ingestion_status(source.name, "done")
        print(f"Done: {summary}")

    # -- Private: lint ------------------------------------------------------

    def _lint_parallel(self, pages: list[dict]):
        failed = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._lint_page, p): p for p in pages}
            for future in as_completed(futures):
                page = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((page["path"], e))
                    print(f"  ERROR linting {page['path']}: {e}")
        if failed:
            print(f"\n{len(failed)} page(s) failed lint:")
            for path, err in failed:
                print(f"  - {path}: {err}")

    def _lint_page(self, page: dict):
        """Lint a single page by reading it and its neighbors."""
        path = page["path"]
        page_content = (self.wiki_dir / path).read_text()
        neighbors = self.manifest.get_neighbors(path)

        # Build context with neighbor content
        neighbor_context = ""
        for n_path in sorted(neighbors):
            n_file = self.wiki_dir / n_path
            if n_file.exists():
                neighbor_context += f"\n\n---\n**{n_path}:**\n{n_file.read_text()}"

        system = f"""\
You are a wiki health auditor. Review a single page and its linked neighbors for issues.

## Wiki Schema
{self._read_schema()}

## Instructions

Review the page below and fix any issues:
- Broken [[wikilinks]] pointing to pages that don't exist
- Missing cross-references to related pages
- Stale or incomplete content
- Contradictions with neighbor pages

If the page needs fixes, call `write_page` to update it.
Then call `done` with a summary of findings.

Do NOT write to index.md or log.md — the system manages those automatically."""

        user_msg = f"Lint this page:\n\n**{path}:**\n{page_content}"
        if neighbor_context:
            user_msg += f"\n\n## Linked pages for context:{neighbor_context}"

        print(f"  Linting {path}...")
        self.backend.run_tool_loop(system, user_msg, LINT_TOOLS, self.execute_tool)
        self.manifest.mark_linted(path)

    # -- Private: rebuild ---------------------------------------------------

    def _rebuild_index(self):
        """Generate index.md from the manifest."""
        pages = self.manifest.list_pages()
        sections = {
            "Sources": [],
            "Entities": [],
            "Concepts": [],
            "Analyses": [],
        }
        type_to_section = {
            "source": "Sources",
            "entity": "Entities",
            "concept": "Concepts",
            "analysis": "Analyses",
        }
        for p in pages:
            section = type_to_section.get(p["type"], "Analyses")
            summary = p.get("summary", "")
            sections[section].append(f"- [{p['title']}]({p['path']}) — {summary}")

        lines = ["# Wiki Index\n"]
        for section_name, entries in sections.items():
            if entries:
                lines.append(f"## {section_name}\n")
                lines.extend(sorted(entries))
                lines.append("")

        (self.wiki_dir / "index.md").write_text("\n".join(lines))

    def _rebuild_log(self):
        """Generate log.md from the ingestions table."""
        log_path = self.wiki_dir / "log.md"
        # Preserve existing log content if it exists
        existing = ""
        if log_path.exists():
            existing = log_path.read_text()

        # Only append new entries for recently finished ingestions
        # For simplicity, regenerate from scratch
        ingestions = self.manifest.recent_ingestions("done")
        lines = ["# Wiki Log\n"]
        for ing in reversed(ingestions):  # chronological order
            ts = ing.get("finished_at") or ing["started_at"]
            # Format timestamp nicely
            try:
                dt = datetime.fromisoformat(ts)
                ts_fmt = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                ts_fmt = ts
            lines.append(f"## [{ts_fmt}] ingest | {ing['filename']}\n")
            lines.append(f"Ingested {ing['filename']} into the wiki.\n")

        log_path.write_text("\n".join(lines))

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _read_source(path: Path) -> str | None:
        """Read a source file, returning its text content or None if unsupported."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            import fitz
            doc = fitz.open(path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages).strip() or None
        try:
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            return None

    def _read_schema(self) -> str:
        schema_path = self.base_dir / "schema.md"
        if schema_path.exists():
            return schema_path.read_text()
        return DEFAULT_SCHEMA
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_engine.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine.py tests/test_engine.py
git commit -m "feat: add wiki engine with manifest-backed tools, two-phase ingest, incremental lint"
```

---

### Task 4: Rewrite cli.py as a thin entry point

**Files:**
- Modify: `cli.py` — replace with thin CLI that imports from `backends.py` and `engine.py`

- [ ] **Step 1: Rewrite cli.py**

```python
#!/usr/bin/env python3
"""LLM Wiki — a persistent, LLM-maintained knowledge base.

Usage:
    python cli.py --dir my_wiki init
    python cli.py --dir my_wiki ingest path/to/article.md
    python cli.py --dir my_wiki --backend openai ingest path/to/article.md
    python cli.py --dir my_wiki query "What are the key themes?"
    python cli.py --dir my_wiki lint
    python cli.py --dir my_wiki rebuild
    python cli.py --dir my_wiki migrate
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from backends import BACKENDS
from engine import WikiEngine


def main():
    parser = argparse.ArgumentParser(description="LLM Wiki — persistent, LLM-maintained knowledge base")
    parser.add_argument("--dir", default=".", help="Wiki base directory (default: current dir)")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic", help="LLM backend (default: anthropic)")
    parser.add_argument("--model", default=None, help="Model override (default: backend-specific)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for ingestion/lint (default: 1)")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="Initialize a new wiki")

    p_ingest = sub.add_parser("ingest", help="Ingest a source document or directory")
    p_ingest.add_argument("source", help="Path to the source file or directory")

    p_query = sub.add_parser("query", help="Query the wiki")
    p_query.add_argument("question", help="Your question")

    p_lint = sub.add_parser("lint", help="Audit wiki health (incremental)")
    p_lint.add_argument("--full", action="store_true", help="Lint all pages, not just changed ones")

    sub.add_parser("rebuild", help="Regenerate index.md and log.md from manifest")
    sub.add_parser("migrate", help="Migrate existing wiki into manifest (one-time)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    backend = BACKENDS[args.backend](model=args.model)
    engine = WikiEngine(args.dir, backend=backend, workers=args.workers)

    match args.command:
        case "init":
            engine.init()
        case "ingest":
            engine.ingest(args.source)
        case "query":
            engine.query(args.question)
        case "lint":
            engine.lint(full=args.full)
        case "rebuild":
            engine.rebuild()
            print("Rebuilt index.md and log.md from manifest.")
        case "migrate":
            engine.migrate()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

Run: `uv run python cli.py --help`
Expected: Shows all commands including `rebuild`, `migrate`, and `lint --full`

- [ ] **Step 3: Verify init command works**

Run: `uv run python cli.py --dir /tmp/test_wiki init`
Expected: Creates directory structure with manifest.db

- [ ] **Step 4: Commit**

```bash
git add cli.py
git commit -m "refactor: rewrite cli.py as thin entry point over engine + backends modules"
```

---

### Task 5: Add pytest dependency and test infrastructure

**Files:**
- Modify: `pyproject.toml` — add pytest as dev dependency
- Create: `tests/__init__.py`

- [ ] **Step 1: Add pytest**

Run: `uv add --dev pytest`

- [ ] **Step 2: Create tests/__init__.py**

```python
# tests/__init__.py
```

- [ ] **Step 3: Run all tests**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests from Task 1 and Task 3 pass

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "chore: add pytest dev dependency and test infrastructure"
```

---

### Task 6: Migrate existing wiki and end-to-end verification

**Files:**
- No new files — uses `migrate` and `rebuild` commands on existing `my_wiki`

- [ ] **Step 1: Run migration on existing wiki**

Run: `uv run python cli.py --dir my_wiki migrate`
Expected output:
```
Scanning existing wiki pages into manifest...
  Migrated .status.json (N entries)
  Indexed 171 pages into manifest
Migration complete.
```

- [ ] **Step 2: Verify manifest.db was created**

Run: `uv run python -c "from manifest import Manifest; m = Manifest('my_wiki/manifest.db'); print(f'{len(m.list_pages())} pages, {len(m.all_paths())} paths')"`
Expected: `171 pages, 171 paths` (approximately)

- [ ] **Step 3: Verify index.md was rebuilt correctly**

Run: `head -30 my_wiki/wiki/index.md`
Expected: Structured index with Sources, Entities, Concepts, Analyses sections

- [ ] **Step 4: Verify .status.json was removed**

Run: `test -f my_wiki/raw/.status.json && echo "STILL EXISTS" || echo "removed"`
Expected: `removed`

- [ ] **Step 5: Test incremental ingest (single file)**

Run: `uv run python cli.py --dir my_wiki --backend openai --model gpt-4o ingest <path-to-a-new-file>`
Expected: Ingests the file, then rebuilds index automatically. No index clobber.

- [ ] **Step 6: Test parallel ingest**

Run: `uv run python cli.py --dir my_wiki --backend openai --model gpt-4o --workers 4 ingest <path-to-directory>`
Expected: Parallel workers create pages, then single consolidation pass rebuilds index.

- [ ] **Step 7: Test incremental lint**

Run: `uv run python cli.py --dir my_wiki --backend openai --model gpt-4o lint`
Expected: Only lints pages changed since last lint (or all on first run). Each page linted individually.

- [ ] **Step 8: Test rebuild standalone**

Run: `uv run python cli.py --dir my_wiki rebuild`
Expected: Regenerates index.md and log.md from manifest without any LLM calls.

- [ ] **Step 9: Commit any fixes**

```bash
git add -A
git commit -m "fix: address issues found during end-to-end verification"
```

---

### Task 7: Clean up old code

**Files:**
- Modify: `cli.py` — verify no dead code remains
- Delete: `my_wiki/raw/.status.json` (if migration didn't remove it)

- [ ] **Step 1: Verify old cli.py code is fully replaced**

Check that `cli.py` no longer contains `WikiEngine`, `AnthropicBackend`, `OpenAIBackend`, `TOOLS_SPEC`, `DEFAULT_SCHEMA`, `_retry_on_rate_limit`, `_file_lock`, `_status_path`, or `_load_status`.

Run: `grep -n "class WikiEngine\|class AnthropicBackend\|class OpenAIBackend\|TOOLS_SPEC\|_file_lock\|_status_path\|_load_status" cli.py`
Expected: No matches

- [ ] **Step 2: Run full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove dead code from old monolithic cli.py"
```
