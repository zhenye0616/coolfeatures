"""Wiki engine with manifest-backed tools, two-phase ingest, and incremental lint.

The engine uses the manifest for all metadata. Tool execution updates the
manifest on writes, and ``index.md``/``log.md`` are system-generated from
manifest data (never written directly by the LLM).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from manifest import Manifest, parse_wikilinks, _resolve_wikilink

# ---------------------------------------------------------------------------
# Tool specifications (backend-neutral)
# ---------------------------------------------------------------------------

_BASE_TOOLS = [
    {
        "name": "list_pages",
        "description": "List all wiki pages with their paths, types, and summaries.",
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
        "description": (
            "Create or overwrite a wiki page. Parent directories are created "
            "automatically. Cannot write to index.md or log.md — system-managed."
        ),
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
    "description": "List titles of all existing entity and concept pages.",
    "parameters": {"type": "object", "properties": {}},
}

INGEST_TOOLS: list[dict] = _BASE_TOOLS + [_LIST_EXISTING_ENTITIES_TOOL]
LINT_TOOLS: list[dict] = list(_BASE_TOOLS)
QUERY_TOOLS: list[dict] = list(_BASE_TOOLS)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _infer_type(path: str) -> str:
    """Infer page type from directory prefix."""
    first_dir = path.split("/")[0] if "/" in path else ""
    mapping = {
        "sources": "source",
        "entities": "entity",
        "concepts": "concept",
        "analyses": "analysis",
    }
    return mapping.get(first_dir, "other")


def _extract_title(content: str) -> str:
    """Extract the first ``# heading`` line from markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return "Untitled"


def _extract_summary(content: str) -> str:
    """Extract first non-heading non-empty line, truncated to 200 chars."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        return stripped[:200]
    return ""


# ---------------------------------------------------------------------------
# Default schema
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA = """\
# Wiki Schema

## Structure

- `index.md` — Master index of all pages with links and one-line summaries (system-generated, do not write to it).
- `log.md` — Chronological record of operations (system-generated, do not write to it).
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
# WikiEngine
# ---------------------------------------------------------------------------


class WikiEngine:
    """Core engine for manifest-backed, LLM-maintained wiki operations."""

    def __init__(self, base_dir: str, backend, workers: int = 1):
        self.base_dir = Path(base_dir).resolve()
        self.raw_dir = self.base_dir / "raw"
        self.wiki_dir = self.base_dir / "wiki"
        self.backend = backend
        self.workers = workers
        self._file_lock = threading.Lock()
        self.manifest = Manifest(self.base_dir / "manifest.db")

    # ── public commands ───────────────────────────────────────────

    def init(self):
        """Create directory structure, schema.md, and run rebuild."""
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
        """Ingest source document(s). Phase 1: LLM ingestion. Phase 2: rebuild."""
        source = Path(source_path).resolve()
        if not source.exists():
            print(f"Error: {source_path} not found", file=sys.stderr)
            sys.exit(1)

        if source.is_dir():
            files = sorted(
                f for f in source.iterdir()
                if f.is_file() and not f.name.startswith(".")
            )
            if not files:
                print(f"No files found in {source_path}", file=sys.stderr)
                sys.exit(1)
            print(f"Found {len(files)} file(s) in {source.name}/\n")
            if self.workers > 1:
                self._ingest_parallel(files)
            else:
                for f in files:
                    self._ingest_file(f)
                    print()
        else:
            self._ingest_file(source)

        # Phase 2: deterministic rebuild
        self.rebuild()

    def query(self, question: str):
        """Query the wiki and synthesize an answer."""
        system = f"""\
You are a research assistant powered by a personal wiki. Answer questions using the wiki as your knowledge base.

## Wiki Schema
{self._read_schema()}

## Instructions

1. Call `list_pages` to see what exists, then read relevant pages.
2. Synthesize a clear answer, citing wiki pages as sources.
3. If your answer is a valuable synthesis, save it as a new page in `analyses/`.
4. Do NOT write to index.md or log.md — the system manages those automatically.
5. Call `done` with your full answer as the summary."""

        user_msg = f"Question: {question}"

        print("Searching wiki...")
        answer = self.backend.run_tool_loop(
            system, user_msg, QUERY_TOOLS, self.execute_tool
        )
        self.rebuild()
        print(f"\n{answer}")

    def lint(self, full: bool = False):
        """Lint wiki pages. If full: all pages, else pages needing lint."""
        if full:
            pages = self.manifest.list_pages()
        else:
            pages = self.manifest.pages_needing_lint()

        if not pages:
            print("No pages need linting.")
            return

        print(f"Linting {len(pages)} page(s)...")
        if self.workers > 1 and len(pages) > 1:
            self._lint_parallel(pages)
        else:
            for page in pages:
                self._lint_page(page)

        self.rebuild()
        print("Lint complete.")

    def rebuild(self):
        """Deterministic rebuild of index.md and log.md from manifest data."""
        self._rebuild_index()
        self._rebuild_log()

    def migrate(self):
        """Scan existing wiki pages into manifest and import .status.json."""
        # Scan wiki pages
        for md_file in sorted(self.wiki_dir.rglob("*.md")):
            rel = str(md_file.relative_to(self.wiki_dir))
            if rel in ("index.md", "log.md"):
                continue
            content = md_file.read_text()
            title = _extract_title(content)
            page_type = _infer_type(rel)
            summary = _extract_summary(content)
            self.manifest.upsert_page(rel, title=title, type=page_type, summary=summary)

            # Resolve wikilinks
            links = parse_wikilinks(content)
            all_paths = self.manifest.all_paths()
            resolved = []
            for link in links:
                target = _resolve_wikilink(link, all_paths)
                if target:
                    resolved.append(target)
            self.manifest.set_links(rel, resolved)

        # Import .status.json if present
        status_path = self.raw_dir / ".status.json"
        if status_path.exists():
            status_data = json.loads(status_path.read_text())
            for filename, state in status_data.items():
                self.manifest.set_ingestion_status(filename, state)

        self.rebuild()
        print("Migration complete.")

    # ── tool execution ────────────────────────────────────────────

    def execute_tool(self, name: str, inputs: dict) -> str:
        """Execute a tool call from the LLM backend."""
        if name == "list_pages":
            pages = self.manifest.list_pages()
            if not pages:
                return "(no pages yet)"
            lines = []
            for p in pages:
                summary = p.get("summary") or ""
                lines.append(f"{p['path']} [{p['type']}] — {summary}")
            return "\n".join(lines)

        if name == "list_existing_entities":
            entities = self.manifest.list_pages(type="entity")
            concepts = self.manifest.list_pages(type="concept")
            lines = []
            if entities:
                lines.append("## Entities")
                for p in entities:
                    lines.append(f"- {p['title']}")
            if concepts:
                lines.append("## Concepts")
                for p in concepts:
                    lines.append(f"- {p['title']}")
            return "\n".join(lines) if lines else "(no entities or concepts yet)"

        if name == "read_page":
            with self._file_lock:
                path = self.wiki_dir / inputs["path"]
                if not path.exists():
                    return f"Error: '{inputs['path']}' does not exist."
                return path.read_text()

        if name == "write_page":
            rel_path = inputs["path"]
            # Block writes to system-managed pages
            if rel_path in ("index.md", "log.md"):
                return f"Error: '{rel_path}' is system-managed and cannot be written to directly."

            content = inputs["content"]
            with self._file_lock:
                full_path = self.wiki_dir / rel_path
                existed = full_path.exists()
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                label = "Updated" if existed else "Created"
                print(f"  {label}: wiki/{rel_path}")

            # Update manifest
            title = _extract_title(content)
            page_type = _infer_type(rel_path)
            summary = _extract_summary(content)
            self.manifest.upsert_page(
                rel_path, title=title, type=page_type, summary=summary
            )

            # Resolve and store wikilinks
            links = parse_wikilinks(content)
            all_paths = self.manifest.all_paths()
            resolved = []
            for link in links:
                target = _resolve_wikilink(link, all_paths)
                if target:
                    resolved.append(target)
            self.manifest.set_links(rel_path, resolved)

            return f"Wrote {rel_path}"

        if name == "done":
            return inputs.get("summary", "Done.")

        return f"Unknown tool: {name}"

    # ── private: ingest ───────────────────────────────────────────

    def _ingest_parallel(self, files: list[Path]):
        """Ingest multiple files in parallel using a thread pool."""
        print(f"Using {self.workers} parallel workers\n")
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
        """Ingest a single file: check status, archive, call LLM."""
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

1. Use `list_existing_entities` to check what entities and concepts already exist before creating duplicates.
2. Call `list_pages` to see what exists, then `read_page` on any relevant pages.
3. Create a summary page for this source in `sources/`.
4. Create or update entity pages in `entities/` for notable people, orgs, products, etc.
5. Create or update concept pages in `concepts/` for key ideas, themes, technologies.
6. Add cross-reference links between related pages.
7. Do NOT write to index.md or log.md — the system manages those automatically.
8. Call `done` with a brief summary.

Be thorough but concise. Extract the most important information."""

        user_msg = f"Ingest this source:\n\n**Filename:** {source.name}\n\n---\n\n{content}"

        print(f"Ingesting {source.name}...")
        summary = self.backend.run_tool_loop(
            system, user_msg, INGEST_TOOLS, self.execute_tool
        )
        self.manifest.set_ingestion_status(source.name, "done")
        print(f"Done: {summary}")

    # ── private: lint ─────────────────────────────────────────────

    def _lint_parallel(self, pages: list[dict]):
        """Lint multiple pages in parallel using a thread pool."""
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
            print(f"\n{len(failed)} page(s) failed linting.")

    def _lint_page(self, page: dict):
        """Lint a single page along with its neighbors."""
        path = page["path"]
        neighbors = self.manifest.get_neighbors(path)

        # Read page content
        page_file = self.wiki_dir / path
        if not page_file.exists():
            return
        page_content = page_file.read_text()

        # Read neighbor contents
        neighbor_sections = []
        for n_path in sorted(neighbors):
            n_file = self.wiki_dir / n_path
            if n_file.exists():
                n_content = n_file.read_text()
                neighbor_sections.append(f"### {n_path}\n\n{n_content}")

        neighbors_text = "\n\n".join(neighbor_sections) if neighbor_sections else "(no neighbors)"

        system = f"""\
You are a wiki page auditor. Review and fix issues in wiki pages.

## Wiki Schema
{self._read_schema()}

## Instructions

Review the page below and fix any issues:
- Broken wikilinks (links to pages that don't exist)
- Missing cross-references to related neighbors
- Stale or contradictory content compared to neighbor pages
- Formatting issues

Do NOT write to index.md or log.md — the system manages those automatically.
Call `done` when finished."""

        user_msg = f"""\
## Page to review: {path}

{page_content}

## Neighbor pages

{neighbors_text}"""

        print(f"  Linting {path}...")
        self.backend.run_tool_loop(
            system, user_msg, LINT_TOOLS, self.execute_tool
        )
        self.manifest.mark_linted(path)

    # ── private: rebuild ──────────────────────────────────────────

    def _rebuild_index(self):
        """Generate index.md from manifest data, grouped by type."""
        type_order = [
            ("source", "Sources"),
            ("entity", "Entities"),
            ("concept", "Concepts"),
            ("analysis", "Analyses"),
            ("other", "Other"),
        ]

        lines = ["# Wiki Index", ""]

        for type_key, heading in type_order:
            pages = self.manifest.list_pages(type=type_key)
            if not pages:
                continue
            lines.append(f"## {heading}")
            lines.append("")
            for p in sorted(pages, key=lambda x: x["title"]):
                summary = p.get("summary") or ""
                lines.append(f"- [{p['title']}]({p['path']}) — {summary}")
            lines.append("")

        index_path = self.wiki_dir / "index.md"
        index_path.write_text("\n".join(lines))

    def _rebuild_log(self):
        """Generate log.md from ingestion records."""
        ingestions = self.manifest.recent_ingestions(status="done")

        lines = ["# Wiki Log", ""]

        if ingestions:
            for ing in ingestions:
                ts = ing.get("finished_at") or ing.get("started_at") or "unknown"
                lines.append(f"- **{ing['filename']}** — ingested ({ts})")
            lines.append("")

        log_path = self.wiki_dir / "log.md"
        log_path.write_text("\n".join(lines))

    # ── private: helpers ──────────────────────────────────────────

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

        # Try reading as text
        try:
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            return None

    def _read_schema(self) -> str:
        schema_path = self.base_dir / "schema.md"
        if schema_path.exists():
            return schema_path.read_text()
        return DEFAULT_SCHEMA
