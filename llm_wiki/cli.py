#!/usr/bin/env python3
"""LLM Wiki — a persistent, LLM-maintained knowledge base.

Inspired by Andrej Karpathy's LLM Wiki pattern. Instead of RAG, the LLM
incrementally builds and maintains a structured wiki of interlinked markdown
files that compounds knowledge over time.

Usage:
    python cli.py --dir my_wiki init
    python cli.py --dir my_wiki ingest path/to/article.md
    python cli.py --dir my_wiki query "What are the key themes?"
    python cli.py --dir my_wiki lint
    python cli.py --dir my_wiki --backend openai ingest path/to/article.md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tool definitions (backend-neutral)
# ---------------------------------------------------------------------------

TOOLS_SPEC = [
    {
        "name": "list_pages",
        "description": "List all wiki pages with their relative paths and sizes.",
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
                    "description": "Relative path within wiki/ (e.g. 'index.md', 'entities/openai.md')",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_page",
        "description": "Create or overwrite a wiki page. Parent directories are created automatically.",
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

# ---------------------------------------------------------------------------
# Default schema — defines wiki conventions for the LLM
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA = """\
# Wiki Schema

## Structure

- `index.md` — Master index of all pages with links and one-line summaries, organized by category.
- `log.md` — Chronological, append-only record of operations (ingests, queries, lint passes).
- `sources/` — One summary page per ingested source document.
- `entities/` — Pages for people, organizations, places, products, etc.
- `concepts/` — Pages for ideas, themes, technologies, methodologies, etc.
- `analyses/` — Comparison tables, syntheses, and query results worth preserving.

## Page Conventions

- Each page starts with a `# Title` heading.
- Use `[[wikilinks]]` for cross-references between pages.
- Entity/concept pages include: brief definition, key facts, and a **References** section linking back to source pages.
- Source pages include: title, author/date if known, key takeaways, entities mentioned, concepts covered.

## Index Format

Each entry in index.md: `- [Page Title](path) — one-line description`
Grouped under category headings (Sources, Entities, Concepts, Analyses).

## Log Format

Each entry: `## [YYYY-MM-DD HH:MM] operation | Subject`
Followed by a brief description of what changed.

## Cross-Referencing

When creating or updating a page, add links to related existing pages. Ensure both sides link to each other where appropriate.
"""


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

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
        execute_tool: Callable[[str, dict], str],
        max_iterations: int = 30,
    ) -> str:
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOLS_SPEC
        ]
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=messages,
            )

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
        execute_tool: Callable[[str, dict], str],
        max_iterations: int = 30,
    ) -> str:
        tools = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in TOOLS_SPEC
        ]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )

            choice = response.choices[0]
            message = choice.message

            if not message.tool_calls:
                return message.content or "Done."

            # Append the assistant message (with tool_calls) to history
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


# ---------------------------------------------------------------------------
# Wiki Engine
# ---------------------------------------------------------------------------

class WikiEngine:
    """Core engine for LLM-maintained wiki operations."""

    def __init__(self, base_dir: str, backend):
        self.base_dir = Path(base_dir).resolve()
        self.raw_dir = self.base_dir / "raw"
        self.wiki_dir = self.base_dir / "wiki"
        self.backend = backend

    # -- Public commands ----------------------------------------------------

    def init(self):
        """Initialize a new wiki directory structure."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        schema_path = self.base_dir / "schema.md"
        if not schema_path.exists():
            schema_path.write_text(DEFAULT_SCHEMA)

        index = self.wiki_dir / "index.md"
        if not index.exists():
            index.write_text("# Wiki Index\n\n*No pages yet. Ingest a source to get started.*\n")

        log = self.wiki_dir / "log.md"
        if not log.exists():
            log.write_text("# Wiki Log\n\n")

        print(f"Initialized wiki at {self.base_dir}")
        print(f"  raw/       — drop source documents here")
        print(f"  wiki/      — LLM-generated wiki pages")
        print(f"  schema.md  — wiki conventions (edit to customize)")

    def ingest(self, source_path: str):
        """Ingest a source document or directory of documents into the wiki."""
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
            for f in files:
                self._ingest_file(f)
                print()
        else:
            self._ingest_file(source)

    def _ingest_file(self, source: Path):
        """Ingest a single file into the wiki."""
        content = self._read_source(source)
        if content is None:
            print(f"  Skipping {source.name} (unsupported format)")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Archive source to raw/
        dest = self.raw_dir / source.name
        if not dest.exists():
            shutil.copy2(source, dest)
            print(f"  Archived: raw/{source.name}")

        system = f"""\
You are a wiki maintainer. Integrate new source documents into an existing wiki.

## Wiki Schema
{self._read_schema()}

## Instructions

A new source document is provided below. Your task:

1. Call `list_pages` to see what exists, then `read_page` on `index.md` and any relevant pages.
2. Create a summary page for this source in `sources/`.
3. Create or update entity pages in `entities/` for notable people, orgs, products, etc.
4. Create or update concept pages in `concepts/` for key ideas, themes, technologies.
5. Update `index.md` to include all new and updated pages.
6. Append an entry to `log.md` (timestamp: {now}).
7. Add cross-reference links between related pages.
8. Call `done` with a brief summary.

Be thorough but concise. Extract the most important information."""

        user_msg = f"Ingest this source:\n\n**Filename:** {source.name}\n\n---\n\n{content}"

        print(f"Ingesting {source.name}...")
        summary = self.backend.run_tool_loop(system, user_msg, self._execute_tool)
        print(f"Done: {summary}")

    def query(self, question: str):
        """Query the wiki and synthesize an answer."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        system = f"""\
You are a research assistant powered by a personal wiki. Answer questions using the wiki as your knowledge base.

## Wiki Schema
{self._read_schema()}

## Instructions

1. Call `list_pages`, then `read_page` on `index.md` to find relevant pages.
2. Read relevant pages to gather information.
3. Synthesize a clear answer, citing wiki pages as sources.
4. If your answer is a valuable synthesis, save it as a new page in `analyses/` and update the index.
5. Append a query entry to `log.md` (timestamp: {now}).
6. Call `done` with your full answer as the summary.

If the wiki lacks information to answer fully, say so and suggest what sources could fill the gap."""

        user_msg = f"Question: {question}"

        print(f"Searching wiki...")
        answer = self.backend.run_tool_loop(system, user_msg, self._execute_tool)
        print(f"\n{answer}")

    def lint(self):
        """Audit the wiki for health issues and fix what's possible."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        system = f"""\
You are a wiki health auditor. Find and fix issues in the wiki.

## Wiki Schema
{self._read_schema()}

## Instructions

1. Call `list_pages` and read every page in the wiki.
2. Check for:
   - Contradictions between pages
   - Orphan pages not linked from index or other pages
   - Missing cross-references between related pages
   - Stale or incomplete content
   - Broken wikilinks pointing to pages that don't exist
   - Index.md being out of date
3. Fix issues directly by rewriting pages where possible.
4. Append a lint entry to `log.md` (timestamp: {now}).
5. Call `done` with a summary of findings and fixes.

Be thorough — read every page."""

        print("Auditing wiki...")
        summary = self.backend.run_tool_loop(system, "Perform a full health audit of the wiki.", self._execute_tool)
        print(f"\nDone: {summary}")

    # -- Helpers -------------------------------------------------------------

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

    # -- Tool execution -----------------------------------------------------

    def _execute_tool(self, name: str, inputs: dict) -> str:
        if name == "list_pages":
            pages = []
            for f in sorted(self.wiki_dir.rglob("*.md")):
                rel = f.relative_to(self.wiki_dir)
                pages.append(f"{rel} ({f.stat().st_size} bytes)")
            return "\n".join(pages) if pages else "(no pages yet)"

        if name == "read_page":
            path = self.wiki_dir / inputs["path"]
            if not path.exists():
                return f"Error: '{inputs['path']}' does not exist."
            return path.read_text()

        if name == "write_page":
            path = self.wiki_dir / inputs["path"]
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(inputs["content"])
            label = "Updated" if existed else "Created"
            print(f"  {label}: wiki/{inputs['path']}")
            return f"Wrote {inputs['path']}"

        if name == "done":
            return inputs.get("summary", "Done.")

        return f"Unknown tool: {name}"


def main():
    parser = argparse.ArgumentParser(description="LLM Wiki — persistent, LLM-maintained knowledge base")
    parser.add_argument("--dir", default=".", help="Wiki base directory (default: current dir)")
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic", help="LLM backend (default: anthropic)")
    parser.add_argument("--model", default=None, help="Model override (default: backend-specific)")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="Initialize a new wiki")
    p_ingest = sub.add_parser("ingest", help="Ingest a source document")
    p_ingest.add_argument("source", help="Path to the source file (markdown, text, etc.)")
    p_query = sub.add_parser("query", help="Query the wiki")
    p_query.add_argument("question", help="Your question")
    sub.add_parser("lint", help="Audit wiki health")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    backend = BACKENDS[args.backend](model=args.model)
    engine = WikiEngine(args.dir, backend=backend)

    match args.command:
        case "init":
            engine.init()
        case "ingest":
            engine.ingest(args.source)
        case "query":
            engine.query(args.question)
        case "lint":
            engine.lint()


if __name__ == "__main__":
    main()
