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

from core.backends import BACKENDS
from core.engine import WikiEngine


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
    sub.add_parser("theme", help="Generate Obsidian CSS snippet for topic color-coding")

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
        case "theme":
            from core.theme import generate_snippet
            out = generate_snippet(engine.manifest, engine.wiki_dir)
            print(f"CSS snippet written to {out}")
            print("Enable in Obsidian: Settings → Appearance → CSS snippets → topic-colors")


if __name__ == "__main__":
    main()
