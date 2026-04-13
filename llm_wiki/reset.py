#!/usr/bin/env python3
"""Factory reset a wiki — wipes all pages, manifest data, and raw archives."""

import argparse
import shutil
import sqlite3
from pathlib import Path


def reset(wiki_dir: str):
    base = Path(wiki_dir).resolve()
    wiki = base / "wiki"
    raw = base / "raw"
    db = base / "manifest.db"

    if not base.exists():
        print(f"Error: {base} does not exist")
        return

    # Clear manifest
    if db.exists():
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM ingestions")
        conn.execute("DELETE FROM pages")
        conn.execute("DELETE FROM links")
        conn.commit()
        conn.close()
        print("  Manifest reset")

    # Remove all wiki pages (keep .obsidian/)
    if wiki.exists():
        for item in wiki.iterdir():
            if item.name == ".obsidian":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print("  Wiki pages removed")

    # Clear raw archives
    if raw.exists():
        for f in raw.iterdir():
            f.unlink()
        print("  Raw archives removed")

    # Rebuild empty index/log
    (wiki / "index.md").write_text("# Wiki Index\n")
    (wiki / "log.md").write_text("# Wiki Log\n")
    print("  Empty index/log created")

    print(f"\nFactory reset complete: {base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory reset a wiki")
    parser.add_argument("--dir", default="my_wiki", help="Wiki directory (default: my_wiki)")
    args = parser.parse_args()
    reset(args.dir)
