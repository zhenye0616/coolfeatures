"""Tests for the SQLite-backed manifest module."""

from __future__ import annotations

import sqlite3
import time

import pytest

from manifest import Manifest, parse_wikilinks


# ── Table creation ────────────────────────────────────────────────


def test_init_creates_tables(tmp_path):
    db = tmp_path / "wiki.db"
    m = Manifest(db)

    conn = sqlite3.connect(str(db))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    conn.close()

    assert "pages" in tables
    assert "links" in tables
    assert "ingestions" in tables


def test_init_wal_mode(tmp_path):
    db = tmp_path / "wiki.db"
    Manifest(db)

    conn = sqlite3.connect(str(db))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"


# ── upsert_page / get_page ───────────────────────────────────────


def test_upsert_and_get_page(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("concepts/llm.md", title="LLM", type="concept", summary="Large language models")

    page = m.get_page("concepts/llm.md")
    assert page is not None
    assert page["title"] == "LLM"
    assert page["type"] == "concept"
    assert page["summary"] == "Large language models"
    assert page["created_at"] is not None
    assert page["updated_at"] is not None


def test_upsert_preserves_created_at(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="first")

    page1 = m.get_page("a.md")
    created_at_1 = page1["created_at"]

    # Small delay to ensure updated_at differs
    time.sleep(0.05)

    m.upsert_page("a.md", title="A v2", type="concept", summary="updated")

    page2 = m.get_page("a.md")
    assert page2["created_at"] == created_at_1, "created_at must be preserved on update"
    assert page2["title"] == "A v2"
    assert page2["summary"] == "updated"


def test_get_page_missing(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    assert m.get_page("nonexistent.md") is None


# ── list_pages ────────────────────────────────────────────────────


def test_list_pages_all(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="a")
    m.upsert_page("b.md", title="B", type="entity", summary="b")

    pages = m.list_pages()
    assert len(pages) == 2


def test_list_pages_by_type(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="a")
    m.upsert_page("b.md", title="B", type="entity", summary="b")
    m.upsert_page("c.md", title="C", type="concept", summary="c")

    concepts = m.list_pages(type="concept")
    assert len(concepts) == 2
    assert all(p["type"] == "concept" for p in concepts)

    entities = m.list_pages(type="entity")
    assert len(entities) == 1


# ── set_links / get_outgoing_links / get_neighbors ────────────────


def test_set_links_and_get_outgoing(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="a")
    m.set_links("a.md", ["b.md", "c.md"])

    outgoing = m.get_outgoing_links("a.md")
    assert set(outgoing) == {"b.md", "c.md"}


def test_set_links_replaces_old(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.set_links("a.md", ["b.md", "c.md"])
    m.set_links("a.md", ["d.md"])

    outgoing = m.get_outgoing_links("a.md")
    assert outgoing == ["d.md"]


def test_get_neighbors(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    # a -> b, a -> c, d -> a
    m.set_links("a.md", ["b.md", "c.md"])
    m.set_links("d.md", ["a.md"])

    neighbors = m.get_neighbors("a.md")
    assert neighbors == {"b.md", "c.md", "d.md"}
    assert "a.md" not in neighbors


# ── ingestion status ──────────────────────────────────────────────


def test_set_and_get_ingestion_status(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.set_ingestion_status("paper.pdf", "running")

    assert m.get_ingestion_status("paper.pdf") == "running"


def test_ingestion_status_missing(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    assert m.get_ingestion_status("nope.pdf") is None


def test_set_ingestion_status_updates(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.set_ingestion_status("paper.pdf", "running")
    m.set_ingestion_status("paper.pdf", "done")

    assert m.get_ingestion_status("paper.pdf") == "done"


def test_recent_ingestions(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.set_ingestion_status("a.pdf", "done")
    m.set_ingestion_status("b.pdf", "running")
    m.set_ingestion_status("c.pdf", "done")

    done = m.recent_ingestions(status="done")
    assert len(done) == 2
    assert all(d["status"] == "done" for d in done)


# ── lint tracking ─────────────────────────────────────────────────


def test_pages_needing_lint(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="a")
    m.upsert_page("b.md", title="B", type="concept", summary="b")

    # Both should need linting (last_linted is NULL)
    needing = m.pages_needing_lint()
    assert len(needing) == 2

    # Lint one
    m.mark_linted("a.md")

    needing = m.pages_needing_lint()
    assert len(needing) == 1
    assert needing[0]["path"] == "b.md"


def test_pages_needing_lint_after_update(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="a")
    m.mark_linted("a.md")

    # At this point a.md should NOT need linting
    assert len(m.pages_needing_lint()) == 0

    # Update the page — now it needs linting again
    time.sleep(0.05)
    m.upsert_page("a.md", title="A v2", type="concept", summary="updated")

    needing = m.pages_needing_lint()
    assert len(needing) == 1
    assert needing[0]["path"] == "a.md"


# ── all_paths ─────────────────────────────────────────────────────


def test_all_paths(tmp_path):
    m = Manifest(tmp_path / "wiki.db")
    m.upsert_page("a.md", title="A", type="concept", summary="a")
    m.upsert_page("b.md", title="B", type="entity", summary="b")

    paths = m.all_paths()
    assert set(paths) == {"a.md", "b.md"}


# ── parse_wikilinks ──────────────────────────────────────────────


def test_parse_wikilinks_basic():
    content = "See [[LLM]] and [[Transformer]] for details."
    links = parse_wikilinks(content)
    assert links == {"LLM", "Transformer"}


def test_parse_wikilinks_deduplication():
    content = "The [[LLM]] is great. Read more about [[LLM]] here."
    links = parse_wikilinks(content)
    assert links == {"LLM"}


def test_parse_wikilinks_empty():
    assert parse_wikilinks("No links here.") == set()


def test_parse_wikilinks_multiline():
    content = "First [[Alpha]]\nSecond [[Beta]]\n[[Alpha]] again"
    links = parse_wikilinks(content)
    assert links == {"Alpha", "Beta"}
