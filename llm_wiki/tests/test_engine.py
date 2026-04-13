"""Tests for the wiki engine module with manifest-backed tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine import (
    DEFAULT_SCHEMA,
    INGEST_TOOLS,
    LINT_TOOLS,
    QUERY_TOOLS,
    WikiEngine,
    _extract_summary,
    _extract_title,
    _infer_type,
)


# ── Helper function tests ─────────────────────────────────────────


def test_infer_type():
    assert _infer_type("sources/paper.md") == "source"
    assert _infer_type("entities/openai.md") == "entity"
    assert _infer_type("concepts/llm.md") == "concept"
    assert _infer_type("analyses/comparison.md") == "analysis"
    assert _infer_type("random/other.md") == "other"


def test_extract_title():
    assert _extract_title("# My Title\n\nSome body.") == "My Title"
    assert _extract_title("No heading here.") == "Untitled"


def test_extract_summary():
    content = "# Title\n\nThis is the first paragraph of the page."
    assert _extract_summary(content) == "This is the first paragraph of the page."


def test_extract_summary_truncation():
    content = "# Title\n\n" + "A" * 300
    assert len(_extract_summary(content)) == 200


# ── WikiEngine init ───────────────────────────────────────────────


def test_init_creates_structure(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "wiki").is_dir()
    assert (tmp_path / "schema.md").is_file()
    assert (tmp_path / "manifest.db").is_file()


# ── execute_tool: write_page ──────────────────────────────────────


def test_write_page_updates_manifest(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    result = engine.execute_tool(
        "write_page",
        {"path": "entities/openai.md", "content": "# OpenAI\n\nAn AI research company."},
    )

    # File should exist on disk
    assert (tmp_path / "wiki" / "entities" / "openai.md").exists()
    assert "Wrote" in result

    # Manifest should have the page
    page = engine.manifest.get_page("entities/openai.md")
    assert page is not None
    assert page["title"] == "OpenAI"
    assert page["type"] == "entity"


def test_write_page_blocks_index_and_log(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    result_index = engine.execute_tool(
        "write_page", {"path": "index.md", "content": "# Hacked Index"}
    )
    assert "error" in result_index.lower() or "Error" in result_index

    result_log = engine.execute_tool(
        "write_page", {"path": "log.md", "content": "# Hacked Log"}
    )
    assert "error" in result_log.lower() or "Error" in result_log

    # Files should NOT have been created/overwritten
    # (init creates index.md/log.md via rebuild, but the content should
    # not be the hacked content)
    index_content = (tmp_path / "wiki" / "index.md").read_text()
    assert "Hacked" not in index_content


# ── execute_tool: list_pages ──────────────────────────────────────


def test_list_pages_uses_manifest(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    engine.execute_tool(
        "write_page",
        {"path": "entities/openai.md", "content": "# OpenAI\n\nAn AI company."},
    )
    engine.execute_tool(
        "write_page",
        {"path": "concepts/llm.md", "content": "# LLM\n\nLarge language models."},
    )

    result = engine.execute_tool("list_pages", {})
    assert "entities/openai.md" in result
    assert "concepts/llm.md" in result
    assert "entity" in result
    assert "concept" in result


# ── execute_tool: list_existing_entities ──────────────────────────


def test_list_existing_entities(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    engine.execute_tool(
        "write_page",
        {"path": "entities/openai.md", "content": "# OpenAI\n\nAn AI company."},
    )
    engine.execute_tool(
        "write_page",
        {"path": "concepts/llm.md", "content": "# LLM\n\nLarge language models."},
    )
    engine.execute_tool(
        "write_page",
        {"path": "sources/paper.md", "content": "# Paper\n\nA research paper."},
    )

    result = engine.execute_tool("list_existing_entities", {})
    assert "OpenAI" in result
    assert "LLM" in result
    # Sources should NOT appear
    assert "Paper" not in result


# ── rebuild ───────────────────────────────────────────────────────


def test_rebuild_index(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    engine.execute_tool(
        "write_page",
        {"path": "sources/paper.md", "content": "# Great Paper\n\nSummary of the paper."},
    )
    engine.execute_tool(
        "write_page",
        {"path": "entities/openai.md", "content": "# OpenAI\n\nAn AI company."},
    )
    engine.execute_tool(
        "write_page",
        {"path": "concepts/llm.md", "content": "# LLM\n\nLarge language models."},
    )

    engine.rebuild()

    index_path = tmp_path / "wiki" / "index.md"
    assert index_path.exists()
    index_content = index_path.read_text()

    assert "# Wiki Index" in index_content
    assert "Sources" in index_content
    assert "Entities" in index_content
    assert "Concepts" in index_content
    assert "Great Paper" in index_content
    assert "OpenAI" in index_content
    assert "LLM" in index_content


def test_rebuild_log_from_ingestions(tmp_path):
    backend = MagicMock()
    engine = WikiEngine(str(tmp_path), backend=backend)
    engine.init()

    engine.manifest.set_ingestion_status("paper.pdf", "done")
    engine.manifest.set_ingestion_status("article.md", "done")

    engine.rebuild()

    log_path = tmp_path / "wiki" / "log.md"
    assert log_path.exists()
    log_content = log_path.read_text()

    assert "# Wiki Log" in log_content
    assert "paper.pdf" in log_content
    assert "article.md" in log_content


# ── Tool specs ────────────────────────────────────────────────────


def test_ingest_tools_exclude_index_log():
    """INGEST_TOOLS should contain list_existing_entities but
    write_page description should warn about index/log."""
    tool_names = {t["name"] for t in INGEST_TOOLS}
    assert "list_pages" in tool_names
    assert "read_page" in tool_names
    assert "write_page" in tool_names
    assert "done" in tool_names
    assert "list_existing_entities" in tool_names

    # write_page description should mention index.md and log.md
    write_tool = next(t for t in INGEST_TOOLS if t["name"] == "write_page")
    assert "index.md" in write_tool["description"]
    assert "log.md" in write_tool["description"]
