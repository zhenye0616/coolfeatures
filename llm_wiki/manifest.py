"""SQLite-backed manifest for tracking wiki pages, links, and ingestions."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")


def parse_wikilinks(content: str) -> set[str]:
    """Extract unique ``[[wikilink]]`` targets from markdown content."""
    return set(_WIKILINK_RE.findall(content))


def _resolve_wikilink(link: str, all_paths: list[str]) -> str | None:
    """Resolve a wikilink name to an actual page path.

    Matches against filename stems (case-insensitive).  Returns the first
    matching path or ``None``.
    """
    link_lower = link.lower()
    for path in all_paths:
        stem = Path(path).stem.lower()
        if stem == link_lower:
            return path
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifest:
    """SQLite-backed manifest tracking wiki pages, links, and ingestions."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()

    # ── schema ────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        self._conn.executescript(
            """\
            CREATE TABLE IF NOT EXISTS pages (
                path        TEXT PRIMARY KEY,
                title       TEXT,
                type        TEXT,
                summary     TEXT,
                created_at  TEXT,
                updated_at  TEXT,
                last_linted TEXT
            );

            CREATE TABLE IF NOT EXISTS links (
                from_path TEXT,
                to_path   TEXT,
                PRIMARY KEY (from_path, to_path)
            );

            CREATE TABLE IF NOT EXISTS ingestions (
                filename    TEXT PRIMARY KEY,
                status      TEXT,
                started_at  TEXT,
                finished_at TEXT
            );
            """
        )

    # ── pages ─────────────────────────────────────────────────────

    def upsert_page(
        self,
        path: str,
        *,
        title: str,
        type: str,
        summary: str,
    ) -> None:
        """Insert or update a page.  Preserves ``created_at`` on update."""
        now = _now()
        self._conn.execute(
            """\
            INSERT INTO pages (path, title, type, summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                title      = excluded.title,
                type       = excluded.type,
                summary    = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (path, title, type, summary, now, now),
        )
        self._conn.commit()

    def get_page(self, path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM pages WHERE path = ?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def list_pages(self, *, type: str | None = None) -> list[dict]:
        if type is None:
            rows = self._conn.execute("SELECT * FROM pages").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pages WHERE type = ?", (type,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── links ─────────────────────────────────────────────────────

    def set_links(self, from_path: str, to_paths: list[str]) -> None:
        """Replace all outgoing links for *from_path*."""
        self._conn.execute(
            "DELETE FROM links WHERE from_path = ?", (from_path,)
        )
        self._conn.executemany(
            "INSERT INTO links (from_path, to_path) VALUES (?, ?)",
            [(from_path, tp) for tp in to_paths],
        )
        self._conn.commit()

    def get_outgoing_links(self, path: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT to_path FROM links WHERE from_path = ?", (path,)
        ).fetchall()
        return [r["to_path"] for r in rows]

    def get_neighbors(self, path: str) -> set[str]:
        """Union of outgoing and incoming links, excluding *path* itself."""
        outgoing = self._conn.execute(
            "SELECT to_path FROM links WHERE from_path = ?", (path,)
        ).fetchall()
        incoming = self._conn.execute(
            "SELECT from_path FROM links WHERE to_path = ?", (path,)
        ).fetchall()
        result = {r["to_path"] for r in outgoing} | {r["from_path"] for r in incoming}
        result.discard(path)
        return result

    # ── ingestions ─────────────────────────────────────────────────

    def set_ingestion_status(self, filename: str, status: str) -> None:
        """Upsert an ingestion record with timestamps."""
        now = _now()
        self._conn.execute(
            """\
            INSERT INTO ingestions (filename, status, started_at, finished_at)
            VALUES (?, ?, ?, NULL)
            ON CONFLICT(filename) DO UPDATE SET
                status      = excluded.status,
                finished_at = CASE WHEN excluded.status IN ('done', 'error')
                                   THEN ?
                                   ELSE ingestions.finished_at
                              END
            """,
            (filename, status, now, now),
        )
        self._conn.commit()

    def get_ingestion_status(self, filename: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM ingestions WHERE filename = ?", (filename,)
        ).fetchone()
        return row["status"] if row else None

    def recent_ingestions(self, status: str = "done") -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM ingestions WHERE status = ? ORDER BY started_at DESC",
            (status,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── lint tracking ─────────────────────────────────────────────

    def pages_needing_lint(self) -> list[dict]:
        """Pages where ``last_linted IS NULL`` or ``last_linted < updated_at``."""
        rows = self._conn.execute(
            """\
            SELECT * FROM pages
            WHERE last_linted IS NULL OR last_linted < updated_at
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_linted(self, path: str) -> None:
        """Set ``last_linted`` to the current timestamp."""
        self._conn.execute(
            "UPDATE pages SET last_linted = ? WHERE path = ?",
            (_now(), path),
        )
        self._conn.commit()

    # ── utilities ─────────────────────────────────────────────────

    def all_paths(self) -> list[str]:
        rows = self._conn.execute("SELECT path FROM pages").fetchall()
        return [r["path"] for r in rows]
