"""Wikipedia API client — search + fetch + chunk for the FRAMES evaluation harness.

Uses the public MediaWiki API (no API key required):
  - search endpoint for full-text search
  - query+extracts endpoint for plain-text page content
"""

from __future__ import annotations

import re

import httpx

from config import Document

_API_URL = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 30.0
_HEADERS = {"User-Agent": "AgenticSearch/0.1 (FRAMES evaluation; https://github.com)"}


def search_wikipedia(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search Wikipedia and return a list of {title, snippet} dicts."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
        resp = client.get(_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("query", {}).get("search", []):
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        results.append({"title": item["title"], "snippet": snippet})
    return results


def fetch_wikipedia(title: str) -> str | None:
    """Fetch the plain-text content of a Wikipedia page by title.

    Returns None if the page doesn't exist.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "utf8": "1",
    }
    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
        resp = client.get(_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        return page.get("extract", "")
    return None


def chunk_page(
    title: str, text: str, chunk_size: int = 512, chunk_overlap: int = 50
) -> list[Document]:
    """Split a Wikipedia page into overlapping chunks.

    Each chunk gets a doc_id like "wiki:PageTitle#chunk_0" and metadata
    including the Wikipedia URL for retrieval F1 scoring.
    """
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[Document] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_tokens = len(para) // 4
        if current_len + para_tokens > chunk_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(
                Document(
                    doc_id=f"wiki:{title}#chunk_{len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "source": f"wiki:{title}",
                        "wiki_url": url,
                        "wiki_title": title,
                        "chunk_index": len(chunks),
                    },
                )
            )
            overlap_paras: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                p_len = len(p) // 4
                if overlap_len + p_len > chunk_overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_len += p_len
            current = overlap_paras
            current_len = overlap_len

        current.append(para)
        current_len += para_tokens

    if current:
        chunk_text = "\n\n".join(current)
        chunks.append(
            Document(
                doc_id=f"wiki:{title}#chunk_{len(chunks)}",
                text=chunk_text,
                metadata={
                    "source": f"wiki:{title}",
                    "wiki_url": url,
                    "wiki_title": title,
                    "chunk_index": len(chunks),
                },
            )
        )
    return chunks
