"""Corpus ingestion — reads markdown files, chunks them, and loads into DocumentStore."""

from __future__ import annotations

import os
import re
from pathlib import Path

from config import Document, SearchConfig
from storage import DocumentStore


def chunk_document(
    text: str,
    doc_id_prefix: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split a document into overlapping chunks on paragraph boundaries.

    Each chunk gets an ID like "filename#chunk_0".
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[Document] = []
    current_chunk: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_tokens = len(para) // 4  # rough token estimate
        if current_len + para_tokens > chunk_size and current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Document(
                    doc_id=f"{doc_id_prefix}#chunk_{len(chunks)}",
                    text=chunk_text,
                    metadata={"source": doc_id_prefix, "chunk_index": len(chunks)},
                )
            )
            # Overlap: keep last paragraph(s) that fit within overlap budget
            overlap_paras: list[str] = []
            overlap_len = 0
            for p in reversed(current_chunk):
                p_len = len(p) // 4
                if overlap_len + p_len > chunk_overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_len += p_len
            current_chunk = overlap_paras
            current_len = overlap_len

        current_chunk.append(para)
        current_len += para_tokens

    # Final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append(
            Document(
                doc_id=f"{doc_id_prefix}#chunk_{len(chunks)}",
                text=chunk_text,
                metadata={"source": doc_id_prefix, "chunk_index": len(chunks)},
            )
        )
    return chunks


def load_corpus(
    corpus_dir: str,
    store: DocumentStore,
    config: SearchConfig | None = None,
) -> int:
    """Read all .md files from *corpus_dir*, chunk them, and add to *store*.

    Returns the total number of chunks ingested.
    """
    cfg = config or SearchConfig()
    corpus_path = Path(corpus_dir)
    total_chunks = 0

    for md_file in sorted(corpus_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        if not text.strip():
            continue
        doc_id_prefix = md_file.stem
        chunks = chunk_document(
            text, doc_id_prefix, cfg.chunk_size, cfg.chunk_overlap
        )
        store.add_documents(chunks)
        total_chunks += len(chunks)

    return total_chunks
