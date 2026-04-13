# Known Capability Gaps

Last updated: 2026-04-13

Architectural and capability limitations of llm_wiki, excluding engineering bugs.

---

## 1. Ingestion: Lossy Single-Pass Extraction

### No attention gradient control
The entire document is sent as one user message. LLMs attend unevenly across long contexts ("lost in the middle" — middle chapters get worse extraction than the beginning or end). No chunking, no iterative deepening, no second pass.

**Direction:** Chunk documents by chapter/section using heading detection or page ranges. Ingest each chunk as a separate sub-document, then run a consolidation pass that merges and deduplicates across chunks.

### Extraction depth is unpredictable
The prompt says "be thorough but concise" — a tension the LLM resolves arbitrarily. Nano extracted 9 pages from 8 books; gpt-4.1 extracted 122. No mechanism to control or guarantee extraction depth.

**Direction:** Add a structured extraction step before free-form page creation: "list every entity, concept, and key claim in this document as a JSON array" → then create pages from that list. This separates *what to extract* from *how to write it up*.

### No structural awareness of source documents
PDFs are flattened to raw text. Chapter structure, headings, tables, figures, equations, footnotes — all lost. The LLM re-infers structure from the text stream.

**Direction:** Use PDF parsers that preserve structure (headings, tables, bounding boxes). Pass structured metadata (chapter titles, page numbers) alongside the text so the LLM can reference specific locations.

### No re-ingestion workflow
Once a file is marked "done," it's skipped forever. No way to re-ingest with a better model, deeper extraction, or updated source without factory resetting.

**Direction:** Add a `reingest` command that clears pages originating from a specific source and re-processes it. Track which pages came from which source in the manifest.

---

## 2. Retrieval: No Semantic Search

### `list_pages` dumps everything
At 122 pages, the tool returns ~122 lines into context. At 1,000+ pages this becomes an unusable wall of text with no relevance ranking. The LLM scans linearly and picks what looks relevant by title/summary.

**Direction:** Add a `search_pages(query)` tool backed by embedding similarity (e.g. sqlite-vss or a lightweight vector store). Return top-K relevant pages instead of all pages.

### No embedding-based retrieval
Queries can't find semantically related content that uses different terminology. "Risk management" won't surface pages about "adverse selection" or "error budgets" unless the LLM happens to read them.

**Direction:** Compute embeddings for each page on write. Add a vector search tool to the query/ingest/lint tool sets. This also enables "related pages" suggestions in Obsidian.

### No graph traversal tool
The manifest stores wikilinks but the LLM can't query them structurally. It can't say "give me all pages within 2 hops of PCA." Manual page-by-page navigation burns tool-loop turns on navigation instead of reasoning.

**Direction:** Add a `get_related(path, depth)` tool that returns the N-hop neighborhood from the links table. Cheap SQLite query, high-value context.

---

## 3. Knowledge Representation: Flat and Untyped

### Wikilinks have no relationship type
All links are identical edges. `[[Naval Ravikant]]` on a leverage page — is this "coined by", "practiced by", "contradicted by"? Can't query "who authored which concepts" or "which concepts contradict each other."

**Direction:** Extend the links table with a `relation_type` column (e.g. "references", "defines", "contradicts", "authored_by"). Update the schema to instruct the LLM to use typed links like `[[Naval Ravikant|authored_by]]`. Enable relation-aware queries.

### No provenance tracking
A claim on a concept page has no trace to which source it came from, which page of that source, or extraction confidence. If two books disagree, both get merged into one page with no conflict indicator.

**Direction:** Add a `provenance` table mapping (page, claim_hash) → (source_filename, page_number, confidence). Source pages already exist — extend concept/entity pages with inline citations like `[^source-name]`.

### Two-level taxonomy is shallow
Topic → subtopic is the only hierarchy. Real knowledge is deeper: "Mathematics → Linear Algebra → Matrix Decomposition → Eigendecomposition → Spectral Theorem." Can't represent or navigate this depth.

**Direction:** Replace the flat topic/subtopic with a hierarchical tag system (e.g. `topic/mathematics/linear-algebra/matrix-decomposition`). Render as a collapsible tree in Obsidian via nested tags.

### No temporal dimension
Pages have `created_at`/`updated_at` but content has no notion of "as of 2019" vs "as of 2024." A page about trading strategies doesn't know if its info is from a 2015 book or a 2024 roadmap.

**Direction:** Add `source_date` to frontmatter and manifest. During ingestion, extract publication date from the document. Surface recency in queries ("prioritize recent sources") and flag stale pages.

---

## 4. Query: Shallow Single-Turn Reasoning

### No multi-hop reasoning
The LLM reads pages and reports what they say. It doesn't follow chains: "PCA uses eigendecomposition → eigendecomposition requires linear transformations → covered in mml-book chapter 2." The tool loop punishes exploration (each hop = another full-context API call).

**Direction:** Add explicit multi-hop instructions to the query prompt: "follow wikilinks across pages to build a complete picture." Add the `get_related` graph traversal tool to reduce the cost of exploration.

### No outside knowledge integration
The system prompt constrains the LLM to "answer using the wiki." Gaps in extraction = gaps in answers, even when the LLM's training data could fill them.

**Direction:** Change the prompt to "use the wiki as primary context, supplement with your own knowledge, clearly marking what comes from the wiki vs your reasoning." Add a `[wiki]` vs `[external]` citation convention.

### No query memory
Each query starts from zero. "Explain PCA" → "How does it relate to SVD?" — the second query has no context from the first. The Streamlit chat shows history but the tool loop gets only the new question.

**Direction:** Pass the last N conversation turns as context to the tool loop. Or maintain a "session summary" that accumulates key findings across queries within a session.

### No question decomposition
Complex questions ("compare all decision-making frameworks across my library") should be broken into sub-queries. The current system sends the whole question to one tool loop.

**Direction:** Add a planning step before the tool loop: decompose the question into sub-queries, execute each, then synthesize. This is the "agentic design pattern" from one of the ingested books.

---

## 5. Knowledge Maintenance: No Continuous Integrity

### Lint is surface-level
Checks broken links, missing cross-references, and formatting. Doesn't check factual consistency, doesn't detect duplicate concepts under different names, doesn't identify knowledge gaps.

**Direction:** Add semantic lint rules: embedding similarity to detect near-duplicate pages, cross-source consistency checks, coverage analysis ("these chapters produced no pages").

### No contradiction detection
If two sources define "expected value" differently, both definitions coexist without any flag. No mechanism to identify, surface, or resolve conflicts.

**Direction:** During ingestion, when updating an existing concept page, compare the new content against what's already there. If they diverge, create a `contradictions/` page or add a "Disputed" section.

### No completeness awareness
The system doesn't know what it *doesn't* know. After ingesting a textbook, it can't tell you "chapters 7-12 produced no concept pages." No coverage map back to the source.

**Direction:** During ingestion, have the LLM output a chapter-by-chapter extraction manifest. Post-ingest, compare extracted pages against this manifest and report gaps.

### No knowledge decay tracking
Pages from a 2019 book aren't flagged as potentially outdated. No mechanism to mark pages as stale, schedule re-review, or prioritize re-ingestion.

**Direction:** Track `source_date` per page. Add a `stale` command that flags pages older than a threshold or whose source has a newer edition available.

---

## 6. Scaling

### Cost scales O(D × N) per operation
Every tool-loop turn re-sends the full document (ingest) or accumulated context (query). A query that reads 5 pages over 10 turns replays all 5 pages 10 times.

**Direction:** Use prompt caching (Anthropic) or context caching (OpenAI) to avoid re-processing static content. Chunk large documents so only the relevant chunk is in context per turn.

### No caching of extracted knowledge
If two queries both read the "probability" page, it's re-read from disk and re-sent each time. No embedding cache, no pre-computed summaries at different detail levels.

**Direction:** Maintain a summary cache at multiple granularities (one-liner, paragraph, full page). The `list_pages` tool already returns one-liners; add a `read_summary(path)` tool that returns the paragraph-level summary without loading the full page.

### No hierarchical index
As the wiki grows, the flat `list_pages` dump degrades. There's no topic-level index, no "table of contents" that the LLM can drill into.

**Direction:** Generate per-topic index pages (like `index.md` but scoped). The LLM reads the topic index first, then drills into specific pages. This turns O(N) scanning into O(T + K) where T = topics and K = pages within a topic.
