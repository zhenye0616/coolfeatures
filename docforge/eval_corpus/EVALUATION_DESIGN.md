# DocForge Evaluation Framework Design

## Overview

This document specifies the evaluation framework for DocForge's template extraction pipeline. Inspired by autoresearch's `evaluate_bpb` pattern (a fixed, scalar metric that the autonomous loop optimizes against), DocForge evaluation produces a **single scalar score from 0.0 to 1.0** that captures extraction quality across multiple dimensions.

The key insight from autoresearch: the metric function is **read-only** and **never modified** by the optimization loop. `evaluate_bpb` in `prepare.py` is the ground truth; `train.py` is what gets changed. Similarly, `evaluate.py` is the ground truth; `docx_template_pipeline.py` is what gets improved.

---

## 1. eval_corpus/ Directory Structure

```
eval_corpus/
  EVALUATION_DESIGN.md          # this file
  evaluate.py                   # the evaluator script (produces scalar score)
  corpus/
    contract_01/
      source.docx               # original document
      expected_fields.json      # ground truth annotations
      known_fill_data.json      # data for round-trip test
      expected_output.txt       # expected plain text after round-trip fill
    contract_02/
      source.docx
      expected_fields.json
      known_fill_data.json
      expected_output.txt
    legal_motion_01/
      ...
    business_letter_01/
      ...
    invoice_01/
      ...
  corpus_manifest.json          # index of all corpus entries + metadata
```

### corpus_manifest.json

```json
{
  "version": 1,
  "entries": [
    {
      "id": "contract_01",
      "document_type": "contract",
      "complexity": "medium",
      "field_count": 12,
      "description": "Standard SaaS service agreement"
    },
    {
      "id": "legal_motion_01",
      "document_type": "legal_motion",
      "complexity": "high",
      "field_count": 25,
      "description": "Motion for summary judgment with multiple parties"
    }
  ]
}
```

**Corpus sizing guidelines:**
- Minimum viable corpus: 5 documents (enough to detect regressions)
- Target corpus: 15-25 documents across 5+ document types
- Each document should have 5-30 expected fields

---

## 2. Ground Truth Format

### expected_fields.json

Each corpus entry has an `expected_fields.json` that mirrors the structure returned by `_analyze_document()` (the LLM analysis in `docx_template_pipeline.py:258`), but only includes the fields we care about verifying.

```json
{
  "document_type": "contract",
  "fields": [
    {
      "field_name": "client_name",
      "original_value": "Acme Corporation",
      "category": "parties",
      "is_critical": true,
      "variations": [
        {"text": "Acme Corporation", "type": "identity"},
        {"text": "ACME CORPORATION", "type": "upper"},
        {"text": "Acme", "type": "short"}
      ]
    },
    {
      "field_name": "effective_date",
      "original_value": "January 15, 2025",
      "category": "dates",
      "is_critical": true,
      "variations": [
        {"text": "January 15, 2025", "type": "identity"}
      ]
    },
    {
      "field_name": "governing_state",
      "original_value": "California",
      "category": "legal",
      "is_critical": false,
      "variations": [
        {"text": "California", "type": "identity"}
      ]
    }
  ],
  "boilerplate_phrases": [
    "WHEREAS",
    "NOW, THEREFORE",
    "IN WITNESS WHEREOF",
    "Section 1.",
    "Article II"
  ]
}
```

**Key design decisions:**

- **`is_critical`**: Distinguishes must-find fields (names, dates) from nice-to-find fields (less obvious variables). Critical field misses are penalized more heavily.
- **`variations`**: Lists known text variations in the document. Matches the `all_variations` structure from `_ANALYSIS_SCHEMA` (line 92-109 in docx_template_pipeline.py).
- **`boilerplate_phrases`**: Phrases that should NOT be marked as variable. Used for precision measurement.

### known_fill_data.json

Structured fill data matching the `fill_schema` format produced by `_build_fill_schema()` (line 359). Used for round-trip testing.

```json
{
  "parties": {
    "client_name": "Widget Industries",
    "client_name_short": "Widget"
  },
  "dates": {
    "effective_date": "March 1, 2026"
  },
  "legal": {
    "governing_state": "New York"
  }
}
```

### expected_output.txt

Plain text rendering of what the filled document should contain (key phrases only, not full document). Used for round-trip accuracy spot-checks.

```
Widget Industries
WIDGET INDUSTRIES
Widget
March 1, 2026
New York
```

---

## 3. Metrics Design

### 3.1 Field Recall (weight: 0.35)

**Question:** Did the pipeline find all expected variable fields?

```
field_recall = matched_fields / total_expected_fields
```

**Matching logic:** An expected field is "matched" if the pipeline's analysis contains a field whose `original_value` matches (exact string match) OR whose `field_name` is semantically equivalent (normalized snake_case comparison).

Matching is done at two levels:
1. **Field-level recall**: Did we find the field at all?
2. **Variation-level recall**: For found fields, did we capture all known variations?

```
variation_recall = total_matched_variations / total_expected_variations
```

**Combined field recall:**
```
R = 0.7 * field_recall + 0.3 * variation_recall
```

Critical fields (`is_critical: true`) that are missed incur double penalty:
```
critical_miss_penalty = missed_critical_fields / total_critical_fields
R_adjusted = R * (1 - 0.3 * critical_miss_penalty)
```

### 3.2 Field Precision (weight: 0.20)

**Question:** Did the pipeline avoid marking boilerplate as variable?

```
field_precision = true_positive_fields / (true_positive_fields + false_positive_fields)
```

A field is a **false positive** if:
- Its `original_value` matches any entry in `boilerplate_phrases` (substring match, case-insensitive)
- Its `original_value` is shorter than 3 characters (matches the `_MIN_REPLACEMENT_LENGTH` guard at line 302)
- Its `original_value` appears to be a section number, statute citation, or standard legal phrase

We also check for **over-segmentation**: if the pipeline splits one logical field into multiple fields (e.g., separating "John" and "Smith" instead of "John Smith"), count each extra split as 0.5 of a false positive.

### 3.3 Template Integrity (weight: 0.15)

**Question:** Does the generated template render without errors?

This is a binary-with-gradations metric:

| Condition | Score |
|-----------|-------|
| Template loads via `DocxTemplate()` and renders with dummy data without exception | 1.0 |
| Template loads but renders with warnings (e.g., undefined placeholders) | 0.7 |
| Template loads but rendering raises an exception | 0.3 |
| Template file is corrupt / cannot be loaded | 0.0 |

**Dummy data generation:** For each placeholder found via `_PLACEHOLDER_RE` (line 45), generate `{"placeholder_name": "TEST_VALUE"}`.

### 3.4 Round-Trip Accuracy (weight: 0.30)

**Question:** If we fill the template with known data, does the output contain the expected values?

This is the end-to-end test. Uses `fill_template()` (line 632) with `fill_data_override` (no LLM call needed for the fill step, but the initial `create_template()` requires one).

```
round_trip_accuracy = matched_expected_phrases / total_expected_phrases
```

**Matching:** For each line in `expected_output.txt`, check if it appears verbatim in the rendered document's extracted text.

**Variation-aware matching:** Also check that case transforms work correctly:
- If fill data has "Widget Industries", check that "WIDGET INDUSTRIES" appears where the template had an `_upper` placeholder
- Uses the transform logic from `_expand_to_placeholders()` (line 461)

### 3.5 Composite Score

```
score = (0.35 * field_recall_adjusted) +
        (0.20 * field_precision) +
        (0.15 * template_integrity) +
        (0.30 * round_trip_accuracy)
```

All sub-metrics are in [0.0, 1.0], so the composite is also in [0.0, 1.0].

**Score interpretation:**
- 0.95+ : Excellent — production-ready extraction
- 0.85-0.95 : Good — minor field misses or precision issues
- 0.70-0.85 : Acceptable — noticeable gaps but functional
- Below 0.70 : Poor — significant extraction failures

---

## 4. Fast-Check vs Full-Eval Split

### 4.1 Fast Check (free, no LLM calls)

Run on every code change before committing. Catches syntax errors, import failures, and template rendering bugs without spending API credits.

**Checks performed:**

1. **Import check**: `import docx_template_pipeline` succeeds
2. **Schema validation**: `_ANALYSIS_SCHEMA` is valid JSON Schema
3. **Regex sanity**: `_PLACEHOLDER_RE` matches expected patterns like `{{ field_name }}`
4. **Template rendering check**: For each corpus entry that already has a cached `template.docx` from a previous full eval:
   - Load via `DocxTemplate()`
   - Render with dummy placeholder values
   - Verify no exceptions
5. **Fill schema validation**: Cached `fill_schema` is valid JSON Schema
6. **Round-trip rendering**: If cached template + fill_data exist, run `fill_template()` with `fill_data_override` (no LLM call) and verify output renders

**Output:** PASS/FAIL + list of failures. Not a scalar score (fast check is a gate, not a metric).

**Expected runtime:** < 5 seconds for a 20-document corpus.

### 4.2 Full Eval (costs money, LLM calls)

Run to measure actual extraction quality. This is the scalar score used by the autonomous loop.

**Steps per corpus entry:**

1. Call `create_template(source.docx, tmp_dir, llm_config)` — this invokes `_analyze_document()` (1 LLM call)
2. Compare returned `analysis["fields"]` against `expected_fields.json` — compute recall + precision
3. Verify template integrity (load + render with dummy data)
4. Call `fill_template(artifacts, "", output_path, llm_config, fill_data_override=known_fill_data)` — render with known data (0 LLM calls since we use override)
5. Extract text from rendered document, compare against `expected_output.txt`

**Total LLM calls per corpus entry:** 1 (the `_analyze_document` call in step 1)

**Output:** Single scalar score 0.0-1.0 printed to stdout, matching autoresearch's output format:

```
---
eval_score:       0.8742
corpus_size:      15
field_recall:     0.8900
field_precision:  0.9200
template_integrity: 0.9333
round_trip:       0.8100
api_cost_usd:     0.45
eval_seconds:     82.3
```

The autonomous loop extracts: `grep "^eval_score:" eval.log`

---

## 5. Cost Estimation

### Per-Document Cost

Each full eval calls `_analyze_document()` once. Based on the `_ANALYSIS_SYSTEM_PROMPT` + document text + tool schema:

| Component | Tokens (est.) |
|-----------|---------------|
| System prompt | ~500 |
| Document text (avg) | ~3,000 |
| Tool schema | ~400 |
| LLM response (fields) | ~2,000 |
| **Total per doc** | **~6,000** |

At Claude 3.5 Sonnet pricing ($3/M input, $15/M output):
- Input: ~3,900 tokens * $3/M = $0.012
- Output: ~2,000 tokens * $15/M = $0.030
- **Per document: ~$0.04**

### Corpus-Level Cost

| Corpus Size | Cost per Full Eval | Evals per Dollar |
|-------------|-------------------|------------------|
| 5 docs | ~$0.20 | ~5 |
| 15 docs | ~$0.60 | ~1.7 |
| 25 docs | ~$1.00 | ~1.0 |
| 50 docs | ~$2.00 | ~0.5 |

### Autonomous Loop Budget

If running the autoresearch-style loop with ~12 experiments/hour:
- 5-doc corpus: ~$2.40/hour
- 15-doc corpus: ~$7.20/hour
- 25-doc corpus: ~$12.00/hour

**Recommendation:** Start with a 5-document corpus for rapid iteration, expand to 15 once the pipeline stabilizes. The fast-check gate prevents wasting API credits on broken code.

### Cost Reduction Strategies

1. **Caching**: Cache `create_template()` results keyed by `(source_docx_hash, pipeline_code_hash)`. Only re-run when either the document or the pipeline code changes.
2. **Incremental eval**: If only prompt engineering changed (not template building logic), re-run only the extraction step, not the full pipeline.
3. **Subset eval**: For rapid iteration, eval on 3 "canary" documents first; run full corpus only when canary score improves.

---

## 6. evaluate.py Script Design

```python
"""
DocForge evaluation script.

Produces a single scalar score (0.0-1.0) measuring template extraction quality.
Analogous to autoresearch's evaluate_bpb in prepare.py — this is the fixed metric.

Usage:
    python evaluate.py                          # full eval, all corpus entries
    python evaluate.py --fast                   # fast check only (no LLM, no score)
    python evaluate.py --subset 3               # eval first N corpus entries
    python evaluate.py --cache-dir /tmp/cache   # cache template artifacts
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass

from docx import Document

# Import the pipeline under test
from docx_template_pipeline import (
    LLMConfig,
    TemplateArtifacts,
    create_template,
    fill_template,
    _PLACEHOLDER_RE,
)


# ── Constants (DO NOT MODIFY — this is the fixed evaluation) ──────────────

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
MANIFEST_PATH = os.path.join(CORPUS_DIR, "..", "corpus_manifest.json")

WEIGHT_RECALL = 0.35
WEIGHT_PRECISION = 0.20
WEIGHT_INTEGRITY = 0.15
WEIGHT_ROUNDTRIP = 0.30


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class CorpusEntry:
    id: str
    source_docx: str
    expected_fields: dict
    known_fill_data: dict | None
    expected_output_lines: list[str]


@dataclass
class EntryResult:
    entry_id: str
    field_recall: float
    field_precision: float
    template_integrity: float
    round_trip_accuracy: float
    score: float
    errors: list[str]


# ── Corpus loading ───────────────────────────────────────────────────────

def load_corpus(subset: int | None = None) -> list[CorpusEntry]:
    """Load corpus entries from eval_corpus/corpus/."""
    entries = []
    corpus_dirs = sorted(
        d for d in os.listdir(CORPUS_DIR)
        if os.path.isdir(os.path.join(CORPUS_DIR, d))
    )
    if subset:
        corpus_dirs = corpus_dirs[:subset]

    for dirname in corpus_dirs:
        entry_dir = os.path.join(CORPUS_DIR, dirname)
        source_docx = os.path.join(entry_dir, "source.docx")
        fields_path = os.path.join(entry_dir, "expected_fields.json")

        if not os.path.exists(source_docx) or not os.path.exists(fields_path):
            continue

        with open(fields_path) as f:
            expected_fields = json.load(f)

        fill_path = os.path.join(entry_dir, "known_fill_data.json")
        known_fill_data = None
        if os.path.exists(fill_path):
            with open(fill_path) as f:
                known_fill_data = json.load(f)

        output_path = os.path.join(entry_dir, "expected_output.txt")
        expected_output_lines = []
        if os.path.exists(output_path):
            with open(output_path) as f:
                expected_output_lines = [
                    line.strip() for line in f if line.strip()
                ]

        entries.append(CorpusEntry(
            id=dirname,
            source_docx=source_docx,
            expected_fields=expected_fields,
            known_fill_data=known_fill_data,
            expected_output_lines=expected_output_lines,
        ))

    return entries


# ── Metric: Field Recall ─────────────────────────────────────────────────

def compute_field_recall(
    predicted_fields: list[dict],
    expected: dict,
) -> float:
    """
    Measure what fraction of expected fields were found.
    Returns adjusted recall in [0.0, 1.0].
    """
    expected_fields = expected["fields"]
    if not expected_fields:
        return 1.0

    # Build lookup: original_value -> expected field
    expected_by_value = {}
    expected_by_name = {}
    for ef in expected_fields:
        expected_by_value[ef["original_value"]] = ef
        expected_by_name[ef["field_name"]] = ef

    # Match predicted fields to expected fields
    matched_fields = set()
    matched_variations = 0
    total_variations = sum(len(ef.get("variations", [])) for ef in expected_fields)

    for pf in predicted_fields:
        pf_value = pf.get("original_value", "")
        pf_name = pf.get("field_name", "")

        # Try matching by original_value first, then by field_name
        ef = expected_by_value.get(pf_value) or expected_by_name.get(pf_name)
        if ef is None:
            continue

        matched_fields.add(ef["field_name"])

        # Check variation coverage
        predicted_var_texts = {
            v["text"] for v in pf.get("all_variations", [])
        }
        for ev in ef.get("variations", []):
            if ev["text"] in predicted_var_texts:
                matched_variations += 1

    field_recall = len(matched_fields) / len(expected_fields)
    variation_recall = (
        matched_variations / total_variations if total_variations > 0 else 1.0
    )

    R = 0.7 * field_recall + 0.3 * variation_recall

    # Critical field penalty
    critical_fields = [f for f in expected_fields if f.get("is_critical", False)]
    if critical_fields:
        missed_critical = sum(
            1 for f in critical_fields if f["field_name"] not in matched_fields
        )
        critical_miss_ratio = missed_critical / len(critical_fields)
        R = R * (1 - 0.3 * critical_miss_ratio)

    return R


# ── Metric: Field Precision ──────────────────────────────────────────────

def compute_field_precision(
    predicted_fields: list[dict],
    expected: dict,
) -> float:
    """
    Measure what fraction of predicted fields are actual variables (not boilerplate).
    Returns precision in [0.0, 1.0].
    """
    if not predicted_fields:
        return 1.0  # No predictions = no false positives (but recall handles misses)

    boilerplate = [
        bp.lower() for bp in expected.get("boilerplate_phrases", [])
    ]
    expected_values = {
        f["original_value"] for f in expected["fields"]
    }
    expected_names = {
        f["field_name"] for f in expected["fields"]
    }

    true_positives = 0
    false_positives = 0

    for pf in predicted_fields:
        pf_value = pf.get("original_value", "")
        pf_name = pf.get("field_name", "")

        # Check if this is a known expected field
        if pf_value in expected_values or pf_name in expected_names:
            true_positives += 1
            continue

        # Check if it looks like boilerplate
        is_boilerplate = any(bp in pf_value.lower() for bp in boilerplate)
        is_too_short = len(pf_value) < 3

        if is_boilerplate or is_too_short:
            false_positives += 1
        else:
            # Unknown field — not in expected, not obviously boilerplate.
            # Give partial credit: it might be a valid field we didn't annotate.
            true_positives += 0.5
            false_positives += 0.5

    total = true_positives + false_positives
    return true_positives / total if total > 0 else 1.0


# ── Metric: Template Integrity ───────────────────────────────────────────

def compute_template_integrity(template_path: str) -> float:
    """
    Check if the template loads and renders without errors.
    Returns score in [0.0, 1.0].
    """
    from docxtpl import DocxTemplate

    try:
        tpl = DocxTemplate(template_path)
    except Exception:
        return 0.0

    # Extract all placeholder names
    try:
        doc = Document(template_path)
        all_text = "\n".join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    all_text += "\n" + cell.text
        placeholders = set(_PLACEHOLDER_RE.findall(all_text))
    except Exception:
        return 0.3

    # Build dummy context
    dummy_context = {name: "TEST_VALUE" for name in placeholders}

    try:
        tpl.render(dummy_context)
        # Try saving to temp file to verify full render
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tpl.save(tmp.name)
        return 1.0
    except Exception:
        return 0.3


# ── Metric: Round-Trip Accuracy ──────────────────────────────────────────

def compute_round_trip(
    artifacts: TemplateArtifacts,
    known_fill_data: dict,
    expected_lines: list[str],
    llm_config: LLMConfig,
) -> float:
    """
    Fill template with known data, check if expected phrases appear in output.
    Returns accuracy in [0.0, 1.0].
    """
    if not expected_lines:
        return 1.0  # No expected output = skip (don't penalize)

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        output_path = tmp.name

    try:
        result = fill_template(
            artifacts,
            new_facts="",  # not used when override is provided
            output_path=output_path,
            llm_config=llm_config,
            fill_data_override=known_fill_data,
        )

        # Extract text from rendered document
        doc = Document(output_path)
        rendered_text = "\n".join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    rendered_text += "\n" + cell.text

        # Check each expected line
        matched = sum(
            1 for line in expected_lines if line in rendered_text
        )
        return matched / len(expected_lines)

    except Exception:
        return 0.0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


# ── Full evaluation ──────────────────────────────────────────────────────

def evaluate_entry(
    entry: CorpusEntry,
    llm_config: LLMConfig,
    cache_dir: str | None = None,
) -> EntryResult:
    """Run full evaluation on one corpus entry. Returns per-entry result."""
    errors = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Run create_template (1 LLM call)
        try:
            artifacts = create_template(
                entry.source_docx, tmp_dir, llm_config
            )
        except Exception as e:
            errors.append(f"create_template failed: {e}")
            return EntryResult(
                entry_id=entry.id,
                field_recall=0.0,
                field_precision=0.0,
                template_integrity=0.0,
                round_trip_accuracy=0.0,
                score=0.0,
                errors=errors,
            )

        # Step 2: Field recall
        predicted_fields = artifacts.analysis.get("fields", [])
        recall = compute_field_recall(predicted_fields, entry.expected_fields)

        # Step 3: Field precision
        precision = compute_field_precision(
            predicted_fields, entry.expected_fields
        )

        # Step 4: Template integrity
        integrity = compute_template_integrity(artifacts.template_path)

        # Step 5: Round-trip accuracy
        if entry.known_fill_data and entry.expected_output_lines:
            roundtrip = compute_round_trip(
                artifacts,
                entry.known_fill_data,
                entry.expected_output_lines,
                llm_config,
            )
        else:
            roundtrip = 1.0  # Skip if no fill data provided

    # Composite score
    score = (
        WEIGHT_RECALL * recall
        + WEIGHT_PRECISION * precision
        + WEIGHT_INTEGRITY * integrity
        + WEIGHT_ROUNDTRIP * roundtrip
    )

    return EntryResult(
        entry_id=entry.id,
        field_recall=recall,
        field_precision=precision,
        template_integrity=integrity,
        round_trip_accuracy=roundtrip,
        score=score,
        errors=errors,
    )


# ── Fast check ───────────────────────────────────────────────────────────

def fast_check() -> bool:
    """
    Free, no-LLM checks. Returns True if all pass.
    Run before every commit / full eval.
    """
    print("=== Fast Check ===")
    passed = True

    # 1. Import check
    try:
        import docx_template_pipeline  # noqa: F401
        print("PASS  import docx_template_pipeline")
    except Exception as e:
        print(f"FAIL  import: {e}")
        passed = False

    # 2. Schema validation
    try:
        from docx_template_pipeline import _ANALYSIS_SCHEMA
        assert "properties" in _ANALYSIS_SCHEMA
        assert "fields" in _ANALYSIS_SCHEMA["properties"]
        print("PASS  _ANALYSIS_SCHEMA is valid")
    except Exception as e:
        print(f"FAIL  schema: {e}")
        passed = False

    # 3. Regex check
    try:
        from docx_template_pipeline import _PLACEHOLDER_RE
        assert _PLACEHOLDER_RE.search("{{ field_name }}")
        assert _PLACEHOLDER_RE.search("{{field_name}}")
        assert not _PLACEHOLDER_RE.search("plain text")
        print("PASS  _PLACEHOLDER_RE works correctly")
    except Exception as e:
        print(f"FAIL  regex: {e}")
        passed = False

    # 4. Template rendering check (cached templates only)
    corpus = load_corpus()
    for entry in corpus:
        cached_template = os.path.join(
            os.path.dirname(entry.source_docx), "cached_template.docx"
        )
        if os.path.exists(cached_template):
            integrity = compute_template_integrity(cached_template)
            status = "PASS" if integrity >= 0.7 else "FAIL"
            print(f"{status}  template integrity [{entry.id}]: {integrity:.2f}")
            if integrity < 0.7:
                passed = False

    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return passed


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DocForge evaluation — produces a scalar score (0.0-1.0)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run fast checks only (no LLM calls, no score)"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Evaluate only the first N corpus entries"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for caching template artifacts"
    )
    parser.add_argument(
        "--model", type=str, default="anthropic/claude-3.5-sonnet",
        help="LLM model to use for extraction"
    )
    args = parser.parse_args()

    if args.fast:
        ok = fast_check()
        sys.exit(0 if ok else 1)

    # Full evaluation requires API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    llm_config = LLMConfig(api_key=api_key, model=args.model)
    corpus = load_corpus(subset=args.subset)

    if not corpus:
        print("ERROR: No corpus entries found in eval_corpus/corpus/", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(corpus)} corpus entries...\n")
    t0 = time.time()
    results = []
    total_cost = 0.0

    for entry in corpus:
        print(f"  [{entry.id}] ", end="", flush=True)
        result = evaluate_entry(entry, llm_config, cache_dir=args.cache_dir)
        results.append(result)
        est_cost = 0.04  # estimated per-document cost
        total_cost += est_cost
        status = "OK" if not result.errors else f"ERRORS: {result.errors}"
        print(f"score={result.score:.4f}  {status}")

    elapsed = time.time() - t0

    # Aggregate: mean across corpus entries
    avg_score = sum(r.score for r in results) / len(results)
    avg_recall = sum(r.field_recall for r in results) / len(results)
    avg_precision = sum(r.field_precision for r in results) / len(results)
    avg_integrity = sum(r.template_integrity for r in results) / len(results)
    avg_roundtrip = sum(r.round_trip_accuracy for r in results) / len(results)

    # Print summary in autoresearch-compatible format
    print("\n---")
    print(f"eval_score:       {avg_score:.6f}")
    print(f"corpus_size:      {len(corpus)}")
    print(f"field_recall:     {avg_recall:.4f}")
    print(f"field_precision:  {avg_precision:.4f}")
    print(f"template_integrity: {avg_integrity:.4f}")
    print(f"round_trip:       {avg_roundtrip:.4f}")
    print(f"api_cost_usd:     {total_cost:.2f}")
    print(f"eval_seconds:     {elapsed:.1f}")


if __name__ == "__main__":
    main()
```

---

## 7. Integration with Autonomous Loop

### How the Loop Uses This Evaluator

Following the autoresearch pattern from `program.md`:

1. The loop modifies `docx_template_pipeline.py` (equivalent to `train.py`)
2. Runs `python evaluate.py > eval.log 2>&1`
3. Extracts metric: `grep "^eval_score:" eval.log`
4. If `eval_score` improved (higher is better, opposite of val_bpb), keep the change
5. If `eval_score` is equal or worse, revert

### Results TSV

```
commit	eval_score	api_cost	status	description
a1b2c3d	0.874200	0.20	keep	baseline
b2c3d4e	0.891500	0.20	keep	improve variation detection in system prompt
c3d4e5f	0.860000	0.20	discard	add aggressive boilerplate filtering
```

### Guard Rails

- **Fast check gate**: Always run `python evaluate.py --fast` before full eval. If fast check fails, don't spend API credits.
- **Canary subset**: For rapid iteration, use `python evaluate.py --subset 3` as a quick signal before running full corpus.
- **Cost cap**: The loop should track cumulative API spend and pause if it exceeds a configured budget.

---

## 8. Building the Corpus

### Bootstrap Strategy

To create the initial corpus:

1. Take 5 real .docx documents from different domains (contracts, letters, invoices, legal motions, HR forms)
2. For each document, manually annotate `expected_fields.json` by reading the document and identifying all variable fields
3. Create `known_fill_data.json` with plausible replacement values
4. Create `expected_output.txt` with the key phrases that should appear after filling
5. Run `python evaluate.py` to establish a baseline score

### Annotation Guidelines

- Read the full document before annotating
- A field is "variable" if it would change when reusing the document for a different case/client/situation
- Standard legal language, statute citations, and section headers are NOT variable
- Mark names, dates, addresses, case numbers, dollar amounts, and entity-specific references as variable
- For each field, list ALL variations found in the document (uppercase, abbreviated, with/without prefix)
- Set `is_critical: true` for fields where missing them would make the template unusable (primary party names, key dates)

---

## 9. Determinism and Reproducibility

LLM outputs are inherently non-deterministic. To manage this:

1. **Multiple runs**: For reliable measurement, average eval_score over 2-3 runs (configurable via `--runs N` flag, future enhancement)
2. **Seed control**: Use `temperature=0` in LLM calls when supported (already implicit with tool_choice forcing)
3. **Cached baselines**: Store the baseline analysis for comparison, so metric changes reflect pipeline changes, not LLM variance
4. **Score thresholds**: Only count an improvement if eval_score improves by >= 0.005 (avoids chasing noise)

---

## 10. Future Extensions

- **Field-type-specific metrics**: Separate scores for dates vs names vs addresses
- **Formatting preservation**: Check that bold/italic/font styling survives the template round-trip
- **Multi-language support**: Corpus entries in non-English languages
- **Regression tests**: Pin specific field extractions as regression tests that must always pass
- **Human-in-the-loop annotation**: Tool to review and correct LLM field extractions, building the corpus incrementally
