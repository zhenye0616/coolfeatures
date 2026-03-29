"""FRAMES benchmark evaluation — runs WikiSearchAgent on FRAMES questions.

Usage:
    uv run python frames_eval.py [--limit N] [--output results.json]

Metrics (matching Context-1 paper):
  - Final Answer Found: LLM judge checks if retrieved context contains the answer
  - Retrieval F1: precision/recall of retrieved Wikipedia URLs vs gold URLs
  - Answer Accuracy: LLM judge on generated answer vs gold answer (FRAMES paper metric)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote

import anthropic
from datasets import load_dataset

from config import LLMConfig, SearchConfig
from generate import AnswerGenerator
from wiki_agent import WikiSearchAgent


# ---------------------------------------------------------------------------
# Data models for evaluation results
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    question_idx: int
    question: str
    gold_answer: str
    gold_urls: list[str]
    retrieved_urls: list[str]
    generated_answer: str
    final_answer_found: bool
    answer_accurate: bool
    precision: float
    recall: float
    f1: float
    steps_taken: int
    context_entries: int
    elapsed_seconds: float
    error: str | None = None


@dataclass
class EvalSummary:
    total_questions: int
    final_answer_found_rate: float
    answer_accuracy_rate: float
    mean_retrieval_precision: float
    mean_retrieval_recall: float
    mean_retrieval_f1: float
    mean_steps: float
    mean_elapsed: float


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_gold_urls(row: dict) -> list[str]:
    """Extract gold Wikipedia URLs from a FRAMES dataset row."""
    urls = []
    # Try the wiki_links JSON field first
    wiki_links = row.get("wiki_links", "")
    if wiki_links:
        try:
            parsed = json.loads(wiki_links) if isinstance(wiki_links, str) else wiki_links
            if isinstance(parsed, list):
                urls.extend(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    # Also check individual link columns
    if not urls:
        for i in range(1, 15):
            col = f"wikipedia_link_{i}"
            if col in row and row[col]:
                urls.append(row[col])

    # Normalize URLs: strip trailing slashes, ensure consistent format
    normalized = []
    for url in urls:
        if isinstance(url, str) and url.strip():
            normalized.append(url.strip().rstrip("/"))
    return normalized


def extract_retrieved_urls(context_snapshot: list) -> list[str]:
    """Extract unique Wikipedia URLs from agent context entries."""
    urls: set[str] = set()
    for entry in context_snapshot:
        # doc_id format: "wiki:PageTitle#chunk_N"
        doc_id = entry.doc_id if hasattr(entry, "doc_id") else entry.get("doc_id", "")
        if doc_id.startswith("wiki:"):
            title = doc_id.split("#")[0].replace("wiki:", "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            urls.add(url)
    return list(urls)


def normalize_wiki_url(url: str) -> str:
    """Normalize a Wikipedia URL to a canonical form for comparison."""
    url = url.strip().rstrip("/")
    # Handle both /wiki/ and full URLs
    if "/wiki/" in url:
        title = url.split("/wiki/")[-1]
        # Decode percent encoding, normalize spaces/underscores
        title = unquote(title).replace("_", " ").strip().lower()
        return title
    return url.lower()


def retrieval_f1(gold_urls: list[str], retrieved_urls: list[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 between gold and retrieved Wikipedia URLs."""
    if not gold_urls:
        return (1.0, 1.0, 1.0) if not retrieved_urls else (0.0, 1.0, 0.0)
    if not retrieved_urls:
        return (0.0, 0.0, 0.0)

    gold_normalized = {normalize_wiki_url(u) for u in gold_urls}
    retrieved_normalized = {normalize_wiki_url(u) for u in retrieved_urls}

    true_positives = len(gold_normalized & retrieved_normalized)
    precision = true_positives / len(retrieved_normalized) if retrieved_normalized else 0.0
    recall = true_positives / len(gold_normalized) if gold_normalized else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (precision, recall, f1)


def judge_final_answer_found(
    context_texts: list[str],
    gold_answer: str,
    client: anthropic.Anthropic,
    model: str,
) -> bool:
    """LLM judge: does the retrieved context contain the answer?"""
    context_combined = "\n\n---\n\n".join(context_texts[:20])  # cap for token budget
    prompt = (
        f"You are an evaluation judge. Determine whether the following retrieved "
        f"context contains sufficient information to answer the question.\n\n"
        f"Gold answer: {gold_answer}\n\n"
        f"Retrieved context:\n{context_combined}\n\n"
        f"Does the context contain the information needed to produce the gold answer? "
        f"Reply with ONLY 'YES' or 'NO'."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return False


def judge_answer_accuracy(
    generated_answer: str,
    gold_answer: str,
    question: str,
    client: anthropic.Anthropic,
    model: str,
) -> bool:
    """LLM judge: does the generated answer match the gold answer?

    Uses the FRAMES paper's evaluation approach: checks if the meaning and
    vital facts of the gold answer are present in the generated answer.
    """
    prompt = (
        f"You are an evaluation judge. Determine whether the predicted answer "
        f"contains the same meaning and vital facts as the gold answer.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Predicted answer: {generated_answer}\n\n"
        f"Does the predicted answer contain the essential meaning and facts of "
        f"the gold answer? Reply with ONLY 'YES' or 'NO'."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    llm_config: LLMConfig,
    search_config: SearchConfig,
    limit: int | None = None,
    output_path: str = "frames_results.json",
) -> EvalSummary:
    """Run FRAMES evaluation end-to-end."""
    print("Loading FRAMES dataset from HuggingFace...")
    ds = load_dataset("google/frames-benchmark", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"Evaluating {len(ds)} questions")

    agent = WikiSearchAgent(llm_config, search_config)
    generator = AnswerGenerator(llm_config)
    judge_client = anthropic.Anthropic(api_key=llm_config.api_key)

    results: list[QuestionResult] = []

    for idx, row in enumerate(ds):
        question = row["Prompt"]
        gold_answer = row["Answer"]
        gold_urls = extract_gold_urls(row)

        print(f"\n[{idx + 1}/{len(ds)}] {question[:100]}...")

        start = time.time()
        try:
            # Tier 1: Agent search
            agent_result = agent.search(question)

            # Tier 2: Generate answer
            gen_result = generator.generate(question, agent_result)

            elapsed = time.time() - start

            # Extract retrieved URLs from context
            retrieved_urls = extract_retrieved_urls(agent_result.context_snapshot)

            # Context texts for judging
            context_texts = [e.text for e in agent_result.context_snapshot]

            # Metrics
            faf = judge_final_answer_found(
                context_texts, gold_answer, judge_client, llm_config.model
            )
            accuracy = judge_answer_accuracy(
                gen_result.answer, gold_answer, question, judge_client, llm_config.model
            )
            prec, rec, f1 = retrieval_f1(gold_urls, retrieved_urls)

            result = QuestionResult(
                question_idx=idx,
                question=question,
                gold_answer=gold_answer,
                gold_urls=gold_urls,
                retrieved_urls=retrieved_urls,
                generated_answer=gen_result.answer,
                final_answer_found=faf,
                answer_accurate=accuracy,
                precision=prec,
                recall=rec,
                f1=f1,
                steps_taken=agent_result.steps_taken,
                context_entries=len(agent_result.context_snapshot),
                elapsed_seconds=elapsed,
            )

            print(f"  FAF={faf} | Acc={accuracy} | F1={f1:.2f} | "
                  f"Steps={agent_result.steps_taken} | {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            result = QuestionResult(
                question_idx=idx,
                question=question,
                gold_answer=gold_answer,
                gold_urls=gold_urls,
                retrieved_urls=[],
                generated_answer="",
                final_answer_found=False,
                answer_accurate=False,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                steps_taken=0,
                context_entries=0,
                elapsed_seconds=elapsed,
                error=str(e),
            )
            print(f"  ERROR: {e}")

        results.append(result)

        # Write intermediate results after each question
        _save_results(results, output_path)

    # Compute summary
    n = len(results)
    summary = EvalSummary(
        total_questions=n,
        final_answer_found_rate=sum(r.final_answer_found for r in results) / n if n else 0,
        answer_accuracy_rate=sum(r.answer_accurate for r in results) / n if n else 0,
        mean_retrieval_precision=sum(r.precision for r in results) / n if n else 0,
        mean_retrieval_recall=sum(r.recall for r in results) / n if n else 0,
        mean_retrieval_f1=sum(r.f1 for r in results) / n if n else 0,
        mean_steps=sum(r.steps_taken for r in results) / n if n else 0,
        mean_elapsed=sum(r.elapsed_seconds for r in results) / n if n else 0,
    )

    _save_results(results, output_path, summary)
    _print_summary(summary)
    return summary


def _save_results(
    results: list[QuestionResult],
    path: str,
    summary: EvalSummary | None = None,
) -> None:
    """Write results to JSON file."""
    data = {
        "results": [asdict(r) for r in results],
    }
    if summary:
        data["summary"] = asdict(summary)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def _print_summary(summary: EvalSummary) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 60)
    print("FRAMES Evaluation Results")
    print("=" * 60)
    print(f"  Questions evaluated:     {summary.total_questions}")
    print(f"  Final Answer Found:      {summary.final_answer_found_rate:.3f}")
    print(f"  Answer Accuracy:         {summary.answer_accuracy_rate:.3f}")
    print(f"  Retrieval Precision:     {summary.mean_retrieval_precision:.3f}")
    print(f"  Retrieval Recall:        {summary.mean_retrieval_recall:.3f}")
    print(f"  Retrieval F1:            {summary.mean_retrieval_f1:.3f}")
    print(f"  Mean Steps/Question:     {summary.mean_steps:.1f}")
    print(f"  Mean Time/Question:      {summary.mean_elapsed:.1f}s")
    print("=" * 60)

    # Context-1 paper comparison
    print("\nContext-1 Paper Comparison (FRAMES):")
    print(f"  {'Metric':<25} {'Ours':>8} {'Sonnet-4.5':>12} {'C1 (1x)':>10} {'C1 (4x)':>10}")
    print(f"  {'Final Answer Found':<25} {summary.final_answer_found_rate:>8.3f} {'0.960':>12} {'0.870':>10} {'0.960':>10}")
    print(f"  {'Retrieval F1':<25} {summary.mean_retrieval_f1:>8.3f} {'0.820':>12} {'0.650':>10} {'0.790':>10}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FRAMES benchmark evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate (default: all 824)")
    parser.add_argument("--output", type=str, default="frames_results.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--max-steps", type=int, default=12, help="Max agent steps per question")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    llm_config = LLMConfig(api_key=api_key, model=args.model)
    search_config = SearchConfig(max_agent_steps=args.max_steps)

    run_evaluation(llm_config, search_config, limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()
