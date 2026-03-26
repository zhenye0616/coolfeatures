# docforge

This is an experiment to have an LLM autonomously improve a document template extraction pipeline.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `docforge/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b docforge/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `evaluate.py` — fixed evaluation harness, corpus loading, scoring. Do not modify.
   - `docx_template_pipeline.py` — the file you modify. LLM prompts, replacement logic, transform types, schema construction, fill expansion.
   - `app.py` — Streamlit UI. Do not modify.
4. **Verify eval corpus exists**: Check that `eval_corpus/` contains subdirectories, each with a source `.docx`, `expected_analysis.json`, and optionally `expected_fill.json`. If empty, tell the human to populate it.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment evaluates the pipeline against a fixed corpus of document templates. You launch the evaluator simply as: `uv run evaluate.py`.

**What you CAN do:**
- Modify `docx_template_pipeline.py` — this is the only file you edit. Everything is fair game: LLM analysis prompt, replacement rules, transform types, fill prompt engineering, schema construction, error handling, text extraction logic, variation matching, etc.

**What you CANNOT do:**
- Modify `evaluate.py`. It is read-only. It contains the fixed evaluation harness, corpus loading, and scoring functions.
- Modify `app.py`. It is read-only. It is the Streamlit UI.
- Modify anything in `eval_corpus/`. The evaluation corpus is the ground truth. Adding, removing, or editing test documents is gaming the eval.
- **Read ANY file inside `eval_corpus/`**. You may only interact with the corpus through running `evaluate.py`. Do not `cat`, `read`, `grep`, or otherwise inspect source documents, ground truth annotations (expected_fields.json, known_fill_data.json, expected_output.txt), or any other file under `eval_corpus/`. Hardcoding logic based on corpus contents is gaming the eval. The evaluator is a black box — you see the score, nothing else.
- Modify `pyproject.toml` or install new packages. You can only use what's already available.
- Modify any files in `templates/`. These are user-facing saved templates, not part of the eval.

**The goal is simple: get the highest eval_score.** The evaluator measures how well the pipeline extracts fields, builds templates, and fills them with new data. A higher score means more fields correctly identified, fewer false positives, better placeholder replacement coverage, and more accurate fill expansion. Everything is fair game: change the analysis prompt, add new transform types, improve replacement logic, handle edge cases. The only constraint is that the code runs without crashing and produces valid results.

**The first run**: Your very first run should always be to establish the baseline, so you will run the evaluator as-is.

## Output format

Once the evaluator finishes it prints a summary like this:

```
---
eval_score:         0.7234
field_precision:    0.8100
field_recall:       0.6500
template_accuracy:  0.7800
fill_accuracy:      0.6500
corpus_size:        16
errors:             1
total_seconds:      142.3
```

You can extract the key metric from the log file:

```
grep "^eval_score:" eval.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	eval_score	status	description
```

1. git commit hash (short, 7 chars)
2. eval_score achieved (e.g. 0.723400) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	eval_score	status	description
a1b2c3d	0.723400	keep	baseline
b2c3d4e	0.751200	keep	add rule 10 to analysis prompt re: short values
c3d4e5f	0.710000	discard	switch to per-field LLM calls
d4e5f6g	0.000000	crash	refactor variation map (KeyError)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `docforge/mar25`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `docx_template_pipeline.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the evaluator: `uv run python eval_corpus/evaluate.py --subset 5 > eval.log 2>&1` (redirect everything — do NOT use tee or let output flood your context. Increase --subset as you scale up tiers.)
5. Read out the results: `grep "^eval_score:" eval.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 eval.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If eval_score improved by **more than 0.005** (higher), you "advance" the branch, keeping the git commit
9. If eval_score improved by **0.005 or less**, or is equal or worse, you git reset back to where you started. Improvements within 0.005 are likely LLM variance noise, not real signal — do not keep them.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Corpus scaling**: You start with `--subset 5`. Scale up when the score stabilizes — defined as **3 consecutive keeps where eval_score improves by less than 0.005 each**. When that happens, scale to the next tier:

| Tier | Command | Trigger |
|------|---------|---------|
| 1 | `uv run python eval_corpus/evaluate.py --subset 5` | Start here |
| 2 | `uv run python eval_corpus/evaluate.py --subset 15` | 3 consecutive keeps with delta < 0.005 on tier 1 |
| 3 | `uv run python eval_corpus/evaluate.py --subset 30` | 3 consecutive keeps with delta < 0.005 on tier 2 |
| 4 | `uv run python eval_corpus/evaluate.py --subset 50` | 3 consecutive keeps with delta < 0.005 on tier 3 |
| 5 | `uv run python eval_corpus/evaluate.py` | 3 consecutive keeps with delta < 0.005 on tier 4 (full corpus) |

When you scale up, your first run at the new tier is a new baseline. The score will likely drop (more documents = harder test). That's expected — it means there's more to improve. Log the tier in your results.tsv description (e.g. "tier2 baseline", "tier3 improve date handling").

**Scaling is one-way. Never scale back down.** Once you move to tier 2, you stay on tier 2 or higher. A regression on a larger corpus is real signal, not noise.

**Timeout**: Each evaluation run depends on corpus size and LLM API latency. Tier 1 (~5 docs) takes ~2 minutes. Tier 5 (full corpus) may take 15+ minutes. If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (a bug, malformed response, API error, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import, a KeyError from restructured data), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**LLM API failures**: The pipeline makes LLM calls (Anthropic API). Transient failures (rate limits, timeouts, 5xx errors) should be retried up to 3 times with a brief pause. If the API is persistently down, wait 2 minutes and try again. Do not give up on an experiment because of a transient API issue — only give up if the code itself is broken.

**Skip-after-N-failures**: If you crash 3 times in a row on different ideas, step back and re-read `docx_template_pipeline.py` from scratch. You may be making assumptions about the code structure that are no longer true after your edits. If you crash 5 times in a row, revert to the last known-good commit and try a fundamentally different approach.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the pipeline code for new angles, study the eval corpus documents for patterns the pipeline misses, try combining previous near-misses, try more radical prompt engineering approaches. The loop runs until the human interrupts you, period.

## What to try

The pipeline has several axes of improvement. Here are concrete directions to explore, roughly ordered from most likely to yield gains to most speculative:

### Prompt engineering — `_ANALYSIS_SYSTEM_PROMPT`
This is the single highest-leverage target. The LLM analysis prompt determines which fields get identified and how well variations are captured. Ideas:
- Add more specific rules for common document patterns (dates in different formats, addresses split across lines, phone/fax numbers)
- Improve the variation ordering instructions (longest-first is critical for replacement correctness)
- Add few-shot examples of correct field extraction
- Better instructions for distinguishing boilerplate from variable content
- More precise instructions about what counts as a "variation" vs. a separate field
- Instructions for handling fields that span multiple paragraphs or table cells

### Replacement logic — `_build_replacement_rules` and `_apply_replacements_to_docx`
The replacement engine does literal string matching. Ideas:
- Handle cases where the same text is split across multiple runs in the .docx XML
- Better handling of whitespace normalization (extra spaces, tabs, line breaks)
- Smarter conflict resolution when two replacement rules overlap
- Handle Unicode normalization issues (curly quotes vs straight quotes, en-dash vs hyphen)

### Transform types — `_build_variation_map` and `_expand_to_placeholders`
The transform system maps canonical values to their variations. Ideas:
- Add new transform types for common patterns (title case, sentence case, initials, possessive forms like "Smith's")
- Better lastname extraction logic (handle multi-word last names, hyphenated names, suffixes like Jr./III)
- Improved affix detection (currently uses simple string find)

### Fill prompt engineering — `_fill_with_llm`
The fill prompt determines how well new data gets mapped to template fields. Ideas:
- Better instructions for date format consistency
- Clearer rules for when to use `[TBD]` vs. inferring values
- Instructions for maintaining cross-field consistency (e.g. if firm name changes, address should too)

### Text extraction — `_extract_document_text`
The document text extraction determines what the LLM sees. Ideas:
- Better handling of nested tables (tables within table cells)
- Extract text from text boxes, shapes, or other non-standard elements
- Preserve more structural information (indentation, list numbering)
- Handle merged cells in tables
- Extract footnotes, endnotes, comments

### Robustness
- Better error handling for malformed .docx files
- Graceful handling of documents with no variable fields
- Handle extremely large documents (truncation strategies for LLM context limits)
- Validate LLM responses more strictly before using them

### Schema construction — `_build_fill_schema`
- Better category grouping logic
- More descriptive field descriptions in the schema
- Handle edge cases in user-supplied suffix detection

## Anti-gaming safeguards

The evaluator includes built-in defenses against metric gaming. Be aware of these — they affect your strategy:

**Code bloat penalty**: The evaluator measures `docx_template_pipeline.py` line count against the baseline. If the file grows beyond 1.5x baseline size, the score is penalized by 5%. This prevents prompt-stuffing — adding dozens of highly specific rules to `_ANALYSIS_SYSTEM_PROMPT` that happen to match the corpus but don't generalize. Prefer concise, general improvements over long, specific ones.

**Noise threshold**: Improvements of 0.005 or less are discarded (see step 8 above). LLM extraction is non-deterministic — the same code can produce slightly different scores between runs. The threshold ensures you only keep changes that produce real, measurable improvement above the noise floor.

**Holdout corpus**: The human periodically runs your best code against a separate holdout corpus you have never seen. If your eval_score goes up but the holdout score stays flat or drops, you are overfitting to the eval corpus. The human may rotate documents in and out of the eval corpus without notice. Your improvements must generalize to unseen documents.

**Corpus rotation**: The eval corpus may change between sessions. Documents can be swapped in or out by the human. Do not rely on the corpus staying the same — build general-purpose improvements, not corpus-specific hacks.

## Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.002 eval_score improvement that adds 30 lines of hacky code? Probably not worth it. A 0.002 eval_score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Example session

As an example use case, a user might leave you running overnight. If each evaluation takes ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to a `results.tsv` full of experimental results and a branch with a meaningfully improved pipeline — all completed by you while they slept.
