# Project Story

## Current state

- M7 implementation complete on branch `M7` (6 commits ahead of main, PR not yet raised).
- 885 tests pass, 2 skipped (Ollama integration tests). 50 new M7 tests added.
- `analyze-grounding` and `inspect-failure` CLI commands fully operational.
- No live grounding analysis run recorded yet — all tests use mock data.

## Latest milestone

**M7: Hallucination and Grounding Analysis** — all 15 sub-milestones complete.

Key deliverables:

- `grounding_types.py` — M7 taxonomy: 5 label tuples (GROUNDING_LABELS, HALLUCINATION_FAILURE_TYPES, ROOT_CAUSE_LABELS, CITATION_SUPPORT_LABELS, CONTEXT_SUFFICIENCY_LABELS) + 3 dataclasses
- `scoring.py` — added `extract_fiscal_periods()` for FY/quarter string detection
- `claim_extraction.py` — `extract_claims()` (sentence-split + classify), `extract_cited_chunk_ids()` (regex `[chunk_id: …]`)
- `context_sufficiency.py` — `check_context_sufficiency()` using evidence_hit + numeric overlap
- `citation_checker.py` — `score_citations()` validating cited IDs against retrieved set
- `hallucination_checks.py` — 5 rule-based flags (no_citation, cited_chunk_not_retrieved, predicted_number_not_in_context, predicted_year_not_in_context, refusal_with_evidence)
- `judge.py` — v2 grounding judge: `render_grounding_prompt_v2()`, `parse_grounding_response_v2()`, `M7_GROUNDING_LABELS`
- `prompts/judges/answer_grounding_v2.txt` — richer M7 judge prompt (failure_types list, context_sufficiency, citation_quality)
- `analysis.py` — added `classify_root_cause()` (6-case deterministic hierarchy)
- `grounding_analysis_config.py` + `configs/grounding_analysis.yaml` — config loader for `analyze-grounding`
- `grounding_analysis.py` — `analyze_grounding()` orchestrator, `join_all_signals()` (M7.10), failure summarizer (M7.14)
- `hallucination_report.py` — 10-section Markdown report with failure slicing by company/question_type
- `failure_inspector.py` — `load_failure_inspection()` + `format_failure_inspection()` for single-question drill-down
- CLI: `analyze-grounding --config … [--report-dir …] [--run-id …]` and `inspect-failure --run … --question-id …`
- `tests/test_hallucination_analysis.py` — 50 tests, all passing

## Next step

1. **Raise and merge M7 PR** — branch is clean, all tests pass.
2. **Run actual grounding analysis** — point `configs/grounding_analysis.yaml` at a completed RAG run dir and execute:
   ```bash
   python -m financebench_eval analyze-grounding --config configs/grounding_analysis.yaml
   ```
3. **Interpret failure report** — check root cause breakdown (retrieval_failure vs generation_failure rates).
4. **Enable LLM judge** — set `grounding_judge.enabled: true` in the config and set `run_dir` to a real RAG run to get richer per-example grounding labels.
5. **M8 (if planned)** — use grounding results to drive prompt or retrieval improvements.

## Known risks or blockers

- `configs/grounding_analysis.yaml` has `run_dir: runs/placeholder_run_id` — must be updated to a real run dir before use.
- The v2 grounding judge (`answer_grounding_v2.txt`) is written but not yet exercised against a live LLM. Parser validates the schema strictly; malformed LLM output raises `JudgeError` and is logged as an error row.
- `classify_root_cause()` uses `grounding_label == "under_refusal"` as the primary signal for `hallucination_under_refusal`, which takes priority over `evidence_hit=False`. This is intentional but means the grounding judge must correctly label refusal cases for that branch to activate.
- `hallucination_checks.py` number-in-context check is string-based (converts float to str and does substring match). Works for clean numeric strings; may miss formatted numbers like "1,577" if the prediction stores "1577".

## Safe next commands

```bash
# Confirm tests still pass
pytest -q

# Raise M7 PR
git push origin M7
gh pr create --title "feat(M7): hallucination and grounding analysis" --body "..."

# Dry-run grounding analysis on an existing run dir
python -m financebench_eval analyze-grounding \
  --config configs/grounding_analysis.yaml

# Inspect a specific question after analysis
python -m financebench_eval inspect-failure \
  --run runs/<run_id> \
  --question-id financebench_001
```
