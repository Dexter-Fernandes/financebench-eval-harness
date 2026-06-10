# Project Story

## Current state

- The repository is an early-stage FinanceBench evaluation harness portfolio project.
- `docs/PRD.md` is the current source of product intent, methodology, milestones, and credibility boundaries.
- Handoff scaffolding now exists for future agents through `AGENTS.md`, `docs/status.md`, and this continuity log.
- No harness source code has been implemented yet.

## Latest milestone

- Completed the employer-facing PRD revision and handoff documentation setup.
- Captured the intended development path: local Ollama models first, then optional stronger hosted/frontier model comparisons through provider abstractions.

## Next step

- Implement PRD Milestone 1: load and validate the public FinanceBench JSONL sample, define experiment configuration conventions, and produce a minimal run artefact shape.

## Known risks or blockers

- Public FinanceBench sample files are not present in the repository yet.
- No Python project scaffold, dependency manager, test framework, or package layout has been selected yet.
- Model-provider choices should remain configurable until real runs are recorded.
- No eval metrics or benchmark claims exist yet.

## Safe next commands

```bash
git status --short
sed -n '1,260p' docs/PRD.md
sed -n '1,220p' docs/status.md
sed -n '1,220p' AGENTS.md
```
