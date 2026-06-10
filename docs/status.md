# Project Status

## Current State

- The repository is in planning and documentation setup.
- `docs/PRD.md` defines the intended FinanceBench evaluation harness, success criteria, milestones, and data boundaries.
- No source code, CLI, dataset loader, model provider integration, eval runner, or report generator has been implemented yet.

## Intended Build Path

- Start with local Ollama models for development and debugging.
- Add hosted/frontier model support later through provider abstractions once the local harness is reproducible.
- Keep exact provider and model names configurable so comparisons can use whichever models are available at evaluation time.

## Next Engineering Milestone

- Implement Milestone 1 from the PRD: dataset loading, JSONL validation, experiment configuration conventions, and a minimal run artefact shape.

## Current Boundaries

- Use only public FinanceBench sample data.
- Keep local datasets, PDFs, vector stores, model caches, secrets, and generated run artefacts out of version control.
- Do not publish benchmark claims or metrics until they are produced by reproducible runs.
