# Agent Instructions

## Project Context

- This repository is an early-stage portfolio project for `financebench-eval-harness`.
- The product intent lives in `docs/PRD.md`; treat that file as the source of truth for scope, evaluation goals, and credibility boundaries.
- The project should demonstrate applied AI research-engineering judgement: repeatable evaluation, hypothesis-driven experiments, failure analysis, and clear communication to product and engineering stakeholders.
- Keep employer-facing language professional and broadly applicable. Do not name a specific target employer in repo docs unless the user explicitly asks.

## Current Build Direction

- Build locally first with Ollama-backed models so the harness can be developed and debugged without paid hosted inference.
- Once the local harness is reliable, add provider abstractions for stronger hosted/frontier models such as Claude, GPT/ChatGPT, or other APIs available at evaluation time.
- Avoid hardcoding future model names in durable docs or code unless they are actually used in a recorded run.
- Prefer configurable providers, model names, prompts, retrieval settings, and evaluation settings.

## Data and Artefact Rules

- Use only public FinanceBench sample data and public or locally available source documents.
- Do not commit private, confidential, proprietary, or licensed documents.
- Do not commit API keys, `.env` files, local model caches, vector stores, raw run outputs, or large generated artefacts.
- Raw/generated eval outputs should remain local unless they are intentionally curated into small, reviewable documentation.
- Do not claim benchmark results unless the repository contains enough reproducible evidence to support them.

## Working Practices

- Keep changes small and aligned with the PRD milestones.
- For code work, prefer Python tooling and clear module boundaries for data loading, retrieval, model providers, evaluation, and reporting.
- Separate retrieval failures from generation, reasoning, grounding, refusal, and evaluation ambiguity.
- Record model provider, model name, prompt version, dataset slice, and evaluation settings for every experiment.
- When updating handoff docs, update `story.md` at the repository root.

## Useful Files

- `docs/PRD.md`: product requirements and evaluation methodology.
- `docs/status.md`: concise human-readable status snapshot.
- `story.md`: continuity log for future sessions.
