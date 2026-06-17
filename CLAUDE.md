@AGENTS.md

## Claude Code

For M3, implement standalone retrieval only.
Do not wire retrieval into run-eval yet.
Keep generated indexes, run outputs, model caches, and API keys out of git.
Run relevant pytest smoke tests before committing.

## M6 Workflow

Follow `docs/M7_spec.md` for all M6 implementation work.

Invoke `/tdd` at the very start before any implementation begins.

After completing each M7.x sub-milestone:
1. Stage relevant changed files with `git add`
2. Invoke `/git-pre-commit` to generate the commit message
3. Use only the first line of the output (the commit subject — not the body or description)
4. Commit with that subject line
