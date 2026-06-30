## What

Short summary of what this PR changes.

## Why

The motivation — what problem does this solve, or which ticket does it close?

Closes #

## How tested

- [ ] `uv run ruff check .` clean
- [ ] `uv run mypy switchyard` clean
- [ ] `uv run pytest tests/` green
- [ ] Manual smoke (describe what was run)

## Checklist

- [ ] One class per file; filename = `snake_case` of the primary class.
- [ ] New public symbols exported from `switchyard/__init__.py.__all__` if intended for downstream use.
- [ ] Unit tests added for new components / bug fixes.
- [ ] README / `--help` updated if customer-facing surface changed.
- [ ] Commits signed off (`Signed-off-by: Your Name <email>`) per the DCO.

## Notes for reviewers

Anything reviewers should pay extra attention to — risky paths, follow-up tickets, intentional trade-offs.
