---
name: run-pre-merge-checks
description: Run the live end-to-end production tests against the real NVIDIA Inference Hub backend, then produce a screenshot-ready summary that can be attached to an MR. Use this skill when asked to run pre-merge checks, run the e2e tests, validate before merging, or generate an MR test summary.
---

# Run Pre-Merge Checks

The pre-merge gate for `switchyard`. This skill runs `tests/e2e/` — the live end-to-end production tests that exercise Switchyard against a real NVIDIA Inference Hub backend — and produces a clean, screenshot-friendly summary the user can paste into an MR description.

Run this **before opening or updating an MR**, especially after touching anything under `switchyard/` or `crates/`. The test suite spawns real `switchyard` subprocesses, calls real upstream APIs, and validates the full chain (request processors, LLM backends, response processors, and translation engine) for all three inbound formats (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages).

## Parameters

Extract from the user's request. Use defaults if not specified:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_path` | `tests/e2e/` | Pytest target. Override to a single file (`tests/e2e/test_passthrough_e2e.py`) or single test (`...::TestPassthroughChatCompletions::test_basic_completion`) when iterating. |
| `extra_args` | _(none)_ | Extra pytest flags to append (e.g. `-k passthrough`, `--lf`, `-s`). |
| `also_lint` | `true` | Also run `uv run ruff check .` after the tests. CI runs this and lint failures block merge. |
| `also_typecheck` | `false` | Also run `uv run mypy switchyard`. Slower; off by default — turn on when the diff includes type changes. |
| `maxfail` | `5` | Stop the run after this many failures. We want to surface as many real regressions as possible in one pass, but bail out if everything's broken (e.g. credentials wrong, server can't bind). |

## Sandbox: must be off

**Run every Bash invocation in this skill with `dangerouslyDisableSandbox: true`.** The skill needs write access to:

- `/tmp/pre-merge-checks/` (log directory — outside the default sandbox `/tmp/claude` allowlist)
- `~/.cache/uv/` (uv's package cache — `uv run` cannot resolve dependencies without it)
- `secrets/secrets.json` (read-only, but on some sandbox configs even reads from arbitrary paths under the repo are mediated)
- the git working tree at `$REPO_ROOT` (test subprocesses spawn from here and write `.pytest_cache/` etc.)

The first time you call Bash without disabling the sandbox you will see errors like `mkdir: /tmp/pre-merge-checks: Operation not permitted` and `failed to open file /Users/linj/.cache/uv/...: Operation not permitted` — that's the signal. Don't try to work around it with `$TMPDIR` or by relocating the uv cache; just disable the sandbox for this skill's commands. The user can manage permanent allowlist entries via `/sandbox` if they want to avoid the per-call prompts.

## What this skill runs (and why)

The e2e suite has multiple test files. Each spawns its own (or shared) subprocess server and hits the real NVIDIA inference API:

| File | What it covers | Why it matters for merge |
|------|---------------|--------------------------|
| `test_passthrough_e2e.py` | OpenAI Chat Completions inbound → OpenAI backend → real API | Validates the most common production path. |
| `test_passthrough_responses_e2e.py` | OpenAI Responses API inbound → translation → Chat Completions backend | Validates the Responses-to-Chat translation engine end-to-end. |
| `test_random_routing_llm_backend.py` | Random-routing processor plus Rust `MultiLlmBackend` dispatch across OpenAI-compatible and Anthropic-native tiers | Validates the random-routing profile path used by TerminalBench / SWE-bench experiments. |
| `test_latency_service_llm_backend.py` | `LatencyServiceLLMBackend` health-aware multi-endpoint routing | Validates the health-poller usage case. |
| `test_verify_e2e.py` | The `switchyard verify` + `switchyard launch {claude,codex} --smoke` CLIs | Validates the user-facing post-clone smoke test. Proxy-only mode shells `switchyard verify`; harness modes shell `switchyard launch <target> --smoke`. Sub-tests skip if `claude` / `codex` CLIs aren't installed; that is expected, not a failure. |

These are **session-scoped subprocess fixtures** — running them costs real upstream tokens. Expect ~1–3 minutes wall time and a few cents of API spend per full run.

## Prerequisites

These need to be in place before the skill can do anything useful. Check all of them up front and fail loud if any are missing.

### 1. Repo root + venv

The skill must run from the Switchyard repo root regardless of where the user has it cloned. Resolve it dynamically — never hardcode a path:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "NOT_IN_GIT_REPO — run this skill from inside a Switchyard checkout"; exit 1
fi
cd "$REPO_ROOT"

# Sanity-check we're actually in Switchyard, not some other repo
if [ ! -d "switchyard" ] || [ ! -d "tests/e2e" ]; then
  echo "WRONG_REPO — $REPO_ROOT does not look like Switchyard"; exit 1
fi

test -d .venv || { echo "MISSING_VENV — run ./setup.sh --dev first"; exit 1; }
```

All subsequent commands in this skill assume pwd is `$REPO_ROOT`. Relative paths (`tests/...`, `secrets/...`, `switchyard/...`) are intentional — they make the skill portable across clones.

### 2. Credentials — NVIDIA API key

The fixtures resolve credentials in this order:

1. `NVIDIA_API_KEY` env var
2. `secrets/secrets.json` (already gitignored)

Check both:

```bash
if [ -n "$NVIDIA_API_KEY" ]; then
  echo "OK: NVIDIA_API_KEY set in env"
elif [ -f secrets/secrets.json ] && grep -q '"nvidia"' secrets/secrets.json; then
  echo "OK: secrets/secrets.json has nvidia section"
else
  echo "MISSING — set NVIDIA_API_KEY or copy secrets/secrets.template.json to secrets/secrets.json and fill in"
  exit 1
fi
```

If neither is present, stop and tell the user:

> `NVIDIA_API_KEY` is required for the e2e tests. Either `export NVIDIA_API_KEY=...` or copy `secrets/secrets.template.json` to `secrets/secrets.json` and fill in the `nvidia` section. Re-run after that.

### 3. Working tree status

Capture the branch and HEAD so the summary is reproducible:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
git status --porcelain | wc -l   # number of dirty files
```

## Workflow

### Step 1: Verify prerequisites

Run the three checks from "Prerequisites" above in parallel via the Bash tool. Report any failure and stop. Print the branch/HEAD/dirty-file count — these go into the summary.

### Step 2: Run the e2e tests

Use `uv run pytest` so the venv resolves automatically. Pipe the output through `tee` so you have both a live transcript and a file the skill can grep at the end.

**Override the project's default `-x` (set in `pyproject.toml`'s `addopts`).** The pre-merge check should surface as many real failures as possible in one run, not stop at the first one — otherwise reviewers can't tell whether downstream tests regressed too. Use `-o addopts= --maxfail={maxfail}` to keep going up to the failure budget.

```bash
mkdir -p /tmp/pre-merge-checks
LOG=/tmp/pre-merge-checks/v2-e2e-$(date +%Y%m%d-%H%M%S).log
uv run pytest {test_path} -v -o addopts= --maxfail={maxfail} {extra_args} 2>&1 | tee "$LOG"
echo "EXIT_CODE=${PIPESTATUS[0]}"
```

Important notes:
- **Do not run the tests in the background.** They take 1–10 min (full suite) and the skill needs the exit code. Use Bash's normal blocking mode with a generous timeout (`timeout: 900000` = 15 min covers a full no-`-x` run with retries).
- **Run with `dangerouslyDisableSandbox: true`** — see the "Sandbox: must be off" section above. The first attempt without will fail at `mkdir /tmp/pre-merge-checks`.
- If pytest reports `skipped`, that is normal — `test_verify_e2e.py` skips Claude / Codex sub-tests when those CLIs aren't installed.

#### Detect incomplete runs

If `--maxfail` was hit, pytest stops early and the suite is **incomplete**. Detect and surface this loudly — both in the terminal and in the final summary. A green-looking partial run is the worst possible outcome.

```bash
COLLECTED=$(grep -oE "collected [0-9]+ items?" "$LOG" | head -1 | grep -oE "[0-9]+")
# Pull the final summary line, e.g. "5 failed, 30 passed, 2 skipped in 412.18s"
SUMMARY=$(grep -E "= [0-9]+ (passed|failed)" "$LOG" | tail -1)
PASSED=$(echo "$SUMMARY" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" || echo 0)
FAILED=$(echo "$SUMMARY" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+" || echo 0)
SKIPPED=$(echo "$SUMMARY" | grep -oE "[0-9]+ skipped" | grep -oE "[0-9]+" || echo 0)
ERRORS=$(echo "$SUMMARY" | grep -oE "[0-9]+ error" | grep -oE "[0-9]+" || echo 0)
RAN=$((PASSED + FAILED + SKIPPED + ERRORS))

if [ "$RAN" -lt "$COLLECTED" ]; then
  NOT_RUN=$((COLLECTED - RAN))
  echo ""
  echo "================================================================"
  echo "⚠️  TEST RUN INCOMPLETE"
  echo "    Stopped after $FAILED failure(s) — reached --maxfail={maxfail} threshold."
  echo "    $NOT_RUN of $COLLECTED tests did NOT execute."
  echo "    Fix the failing tests and re-run before treating this as a clean signal."
  echo "================================================================"
fi
```

When `RAN < COLLECTED`, the final summary's `Status` line must be `❌ FAIL — INCOMPLETE` (not just `❌ FAIL`), and the markdown summary must include the `not run` count so the MR reviewer immediately sees that some tests were never reached.

### Step 3: (Optional) Run lint

If `also_lint` is true (the default):

```bash
uv run ruff check . 2>&1 | tee /tmp/pre-merge-checks/ruff-$(date +%Y%m%d-%H%M%S).log
echo "RUFF_EXIT=${PIPESTATUS[0]}"
```

Lint failures block CI and are an embarrassing-but-cheap gate to catch before push. AGENTS.md says: "must pass with zero errors before any commit or push."

### Step 4: (Optional) Run mypy

If `also_typecheck` is true:

```bash
uv run mypy switchyard 2>&1 | tee /tmp/pre-merge-checks/mypy-$(date +%Y%m%d-%H%M%S).log
echo "MYPY_EXIT=${PIPESTATUS[0]}"
```

mypy runs in `strict` mode per `pyproject.toml`. It's slower than ruff (~30s+); only run when the user explicitly asks or the diff has substantial type changes.

### Step 5: Parse results

Extract the summary line pytest writes at the end. Examples:

```
================== 27 passed, 3 skipped in 142.44s (0:02:22) ==================
================ 25 passed, 1 failed, 4 skipped in 138.91s ===================
```

Pull these counts (passed / failed / skipped / errors / wall time) from the log. If there are failures, also pull the names of the failed tests — they live above the summary in the `FAILED` lines:

```bash
grep -E "^FAILED " "$LOG"
```

### Step 6: Generate the screenshot-ready summary

This is the deliverable. The user is going to screenshot it and paste into the MR. Format it as a single Markdown block in the chat — no surrounding chatter, no follow-up questions, no "Let me know if…". Just the block. Headers, tables, and a clean status line at the top.

Use this exact template (fill in the `{}` placeholders):

```markdown
## Pre-merge checks — {branch} @ {short_sha}{dirty_marker}

**Status: {✅ PASS | ❌ FAIL | ⛔ FAIL — INCOMPLETE}**  ·  {date_iso}  ·  {wall_time}

### Live e2e production tests — `tests/e2e/`

| Result | Count |
|--------|------:|
| Passed | {n_passed} |
| Failed | {n_failed} |
| Skipped | {n_skipped} |
| Errors | {n_errors} |
| **Not run** | **{n_not_run}** |

{if n_not_run > 0:}
> ⚠️  **Run incomplete.** Stopped after {n_failed} failures (--maxfail={maxfail}). {n_not_run} of {n_collected} tests did not execute. Fix the failures and re-run before treating this as a clean signal.
{/if}

{if n_failed > 0:}
**Failed tests:**
{for each failed test:}
- `{test_node_id}`
{/for}
{/if}

### Lint — `uv run ruff check .`

{✅ clean | ❌ {N} errors — see log}

### Type check — `uv run mypy switchyard`

{✅ clean | ❌ {N} errors — see log | ⏭ skipped}

---

<sub>Generated by `run-pre-merge-checks` skill · log: `{log_path}`</sub>
```

Notes for filling in:
- `{dirty_marker}` is `+dirty` if `git status --porcelain` had any output, otherwise empty.
- `{date_iso}` is `date -u +"%Y-%m-%dT%H:%MZ"`.
- `{wall_time}` is the `in 142.44s` portion of the pytest summary, formatted as `2m 22s` or `38s`.
- `{n_collected}` comes from the pytest `collected N items` line; `{n_not_run} = n_collected − (passed + failed + skipped + errors)`.
- The top-level `Status` is:
  - `⛔ FAIL — INCOMPLETE` if `n_not_run > 0` (--maxfail was hit). This is the worst signal — surface it loudly.
  - `❌ FAIL` if pytest, ruff, or mypy returned non-zero but every collected test ran.
  - `✅ PASS` only when every gate is green AND every collected test ran.
- The "Not run" table row and the warning callout are omitted entirely when `n_not_run == 0` (don't render empty rows).
- The "Failed tests" section is omitted entirely when `n_failed == 0`.
- The mypy row is omitted entirely when `also_typecheck=false` (don't render `⏭ skipped` unless the user explicitly asked for typecheck and we couldn't run it).

### Step 7: After the summary

Below the Markdown block, in plain text (one short paragraph), tell the user:
1. Where the full log is (`/tmp/pre-merge-checks/...`).
2. If anything failed, the most likely failure mode based on the failed test names — see "Failure triage" below.
3. Nothing else. No "let me know" / "should I retry" / "anything else" — the user knows what to do next.

## Failure triage

When a test fails, surface the right next step instead of dumping the full traceback into chat. Common patterns:

| Failure pattern | Likely cause | What to tell the user |
|-----------------|--------------|------------------------|
| `pytest.skip("NVIDIA API key not found...")` on every test | Credentials not picked up despite Step 1 passing | Confirm `secrets/secrets.json` has a `nvidia` section with `api_key` *and* `base_url` *and* `model`. Env var alone is not enough — fixtures also need the model name. |
| `passthrough server failed to start within 60s` | Subprocess crashed before binding the port | Tail the captured stderr from the test output; usually a `ModuleNotFoundError` (stale install) or a port conflict. Suggest `uv sync --dev` then retry. |
| `httpx.HTTPStatusError: 401` from any test | API key valid but rejected by NVIDIA hub | Either the key was revoked, or the `base_url` in `secrets/secrets.json` doesn't match the key's tenant. |
| `assert response.choices[0].message.content` is empty | Upstream model returned empty body — flaky inference, not a code bug | Suggest re-running just that test (`pytest <node_id> -v`); if it reproduces twice, dig in. |
| `test_verify_e2e.py::TestVerifyClaudeE2E` skipped | `claude` CLI not installed in this env | Expected. Not a failure. Mention this in the plain-text addendum so the user knows skips ≠ regressions. |
| Anthropic translation tests fail with `KeyError: 'role'` or similar | Bug in `foundation/translation/anthropic_openai.py` from the user's diff | Surface the failed test names; this is a real bug to fix. |
| `RuntimeError: Event loop is closed` | Pytest-asyncio teardown race with subprocess fixture | Re-run; if it persists, file an issue. Not a real failure of the user's code. |

## Iteration helpers

If the user says "run again, just the failed ones":
```bash
uv run pytest --lf -v
```

If the user says "run with print output":
```bash
uv run pytest tests/e2e/ -v -s
```

If the user asks why a particular test failed (after a failed run):
```bash
grep -A 60 "FAILED {test_node_id}" "$LOG"
```

## Boundaries

### Always do

- Treat `tests/e2e/` as the canonical live pre-merge gate. If the user wants something narrower, accept a `test_path` override but still default to the full suite.
- Block the run on missing credentials. Don't proceed to pytest if `NVIDIA_API_KEY` and `secrets/secrets.json` are both absent — pytest will skip *every* test, producing a misleading green summary.
- Run `ruff check .` alongside the tests by default. CI runs it and the user will catch lint failures here before push.
- Output the summary as a single self-contained Markdown block so the user can screenshot it without editing.

### Ask first

- Running anything more expensive than this skill, such as the full benchmark suite or ad hoc live provider suites outside `tests/e2e/`.
- Modifying `secrets/secrets.json`, even to fill in obvious defaults — it's gitignored for a reason.

### Never do

- Run broader live or benchmark suites by default — those may need different credentials and take much longer.
- Suppress test failures or downgrade the summary to "PASS" because the failures look "minor." If pytest's exit code is non-zero, the summary says FAIL.
- Commit, push, or open a PR from inside this skill. The user explicitly invokes this *to inform their MR*; the MR ceremony is theirs.
