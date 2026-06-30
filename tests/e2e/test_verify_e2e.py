# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end production gate for the smoke-test CLI surface.

Shells out to the same CLI a fresh user runs after cloning the repo,
so the test and the user-facing UX share a single source of truth.
Each test invokes one of:

* ``switchyard verify ...`` (proxy + backend; no harness binary)
* ``switchyard launch claude --smoke ...`` (full claude e2e)
* ``switchyard launch codex --smoke ...`` (full codex e2e)

…as a subprocess, captures its stdout/stderr, and asserts
``returncode == 0`` plus the final ``verify <mode>: PASS`` summary
line is present. (The ``--smoke`` flag dispatches to the same
:func:`verify_claude` / :func:`verify_codex` helpers the standalone
``verify claude`` / ``verify codex`` subcommands used to drive, so
the summary line wording is preserved.)

Three mode-specific test classes:

* :class:`TestVerifyProxyE2E` — runs unconditionally when secrets are
  available.  No harness binary required.  Cheapest gate.
* :class:`TestVerifyClaudeE2E` — gated on ``claude --version`` being
  installable; ``pytest.skip`` otherwise.
* :class:`TestVerifyCodexE2E` — gated on ``codex --version``; same
  skip pattern.

Why subprocess-out instead of calling ``verify_proxy()`` directly:

1. The subprocess captures the user-facing UX exactly — flag parsing,
   credential resolution, exit codes, output formatting.  An in-process
   call would test the implementation, not the CLI.
2. Spawning ``claude`` / ``codex`` from inside pytest's own process
   would fight with the surrounding pytest event loop (the verifier
   hosts uvicorn in a thread, and pytest-asyncio's autouse loop can
   interfere with subprocess teardown timing).  A subprocess shell-out
   avoids both classes of cross-talk.

Run with::

    OPENROUTER_API_KEY=sk-or-... pytest tests/e2e/test_verify_e2e.py -v
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest

from .conftest import REPO_ROOT

pytestmark = pytest.mark.integration

logger = logging.getLogger("e2e.verify")

# Per-mode budget for the entire ``verify`` subprocess.  Step-level
# timeouts inside ``verify`` (60s per harness invocation, 60s round-
# trip, 10s per probe) sum to ~3 minutes worst case; 240s gives slack
# for subprocess startup + Python's own venv-resolution latency.
_VERIFY_TIMEOUT_S = 240.0


def _run_verify(
    mode: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Shell out to the smoke-test CLI for ``mode``.

    Modes dispatch to:

    * ``"proxy"``  → ``switchyard verify ...``
    * ``"claude"`` → ``switchyard launch claude --smoke ...``
    * ``"codex"``  → ``switchyard launch codex  --smoke ...``

    All three end up running the same orchestrators in
    ``switchyard.server.verify`` (``verify_proxy`` / ``verify_claude``
    / ``verify_codex``) and emit the same ``verify <mode>: PASS`` /
    ``FAIL`` summary line; the CLI shape is the only thing that
    changed in the unification.

    Invokes via ``python -m`` (not ``.venv/bin/switchyard``) to match
    the convention :mod:`tests.e2e.conftest` uses — keeps the tests
    independent of whether the editable-install script shim has been
    regenerated since the last package install.
    """
    base_cmd = [sys.executable, "-m", "switchyard.cli.switchyard_cli"]
    if mode == "proxy":
        verb_args = ["verify"]
    elif mode == "claude":
        verb_args = ["launch", "claude", "--smoke"]
    elif mode == "codex":
        verb_args = ["launch", "codex", "--smoke"]
    else:
        raise ValueError(f"unsupported smoke mode: {mode!r}")
    cmd = [
        *base_cmd,
        *verb_args,
        "--model", model,
        "--api-key", api_key,
        "--base-url", base_url,
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    # Belt-and-braces: the verify orchestrator also resolves from env
    # vars, but we set OPENAI_API_KEY here so any deeper subprocess
    # (e.g. the codex CLI shelled out by ``launch codex --smoke``)
    # finds it without requiring the same waterfall.
    env.setdefault("OPENAI_API_KEY", api_key)

    logger.info("[Verify] Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_VERIFY_TIMEOUT_S,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )


def _assert_pass(result: subprocess.CompletedProcess, mode: str) -> None:
    """Common pass assertion: rc=0 + a ``verify <mode>: PASS`` line.

    Both checks are intentional.  Exit 0 alone could be a parse error
    that exited cleanly without ever reaching the runner; the summary
    line is the only positive signal that the orchestrator ran end-to-
    end.  Pinning both gives loud failures with surfaceable output —
    the assertion message dumps stdout + stderr so CI logs are
    diagnosable without re-running the test.
    """
    assert result.returncode == 0, (
        f"verify {mode} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )
    assert f"verify {mode}: PASS" in result.stdout, (
        f"missing 'verify {mode}: PASS' summary line\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )


# ---------------------------------------------------------------------------
# verify proxy — backend + chain only, no harness binary
# ---------------------------------------------------------------------------


class TestVerifyProxyE2E:
    """The cheapest gate — runs unconditionally when secrets are available.

    Smoke-tests the verify CLI's six steps end-to-end against the selected
    live backend: credentials, /models reachability, /v1/messages probe,
    proxy spin-up, chat-completion round-trip, teardown.

    A failure here means *something* in the chain is broken (or the
    upstream is down) — useful tripwire even before we get to the
    harness-binary tests.
    """

    def test_verify_proxy_passes(self, nvidia_config: dict) -> None:
        result = _run_verify(
            "proxy",
            api_key=nvidia_config["api_key"],
            base_url=nvidia_config["base_url"],
            model=nvidia_config["model"],
        )
        _assert_pass(result, "proxy")

        # All six steps must report — count check guards against the
        # runner short-circuiting silently after a refactor.
        for step_marker in ("[1/6]", "[2/6]", "[3/6]", "[4/6]", "[5/6]", "[6/6]"):
            assert step_marker in result.stdout, (
                f"missing step header {step_marker}\n--- stdout ---\n{result.stdout}"
            )

    def test_verify_proxy_fails_loud_on_bad_credentials(
        self, nvidia_config: dict,
    ) -> None:
        """Negative test: a bogus key must fail at step 2 (Reach backend),
        not silently pass — this is the contract that lets
        ``returncode == 0`` mean "everything is wired correctly".

        Picks an obviously-invalid key shape so the test doesn't
        depend on rate-limit behavior of any specific upstream.
        """
        result = _run_verify(
            "proxy",
            api_key="sk-deliberately-invalid-credential-for-test",
            base_url=nvidia_config["base_url"],
            model=nvidia_config["model"],
        )
        assert result.returncode != 0, (
            f"verify proxy must fail with a bad key; got rc=0\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
        assert "verify proxy: FAIL" in result.stdout, (
            f"missing FAIL summary line\n"
            f"--- stdout ---\n{result.stdout}\n"
        )


# ---------------------------------------------------------------------------
# verify claude — full e2e through Claude Code's CLI
# ---------------------------------------------------------------------------


def _is_claude_cli_available() -> bool:
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestVerifyClaudeE2E:
    """Runs the full ``launch claude``-equivalent path: proxy + spawn
    ``claude -p "..."``.  Skipped when ``claude`` isn't installed.
    """

    def test_verify_claude_passes(self, nvidia_config: dict) -> None:
        if not _is_claude_cli_available():
            pytest.skip(
                "claude CLI not installed; install with "
                "`curl -fsSL https://claude.ai/install.sh | bash`"
            )

        result = _run_verify(
            "claude",
            api_key=nvidia_config["api_key"],
            base_url=nvidia_config["base_url"],
            model=nvidia_config["model"],
        )
        _assert_pass(result, "claude")

        # All eight steps — pin the count to catch drift between
        # claude (8 steps with the Anthropic probe) and codex (7 steps).
        for step_marker in (
            "[1/8]", "[2/8]", "[3/8]", "[4/8]", "[5/8]", "[6/8]", "[7/8]", "[8/8]",
        ):
            assert step_marker in result.stdout, (
                f"missing step header {step_marker}\n--- stdout ---\n{result.stdout}"
            )

        # Spawn step-specific check: the ``Spawning claude`` step must
        # have appeared with a reply preview, not just OK.
        assert "Spawning claude" in result.stdout
        assert "reply=" in result.stdout, (
            f"missing reply preview from claude\n--- stdout ---\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# verify codex — full e2e through OpenAI Codex CLI
# ---------------------------------------------------------------------------


def _is_codex_cli_available() -> bool:
    """Mirror of :func:`_is_claude_cli_available` for codex."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestVerifyCodexE2E:
    """Runs the full ``launch codex``-equivalent path: proxy + spawn
    ``codex exec "..."``.  Skipped when ``codex`` isn't installed.
    """

    # Pinned model — overrides whatever provider model env is set to for
    # this test only.  Codex sends a Responses API request whose
    # ``input`` items can place a ``role: "system"`` message after
    # the first user turn; strict providers (notably
    # ``nvidia/qwen/qwen3.5-35b-a3b`` on the NVIDIA Inference Hub)
    # reject that with HTTP 400 ``"System message must be at the
    # beginning."``  Nemotron Super v3 accepts it, so we hard-pin to
    # that model so this test validates the codex e2e path
    # independently of whatever the local default model happens to be.
    _CODEX_TEST_MODEL = "nvidia/nvidia/nemotron-3-super-v3"

    def test_verify_codex_passes(self, nvidia_config: dict) -> None:
        if not _is_codex_cli_available():
            pytest.skip(
                "codex CLI not installed; install with "
                "`npm install -g @openai/codex`"
            )

        result = _run_verify(
            "codex",
            api_key=nvidia_config["api_key"],
            base_url=nvidia_config["base_url"],
            model=self._CODEX_TEST_MODEL,
        )
        _assert_pass(result, "codex")

        # Codex skips the Anthropic probe — pin count at 7 to catch
        # drift toward claude's 8-step shape.
        for step_marker in (
            "[1/7]", "[2/7]", "[3/7]", "[4/7]", "[5/7]", "[6/7]", "[7/7]",
        ):
            assert step_marker in result.stdout, (
                f"missing step header {step_marker}\n--- stdout ---\n{result.stdout}"
            )
        assert "[8/" not in result.stdout, (
            "verify codex must run 7 steps, not 8 — Anthropic probe "
            "should be skipped for the codex path."
        )

        assert "Spawning codex" in result.stdout
        assert "reply=" in result.stdout
