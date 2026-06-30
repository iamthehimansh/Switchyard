# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for switchyard end-to-end production tests.

These tests hit a real OpenAI-compatible backend (OpenRouter by default,
but any compatible URL works via env vars) through a subprocess-launched
``switchyard`` CLI. They're gated on an API key env var being set; without
one every test in this directory skips.

Configuration (env vars, in resolution order):

* ``OPENROUTER_API_KEY`` / ``NVIDIA_API_KEY`` — required; the test suite
  skips without one
* ``OPENROUTER_BASE_URL`` / ``NVIDIA_BASE_URL`` — defaults to the selected
  provider's OpenAI-compatible base URL
* ``OPENROUTER_MODEL`` / ``NVIDIA_MODEL`` — defaults to the selected provider's
  GPT-5.2 model id

Run with::

    OPENROUTER_API_KEY=sk-or-... uv run pytest tests/e2e/ -v
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("e2e")

REPO_ROOT = Path(__file__).parent.parent.parent

SERVER_STARTUP_TIMEOUT = 60.0


def get_nvidia_config() -> dict:
    """Resolve backend configuration from env vars.

    Returns a dict with ``api_key``, ``base_url``, and ``model``.
    ``api_key`` is ``None`` when nothing's set — fixtures that depend
    on it call ``pytest.skip``.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "provider": "openrouter",
            "api_key": openrouter_key,
            "base_url": (
                os.environ.get("OPENROUTER_BASE_URL")
                or "https://openrouter.ai/api/v1"
            ),
            "model": os.environ.get("OPENROUTER_MODEL") or "openai/gpt-5.2",
        }

    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if nvidia_key:
        return {
            "provider": "nvidia",
            "api_key": nvidia_key,
            "base_url": (
                os.environ.get("NVIDIA_BASE_URL")
                or "https://inference-api.nvidia.com/v1"
            ),
            "model": os.environ.get("NVIDIA_MODEL") or "openai/openai/gpt-5.2",
        }

    return {
        "provider": "openrouter",
        "api_key": None,
        "base_url": (
            os.environ.get("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        ),
        "model": os.environ.get("OPENROUTER_MODEL") or "openai/gpt-5.2",
    }


def find_free_port() -> int:
    """Bind a socket to port 0 to claim a free port, then close it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 30.0, server_name: str = "server") -> bool:
    """Poll ``127.0.0.1:port`` until it accepts TCP connections, or timeout."""
    start_time = time.time()
    last_log_time = start_time
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = time.time() - start_time
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(("127.0.0.1", port))
                logger.info(f"  {server_name} ready after {elapsed:.1f}s (attempt {attempt})")
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            if time.time() - last_log_time >= 5.0:
                logger.info(
                    f"  Still waiting for {server_name}... "
                    f"({elapsed:.1f}s elapsed, attempt {attempt})"
                )
                last_log_time = time.time()
            time.sleep(0.5)

    logger.warning(f"  {server_name} failed to start after {timeout:.1f}s ({attempt} attempts)")
    return False


def stop_server_subprocess(proc: subprocess.Popen, kill_timeout: float = 5.0) -> None:
    """Gracefully terminate a subprocess, force-killing if it hangs."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        logger.warning("[Subprocess] Did not stop gracefully, killing...")
        proc.kill()
        proc.wait()


@pytest.fixture(scope="session")
def nvidia_config() -> dict:
    """Backend configuration shared across the e2e suite.

    Skips the entire dependent test if no backend API key is set —
    we don't want silent successes from a no-op backend.
    """
    config = get_nvidia_config()
    if not config["api_key"]:
        pytest.skip("OPENROUTER_API_KEY or NVIDIA_API_KEY not set — required for e2e tests")
    return config


def _start_passthrough_server(
    port: int,
    api_key: str,
    base_url: str,
) -> subprocess.Popen:
    """Launch ``switchyard passthrough`` as a subprocess.

    Invoked via ``python -m switchyard.cli.switchyard_cli`` (not
    ``.venv/bin/switchyard``) so the tests are independent of whether
    the editable-install script shim has been regenerated since the
    last package install.

    Uses ``--inbound both`` so the one server exposes all three
    inbound formats (``/v1/chat/completions``, ``/v1/responses``,
    ``/v1/messages``) simultaneously — the Chat Completions, Responses,
    and Anthropic Messages tests share this one process.
    """
    cmd = [
        sys.executable,
        "-m", "switchyard.cli.switchyard_cli",
        "passthrough",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--inbound", "both",
        "--api-key", api_key,
        "--base-url", base_url,
    ]

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", api_key)

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
        env=env,
    )


@pytest.fixture(scope="session")
def passthrough_server(nvidia_config: dict) -> Generator[dict, None, None]:
    """Start a real switchyard passthrough server on a free port.

    Session-scoped: every e2e file (Chat Completions, Responses,
    Anthropic Messages) shares one running server, saving subprocess
    startup overhead.

    Yields a dict with ``process``, ``port``, ``base_url``, ``model``.
    """
    port = find_free_port()

    logger.info("")
    logger.info(f"[Passthrough] {'=' * 60}")
    logger.info("[Passthrough] Starting switchyard passthrough server")
    logger.info(f"[Passthrough] Port:    {port}")
    logger.info(f"[Passthrough] Backend: {nvidia_config['base_url']}")
    logger.info(f"[Passthrough] Model:   {nvidia_config['model']}")
    logger.info(f"[Passthrough] {'=' * 60}")

    proc = _start_passthrough_server(
        port=port,
        api_key=nvidia_config["api_key"],
        base_url=nvidia_config["base_url"],
    )

    server_ready = wait_for_server(
        port,
        timeout=SERVER_STARTUP_TIMEOUT,
        server_name="Passthrough",
    )
    if not server_ready:
        stop_server_subprocess(proc)
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        pytest.fail(
            f"Passthrough server failed to start within "
            f"{SERVER_STARTUP_TIMEOUT}s.\n"
            f"stdout: {stdout[:2000]}\n"
            f"stderr: {stderr[:2000]}"
        )

    base_url = f"http://127.0.0.1:{port}"
    logger.info(f"[Passthrough] Server ready at {base_url}")

    yield {
        "process": proc,
        "port": port,
        "base_url": base_url,
        "model": nvidia_config["model"],
    }

    logger.info("[Passthrough] Shutting down server...")
    stop_server_subprocess(proc)
    logger.info("[Passthrough] Server stopped")
