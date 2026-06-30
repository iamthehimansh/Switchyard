# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for :class:`PluginClient` against real subprocesses.

Each test writes a small Python script to ``tmp_path`` that implements a
slice of the JSON-RPC contract — handshake-only, malformed responses,
crashes — so we exercise the actual subprocess + reader code path rather
than mocking the transport.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from switchyard.lib.plugin import (
    PROTOCOL_VERSION,
    PluginClient,
    PluginCrashError,
    PluginHandshakeError,
    PluginRoutingError,
    PluginTimeoutError,
    RouteDecision,
    RouteError,
    RouteRequest,
    RoutingRequestSummary,
)


def _write_plugin(tmp_path: Path, body: str) -> str:
    script = tmp_path / "plugin.py"
    script.write_text(textwrap.dedent(body).lstrip())
    return str(script)


def _route_request(*, request_id: str = "r1", tiers: tuple[str, ...] = ("strong", "weak")) -> RouteRequest:
    return RouteRequest(
        request_id=request_id,
        available_tiers=tiers,
        summary=RoutingRequestSummary(
            input_token_estimate=10,
            message_count=1,
            has_tool_use=False,
        ),
    )


_HEADER = """
    import json, sys
    def respond(rid, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": rid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()
"""


# ---------------------------------------------------------------------------
# Happy path: handshake + a single route
# ---------------------------------------------------------------------------


async def test_route_returns_decision(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + """
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
        elif env["method"] == "route":
            respond(env["id"], result={"tier": "weak", "metadata": {"score": 0.4}})
    """)

    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong", "weak"),
    )
    try:
        outcome = await client.route(_route_request())
        assert isinstance(outcome, RouteDecision)
        assert outcome.tier == "weak"
        assert outcome.metadata == {"score": 0.4}
    finally:
        await client.shutdown()


async def test_concurrent_routes_multiplexed_by_id(tmp_path: Path) -> None:
    """Two in-flight ``route`` calls must each get their matching response."""
    import asyncio

    plugin_script = _write_plugin(tmp_path, _HEADER + """
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
        elif env["method"] == "route":
            tier = "strong" if env["params"]["request_id"] == "alpha" else "weak"
            respond(env["id"], result={"tier": tier, "metadata": {}})
    """)

    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong", "weak"),
    )
    try:
        a, b = await asyncio.gather(
            client.route(_route_request(request_id="alpha")),
            client.route(_route_request(request_id="beta")),
        )
        assert isinstance(a, RouteDecision) and a.tier == "strong"
        assert isinstance(b, RouteDecision) and b.tier == "weak"
    finally:
        await client.shutdown()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_handshake_version_mismatch_raises(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + f"""
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={{"protocol_version": {PROTOCOL_VERSION + 99}, "plugin_name": "p", "plugin_version": "0"}})
    """)

    with pytest.raises(PluginHandshakeError, match="protocol"):
        await PluginClient.start(
            command=[sys.executable, "-u", plugin_script],
            available_tiers=("strong", "weak"),
        )


async def test_handshake_timeout_raises(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, """
        import sys, time
        # Read input but never respond.
        for _ in sys.stdin:
            time.sleep(60)
    """)

    with pytest.raises(PluginHandshakeError, match="handshake"):
        await PluginClient.start(
            command=[sys.executable, "-u", plugin_script],
            available_tiers=("strong",),
            handshake_timeout_s=0.5,
        )


async def test_route_timeout_raises(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + """
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
        # Never respond to route.
    """)

    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong", "weak"),
        request_timeout_s=0.5,
    )
    try:
        with pytest.raises(PluginTimeoutError):
            await client.route(_route_request())
    finally:
        await client.shutdown()


async def test_route_unknown_tier_raises(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + """
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
        elif env["method"] == "route":
            respond(env["id"], result={"tier": "ULTRA", "metadata": {}})
    """)

    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong", "weak"),
    )
    try:
        with pytest.raises(PluginRoutingError, match="unknown tier"):
            await client.route(_route_request())
    finally:
        await client.shutdown()


async def test_route_error_envelope_returned_as_route_error(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + """
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
        elif env["method"] == "route":
            respond(env["id"], error={"code": -32000, "message": "scorer dead", "data": {"fallback": "weak"}})
    """)

    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong", "weak"),
    )
    try:
        outcome = await client.route(_route_request())
        assert isinstance(outcome, RouteError)
        assert outcome.code == -32000
        assert outcome.fallback_tier == "weak"
    finally:
        await client.shutdown()


async def test_plugin_crash_after_handshake_surfaces_as_crash_error(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + """
    line = sys.stdin.readline()
    env = json.loads(line)
    respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
    sys.exit(0)
    """)

    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong", "weak"),
    )
    try:
        # Give the reader loop a tick to observe the closed stdout.
        import asyncio
        await asyncio.sleep(0.1)
        with pytest.raises(PluginCrashError):
            await client.route(_route_request())
    finally:
        await client.shutdown()


async def test_spawn_fails_when_binary_missing(tmp_path: Path) -> None:  # noqa: ARG001
    with pytest.raises(PluginHandshakeError, match="spawn"):
        await PluginClient.start(
            command=["/nonexistent/binary"],
            available_tiers=("strong",),
        )


async def test_shutdown_is_idempotent(tmp_path: Path) -> None:
    plugin_script = _write_plugin(tmp_path, _HEADER + """
    for line in sys.stdin:
        env = json.loads(line)
        if env["method"] == "handshake":
            respond(env["id"], result={"protocol_version": 1, "plugin_name": "p", "plugin_version": "0"})
    """)
    client = await PluginClient.start(
        command=[sys.executable, "-u", plugin_script],
        available_tiers=("strong",),
    )
    await client.shutdown()
    await client.shutdown()  # second call must not raise
