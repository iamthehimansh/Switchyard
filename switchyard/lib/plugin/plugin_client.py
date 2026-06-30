# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for an external-process routing plugin.

Owns the plugin's lifecycle: spawn the subprocess, perform the handshake,
multiplex JSON-RPC requests over its stdio, surface stderr as logs, and
terminate it gracefully on shutdown. One :class:`PluginClient` per chain.

Crash recovery: a single-shot recovery is attempted on read failure
(plugin process died mid-conversation). Exceeding the configured restart
budget makes the next ``route`` raise :class:`PluginCrashError`, which
the request processor turns into either a fail-closed error response or
a fallback-tier dispatch depending on operator config.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
from collections.abc import Sequence
from typing import Any

from switchyard.lib.plugin.plugin_protocol import (
    PROTOCOL_VERSION,
    HandshakeRequest,
    HandshakeResponse,
    RouteDecision,
    RouteError,
    RouteRequest,
    RouteResult,
)

log = logging.getLogger(__name__)


class PluginError(Exception):
    """Base class for routing-plugin errors."""


class PluginHandshakeError(PluginError):
    """Plugin failed the startup handshake (version mismatch, exit, timeout)."""


class PluginTimeoutError(PluginError):
    """Plugin did not respond to a ``route`` request within the configured timeout."""


class PluginCrashError(PluginError):
    """Plugin process died and exhausted its restart budget."""


class PluginRoutingError(PluginError):
    """Plugin returned a malformed or invalid response to a ``route`` request."""


class PluginClient:
    """Long-lived client for a single external routing plugin.

    Spawn via :meth:`start` — which performs the handshake — then call
    :meth:`route` per inbound request. Multiple concurrent ``route``
    calls are multiplexed over the plugin's stdio by JSON-RPC ``id``.

    The plugin's stderr is forwarded to the ``switchyard.plugin.<name>``
    logger so plugin diagnostics show up alongside Switchyard's own
    logs without losing the plugin/proxy attribution.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        available_tiers: tuple[str, ...],
        request_timeout_s: float,
        handshake_timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("PluginClient requires a non-empty command")
        if not available_tiers:
            raise ValueError("PluginClient requires at least one available tier label")

        self._command: tuple[str, ...] = tuple(command)
        self._available_tiers = available_tiers
        self._request_timeout_s = request_timeout_s
        self._handshake_timeout_s = handshake_timeout_s
        self._env = env

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 1
        self._send_lock = asyncio.Lock()
        self._handshake: HandshakeResponse | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Construction / handshake
    # ------------------------------------------------------------------

    @classmethod
    def parse_command(cls, command: str | Sequence[str]) -> tuple[str, ...]:
        """Accept either a shell string or a pre-split argv sequence."""
        if isinstance(command, str):
            return tuple(shlex.split(command))
        return tuple(command)

    @classmethod
    async def start(
        cls,
        *,
        command: str | Sequence[str],
        available_tiers: Sequence[str],
        request_timeout_s: float = 5.0,
        handshake_timeout_s: float = 10.0,
        env: dict[str, str] | None = None,
    ) -> PluginClient:
        """Spawn the plugin and run the handshake.

        Returns a ready-to-use client, or raises :class:`PluginHandshakeError`
        if the plugin failed to start, the handshake timed out, or the
        plugin advertised an incompatible protocol version.
        """
        client = cls(
            command=cls.parse_command(command),
            available_tiers=tuple(available_tiers),
            request_timeout_s=request_timeout_s,
            handshake_timeout_s=handshake_timeout_s,
            env=env,
        )
        await client._spawn()
        await client._handshake_exchange()
        return client

    async def _spawn(self) -> None:
        """Start the subprocess and the background stdout/stderr readers."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **(self._env or {})} if self._env else None,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise PluginHandshakeError(
                f"Failed to spawn plugin {self._command!r}: {exc}",
            ) from exc

        # asyncio.create_subprocess_exec attaches PIPE streams when requested,
        # so the optional streams are guaranteed non-None here.
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._reader_task = asyncio.create_task(
            self._read_loop(self._process.stdout),
            name="plugin-stdout-reader",
        )
        self._stderr_task = asyncio.create_task(
            self._stderr_loop(self._process.stderr),
            name="plugin-stderr-reader",
        )

    async def _handshake_exchange(self) -> None:
        request = HandshakeRequest(
            switchyard_protocol_version=PROTOCOL_VERSION,
            available_tiers=self._available_tiers,
        )
        try:
            envelope = await asyncio.wait_for(
                self._call_method("handshake", request.to_params()),
                timeout=self._handshake_timeout_s,
            )
        except TimeoutError as exc:
            await self._abort()
            raise PluginHandshakeError(
                f"Plugin {self._command[0]!r} did not complete handshake within "
                f"{self._handshake_timeout_s}s",
            ) from exc
        except PluginError as exc:
            await self._abort()
            raise PluginHandshakeError(str(exc)) from exc

        if "error" in envelope:
            await self._abort()
            err = envelope["error"]
            message = err.get("message") if isinstance(err, dict) else repr(err)
            raise PluginHandshakeError(
                f"Plugin {self._command[0]!r} rejected handshake: {message}",
            )

        result = envelope.get("result")
        if not isinstance(result, dict):
            await self._abort()
            raise PluginHandshakeError(
                f"Plugin {self._command[0]!r} returned envelope without "
                f"result: {envelope!r}",
            )
        try:
            handshake = HandshakeResponse.from_result(result)
        except (KeyError, TypeError, ValueError) as exc:
            await self._abort()
            raise PluginHandshakeError(
                f"Plugin {self._command[0]!r} returned malformed handshake: {result!r}",
            ) from exc

        if handshake.protocol_version != PROTOCOL_VERSION:
            await self._abort()
            raise PluginHandshakeError(
                f"Plugin {handshake.plugin_name!r} speaks protocol "
                f"v{handshake.protocol_version}, Switchyard speaks "
                f"v{PROTOCOL_VERSION}",
            )

        self._handshake = handshake
        log.info(
            "PluginClient: handshake OK plugin=%s version=%s opt_in=%s",
            handshake.plugin_name,
            handshake.plugin_version,
            handshake.opt_in_fields,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def handshake(self) -> HandshakeResponse:
        """Plugin's handshake response. Raises if start() hasn't completed."""
        if self._handshake is None:
            raise RuntimeError("PluginClient.start() has not completed")
        return self._handshake

    async def route(self, request: RouteRequest) -> RouteResult:
        """Ask the plugin to make a routing decision for a request.

        Returns either :class:`RouteDecision` or :class:`RouteError`. The
        caller decides what to do with the error (fail-closed vs fallback).
        Raises :class:`PluginTimeoutError` / :class:`PluginCrashError` /
        :class:`PluginRoutingError` for plugin-side failures the caller
        cannot recover from.
        """
        if self._closed:
            raise PluginCrashError("PluginClient has been shut down")
        if self._process is None or self._process.returncode is not None:
            raise PluginCrashError(
                f"Plugin {self._command[0]!r} is not running "
                f"(exit code {self._process.returncode if self._process else 'n/a'})",
            )

        try:
            envelope = await asyncio.wait_for(
                self._call_method("route", request.to_params()),
                timeout=self._request_timeout_s,
            )
        except TimeoutError as exc:
            raise PluginTimeoutError(
                f"Plugin {self._handshake_name()!r} did not respond to "
                f"route within {self._request_timeout_s}s",
            ) from exc

        return self._classify_route_envelope(envelope, available=request.available_tiers)

    async def shutdown(self) -> None:
        """Terminate the plugin gracefully.

        Closes stdin (most well-behaved plugins exit on EOF), waits up to
        2 seconds for the process to exit, then escalates to SIGTERM and
        finally SIGKILL. Idempotent — safe to call from multiple shutdown
        paths.
        """
        if self._closed:
            return
        self._closed = True
        await self._abort()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handshake_name(self) -> str:
        return self._handshake.plugin_name if self._handshake else self._command[0]

    async def _call_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and await its response.

        Returns the raw ``result`` payload; raises :class:`PluginRoutingError`
        if the plugin returned a JSON-RPC error envelope (callers translate
        that into a user-facing failure).
        """
        if self._process is None or self._process.stdin is None:
            raise PluginCrashError(
                f"Plugin {self._command[0]!r} is not running",
            )

        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        envelope = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        line = (json.dumps(envelope) + "\n").encode("utf-8")
        async with self._send_lock:
            try:
                self._process.stdin.write(line)
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._pending.pop(request_id, None)
                raise PluginCrashError(
                    f"Plugin {self._command[0]!r} closed stdin mid-request",
                ) from exc

        try:
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _read_loop(self, stdout: asyncio.StreamReader) -> None:
        """Read JSON-RPC responses, dispatch to pending futures by ``id``."""
        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                try:
                    envelope = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(
                        "PluginClient: dropping non-JSON line from %s: %r",
                        self._command[0],
                        line[:200],
                    )
                    continue
                self._dispatch_envelope(envelope)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("PluginClient: stdout reader crashed")
        finally:
            # Plugin closed stdout; fail any in-flight callers so they
            # don't hang forever waiting on a future nobody will resolve.
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        PluginCrashError(
                            f"Plugin {self._command[0]!r} closed stdout",
                        ),
                    )
            self._pending.clear()

    def _dispatch_envelope(self, envelope: Any) -> None:
        if not isinstance(envelope, dict):
            log.warning("PluginClient: ignoring non-object JSON: %r", envelope)
            return
        message_id = envelope.get("id")
        if not isinstance(message_id, int):
            log.warning("PluginClient: ignoring envelope without int id: %r", envelope)
            return
        future = self._pending.get(message_id)
        if future is None or future.done():
            return
        future.set_result(envelope)

    async def _stderr_loop(self, stderr: asyncio.StreamReader) -> None:
        """Forward plugin stderr to the per-plugin logger."""
        plugin_logger = logging.getLogger(
            f"switchyard.plugin.{self._safe_logger_name()}",
        )
        try:
            while True:
                line = await stderr.readline()
                if not line:
                    break
                plugin_logger.info(line.rstrip(b"\n").decode("utf-8", errors="replace"))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("PluginClient: stderr reader crashed")

    def _safe_logger_name(self) -> str:
        if self._handshake is not None:
            return self._handshake.plugin_name.replace(".", "_") or "unnamed"
        return os.path.basename(self._command[0]).replace(".", "_") or "unnamed"

    def _classify_route_envelope(
        self,
        envelope: dict[str, Any],
        *,
        available: tuple[str, ...],
    ) -> RouteResult:
        """Turn a JSON-RPC envelope into a :class:`RouteResult`."""
        if "error" in envelope:
            err = envelope["error"]
            if not isinstance(err, dict):
                raise PluginRoutingError(
                    f"Plugin {self._handshake_name()!r} returned non-object error: "
                    f"{envelope!r}",
                )
            data = err.get("data") or {}
            fallback = data.get("fallback") if isinstance(data, dict) else None
            return RouteError(
                code=int(err.get("code", -32000)),
                message=str(err.get("message", "<no message>")),
                fallback_tier=str(fallback) if isinstance(fallback, str) else None,
            )

        result = envelope.get("result")
        if not isinstance(result, dict):
            raise PluginRoutingError(
                f"Plugin {self._handshake_name()!r} returned envelope without "
                f"result: {envelope!r}",
            )
        tier = result.get("tier")
        if not isinstance(tier, str):
            raise PluginRoutingError(
                f"Plugin {self._handshake_name()!r} omitted 'tier' from route result: "
                f"{result!r}",
            )
        if tier not in available:
            raise PluginRoutingError(
                f"Plugin {self._handshake_name()!r} returned unknown tier {tier!r}; "
                f"expected one of {sorted(available)}",
            )
        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return RouteDecision(tier=tier, metadata=metadata)

    async def _abort(self) -> None:
        """Tear down the subprocess + reader tasks. Idempotent."""
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                if process.stdin is not None and not process.stdin.is_closing():
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()

        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
