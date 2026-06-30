# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol for the external-process routing plugin.

The plugin and Switchyard speak JSON-RPC 2.0 over the plugin's stdio. Each
message is one newline-terminated JSON object. The plugin reads requests
from stdin, writes responses to stdout, and may log to stderr — Switchyard
captures the stderr stream and tags log lines with the plugin name.

Two RPC methods exist in v1:

* ``handshake`` — exchanged once at startup; lets Switchyard fail closed
  on protocol-version mismatch before any traffic flows.
* ``route`` — called per inbound request; the plugin returns the tier
  label to dispatch to (e.g. ``"strong"`` / ``"weak"``).

A ``health`` ping and a ``shutdown`` notification are reserved for v2 —
SIGTERM is the source of truth for shutdown today.

This module owns the typed dataclasses for every payload. Marshalling
to/from JSON-RPC envelopes lives in :mod:`switchyard.lib.plugin.plugin_client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Wire-format version. Bumped on any breaking change to the request or
#: response schemas. The plugin advertises the version it speaks during
#: ``handshake``; Switchyard fails closed if the values don't match.
PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# handshake
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandshakeRequest:
    """Sent by Switchyard once, immediately after spawning the plugin.

    Tells the plugin what protocol version Switchyard speaks and which
    tier labels it may legally return from ``route``. Plugins MUST reject
    the handshake (by returning a JSON-RPC error) if any of these are
    incompatible — Switchyard treats that as a startup failure.
    """

    switchyard_protocol_version: int
    available_tiers: tuple[str, ...]

    def to_params(self) -> dict[str, Any]:
        return {
            "switchyard_protocol_version": self.switchyard_protocol_version,
            "available_tiers": list(self.available_tiers),
        }


@dataclass(frozen=True)
class HandshakeResponse:
    """Plugin's reply to ``handshake``.

    ``opt_in_fields`` lets a plugin request optional payload fields (e.g.
    ``"raw_messages"`` for a semantic router) — the operator's allowlist
    decides whether Switchyard honours each request.
    """

    protocol_version: int
    plugin_name: str
    plugin_version: str
    opt_in_fields: tuple[str, ...] = ()

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> HandshakeResponse:
        return cls(
            protocol_version=int(result["protocol_version"]),
            plugin_name=str(result.get("plugin_name", "<unnamed>")),
            plugin_version=str(result.get("plugin_version", "0.0.0")),
            opt_in_fields=tuple(result.get("opt_in_fields", ())),
        )


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingRequestSummary:
    """The minimal request signal Switchyard always sends to the plugin.

    Excludes raw message content and backend credentials by design. A
    plugin that needs richer signal must declare it via
    :attr:`HandshakeResponse.opt_in_fields` and the operator must
    explicitly allowlist that opt-in.
    """

    input_token_estimate: int
    message_count: int
    has_tool_use: bool
    tool_names: tuple[str, ...] = ()
    system_prompt_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "input_token_estimate": self.input_token_estimate,
            "message_count": self.message_count,
            "has_tool_use": self.has_tool_use,
            "tool_names": list(self.tool_names),
        }
        if self.system_prompt_fingerprint is not None:
            out["system_prompt_fingerprint"] = self.system_prompt_fingerprint
        return out


@dataclass(frozen=True)
class RouteRequest:
    """Per-request payload Switchyard sends on every ``route`` call."""

    request_id: str
    available_tiers: tuple[str, ...]
    summary: RoutingRequestSummary
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "available_tiers": list(self.available_tiers),
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RouteDecision:
    """Successful routing decision returned by the plugin.

    ``tier`` MUST be one of the labels Switchyard advertised via
    :attr:`RouteRequest.available_tiers`. Anything else surfaces as
    :class:`PluginRoutingError` with structured ``error.code``.
    """

    tier: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteError:
    """Plugin signalled a per-request failure.

    ``fallback_tier`` is an optional hint — when set, and when the
    operator opted into fallbacks, Switchyard dispatches to that tier
    instead of failing the request closed.
    """

    code: int
    message: str
    fallback_tier: str | None = None


#: Discriminated-union return type for ``route``. The client classifies the
#: JSON-RPC envelope and returns one of these two shapes; callers branch on
#: ``isinstance`` to handle the two outcomes.
RouteResult = RouteDecision | RouteError
