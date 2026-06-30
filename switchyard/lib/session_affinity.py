# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-conversation decision pin shared across routers (sticky routing)."""

from __future__ import annotations

from switchyard.lib.conversation_turn import conversation_turn_number
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.session_cache import SessionCache
from switchyard.lib.session_key import session_key_from_body
from switchyard_rust.core import ChatRequest

#: ``ProxyContext.metadata`` key memoizing the per-request session key, so
#: callers that consult affinity more than once don't re-hash the (growing)
#: request body.
CTX_SESSION_KEY = "_session_affinity_key"


class SessionAffinity:
    """Pins a routing decision per conversation and reuses it on later turns.

    The shared building block both routing paths use for the *common* part of
    stickiness: derive a stable conversation key (system prompt + first user
    message, memoized on the request context) and store/look up a pinned value
    in a bounded LRU. The pinned value is opaque — a tier label for the
    classifier router, an endpoint id for the latency backend. Each caller keeps
    its own *policy* (when to write a pin, whether to honor one) on top.

    A disabled instance no-ops (and never hashes the body). Not thread-safe —
    touched only on the request path.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_sessions: int = 10_000,
        warmup_turns: int = 0,
    ) -> None:
        if warmup_turns < 0:
            raise ValueError("warmup_turns must be >= 0")
        self._enabled = enabled
        self._warmup_turns = warmup_turns
        self._pins: SessionCache = SessionCache(max_sessions)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def max_sessions(self) -> int:
        """Configured maximum number of pinned conversations."""
        return self._pins.max_sessions

    @property
    def warmup_turns(self) -> int:
        """Number of initial turns that cannot read or write affinity pins."""
        return self._warmup_turns

    def pinned(self, ctx: ProxyContext, request: ChatRequest) -> str | None:
        """Return the value pinned to ``request``'s conversation, or ``None``."""
        if not self._enabled or self._is_warmup_turn(request):
            return None
        return self._pins.get(self._session_key(ctx, request))

    def pin(self, ctx: ProxyContext, request: ChatRequest, value: str) -> None:
        """Pin ``request``'s conversation to ``value`` (no-op when disabled)."""
        if not self._enabled or self._is_warmup_turn(request):
            return
        self._pins.put(self._session_key(ctx, request), value)

    def __len__(self) -> int:
        """Number of conversations currently pinned."""
        return len(self._pins)

    def _session_key(self, ctx: ProxyContext, request: ChatRequest) -> str:
        """Derive the conversation key once per request, memoized on ``ctx``."""
        cached = ctx.metadata.get(CTX_SESSION_KEY)
        if isinstance(cached, str):
            return cached
        key = session_key_from_body(request.body)
        ctx.metadata[CTX_SESSION_KEY] = key
        return key

    def _is_warmup_turn(self, request: ChatRequest) -> bool:
        """Return whether this request is still inside the no-stick warmup."""
        return conversation_turn_number(request) <= self._warmup_turns


__all__ = ["CTX_SESSION_KEY", "SessionAffinity"]
