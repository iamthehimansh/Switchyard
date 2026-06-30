# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force every request to a single pre-chosen tier — no LLM call.

Drop-in replacement for the
:class:`LLMClassifierRequestProcessor` + :class:`SignalTierSelectorRequestProcessor`
pair when you want an all-strong or all-weak baseline run.  Skipping
the classifier round-trip eliminates the per-request token cost and
the ~3–17 s wall-time tax of a real classification call, so the
forced-tier baseline measures only the upstream backend's behaviour
(solve rate, cost, latency) without any classifier contamination.

Stamps :data:`CTX_DETERMINISTIC_ROUTING_TIER` directly so the
downstream :class:`DeterministicRoutingLLMBackend` dispatches without
needing a synthetic :class:`RouteDecision` to materialise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest


class FixedTierRequestProcessor:
    """Stamp a fixed tier label onto every request.

    Args:
        tier: Tier label to stamp.  Must match one of the labels the
            downstream :class:`DeterministicRoutingLLMBackend` knows
            about (typically ``"strong"`` or ``"weak"``).
    """

    def __init__(self, tier: str) -> None:
        if not isinstance(tier, str) or not tier:
            raise ValueError(f"tier must be a non-empty string, got {tier!r}")
        self._tier = tier

    @property
    def tier(self) -> str:
        return self._tier

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = self._tier
        return request


__all__ = ["FixedTierRequestProcessor"]
