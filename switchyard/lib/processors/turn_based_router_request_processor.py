# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic turn-based router — cadenced strong/weak dispatch.

Sibling routing primitive to :mod:`switchyard.lib.processors.llm_classifier`
and :mod:`switchyard.lib.processors.plan_execute`.  Where the classifier
inspects the prompt and routes by semantic signals, and the random
router flips a weighted coin, this router routes by **cadence**: turn
1 of any task is always strong; subsequent turns are weak by default,
with strong "anchor" turns scheduled at a fixed long-run fraction.

The hypothesis: strong models earn their cost at *anchor* turns —
turn 1 for initial framing, periodic later turns for course correction
— while routine execution turns can use a cheaper executor without
hurting solve rate.  Random routing tests this with a coin flip;
turn-based routing tests it deterministically so the cadence is
predictable, transparent, and the same on every benchmark replay.

Mirrors :class:`switchyard.lib.profiles.random_routing.RandomRoutingConfig`'s
API exactly: same ``strong_probability`` knob, same long-run
distribution, just no randomness.  Drop-in replacement for benchmark
sweeps where reproducibility matters.

Tier selection formula
----------------------

For a target strong fraction ``p ∈ [0, 1]`` and 1-indexed turn ``t``::

    select strong  iff  ceil(t * p) > ceil((t - 1) * p)

This produces:

* Turn 1 always strong whenever ``p > 0`` (``ceil(p) ≥ 1 > 0``).
* Long-run strong rate exactly ``p`` (Bresenham line algorithm).
* Stateless — the decision depends only on the current turn number,
  which is computed from the inbound conversation history.

Distribution sanity check::

    p=1.0:  S S S S S S S S S S        always strong
    p=0.7:  S S S W S S W S S W ...    ~70% strong
    p=0.5:  S W S W S W S W S W ...    every other turn = strong
    p=0.3:  S W W S W W S W W W ...    ~30% strong (every 3-4 turns)
    p=0.2:  S W W W W S W W W W S ...  1 in 5 turns strong
    p=0.0:  W W W W W W W W W W        always weak

Turn counting
-------------

The "turn number" is the 1-indexed count of LLM invocations for this
task — i.e. *how deep in the agent loop are we?*  Computed from the
inbound conversation history as ``count(LLM-side prior items) + 1``.

Across the three inbound formats switchyard accepts:

* **OpenAI Chat Completions** — count ``messages[*].role == "assistant"``.
  Each LLM round produces exactly one assistant message (text and / or
  ``tool_calls``); tool results live in separate ``role: "tool"``
  messages so they don't inflate the count.
* **Anthropic Messages** — same: count ``messages[*].role == "assistant"``.
  Note that Anthropic puts ``tool_result`` blocks inside ``role: "user"``
  messages (not ``role: "tool"`` like OpenAI), but we don't count
  ``user``, so the asymmetry doesn't matter.
* **OpenAI Responses** — coarser fallback: with ``input: <string>`` the
  count is 0 (always turn 1); with ``input: <list>`` we count items
  that are agent-side (``role == "user"`` or ``type ==
  "function_call_output"``) and use that as an approximation of
  *completed agent acks*, which is one-less than completed LLM rounds.
  Adjusted by 1 if the trailing item is itself agent-side.  This is
  approximate — exact block-counting would require segmentation logic.
  Agent loops using the Responses API for tool calls may see slightly
  off cadence; for benchmark workloads we run (terminus-2 over OpenAI
  Chat Completions) this branch is never hit.

Unknown formats default to turn 1 (strong on first call).  Better to
have a known-safe baseline behavior than to fail.

What gets stamped
-----------------

* :data:`CTX_DETERMINISTIC_ROUTING_TIER` — the chosen tier label
  (``"strong"`` or ``"weak"`` by default, configurable via
  :attr:`TurnBasedRoutingConfig.strong_tier` /
  :attr:`TurnBasedRoutingConfig.weak_tier`).  Same key
  :class:`switchyard.lib.backends.deterministic_routing_llm_backend.DeterministicRoutingLLMBackend` reads, so
  the existing per-tier backend dispatch plugs in unchanged.

Observability
-------------

One ``turn_based_decision={...}`` JSON line per request to stderr
(mirrors the classifier's ``classifier_signals=…`` and the planner's
``planner_decision=…`` pattern).  Lands in captured server logs
regardless of uvicorn log level.  Fields::

    {
      "turn": 6,
      "strong_probability": 0.5,
      "tier": "strong",
      "request_type": "openai_chat",
    }

Grep ``turn_based_decision=`` on the server log to tally cadence
distribution + verify the routing matched the expected pattern.
"""

from __future__ import annotations

import json
import math
import sys
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)
from switchyard.lib.conversation_turn import conversation_turn_number

if TYPE_CHECKING:
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard_rust.core import ChatRequest


#: ``ProxyContext.metadata`` key for the turn number this request was
#: routed at.  Always stamped; useful for downstream audit / cost
#: attribution that wants to slice executor spend by turn position.
CTX_TURN_BASED_TURN = "_turn_based_turn"


class TurnBasedRoutingConfig(BaseModel):
    """Configuration for :class:`TurnBasedRouterRequestProcessor`.

    Mirrors
    :class:`switchyard.lib.profiles.random_routing.RandomRoutingConfig`'s
    surface so the two router primitives are drop-in interchangeable
    for benchmark sweeps that vary the routing *strategy* against the
    same upstream tier targets.
    """

    model_config = ConfigDict(frozen=True)

    strong_tier: str = Field(default="strong", min_length=1)
    """Tier label stamped for strong-routed turns.  Must match one of
    the labels the downstream
    :class:`switchyard.lib.backends.deterministic_routing_llm_backend.DeterministicRoutingLLMBackend` knows
    about."""

    weak_tier: str = Field(default="weak", min_length=1)
    """Tier label stamped for weak-routed turns."""

    strong_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    """Target long-run fraction of turns routed to the strong tier.
    ``1.0`` = every turn strong (equivalent to ``FixedTierRequestProcessor("strong")``);
    ``0.0`` = every turn weak (equivalent to
    ``FixedTierRequestProcessor("weak")``); intermediate values
    produce a deterministic cadence whose long-run rate matches.
    Default ``0.5`` (every other turn = strong) mirrors the
    ``RandomRoutingConfig`` default."""


class TurnBasedRouterRequestProcessor:
    """Stamp a strong/weak tier label based on a deterministic per-turn cadence.

    Reads the inbound conversation history to compute the current
    turn number, applies the Bresenham ceil formula to decide whether
    this turn should hit the strong or weak tier, and stamps the
    label so the downstream
    :class:`DeterministicRoutingLLMBackend` dispatches accordingly.

    Emits one ``turn_based_decision={...}`` JSON audit line per request
    to stderr for observability.
    """

    def __init__(self, config: TurnBasedRoutingConfig) -> None:
        self._config = config

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        turn = conversation_turn_number(request)
        tier = _select_tier(
            turn=turn,
            strong_probability=self._config.strong_probability,
            strong_tier=self._config.strong_tier,
            weak_tier=self._config.weak_tier,
        )
        ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = tier
        ctx.metadata[CTX_TURN_BASED_TURN] = turn

        # Audit line — mirrors classifier_signals=... and
        # planner_decision=... so the three routers share a grep
        # convention.  Written directly to stderr so the line lands in
        # captured server logs regardless of uvicorn's logger config.
        request_type = request.request_type.value
        audit_payload = {
            "turn": turn,
            "strong_probability": self._config.strong_probability,
            "tier": tier,
            "request_type": request_type,
        }
        sys.stderr.write(
            f"turn_based_decision={json.dumps(audit_payload, sort_keys=True)}\n"
        )
        sys.stderr.flush()
        return request


def _select_tier(
    *,
    turn: int,
    strong_probability: float,
    strong_tier: str,
    weak_tier: str,
) -> str:
    """Return the tier label for ``turn`` under the given ``strong_probability``.

    Bresenham ceil formula: strong iff ``ceil(t * p) > ceil((t-1) * p)``.

    The two extremes are short-circuited because Python's ``math.ceil``
    is finicky at ``p == 0`` (would give weak for every turn including
    turn 1 — fine but wasteful to call) and ``p == 1`` would always
    pass (correct, but explicit is clearer).
    """
    if strong_probability >= 1.0:
        return strong_tier
    if strong_probability <= 0.0:
        return weak_tier
    cur = math.ceil(turn * strong_probability)
    prev = math.ceil((turn - 1) * strong_probability)
    return strong_tier if cur > prev else weak_tier


__all__ = [
    "CTX_TURN_BASED_TURN",
    "TurnBasedRouterRequestProcessor",
    "TurnBasedRoutingConfig",
]
