# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-backend e2e test for the **full** turn-based router chain.

Sibling to :mod:`tests.e2e.test_planning_request_processor_e2e` and
:mod:`tests.e2e.test_classifier_planner_chain_e2e`. This file walks
the turn-based router end-to-end against a live backend::

    inbound ChatRequest
       -> TurnBasedRouterRequestProcessor   (no LLM call; counts
                                              assistant messages,
                                              applies Bresenham
                                              ceil schedule)
       -> DeterministicRoutingLLMBackend    (dispatches to per-tier
                                              backend based on the
                                              tier label stamped on
                                              ctx by the router)
           -> strong tier: real LLM call    (gpt-5.2)
           -> weak tier:   real LLM call    (V4 Pro)
       -> ChatResponse

The router itself does no LLM call (just compute), so the cost on
this test comes entirely from the executor tier the router picked.
At ``strong_probability=0.5`` over three crafted requests the
expected pattern is **strong → weak → strong**, with each request
landing on the correct tier model.

Strong vs weak distinct models is the critical thing this test
exercises that the unit tests cannot: it verifies the
``ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]`` stamp actually
drives the multi-tier dispatcher's per-tier dispatch.  The unit
tests stub the backend entirely; this test wires
:class:`DeterministicRoutingLLMBackend` with two real backends
pointed at two real models, and verifies the response's ``model``
field matches the tier the router picked.

Prerequisites:
    - ``OPENROUTER_API_KEY`` or ``NVIDIA_API_KEY`` env var (skips otherwise).

Run with::

    OPENROUTER_API_KEY=sk-or-... uv run pytest \\
        tests/e2e/test_turn_based_router_chain_e2e.py -v
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pytest

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.chat_response.base import ChatResponse
from switchyard.lib.processors.turn_based_router_request_processor import (
    CTX_TURN_BASED_TURN,
    TurnBasedRouterRequestProcessor,
    TurnBasedRoutingConfig,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequest

from .conftest import get_nvidia_config

# Three real LLM calls per cadence test (one per crafted turn).  120s
# headroom is generous for the small-prompt smoke calls these tests
# do (each request is ~10 tokens out).
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]

logger = logging.getLogger("e2e.turn_based_router_chain")

HTTP_TIMEOUT_S = 60.0

_nvidia = get_nvidia_config()
_skip_reason = "OPENROUTER_API_KEY or NVIDIA_API_KEY not set"

BACKEND_BASE_URL = _nvidia["base_url"]

# Distinct strong / weak model ids so the test can verify routing by
# inspecting ``response.body["model"]``.  Both are OpenAI-Chat-Completions
# compatible on the selected backend — no Anthropic-native translation
# required, keeping the test focused on routing rather than wire-format
# wrangling.
if _nvidia["provider"] == "openrouter":
    STRONG_MODEL = _nvidia["model"]
    WEAK_MODEL = os.environ.get("OPENROUTER_WEAK_MODEL") or "moonshotai/kimi-k2.6"
else:
    STRONG_MODEL = "openai/openai/gpt-5.2"
    WEAK_MODEL = "nvidia/deepseek-ai/evals-deepseek-v4-pro"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_full_chain(
    *,
    strong_probability: float,
) -> tuple[list[Any], LLMBackend]:
    """Build (request_processors, multi-tier backend) for the turn-based chain.

    Mirrors the turn-based routing benchmark chain in-process with direct
    OpenAiPassthrough-equivalent tier
    backends (via :meth:`DeterministicRoutingLLMBackend.from_tiers`,
    which builds standard OpenAI / Anthropic backends from
    :class:`LlmTarget` config).  No stats wrapping — the router itself
    doesn't record stats anyway, and tier-side stats are out of scope
    for this routing-correctness test.
    """
    router = TurnBasedRouterRequestProcessor(
        TurnBasedRoutingConfig(
            strong_tier="strong",
            weak_tier="weak",
            strong_probability=strong_probability,
        ),
    )
    backend = DeterministicRoutingLLMBackend.from_tiers(
        tiers={
            "strong": LlmTarget(
                model=STRONG_MODEL,
                format=BackendFormat.OPENAI,
                api_key=_nvidia["api_key"],
                base_url=BACKEND_BASE_URL,
                timeout=HTTP_TIMEOUT_S,
            ),
            "weak": LlmTarget(
                model=WEAK_MODEL,
                format=BackendFormat.OPENAI,
                api_key=_nvidia["api_key"],
                base_url=BACKEND_BASE_URL,
                timeout=HTTP_TIMEOUT_S,
                # DeepSeek V4 needs enable_thinking=False or its JSON
                # output gets misrouted into reasoning_content.  This
                # call doesn't request structured output so the risk
                # is mostly latency (V4 with thinking on is 10-15s
                # slower per call) — set anyway for parity with the
                # benchmark server's auto-detect.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                extra_headers={"X-Inference-Priority": "batch"},
            ),
        },
        default_tier="strong",
    )
    return [router], backend


async def _call_through_chain(
    request_processors: list[Any],
    backend: LLMBackend,
    request: ChatRequest,
) -> tuple[ProxyContext, ChatResponse]:
    """Run a request through router + multi-tier backend in-process."""
    ctx = ProxyContext()
    for proc in request_processors:
        request = await proc.process(ctx, request)
    response = await backend.call(ctx, request)
    return ctx, response


def _chat_request_with_n_prior_assistants(
    *,
    n_prior_assistants: int,
    user_message: str = "Reply with: ok",
) -> ChatRequest:
    """Build an OpenAI Chat Completions ``ChatRequest`` whose history
    contains exactly ``n_prior_assistants`` assistant messages.

    Used to drive the router's turn counter — the router computes
    ``turn = count(assistant) + 1``, so ``n_prior_assistants=0`` →
    turn 1, ``n_prior_assistants=1`` → turn 2, etc.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "initial task"},
    ]
    for i in range(n_prior_assistants):
        messages.append({"role": "assistant", "content": f"r{i + 1}"})
        messages.append({"role": "user", "content": f"continue ({i + 1})"})
    # Replace the trailing "continue" with the actual prompt for the
    # current turn so the model has something concrete to answer.
    if n_prior_assistants > 0:
        messages[-1] = {"role": "user", "content": user_message}
    else:
        messages[-1] = {"role": "user", "content": user_message}
    return ChatRequest.openai_chat({
        "model": "client-placeholder",
        "messages": messages,
        "max_tokens": 10,
    })


def _model_from_response(response: ChatResponse) -> str:
    """The upstream-reported model in the response body."""
    return str(response.body.get("model") or "")


def _content_from_response(response: ChatResponse) -> str:
    choice = response.body["choices"][0]
    message = choice.get("message", {})
    return str(message.get("content") or "")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestTurnBasedRouterChainLiveE2E:
    """Full turn-based router chain (router + multi-tier backend) against the live hub."""

    async def test_cadence_routes_through_both_tiers_at_half_probability(self) -> None:
        """At ``strong_probability=0.5`` three crafted turns yield S → W → S.

        The router stamps ``CTX_DETERMINISTIC_ROUTING_TIER`` on each
        request; :class:`DeterministicRoutingLLMBackend` dispatches to
        the matching tier; the response's ``model`` field reflects the
        upstream model that was actually called.  This test wires
        distinct strong / weak models specifically so we can verify
        the dispatch end-to-end — the response model identifying which
        tier handled the request.

        The cadence assertion is the user-facing invariant: turn 1
        always strong, turn 2 weak at p=0.5, turn 3 strong again.
        """
        req_processors, backend = _build_full_chain(strong_probability=0.5)

        # --- Turn 1 (no prior assistants) → strong ------------------
        req1 = _chat_request_with_n_prior_assistants(
            n_prior_assistants=0,
            user_message="Reply with one word: hi",
        )
        ctx1, resp1 = await _call_through_chain(req_processors, backend, req1)

        assert ctx1.metadata[CTX_TURN_BASED_TURN] == 1
        assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
        model1 = _model_from_response(resp1)
        assert model1 == STRONG_MODEL, (
            f"turn 1 should hit the strong tier ({STRONG_MODEL!r}); "
            f"response reported model={model1!r}"
        )
        content1 = _content_from_response(resp1)
        assert content1, "expected non-empty content from strong-tier executor"

        # --- Turn 2 (1 prior assistant) → weak ----------------------
        req2 = _chat_request_with_n_prior_assistants(
            n_prior_assistants=1,
            user_message="Reply with one word: ok",
        )
        ctx2, resp2 = await _call_through_chain(req_processors, backend, req2)

        assert ctx2.metadata[CTX_TURN_BASED_TURN] == 2
        assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
        model2 = _model_from_response(resp2)
        assert model2 == WEAK_MODEL, (
            f"turn 2 at p=0.5 should hit the weak tier ({WEAK_MODEL!r}); "
            f"response reported model={model2!r}"
        )
        content2 = _content_from_response(resp2)
        assert content2, "expected non-empty content from weak-tier executor"

        # --- Turn 3 (2 prior assistants) → strong --------------------
        req3 = _chat_request_with_n_prior_assistants(
            n_prior_assistants=2,
            user_message="Reply with one word: done",
        )
        ctx3, resp3 = await _call_through_chain(req_processors, backend, req3)

        assert ctx3.metadata[CTX_TURN_BASED_TURN] == 3
        assert ctx3.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
        model3 = _model_from_response(resp3)
        assert model3 == STRONG_MODEL, (
            f"turn 3 at p=0.5 should hit the strong tier ({STRONG_MODEL!r}); "
            f"response reported model={model3!r}"
        )

        logger.info(
            "  [turn-based-e2e][p=0.5] cadence=[t1=%s/%s, t2=%s/%s, t3=%s/%s]",
            ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER], model1,
            ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER], model2,
            ctx3.metadata[CTX_DETERMINISTIC_ROUTING_TIER], model3,
        )

    async def test_low_probability_anchors_first_turn_then_routes_weak(self) -> None:
        """At ``strong_probability=0.2`` turn 1 routes strong, turns 2-5 route weak.

        Verifies the "anchor + sparse strong" cadence the turn-based
        router was designed for: cheap executor handles the bulk of
        the loop while turn 1 (and every Nth turn after) gets a
        strong-model anchor.  At p=0.2 the next strong turn after
        turn 1 is turn 6.

        Sweeps turn 1 + a sample of subsequent turns rather than the
        full 1-6 to keep the test under a few seconds of LLM calls;
        the unit tests cover the exact cadence pattern over a longer
        window.
        """
        req_processors, backend = _build_full_chain(strong_probability=0.2)

        # Turn 1: always strong (the anchor).
        req1 = _chat_request_with_n_prior_assistants(
            n_prior_assistants=0,
            user_message="Reply with: anchor",
        )
        ctx1, resp1 = await _call_through_chain(req_processors, backend, req1)
        assert ctx1.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
        assert _model_from_response(resp1) == STRONG_MODEL

        # Turn 2: weak (cadence at p=0.2 only strong at turns 1, 6, 11, …).
        req2 = _chat_request_with_n_prior_assistants(
            n_prior_assistants=1,
            user_message="Reply with: step2",
        )
        ctx2, resp2 = await _call_through_chain(req_processors, backend, req2)
        assert ctx2.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
        assert _model_from_response(resp2) == WEAK_MODEL

        # Turn 5: still weak (next strong is turn 6).
        req5 = _chat_request_with_n_prior_assistants(
            n_prior_assistants=4,
            user_message="Reply with: step5",
        )
        ctx5, resp5 = await _call_through_chain(req_processors, backend, req5)
        assert ctx5.metadata[CTX_TURN_BASED_TURN] == 5
        assert ctx5.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
        assert _model_from_response(resp5) == WEAK_MODEL

        logger.info(
            "  [turn-based-e2e][p=0.2] sparse-strong: t1=strong/%s, "
            "t2=weak/%s, t5=weak/%s",
            _model_from_response(resp1),
            _model_from_response(resp2),
            _model_from_response(resp5),
        )
