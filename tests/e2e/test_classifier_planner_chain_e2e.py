# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-backend e2e test for the classifier + planner composition.

Exercises the full chain that pairs LLM-classifier routing with the
LLM planner / executor pattern.  The two are independent primitives
(:mod:`switchyard.lib.processors.llm_classifier` and
:mod:`switchyard.lib.processors.plan_execute`); this file tests them
**together** to confirm classifier-stamped signals do not interfere
with planner injection and vice versa, end to end against a live backend.

Chain shape::

    inbound ChatRequest
       -> LLMClassifierRequestProcessor     -> real classifier LLM call
       -> SignalTierSelectorRequestProcessor
       -> PlanningRequestProcessor          -> real planner LLM call
       -> DeterministicRoutingLLMBackend
           -> per-tier OpenAiPassthroughBackend     -> real tier LLM call
                                               (request has planner system
                                                prompt injected)
       -> ChatResponse

Sibling pattern: in-process — drive the processors + backend manually
rather than via ``Switchyard.call()`` so we get back the typed
:class:`ChatResponse`.

Dedicated planner-only tests live in
:mod:`tests.e2e.test_planning_request_processor_e2e`; this file is
intentionally focused on **composition**.

Prerequisites:
    - ``OPENROUTER_API_KEY`` or ``NVIDIA_API_KEY`` env var (skips otherwise).

Run with::

    OPENROUTER_API_KEY=sk-or-... uv run pytest \
        tests/e2e/test_classifier_planner_chain_e2e.py -v
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.chat_response.base import ChatResponse
from switchyard.lib.processors.llm_classifier import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    CTX_DETERMINISTIC_TIER_DECISION,
    LLMClassifierConfig,
    LLMClassifierRequestProcessor,
    RouteSignals,
    SignalTierSelectorConfig,
    SignalTierSelectorRequestProcessor,
    TierSelectionDecision,
)
from switchyard.lib.processors.plan_execute import (
    CTX_PLANNER_DECISION,
    PlannerDecision,
    PlanningConfig,
    PlanningRequestProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequest

from .conftest import get_nvidia_config

# Three real LLM calls per test (classifier + planner + tier).  120s is
# enough headroom for the slowest case under typical live-backend load
# while still failing fast if anything hangs.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]

logger = logging.getLogger("e2e.classifier_planner_chain")

HTTP_TIMEOUT_S = 60.0

_nvidia = get_nvidia_config()
_skip_reason = "OPENROUTER_API_KEY or NVIDIA_API_KEY not set"

BACKEND_BASE_URL = _nvidia["base_url"]

# Same model id for the classifier, planner, and tier backends.  This
# test validates end-to-end wiring of the composition, not relative
# capability per role.
CLASSIFIER_MODEL = _nvidia["model"]
PLANNER_MODEL = _nvidia["model"]
BACKEND_MODEL = _nvidia["model"]

# Tier labels mirror the four ``RouteTier`` values so whichever tier
# the live classifier picks resolves to a valid backend.
TIER_LABELS = ("simple", "medium", "complex", "reasoning")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_request(
    *,
    user_message: str,
    model: str = "placeholder-rewritten-by-router",
    **body_overrides: Any,
) -> ChatRequest:
    """Build an OpenAI chat :class:`ChatRequest` for the chain.

    The default ``model`` is a placeholder because the deterministic
    routing backend rewrites the model per tier.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 1024,
    }
    body.update(body_overrides)
    return ChatRequest.openai_chat(body)


def _build_classifier_planner_chain() -> tuple[
    list[Any], DeterministicRoutingLLMBackend
]:
    """Build (processors, backend) for the classifier + planner chain.

    Order: classifier -> tier selector -> planner -> tier backend.
    Putting planner *after* the tier selector keeps signal stamping
    (which the planner doesn't read) independent of plan generation.
    """
    classifier = LLMClassifierRequestProcessor(
        LLMClassifierConfig(
            model=CLASSIFIER_MODEL,
            api_key=_nvidia["api_key"],
            base_url=BACKEND_BASE_URL,
            timeout_s=HTTP_TIMEOUT_S,
        ),
    )
    tier_selector = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(default_tier="simple"),
    )
    planner = PlanningRequestProcessor(
        PlanningConfig(
            model=PLANNER_MODEL,
            api_key=_nvidia["api_key"],
            base_url=BACKEND_BASE_URL,
            timeout_s=HTTP_TIMEOUT_S,
            inject_plan=True,
        ),
    )
    backend = DeterministicRoutingLLMBackend.from_tiers(
        tiers={
            label: LlmTarget(
                model=BACKEND_MODEL,
                format=BackendFormat.OPENAI,
                api_key=_nvidia["api_key"],
                base_url=BACKEND_BASE_URL,
                timeout=HTTP_TIMEOUT_S,
            )
            for label in TIER_LABELS
        },
        default_tier="simple",
    )
    return [classifier, tier_selector, planner], backend


async def _call_through_chain(
    processors: list[Any],
    backend: LLMBackend,
    request: ChatRequest,
) -> tuple[ProxyContext, ChatResponse]:
    """Run a request through processors + backend in-process."""
    ctx = ProxyContext()
    for proc in processors:
        request = await proc.process(ctx, request)
    response = await backend.call(ctx, request)
    return ctx, response


def _assistant_text(response: ChatResponse) -> tuple[str, str]:
    choice = response.body["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    return content, reasoning


def _finish_reason(response: ChatResponse) -> object:
    return response.body["choices"][0].get("finish_reason")


# ---------------------------------------------------------------------------
# Tests — classifier + planner + tier backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestClassifierPlannerChainLiveE2E:
    """Full chain: real classifier + real planner + real tier backend."""

    async def test_classifier_planner_chain_routes_plans_and_serves(self) -> None:
        """End-to-end: classifier picks a tier, planner enriches, backend answers.

        Asserts that:

        * the classifier stamped :class:`RouteSignals` and the tier
          selector stamped a :class:`TierSelectionDecision`
        * the planner stamped a typed :class:`ExecutionPlan` and
          rendered plan text on ctx
        * the rendered plan text was prepended as a ``system`` turn on
          the request the backend received
        * the tier backend rewrote the model name to ``BACKEND_MODEL``
        * the live backend produced non-empty assistant text
        """
        processors, backend = _build_classifier_planner_chain()
        # Task-shaped prompt: the TB-Lite-tuned planner fast-paths
        # ("how would you ...") Q&A asks with ``plan_needed=false``,
        # so we phrase as a concrete multi-step engineering task.
        request = _chat_request(
            user_message=(
                "Debug a slow Postgres query in /app/db/reports.py "
                "that regresses under concurrent load. Determine "
                "whether the cause is lock contention or plan-cache "
                "pollution, then implement a fix and verify the "
                "p99 latency stays under 100ms at 50-way concurrency."
            ),
        )

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)

        signals = ctx.metadata.get(CTX_DETERMINISTIC_ROUTE_SIGNALS)
        assert isinstance(signals, RouteSignals), (
            f"classifier did not stamp RouteSignals; metadata keys: "
            f"{list(ctx.metadata.keys())}"
        )

        decision = ctx.metadata.get(CTX_DETERMINISTIC_TIER_DECISION)
        assert isinstance(decision, TierSelectionDecision)
        tier = ctx.metadata.get(CTX_DETERMINISTIC_ROUTING_TIER)
        assert tier in set(TIER_LABELS), f"unexpected tier picked: {tier!r}"
        assert decision.tier == tier

        decision = ctx.metadata.get(CTX_PLANNER_DECISION)
        assert isinstance(decision, PlannerDecision), (
            f"planner did not stamp PlannerDecision; metadata keys: "
            f"{list(ctx.metadata.keys())}"
        )
        assert decision.plan_needed, (
            f"expected plan_needed=True for a multi-step debugging prompt; "
            f"planner declined with plan_text_len={len(decision.plan_text)}"
        )
        assert decision.plan_text.strip(), "PlannerDecision.plan_text was empty"

        messages = request.body["messages"]
        # Original user turn preserved at messages[0]; plan prefilled
        # as the trailing assistant turn (the executor continues from
        # there).  The processor wraps plan_text in ``<plan>...</plan>``
        # before appending; we validate the prefill directly since the
        # plan is never echoed back to the client.
        assert messages[0]["role"] == "user"
        assert "slow Postgres query" in messages[0]["content"]
        assert messages[-1]["role"] == "assistant"
        prefill_content = messages[-1]["content"]
        assert "<plan>" in prefill_content and "</plan>" in prefill_content
        assert decision.plan_text in prefill_content

        actual_model = ctx.selected_model
        assert actual_model == BACKEND_MODEL, (
            f"unexpected actual model: {actual_model!r}"
        )

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        logger.info(
            "  [classifier+planner-e2e] tier=%s task=%s recommended=%s "
            "confidence=%.2f plan_text_len=%d content[:80]=%r",
            tier,
            signals.task_type.value, signals.recommended_tier.value,
            signals.confidence,
            len(decision.plan_text), content[:80],
        )
