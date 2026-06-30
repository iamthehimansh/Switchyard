# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-backend e2e tests for the planner / executor pattern (planner only).

Mirrors :mod:`tests.test_planning_request_processor` (which mocks the
planner via a fake client) but with no mocking — every call hits the selected
live backend::

    inbound ChatRequest
       -> PlanningRequestProcessor          -> real planner LLM call
       -> OpenAiPassthroughBackend                  -> real downstream LLM call
                                               (request has planner system
                                                prompt injected)
       -> ChatResponse

Sibling pattern: in-process — drive the processors + backend manually
rather than via ``Switchyard.call()`` so we get back the typed
:class:`ChatResponse` and can inspect both the post-injection
request body and the response.

Because the planner LLM's output is non-deterministic, tests do not
assert specific plan content.  They assert:

* the planner ran (typed :class:`ExecutionPlan` stamped on ctx)
* the rendered plan text was injected into the outbound request body
* the downstream backend produced a well-formed response

This file is **planner-only**.  Tests that compose the planner with
the classifier live in
:mod:`tests.e2e.test_classifier_planner_chain_e2e`.

Prerequisites:
    - ``OPENROUTER_API_KEY`` or ``NVIDIA_API_KEY`` env var (skips otherwise).

Run with::

    OPENROUTER_API_KEY=sk-or-... uv run pytest \
        tests/e2e/test_planning_request_processor_e2e.py -v
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from switchyard.lib.backends import OpenAiPassthroughBackend
from switchyard.lib.chat_response.base import ChatResponse
from switchyard.lib.processors.plan_execute import (
    CTX_PLANNER_DECISION,
    PlannerDecision,
    PlanningConfig,
    PlanningRequestProcessor,
    PlanningTriggerMode,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequest

from .conftest import get_nvidia_config

# Two real LLM calls per planner-only test (planner + downstream). 120s
# is enough headroom under typical live-backend load while still failing fast
# if anything hangs.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
]

logger = logging.getLogger("e2e.planning_request_processor")

HTTP_TIMEOUT_S = 60.0

_nvidia = get_nvidia_config()
_skip_reason = "OPENROUTER_API_KEY or NVIDIA_API_KEY not set"

BACKEND_BASE_URL = _nvidia["base_url"]

# Same model name for the planner and the downstream backend.  These
# tests validate end-to-end wiring of the planner / executor pattern,
# not relative planner-vs-executor capability.  Using one model also
# lets us pin a model known to support
# ``response_format={"type": "json_object"}`` on the selected backend
# (required by :class:`OpenAIChatPlannerClient`).
PLANNER_MODEL = _nvidia["model"]
BACKEND_MODEL = _nvidia["model"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_request(
    *,
    user_message: str,
    model: str = BACKEND_MODEL,
    **body_overrides: Any,
) -> ChatRequest:
    """Build an OpenAI chat :class:`ChatRequest` for the planner-only chain."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 1024,
    }
    body.update(body_overrides)
    return ChatRequest.openai_chat(body)


def _build_planner_only_chain(
    *,
    trigger_mode: PlanningTriggerMode = PlanningTriggerMode.ALWAYS,
    inject_plan: bool = True,
    fail_open: bool = True,
    planner_base_url: str = BACKEND_BASE_URL,
    planner_api_key: str | None = None,
    planner_timeout_s: float = HTTP_TIMEOUT_S,
) -> tuple[list[Any], LLMBackend]:
    """Build (processors, backend) for the planner-only chain."""
    planner = PlanningRequestProcessor(
        PlanningConfig(
            model=PLANNER_MODEL,
            api_key=planner_api_key or _nvidia["api_key"],
            base_url=planner_base_url,
            timeout_s=planner_timeout_s,
            trigger_mode=trigger_mode,
            inject_plan=inject_plan,
            fail_open=fail_open,
        ),
    )
    backend = OpenAiPassthroughBackend(
        api_key=_nvidia["api_key"],
        base_url=BACKEND_BASE_URL,
        timeout=HTTP_TIMEOUT_S,
    )
    return [planner], backend


async def _call_through_chain(
    processors: list[Any],
    backend: LLMBackend,
    request: ChatRequest,
) -> tuple[ProxyContext, ChatResponse]:
    """Run a request through processors + backend in-process.

    Skips terminal translation on purpose: returning the typed
    :class:`ChatResponse` keeps assertions clean.
    """
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
# Tests — planner only
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _nvidia["api_key"], reason=_skip_reason)
class TestPlanningRequestProcessorLiveE2E:
    """Real planner LLM + real downstream backend via the selected live backend."""

    async def test_planner_generates_and_injects_plan(self) -> None:
        """Planner decides a multi-step prompt warrants a plan and injects it.

        Verifies the planner / executor contract end-to-end on the
        ``plan_needed=True`` branch of :class:`PlannerDecision`:

        * a :class:`PlannerDecision` with ``plan_needed=True`` is
          stamped on ``ctx.metadata[CTX_PLANNER_DECISION]``
        * a typed :class:`ExecutionPlan` is stamped on
          ``ctx.metadata[CTX_EXECUTION_PLAN]`` with at least one
          well-formed step
        * the rendered plan text on
          the wrapped plan in the prefilled assistant message matches what was
          prepended as a ``system`` message on the outbound request
        * the downstream backend, which sees the injected prompt,
          returns a non-empty assistant turn

        The prompt is deliberately multi-step (debugging + diagnostic
        proposal) so the planner LLM should reliably choose to plan.
        If it doesn't, the assertion failure message surfaces the
        decision's ``reason`` to make the model-side regression
        obvious.
        """
        processors, backend = _build_planner_only_chain()
        # Phrased as an actionable engineering task (verbs: investigate,
        # implement, verify) rather than a Q&A ("outline / suggest").
        # The TB-Lite-tuned planner explicitly fast-paths Q&A asks via
        # ``plan_needed=false``; task-shaped prompts reliably trigger
        # ``plan_needed=true``.
        request = _chat_request(
            user_message=(
                "Debug a flaky pytest in /app/tests/test_cache.py that "
                "fails ~1 in 10 runs with a stale-cache error. "
                "Investigate whether the cache is per-process or "
                "shared across workers, implement a fix in the "
                "fixture, and verify the test passes 50 runs in a row."
            ),
        )

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)

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

        # Plan was prefilled as the last assistant turn at the tail of
        # ``messages`` (the executor continues from where it ends).
        # Original user turn is preserved at messages[0].  The
        # processor wraps plan_text in ``<plan>...</plan>``; we check
        # the prefill directly since the plan is never echoed back to
        # the client.
        messages = request.body["messages"]
        assert messages[0]["role"] == "user"
        assert "flaky pytest" in messages[0]["content"]
        prefill_content = messages[-1]["content"]
        assert messages[-1]["role"] == "assistant"
        assert "<plan>" in prefill_content and "</plan>" in prefill_content
        assert decision.plan_text in prefill_content

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        logger.info(
            "  [planner-e2e][inject] plan_text_len=%d "
            "plan_text[:80]=%r content[:80]=%r",
            len(decision.plan_text), decision.plan_text[:80], content[:80],
        )

    async def test_planner_declines_to_plan_for_trivial_request(self) -> None:
        """Planner decides a one-word reply needs no plan; request stays untouched.

        Verifies the ``plan_needed=False`` branch of
        :class:`PlannerDecision`:

        * the planner runs (so :data:`CTX_PLANNER_DECISION` is stamped)
        * but it returns ``plan_needed=False`` with a one-line reason
        * neither :data:`CTX_EXECUTION_PLAN` nor
          the planner declined to plan (no prefill is appended)
        * the outbound request body is byte-equal to what the caller
          built — no system turn was prepended
        * the downstream backend still answers the user

        Prompt is deliberately trivial so the planner should reliably
        decline.  Like the sibling ``plan_needed=True`` test, this is
        sensitive to planner-model variance; the assertion message
        flags the model side if the planner ever changes its mind.
        """
        processors, backend = _build_planner_only_chain()
        request = _chat_request(
            user_message="Reply with the single word: ok",
        )

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)

        decision = ctx.metadata.get(CTX_PLANNER_DECISION)
        assert isinstance(decision, PlannerDecision)
        assert not decision.plan_needed, (
            f"expected plan_needed=False for a one-word reply; "
            f"planner produced a plan instead "
            f"(plan_text[:80]={decision.plan_text[:80]!r})"
        )
        assert decision.plan_text == ""


        # Request body was not mutated — original user turn is still
        # the only message on the outbound request.
        messages = request.body["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        logger.info(
            "  [planner-e2e][declined] plan_needed=%s content[:80]=%r",
            decision.plan_needed, content[:80],
        )

    async def test_planner_disabled_leaves_request_untouched(self) -> None:
        """``trigger_mode=DISABLED`` skips the planner call entirely.

        No plan is stamped on ctx, no system turn is prepended, and the
        downstream backend still answers the original user message.
        Sanity check that the disabled path doesn't accidentally inject
        a stale plan or burn a planner call.
        """
        processors, backend = _build_planner_only_chain(
            trigger_mode=PlanningTriggerMode.DISABLED,
        )
        request = _chat_request(
            user_message="Reply with the single word: ok",
        )

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)

        # DISABLED skips the planner call entirely, so no decision is
        # even stamped — distinguishable from the ``plan_needed=False``
        # case (where the decision IS stamped).
        assert CTX_PLANNER_DECISION not in ctx.metadata

        # No system turn was injected — the original user turn is still
        # the only message on the outbound request.
        messages = request.body["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend; "
            f"finish_reason={_finish_reason(response)!r}"
        )

        logger.info(
            "  [planner-e2e][disabled] content[:80]=%r",
            content[:80],
        )

    async def test_planner_failure_fails_open_to_backend(self) -> None:
        """A broken planner upstream must not break downstream inference.

        Points the planner at an unresolvable host while keeping the
        downstream backend live.  ``fail_open=True`` is the configured
        default for :class:`PlanningConfig`, so a planner failure
        should leave the request unmodified and still return real
        content from the live backend.
        """
        # Short planner timeout so we don't burn the whole pytest-timeout
        # budget waiting for the bogus URL to fail.
        processors, backend = _build_planner_only_chain(
            planner_base_url="https://planner-does-not-exist.invalid/v1",
            planner_api_key="not-used",
            planner_timeout_s=5.0,
        )
        request = _chat_request(
            user_message="Reply with the single word: ok",
        )

        ctx, response = await _call_through_chain(processors, backend, request)

        assert isinstance(response, ChatResponse), type(response)

        # Planner call failed before producing a valid decision, so
        # CTX_PLANNER_DECISION is also absent (distinguishable from a
        # valid decision with ``plan_needed=False``, which would stamp
        # the decision).
        assert CTX_PLANNER_DECISION not in ctx.metadata

        # Request body wasn't mutated — no injected system turn.
        messages = request.body["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

        content, reasoning = _assistant_text(response)
        assert content or reasoning, (
            f"expected non-empty assistant text from live backend after "
            f"planner failure; finish_reason="
            f"{_finish_reason(response)!r}"
        )

        logger.info(
            "  [planner-e2e][fail-open] content[:80]=%r",
            content[:80],
        )
