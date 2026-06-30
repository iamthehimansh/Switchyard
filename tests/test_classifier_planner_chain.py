# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production offline e2e test for the classifier + planner composition.

Exercises the chain that pairs LLM-classifier routing with the LLM
planner / executor pattern.  The two are independent primitives
(:mod:`switchyard.lib.processors.llm_classifier` and
:mod:`switchyard.lib.processors.plan_execute`); this file tests them
**together** to confirm classifier-stamped signals do not interfere
with planner injection and vice versa.

Chain shape::

    inbound HTTP request
       -> Switchyard chain
           LLMClassifierRequestProcessor    (real OpenAILLMClient -> openai SDK -> httpx)
           SignalTierSelectorRequestProcessor
           PlanningRequestProcessor          (real OpenAILLMClient -> openai SDK -> httpx)
       -> DeterministicRoutingLLMBackend
           -> per-tier Rust OpenAI backend           (real reqwest -> loopback HTTP)
       -> TranslationEngine
       -> outbound HTTP response

Classifier and planner HTTPS calls are intercepted by ``respx`` on
distinct URLs.  Backend tier calls go through local loopback HTTP
servers because Rust reqwest bypasses ``respx``.

Dedicated planner-only tests live in
:mod:`tests.test_planning_request_processor` (mock planner client) and
classifier-only tests in :mod:`tests.test_llm_classifier_e2e`.  This
file is intentionally focused on **composition**.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    DeterministicRoutingLLMBackend,
)
from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.processors.llm_classifier import (
    LLMClassifierConfig,
    LLMClassifierRequestProcessor,
    SignalTierSelectorConfig,
    SignalTierSelectorRequestProcessor,
)
from switchyard.lib.processors.plan_execute import (
    PlanningConfig,
    PlanningRequestProcessor,
)
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.translation import TranslationEngine
from tests._chain_test_helpers import (
    _CLASSIFIER_BASE,
    _CLASSIFIER_MODEL,
    _CLASSIFIER_URL,
    _COMPLEX_MODEL,
    _SIMPLE_MODEL,
    _backend_payload,
    _classifier_payload,
    _ClassifierHarness,
    _last_body,
    _OpenAICompatStub,
    _signals_json,
)

# ---------------------------------------------------------------------------
# Planner-specific URLs / payloads (orthogonal to the classifier helpers, so
# they live here rather than in tests/_chain_test_helpers.py).
# ---------------------------------------------------------------------------

_PLANNER_BASE = "https://planner.test/v1"
_PLANNER_URL = f"{_PLANNER_BASE}/chat/completions"

_PLANNER_MODEL = "router-planner-llm"


def _planner_payload(content: str) -> dict[str, object]:
    """An OpenAI Chat Completion JSON body whose content is the planner output."""
    return {
        "id": "chatcmpl-planner",
        "object": "chat.completion",
        "created": 1700000003,
        "model": _PLANNER_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
    }


_PLAN_TEXT = (
    "I'll approach the traceback systematically:\n"
    "1. Inspect the traceback and failing path — identify the component "
    "and likely regression surface.\n"
    "2. Patch the narrow failure, keeping the change scoped.\n"
    "3. Run focused regression tests to verify the fix.\n\n"
    "Risk: the traceback may be a symptom of upstream request mutation.\n\n"
    "Starting with step 1 now."
)
_WRAPPED_PLAN_TEXT = f"<plan>\n{_PLAN_TEXT}\n</plan>"


def _plan_decision_json(plan_text: str = _PLAN_TEXT) -> str:
    """Stub planner output: a ``plan_needed=True`` decision with literal plan text."""
    return json.dumps({"plan_needed": True, "plan_text": plan_text})


# ---------------------------------------------------------------------------
# Fixture — assembles the classifier + planner composition by hand.
# ---------------------------------------------------------------------------


def _build_switchyard_with_planner(
    *, simple_base_url: str, complex_base_url: str,
) -> Switchyard:
    """Classifier routing chain plus plan-enrichment processor.

    Order mirrors the production wiring: classifier and tier selector
    first, planner last among the request processors so the plan is
    injected after the tier has been pinned.
    """
    classifier = LLMClassifierRequestProcessor(
        LLMClassifierConfig(
            model=_CLASSIFIER_MODEL,
            api_key="classifier-key-not-validated",
            base_url=_CLASSIFIER_BASE,
        ),
    )
    tier_selector = SignalTierSelectorRequestProcessor(
        SignalTierSelectorConfig(default_tier="simple"),
    )
    planner = PlanningRequestProcessor(
        PlanningConfig(
            model=_PLANNER_MODEL,
            api_key="planner-key-not-validated",
            base_url=_PLANNER_BASE,
            inject_plan=True,
        ),
    )
    backend = DeterministicRoutingLLMBackend.from_tiers(
        tiers={
            "simple": LlmTarget(
                model=_SIMPLE_MODEL,
                format=BackendFormat.OPENAI,
                api_key="simple-key",
                base_url=simple_base_url,
            ),
            "complex": LlmTarget(
                model=_COMPLEX_MODEL,
                format=BackendFormat.OPENAI,
                api_key="complex-key",
                base_url=complex_base_url,
            ),
        },
        default_tier="simple",
    )
    return Switchyard(
        request_processors=[classifier, tier_selector, planner],
        backend=backend,
        translator=TranslationEngine(),
    )


@pytest.fixture
async def classifier_planner_harness() -> AsyncIterator[_ClassifierHarness]:
    """ASGI-transport client wired through classifier, selector, planner, backend."""
    with _OpenAICompatStub() as simple, _OpenAICompatStub() as complex:
        app = build_switchyard_app(
            _build_switchyard_with_planner(
                simple_base_url=simple.base_url,
                complex_base_url=complex.base_url,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            yield _ClassifierHarness(client=client, simple=simple, complex=complex)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifierPlannerChainE2E:
    """Production offline e2e — classifier routes, planner enriches, backend serves."""

    @respx.mock
    async def test_planner_injects_plan_before_backend_dispatch(
        self, classifier_planner_harness: _ClassifierHarness,
    ) -> None:
        """Planner runs as an independent feature and injects plan into backend prompt."""
        classifier_route = respx.post(_CLASSIFIER_URL).mock(
            return_value=httpx.Response(
                200,
                json=_classifier_payload(
                    _signals_json(
                        task_type="coding",
                        complexity="complex",
                        reasoning_depth="multi_step",
                        recommended_tier="complex",
                        confidence=0.92,
                        reason_code="coding_complex",
                    ),
                ),
            )
        )
        planner_route = respx.post(_PLANNER_URL).mock(
            return_value=httpx.Response(
                200,
                json=_planner_payload(_plan_decision_json()),
            )
        )
        classifier_planner_harness.complex.respond_json(
            _backend_payload(content="planned-complex-reply", model=_COMPLEX_MODEL)
        )
        classifier_planner_harness.simple.respond_json(
            _backend_payload(content="simple-reply", model=_SIMPLE_MODEL)
        )

        resp = await classifier_planner_harness.client.post(
            "/v1/chat/completions",
            json={
                "model": "client-requested-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "Please debug this traceback and implement the fix.",
                    }
                ],
            },
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "planned-complex-reply"

        assert classifier_route.called
        assert planner_route.called
        assert classifier_planner_harness.complex.called
        assert not classifier_planner_harness.simple.called

        planner_req = json.loads(planner_route.calls.last.request.content)
        assert planner_req["model"] == _PLANNER_MODEL
        assert planner_req.get("response_format") == {"type": "json_object"}
        # The chain has no prior <plan> in history → initial planner
        # prompt (which casts the planner as drafting the agent's
        # opening thought).
        assert "opening thought" in planner_req["messages"][0]["content"].lower()
        assert "Please debug this traceback" in planner_req["messages"][1]["content"]
        # The planner sees the inbound body but NOT the classifier's
        # RouteSignals — the two primitives are decoupled at the prompt
        # level too, not just at the import level.
        assert '"signals"' not in planner_req["messages"][1]["content"]

        backend_req = _last_body(classifier_planner_harness.complex)
        assert backend_req["model"] == _COMPLEX_MODEL
        messages = backend_req["messages"]
        # User turn is preserved at messages[0]; plan is prefilled as
        # the trailing assistant turn (the executor continues from
        # there).  The prefill is wrapped in ``<plan>...</plan>`` so
        # the executor sees an explicit, delimited block during
        # generation.  The plan is never echoed back to the client.
        assert messages[0]["role"] == "user"
        assert "Please debug this traceback" in messages[0]["content"]
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == _WRAPPED_PLAN_TEXT
