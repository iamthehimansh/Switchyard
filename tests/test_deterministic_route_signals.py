# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic routing signal types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from switchyard.lib.processors.llm_classifier import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    Complexity,
    ContextDependency,
    PrecisionRequirement,
    ReasonCode,
    ReasoningDepth,
    RiskLevel,
    RouteSignals,
    RouteTier,
    TaskType,
)


def test_route_signals_accepts_minimal_mvp_payload() -> None:
    signals = RouteSignals(
        task_type=TaskType.CODING,
        complexity=Complexity.COMPLEX,
        reasoning_depth=ReasoningDepth.MULTI_STEP,
        tool_planning_required=True,
        precision_requirement=PrecisionRequirement.HIGH,
        context_dependency=ContextDependency.CONVERSATION,
        structured_output_risk=RiskLevel.MEDIUM,
        recommended_tier=RouteTier.COMPLEX,
        confidence=0.82,
        reason_code=ReasonCode.CODING_COMPLEX,
    )

    assert signals.task_type is TaskType.CODING
    assert signals.recommended_tier is RouteTier.COMPLEX
    assert signals.abstain is False
    assert CTX_DETERMINISTIC_ROUTE_SIGNALS == "_deterministic_route_signals"


def test_route_signals_coerce_json_friendly_strings_to_enums() -> None:
    signals = RouteSignals(
        task_type="summarization",
        complexity="simple",
        reasoning_depth="light",
        tool_planning_required=False,
        precision_requirement="medium",
        context_dependency="latest_message",
        structured_output_risk="low",
        recommended_tier="simple",
        confidence=0.91,
        reason_code="summarization",
    )

    assert signals.task_type is TaskType.SUMMARIZATION
    assert signals.reason_code is ReasonCode.SUMMARIZATION


def test_route_signals_reject_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        RouteSignals(
            task_type=TaskType.CHAT,
            complexity=Complexity.SIMPLE,
            reasoning_depth=ReasoningDepth.NONE,
            tool_planning_required=False,
            precision_requirement=PrecisionRequirement.LOW,
            context_dependency=ContextDependency.LATEST_MESSAGE,
            structured_output_risk=RiskLevel.LOW,
            recommended_tier=RouteTier.SIMPLE,
            confidence=1.5,
            reason_code=ReasonCode.SIMPLE_QA,
        )


def test_route_signals_are_immutable() -> None:
    signals = RouteSignals(
        task_type=TaskType.CHAT,
        complexity=Complexity.SIMPLE,
        reasoning_depth=ReasoningDepth.NONE,
        tool_planning_required=False,
        precision_requirement=PrecisionRequirement.LOW,
        context_dependency=ContextDependency.LATEST_MESSAGE,
        structured_output_risk=RiskLevel.LOW,
        recommended_tier=RouteTier.SIMPLE,
        confidence=0.7,
        reason_code=ReasonCode.SIMPLE_QA,
    )

    with pytest.raises(ValidationError):
        signals.confidence = 0.2
