# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM classifier profile bundles."""

from __future__ import annotations

import json
from typing import Any, cast

from switchyard.lib.processors.llm_classifier import (
    CTX_DETERMINISTIC_ROUTE_SIGNALS,
    DEFAULT_MAX_REQUEST_CHARS,
    ChannelKind,
    ClassifierCompletion,
    CodeModificationScope,
    CodingAgentRouteDecision,
    CodingAgentTurnType,
    LLMClassifierPresets,
    LLMClassifierProfile,
    LLMClassifierRequestProcessor,
    MemoryDependency,
    OpenClawRouteDecision,
    OpenClawTurnType,
    RouteSignals,
    RouteTier,
    SignalTierSelectorConfig,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

WEAK = "weak"
STRONG = "strong"


class _FakeClassifierClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.system_prompt: str | None = None

    async def classify(
        self,
        *,
        model: str,
        system_prompt: str,
        request_summary: str,
    ) -> ClassifierCompletion:
        self.system_prompt = system_prompt
        return ClassifierCompletion(content=self.response)


def _request() -> ChatRequest:
    body: dict[str, Any] = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "list files in src"}],
    }
    return ChatRequest.openai_chat(cast(Any, body))


def test_general_profile_keeps_conservative_mapping() -> None:
    profile = LLMClassifierPresets.general_2_tier(weak=WEAK, strong=STRONG)

    assert profile.name == "general_2_tier"
    assert profile.signal_schema is RouteSignals
    assert profile.tier_mapping[RouteTier.SIMPLE] == WEAK
    assert profile.tier_mapping[RouteTier.MEDIUM] == STRONG
    assert profile.tier_mapping[RouteTier.COMPLEX] == STRONG
    assert profile.tier_mapping[RouteTier.REASONING] == STRONG
    assert profile.default_tier == STRONG


def test_coding_agent_profile_routes_simple_and_medium_to_weak() -> None:
    profile = LLMClassifierPresets.coding_agent_2_tier(weak=WEAK, strong=STRONG)

    assert profile.name == "coding_agent_2_tier"
    assert profile.signal_schema is CodingAgentRouteDecision
    assert profile.tier_mapping[RouteTier.SIMPLE] == WEAK
    assert profile.tier_mapping[RouteTier.MEDIUM] == WEAK
    assert profile.tier_mapping[RouteTier.COMPLEX] == STRONG
    assert profile.tier_mapping[RouteTier.REASONING] == STRONG
    assert profile.default_tier == STRONG


def test_openclaw_profile_routes_simple_and_medium_to_weak() -> None:
    profile = LLMClassifierPresets.openclaw_2_tier(weak=WEAK, strong=STRONG)

    assert profile.name == "openclaw_2_tier"
    assert profile.signal_schema is OpenClawRouteDecision
    assert profile.tier_mapping[RouteTier.SIMPLE] == WEAK
    assert profile.tier_mapping[RouteTier.MEDIUM] == WEAK
    assert profile.tier_mapping[RouteTier.COMPLEX] == STRONG
    assert profile.tier_mapping[RouteTier.REASONING] == STRONG
    assert profile.default_tier == STRONG


def test_profile_make_classifier_config_bakes_in_system_prompt() -> None:
    profile = LLMClassifierPresets.coding_agent_2_tier(weak=WEAK, strong=STRONG)
    config = profile.make_classifier_config(model="router-model")

    assert config.system_prompt == profile.system_prompt
    assert config.model == "router-model"
    assert config.fail_open is True
    assert config.max_request_chars == DEFAULT_MAX_REQUEST_CHARS


def test_profile_make_classifier_config_accepts_prompt_and_context_override() -> None:
    profile = LLMClassifierPresets.coding_agent_2_tier(weak=WEAK, strong=STRONG)
    config = profile.make_classifier_config(
        model="router-model",
        system_prompt="custom prompt",
        max_request_chars=1024,
    )

    assert config.system_prompt == "custom prompt"
    assert config.max_request_chars == 1024


def test_profile_make_tier_selector_config_matches_mapping() -> None:
    profile = LLMClassifierPresets.openclaw_2_tier(weak=WEAK, strong=STRONG)
    config = profile.make_tier_selector_config()

    assert isinstance(config, SignalTierSelectorConfig)
    assert dict(config.tier_mapping) == dict(profile.tier_mapping)
    assert config.default_tier == profile.default_tier
    assert config.min_confidence == 0.0


def test_profile_min_confidence_override() -> None:
    profile = LLMClassifierPresets.coding_agent_2_tier(weak=WEAK, strong=STRONG)
    config = profile.make_tier_selector_config(min_confidence=0.55)

    assert config.min_confidence == 0.55


async def test_coding_agent_profile_round_trips_through_request_processor() -> None:
    profile = LLMClassifierPresets.coding_agent_2_tier(weak=WEAK, strong=STRONG)
    payload = {
        "recommended_tier": "simple",
        "confidence": 0.91,
        "abstain": False,
        "turn_type": "exploration",
        "code_modification_scope": "none",
        "tool_call_count_estimate": 2,
        "requires_codebase_context": False,
    }
    fake = _FakeClassifierClient(json.dumps(payload))

    processor = LLMClassifierRequestProcessor(
        profile.make_classifier_config(model="router-model"),
        client=fake,
        signal_schema=profile.signal_schema,
    )
    ctx = ProxyContext()
    await processor.process(ctx, _request())

    stamped = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
    assert isinstance(stamped, CodingAgentRouteDecision)
    assert stamped.recommended_tier is RouteTier.SIMPLE
    assert stamped.turn_type is CodingAgentTurnType.EXPLORATION
    assert stamped.code_modification_scope is CodeModificationScope.NONE
    assert stamped.tool_call_count_estimate == 2
    assert fake.system_prompt == profile.system_prompt


async def test_openclaw_profile_round_trips_through_request_processor() -> None:
    profile = LLMClassifierPresets.openclaw_2_tier(weak=WEAK, strong=STRONG)
    payload = {
        "recommended_tier": "simple",
        "confidence": 0.84,
        "abstain": False,
        "turn_type": "lookup",
        "tool_call_count_estimate": 1,
        "memory_dependency": "light",
        "external_action_required": False,
        "precision_requirement": "low",
        "ambiguity": "low",
        "channel_kind": "casual",
    }
    fake = _FakeClassifierClient(json.dumps(payload))

    processor = LLMClassifierRequestProcessor(
        profile.make_classifier_config(model="router-model"),
        client=fake,
        signal_schema=profile.signal_schema,
    )
    ctx = ProxyContext()
    await processor.process(ctx, _request())

    stamped = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
    assert isinstance(stamped, OpenClawRouteDecision)
    assert stamped.recommended_tier is RouteTier.SIMPLE
    assert stamped.turn_type is OpenClawTurnType.LOOKUP
    assert stamped.memory_dependency is MemoryDependency.LIGHT
    assert stamped.channel_kind is ChannelKind.CASUAL
    assert fake.system_prompt == profile.system_prompt


async def test_classifier_fail_open_uses_profile_schema_for_abstain() -> None:
    profile = LLMClassifierPresets.coding_agent_2_tier(weak=WEAK, strong=STRONG)

    class _Boom:
        async def classify(self, **_: Any) -> ClassifierCompletion:
            raise RuntimeError("classifier exploded")

    processor = LLMClassifierRequestProcessor(
        profile.make_classifier_config(model="router-model"),
        client=_Boom(),
        signal_schema=profile.signal_schema,
    )
    ctx = ProxyContext()
    await processor.process(ctx, _request())

    stamped = ctx.metadata[CTX_DETERMINISTIC_ROUTE_SIGNALS]
    assert isinstance(stamped, CodingAgentRouteDecision)
    assert stamped.abstain is True
    assert stamped.confidence == 0.0


def test_profile_is_immutable_dataclass() -> None:
    profile = LLMClassifierPresets.general_2_tier(weak=WEAK, strong=STRONG)
    assert isinstance(profile, LLMClassifierProfile)
    try:
        profile.name = "mutated"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected LLMClassifierProfile to be frozen")
