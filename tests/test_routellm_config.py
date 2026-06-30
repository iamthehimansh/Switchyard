# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``RouteLLMConfig`` validation and legacy coercion."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from switchyard.lib.backends.llm_target import (
    BackendFormat,
    LlmTarget,
)
from switchyard.lib.profiles.routellm import RouteLLMConfig


def _tier(model: str = "gpt-4o-mini") -> LlmTarget:
    return LlmTarget(model=model)


class TestRouteLLMConfigBasics:
    def test_defaults(self):
        cfg = RouteLLMConfig(strong=_tier("gpt-4"), weak=_tier("gpt-4o-mini"), fallback_target_on_evict="strong")
        assert cfg.threshold == 0.5
        assert cfg.router_type == "mf"
        assert cfg.classifier_model is None
        assert cfg.enable_stats is True

    def test_frozen(self):
        cfg = RouteLLMConfig(strong=_tier("gpt-4"), weak=_tier("gpt-4o-mini"), fallback_target_on_evict="strong")
        with pytest.raises(ValidationError):
            cfg.threshold = 0.9  # type: ignore[misc]

    def test_threshold_range_low(self):
        with pytest.raises(ValidationError, match="threshold"):
            RouteLLMConfig(strong=_tier("a"), weak=_tier("b"), threshold=-0.1, fallback_target_on_evict="strong")

    def test_threshold_range_high(self):
        with pytest.raises(ValidationError, match="threshold"):
            RouteLLMConfig(strong=_tier("a"), weak=_tier("b"), threshold=1.5, fallback_target_on_evict="strong")

    def test_threshold_zero_and_one_allowed(self):
        # Boundary values stay valid — extreme thresholds are useful for
        # forcing one tier in tests.
        RouteLLMConfig(strong=_tier("a"), weak=_tier("b"), threshold=0.0, fallback_target_on_evict="strong")
        RouteLLMConfig(strong=_tier("a"), weak=_tier("b"), threshold=1.0, fallback_target_on_evict="strong")

    def test_empty_model_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            RouteLLMConfig(strong={"model": ""}, weak=_tier("b"), fallback_target_on_evict="strong")

    def test_tier_format_preserved(self):
        cfg = RouteLLMConfig(
            strong=LlmTarget(model="claude", format=BackendFormat.ANTHROPIC),
            weak=LlmTarget(model="gpt-4o-mini", format=BackendFormat.OPENAI),
        fallback_target_on_evict="strong")
        assert cfg.strong.format is BackendFormat.ANTHROPIC
        assert cfg.weak.format is BackendFormat.OPENAI


class TestRouteLLMLegacyCoercion:
    """Legacy ``routellm_config`` carried bare ``strong_model`` / ``weak_model`` strings."""

    def test_legacy_strings_coerced(self):
        cfg = RouteLLMConfig(  # type: ignore[arg-type]
            **{"strong_model": "gpt-4", "weak_model": "gpt-4o-mini", "threshold": 0.3},
        fallback_target_on_evict="strong")
        assert cfg.strong.model == "gpt-4"
        assert cfg.weak.model == "gpt-4o-mini"
        assert cfg.threshold == 0.3

    def test_legacy_default_format_is_openai(self):
        # Legacy payloads did not carry a format. Keep that compatibility
        # path OpenAI even though direct LlmTarget construction defaults
        # to AUTO.
        cfg = RouteLLMConfig(  # type: ignore[arg-type]
            **{"strong_model": "gpt-4", "weak_model": "gpt-4o-mini"},
        fallback_target_on_evict="strong")
        assert cfg.strong.format is BackendFormat.OPENAI
        assert cfg.weak.format is BackendFormat.OPENAI

    def test_modern_payload_unchanged(self):
        # Modern callers pass typed LlmTarget dicts; the legacy
        # validator is a no-op for them.
        cfg = RouteLLMConfig(
            strong=LlmTarget(model="x", format=BackendFormat.ANTHROPIC),
            weak=_tier("y"),
        fallback_target_on_evict="strong")
        assert cfg.strong.format is BackendFormat.ANTHROPIC

    def test_mixed_strong_str_weak_tier_rejected(self):
        # If one legacy field and one typed field collide on the same
        # slot, we don't try to merge — Pydantic just rejects.
        with pytest.raises(ValidationError):
            RouteLLMConfig(  # type: ignore[arg-type]
                **{
                    "strong_model": "x",
                    "strong": _tier("y"),  # both set on `strong`
                    "weak_model": "z",
                },
            fallback_target_on_evict="strong")
