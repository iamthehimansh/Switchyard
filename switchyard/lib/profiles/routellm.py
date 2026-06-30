# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned RouteLLM construction."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.roles import LLMBackend


@profile_config("routellm")
class RouteLLMProfileConfig:
    """Profile config wrapper for classifier-driven RouteLLM routing."""

    config: RouteLLMConfig

    @classmethod
    def from_config(cls, config: RouteLLMConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the RouteLLM profile runtime."""
        from switchyard.lib.backends.multi_llm_backend import build_multi_llm_backend
        from switchyard.lib.processors.routellm_request_processor import (
            RouteLLMRequestProcessor,
        )

        config = self.config
        backend: LLMBackend = build_multi_llm_backend(
            (config.strong, config.weak),
            default_target_id=config.strong.id,
        )

        return ComponentChainProfile(
            request_processors=[RouteLLMRequestProcessor(config)],
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


class RouteLLMConfig(BaseModel):
    """Validated parsing config for classifier-driven RouteLLM routing."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    strong: LlmTarget
    weak: LlmTarget
    fallback_target_on_evict: str
    threshold: float = 0.5
    router_type: str = "mf"
    classifier_model: str | None = None
    enable_stats: bool = True

    @field_validator("strong", "weak", mode="before")
    @classmethod
    def _coerce_target(cls, value: object, info: ValidationInfo) -> LlmTarget:
        """Accept dicts and existing ``LlmTarget`` instances for tiers."""
        return coerce_llm_target(value, default_id=info.field_name or "target")

    @field_validator("strong", "weak")
    @classmethod
    def _tier_model_non_empty(cls, tier: LlmTarget) -> LlmTarget:
        """Reject empty model names before classifier startup."""
        if not tier.model:
            raise ValueError("tier.model must be a non-empty string")
        return tier

    @field_validator("threshold")
    @classmethod
    def _threshold_range(cls, value: float) -> float:
        """Validate the RouteLLM decision threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {value!r}")
        return value

    @field_validator("fallback_target_on_evict")
    @classmethod
    def _fallback_matches_existing_target(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        """Ensure evict-and-retry can only fall back to a configured tier."""
        valid_ids = {info.data[key].id for key in ("strong", "weak") if key in info.data}
        if value not in valid_ids:
            raise ValueError(
                f"fallback_target_on_evict={value!r} must match one of "
                f"{sorted(valid_ids)} (the configured target ids)"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_string_models(cls, raw: Any) -> Any:
        """Promote legacy ``strong_model`` / ``weak_model`` strings into tiers."""
        if not isinstance(raw, dict):
            return raw

        result = dict(raw)
        for slot, legacy_key in (("strong", "strong_model"), ("weak", "weak_model")):
            if legacy_key not in result:
                continue
            if slot in result:
                raise ValueError(
                    f"RouteLLMConfig: cannot specify both {slot!r} and "
                    f"{legacy_key!r} — pick one",
                )
            result[slot] = {"model": result.pop(legacy_key)}
        return result

__all__ = ["RouteLLMConfig", "RouteLLMProfileConfig"]
