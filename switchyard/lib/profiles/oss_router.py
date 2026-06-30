# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned external OSS-router plugin construction."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from switchyard.lib.backends.llm_target import LlmTarget, coerce_llm_target
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.roles import LLMBackend


class OSSRouterTier(BaseModel):
    """One tier an external routing plugin may select."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    label: str = Field(min_length=1)
    tier: LlmTarget

    @model_validator(mode="before")
    @classmethod
    def _coerce_tier(cls, raw: Any) -> Any:
        """Accept dict target definitions under the ``tier`` field."""
        if not isinstance(raw, dict) or "tier" not in raw:
            return raw
        result = dict(raw)
        result["tier"] = coerce_llm_target(
            result["tier"],
            default_id=str(result.get("label", "")),
        )
        return result


@profile_config("oss_router")
class OSSRouterProfileConfig:
    """Profile config wrapper for plugin-driven profile routing."""

    config: OSSRouterConfig

    @classmethod
    def from_config(cls, config: OSSRouterConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the OSS-router profile runtime."""
        from switchyard.lib.backends.multi_llm_backend import build_multi_llm_backend
        from switchyard.lib.processors.plugin_routing_request_processor import (
            PluginRoutingRequestProcessor,
        )

        config = self.config
        request_processors: list[Any] = []
        request_processors.append(
            PluginRoutingRequestProcessor(
                plugin_command=config.plugin_command,
                tier_models={t.label: t.tier.model for t in config.tiers},
                tier_target_ids={t.label: t.tier.id for t in config.tiers},
                fallback_tier=config.fallback_tier,
                request_timeout_s=config.request_timeout_s,
                handshake_timeout_s=config.handshake_timeout_s,
                env=config.env,
                expose_metadata_keys=config.expose_metadata_keys,
            )
        )

        backend: LLMBackend = build_multi_llm_backend([t.tier for t in config.tiers])

        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


class OSSRouterConfig(BaseModel):
    """Validated parsing config for plugin-driven profile routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plugin_command: list[str] | str
    tiers: tuple[OSSRouterTier, ...]
    fallback_tier: str | None = None
    fallback_target_on_evict: str
    request_timeout_s: float = 5.0
    handshake_timeout_s: float = 10.0
    env: dict[str, str] | None = None
    expose_metadata_keys: tuple[str, ...] = ()
    enable_stats: bool = True

    @field_validator("plugin_command")
    @classmethod
    def _command_non_empty(cls, value: list[str] | str) -> list[str] | str:
        """Reject empty plugin commands before spawning a subprocess."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("plugin_command must be a non-empty string")
        if isinstance(value, list) and not value:
            raise ValueError("plugin_command must be a non-empty list")
        return value

    @field_validator("tiers")
    @classmethod
    def _tiers_non_empty(cls, value: tuple[OSSRouterTier, ...]) -> tuple[OSSRouterTier, ...]:
        """Require at least one unique plugin-visible tier label."""
        if not value:
            raise ValueError("OSSRouterConfig requires at least one tier")
        labels = [tier.label for tier in value]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"OSSRouterConfig tier labels must be unique; got {labels}",
            )
        return value

    @model_validator(mode="after")
    def _fallback_tier_known(self) -> OSSRouterConfig:
        """Validate fallback labels and target ids against configured tiers."""
        labels = {tier.label for tier in self.tiers}
        if self.fallback_tier is not None and self.fallback_tier not in labels:
            raise ValueError(
                f"fallback_tier {self.fallback_tier!r} is not in "
                f"tiers {sorted(labels)}",
            )
        target_ids = {tier.tier.id for tier in self.tiers}
        if self.fallback_target_on_evict not in target_ids:
            raise ValueError(
                f"fallback_target_on_evict {self.fallback_target_on_evict!r} "
                f"is not in tier target ids {sorted(target_ids)}",
            )
        return self


__all__ = ["OSSRouterConfig", "OSSRouterProfileConfig", "OSSRouterTier"]
