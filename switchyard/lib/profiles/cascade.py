# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned signal cascade routing construction."""

from __future__ import annotations

import functools
from typing import Any, Self

from switchyard.lib.processors.cascade import CascadeDecisionLog, TierClassifier
from switchyard.lib.processors.cascade_request_processor import (
    BUILTIN_PICKERS,
    CascadeRequestProcessor,
    TierPicker,
)
from switchyard.lib.processors.reasoning_hint import model_accepts_reasoning_hint
from switchyard.lib.profiles.cascade_config import (
    CascadeConfig,
    ClassifierConfig,
)
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config
from switchyard.lib.roles import LLMBackend


@profile_config("cascade")
class CascadeProfileConfig:
    """Profile config wrapper for signal-driven strong/weak cascade profiles."""

    config: CascadeConfig

    @classmethod
    def from_config(cls, config: CascadeConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the cascade profile runtime."""
        from switchyard.lib.backends.multi_llm_backend import build_multi_llm_backend
        from switchyard_rust.components import DimensionCollector

        config = self.config
        request_processors: list[Any] = []
        request_processors.append(
            DimensionCollector(recent_window=config.signal_recent_window)
        )
        decision_log = CascadeDecisionLog()
        classifier = _build_classifier(config.classifier)
        request_processors.append(
            CascadeRequestProcessor(
                targets=(config.weak, config.strong),
                picker=_build_tier_picker(config, decision_log, classifier),
                classifier=classifier,
                decision_log=decision_log,
            )
        )

        backend: LLMBackend = build_multi_llm_backend((config.weak, config.strong))

        return ComponentChainProfile(
            request_processors=request_processors,
            backend=backend,
            fallback_target_on_evict=config.fallback_target_on_evict,
        )


def _build_tier_picker(
    config: CascadeConfig,
    decision_log: CascadeDecisionLog,
    classifier: TierClassifier | None,
) -> TierPicker:
    """Resolve the named cascade picker and bind its runtime knobs."""
    picker_fn = BUILTIN_PICKERS.get(config.picker)
    if picker_fn is None:
        allowed = ", ".join(sorted(BUILTIN_PICKERS))
        raise ValueError(f"unknown picker {config.picker!r}; allowed: {allowed}")
    return functools.partial(
        picker_fn,
        confidence_threshold=config.confidence_threshold,
        classifier=classifier,
        decision_log=decision_log,
    )


def _build_classifier(config: ClassifierConfig | None) -> TierClassifier | None:
    """Build the optional LLM fallback classifier for cascade routing."""
    if config is None:
        return None
    return TierClassifier(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_secs=config.timeout_secs,
        recent_turn_window=config.recent_turn_window,
        disable_reasoning=model_accepts_reasoning_hint(config.model),
    )


__all__ = ["CascadeProfileConfig"]
