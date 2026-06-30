# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cascade — weighted scorer + selective LLM-classifier."""

from switchyard.lib.processors.cascade.classifier import (
    STRONG_TIER,
    WEAK_TIER,
    TierClassifier,
)
from switchyard.lib.processors.cascade.decision_log import (
    CONTEXT_KEY,
    CascadeDecisionLog,
    DecisionSource,
)
from switchyard.lib.processors.cascade.dimensions import (
    CodingAgentDimensions,
    from_signal,
)
from switchyard.lib.processors.cascade.picker import (
    STRONG,
    WEAK,
    pick_strong_default,
    pick_weak_default,
)
from switchyard.lib.processors.cascade.scorer import (
    DEFAULT_WEIGHTS,
    ScoreResult,
    score,
)

__all__ = [
    "CONTEXT_KEY",
    "DEFAULT_WEIGHTS",
    "STRONG",
    "STRONG_TIER",
    "CascadeDecisionLog",
    "CodingAgentDimensions",
    "DecisionSource",
    "ScoreResult",
    "TierClassifier",
    "WEAK",
    "WEAK_TIER",
    "from_signal",
    "pick_strong_default",
    "pick_weak_default",
    "score",
]
