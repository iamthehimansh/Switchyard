# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named :class:`PlanExecuteConfig` presets keyed by shipping bundle.

The shipping default :meth:`PlanExecutePresets.coding_agent_default`
pairs a **strong planner** with a **weak executor** — the documented
project stance since commit ``ca5fcd8a`` (April 2026, then on the
LangGraph orchestrator; the principle migrated forward when the
orchestrator was deprecated in favour of the
:class:`switchyard.lib.processors.plan_execute.PlanningRequestProcessor`).

Rationale (verbatim from ``ca5fcd8a``):

    Planning is the high-leverage cognitive task — a bad plan poisons
    the whole run and no downstream executor can recover from "step 1:
    do the impossible". Decomposition is therefore worth paying a
    strong model for. By contrast, each step in a well-decomposed plan
    is small, self-contained, and mostly mechanical, so a
    weaker/cheaper model can drive it. For plans with more than a
    handful of steps the executor cost dominates, and running that
    dominant term on the cheap model is the whole point.

Example::

    from switchyard import PlanExecutePresets, PlanExecuteProfileConfig, ProfileSwitchyard

    config = PlanExecutePresets.coding_agent_default(
        api_key=nvidia_api_key,
    )
    switchyard = ProfileSwitchyard(PlanExecuteProfileConfig.from_config(config).build())
"""

from __future__ import annotations

from switchyard.lib.backends.llm_target import BackendFormat, LlmTarget
from switchyard.lib.profiles.plan_execute_config import PlanExecuteConfig

# Shipping presets route through OpenRouter's OpenAI-compatible endpoint
# by default; callers override with ``base_url=`` for another gateway.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Planner — Claude Opus 4.6 on OpenRouter.
_MODEL_OPUS_4_6_PLANNER = "anthropic/claude-opus-4.6"

# Executor — Kimi K2.6, an OpenRouter-available weak tier for the
# code-level default. NVIDIA executor presets remain available through
# explicit routing profiles.
_MODEL_KIMI_K2_6_EXECUTOR = "moonshotai/kimi-k2.6"


class PlanExecutePresets:
    """Builder of pre-built :class:`PlanExecuteConfig` bundles."""

    @staticmethod
    def coding_agent_default(
        *,
        api_key: str,
        base_url: str = _OPENROUTER_BASE_URL,
        timeout_secs: float | None = 600.0,
    ) -> PlanExecuteConfig:
        """The coding-agent-launcher default plan-execute pairing.

        Planner: Claude Opus 4.6 (strong reasoning, JSON-strict).
        Executor: Kimi K2.6 (cheap, fast, tool-friendly).
        Cadence: every 2 assistant turns (validated default).

        Use for any coding-agent launcher (Claude Code, Codex, OpenClaw)
        unless you have benchmark numbers showing a different pairing
        wins on your workload.
        """
        return PlanExecuteConfig(
            planner=LlmTarget(
                id="planner",
                model=_MODEL_OPUS_4_6_PLANNER,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
                timeout_secs=timeout_secs,
            ),
            executor=LlmTarget(
                id="executor",
                model=_MODEL_KIMI_K2_6_EXECUTOR,
                format=BackendFormat.OPENAI,
                api_key=api_key,
                base_url=base_url,
                timeout_secs=timeout_secs,
            ),
            cadence_n=2,
            disable_reasoning=False,
            fail_open=True,
            enable_stats=True,
            fallback_target_on_evict="planner",
            preset="coding_agent_default",
        )


__all__ = ["PlanExecutePresets"]
