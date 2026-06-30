# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-backed planner / executor primitives.

A request-augmentation pattern that calls a planner LLM, which decides
per turn whether the executor needs a (fresh) plan and, if so, emits
``plan_text`` — first-person prefill content that the processor appends
as an ``assistant`` turn at the end of the outbound conversation so
the executor continues from there.  When the planner declines
(``plan_needed=False``), the request body is left untouched and the
:class:`PlannerDecision` is stamped on ctx for observability.

The plan is *only* prefilled into the executor's request — it is never
echoed back to the client.  Plans guide one turn of executor
generation and do not persist into the agent harness's conversation
history.  See the benchmark results in
the project history for the rationale: echoing plans into the response
broke strict-JSON parsers downstream (terminus-2 reported ``Extra text
detected before JSON object``) and provided no measurable lift over a
no-planner baseline.

Independent of :mod:`switchyard.lib.processors.llm_classifier`: the planner
does not read :class:`RouteSignals` or the tier decision, and the
classifier does not read the plan.  The two are composable —
typically classifier first to pick a tier, planner after to enrich
the prompt — but should be imported, configured, and reasoned about
separately.

Import explicitly from this subpackage::

    from switchyard.lib.processors.plan_execute import (
        PlanningRequestProcessor,
        PlanningConfig,
        PlannerDecision,
        CTX_PLANNER_DECISION,
    )
"""

from switchyard.lib.processors.plan_execute.plan import (
    CTX_PLANNER_DECISION,
    PlannerDecision,
)
from switchyard.lib.processors.plan_execute.request_processor import (
    DEFAULT_INITIAL_PLANNER_SYSTEM_PROMPT,
    DEFAULT_REVISION_PLANNER_SYSTEM_PROMPT,
    OpenAIChatPlannerClient,
    PlannerClient,
    PlannerCompletion,
    PlanningConfig,
    PlanningError,
    PlanningRequestProcessor,
    PlanningTriggerMode,
    is_anthropic_model,
    parse_planner_decision,
)

__all__ = [
    "CTX_PLANNER_DECISION",
    "DEFAULT_INITIAL_PLANNER_SYSTEM_PROMPT",
    "DEFAULT_REVISION_PLANNER_SYSTEM_PROMPT",
    "OpenAIChatPlannerClient",
    "PlannerClient",
    "PlannerCompletion",
    "PlannerDecision",
    "PlanningConfig",
    "PlanningError",
    "PlanningRequestProcessor",
    "PlanningTriggerMode",
    "is_anthropic_model",
    "parse_planner_decision",
]
