# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slim planner-decision contract for the planner / executor pattern.

The planner LLM emits a two-field JSON object:

* ``plan_needed`` — boolean gate.  ``False`` means the planner declined
  to plan (fast path; ``plan_text`` is empty).  ``True`` means the
  planner is contributing a plan to inject.
* ``plan_text`` — the literal content the executor will see as an
  assistant prefill.  When ``plan_needed=True`` this is the agent's
  drafted thinking written in first-person ("I'll approach this as
  follows: ..."); when ``plan_needed=False`` this is empty.

Why so slim:

* Earlier versions emitted a structured :class:`ExecutionPlan` (steps,
  risks, test strategy, confidence) and ran the output through a
  ``render_execution_plan`` templater.  Strict JSON-schema decoder
  enforcement on a nested-object shape was fragile (the Pydantic
  ``title`` field on the old ``PlanStep`` class collided with
  ``_STRICT_DROP_KEYS`` and silently stripped the property — see the
  929/929 silent fail-open on the 2026-05-14 cheap-planner-control
  sweep).
* The injection mechanism is now assistant-prefill: we paste the
  planner's text directly into the conversation as the executor's
  opening response.  No template, no field extraction — the planner's
  output literally is the prefill content.  A flat two-field schema
  has no nested objects to collide with strict-mode drop rules, and
  the planner can format ``plan_text`` however fits the task
  (numbered steps, paragraph, bullets — the executor sees prose, not
  structured data).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

#: ``ProxyContext.metadata`` key for the full :class:`PlannerDecision`.
#:
#: Always stamped when the planner runs successfully (both the
#: ``plan_needed=True`` and ``plan_needed=False`` branches).
CTX_PLANNER_DECISION = "_planner_decision"


class PlannerDecision(BaseModel):
    """Slim two-field wrapper for the planner LLM's output.

    Shape:

    * ``plan_needed=False``: ``plan_text`` must be empty.  The fast
      path — the planner is saying "no new plan is warranted on this
      turn; the existing conversation anchor is sufficient."  The
      processor leaves the request body untouched.
    * ``plan_needed=True``: ``plan_text`` is non-empty and contains
      the prefill content for the executor.  The processor appends an
      ``{"role": "assistant", "content": plan_text}`` entry to the
      outbound conversation so the executor continues from there.

    No nested objects, no rendered-template step types — see the
    module docstring for why.
    """

    model_config = ConfigDict(frozen=True)

    plan_needed: bool
    plan_text: str = ""

    @model_validator(mode="after")
    def _validate_consistency(self) -> PlannerDecision:
        if not self.plan_needed and self.plan_text:
            # Be forgiving but normalize: trust the gate, drop the
            # spurious text.  The planner system prompt explicitly
            # contracts ``plan_needed=false -> plan_text=""``; if the
            # model emitted both, the gate wins.
            object.__setattr__(self, "plan_text", "")
        if self.plan_needed and not self.plan_text.strip():
            raise ValueError("plan_needed=True requires non-empty plan_text")
        return self


__all__ = [
    "CTX_PLANNER_DECISION",
    "PlannerDecision",
]
