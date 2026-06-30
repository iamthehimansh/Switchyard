# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`TurnBasedRouterRequestProcessor`.

Pure compute, no LLM, no network — covers:

* The Bresenham ceil formula's long-run distribution and first-turn-strong
  invariant at edges (p=0.0, 1.0) and intermediate values (0.3, 0.5, 0.7).
* OpenAI Chat Completions turn counting (role=='assistant').
* Anthropic Messages turn counting (verifies the role=='user' tool-result
  asymmetry doesn't inflate the counter).
* OpenAI Responses turn counting (string input and list input coarse path).
* Edge cases: empty / missing messages, malformed entries.
* ctx-metadata stamping for both the tier label and the turn number.
* The ``turn_based_decision={...}`` audit line on stderr.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)
from switchyard.lib.processors.turn_based_router_request_processor import (
    CTX_TURN_BASED_TURN,
    TurnBasedRouterRequestProcessor,
    TurnBasedRoutingConfig,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_openai_chat_request(messages: list[dict[str, Any]]) -> ChatRequest:
    """Wrap a messages list in a minimal OpenAI Chat Completions request."""
    return ChatRequest.openai_chat(cast(Any, {
        "model": "placeholder",
        "messages": messages,
    }))


def _build_anthropic_request(messages: list[dict[str, Any]]) -> ChatRequest:
    """Wrap messages in a minimal Anthropic Messages request."""
    return ChatRequest.anthropic(cast(Any, {
        "model": "claude-test",
        "max_tokens": 128,
        "messages": messages,
    }))


def _build_responses_request(input_value: Any) -> ChatRequest:
    """Wrap input in a minimal OpenAI Responses request."""
    return ChatRequest.openai_responses(cast(Any, {
        "model": "placeholder",
        "input": input_value,
    }))


def _conversation_with_n_assistant_turns(n: int) -> list[dict[str, Any]]:
    """Synthesize an OpenAI Chat history with ``n`` prior assistant turns.

    Layout: initial user prompt, then alternating assistant/tool pairs
    matching the terminus-2 pattern.  Sufficient to drive turn counting
    without needing a live agent.
    """
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "initial task"},
    ]
    for i in range(n):
        msgs.append({
            "role": "assistant",
            "content": f"step {i + 1}",
            "tool_calls": [{
                "id": f"c{i}",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": f"output {i + 1}",
        })
    return msgs


# ---------------------------------------------------------------------------
# Cadence formula
# ---------------------------------------------------------------------------


class TestCadenceFormula:
    """Direct tests on _select_tier / the Bresenham ceil schedule."""

    async def test_first_turn_always_strong_when_probability_positive(self) -> None:
        """Turn 1 must always route strong when strong_probability > 0.

        This is the user-facing invariant — "first turn always goes
        to strong".  Verified at multiple probabilities.
        """
        for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            processor = TurnBasedRouterRequestProcessor(
                TurnBasedRoutingConfig(strong_probability=p),
            )
            req = _build_openai_chat_request(
                [{"role": "user", "content": "first turn"}],
            )
            ctx = ProxyContext()
            await processor.process(ctx, req)
            assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong", (
                f"turn 1 must be strong at p={p}; got "
                f"{ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER]!r}"
            )
            assert ctx.metadata[CTX_TURN_BASED_TURN] == 1

    async def test_zero_probability_routes_every_turn_weak(self) -> None:
        """strong_probability=0.0 routes every turn — including turn 1 — to weak.

        This is the documented degenerate behavior: p=0 is equivalent
        to FixedTierRequestProcessor("weak").
        """
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.0),
        )
        for n_prior in range(5):
            ctx = ProxyContext()
            req = _build_openai_chat_request(
                _conversation_with_n_assistant_turns(n_prior),
            )
            await processor.process(ctx, req)
            assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"
            assert ctx.metadata[CTX_TURN_BASED_TURN] == n_prior + 1

    async def test_one_probability_routes_every_turn_strong(self) -> None:
        """strong_probability=1.0 routes every turn to strong."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=1.0),
        )
        for n_prior in range(5):
            ctx = ProxyContext()
            req = _build_openai_chat_request(
                _conversation_with_n_assistant_turns(n_prior),
            )
            await processor.process(ctx, req)
            assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"

    async def test_half_probability_alternates_strong_weak(self) -> None:
        """strong_probability=0.5 should produce S W S W S W... cadence.

        Verifies the ceil(t*0.5) > ceil((t-1)*0.5) formula gives the
        expected alternation: odd turns strong, even turns weak.
        """
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        expected = ["strong", "weak"] * 10  # 20 turns
        actual: list[str] = []
        for n_prior in range(20):
            ctx = ProxyContext()
            req = _build_openai_chat_request(
                _conversation_with_n_assistant_turns(n_prior),
            )
            await processor.process(ctx, req)
            actual.append(ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER])

        assert actual == expected, (
            f"expected strict alternation at p=0.5; got: {actual}"
        )

    async def test_thirty_percent_probability_hits_roughly_thirty_percent_strong(self) -> None:
        """strong_probability=0.3 over 100 turns should produce exactly 30 strong.

        The Bresenham formula gives an *exact* long-run rate (not just
        approximate) at any rational p with denominator dividing the
        turn count.  At p=0.3 over 100 turns, count(strong) == 30.
        """
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.3),
        )
        strong_count = 0
        for n_prior in range(100):
            ctx = ProxyContext()
            req = _build_openai_chat_request(
                _conversation_with_n_assistant_turns(n_prior),
            )
            await processor.process(ctx, req)
            if ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong":
                strong_count += 1
        assert strong_count == 30, (
            f"at p=0.3, expected exactly 30 strong over 100 turns; got {strong_count}"
        )

    async def test_seventy_percent_probability_hits_roughly_seventy_percent_strong(self) -> None:
        """Symmetric: p=0.7 over 100 turns yields exactly 70 strong."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.7),
        )
        strong_count = 0
        for n_prior in range(100):
            ctx = ProxyContext()
            req = _build_openai_chat_request(
                _conversation_with_n_assistant_turns(n_prior),
            )
            await processor.process(ctx, req)
            if ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong":
                strong_count += 1
        assert strong_count == 70, (
            f"at p=0.7, expected exactly 70 strong over 100 turns; got {strong_count}"
        )

    async def test_low_probability_strong_turns_are_spaced(self) -> None:
        """At p=0.2, strong turns should appear at indices 1, 6, 11, 16, ...

        Bresenham formula at p=0.2 yields ``ceil(t*0.2) > ceil((t-1)*0.2)``
        when ``t * 0.2`` crosses an integer — that's t=1, 5, 10, 15...
        but offset by 1 because turn 1 is strong (ceil(0.2)=1 > ceil(0)=0).

        Let me compute directly to be sure:
        - t=1: ceil(0.2)=1, ceil(0)=0, 1>0 → strong
        - t=2: ceil(0.4)=1, ceil(0.2)=1, NO → weak
        - t=3: ceil(0.6)=1, ceil(0.4)=1, NO → weak
        - t=4: ceil(0.8)=1, ceil(0.6)=1, NO → weak
        - t=5: ceil(1.0)=1, ceil(0.8)=1, NO → weak
        - t=6: ceil(1.2)=2, ceil(1.0)=1, YES → strong
        - t=7-10: weak
        - t=11: strong
        """
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.2),
        )
        strong_indices: list[int] = []
        for n_prior in range(20):
            turn = n_prior + 1
            ctx = ProxyContext()
            req = _build_openai_chat_request(
                _conversation_with_n_assistant_turns(n_prior),
            )
            await processor.process(ctx, req)
            if ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong":
                strong_indices.append(turn)

        assert strong_indices == [1, 6, 11, 16], (
            f"expected strong turns at [1, 6, 11, 16] at p=0.2 over 20 turns; "
            f"got: {strong_indices}"
        )


# ---------------------------------------------------------------------------
# Per-format turn counting
# ---------------------------------------------------------------------------


class TestOpenAIChatTurnCounting:
    """OpenAI Chat Completions: count role=='assistant', add 1."""

    async def test_no_messages_treated_as_turn_one(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_openai_chat_request([])
        ctx = ProxyContext()
        await processor.process(ctx, req)
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 1

    async def test_user_only_history_is_turn_one(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_openai_chat_request([
            {"role": "system", "content": "..."},
            {"role": "user", "content": "first turn"},
        ])
        ctx = ProxyContext()
        await processor.process(ctx, req)
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 1

    async def test_three_prior_assistants_is_turn_four(self) -> None:
        """3 prior assistant turns → this is the 4th LLM invocation."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_openai_chat_request(
            _conversation_with_n_assistant_turns(3),
        )
        ctx = ProxyContext()
        await processor.process(ctx, req)
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 4

    async def test_tool_messages_do_not_inflate_count(self) -> None:
        """role=='tool' messages must not be counted as turns."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "step 1"},
            # 5 tool messages — pure agent-side, shouldn't count.
            *[{"role": "tool", "tool_call_id": f"c{i}", "content": "..."} for i in range(5)],
        ]
        req = _build_openai_chat_request(messages)
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # One assistant → turn 2.
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 2

    async def test_system_messages_do_not_inflate_count(self) -> None:
        """role=='system' messages must not be counted as turns."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        messages = [
            {"role": "system", "content": "agent system prompt"},
            {"role": "system", "content": "another system note"},
            {"role": "user", "content": "task"},
        ]
        req = _build_openai_chat_request(messages)
        ctx = ProxyContext()
        await processor.process(ctx, req)
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 1


class TestAnthropicTurnCounting:
    """Anthropic Messages: count role=='assistant' — verify role=='user'
    tool_result messages don't accidentally inflate the count."""

    async def test_initial_anthropic_request_is_turn_one(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_anthropic_request([
            {"role": "user", "content": "initial task"},
        ])
        ctx = ProxyContext()
        await processor.process(ctx, req)
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 1

    async def test_anthropic_user_tool_result_messages_not_counted(self) -> None:
        """In Anthropic Messages, tool_result is inside role=='user' messages.

        The counter is on role=='assistant', so these don't inflate.
        Verifies the format-asymmetry trap doesn't bite — we picked
        ``role == "assistant"`` as the discriminator precisely because
        it's invariant across the two message formats.
        """
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "looking..."},
                    {"type": "tool_use", "id": "tu1", "name": "bash", "input": {}},
                ],
            },
            # The Anthropic-shape tool_result: comes back as role=='user'.
            # Counting role=='user' would now wrongly say 2 turns.
            # We count role=='assistant', so this should still be turn 2.
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "stdout"},
                ],
            },
        ]
        req = _build_anthropic_request(messages)
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # One assistant → turn 2.
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 2

    async def test_multi_round_anthropic_conversation_counts_correctly(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [{"type": "text", "text": "r1"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "out1"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "r2"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "out2"}]},
        ]
        req = _build_anthropic_request(messages)
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # Two assistants → turn 3.
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 3


class TestOpenAIResponsesTurnCounting:
    """OpenAI Responses: string input → turn 1; list input → coarse count."""

    async def test_string_input_is_turn_one(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_responses_request("initial task")
        ctx = ProxyContext()
        await processor.process(ctx, req)
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 1

    async def test_list_input_with_only_user_message_is_turn_one(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_responses_request([
            {"role": "user", "content": "initial task"},
        ])
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # One agent-side ack (the user prompt) → coarse turn 1.
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 1

    async def test_list_input_with_function_call_outputs_advances_turn(self) -> None:
        """function_call_output items mark prior LLM-round acks."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_responses_request([
            {"role": "user", "content": "task"},
            {"type": "function_call", "id": "fc1", "name": "bash", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc1", "output": "stdout"},
            {"type": "function_call", "id": "fc2", "name": "bash", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc2", "output": "stdout"},
        ])
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # 1 user ack + 2 function_call_output acks = 3.
        # Coarse approximation: turn 3 (we're 2 rounds in).
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 3


# ---------------------------------------------------------------------------
# Stamping + audit line
# ---------------------------------------------------------------------------


class TestStampingAndAudit:
    """ctx-metadata stamping + stderr audit-line shape."""

    async def test_stamps_both_tier_and_turn_metadata_keys(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_openai_chat_request(
            _conversation_with_n_assistant_turns(2),
        )
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # Two prior assistants → turn 3 → odd → strong at p=0.5.
        assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
        assert ctx.metadata[CTX_TURN_BASED_TURN] == 3

    async def test_custom_tier_labels_are_stamped(self) -> None:
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(
                strong_tier="premium",
                weak_tier="economy",
                strong_probability=0.5,
            ),
        )
        req = _build_openai_chat_request(
            [{"role": "user", "content": "task"}],
        )
        ctx = ProxyContext()
        await processor.process(ctx, req)
        # Turn 1 always strong.
        assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "premium"

    async def test_audit_line_lands_on_stderr_with_expected_fields(self, capsys: Any) -> None:
        """One ``turn_based_decision=...`` line per invocation on stderr."""
        processor = TurnBasedRouterRequestProcessor(
            TurnBasedRoutingConfig(strong_probability=0.5),
        )
        req = _build_openai_chat_request(
            [{"role": "user", "content": "task"}],
        )
        ctx = ProxyContext()
        await processor.process(ctx, req)

        captured = capsys.readouterr()
        # The audit line is on stderr (mirrors classifier + planner).
        lines = [
            ln for ln in captured.err.splitlines()
            if ln.startswith("turn_based_decision=")
        ]
        assert len(lines) == 1, (
            f"expected exactly 1 turn_based_decision line on stderr; "
            f"got {len(lines)}.  Stderr was: {captured.err!r}"
        )
        payload = json.loads(lines[0][len("turn_based_decision="):])
        assert payload["turn"] == 1
        assert payload["tier"] == "strong"
        assert math.isclose(payload["strong_probability"], 0.5)
        assert payload["request_type"] == "openai_chat"
