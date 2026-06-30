# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic tests for Codex multi-turn tool call history merging.

Uses actual trace data from tmp/output/logs to verify whether the observed
"missing intermediate assistant-tool turns" is:
  (a) Codex CLI behavior (client sends merged history)
  (b) A bug in our Responses API → Chat Completions translation layer
  (c) User misinterpretation of the logs

CONCLUSION: It's (b) — a bug in Responses input-item translation. The
translator didn't flush the pending tool block at turn boundaries
(function_call_output → function_call transition), so tool calls across turns
were merged into one assistant message.
"""

import json
from pathlib import Path

import pytest

from switchyard_rust.translation import TranslationEngine

TRACE_DIR = Path(__file__).parent.parent / "tmp" / "output" / "logs" / "openai"
ENGINE = TranslationEngine()


def _responses_items_to_messages(items: list) -> list:
    body = ENGINE.translate_request("openai_responses", "openai_chat", {"input": items})
    return list(body.get("messages", []))


# ---------------------------------------------------------------------------
# Helper: load trace files sorted by timestamp
# ---------------------------------------------------------------------------

def _load_traces():
    """Load all trace JSON files from the output directory, sorted by timestamp."""
    if not TRACE_DIR.exists():
        pytest.skip(f"Trace directory not found: {TRACE_DIR}")
    files = sorted(TRACE_DIR.glob("*.json"))
    if not files:
        pytest.skip("No trace files found")
    traces = []
    for f in files:
        with open(f) as fh:
            traces.append(json.load(fh))
    return traces


# ---------------------------------------------------------------------------
# Test 1: Verify what the traces actually show (structural analysis)
# ---------------------------------------------------------------------------

class TestTraceStructuralAnalysis:
    """Analyze the actual trace files to understand the conversation structure."""

    def test_trace_count(self):
        """There should be 4 traces for this Codex session."""
        traces = _load_traces()
        assert len(traces) == 4, f"Expected 4 traces, got {len(traces)}"

    def test_trace1_is_initial_request(self):
        """Trace 1: Initial request with only user messages, no history."""
        traces = _load_traces()
        t1 = traces[0]
        msgs = t1["request"]["messages"]

        # Should have: system, developer, user(AGENTS.md), user(env), user(task)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "developer", "user", "user", "user"]

        # Response should have 2 tool_calls (update_plan + apply_patch)
        response_tool_calls = t1["response"]["choices"][0]["message"]["tool_calls"]
        assert len(response_tool_calls) == 2
        names = [tc["function"]["name"] for tc in response_tool_calls]
        assert names == ["update_plan", "apply_patch"]

    def test_trace2_has_turn1_history(self):
        """Trace 2: Should include turn 1's assistant + tool results as history."""
        traces = _load_traces()
        t2 = traces[1]
        msgs = t2["request"]["messages"]

        roles = [m["role"] for m in msgs]
        # Expected: system, developer, user, user, user, assistant, tool, tool
        assert roles == ["system", "developer", "user", "user", "user",
                         "assistant", "tool", "tool"]

        # The assistant message should have exactly 2 tool_calls (from turn 1)
        assistant_msg = msgs[5]
        assert len(assistant_msg["tool_calls"]) == 2
        names = [tc["function"]["name"] for tc in assistant_msg["tool_calls"]]
        assert names == ["update_plan", "apply_patch"]

    def test_trace3_shows_merged_turns_pre_fix(self):
        """Trace 3: Captured before fix — turns 1+2 merged into single assistant msg.

        These traces were recorded with the old buggy translation layer.
        After the fix in _responses_items_to_messages(), new traces
        would show separate assistant messages per turn.
        """
        traces = _load_traces()
        t3 = traces[2]
        msgs = t3["request"]["messages"]

        [m["role"] for m in msgs]

        # Count assistant messages in the history
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]

        # BUG: There's only 1 assistant message when there should be 2
        assert len(assistant_msgs) == 1, (
            f"BUG CONFIRMED: Only {len(assistant_msgs)} assistant message(s) — "
            f"turns 1+2 are merged into a single assistant message"
        )

        # The single assistant message has 9 tool_calls (2 from turn 1 + 7 from turn 2)
        merged_tool_calls = assistant_msgs[0]["tool_calls"]
        assert len(merged_tool_calls) == 9, (
            f"Expected 9 merged tool_calls (2+7), got {len(merged_tool_calls)}"
        )

        # Verify the first 2 are from turn 1
        assert merged_tool_calls[0]["function"]["name"] == "update_plan"
        assert merged_tool_calls[1]["function"]["name"] == "apply_patch"
        # The remaining 7 are from turn 2
        turn2_names = [tc["function"]["name"] for tc in merged_tool_calls[2:]]
        assert all(n == "exec_command" for n in turn2_names)

        # There should be 9 tool messages too
        assert len(tool_msgs) == 9

    def test_trace4_shows_all_turns_merged_pre_fix(self):
        """Trace 4: Captured before fix — all turns merged into single assistant msg."""
        traces = _load_traces()
        t4 = traces[3]
        msgs = t4["request"]["messages"]

        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        [m for m in msgs if m["role"] == "tool"]

        # BUG: Still only 1 assistant message
        assert len(assistant_msgs) == 1, (
            f"BUG CONFIRMED: Only {len(assistant_msgs)} assistant message(s) — "
            f"all turns merged"
        )

        # Should have even more tool_calls (2 + 7 + 3 = 12 from turn 3 additions)
        total_tool_calls = len(assistant_msgs[0]["tool_calls"])
        assert total_tool_calls > 9, (
            f"Expected >9 tool_calls in merged message, got {total_tool_calls}"
        )


# ---------------------------------------------------------------------------
# Test 2: Reproduce the bug with reconstructed Responses API input
# ---------------------------------------------------------------------------

class TestTranslationLayerMultiTurnBug:
    """Reproduce the turn-merging bug using _responses_items_to_messages()
    with input items reconstructed from the trace data."""

    @staticmethod
    def _make_two_turn_input():
        """Reconstruct the Responses API input Codex would send for trace 3.

        Turn 1: 2 parallel tool calls (update_plan + apply_patch) + results
        Turn 2: 7 parallel tool calls (5× ls + 2× echo) + results

        This is the standard Responses API format: all calls from a turn
        grouped together, then all outputs, then next turn's calls, etc.
        """
        return [
            # --- User message ---
            {"type": "message", "role": "user", "content": "Create a Python script"},

            # --- Turn 1: 2 tool calls ---
            {
                "type": "function_call",
                "name": "update_plan",
                "call_id": "call_turn1_a",
                "arguments": '{"plan": [{"step": "Create script", "status": "completed"}]}',
            },
            {
                "type": "function_call",
                "name": "apply_patch",
                "call_id": "call_turn1_b",
                "arguments": '{"command": "*** Begin Patch..."}',
            },
            # --- Turn 1: results ---
            {
                "type": "function_call_output",
                "call_id": "call_turn1_a",
                "output": "Plan updated",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn1_b",
                "output": "unsupported call: apply_patch",
            },

            # --- Turn 2: 7 tool calls ---
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_a",
                "arguments": '{"cmd": "ls -R ."}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_b",
                "arguments": '{"cmd": "ls -R ."}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_c",
                "arguments": '{"cmd": "ls -R ."}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_d",
                "arguments": '{"cmd": "ls -R ."}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_e",
                "arguments": '{"cmd": "ls -R .", "max_output_tokens": 2000}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_f",
                "arguments": '{"cmd": "echo hello", "max_output_tokens": 20}',
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_turn2_g",
                "arguments": '{"cmd": "echo hello", "shell": "/bin/bash"}',
            },
            # --- Turn 2: results ---
            {
                "type": "function_call_output",
                "call_id": "call_turn2_a",
                "output": ".:\ninput.json\nlogs\n",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn2_b",
                "output": ".:\ninput.json\nlogs\n",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn2_c",
                "output": ".:\ninput.json\nlogs\n",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn2_d",
                "output": ".:\ninput.json\nlogs\n",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn2_e",
                "output": ".:\ninput.json\nlogs\n",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn2_f",
                "output": "hello\n",
            },
            {
                "type": "function_call_output",
                "call_id": "call_turn2_g",
                "output": "hello\n",
            },
        ]

    def test_two_turns_produce_separate_assistant_messages(self):
        """Two turns of tool calls produce separate assistant messages.

        The proxy detects the function_call_output → function_call turn
        boundary and flushes, producing:
          user → assistant(2 calls) → 2 tools → assistant(7 calls) → 7 tools
        """
        items = self._make_two_turn_input()
        messages = _responses_items_to_messages(items)

        user_msgs = [m for m in messages if m["role"] == "user"]
        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]

        assert len(user_msgs) == 1
        assert len(asst_msgs) == 2, "Should have 2 assistant messages (one per turn)"
        assert len(asst_msgs[0]["tool_calls"]) == 2, "Turn 1: 2 tool calls"
        assert len(asst_msgs[1]["tool_calls"]) == 7, "Turn 2: 7 tool calls"
        assert len(tool_msgs) == 9, "Total 9 tool results"

        # Verify message ordering
        expected_roles = [
            "user",       # original request
            "assistant",  # turn 1: 2 tool calls
            "tool",       # turn 1 result 1
            "tool",       # turn 1 result 2
            "assistant",  # turn 2: 7 tool calls
            "tool",       # turn 2 results...
            "tool",
            "tool",
            "tool",
            "tool",
            "tool",
            "tool",
        ]
        actual_roles = [m["role"] for m in messages]
        assert actual_roles == expected_roles

    def test_single_turn_still_works(self):
        """Single-turn tool calls should still be merged into one assistant message."""
        items = [
            {"type": "message", "role": "user", "content": "Do two things"},
            {
                "type": "function_call",
                "name": "tool_a",
                "call_id": "call_a",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "name": "tool_b",
                "call_id": "call_b",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_a",
                "output": "result_a",
            },
            {
                "type": "function_call_output",
                "call_id": "call_b",
                "output": "result_b",
            },
        ]
        messages = _responses_items_to_messages(items)

        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(asst_msgs) == 1
        assert len(asst_msgs[0]["tool_calls"]) == 2

    def test_three_turns_produce_three_assistant_messages(self):
        """Three turns of tool calls produce 3 separate assistant messages.

        Reconstructs what Codex sends for trace 4 (turn 1 + turn 2 + turn 3).
        """
        items = [
            {"type": "message", "role": "user", "content": "Create a script"},

            # Turn 1: 2 calls
            {"type": "function_call", "name": "update_plan", "call_id": "t1_a", "arguments": "{}"},
            {"type": "function_call", "name": "apply_patch", "call_id": "t1_b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "t1_a", "output": "Plan updated"},
            {"type": "function_call_output", "call_id": "t1_b", "output": "unsupported"},

            # Turn 2: 3 calls
            {"type": "function_call", "name": "exec_command", "call_id": "t2_a", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call", "name": "exec_command", "call_id": "t2_b", "arguments": '{"cmd":"echo hi"}'},
            {"type": "function_call", "name": "exec_command", "call_id": "t2_c", "arguments": '{"cmd":"pwd"}'},
            {"type": "function_call_output", "call_id": "t2_a", "output": "file1 file2"},
            {"type": "function_call_output", "call_id": "t2_b", "output": "hi"},
            {"type": "function_call_output", "call_id": "t2_c", "output": "/workspace"},

            # Turn 3: 2 calls
            {"type": "function_call", "name": "exec_command", "call_id": "t3_a", "arguments": '{"cmd":"cat file1"}'},
            {"type": "function_call", "name": "exec_command", "call_id": "t3_b", "arguments": '{"cmd":"cat file2"}'},
            {"type": "function_call_output", "call_id": "t3_a", "output": "content1"},
            {"type": "function_call_output", "call_id": "t3_b", "output": "content2"},
        ]
        messages = _responses_items_to_messages(items)

        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]

        assert len(asst_msgs) == 3, "Should have 3 assistant messages"
        assert len(asst_msgs[0]["tool_calls"]) == 2, "Turn 1: 2 calls"
        assert len(asst_msgs[1]["tool_calls"]) == 3, "Turn 2: 3 calls"
        assert len(asst_msgs[2]["tool_calls"]) == 2, "Turn 3: 2 calls"
        assert len(tool_msgs) == 7, "Total 7 tool results"


# ---------------------------------------------------------------------------
# Test 3: Verify the specific bug location
# ---------------------------------------------------------------------------

class TestBugLocation:
    """Pinpoint the exact code path causing the merge."""

    def test_turn_boundary_detected_at_output_to_call_transition(self):
        """The converter flushes at function_call_output → function_call transitions.

        This transition marks a turn boundary: the previous turn's outputs are
        done and a new LLM response is starting.
        """
        # Minimal case: two turns with no message item between them
        items = [
            # Turn 1
            {"type": "function_call", "name": "tool_a", "call_id": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "done_a"},
            # Turn 2 (no message item separating from turn 1)
            {"type": "function_call", "name": "tool_b", "call_id": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "b", "output": "done_b"},
        ]
        messages = _responses_items_to_messages(items)

        asst_msgs = [m for m in messages if m["role"] == "assistant"]

        # Correctly produces 2 separate assistant messages
        assert len(asst_msgs) == 2
        assert len(asst_msgs[0]["tool_calls"]) == 1
        assert asst_msgs[0]["tool_calls"][0]["id"] == "a"
        assert len(asst_msgs[1]["tool_calls"]) == 1
        assert asst_msgs[1]["tool_calls"][0]["id"] == "b"

    def test_message_item_correctly_triggers_flush(self):
        """When a message item separates tool blocks, flush works correctly."""
        items = [
            # Turn 1
            {"type": "function_call", "name": "tool_a", "call_id": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "done_a"},
            # Explicit message item triggers flush
            {"type": "message", "role": "assistant", "content": "Intermediate text"},
            # Turn 2
            {"type": "function_call", "name": "tool_b", "call_id": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "b", "output": "done_b"},
        ]
        messages = _responses_items_to_messages(items)

        asst_msgs = [m for m in messages if m["role"] == "assistant"]
        # This works correctly: 3 assistant messages (tool_a, text, tool_b)
        assert len(asst_msgs) == 3
        assert len(asst_msgs[0]["tool_calls"]) == 1  # tool_a
        assert asst_msgs[1]["content"] == "Intermediate text"
        assert len(asst_msgs[2]["tool_calls"]) == 1  # tool_b


# ---------------------------------------------------------------------------
# Test 4: Verify this matches the trace data from tmp/output
# ---------------------------------------------------------------------------

class TestTraceDataConsistency:
    """Cross-reference trace data with the translation function output."""

    def test_trace2_tool_call_ids_match(self):
        """The tool_call IDs in trace 2's messages should match trace 1's response."""
        traces = _load_traces()
        t1_response_calls = traces[0]["response"]["choices"][0]["message"]["tool_calls"]
        t2_history_assistant = next(
            m for m in traces[1]["request"]["messages"] if m["role"] == "assistant"
        )

        t1_ids = [tc["id"] for tc in t1_response_calls]
        t2_ids = [tc["id"] for tc in t2_history_assistant["tool_calls"]]

        assert t1_ids == t2_ids, (
            "Turn 1 response tool_call IDs should appear in trace 2's history"
        )

    def test_trace3_contains_turn1_and_turn2_calls_merged(self):
        """Trace 3's single assistant message has turn 1 + turn 2 tool_calls merged."""
        traces = _load_traces()
        t3_assistant = next(
            m for m in traces[2]["request"]["messages"] if m["role"] == "assistant"
        )

        t1_ids = [tc["id"] for tc in traces[0]["response"]["choices"][0]["message"]["tool_calls"]]
        t2_ids = [tc["id"] for tc in traces[1]["response"]["choices"][0]["message"]["tool_calls"]]

        merged_ids = [tc["id"] for tc in t3_assistant["tool_calls"]]

        # The merged assistant message contains ALL IDs from both turns
        assert merged_ids[:len(t1_ids)] == t1_ids, "First tool_calls should be from turn 1"
        assert merged_ids[len(t1_ids):] == t2_ids, "Remaining tool_calls should be from turn 2"

    def test_all_traces_same_session(self):
        """All traces belong to the same Codex session."""
        traces = _load_traces()
        session_ids = {t["sessionID"] for t in traces}
        assert len(session_ids) == 1, f"Expected 1 session, got {session_ids}"

    def test_tool_results_in_trace2_match_tool_call_ids(self):
        """The tool_results attached to trace 1 should be consistent with trace 2's history."""
        traces = _load_traces()
        t1 = traces[0]

        # tool_results from the deferred logging
        if "tool_results" not in t1:
            pytest.skip("Trace 1 doesn't have tool_results (may be first-turn pattern)")

        t1_result_ids = {tr["tool_call_id"] for tr in t1["tool_results"]}
        t1_call_ids = {tc["id"] for tc in t1["response"]["choices"][0]["message"]["tool_calls"]}

        assert t1_result_ids == t1_call_ids, (
            "tool_results should correspond to the response's tool_calls"
        )
