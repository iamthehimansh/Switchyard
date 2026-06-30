#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test: multi-turn Responses API with GPT-OSS-120B via NVIDIA API.

Simulates Codex-like multi-turn tool-call conversations through the proxy's
/v1/responses endpoint.  After all turns complete, loads the saved traces
from --rl-log-dir and verifies that each turn produced a separate assistant
message (i.e. the turn-merge bug is fixed).

Usage:
  1. Start the proxy in a separate terminal:
       source .venv/bin/activate
       switchyard passthrough \
         --port 4000 \
         --api-key "$OPENAI_API_KEY" \
         --base-url https://inference-api.nvidia.com/v1 \
         --enable-rl-logging --rl-log-dir ./e2e_traces

  2. Run this script:
       python tests/e2e_multiturn_responses.py --proxy-url http://localhost:4000

  The script exits 0 on success, 1 on failure.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PROXY_URL = "http://localhost:4000"
DEFAULT_MODEL = "nvidia/openai/gpt-oss-120b"
DEFAULT_TRACE_DIR = Path(__file__).parent.parent / "e2e_traces"
SESSION_ID = str(uuid.uuid4())

TOOLS = [
    {
        "type": "function",
        "name": "exec_command",
        "description": "Runs a shell command and returns stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def send_responses_request(
    proxy_url: str,
    model: str,
    input_items: list,
    session_id: str,
    stream: bool = True,
    timeout: float = 120.0,
) -> dict:
    """Send a Responses API request and return the parsed response.

    For streaming, collects SSE events and returns the response.completed payload.
    For non-streaming, returns the JSON response directly.
    """
    body = {
        "model": model,
        "input": input_items,
        "tools": TOOLS,
        "tool_choice": "auto",
        "stream": stream,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
        "proxy_x_session_id": session_id,
    }

    url = f"{proxy_url}/v1/responses"

    if not stream:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # Streaming: collect SSE events
    completed_response = None
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)
                    lines = event_str.strip().split("\n")
                    event_type = None
                    event_data = None
                    for line in lines:
                        if line.startswith("event: "):
                            event_type = line[7:]
                        elif line.startswith("data: "):
                            event_data = line[6:]

                    if event_type == "response.completed" and event_data:
                        completed_response = json.loads(event_data)
                    elif event_type == "error" and event_data:
                        err = json.loads(event_data)
                        print(f"  SSE error: {err}", file=sys.stderr)

    if completed_response is None:
        raise RuntimeError("Never received response.completed event")

    return completed_response


def extract_tool_calls(response: dict) -> list[dict]:
    """Extract function_call items from a Responses API response."""
    output = response.get("response", response).get("output", [])
    return [item for item in output if item.get("type") == "function_call"]


def extract_text(response: dict) -> str:
    """Extract text content from a Responses API response."""
    output = response.get("response", response).get("output", [])
    parts = []
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
    return "\n".join(parts)


def simulate_tool_results(tool_calls: list[dict]) -> list[dict]:
    """Generate fake tool results for the given tool calls."""
    results = []
    for tc in tool_calls:
        args = json.loads(tc.get("arguments", "{}"))
        cmd = args.get("cmd", "unknown")
        # Fake output based on command
        if "ls" in cmd:
            output = "file1.py\nfile2.py\nREADME.md\n"
        elif "echo" in cmd:
            output = "hello\n"
        elif "cat" in cmd:
            output = "# File content\nprint('hello world')\n"
        elif "pwd" in cmd:
            output = "/workspace\n"
        else:
            output = f"Executed: {cmd}\n"

        results.append({
            "type": "function_call_output",
            "call_id": tc.get("call_id", ""),
            "output": output,
        })
    return results


def build_history(turns: list[tuple[list[dict], list[dict]]]) -> list[dict]:
    """Build Responses API input items from a list of (tool_calls, tool_results) turns.

    Each turn's function_calls come first (grouped), then function_call_outputs.
    This matches how Codex sends multi-turn history.
    """
    items = []
    for tool_calls, tool_results in turns:
        for tc in tool_calls:
            items.append({
                "type": "function_call",
                "name": tc.get("name", ""),
                "call_id": tc.get("call_id", ""),
                "arguments": tc.get("arguments", "{}"),
            })
        for tr in tool_results:
            items.append(tr)
    return items


# ---------------------------------------------------------------------------
# Load and verify traces
# ---------------------------------------------------------------------------

def load_session_traces(trace_dir: Path, session_id: str) -> list[dict]:
    """Load trace files for a specific session, sorted by timestamp."""
    if not trace_dir.exists():
        return []
    traces = []
    for f in sorted(trace_dir.glob("*.json")):
        with open(f) as fh:
            t = json.load(fh)
        if t.get("sessionID") == session_id:
            traces.append(t)
    # Also check subdirectories (e.g. openai/)
    for f in sorted(trace_dir.rglob("*.json")):
        with open(f) as fh:
            t = json.load(fh)
        if t.get("sessionID") == session_id and t not in traces:
            traces.append(t)
    # Sort by timestamp
    traces.sort(key=lambda t: t.get("timestamp", ""))
    return traces


def verify_traces(traces: list[dict], expected_turns: int) -> bool:
    """Verify that traces have correct multi-turn structure.

    Returns True if all traces show separate assistant messages per turn.
    """
    ok = True

    for i, trace in enumerate(traces):
        msgs = trace["request"]["messages"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]

        # Trace i should have exactly i assistant messages in its history
        # (one per previous turn)
        expected_asst = i  # trace 0 has 0, trace 1 has 1, etc.

        if len(asst_msgs) != expected_asst:
            print(
                f"  FAIL trace {i+1}: expected {expected_asst} assistant msg(s) "
                f"in history, got {len(asst_msgs)}"
            )
            if len(asst_msgs) > 0:
                for j, am in enumerate(asst_msgs):
                    n_calls = len(am.get("tool_calls", []))
                    print(f"    assistant msg {j}: {n_calls} tool_calls")
            ok = False
        else:
            # Verify each assistant message has the right tool_calls count
            total_tool_calls = sum(
                len(am.get("tool_calls", [])) for am in asst_msgs
            )
            print(
                f"  OK   trace {i+1}: {len(asst_msgs)} assistant msg(s), "
                f"{total_tool_calls} total tool_calls, {len(tool_msgs)} tool results"
            )

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E2E multi-turn Responses API test")
    parser.add_argument("--proxy-url", default=DEFAULT_PROXY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-stream", action="store_true", help="Use non-streaming mode")
    parser.add_argument("--max-turns", type=int, default=3, help="Max conversation turns")
    args = parser.parse_args()

    stream = not args.no_stream
    print(f"Session ID: {SESSION_ID}")
    print(f"Proxy: {args.proxy_url}")
    print(f"Model: {args.model}")
    print(f"Stream: {stream}")
    print(f"Trace dir: {args.trace_dir}")
    print()

    # Initial user message
    user_prompt = (
        "You are a helpful coding assistant with access to exec_command tool. "
        "Please: 1) List files in the current directory, 2) Show the current "
        "working directory, 3) Echo 'hello world'. Do all three using exec_command."
    )

    # Build the conversation input, starting with just the user message
    input_items: list[dict] = [
        {"type": "message", "role": "user", "content": user_prompt},
    ]

    turns: list[tuple[list[dict], list[dict]]] = []
    turn = 0

    while turn < args.max_turns:
        turn += 1
        print(f"--- Turn {turn} ---")
        print(f"  Sending {len(input_items)} input items...")

        try:
            response = send_responses_request(
                args.proxy_url, args.model, input_items, SESSION_ID,
                stream=stream, timeout=120.0,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        tool_calls = extract_tool_calls(response)
        text = extract_text(response)

        if text:
            print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        if not tool_calls:
            print(f"  No tool calls — conversation complete after {turn} turn(s)")
            break

        print(f"  Got {len(tool_calls)} tool call(s):")
        for tc in tool_calls:
            args_str = tc.get("arguments", "{}")
            cmd = json.loads(args_str).get("cmd", "?")
            print(f"    - {tc['name']}({cmd})")

        # Simulate tool execution
        tool_results = simulate_tool_results(tool_calls)
        turns.append((tool_calls, tool_results))

        # Rebuild input with full history for next turn
        history_items = build_history(turns)
        input_items = [
            {"type": "message", "role": "user", "content": user_prompt},
            *history_items,
        ]

        # Small delay to let trace writes flush
        time.sleep(1)

    print()

    # Wait for background trace writer to flush
    print("Waiting 3s for trace writer to flush...")
    time.sleep(3)

    # Load and verify traces
    print(f"\n--- Verifying traces in {args.trace_dir} ---")
    traces = load_session_traces(args.trace_dir, SESSION_ID)

    if not traces:
        print(f"  WARNING: No traces found for session {SESSION_ID}")
        print(f"  Make sure --enable-rl-logging and --rl-log-dir={args.trace_dir} are set")
        sys.exit(1)

    print(f"  Found {len(traces)} trace(s) for session {SESSION_ID}")

    ok = verify_traces(traces, turn)

    if ok:
        print("\nPASS: All traces show correct multi-turn structure!")
        print("The turn-merge bug is fixed.")
        sys.exit(0)
    else:
        print("\nFAIL: Traces show merged turns — the bug is NOT fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
