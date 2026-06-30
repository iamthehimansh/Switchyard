# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production end-to-end smoke test for ``switchyard`` passthrough.

Exercises the real production path end-to-end::

    openai SDK → HTTP (127.0.0.1) → switchyard passthrough
               → real backend → response

Uses the official ``openai`` Python SDK as the client so we're
validating the exact request shape a real customer integration would
produce, and we get SDK-side validation of the response wire shape
for free.

Unlike the in-process integration tests under ``tests/`` — which use
FastAPI's ``TestClient`` with a mocked backend — this test hits a real
OpenAI-compatible backend through the subprocess-launched
``switchyard`` CLI entry point.

The subprocess / server fixture lives in this package's
``conftest.py`` so the Responses and Anthropic sibling suites can share
the same running server.

Prerequisites:
    - ``OPENROUTER_API_KEY`` or ``NVIDIA_API_KEY`` env var (skips otherwise)

Run with::

    OPENROUTER_API_KEY=sk-or-... pytest tests/e2e/test_passthrough_e2e.py -v
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

import openai
import pytest

from ._helpers import (
    CHAT_COMPLETIONS_WEATHER_TOOL,
    resolve_tool_capable_model,
)

pytestmark = pytest.mark.integration

logger = logging.getLogger("e2e.passthrough")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_openai_client(passthrough_server: dict) -> openai.OpenAI:
    """Build an OpenAI SDK client pointed at the passthrough server.

    The passthrough server injects its own upstream ``api_key`` into
    calls it makes to the real backend, so the ``api_key`` we give the
    SDK here is just a non-empty placeholder the SDK needs for its
    ``Authorization`` header (the passthrough inbound layer doesn't
    validate it).
    """
    return openai.OpenAI(
        base_url=f"{passthrough_server['base_url']}/v1",
        api_key="not-used-passthrough-forwards-its-own",
        timeout=60.0,
        max_retries=0,
    )


def _reasoning_text(delta_or_message: Any) -> str | None:
    """Return ``reasoning`` / ``reasoning_content`` from a pydantic field.

    Both names are vendor extensions produced by reasoning models
    (different providers use different spellings), so they live in the
    pydantic ``model_extra`` bag rather than as typed attributes on
    ``ChatCompletionMessage`` / ``ChoiceDelta``.
    """
    extra = getattr(delta_or_message, "model_extra", None) or {}
    return extra.get("reasoning") or extra.get("reasoning_content")


def _collect_sdk_stream_deltas(
    chunks: Iterable[Any],
) -> tuple[int, str, str, list[dict]]:
    """Aggregate deltas from an OpenAI SDK ``ChatCompletionChunk`` stream.

    Reassembles fragmented fields across chunks using the OpenAI
    streaming contract:

    * ``delta.content`` fragments → concatenated string
    * ``delta.reasoning`` / ``delta.reasoning_content`` fragments (vendor
      extensions for reasoning models) → concatenated string
    * ``delta.tool_calls[*]`` fragments → rebuilt per-``index`` tool-call
      dicts matching the non-streaming ``message.tool_calls`` shape
      (``id``, ``type``, ``function.name``, ``function.arguments``)

    Returns ``(chunk_count, content, reasoning, tool_calls)``.
    """
    chunk_count = 0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict] = {}

    for chunk in chunks:
        chunk_count += 1
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        if delta.content:
            content_parts.append(delta.content)

        reasoning = _reasoning_text(delta)
        if reasoning:
            reasoning_parts.append(str(reasoning))

        for tc_delta in delta.tool_calls or []:
            idx = tc_delta.index if tc_delta.index is not None else 0
            entry = tool_calls_by_index.setdefault(
                idx,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tc_delta.id:
                entry["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    entry["function"]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    entry["function"]["arguments"] += tc_delta.function.arguments

    tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]
    return (
        chunk_count,
        "".join(content_parts),
        "".join(reasoning_parts),
        tool_calls,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPassthroughE2E:
    """OpenAI SDK → Switchyard chain → real backend round-trip."""

    # ------------------------------------------------------------------
    # Basic say-hi
    # ------------------------------------------------------------------

    def test_say_hi_get_response_back(self, passthrough_server: dict) -> None:
        """``client.chat.completions.create("hi")`` round-trips successfully.

        Validates the minimum contract of the passthrough chain:

        * the SDK can reach the server and the request deserializes
          cleanly into a ``ChatCompletion`` object
        * the assistant role is returned
        * non-empty assistant text comes back — in ``content`` for
          non-reasoning models, or ``reasoning_content`` for reasoning
          models that still had tokens left in their thinking phase

        ``max_tokens`` is deliberately generous to accommodate
        *reasoning* models (e.g. Qwen 3.5 / GPT-5 series) that spend
        the first several hundred tokens on internal reasoning before
        emitting user-visible ``content``.
        """
        client = _make_openai_client(passthrough_server)
        response = client.chat.completions.create(
            model=passthrough_server["model"],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2048,
        )

        assert response.object == "chat.completion"
        assert response.choices and len(response.choices) >= 1

        choice = response.choices[0]
        message = choice.message
        assert message.role == "assistant"

        content = message.content
        reasoning = _reasoning_text(message)
        assert content or reasoning, (
            f"No assistant text in either 'content' or 'reasoning_content' "
            f"(finish_reason={choice.finish_reason!r}): "
            f"{response.model_dump()}"
        )

        logger.info(f"  [passthrough] finish_reason:       {choice.finish_reason!r}")
        if content:
            logger.info(f"  [passthrough] content[:100]:      {content[:100]!r}")
        if reasoning:
            logger.info(f"  [passthrough] reasoning[:100]:    {reasoning[:100]!r}")

    def test_say_hi_streaming(self, passthrough_server: dict) -> None:
        """``stream=True`` yields ``ChatCompletionChunk`` objects.

        Validates the streaming contract end-to-end: the SDK can
        iterate the SSE stream (so the passthrough's envelope —
        ``Content-Type: text/event-stream``, ``data: ...`` frames,
        ``[DONE]`` terminator — must be well-formed), chunks
        deserialize as ``chat.completion.chunk``, and at least one
        delta carries visible text (``delta.content`` or vendor
        ``delta.reasoning``).
        """
        client = _make_openai_client(passthrough_server)
        stream = client.chat.completions.create(
            model=passthrough_server["model"],
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2048,
            stream=True,
        )

        chunk_count, content, reasoning, _ = _collect_sdk_stream_deltas(stream)

        assert chunk_count >= 1, "expected at least one streamed chunk"
        assert content or reasoning, (
            f"no delta carried visible content or reasoning tokens "
            f"across {chunk_count} chunks"
        )

        logger.info(f"  [passthrough] streamed {chunk_count} chunk frames + [DONE]")

    # ------------------------------------------------------------------
    # System prompt — verifies a ``system`` role is forwarded through
    # the chain to the backend and the response is well-formed.
    # ------------------------------------------------------------------

    def test_system_prompt_non_streaming(self, passthrough_server: dict) -> None:
        """``system`` + ``user`` messages flow through and the backend honors them.

        Planted keyword ``pineapple`` is instructed via the system
        prompt; when the model produces user-visible ``content`` we
        assert the keyword is present, proving the system turn reached
        the backend.  For reasoning-only truncations (content empty,
        all budget spent in ``reasoning_content``) we fall back to
        asserting well-formedness — the chain still faithfully carried
        the system turn even if the model didn't get to emit it.
        """
        client = _make_openai_client(passthrough_server)
        response = client.chat.completions.create(
            model=passthrough_server["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Always include "
                        "the exact word 'pineapple' in your response."
                    ),
                },
                {"role": "user", "content": "Say hello."},
            ],
            max_tokens=2048,
        )

        assert response.object == "chat.completion"
        choice = response.choices[0]
        message = choice.message
        assert message.role == "assistant"

        content = message.content or ""
        reasoning = _reasoning_text(message) or ""
        assert content or reasoning, (
            f"no assistant text: {response.model_dump()}"
        )

        if content:
            assert "pineapple" in content.lower(), (
                f"system prompt keyword missing from content — "
                f"system turn may not have been forwarded. "
                f"content={content[:200]!r}"
            )

        logger.info(f"  [passthrough][system] finish_reason={choice.finish_reason!r}")
        if content:
            logger.info(f"  [passthrough][system] content[:150]: {content[:150]!r}")

    def test_system_prompt_streaming(self, passthrough_server: dict) -> None:
        """Same system-prompt round-trip, but over SSE via SDK streaming."""
        client = _make_openai_client(passthrough_server)
        stream = client.chat.completions.create(
            model=passthrough_server["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Always include "
                        "the exact word 'pineapple' in your response."
                    ),
                },
                {"role": "user", "content": "Say hello."},
            ],
            max_tokens=2048,
            stream=True,
        )

        chunk_count, content, reasoning, _ = _collect_sdk_stream_deltas(stream)
        assert content or reasoning, (
            f"no streamed assistant text across {chunk_count} chunks"
        )

        if content:
            assert "pineapple" in content.lower(), (
                f"system prompt keyword missing from streamed content — "
                f"system turn may not have been forwarded. "
                f"content={content[:200]!r}"
            )

        logger.info(
            f"  [passthrough][system-stream] {chunk_count} chunks, "
            f"content[:100]={content[:100]!r}"
        )

    # ------------------------------------------------------------------
    # Multi-turn — verifies prior ``user`` + ``assistant`` turns are
    # forwarded so the backend sees full conversation history.
    # ------------------------------------------------------------------

    def test_multi_turn_non_streaming(self, passthrough_server: dict) -> None:
        """Prior conversation history flows through the chain.

        Plants the name ``Alice`` two turns back, then asks the model
        to recall it.  If the history reached the backend, the name
        appears in the reply; if only the last user turn reached it,
        the model has nothing to recall.  Reasoning-only cut-offs are
        forgiven (same convention as the system-prompt test above).
        """
        client = _make_openai_client(passthrough_server)
        response = client.chat.completions.create(
            model=passthrough_server["model"],
            messages=[
                {
                    "role": "user",
                    "content": "My name is Alice. Please remember this.",
                },
                {
                    "role": "assistant",
                    "content": "Got it, Alice. I will remember your name.",
                },
                {
                    "role": "user",
                    "content": (
                        "What is my name? Reply with only the name "
                        "itself, nothing else."
                    ),
                },
            ],
            max_tokens=2048,
        )

        assert response.object == "chat.completion"
        choice = response.choices[0]
        message = choice.message
        assert message.role == "assistant"

        content = message.content or ""
        reasoning = _reasoning_text(message) or ""
        assert content or reasoning, (
            f"no assistant text: {response.model_dump()}"
        )

        if content:
            assert "alice" in content.lower(), (
                f"prior-turn context not recalled — multi-turn history "
                f"may not have been forwarded. content={content[:200]!r}"
            )

        logger.info(f"  [passthrough][multi-turn] finish_reason={choice.finish_reason!r}")
        if content:
            logger.info(f"  [passthrough][multi-turn] content[:150]: {content[:150]!r}")

    def test_multi_turn_streaming(self, passthrough_server: dict) -> None:
        """Same multi-turn round-trip, but over SSE via SDK streaming."""
        client = _make_openai_client(passthrough_server)
        stream = client.chat.completions.create(
            model=passthrough_server["model"],
            messages=[
                {
                    "role": "user",
                    "content": "My name is Alice. Please remember this.",
                },
                {
                    "role": "assistant",
                    "content": "Got it, Alice. I will remember your name.",
                },
                {
                    "role": "user",
                    "content": (
                        "What is my name? Reply with only the name "
                        "itself, nothing else."
                    ),
                },
            ],
            max_tokens=2048,
            stream=True,
        )

        chunk_count, content, reasoning, _ = _collect_sdk_stream_deltas(stream)
        assert content or reasoning, (
            f"no streamed assistant text across {chunk_count} chunks"
        )

        if content:
            assert "alice" in content.lower(), (
                f"prior-turn context not recalled over SSE — multi-turn "
                f"history may not have been forwarded. "
                f"content={content[:200]!r}"
            )

        logger.info(
            f"  [passthrough][multi-turn-stream] {chunk_count} chunks, "
            f"content[:100]={content[:100]!r}"
        )

    # ------------------------------------------------------------------
    # Tool calls — verifies ``tools`` definitions flow to the backend
    # and an OpenAI-format tool call comes back through the chain.
    # ------------------------------------------------------------------

    def test_tool_call_non_streaming(self, passthrough_server: dict) -> None:
        """``tools`` + ``tool_choice`` round-trip end-to-end.

        Validates that:

        * the ``tools`` definition is forwarded to the backend,
        * the backend's ``tool_calls`` response survives the chain,
        * the first tool call has the expected OpenAI shape
          (``id``, ``type=function``, ``function.name``,
          ``function.arguments``) and the arguments parse as JSON
          containing the requested parameter.

        Uses :func:`resolve_tool_capable_model` to avoid vLLM-hosted
        models that lack ``--enable-auto-tool-choice`` and would reject
        ``tool_choice`` with HTTP 400 — we want to test the
        passthrough's tool wiring, not the backend's flags.
        """
        tool_model = resolve_tool_capable_model(passthrough_server["model"])
        logger.info(f"  [passthrough][tools] using tool-capable model: {tool_model!r}")

        client = _make_openai_client(passthrough_server)
        response = client.chat.completions.create(
            model=tool_model,
            messages=[{
                "role": "user",
                "content": (
                    "What is the weather in Tokyo? "
                    "You MUST use the get_weather tool."
                ),
            }],
            tools=[CHAT_COMPLETIONS_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=2048,
        )

        assert response.object == "chat.completion"
        message = response.choices[0].message
        assert message.role == "assistant"

        tool_calls = message.tool_calls or []
        assert tool_calls, (
            f"expected at least one tool call from the backend, "
            f"got message={message.model_dump()}"
        )

        first = tool_calls[0]
        assert first.type == "function"
        assert first.function.name == "get_weather", (
            f"unexpected function name: {first.function.name!r}"
        )

        raw_args = first.function.arguments or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        assert "city" in args, (
            f"expected 'city' in arguments, got {args!r}"
        )

        logger.info(
            f"  [passthrough][tools] tool_calls={len(tool_calls)}, "
            f"fn={first.function.name!r}, args={args!r}"
        )

    def test_tool_call_streaming(self, passthrough_server: dict) -> None:
        """Streaming tool-call deltas reassemble into a valid call.

        Each chunk's ``delta.tool_calls`` carries incremental fragments
        (``id`` on the first chunk, ``function.name`` on one chunk,
        ``function.arguments`` split across many).  The helper
        :func:`_collect_sdk_stream_deltas` rebuilds them using the
        OpenAI streaming contract; this test then runs the same
        structural assertions as the non-streaming variant.

        Uses :func:`resolve_tool_capable_model` for the same reason as
        the non-streaming tool test.
        """
        tool_model = resolve_tool_capable_model(passthrough_server["model"])
        logger.info(f"  [passthrough][tools-stream] using tool-capable model: {tool_model!r}")

        client = _make_openai_client(passthrough_server)
        stream = client.chat.completions.create(
            model=tool_model,
            messages=[{
                "role": "user",
                "content": (
                    "What is the weather in Tokyo? "
                    "You MUST use the get_weather tool."
                ),
            }],
            tools=[CHAT_COMPLETIONS_WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=2048,
            stream=True,
        )

        chunk_count, content, reasoning, tool_calls = _collect_sdk_stream_deltas(
            stream,
        )
        assert tool_calls, (
            f"expected at least one streamed tool call, got "
            f"{chunk_count} chunks, "
            f"content[:100]={content[:100]!r}, "
            f"reasoning[:100]={reasoning[:100]!r}"
        )

        first = tool_calls[0]
        assert first.get("type") == "function"
        fn = first.get("function") or {}
        assert fn.get("name") == "get_weather", (
            f"unexpected function name: {fn.get('name')!r}"
        )

        raw_args = fn.get("arguments") or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        assert "city" in args, (
            f"expected 'city' in reassembled arguments, got {args!r}"
        )

        logger.info(
            f"  [passthrough][tools-stream] {chunk_count} chunks, "
            f"tool_calls={len(tool_calls)}, fn={fn.get('name')!r}, "
            f"args={args!r}"
        )
