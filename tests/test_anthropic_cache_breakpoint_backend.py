# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`AnthropicCacheBreakpointBackend`.

The wrapper has two responsibilities:

1. **Translate** the inbound request to Anthropic Messages format
   (no-op if already Anthropic).
2. **Inject** ``cache_control: {"type": "ephemeral"}`` markers on
   the system field's last text block and the last message's last
   block.

It then delegates to an inner :class:`LLMBackend`.  These tests use
a fake inner backend to capture exactly what got passed through so
the markers + format change can be asserted without touching the
network.
"""

from __future__ import annotations

from typing import Any, cast

from switchyard.lib.backends.anthropic_cache_breakpoint_backend import (
    AnthropicCacheBreakpointBackend,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _CapturingBackend(LLMBackend):
    """Records the request handed to it and returns a canned response."""

    def __init__(self) -> None:
        self.calls: list[ChatRequest] = []
        self.startup_count = 0
        self.shutdown_count = 0

    async def startup(self) -> None:
        self.startup_count += 1

    async def shutdown(self) -> None:
        self.shutdown_count += 1

    async def call(
        self,
        ctx: ProxyContext,  # noqa: ARG002
        request: ChatRequest,
    ) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse.anthropic_completion(cast(Any, {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "test",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }))


def _openai_chat(messages: list[dict[str, Any]]) -> ChatRequest:
    return ChatRequest.openai_chat(cast(Any, {
        "model": "placeholder",
        "messages": messages,
    }))


def _anthropic(messages: list[dict[str, Any]], system: Any = None) -> ChatRequest:
    body: dict[str, Any] = {
        "model": "claude-test",
        "max_tokens": 128,
        "messages": messages,
    }
    if system is not None:
        body["system"] = system
    return ChatRequest.anthropic(cast(Any, body))


# ---------------------------------------------------------------------------
# Translation + marker placement
# ---------------------------------------------------------------------------


class TestOpenAIChatInbound:
    """Inbound OpenAI Chat ⇒ wrapper translates + marks before delegating."""

    async def test_inner_receives_anthropic_typed_request(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _openai_chat([
            {"role": "system", "content": "stable system prompt"},
            {"role": "user", "content": "do the thing"},
        ])
        await wrapper.call(ProxyContext(), request)
        assert len(inner.calls) == 1
        forwarded = inner.calls[0]
        assert forwarded.request_type.value == "anthropic"

    async def test_system_field_gets_ephemeral_marker(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _openai_chat([
            {"role": "system", "content": "stable system prompt"},
            {"role": "user", "content": "do the thing"},
        ])
        await wrapper.call(ProxyContext(), request)
        body = cast(dict[str, Any], inner.calls[0].body)
        system = body.get("system")
        assert isinstance(system, list), f"want list-shape system; got {system!r}"
        # Last text block carries the marker
        marker = None
        for blk in reversed(system):
            if isinstance(blk, dict) and blk.get("type") == "text":
                marker = blk
                break
        assert marker is not None
        assert marker["cache_control"] == {"type": "ephemeral"}
        assert "stable system prompt" in marker["text"]

    async def test_last_message_gets_rolling_marker(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _openai_chat([
            {"role": "user", "content": "first user message"},
            {"role": "assistant", "content": "midway response"},
            {"role": "user", "content": "the most recent turn"},
        ])
        await wrapper.call(ProxyContext(), request)
        body = cast(dict[str, Any], inner.calls[0].body)
        messages = body["messages"]
        last_content = messages[-1]["content"]
        assert isinstance(last_content, list), (
            f"last message content should be a list of typed blocks; got {last_content!r}"
        )
        marker = None
        for blk in reversed(last_content):
            if isinstance(blk, dict) and blk.get("type") == "text":
                marker = blk
                break
        assert marker is not None
        assert marker["cache_control"] == {"type": "ephemeral"}

    async def test_chain_level_request_unchanged(self) -> None:
        """The wrapper must not mutate the outer request.

        The terminal :class:`TranslationEngine` step uses
        ``request.request_type`` to pick the response format for the
        client.  If we mutated the outer request_type to Anthropic,
        OpenAI Chat clients (Harbor's LiteLLM) would receive
        Anthropic-shape responses and crash.
        """
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _openai_chat([
            {"role": "user", "content": "do the thing"},
        ])
        original_type = request.request_type
        original_body_snapshot = dict(cast(dict[str, Any], request.body))
        await wrapper.call(ProxyContext(), request)
        # Outer request must be untouched — only a *local* Anthropic
        # request was constructed for the inner backend.
        assert request.request_type is original_type
        assert dict(cast(dict[str, Any], request.body)) == original_body_snapshot


class TestAnthropicInbound:
    """Inbound Anthropic ⇒ wrapper skips translation, marks body in place."""

    async def test_already_anthropic_short_circuits_engine(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _anthropic(
            messages=[{"role": "user", "content": "task"}],
            system="stable system",
        )
        await wrapper.call(ProxyContext(), request)
        # Engine was never built since translation was unnecessary
        assert wrapper._engine is None
        # System still got marked
        body = cast(dict[str, Any], inner.calls[0].body)
        system = body["system"]
        assert isinstance(system, list)
        assert system[-1]["cache_control"] == {"type": "ephemeral"}

    async def test_last_message_string_content_gets_marker(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _anthropic(
            messages=[{"role": "user", "content": "task"}],
        )
        await wrapper.call(ProxyContext(), request)
        body = cast(dict[str, Any], inner.calls[0].body)
        last_content = body["messages"][-1]["content"]
        assert isinstance(last_content, list)
        assert last_content[-1]["cache_control"] == {"type": "ephemeral"}

    async def test_client_supplied_markers_pass_through_untouched(self) -> None:
        """A client that already set cache_control (Claude Code) is left as-is.

        Stacking our two breakpoints on top of the client's own could push the
        request past Anthropic's four-breakpoint cap, so the wrapper must not
        re-mark a body that already carries markers.
        """
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        request = _anthropic(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "task",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            system=[
                {
                    "type": "text",
                    "text": "stable system",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        )
        original_body = dict(cast(dict[str, Any], request.body))
        await wrapper.call(ProxyContext(), request)
        # No translation engine built, and the inner backend saw the body
        # verbatim — we did not add a third/fourth breakpoint.
        assert wrapper._engine is None
        forwarded = cast(dict[str, Any], inner.calls[0].body)
        assert forwarded == original_body


# ---------------------------------------------------------------------------
# Lifecycle forwarding
# ---------------------------------------------------------------------------


class TestLifecycle:
    """startup / shutdown forward to inner."""

    async def test_startup_forwards(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        await wrapper.startup()
        assert inner.startup_count == 1

    async def test_shutdown_forwards(self) -> None:
        inner = _CapturingBackend()
        wrapper = AnthropicCacheBreakpointBackend(inner)
        await wrapper.shutdown()
        assert inner.shutdown_count == 1
