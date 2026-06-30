# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified TranslationEngine request wrapper."""

import copy
import json

from switchyard_rust.core import ChatRequest, ChatRequestType, request_type_matches
from switchyard_rust.translation import TranslationEngine

E = TranslationEngine()


# =========================================================================
# to_openai_chat
# =========================================================================


class TestToOpenAIChat:
    def test_openai_passthrough(self):
        """OpenAI chat requests pass through unchanged (same object)."""
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result is req

    def test_anthropic_to_openai_basic(self):
        """Anthropic requests are converted to OpenAI chat requests."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "system": "Be helpful.",
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)

        assert request_type_matches(result, ChatRequestType.OPENAI_CHAT)
        msgs = result.body["messages"]
        # System prompt becomes first message
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be helpful."
        # User message follows
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"
        assert result.body["model"] == "claude-sonnet-4-20250514"

    def test_anthropic_to_openai_with_tools(self):
        """Anthropic tools are converted to OpenAI tool format."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)

        assert request_type_matches(result, ChatRequestType.OPENAI_CHAT)
        tools = result.body.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"

    def test_anthropic_to_openai_does_not_mutate_original(self):
        """The original Anthropic request body is not modified."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        original = copy.deepcopy(body)
        req = ChatRequest.anthropic(body)
        E.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert req.body == original

    def test_responses_to_openai_basic(self):
        """Responses requests are converted to OpenAI chat requests."""
        body = {
            "model": "gpt-4o",
            "input": "Hello world",
            "instructions": "Be brief.",
        }
        req = ChatRequest.openai_responses(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)

        assert request_type_matches(result, ChatRequestType.OPENAI_CHAT)
        msgs = result.body["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be brief."
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello world"

    def test_responses_to_openai_with_tools(self):
        """Responses API tools are converted to OpenAI tool format."""
        body = {
            "model": "gpt-4o",
            "input": "Get weather",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"loc": {"type": "string"}},
                    },
                }
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)

        tools = result.body.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"

    def test_responses_to_openai_does_not_mutate_original(self):
        """The original Responses request body is not modified."""
        body = {"model": "gpt-4o", "input": "Hi"}
        original = copy.deepcopy(body)
        req = ChatRequest.openai_responses(body)
        E.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert req.body == original

    def test_anthropic_only_top_level_fields_dropped(self):
        """Anthropic-only fields must not leak into the OpenAI request — the
        OpenAI SDK rejects them with TypeError at call time.

        Claude Code sends several of these on every request (``thinking`` for
        extended thinking, ``cache_control`` for prompt caching,
        ``context_management`` for long-context management, ``container`` for
        Claude containers). Including a synthetic ``made_up_beta_field`` to
        assert the whitelist strategy also handles future Anthropic-only
        fields we don't know about yet.
        """
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "cache_control": {"type": "ephemeral"},
            "container": "my-container",
            "inference_geo": "us-east-1",
            "output_config": {"some": "config"},
            "context_management": {"strategy": "auto"},
            "made_up_beta_field": "future-proofing",
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)
        for field in (
            "thinking", "cache_control", "container", "inference_geo",
            "output_config", "context_management", "made_up_beta_field",
        ):
            assert field not in result.body, (
                f"{field!r} leaked into OpenAI request — "
                "OpenAI SDK would reject with TypeError"
            )

    def test_anthropic_thinking_content_blocks_do_not_leak_to_openai(self):
        """Anthropic thinking blocks are preserved internally, not sent as Chat content."""
        body = {
            "model": "claude-opus-4-20250514",
            "messages": [
                {"role": "user", "content": "Use the tool."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should call the tool.",
                            "signature": "sig-abc",
                        },
                        {"type": "redacted_thinking", "data": "encrypted"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"query": "status"},
                        },
                    ],
                },
            ],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "max_tokens": 2048,
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)

        assistant = result.body["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] is None
        assert "reasoning_content" not in assistant
        assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
        assert "thinking" not in str(result.body)
        assert "redacted_thinking" not in str(result.body)


# =========================================================================
# to_anthropic
# =========================================================================


class TestToAnthropic:
    def test_anthropic_passthrough(self):
        """Anthropic requests pass through unchanged."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)
        assert result is req

    def test_openai_to_anthropic(self):
        """OpenAI chat requests are converted to Anthropic requests."""
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)

        assert request_type_matches(result, ChatRequestType.ANTHROPIC)
        assert result.body["system"] == "Be helpful."
        assert result.body["model"] == "gpt-4o"

    def test_openai_to_anthropic_developer_and_system_concat(self):
        """OpenAI developer/system messages must not leak as invalid Anthropic roles."""
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "System rules."},
                {"role": "developer", "content": "Developer rules."},
                {"role": "user", "content": "Hello"},
            ],
        }
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)

        assert result.body["system"] == "System rules.\n\nDeveloper rules."
        assert [m["role"] for m in result.body["messages"]] == ["user"]

    def test_openai_to_anthropic_uses_max_completion_tokens(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 512,
        }
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)
        assert result.body["max_tokens"] == 512

    def test_openai_to_anthropic_maps_reasoning_effort(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
        }
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)
        assert result.body["thinking"] == {"type": "adaptive"}
        assert result.body["output_config"] == {"effort": "high"}

    def test_openai_to_anthropic_maps_image_url_content(self):
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.test/image.png",
                            },
                        },
                    ],
                }
            ],
        }
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)
        content = result.body["messages"][0]["content"]
        assert content == [
            {"type": "text", "text": "Describe"},
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.test/image.png",
                },
            },
        ]

    def test_openai_to_anthropic_merges_consecutive_tool_results(self):
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "call tools"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "a", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "one"},
                {"role": "tool", "tool_call_id": "call_2", "content": "two"},
            ],
        }
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)

        tool_result_msg = result.body["messages"][2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"] == [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "one"},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "two"},
        ]

    def test_anthropic_tool_result_followup_text_is_preserved(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "72F",
                        },
                        {"type": "text", "text": "Now summarize it."},
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.OPENAI_CHAT, req)
        assert result.body["messages"] == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "72F"},
            {"role": "user", "content": "Now summarize it."},
        ]

    def test_openai_to_anthropic_sanitizes_tool_call_ids(self):
        """Tool call IDs and matching tool results must stay valid together."""
        invalid_id = "call.bad:id/with space"
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": invalid_id,
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"test"}',
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": invalid_id,
                    "content": "ok",
                },
            ],
            "max_tokens": 1024,
        }
        req = ChatRequest.openai_chat(body)

        result = E.request_to(ChatRequestType.ANTHROPIC, req)

        messages = result.body["messages"]
        sanitized_id = "call_bad_id_with_space"
        assert messages[1]["content"][0]["id"] == sanitized_id
        assert messages[2]["content"][0]["tool_use_id"] == sanitized_id

    def test_openai_unknown_content_does_not_leak_to_anthropic(self):
        """Unknown OpenAI content is kept as text instead of raw Anthropic blocks."""
        unknown_part = {"type": "future_openai_part", "payload": {"keep": True}}
        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        unknown_part,
                    ],
                }
            ],
        }
        req = ChatRequest.openai_chat(body)

        result = E.request_to(ChatRequestType.ANTHROPIC, req)

        content = result.body["messages"][0]["content"]
        assert content[1]["type"] == "text"
        assert json.loads(content[1]["text"]) == unknown_part
        assert "future_openai_part" not in [
            block.get("type") for block in content if isinstance(block, dict)
        ]

    def test_responses_to_anthropic(self):
        """Responses requests can be converted to Anthropic requests."""
        body = {"model": "gpt-4o", "input": "Hi"}
        req = ChatRequest.openai_responses(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)
        assert request_type_matches(result, ChatRequestType.ANTHROPIC)
        assert result.body["messages"] == [{"role": "user", "content": "Hi"}]

    def test_responses_tool_arguments_to_anthropic_are_object_shaped(self):
        """Responses tool-call argument strings become valid Anthropic tool input objects."""
        body = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "List files"},
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_1",
                    "arguments": '{"cmd":"ls","limit":2}',
                },
            ],
        }
        req = ChatRequest.openai_responses(body)
        result = E.request_to(ChatRequestType.ANTHROPIC, req)

        tool_use = result.body["messages"][1]["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["input"] == {"cmd": "ls", "limit": 2}


# =========================================================================
# to_responses
# =========================================================================


class TestToResponses:
    def test_responses_passthrough(self):
        """Responses requests pass through unchanged."""
        body = {"model": "gpt-4o", "input": "Hi"}
        req = ChatRequest.openai_responses(body)
        result = E.request_to(ChatRequestType.OPENAI_RESPONSES, req)
        assert result is req

    def test_openai_to_responses(self):
        """OpenAI chat requests can be converted to Responses requests."""
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        req = ChatRequest.openai_chat(body)
        result = E.request_to(ChatRequestType.OPENAI_RESPONSES, req)
        assert request_type_matches(result, ChatRequestType.OPENAI_RESPONSES)
        assert result.body["input"] == "Hi"

    def test_anthropic_to_responses(self):
        """Anthropic requests can be converted to Responses requests."""
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        req = ChatRequest.anthropic(body)
        result = E.request_to(ChatRequestType.OPENAI_RESPONSES, req)
        assert request_type_matches(result, ChatRequestType.OPENAI_RESPONSES)
        assert result.body["input"] == "Hi"
