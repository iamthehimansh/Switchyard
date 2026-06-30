# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Rust-backed ChatRequest values."""

import pytest

from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    request_type_enum,
    request_type_matches,
)


class TestChatRequestType:
    def test_enum_values(self):
        assert ChatRequestType.OPENAI_CHAT.value == "openai_chat"
        assert ChatRequestType.OPENAI_RESPONSES.value == "openai_responses"
        assert ChatRequestType.ANTHROPIC.value == "anthropic"

    def test_enum_members(self):
        assert [
            request_type_enum("openai_chat").value,
            request_type_enum("openai_responses").value,
            request_type_enum("anthropic").value,
        ] == ["openai_chat", "openai_responses", "anthropic"]

    def test_unknown_request_type_is_rejected(self):
        with pytest.raises(ValueError):
            request_type_enum("not_a_real_format")


class TestChatRequestConstructor:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ChatRequest()  # type: ignore[abstract]


class TestOpenAIChatBinding:
    @pytest.fixture()
    def body(self):
        return {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

    def test_request_type(self, body):
        req = ChatRequest.openai_chat(body)
        assert request_type_matches(req, ChatRequestType.OPENAI_CHAT)

    def test_body_access(self, body):
        req = ChatRequest.openai_chat(body)
        assert req.body == body
        assert req.body is not body

    def test_body_field_access(self, body):
        req = ChatRequest.openai_chat(body)
        assert req.body["model"] == "gpt-4o"
        assert req.body["messages"][0]["role"] == "user"

    def test_isinstance(self, body):
        req = ChatRequest.openai_chat(body)
        assert isinstance(req, ChatRequest)
        assert request_type_matches(req, ChatRequestType.OPENAI_CHAT)

    def test_body_unpack(self, body):
        """Verify **request.body works for SDK create() calls."""
        req = ChatRequest.openai_chat(body)
        unpacked = {**req.body}
        assert unpacked == body

    def test_body_mutation(self, body):
        req = ChatRequest.openai_chat(body)
        exported = req.body
        exported["model"] = "mutating-exported-copy-does-not-write-through"
        assert req.body["model"] == "gpt-4o"
        req.set_model("gpt-4o-mini")
        assert req.body["model"] == "gpt-4o-mini"


class TestResponsesChatBinding:
    @pytest.fixture()
    def body(self):
        return {
            "model": "gpt-4o",
            "input": "Hello, world!",
            "instructions": "Be helpful.",
        }

    def test_request_type(self, body):
        req = ChatRequest.openai_responses(body)
        assert request_type_matches(req, ChatRequestType.OPENAI_RESPONSES)

    def test_body_access(self, body):
        req = ChatRequest.openai_responses(body)
        assert req.body == body
        assert req.body is not body

    def test_body_field_access(self, body):
        req = ChatRequest.openai_responses(body)
        assert req.body["model"] == "gpt-4o"
        assert req.body["input"] == "Hello, world!"
        assert req.body["instructions"] == "Be helpful."

    def test_isinstance(self, body):
        req = ChatRequest.openai_responses(body)
        assert isinstance(req, ChatRequest)
        assert request_type_matches(req, ChatRequestType.OPENAI_RESPONSES)

    def test_responses_specific_fields(self):
        body = {
            "model": "gpt-4o",
            "input": "Continue",
            "previous_response_id": "resp_abc123",
            "truncation": "auto",
        }
        req = ChatRequest.openai_responses(body)
        assert req.body["previous_response_id"] == "resp_abc123"
        assert req.body["truncation"] == "auto"


class TestAnthropicChatBinding:
    @pytest.fixture()
    def body(self):
        return {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }

    def test_request_type(self, body):
        req = ChatRequest.anthropic(body)
        assert request_type_matches(req, ChatRequestType.ANTHROPIC)

    def test_body_access(self, body):
        req = ChatRequest.anthropic(body)
        assert req.body == body
        assert req.body is not body

    def test_body_field_access(self, body):
        req = ChatRequest.anthropic(body)
        assert req.body["model"] == "claude-sonnet-4-20250514"
        assert req.body["max_tokens"] == 1024

    def test_isinstance(self, body):
        req = ChatRequest.anthropic(body)
        assert isinstance(req, ChatRequest)
        assert request_type_matches(req, ChatRequestType.ANTHROPIC)

    def test_anthropic_specific_fields(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 2048,
            "system": "You are a helpful assistant.",
            "stream": True,
        }
        req = ChatRequest.anthropic(body)
        assert req.body["system"] == "You are a helpful assistant."
        assert req.body["stream"] is True


class TestRequestTypeDispatch:
    """Verify request-format dispatch uses the Rust request tag."""

    def _dispatch(self, request: ChatRequest) -> str:
        if request_type_matches(request, ChatRequestType.OPENAI_CHAT):
            return "openai_chat"
        if request_type_matches(request, ChatRequestType.OPENAI_RESPONSES):
            return "responses"
        if request_type_matches(request, ChatRequestType.ANTHROPIC):
            return "anthropic"
        return "unknown"

    def test_dispatch_openai(self):
        req = ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})
        assert self._dispatch(req) == "openai_chat"

    def test_dispatch_responses(self):
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hi"})
        assert self._dispatch(req) == "responses"

    def test_dispatch_anthropic(self):
        req = ChatRequest.anthropic({"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 1024})
        assert self._dispatch(req) == "anthropic"


class TestMatchDispatch:
    """Verify match dispatch uses the request-format enum."""

    def _dispatch(self, request: ChatRequest) -> str:
        match request_type_enum(request.request_type):
            case ChatRequestType.OPENAI_CHAT:
                return "openai_chat"
            case ChatRequestType.OPENAI_RESPONSES:
                return "responses"
            case ChatRequestType.ANTHROPIC:
                return "anthropic"
            case _:
                return "unknown"

    def test_match_openai(self):
        req = ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})
        assert self._dispatch(req) == "openai_chat"

    def test_match_responses(self):
        req = ChatRequest.openai_responses({"model": "gpt-4o", "input": "hi"})
        assert self._dispatch(req) == "responses"

    def test_match_anthropic(self):
        req = ChatRequest.anthropic({"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 1024})
        assert self._dispatch(req) == "anthropic"
