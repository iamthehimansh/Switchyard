# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core role helpers and plain processor components."""

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    response_type_matches,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_ctx() -> ProxyContext:
    return ProxyContext()


def make_request() -> ChatRequest:
    return ChatRequest.openai_chat({"model": "gpt-4o", "messages": []})


# ---------------------------------------------------------------------------
# Backend instantiation guard
# ---------------------------------------------------------------------------


class TestABCsCannotBeInstantiated:
    def test_llm_backend(self):
        with pytest.raises(TypeError):
            LLMBackend()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete implementations for testing
# ---------------------------------------------------------------------------


class PassthroughProcessor:
    async def process(self, ctx, request):
        return request


class TaggedRequestProcessor:
    def __init__(self, tag: str) -> None:
        self._tag = tag

    async def process(self, ctx, request):
        ctx.metadata.setdefault("request_order", []).append(self._tag)
        return request


class EchoBackend(LLMBackend):
    @property
    def supported_request_types(self):
        return [ChatRequestType.OPENAI_CHAT]

    async def call(self, ctx, request):
        from openai.types.completion_usage import CompletionUsage

        return ChatResponse.openai_completion(
            ChatCompletion(
                id="chatcmpl-test",
                object="chat.completion",
                created=1700000000,
                model="gpt-4o",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="echo"),
                        finish_reason="stop",
                    )
                ],
                usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )


class NoopResponseProcessor:
    async def process(self, ctx, response):
        return response


class TaggedResponseProcessor:
    def __init__(self, tag: str) -> None:
        self._tag = tag

    async def process(self, ctx, response):
        ctx.metadata.setdefault("response_order", []).append(self._tag)
        return response


# ---------------------------------------------------------------------------
# Request-side component
# ---------------------------------------------------------------------------


class TestRequestProcessor:
    async def test_process(self):
        proc = PassthroughProcessor()
        req = make_request()
        result = await proc.process(make_ctx(), req)
        assert result is req


# ---------------------------------------------------------------------------
# LLMBackend
# ---------------------------------------------------------------------------


class TestLLMBackend:
    async def test_call(self):
        backend = EchoBackend()
        resp = await backend.call(make_ctx(), make_request())
        assert response_type_matches(resp, ChatResponseType.OPENAI_COMPLETION)
        assert resp.body["choices"][0]["message"]["content"] == "echo"


# ---------------------------------------------------------------------------
# Response-side component
# ---------------------------------------------------------------------------


class TestResponseProcessor:
    async def test_process(self):

        resp = ChatResponse.openai_completion(
            ChatCompletion(
                id="test", object="chat.completion", created=0, model="m",
                choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="x"), finish_reason="stop")],
            )
        )
        proc = NoopResponseProcessor()
        result = await proc.process(make_ctx(), resp)
        assert result is resp
