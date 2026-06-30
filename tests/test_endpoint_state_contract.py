# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the ``app.state`` wiring between
``build_switchyard_app`` and the three inbound endpoint handlers.

The bug fixed in PR #28 was that the factory wrote ``app.state.switchyard``
while every endpoint read ``app.state.switchyard`` — the server started
cleanly and unit tests stayed green, but every request hit
``AttributeError`` at runtime.

These tests pin both sides of the contract against ``Switchyard.state_key``
so a future rename either updates everything together or fails CI.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchyard.lib.endpoints.anthropic_messages_endpoint import (
    AnthropicMessagesEndpoint,
)
from switchyard.lib.endpoints.openai_chat_endpoint import (
    OpenAIChatEndpoint,
)
from switchyard.lib.endpoints.responses_endpoint import (
    ResponsesEndpoint,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.roles import LLMBackend
from switchyard.lib.switchyard import Switchyard
from switchyard.server.switchyard_app import build_switchyard_app
from switchyard_rust.core import ChatRequest, ChatRequestType, ChatResponse
from switchyard_rust.translation import TranslationEngine

_ENDPOINT_CLASSES = (
    OpenAIChatEndpoint,
    AnthropicMessagesEndpoint,
    ResponsesEndpoint,
)
_ALL_REQUEST_TYPES = [
    ChatRequestType.OPENAI_CHAT,
    ChatRequestType.OPENAI_RESPONSES,
    ChatRequestType.ANTHROPIC,
]


class _StubBackend(LLMBackend):
    def supported_request_types(self) -> list[ChatRequestType]:
        return list(_ALL_REQUEST_TYPES)

    async def call(self, ctx: ProxyContext, request: ChatRequest) -> ChatResponse:
        completion = ChatCompletion(
            id="chatcmpl-stub",
            object="chat.completion",
            created=1700000000,
            model="stub",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        return ChatResponse.openai_completion(completion)


def _build_app() -> tuple[FastAPI, Switchyard]:
    switchyard = Switchyard(backend=_StubBackend(), translator=TranslationEngine())
    app = build_switchyard_app(switchyard)
    return app, switchyard


def test_state_key_constant_is_switchyard() -> None:
    """Pin the expected key. If this changes, every reader must change too."""
    assert Switchyard.state_key == "switchyard"


def test_factory_writes_switchyard_to_state_key_constant() -> None:
    """``build_switchyard_app`` must store the instance under ``Switchyard.state_key``.

    Direct regression guard for PR #28: the factory previously wrote to
    ``app.state.switchyard`` (wrong) instead of ``app.state.switchyard``.
    """
    app, switchyard = _build_app()
    stored = getattr(app.state, Switchyard.state_key)
    assert stored is switchyard


@pytest.mark.parametrize(
    "endpoint_cls",
    _ENDPOINT_CLASSES,
    ids=lambda c: c.__name__,
)
def test_endpoint_module_reads_state_key_constant(endpoint_cls: type) -> None:
    """Each endpoint module must read from the same key the factory writes.

    Source-level check rather than a runtime assertion because the contract
    is "the literal attribute name agrees" — a runtime test still passes if
    both sides are wrong-and-matching, but the source check fails the moment
    either side drifts from ``Switchyard.state_key``.
    """
    module = inspect.getmodule(endpoint_cls)
    assert module is not None
    expected = f"request.app.state.{Switchyard.state_key}"
    source = inspect.getsource(module)
    assert expected in source, (
        f"{endpoint_cls.__name__} does not read from {expected!r}; "
        f"the factory and the endpoint disagree on app.state."
    )
