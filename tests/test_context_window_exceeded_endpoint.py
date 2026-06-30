# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-level tests: SwitchyardContextWindowExceededError on a single-target
route must return HTTP 400 context_length_exceeded, not HTTP 500."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from switchyard.lib.endpoints.anthropic_messages_endpoint import AnthropicMessagesEndpoint
from switchyard.lib.endpoints.openai_chat_endpoint import OpenAIChatEndpoint
from switchyard.lib.endpoints.responses_endpoint import ResponsesEndpoint
from switchyard_rust.core import SwitchyardContextWindowExceededError

_CHAT_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_ANTHROPIC_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
_RESPONSES_BODY = {"model": "m", "input": "hi"}


def _app_raising(exc: Exception) -> FastAPI:
    """Build a minimal FastAPI app whose chain always raises *exc*."""
    app = FastAPI()
    mock_sw = AsyncMock()
    mock_sw.call = AsyncMock(side_effect=exc)
    app.state.switchyard = mock_sw
    OpenAIChatEndpoint().register(app)
    AnthropicMessagesEndpoint().register(app)
    ResponsesEndpoint().register(app)
    return app


@pytest.fixture
async def window_client() -> AsyncIterator[httpx.AsyncClient]:
    """Async client wired to an app whose chain raises SwitchyardContextWindowExceededError."""
    exc = SwitchyardContextWindowExceededError("context window exceeded on single target")
    app = _app_raising(exc)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_openai_chat_window_exceeded_returns_400(window_client: httpx.AsyncClient) -> None:
    """POST /v1/chat/completions returns 400 context_length_exceeded, not 500."""
    resp = await window_client.post("/v1/chat/completions", json=_CHAT_BODY)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["type"] == "invalid_request_error"


async def test_anthropic_messages_window_exceeded_returns_400(window_client: httpx.AsyncClient) -> None:
    """POST /v1/messages returns 400 invalid_request_error, not 500."""
    resp = await window_client.post("/v1/messages", json=_ANTHROPIC_BODY)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["type"] == "invalid_request_error"


async def test_responses_window_exceeded_returns_400(window_client: httpx.AsyncClient) -> None:
    """POST /v1/responses returns 400 context_length_exceeded, not 500."""
    resp = await window_client.post("/v1/responses", json=_RESPONSES_BODY)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["type"] == "invalid_request_error"
