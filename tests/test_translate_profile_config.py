# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-owned translate profile."""

import pytest
from anthropic.types import Message, TextBlock, Usage

from switchyard import ProfileInput, TranslateProfileConfig, build_profile
from switchyard.lib.processors.format_translate import TranslateConfig
from switchyard.lib.proxy_context import (
    CTX_ORIGINAL_FORMAT,
    CTX_TARGET_FORMAT,
    ProxyContext,
)
from switchyard_rust.core import (
    ChatRequest,
    ChatRequestType,
    ChatResponse,
    ChatResponseType,
    SwitchyardBackendError,
    request_type_matches,
    response_type_matches,
)


class _SelectModelProcessor:
    """Test-only selector that mimics a host shim stamping the chosen model."""

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        """Stamp the selected model before translate lookup runs."""
        ctx.selected_model = "anthropic-target"
        return request


def _openai_request() -> ChatRequest:
    """Build the inbound client request used by translate-profile tests."""
    return ChatRequest.openai_chat({
        "model": "client-model",
        "messages": [{"role": "user", "content": "Say ok"}],
    })


def _anthropic_response() -> ChatResponse:
    """Build the host backend response returned between process and rprocess."""
    return ChatResponse.anthropic_completion(
        Message(
            id="msg_translate_profile",
            type="message",
            role="assistant",
            content=[TextBlock(type="text", text="ok")],
            model="anthropic-target",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=4, output_tokens=1),
        )
    )


async def test_translate_profile_process_and_rprocess_wrap_existing_processors() -> None:
    """Translate profile supports the host-owned process/backend/rprocess split."""
    profile = build_profile(TranslateProfileConfig(
        config=TranslateConfig(
            models=[{"model": "anthropic-target", "backend_format": "anthropic"}]
        ),
        model_selection_processors=(_SelectModelProcessor(),),
    ))

    processed = await profile.process(ProfileInput(_openai_request()))
    assert request_type_matches(processed.request, ChatRequestType.ANTHROPIC)
    assert processed.ctx.metadata[CTX_ORIGINAL_FORMAT] == ChatRequestType.OPENAI_CHAT
    assert processed.ctx.metadata[CTX_TARGET_FORMAT] == ChatRequestType.ANTHROPIC

    response = await profile.rprocess(processed, _anthropic_response())

    assert response_type_matches(response, ChatResponseType.OPENAI_COMPLETION)
    assert response.body["choices"][0]["message"]["content"] == "ok"


async def test_translate_profile_run_fails_clearly_without_host_backend() -> None:
    """Translate profile is hook-only because the embedding host owns the call."""
    profile = build_profile(TranslateProfileConfig())

    with pytest.raises(SwitchyardBackendError, match="hook-only"):
        await profile.run(ProfileInput(_openai_request()))
