# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :meth:`TranslationEngine.request_to_any_of`.

Covers the four behaviors:

- Passthrough when the inbound type is in ``supported``.
- Translation to ``supported[0]`` when the inbound type isn't supported.
- ``ValueError`` on empty ``supported``.
- Any built-in request format can translate to any other built-in format.
"""

import pytest

from switchyard_rust.core import ChatRequest, ChatRequestType, request_type_matches
from switchyard_rust.translation import TranslationEngine

ENGINE = TranslationEngine()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _openai_req() -> ChatRequest:
    return ChatRequest.openai_chat({  # type: ignore[arg-type]
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    })


def _anthropic_req() -> ChatRequest:
    return ChatRequest.anthropic({  # type: ignore[arg-type]
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1024,
    })


def _responses_req() -> ChatRequest:
    return ChatRequest.openai_responses({  # type: ignore[arg-type]
        "model": "gpt-4o",
        "input": "hi",
    })


# ---------------------------------------------------------------------------
# Passthrough (inbound type is in `supported`)
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_openai_in_singleton_supported(self):
        req = _openai_req()
        out = ENGINE.request_to_any_of(
            req, [ChatRequestType.OPENAI_CHAT],
        )
        assert out is req

    def test_anthropic_in_singleton_supported(self):
        req = _anthropic_req()
        out = ENGINE.request_to_any_of(
            req, [ChatRequestType.ANTHROPIC],
        )
        assert out is req

    def test_responses_in_singleton_supported(self):
        req = _responses_req()
        out = ENGINE.request_to_any_of(
            req, [ChatRequestType.OPENAI_RESPONSES],
        )
        assert out is req

    def test_openai_in_multi_element_supported(self):
        """Passthrough still wins even when supported lists more formats."""
        req = _openai_req()
        out = ENGINE.request_to_any_of(
            req,
            [ChatRequestType.OPENAI_RESPONSES, ChatRequestType.OPENAI_CHAT],
        )
        assert out is req

    def test_anthropic_in_multi_element_supported(self):
        req = _anthropic_req()
        out = ENGINE.request_to_any_of(
            req,
            [ChatRequestType.OPENAI_CHAT, ChatRequestType.ANTHROPIC],
        )
        assert out is req


# ---------------------------------------------------------------------------
# Translation (inbound type not in `supported`, translate to supported[0])
# ---------------------------------------------------------------------------


class TestTranslation:
    def test_openai_chat_to_anthropic(self):
        req = _openai_req()
        out = ENGINE.request_to_any_of(
            req, [ChatRequestType.ANTHROPIC],
        )
        assert request_type_matches(out, ChatRequestType.ANTHROPIC)
        assert out is not req

    def test_anthropic_to_openai_chat(self):
        req = _anthropic_req()
        out = ENGINE.request_to_any_of(
            req, [ChatRequestType.OPENAI_CHAT],
        )
        assert request_type_matches(out, ChatRequestType.OPENAI_CHAT)
        assert out is not req

    def test_responses_to_openai_chat(self):
        req = _responses_req()
        out = ENGINE.request_to_any_of(
            req, [ChatRequestType.OPENAI_CHAT],
        )
        assert request_type_matches(out, ChatRequestType.OPENAI_CHAT)
        assert out is not req

    def test_translation_target_is_supported_zero(self):
        """When multiple unsupported formats translate, supported[0] is chosen."""
        req = _anthropic_req()
        # Anthropic is not in supported; supported[0] is OPENAI_CHAT, so we
        # should translate to OpenAI Chat (not to Responses, the second entry).
        out = ENGINE.request_to_any_of(
            req,
            [ChatRequestType.OPENAI_CHAT, ChatRequestType.OPENAI_RESPONSES],
        )
        assert request_type_matches(out, ChatRequestType.OPENAI_CHAT)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_supported_raises_value_error(self):
        req = _openai_req()
        with pytest.raises(ValueError, match="must be non-empty"):
            ENGINE.request_to_any_of(req, [])

    def test_openai_chat_to_responses(self):
        """OpenAI Chat can translate to Responses."""
        req = _openai_req()
        out = ENGINE.request_to_any_of(req, [ChatRequestType.OPENAI_RESPONSES])
        assert request_type_matches(out, ChatRequestType.OPENAI_RESPONSES)

    def test_anthropic_to_responses(self):
        """Anthropic can translate to Responses."""
        req = _anthropic_req()
        out = ENGINE.request_to_any_of(req, [ChatRequestType.OPENAI_RESPONSES])
        assert request_type_matches(out, ChatRequestType.OPENAI_RESPONSES)

    def test_responses_to_anthropic(self):
        """Responses can translate to Anthropic."""
        req = _responses_req()
        out = ENGINE.request_to_any_of(req, [ChatRequestType.ANTHROPIC])
        assert request_type_matches(out, ChatRequestType.ANTHROPIC)
