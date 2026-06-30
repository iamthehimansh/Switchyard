# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`probe_anthropic_messages_support`.

The probe is a single HTTP POST with empty body + real auth; responses
partition cleanly into "endpoint wired" vs "not wired / broken". Tests
mock httpx at the ``AsyncClient`` level.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from switchyard.lib.backends.backend_format_resolver import (
    probe_anthropic_messages_support,
    strip_v1_suffix,
)


def _fake_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _patch_httpx(response: MagicMock | Exception):
    """Return a context manager patching AsyncClient.post.

    If ``response`` is an Exception, it's raised from the POST; otherwise
    it's returned.
    """
    async_client = MagicMock()
    if isinstance(response, Exception):
        async_client.post = AsyncMock(side_effect=response)
    else:
        async_client.post = AsyncMock(return_value=response)

    # AsyncClient is used as an async context manager
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=async_client)
    async_cm.__aexit__ = AsyncMock(return_value=False)

    return patch.object(httpx, "AsyncClient", return_value=async_cm)


class TestProbeAnthropicMessagesSupport:
    async def test_404_returns_false(self):
        with _patch_httpx(_fake_response(404)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is False

    async def test_400_returns_true(self):
        # Endpoint exists, validator ran and rejected our empty body.
        with _patch_httpx(_fake_response(400)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is True

    async def test_422_returns_true(self):
        with _patch_httpx(_fake_response(422)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is True

    async def test_200_returns_true(self):
        with _patch_httpx(_fake_response(200)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is True

    async def test_401_returns_false(self):
        with _patch_httpx(_fake_response(401)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is False

    async def test_5xx_returns_false(self):
        with _patch_httpx(_fake_response(500)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is False

    async def test_timeout_returns_false(self):
        with _patch_httpx(httpx.TimeoutException("slow")):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is False

    async def test_timeout_does_not_log_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        with _patch_httpx(httpx.TimeoutException("slow")):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is False

        assert "Anthropic /v1/messages probe" not in caplog.text

    async def test_connection_error_returns_false(self):
        with _patch_httpx(httpx.ConnectError("dns fail")):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is False

    async def test_probe_url_is_root_plus_v1_messages(self):
        """``--base-url`` follows OpenAI convention (ends in ``/v1``) but
        the probe's target is ``{root}/v1/messages``. Make sure we don't
        produce ``/v1/v1/messages`` or strip essential path components.
        """
        captured: dict = {}

        async def fake_post(url, headers, json):
            captured["url"] = url
            return _fake_response(400)

        async_client = MagicMock()
        async_client.post = AsyncMock(side_effect=fake_post)
        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(return_value=async_client)
        async_cm.__aexit__ = AsyncMock(return_value=False)

        with patch.object(httpx, "AsyncClient", return_value=async_cm):
            await probe_anthropic_messages_support(
                base_url="https://inference-api.nvidia.com/v1", api_key="sk-test",
            )
        assert captured["url"] == "https://inference-api.nvidia.com/v1/messages"

    @pytest.mark.parametrize("status", [402, 403, 405, 429])
    async def test_other_4xx_treated_as_endpoint_exists(self, status):
        with _patch_httpx(_fake_response(status)):
            assert await probe_anthropic_messages_support(
                base_url="https://x.example/v1", api_key="sk-test",
            ) is True


class TestStripV1Suffix:
    def test_strips_trailing_v1(self):
        assert strip_v1_suffix("https://host/v1") == "https://host"

    def test_strips_trailing_v1_with_slash(self):
        assert strip_v1_suffix("https://host/v1/") == "https://host"

    def test_no_v1_suffix_untouched(self):
        assert strip_v1_suffix("https://host") == "https://host"

    def test_trailing_slash_stripped_even_when_no_v1(self):
        assert strip_v1_suffix("https://host/") == "https://host"

    def test_v1_in_middle_of_path_untouched(self):
        # Defensive: /v1 embedded in a longer path must not be stripped.
        assert strip_v1_suffix("https://host/v1/sub") == "https://host/v1/sub"
