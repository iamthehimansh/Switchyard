# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`OpenAILLMClient`."""

from unittest.mock import AsyncMock, MagicMock

from openai import AsyncOpenAI

from switchyard.lib.llm_client import OpenAILLMClient


def test_constructs_only_the_async_client() -> None:
    """Only the async client is built.

    Backends call ``acompletion`` exclusively, so a sync ``OpenAI`` client
    would only allocate a second, never-used httpx connection pool (1000
    connections by default) per instance.
    """
    client = OpenAILLMClient(api_key="test-key")
    assert isinstance(client.async_client, AsyncOpenAI)
    assert not hasattr(client, "client")


def test_max_retries_reaches_the_sdk_client() -> None:
    """``max_retries`` is forwarded to the underlying SDK client."""
    client = OpenAILLMClient(api_key="test-key", max_retries=0)
    assert client.async_client.max_retries == 0


class TestAcompletionApiKeyOverride:
    """``acompletion`` overrides the construction-time key only for a real key.

    A blank or absent per-call ``api_key`` must fall back to the
    construction-time key (the configured endpoint key) instead of overriding
    it with nothing, which would unauthenticate the upstream call.
    """

    @staticmethod
    def _client_with_spied_options() -> tuple[OpenAILLMClient, MagicMock]:
        client = OpenAILLMClient(api_key="endpoint-key")
        client.async_client = MagicMock()
        client.async_client.chat.completions.create = AsyncMock(return_value="base")
        client.async_client.responses.create = AsyncMock(return_value="responses-base")
        overridden = MagicMock()
        overridden.chat.completions.create = AsyncMock(return_value="overridden")
        overridden.responses.create = AsyncMock(return_value="responses-overridden")
        client.async_client.with_options.return_value = overridden
        return client, client.async_client

    async def test_real_caller_key_uses_with_options(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.acompletion(api_key="caller-key", model="m")
        async_client.with_options.assert_called_once_with(api_key="caller-key")
        assert result == "overridden"

    async def test_none_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.acompletion(api_key=None, model="m")
        async_client.with_options.assert_not_called()
        assert result == "base"

    async def test_blank_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.acompletion(api_key="   ", model="m")
        async_client.with_options.assert_not_called()
        assert result == "base"

    async def test_responses_real_caller_key_uses_with_options(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.aresponses(api_key="caller-key", model="m", input="hi")
        async_client.with_options.assert_called_once_with(api_key="caller-key")
        assert result == "responses-overridden"

    async def test_responses_missing_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.aresponses(api_key=None, model="m", input="hi")
        async_client.with_options.assert_not_called()
        assert result == "responses-base"

    async def test_responses_blank_key_falls_back_to_construction_key(self) -> None:
        client, async_client = self._client_with_spied_options()
        result = await client.aresponses(api_key="   ", model="m", input="hi")
        async_client.with_options.assert_not_called()
        assert result == "responses-base"
