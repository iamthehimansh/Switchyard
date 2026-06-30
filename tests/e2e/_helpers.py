# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for passthrough e2e tests.

Hosts the three vendor-specific ``get_weather`` tool definitions and
the tool-capable-model resolver used by the Chat Completions,
Responses, and Anthropic Messages e2e suites.  Pure Python — no
fixtures, so we keep this as a regular module rather than a
``conftest.py``.
"""

from __future__ import annotations

import os


def resolve_tool_capable_model(default_model: str) -> str:
    """Pick a backend model that supports OpenAI-style tool calling.

    Some NVIDIA-hosted vLLM deployments (notably the
    ``nvidia/qwen/qwen3.5-*`` family) aren't launched with
    ``--enable-auto-tool-choice`` / ``--tool-call-parser`` and reject
    ``tool_choice`` with HTTP 400 — the passthrough chain itself is
    fine, the backend just won't parse tool calls.  Tool-call tests
    should therefore run against a known tool-capable model so we're
    validating the passthrough's tool wiring rather than the backend's
    vLLM flags.

    Resolution order:

    1. ``OPENROUTER_TOOL_MODEL`` / ``NVIDIA_TOOL_MODEL`` env var —
       explicit override for any backend or any model
    2. ``openai/openai/gpt-5.2`` when ``default_model`` starts with
       ``nvidia/`` — that's the vLLM-hosted family described above,
       and ``openai/openai/gpt-5.2`` is known to support tool calling
       on ``inference-api.nvidia.com``
    3. ``default_model`` otherwise — so non-NVIDIA backends keep
       whatever the test suite was configured with
    """
    override = os.environ.get("OPENROUTER_TOOL_MODEL") or os.environ.get("NVIDIA_TOOL_MODEL")
    if override:
        return override
    if default_model.startswith("nvidia/"):
        return "openai/openai/gpt-5.2"
    return default_model


# ---------------------------------------------------------------------------
# Tool definitions — same semantic tool (``get_weather(city)``) rendered
# in each vendor's wire format, because each API has subtly different
# expectations:
#
# * Chat Completions nests the function metadata under ``function``.
# * Responses API flattens it out alongside ``type=function``.
# * Anthropic uses ``input_schema`` instead of ``parameters`` and has
#   no ``type`` wrapper at all.
# ---------------------------------------------------------------------------


_WEATHER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "The city name, e.g. 'Tokyo'.",
        },
    },
    "required": ["city"],
}

_WEATHER_NAME = "get_weather"
_WEATHER_DESC = "Get the current weather for a city."


# Chat Completions format: ``{"type": "function", "function": {...}}``.
CHAT_COMPLETIONS_WEATHER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": _WEATHER_NAME,
        "description": _WEATHER_DESC,
        "parameters": _WEATHER_SCHEMA,
    },
}


# Responses API format: flat — ``type``, ``name``, ``description``,
# ``parameters`` all at the top level.
RESPONSES_WEATHER_TOOL: dict = {
    "type": "function",
    "name": _WEATHER_NAME,
    "description": _WEATHER_DESC,
    "parameters": _WEATHER_SCHEMA,
}


# Anthropic Messages format: no ``type``, ``input_schema`` instead of
# ``parameters``.
ANTHROPIC_WEATHER_TOOL: dict = {
    "name": _WEATHER_NAME,
    "description": _WEATHER_DESC,
    "input_schema": _WEATHER_SCHEMA,
}
