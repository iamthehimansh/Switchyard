# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic Messages to OpenAI Chat request translation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from switchyard_rust.translation import TranslationEngine

ENGINE = TranslationEngine()


def _translate_anthropic_request_to_openai(
    *,
    messages: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    return ENGINE.translate_request(
        "anthropic_messages",
        "openai_chat",
        {"messages": list(messages), **kwargs},
    )


def _system_messages(result: dict) -> list[dict]:
    return [m for m in result["messages"] if m.get("role") == "system"]


def test_system_as_string():
    result = _translate_anthropic_request_to_openai(
        messages=[{"role": "user", "content": "Hello"}],
        system="You are helpful.",
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
    )
    sys_msgs = _system_messages(result)
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "You are helpful."


def test_system_as_list_of_blocks():
    result = _translate_anthropic_request_to_openai(
        messages=[{"role": "user", "content": "Hello"}],
        system=[{"type": "text", "text": "You are helpful."}],
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
    )
    sys_msgs = _system_messages(result)
    assert len(sys_msgs) == 1
    assert "You are helpful." in sys_msgs[0]["content"]


def test_system_as_tuple_of_blocks_not_silently_dropped():
    """Tuples are valid Iterable[TextBlockParam] but isinstance(..., list) is False.

    Before the fix this test fails — the system prompt is silently dropped.
    """
    system = ({"type": "text", "text": "You are helpful."},)
    result = _translate_anthropic_request_to_openai(
        messages=[{"role": "user", "content": "Hello"}],
        system=system,
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
    )
    sys_msgs = _system_messages(result)
    assert len(sys_msgs) == 1, (
        f"System prompt from tuple was silently dropped. messages={result['messages']}"
    )
    assert "You are helpful." in sys_msgs[0]["content"]


def test_system_as_generator_of_blocks_not_silently_dropped():
    """Generators are valid Iterable[TextBlockParam] but isinstance(..., list) is False."""

    def block_gen():
        yield {"type": "text", "text": "You are a generator."}

    result = _translate_anthropic_request_to_openai(
        messages=[{"role": "user", "content": "Hello"}],
        system=block_gen(),
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
    )
    sys_msgs = _system_messages(result)
    assert len(sys_msgs) == 1, (
        f"System prompt from generator was silently dropped. messages={result['messages']}"
    )
    assert "You are a generator." in sys_msgs[0]["content"]


def test_system_none_produces_no_system_message():
    result = _translate_anthropic_request_to_openai(
        messages=[{"role": "user", "content": "Hello"}],
        system=None,
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
    )
    assert _system_messages(result) == []
