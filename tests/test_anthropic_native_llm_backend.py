# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python binding tests for the Rust-owned Anthropic native backend."""

from __future__ import annotations

import pytest

from switchyard.lib.backends.multi_llm_backend import build_native_backend
from switchyard_rust.components import AnthropicNativeBackend, BackendFormat, LlmTarget
from switchyard_rust.core import ChatRequestType


def _anthropic_target(**overrides: object) -> LlmTarget:
    data: dict[str, object] = {
        "id": "anthropic",
        "model": "claude-sonnet-test",
        "format": BackendFormat.ANTHROPIC,
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-ant-test",
        "timeout_secs": 12.5,
    }
    data.update(overrides)
    return LlmTarget(**data)


def test_constructs_from_resolved_anthropic_target() -> None:
    target = _anthropic_target()

    backend = AnthropicNativeBackend(target)

    assert backend.target == target
    assert backend.target.endpoint.base_url == "https://api.anthropic.com"
    assert backend.target.endpoint.api_key == "sk-ant-test"
    assert backend.target.endpoint.timeout_secs == 12.5


def test_supported_request_types_is_anthropic_only() -> None:
    backend = AnthropicNativeBackend(_anthropic_target())

    assert backend.supported_request_types == [ChatRequestType.ANTHROPIC]


@pytest.mark.parametrize(
    "target_format",
    [BackendFormat.OPENAI, BackendFormat.AUTO, "openai", "auto"],
)
def test_rejects_unresolved_or_non_anthropic_target_format(target_format: object) -> None:
    with pytest.raises(RuntimeError, match="resolved Anthropic format"):
        AnthropicNativeBackend(_anthropic_target(format=target_format))


def test_build_native_backend_selects_anthropic_binding() -> None:
    backend = build_native_backend(_anthropic_target())

    assert isinstance(backend, AnthropicNativeBackend)


def test_target_is_immutable_after_binding_construction() -> None:
    target = _anthropic_target()
    backend = AnthropicNativeBackend(target)

    with pytest.raises(AttributeError):
        backend.target.model = "mutated"
    assert backend.target.model == "claude-sonnet-test"


def test_request_types_are_value_objects_not_strings() -> None:
    request_type = AnthropicNativeBackend(_anthropic_target()).supported_request_types[0]

    assert request_type is ChatRequestType.ANTHROPIC
    assert request_type.value == "anthropic"
    assert not isinstance(request_type, str)
