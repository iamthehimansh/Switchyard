# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible model listing payload helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

SUPPORTED_INBOUND_FORMATS: tuple[str, ...] = (
    "openai-chat-completions",
    "openai-responses",
    "anthropic-messages",
)

DEFAULT_CONTEXT_WINDOW = 128_000

_CONTEXT_WINDOW_BY_MODEL_FRAGMENT: tuple[tuple[str, int], ...] = (
    ("nemotron-3-super", 1_000_000),
    ("nemotron-3-nano", 262_000),
    ("deepseek-v4", 1_000_000),
    ("claude", 200_000),
    ("kimi-k2", 256_000),
)


def _default_capabilities() -> dict[str, Any]:
    return {
        "streaming": True,
        "tool_calling": True,
        "context_window": DEFAULT_CONTEXT_WINDOW,
        "supported_inbound_formats": list(SUPPORTED_INBOUND_FORMATS),
    }


def model_capabilities(
    model_id: str,
    *,
    context_window: int | None = None,
    tool_calling: bool = True,
) -> dict[str, Any]:
    """Infer capability metadata for a Switchyard-advertised model id."""
    return {
        "tool_calling": tool_calling,
        "context_window": (
            context_window if context_window is not None else inferred_context_window(model_id)
        ),
    }


def combined_model_capabilities(model_ids: Sequence[str]) -> dict[str, Any]:
    """Return conservative capabilities for a route spanning multiple models."""
    windows = [inferred_context_window(model_id) for model_id in model_ids]
    return {
        "tool_calling": True,
        "context_window": min(windows) if windows else DEFAULT_CONTEXT_WINDOW,
    }


def inferred_context_window(model_id: str) -> int:
    """Infer a context window from known model-family fragments."""
    normalized = model_id.lower()
    for fragment, context_window in _CONTEXT_WINDOW_BY_MODEL_FRAGMENT:
        if fragment in normalized:
            return context_window
    return DEFAULT_CONTEXT_WINDOW


def model_entry(
    model_id: str,
    display_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI-compatible model object with Switchyard metadata."""
    extra = dict(metadata or {})
    display_name = display_name or str(extra.pop("display_name", model_id))
    extra.pop("id", None)
    extra.pop("object", None)
    extra.pop("type", None)

    capabilities = _default_capabilities()
    raw_capabilities = extra.pop("capabilities", None)
    if isinstance(raw_capabilities, Mapping):
        capabilities.update(raw_capabilities)

    entry: dict[str, Any] = {
        "id": model_id,
        "object": "model",
        "type": "model",
        "created": extra.pop("created", 0),
        "owned_by": extra.pop("owned_by", "switchyard"),
        "display_name": display_name,
        "capabilities": capabilities,
    }
    entry.update(extra)
    return entry


def model_list_payload(
    entries: Sequence[Mapping[str, Any]],
    default_model: str | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the ``GET /v1/models`` response envelope."""
    data = [dict(entry) for entry in entries]
    model_ids = [str(entry["id"]) for entry in data if "id" in entry]
    advertised_default = model_ids[0] if model_ids else None
    if default_model in model_ids:
        advertised_default = default_model
    payload: dict[str, Any] = {
        "object": "list",
        "data": data,
        "first_id": model_ids[0] if model_ids else None,
        "last_id": model_ids[-1] if model_ids else None,
        "has_more": False,
        "default_model": advertised_default,
        "model_pool": model_ids,
    }
    if warnings:
        payload["warnings"] = list(warnings)
    return payload


__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "SUPPORTED_INBOUND_FORMATS",
    "combined_model_capabilities",
    "inferred_context_window",
    "model_capabilities",
    "model_entry",
    "model_list_payload",
]
