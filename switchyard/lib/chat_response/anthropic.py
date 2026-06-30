# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages API stream adapter re-export."""

from typing import TypeAlias

from switchyard_rust.core import ChatResponse as _ChatResponse
from switchyard_rust.core import ChatResponseStream as _ChatResponseStream

AnthropicChatResponse: TypeAlias = _ChatResponse
AnthropicStreamingChatResponse: TypeAlias = _ChatResponse
AnthropicResponseStream: TypeAlias = _ChatResponseStream

__all__ = [
    "AnthropicChatResponse",
    "AnthropicResponseStream",
    "AnthropicStreamingChatResponse",
]
