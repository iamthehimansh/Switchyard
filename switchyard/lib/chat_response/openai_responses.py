# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses API stream adapter re-export."""

from typing import TypeAlias

from switchyard_rust.core import ChatResponse as _ChatResponse
from switchyard_rust.core import ChatResponseStream as _ChatResponseStream

ResponsesApiChatResponse: TypeAlias = _ChatResponse
ResponsesApiStreamingChatResponse: TypeAlias = _ChatResponse
ResponsesApiStream: TypeAlias = _ChatResponseStream

__all__ = [
    "ResponsesApiChatResponse",
    "ResponsesApiStream",
    "ResponsesApiStreamingChatResponse",
]
