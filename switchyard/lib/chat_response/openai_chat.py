# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Chat Completions stream adapter re-export."""

from typing import TypeAlias

from switchyard_rust.core import ChatResponse as _ChatResponse
from switchyard_rust.core import ChatResponseStream as _ChatResponseStream

CompletionChatResponse: TypeAlias = _ChatResponse
StreamingChatResponse: TypeAlias = _ChatResponse
ResponseStream: TypeAlias = _ChatResponseStream

__all__ = ["CompletionChatResponse", "ResponseStream", "StreamingChatResponse"]
