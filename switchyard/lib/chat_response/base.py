# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-export for Rust-owned response values."""

from typing import TypeAlias

from switchyard_rust.core import ChatResponse as _ChatResponse
from switchyard_rust.core import ChatResponseType as _ChatResponseType

ChatResponse: TypeAlias = _ChatResponse
ChatResponseType: TypeAlias = _ChatResponseType

__all__ = ["ChatResponse", "ChatResponseType"]
