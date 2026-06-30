# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-export for Rust-owned request values."""

from typing import TypeAlias

from switchyard_rust.core import ChatRequest as _ChatRequest
from switchyard_rust.core import ChatRequestType as _ChatRequestType

ChatRequest: TypeAlias = _ChatRequest
ChatRequestType: TypeAlias = _ChatRequestType

__all__ = ["ChatRequest", "ChatRequestType"]
