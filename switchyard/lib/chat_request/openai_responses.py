# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses API request compatibility alias."""

from typing import TypeAlias

from switchyard_rust.core import ChatRequest as _ChatRequest

ResponsesChatRequest: TypeAlias = _ChatRequest

__all__ = ["ResponsesChatRequest"]
