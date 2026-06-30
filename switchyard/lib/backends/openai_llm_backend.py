# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import path for the Rust-owned OpenAI passthrough backend."""

from switchyard_rust.components import OpenAiPassthroughBackend

__all__ = ["OpenAiPassthroughBackend"]
