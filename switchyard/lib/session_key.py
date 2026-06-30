# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-backed stable per-conversation key.

Derived from the prefix an agent harness never rewrites — the system prompt
plus the first user message — so every turn of one conversation hashes alike.
"""

from __future__ import annotations

from switchyard_rust.core import session_key_from_body

__all__ = ["session_key_from_body"]
