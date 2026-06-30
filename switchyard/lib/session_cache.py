# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-backed bounded, access-ordered LRU cache keyed by a session key.

Pair with :func:`switchyard.lib.session_key.session_key_from_body` for the key.
"""

from __future__ import annotations

from switchyard_rust.core import SessionCache

__all__ = ["SessionCache"]
