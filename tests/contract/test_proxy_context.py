# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProxyContext + metadata-key constant contract.

Platform reads ``CTX_TARGET_FORMAT`` and writes ``CTX_ORIGINAL_FORMAT`` into
``ProxyContext.metadata`` to bridge state across IGW request/response hooks
(see ``plugins/nemo-switchyard/src/nemo_switchyard/_processors.py`` and
``middleware.py`` in the Platform repo).

If the constant is renamed or its sentinel value changes, request- and
response-side translate stops sharing state — the user sees broken responses
at runtime with no obvious error.
"""

from __future__ import annotations

import pytest

from switchyard.lib.proxy_context import CTX_TARGET_FORMAT, ProxyContext


def test_ctx_target_format_is_a_stable_key() -> None:
    """``CTX_TARGET_FORMAT`` must be a hashable value usable as a dict key.

    Platform stores it verbatim under ``ctx.metadata[CTX_TARGET_FORMAT]`` and
    reads it back later. Any change here (string rename, enum migration) breaks
    cross-phase metadata bridging silently.
    """
    # Must be hashable (used as dict key)
    probe = {CTX_TARGET_FORMAT: "sentinel"}
    assert probe[CTX_TARGET_FORMAT] == "sentinel"
    # Must not be None (Platform branches on key-present-vs-missing)
    assert CTX_TARGET_FORMAT is not None


def test_proxy_context_metadata_round_trip() -> None:
    """ProxyContext.metadata must accept arbitrary keys + survive round-trip.

    Platform's _bridge.py copies metadata between an IGW context and a
    switchyard ProxyContext on each phase. A mapping that drops unknown keys or
    enforces a typed schema breaks Platform.
    """
    ctx = ProxyContext()
    ctx.metadata[CTX_TARGET_FORMAT] = "openai_chat"
    ctx.metadata["arbitrary_string_key"] = {"nested": "value"}
    assert ctx.metadata[CTX_TARGET_FORMAT] == "openai_chat"
    assert ctx.metadata["arbitrary_string_key"] == {"nested": "value"}


def test_proxy_context_is_constructible_with_no_args() -> None:
    """Platform's bridge constructs ``ProxyContext()`` with no positional args
    when building a side-pipeline. Required positional/kwargs is a breaking
    change."""
    ctx = ProxyContext()
    assert ctx is not None
    assert hasattr(ctx, "metadata"), "ProxyContext must expose `.metadata` mapping"


@pytest.mark.parametrize(
    "attr",
    ["metadata"],
)
def test_proxy_context_exposes_required_attrs(attr: str) -> None:
    """Attribute presence — exercised independently of constructor args."""
    ctx = ProxyContext()
    assert hasattr(ctx, attr), f"ProxyContext.{attr} is required by Platform but is missing"
