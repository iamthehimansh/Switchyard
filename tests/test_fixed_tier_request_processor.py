# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`FixedTierRequestProcessor`."""

from __future__ import annotations

from typing import Any, cast

import pytest

from switchyard.lib.backends.deterministic_routing_llm_backend import (
    CTX_DETERMINISTIC_ROUTING_TIER,
)
from switchyard.lib.processors.fixed_tier_request_processor import (
    FixedTierRequestProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest


def _request() -> ChatRequest:
    return ChatRequest.openai_chat(
        cast(Any, {"model": "client-model", "messages": [{"role": "user", "content": "hi"}]}),
    )


async def test_fixed_tier_stamps_tier_on_ctx() -> None:
    proc = FixedTierRequestProcessor("strong")
    ctx = ProxyContext()

    await proc.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"


async def test_fixed_tier_weak() -> None:
    proc = FixedTierRequestProcessor("weak")
    ctx = ProxyContext()

    await proc.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "weak"


async def test_fixed_tier_request_passes_through_unchanged() -> None:
    """The processor must not mutate the request body — pure tier stamp."""
    proc = FixedTierRequestProcessor("strong")
    req = _request()
    body_before = dict(req.body)

    returned = await proc.process(ProxyContext(), req)

    assert returned is req
    assert dict(req.body) == body_before


async def test_fixed_tier_rejects_empty_label() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        FixedTierRequestProcessor("")


async def test_fixed_tier_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        FixedTierRequestProcessor(cast(Any, None))


async def test_fixed_tier_overwrites_existing_tier() -> None:
    """If an upstream processor stamped a tier, the fixed processor still wins.

    Force-mode is meant to override whatever else might be in the chain;
    this guards against an accidental compose with a real classifier.
    """
    proc = FixedTierRequestProcessor("strong")
    ctx = ProxyContext()
    ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] = "weak"

    await proc.process(ctx, _request())

    assert ctx.metadata[CTX_DETERMINISTIC_ROUTING_TIER] == "strong"
