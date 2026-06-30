# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request processor that pins every inbound request to one backend model."""

from __future__ import annotations

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest


class ModelRewriteRequestProcessor:
    """Force-rewrite ``request.body["model"]`` to a fixed value.

    All request subclasses expose a top-level ``model`` key in their provider
    body. Launchers use this as a safety net so the child process can display a
    model while Switchyard remains authoritative about the upstream route.
    """

    def __init__(self, model: str) -> None:
        self._model = model

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        del ctx
        request.set_model(self._model)
        return request
