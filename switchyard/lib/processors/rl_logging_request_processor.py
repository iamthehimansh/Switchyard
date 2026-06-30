# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-side processor that snapshots the inbound request for RL trace logging."""

from __future__ import annotations

from copy import deepcopy

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest, ChatRequestType
from switchyard_rust.translation import TranslationEngine

#: Context metadata key holding the OpenAI-Chat-shaped snapshot of the inbound
#: request body. Written by :class:`RlLoggingRequestProcessor`; read by
#: :class:`~switchyard.lib.processors.rl_logging_response_processor.RlLoggingResponseProcessor`
#: to reconstruct the logged conversation.
CTX_RL_LOGGING_REQUEST = "_rl_logging_request"


class RlLoggingRequestProcessor:
    """Snapshot the inbound request as an OpenAI-Chat body for RL trace logging.

    The response-side logger only receives ``(ctx, response)``, so the request
    has to be captured on the request side. We store the *translated* body (a
    plain dict) rather than the request wrapper so the snapshot is both
    format-normalized (every inbound format becomes OpenAI Chat) and immune to
    any in-place mutation later processors might perform.
    """

    def __init__(self) -> None:
        self._translation = TranslationEngine()

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:
        """Snapshot the OpenAI-Chat-translated request body onto ``ctx``.

        Writes a deep copy under :data:`CTX_RL_LOGGING_REQUEST` (so the snapshot
        stays stable even if later processors mutate the request) and returns
        the request unchanged.
        """
        openai_request = self._translation.request_to(ChatRequestType.OPENAI_CHAT, request)
        ctx.metadata[CTX_RL_LOGGING_REQUEST] = deepcopy(dict(openai_request.body))
        return request
