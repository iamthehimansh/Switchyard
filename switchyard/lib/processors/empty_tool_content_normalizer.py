# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replace empty message content that NIH's Nemotron Nano adapter rejects.

NVIDIA Inference Hub's LiteLLM adapter for ``Nemotron-3-Nano-30B-A3B``
rejects any message whose ``content`` is empty (or whitespace-only)
with HTTP 400 ``message content cannot be empty``. Direct probes on
2026-05-15 mapped exactly which roles trip the guard:

* ``role: "tool"`` with ``content == ""`` or ``None`` — rejected.
* ``role: "assistant"`` with ``content == ""`` or ``None`` **and no
  ``tool_calls``** — rejected.  (Assistant + ``tool_calls`` + empty
  content is accepted, matching the OpenAI Chat Completions spec.)
* ``role: "system" / "user"`` with empty/null content — accepted (NIH
  doesn't enforce there).
* Whitespace-only strings (``" "``, ``"\\n"``) are *also* rejected —
  the adapter strips before checking.  Placeholder must contain a
  non-whitespace character.

Harbor + terminus-2 routinely produces both shapes (empty shell
output, retried assistant turns with no content).  This processor
normalises them to ``"(no output)"`` so the upstream never sees an
empty payload.
"""

from __future__ import annotations

import logging

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.core import ChatRequest

log = logging.getLogger(__name__)

_PLACEHOLDER = "(no output)"


def _is_empty(content: object) -> bool:
    """True iff ``content`` is None, empty string, or whitespace-only string."""
    if content is None:
        return True
    if isinstance(content, str) and content.strip() == "":
        return True
    return False


class EmptyToolContentNormalizer:
    """Replace empty content on ``tool`` and bare-``assistant`` messages.

    Mutates only the shapes NIH's Nemotron adapter rejects (see module
    docstring); leaves everything else untouched.
    """

    async def process(self, ctx: ProxyContext, request: ChatRequest) -> ChatRequest:  # noqa: ARG002
        body = request.body
        if not isinstance(body, dict):
            return request
        messages = body.get("messages")
        if not isinstance(messages, list):
            return request

        mutated = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "tool":
                if _is_empty(msg.get("content")):
                    msg["content"] = _PLACEHOLDER
                    mutated = True
            elif role == "assistant":
                # Assistant with tool_calls is allowed to have empty
                # content per the OpenAI spec — and the Nano adapter
                # accepts it.  Only normalise the bare case.
                if not msg.get("tool_calls") and _is_empty(msg.get("content")):
                    msg["content"] = _PLACEHOLDER
                    mutated = True

        if mutated:
            request.replace_body(body)
        return request


__all__ = ["EmptyToolContentNormalizer"]
