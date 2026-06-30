# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-surface contract: every symbol the NeMo Platform `nemo-switchyard`
plugin imports from switchyard must resolve.

A failure here means a downstream IGW middleware plugin will hit
``ModuleNotFoundError`` or ``ImportError`` at process startup. The single
source of truth for this list is the platform plugin source tree at
``plugins/nemo-switchyard/src/nemo_switchyard/`` in the NVIDIA-NeMo/Platform
repo — keep these two in lockstep when adding or removing imports there.
"""

from __future__ import annotations

import importlib

import pytest

# (module path, [attribute names that must exist on the module])
#
# Listed by source file in the downstream plugin so reviewers can grep the
# Platform side easily:
#   _format.py    — chat_request.{anthropic,base,openai_chat,openai_responses}
#   _bridge.py    — chat_request.base, chat_response.*, proxy_context.ProxyContext
#   _processors.py — proxy_context.CTX_TARGET_FORMAT
#   middleware.py — proxy_context
PLATFORM_IMPORT_SURFACE: list[tuple[str, list[str]]] = [
    ("switchyard.lib.chat_request.anthropic", ["AnthropicChatRequest"]),
    ("switchyard.lib.chat_request.base", ["ChatRequest"]),
    ("switchyard.lib.chat_request.openai_chat", ["OpenAIChatRequest"]),
    ("switchyard.lib.chat_request.openai_responses", ["ResponsesChatRequest"]),
    (
        "switchyard.lib.chat_response.anthropic",
        ["AnthropicChatResponse", "AnthropicStreamingChatResponse", "AnthropicResponseStream"],
    ),
    ("switchyard.lib.chat_response.base", ["ChatResponse"]),
    (
        "switchyard.lib.chat_response.openai_chat",
        ["CompletionChatResponse", "StreamingChatResponse", "ResponseStream"],
    ),
    (
        "switchyard.lib.chat_response.openai_responses",
        ["ResponsesApiChatResponse", "ResponsesApiStreamingChatResponse", "ResponsesApiStream"],
    ),
    ("switchyard.lib.proxy_context", ["CTX_TARGET_FORMAT", "ProxyContext"]),
    ("switchyard.lib.processors.format_translate", ["TranslateConfig"]),
]


@pytest.mark.parametrize(
    ("module_path", "attr"),
    [(mod, attr) for mod, attrs in PLATFORM_IMPORT_SURFACE for attr in attrs],
)
def test_platform_import_resolves(module_path: str, attr: str) -> None:
    """Each symbol the Platform plugin imports must resolve from upstream switchyard.

    Failure mode: a switchyard PR that deletes ``module_path`` or renames
    ``attr`` ships a broken contract. Downstream IGW startup will fail with
    ``ModuleNotFoundError`` or ``ImportError`` at the plugin entry-point load
    step.
    """
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), (
        f"Platform's nemo-switchyard plugin imports `{attr}` from `{module_path}`, "
        f"but it is no longer exported. Coordinate the migration with the Platform team "
        f"before merging."
    )
