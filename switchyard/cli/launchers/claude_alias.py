# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code `/model` picker alias rule.

Claude Code's gateway-discovery filter only adds models to its `/model`
picker when the id starts with ``claude`` or ``anthropic``. The claude
launcher exposes every table id under both spellings so users can
pick (or save) either form. Centralized here so the rule lives in
exactly one place — see ``claude_code_launcher._with_claude_aliases``,
``launch_command._alias_variants``, and ``configure_command``'s
``default_claude_route_model``.
"""

from __future__ import annotations

#: Prefixes Claude Code's gateway-discovery filter accepts.
CLAUDE_PICKER_PREFIXES = ("claude", "anthropic")


def claude_alias_for(model_id: str) -> str | None:
    """Return ``claude-<model_id>`` when *model_id* isn't already prefixed."""
    if model_id.startswith(CLAUDE_PICKER_PREFIXES):
        return None
    return f"claude-{model_id}"


def de_claude_alias(model_id: str) -> str | None:
    """Return *model_id* with a leading ``claude-`` / ``anthropic-`` stripped.

    Returns ``None`` when *model_id* isn't prefixed (no de-alias to add).
    """
    for prefix in ("claude-", "anthropic-"):
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return None


__all__ = [
    "CLAUDE_PICKER_PREFIXES",
    "claude_alias_for",
    "de_claude_alias",
]
