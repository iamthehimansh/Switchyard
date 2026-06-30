# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal capability detection for optional interactive TUI flows."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

NO_TUI_ENV_VAR = "SWITCHYARD_NO_TUI"


class SupportsIsAtty(Protocol):
    def isatty(self) -> bool:
        ...


@dataclass(frozen=True)
class TerminalCapabilities:
    """Small, testable verdict for interactive terminal features."""

    interactive: bool
    color: bool
    tui: bool
    reason: str | None = None


def _stream_isatty(stream: SupportsIsAtty) -> bool:
    try:
        return stream.isatty()
    except OSError:
        return False


def detect_terminal_capabilities(
    *,
    force_plain: bool = False,
    stdin: SupportsIsAtty | None = None,
    stdout: SupportsIsAtty | None = None,
    environ: Mapping[str, str] | None = None,
) -> TerminalCapabilities:
    """Return whether the current process can use rich terminal prompts."""

    env = environ or os.environ
    if force_plain:
        return TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason="plain mode requested",
        )
    if env.get(NO_TUI_ENV_VAR):
        return TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason=f"{NO_TUI_ENV_VAR} is set",
        )

    effective_stdin = stdin or sys.stdin
    effective_stdout = stdout or sys.stdout
    interactive = _stream_isatty(effective_stdin) and _stream_isatty(effective_stdout)
    if not interactive:
        return TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason="stdin/stdout are not both terminals",
        )

    term = env.get("TERM", "")
    if term == "dumb":
        return TerminalCapabilities(
            interactive=True,
            color=False,
            tui=False,
            reason="TERM=dumb",
        )

    return TerminalCapabilities(
        interactive=True,
        color="NO_COLOR" not in env,
        tui=True,
    )
