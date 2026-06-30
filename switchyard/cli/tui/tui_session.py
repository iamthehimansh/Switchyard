# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable prompt session for Switchyard CLI TUI screens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from switchyard.cli.tui.terminal_capabilities import (
    TerminalCapabilities,
    detect_terminal_capabilities,
)

InputFn = Callable[[str], str]
SecretFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def _prompt_toolkit_available() -> bool:
    try:
        import prompt_toolkit  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class TuiSession:
    """Terminal prompt session shared by current and future CLI screens."""

    capabilities: TerminalCapabilities
    input_fn: InputFn
    secret_fn: SecretFn
    output_fn: OutputFn

    @classmethod
    def from_current_terminal(
        cls,
        *,
        force_plain: bool,
        input_fn: InputFn,
        secret_fn: SecretFn,
        output_fn: OutputFn,
    ) -> TuiSession:
        return cls(
            capabilities=detect_terminal_capabilities(force_plain=force_plain),
            input_fn=input_fn,
            secret_fn=secret_fn,
            output_fn=output_fn,
        )

    @property
    def enabled(self) -> bool:
        """Return whether prompt-toolkit backed TUI prompts should be used."""

        return self.capabilities.tui and _prompt_toolkit_available()

    def print_header(self, title: str, subtitle: str | None = None) -> None:
        if self.enabled and self.capabilities.color:
            self.output_fn(f"\033[1;32m{title}\033[0m")
        else:
            self.output_fn(title)
        if subtitle:
            self.output_fn(subtitle)

    def prompt_text(self, label: str, default: str | None = None) -> str:
        if self.enabled:
            return self._prompt_toolkit_text(label, default=default, is_password=False)

        suffix = f" [{default}]" if default else ""
        value = self.input_fn(f"{label}{suffix}: ").strip()
        return value or default or ""

    def prompt_secret(self, label: str, default: str | None = None) -> str:
        if self.enabled:
            return self._prompt_toolkit_text(label, default=default, is_password=True)

        suffix = " [press enter to keep existing]" if default else ""
        value = self.secret_fn(f"{label}{suffix}: ").strip()
        return value or default or ""

    def _prompt_toolkit_text(
        self,
        label: str,
        *,
        default: str | None,
        is_password: bool,
    ) -> str:
        from prompt_toolkit import prompt  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.styles import Style  # pyright: ignore[reportMissingImports]

        prompt_style = Style.from_dict({
            "prompt": "bold #76b900",
            "bottom-toolbar": "#888888",
        })
        try:
            value = str(
                prompt(
                    f"{label}: ",
                    default=default or "",
                    is_password=is_password,
                    style=prompt_style,
                    bottom_toolbar="Enter to accept the shown default.",
                )
            ).strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise SystemExit("Setup canceled.") from exc

        return value or default or ""
