# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Searchable model selector used by launcher configuration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from switchyard.cli.tui.tui_session import TuiSession

CandidateGroup = Literal["default", "recommended", "available"]


@dataclass(frozen=True)
class ModelCandidate:
    """One selectable model row."""

    model_id: str
    group: CandidateGroup

    @property
    def display_group(self) -> str:
        if self.group == "default":
            return "current default"
        if self.group == "recommended":
            return "recommended"
        return "available"


def _append_unique(
    output: list[ModelCandidate],
    seen: set[str],
    model_ids: Iterable[str],
    group: CandidateGroup,
) -> None:
    for model_id in model_ids:
        if model_id in seen:
            continue
        output.append(ModelCandidate(model_id=model_id, group=group))
        seen.add(model_id)


def build_model_candidates(
    *,
    preferred_model_ids: Iterable[str],
    all_model_ids: Iterable[str],
    default: str | None,
) -> list[ModelCandidate]:
    """Return a selector list with defaults first, then ranked suggestions."""

    candidates: list[ModelCandidate] = []
    seen: set[str] = set()
    if default:
        _append_unique(candidates, seen, (default,), "default")
    _append_unique(candidates, seen, preferred_model_ids, "recommended")
    _append_unique(candidates, seen, all_model_ids, "available")
    return candidates


@dataclass(frozen=True)
class ModelSelector:
    """Model chooser with prompt-toolkit TUI and plain-terminal fallback."""

    title: str
    candidates: list[ModelCandidate]
    default: str | None = None
    allow_manual: bool = True
    prompt_label: str = "model"
    manual_entry_label: str = "model id"
    cycle_label_plural: str = "models"

    def _ordered_candidates(self) -> list[ModelCandidate]:
        """Return candidates with the default value first when present."""

        ordered: list[ModelCandidate] = []
        seen: set[str] = set()
        if self.default:
            ordered.append(ModelCandidate(model_id=self.default, group="default"))
            seen.add(self.default)
        for candidate in self.candidates:
            if candidate.model_id in seen:
                continue
            ordered.append(candidate)
            seen.add(candidate.model_id)
        return ordered

    def filter_candidates(self, query: str) -> list[ModelCandidate]:
        needle = query.strip().lower()
        candidates = self._ordered_candidates()
        if not needle:
            return candidates
        return [
            candidate
            for candidate in candidates
            if needle in candidate.model_id.lower()
            or needle in candidate.display_group.lower()
        ]

    def select(self, session: TuiSession) -> str:
        if session.enabled:
            return self._select_with_prompt_toolkit(session)
        return self._select_plain(session)

    def _select_plain(self, session: TuiSession) -> str:
        candidates = self._ordered_candidates()
        visible = candidates[:20]
        if visible:
            session.output_fn("")
            session.output_fn(f"{self.title} candidates:")
            for index, candidate in enumerate(visible, start=1):
                marker = " *" if candidate.model_id == self.default else ""
                session.output_fn(
                    f"  {index}. {candidate.model_id} ({candidate.display_group}){marker}"
                )
            if len(candidates) > len(visible):
                session.output_fn(
                    f"  ... {len(candidates) - len(visible)} more. "
                    f"Type a {self.manual_entry_label} to use one not shown."
                )

        suffix = f" [{self.default}]" if self.default else ""
        value = str(
            session.input_fn(
                f"{self.title}{suffix} (number or {self.manual_entry_label}): "
            )
        ).strip()
        if not value:
            if self.default:
                return self.default
            raise SystemExit(f"{self.title} is required.")
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(visible):
                return visible[index - 1].model_id
            raise SystemExit(f"{self.title} choice {value} is out of range.")
        if not self.allow_manual and value not in {candidate.model_id for candidate in candidates}:
            raise SystemExit(
                f"{self.title} must be one of: "
                f"{', '.join(candidate.model_id for candidate in candidates)}"
            )
        return value

    def _select_with_prompt_toolkit(self, session: TuiSession) -> str:
        from prompt_toolkit import prompt  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.application.current import (  # pyright: ignore[reportMissingImports]
            get_app,
        )
        from prompt_toolkit.completion import (  # pyright: ignore[reportMissingImports]
            FuzzyWordCompleter,
        )
        from prompt_toolkit.formatted_text import HTML  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.key_binding import KeyBindings  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.styles import Style  # pyright: ignore[reportMissingImports]

        style = Style.from_dict({
            "prompt": "bold #76b900",
            "completion-menu.completion": "bg:#1f2937 #d1d5db",
            "completion-menu.completion.current": "bg:#76b900 #111827",
            "completion-menu.meta.completion": "bg:#111827 #9ca3af",
            "bottom-toolbar": "#888888",
        })
        candidates = self._ordered_candidates()

        session.output_fn("")
        session.output_fn(f"{self.title}")
        if candidates:
            session.output_fn(
                "Type to filter, use arrow keys to move, enter to select. "
                f"You can paste a {self.manual_entry_label} directly."
            )
        else:
            session.output_fn(
                f"No catalog was discovered. Enter a {self.manual_entry_label}."
            )

        meta_by_model = {
            candidate.model_id: candidate.display_group
            for candidate in candidates
        }
        key_bindings = KeyBindings()

        def _cycle_next(event: object) -> None:
            buffer = get_app().current_buffer
            if buffer.complete_state:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        def _cycle_previous(event: object) -> None:
            buffer = get_app().current_buffer
            if buffer.complete_state:
                buffer.complete_previous()
            else:
                buffer.start_completion(select_last=True)

        def _accept_selection(event: object) -> None:
            buffer = get_app().current_buffer
            complete_state = buffer.complete_state
            if complete_state and complete_state.current_completion:
                buffer.apply_completion(complete_state.current_completion)
            buffer.validate_and_handle()

        key_bindings.add("down")(_cycle_next)
        key_bindings.add("up")(_cycle_previous)
        key_bindings.add("enter")(_accept_selection)

        def _open_completion_menu() -> None:
            if candidates:
                get_app().current_buffer.start_completion(select_first=True)

        toolbar = (
            f"Up/down cycles {self.cycle_label_plural}. Enter selects. "
            "Type to filter. Ctrl-C cancels."
        )
        if self.default:
            toolbar = f"Default: {self.default} | {toolbar}"

        try:
            selected = str(
                prompt(
                    HTML(f"<ansigreen><b>{self.prompt_label}</b></ansigreen> "),
                    default="",
                    accept_default=False,
                    completer=FuzzyWordCompleter(
                        [candidate.model_id for candidate in candidates],
                        meta_dict=meta_by_model,
                    ),
                    complete_while_typing=True,
                    key_bindings=key_bindings,
                    pre_run=_open_completion_menu,
                    reserve_space_for_menu=12,
                    style=style,
                    bottom_toolbar=toolbar,
                )
            ).strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise SystemExit("Setup canceled.") from exc

        if not selected:
            if self.default:
                return self.default
            raise SystemExit(f"{self.title} is required.")
        if not self.allow_manual and selected not in meta_by_model:
            raise SystemExit(
                f"{self.title} must be one of: "
                f"{', '.join(candidate.model_id for candidate in candidates)}"
            )
        return selected
