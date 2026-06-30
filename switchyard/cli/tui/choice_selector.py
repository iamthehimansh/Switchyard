# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small reusable selector for non-model CLI choices."""

from __future__ import annotations

from dataclasses import dataclass

from switchyard.cli.tui.model_selector import (
    ModelCandidate,
    ModelSelector,
)
from switchyard.cli.tui.tui_session import TuiSession


@dataclass(frozen=True)
class ChoiceOption:
    """One selectable choice in an interactive prompt."""

    value: str
    label: str


@dataclass(frozen=True)
class ChoiceSelector:
    """Selector for a short list of named choices."""

    title: str
    options: list[ChoiceOption]
    default: str

    def select(self, session: TuiSession) -> str:
        label_to_value = {option.label: option.value for option in self.options}
        value_to_label = {option.value: option.label for option in self.options}
        default_label = value_to_label.get(self.default, self.options[0].label)
        selected_label = ModelSelector(
            title=self.title,
            candidates=[
                ModelCandidate(
                    model_id=option.label,
                    group="default" if option.value == self.default else "available",
                )
                for option in self.options
            ],
            default=default_label,
            allow_manual=False,
            prompt_label=self.title.lower(),
            manual_entry_label="option",
            cycle_label_plural="options",
        ).select(session)
        return label_to_value[selected_label]


__all__ = ["ChoiceOption", "ChoiceSelector"]
