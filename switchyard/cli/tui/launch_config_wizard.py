# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive launcher configuration wizard."""

from __future__ import annotations

from dataclasses import dataclass

from switchyard.cli.tui.choice_selector import (
    ChoiceOption,
    ChoiceSelector,
)
from switchyard.cli.tui.model_selector import (
    ModelSelector,
    build_model_candidates,
)
from switchyard.cli.tui.tui_session import TuiSession


@dataclass(frozen=True)
class LaunchConfigWizard:
    """Prompt orchestration for provider and per-launcher defaults."""

    session: TuiSession

    def start(self, *, target: str) -> None:
        self.session.print_header(
            "Switchyard setup",
            f"Configuring {target} defaults. Existing values are preselected.",
        )

    def prompt_default_base_url(self, default: str) -> str:
        return self.session.prompt_text("Default base URL", default)

    def prompt_default_api_key(
        self,
        existing_api_key: str | None,
        *,
        default_source: str | None = None,
    ) -> str:
        label = "Default API key"
        if existing_api_key and default_source:
            label = f"{label} (default: {default_source})"
        return self.session.prompt_secret(label, existing_api_key)

    def select_endpoint_mode(self, label: str, *, default: str) -> str:
        return ChoiceSelector(
            title=f"{label} endpoint",
            options=[
                ChoiceOption(value="default", label="Use default endpoint"),
                ChoiceOption(value="custom", label="Customize endpoint"),
            ],
            default=default,
        ).select(self.session)

    def prompt_endpoint_base_url(self, label: str, default: str) -> str:
        return self.session.prompt_text(f"{label} base URL", default)

    def prompt_endpoint_api_key(
        self, label: str, existing_api_key: str | None,
    ) -> str:
        return self.session.prompt_secret(f"{label} API key", existing_api_key)

    def select_model(
        self,
        label: str,
        *,
        preferred_model_ids: list[str],
        all_model_ids: list[str],
        default: str | None,
    ) -> str:
        return ModelSelector(
            title=f"{label} model",
            candidates=build_model_candidates(
                preferred_model_ids=preferred_model_ids,
                all_model_ids=all_model_ids,
                default=default,
            ),
            default=default,
            allow_manual=True,
            prompt_label=f"{label} model",
            manual_entry_label="model id",
            cycle_label_plural="models",
        ).select(self.session)

    def prompt_routing_profiles(self, *, default: str | None) -> str | None:
        """Prompt for an optional routing-profile YAML path.

        Used by ``switchyard serve`` and as a fallback for
        ``switchyard launch claude/codex``. Returns ``None`` for an empty
        response or when the user types ``-`` to clear, so the saved
        path is removed rather than persisted as an empty string.
        """
        prompt_label = (
            "Routing-profiles YAML path "
            "(press enter to skip; type '-' to clear)"
        )
        value = self.session.prompt_text(prompt_label, default or "")
        cleaned = value.strip()
        if cleaned == "-" or not cleaned:
            return None
        return cleaned
