# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from switchyard.cli.tui.launch_config_wizard import LaunchConfigWizard
from switchyard.cli.tui.model_selector import (
    ModelSelector,
    build_model_candidates,
)
from switchyard.cli.tui.terminal_capabilities import TerminalCapabilities
from switchyard.cli.tui.tui_session import TuiSession


def _plain_session(input_value: str) -> TuiSession:
    output: list[str] = []
    return TuiSession(
        capabilities=TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason="test",
        ),
        input_fn=lambda _prompt: input_value,
        secret_fn=lambda _prompt: input_value,
        output_fn=output.append,
    )


def test_build_model_candidates_puts_default_and_recommendations_first():
    candidates = build_model_candidates(
        preferred_model_ids=[
            "aws/anthropic/bedrock-claude-opus-4-7",
            "azure/anthropic/claude-sonnet-4-6",
        ],
        all_model_ids=[
            "nvidia/moonshotai/kimi-k2.5",
            "aws/anthropic/bedrock-claude-opus-4-7",
        ],
        default="aws/anthropic/bedrock-claude-opus-4-7",
    )

    assert [candidate.model_id for candidate in candidates] == [
        "aws/anthropic/bedrock-claude-opus-4-7",
        "azure/anthropic/claude-sonnet-4-6",
        "nvidia/moonshotai/kimi-k2.5",
    ]
    assert [candidate.group for candidate in candidates] == [
        "default",
        "recommended",
        "available",
    ]


def test_selector_filter_matches_model_id_and_group():
    selector = ModelSelector(
        title="Claude Code model",
        candidates=build_model_candidates(
            preferred_model_ids=["aws/anthropic/bedrock-claude-opus-4-7"],
            all_model_ids=["nvidia/moonshotai/kimi-k2.5"],
            default=None,
        ),
    )

    assert [
        candidate.model_id for candidate in selector.filter_candidates("opus")
    ] == ["aws/anthropic/bedrock-claude-opus-4-7"]
    assert [
        candidate.model_id for candidate in selector.filter_candidates("available")
    ] == ["nvidia/moonshotai/kimi-k2.5"]


def test_selector_moves_default_to_front_when_candidates_are_not_ordered():
    selector = ModelSelector(
        title="Claude reasoning effort",
        candidates=build_model_candidates(
            preferred_model_ids=["none", "disabled", "low", "medium", "high"],
            all_model_ids=[],
            default=None,
        ),
        default="high",
        allow_manual=False,
    )

    candidates = selector.filter_candidates("")
    assert [candidate.model_id for candidate in candidates] == [
        "high",
        "none",
        "disabled",
        "low",
        "medium",
    ]
    assert candidates[0].group == "default"


def test_plain_selector_accepts_blank_default():
    selector = ModelSelector(
        title="Codex model",
        candidates=build_model_candidates(
            preferred_model_ids=["openai/openai/openai/gpt-5.5"],
            all_model_ids=[],
            default="openai/openai/openai/gpt-5.5",
        ),
        default="openai/openai/openai/gpt-5.5",
    )

    assert selector.select(_plain_session("")) == "openai/openai/openai/gpt-5.5"


def test_plain_selector_accepts_manual_model_id():
    selector = ModelSelector(
        title="Codex model",
        candidates=[],
        default=None,
        allow_manual=True,
    )

    assert selector.select(_plain_session("custom/model")) == "custom/model"


def test_plain_selector_uses_custom_manual_entry_label():
    captured: dict[str, str] = {}
    session = TuiSession(
        capabilities=TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason="test",
        ),
        input_fn=lambda prompt: (captured.setdefault("prompt", prompt), "2")[1],
        secret_fn=lambda _prompt: "",
        output_fn=lambda _line: None,
    )
    selector = ModelSelector(
        title="Routing mode",
        candidates=build_model_candidates(
            preferred_model_ids=["No routing", "Random between two models"],
            all_model_ids=[],
            default="No routing",
        ),
        default="No routing",
        allow_manual=False,
        manual_entry_label="option",
    )

    assert selector.select(session) == "Random between two models"
    assert "(number or option)" in captured["prompt"]


def test_launch_wizard_uses_provider_neutral_default_api_key_label():
    captured: dict[str, str] = {}
    session = TuiSession(
        capabilities=TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason="test",
        ),
        input_fn=lambda _prompt: "",
        secret_fn=lambda prompt: (captured.setdefault("prompt", prompt), "sk-test")[1],
        output_fn=lambda _line: None,
    )

    assert LaunchConfigWizard(session).prompt_default_api_key(None) == "sk-test"
    assert "Default API key" in captured["prompt"]
    assert "NVIDIA" not in captured["prompt"]


def test_launch_wizard_shows_api_key_default_source_without_secret():
    captured: dict[str, str] = {}
    session = TuiSession(
        capabilities=TerminalCapabilities(
            interactive=False,
            color=False,
            tui=False,
            reason="test",
        ),
        input_fn=lambda _prompt: "",
        secret_fn=lambda prompt: (captured.setdefault("prompt", prompt), "")[1],
        output_fn=lambda _line: None,
    )

    assert (
        LaunchConfigWizard(session).prompt_default_api_key(
            "sk-env",
            default_source="$NVIDIA_API_KEY",
        )
        == "sk-env"
    )
    assert "Default API key (default: $NVIDIA_API_KEY)" in captured["prompt"]
    assert "sk-env" not in captured["prompt"]
