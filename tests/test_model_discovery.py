# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from switchyard.cli.model_catalog.model_discovery import (
    choose_default_claude_model,
    choose_default_codex_model,
    claude_model_candidates,
    codex_model_candidates,
)


def test_choose_default_claude_model_prefers_opus_47():
    models = [
        "azure/anthropic/claude-sonnet-4-6",
        "aws/anthropic/bedrock-claude-opus-4-6",
        "aws/anthropic/bedrock-claude-opus-4-7",
        "nvidia/moonshotai/kimi-k2.6",
    ]

    assert (
        choose_default_claude_model(models)
        == "aws/anthropic/bedrock-claude-opus-4-7"
    )
    assert claude_model_candidates(models)[0] == "aws/anthropic/bedrock-claude-opus-4-7"


def test_choose_default_claude_model_prefers_newer_opus_without_ceiling():
    models = [
        "azure/anthropic/claude-sonnet-5-0",
        "aws/anthropic/bedrock-claude-opus-4-7",
        "aws/anthropic/bedrock-claude-opus-4-8",
    ]

    assert (
        choose_default_claude_model(models)
        == "aws/anthropic/bedrock-claude-opus-4-8"
    )
    assert claude_model_candidates(models)[0] == "aws/anthropic/bedrock-claude-opus-4-8"


def test_choose_default_claude_model_prefers_stable_over_future_preview():
    models = [
        "anthropic/claude-opus-5-0-preview",
        "aws/anthropic/bedrock-claude-opus-4-7",
    ]

    assert (
        choose_default_claude_model(models)
        == "aws/anthropic/bedrock-claude-opus-4-7"
    )


def test_choose_default_claude_model_ignores_date_suffix_as_model_version():
    models = [
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-sonnet-3-7",
    ]

    assert (
        choose_default_claude_model(models)
        == "anthropic/claude-sonnet-3-7"
    )


def test_choose_default_claude_model_handles_legacy_version_before_family():
    models = [
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-7-sonnet-20250219",
    ]

    assert (
        choose_default_claude_model(models)
        == "anthropic/claude-3-7-sonnet-20250219"
    )


def test_choose_default_claude_model_handles_vertex_snapshot_separator():
    models = [
        "anthropic/claude-sonnet-4-5@20251001",
        "anthropic/claude-sonnet-4-6",
    ]

    assert choose_default_claude_model(models) == "anthropic/claude-sonnet-4-6"


def test_choose_default_claude_model_falls_back_to_sonnet():
    models = [
        "nvidia/moonshotai/kimi-k2.6",
        "azure/anthropic/claude-sonnet-4-6",
    ]

    assert choose_default_claude_model(models) == "azure/anthropic/claude-sonnet-4-6"


def test_choose_default_claude_model_returns_none_without_candidates():
    models = [
        "nvidia/moonshotai/kimi-k2.6",
        "openai/openai/openai/gpt-5.5",
    ]

    assert choose_default_claude_model(models) is None
    assert claude_model_candidates(models) == []


def test_choose_default_codex_model_prefers_gpt_55():
    models = [
        "openai/openai/openai/gpt-5.2",
        "openai/openai/openai/gpt-5.5",
        "openai/openai/openai/gpt-4.1",
        "aws/anthropic/bedrock-claude-opus-4-7",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.5"
    assert codex_model_candidates(models)[0] == "openai/openai/openai/gpt-5.5"


def test_choose_default_codex_model_prefers_future_gpt_without_ceiling():
    models = [
        "openai/openai/openai/gpt-5.5",
        "openai/openai/openai/gpt-5.6",
        "openai/openai/openai/gpt-5.4-codex",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.6"
    assert codex_model_candidates(models)[0] == "openai/openai/openai/gpt-5.6"


def test_choose_default_codex_model_prefers_stable_over_preview():
    models = [
        "openai/openai/openai/gpt-5.5",
        "openai/openai/openai/gpt-6.0-preview",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.5"


def test_choose_default_codex_model_uses_exact_gpt_segment():
    models = [
        "notopenai/vendor/egpt-9",
        "openai/openai/openai/gpt-5.5",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.5"
    assert codex_model_candidates(models) == ["openai/openai/openai/gpt-5.5"]


def test_choose_default_codex_model_skips_non_chat_models():
    models = [
        "openai/openai/openai/gpt-5.5-embedding",
        "openai/openai/openai/gpt-5.4",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.4"


def test_choose_default_codex_model_prefers_newer_gpt_5_when_no_55():
    models = [
        "openai/openai/openai/gpt-5.2",
        "openai/openai/openai/gpt-5.4",
        "openai/openai/openai/gpt-5.3-codex",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.4"


def test_choose_default_codex_model_prefers_multi_digit_minor_version():
    models = [
        "openai/openai/openai/gpt-5.9",
        "openai/openai/openai/gpt-5.10",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.10"


def test_choose_default_codex_model_prefers_codex_variant_on_same_version():
    models = [
        "openai/openai/openai/gpt-5.2",
        "openai/openai/openai/gpt-5.2-codex",
    ]

    assert choose_default_codex_model(models) == "openai/openai/openai/gpt-5.2-codex"


def test_choose_default_codex_model_returns_none_without_candidates():
    models = [
        "nvidia/moonshotai/kimi-k2.6",
        "aws/anthropic/bedrock-claude-opus-4-7",
    ]

    assert choose_default_codex_model(models) is None
    assert codex_model_candidates(models) == []
