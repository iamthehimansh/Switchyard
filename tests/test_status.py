# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from switchyard.cli.status import StatusRequest, render_status


def test_status_base_url_source_matches_selected_env_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SWITCHYARD_CONFIG_DIR", str(tmp_path))
    for env_var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/v1")

    status = render_status(StatusRequest())

    assert "provider: nvidia" in status
    assert "base URL: https://inference-api.nvidia.com/v1" in status
    assert "base URL source: built-in default" in status
