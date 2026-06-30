# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVDataflow project CLI/env wiring into the intake sink config."""

from __future__ import annotations

import argparse

from switchyard.cli.intake_cli_config import IntakeCliConfig
from switchyard.cli.launchers.launch_intake_config import LaunchIntakeConfig


def test_launch_args_resolve_project_from_flag() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_nvdataflow_project="ipp-nova-tokenomics")
    resolved = IntakeCliConfig.from_launch_args(args, env={})
    assert resolved.nvdataflow_project == "ipp-nova-tokenomics"


def test_launch_args_resolve_project_from_env() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_nvdataflow_project=None)
    resolved = IntakeCliConfig.from_launch_args(
        args, env={"SWITCHYARD_NVDATAFLOW_PROJECT": "from-env"}
    )
    assert resolved.nvdataflow_project == "from-env"


def test_flag_wins_over_env() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_nvdataflow_project="from-flag")
    resolved = IntakeCliConfig.from_launch_args(
        args, env={"SWITCHYARD_NVDATAFLOW_PROJECT": "from-env"}
    )
    assert resolved.nvdataflow_project == "from-flag"


def test_server_args_resolve_project() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_nvdataflow_project="ipp-nova-tokenomics")
    resolved = IntakeCliConfig.from_server_args(args, env={})
    assert resolved.nvdataflow_project == "ipp-nova-tokenomics"


def test_project_absent_defaults_to_none() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_nvdataflow_project=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={})
    assert resolved.nvdataflow_project is None


def test_to_sink_config_passes_project_through_binding() -> None:
    config = LaunchIntakeConfig(
        base_url=None,
        workspace=None,
        api_key=None,
        app="claude-code",
        task="developer-session",
        session_id="sess-1",
        user_id="0badf00d",
        nvdataflow_project="ipp-nova-tokenomics",
    )
    assert config.to_sink_config().nvdataflow_project == "ipp-nova-tokenomics"


def test_launch_env_var_enables_intake() -> None:
    # SWITCHYARD_INTAKE_ENABLED turns intake on for launch (parity with serve).
    args = argparse.Namespace(intake_enabled=False, intake_nvdataflow_project=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={"SWITCHYARD_INTAKE_ENABLED": "1"})
    assert resolved.enabled is True


def test_launch_disabled_without_flag_or_env() -> None:
    args = argparse.Namespace(intake_enabled=False, intake_nvdataflow_project=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={})
    assert resolved.enabled is False


def test_server_env_var_enables_intake() -> None:
    args = argparse.Namespace(intake_enabled=False, intake_nvdataflow_project=None)
    resolved = IntakeCliConfig.from_server_args(args, env={"SWITCHYARD_INTAKE_ENABLED": "1"})
    assert resolved.enabled is True
