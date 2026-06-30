# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the anonymous per-machine intake user id."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from switchyard.cli.intake_cli_config import IntakeCliConfig
from switchyard.cli.launchers.launch_intake_config import (
    LaunchIntakeConfig,
    resolve_machine_user_id,
)

_HEX_8 = re.compile(r"^[0-9a-f]{8}$")


def test_resolve_creates_stable_anonymous_id(tmp_path: Path) -> None:
    target = tmp_path / "user_id"
    first = resolve_machine_user_id(target)
    assert _HEX_8.match(first)
    assert target.read_text(encoding="utf-8").strip() == first
    # Second call reuses the persisted id rather than minting a new one.
    assert resolve_machine_user_id(target) == first


def test_resolve_reads_existing_id(tmp_path: Path) -> None:
    target = tmp_path / "user_id"
    target.write_text("cafebabe", encoding="utf-8")
    assert resolve_machine_user_id(target) == "cafebabe"


def test_resolve_is_fail_open_when_unwritable(tmp_path: Path) -> None:
    # A directory where the id file path is itself a directory cannot be read
    # or written; resolution must still return a usable id.
    target = tmp_path / "user_id"
    target.mkdir()
    assert _HEX_8.match(resolve_machine_user_id(target))


def test_resolve_fails_open_when_home_unresolvable(monkeypatch) -> None:
    # Path.home() raises RuntimeError when no home dir resolves; resolution
    # must still return a usable id rather than abort the launch.
    import switchyard.cli.launchers.launch_intake_config as mod

    def _boom() -> Path:
        raise RuntimeError("no home directory")

    monkeypatch.setattr(mod, "_machine_user_id_path", _boom)
    assert _HEX_8.match(resolve_machine_user_id())


def test_resolve_regenerates_on_malformed_or_corrupt_file(tmp_path: Path) -> None:
    # Non-hex text and undecodable bytes must fail open to a fresh valid id,
    # not abort the launch.
    text_target = tmp_path / "text_user_id"
    text_target.write_text("not-a-valid-id", encoding="utf-8")
    assert _HEX_8.match(resolve_machine_user_id(text_target))

    bytes_target = tmp_path / "bytes_user_id"
    bytes_target.write_bytes(b"\xff\xfe\x00bad")
    assert _HEX_8.match(resolve_machine_user_id(bytes_target))


def test_from_resolved_prefers_explicit_user_id(tmp_path: Path) -> None:
    config = LaunchIntakeConfig.from_resolved(
        base_url=None,
        workspace=None,
        api_key=None,
        app="claude-code",
        task="developer-session",
        session_id="sess-1",
        user_id="deadbeef",
        target="claude",
    )
    assert config.user_id == "deadbeef"


def test_to_sink_config_passes_user_id() -> None:
    config = LaunchIntakeConfig(
        base_url=None,
        workspace=None,
        api_key=None,
        app="claude-code",
        task="developer-session",
        session_id="sess-1",
        user_id="deadbeef",
    )
    assert config.to_sink_config().user_id == "deadbeef"


def test_from_launch_args_resolves_user_id_from_env() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_user_id=None)
    resolved = IntakeCliConfig.from_launch_args(args, env={"SWITCHYARD_USER_ID": "0badf00d"})
    assert resolved.user_id == "0badf00d"


def test_from_launch_args_prefers_flag_over_env() -> None:
    args = argparse.Namespace(intake_enabled=True, intake_user_id="11112222")
    resolved = IntakeCliConfig.from_launch_args(args, env={"SWITCHYARD_USER_ID": "0badf00d"})
    assert resolved.user_id == "11112222"
