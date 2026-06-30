# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests asserting ``switchyard launch codex --plan-execute`` is gone.

The ``--plan-execute`` flag has been removed from the ``launch codex``
subparser. Plan-execute routing is now reachable only via a
``type: plan_execute`` route in a ``--routing-profiles`` YAML bundle.
"""

from __future__ import annotations

import pytest


class TestArgparse:
    def test_plan_execute_flag_removed(self) -> None:
        from switchyard.cli.switchyard_cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["launch", "codex", "--plan-execute"])
