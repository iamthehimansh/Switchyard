# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal UI helpers for the Switchyard CLI."""

from switchyard.cli.tui.choice_selector import (
    ChoiceOption,
    ChoiceSelector,
)
from switchyard.cli.tui.launch_config_wizard import LaunchConfigWizard
from switchyard.cli.tui.model_selector import ModelCandidate, ModelSelector
from switchyard.cli.tui.terminal_capabilities import (
    NO_TUI_ENV_VAR,
    TerminalCapabilities,
    detect_terminal_capabilities,
)
from switchyard.cli.tui.tui_session import TuiSession

__all__ = [
    "NO_TUI_ENV_VAR",
    "ChoiceOption",
    "ChoiceSelector",
    "LaunchConfigWizard",
    "ModelCandidate",
    "ModelSelector",
    "TerminalCapabilities",
    "TuiSession",
    "detect_terminal_capabilities",
]
