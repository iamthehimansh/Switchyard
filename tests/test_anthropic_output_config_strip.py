# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inbound Anthropic bodies must drop output_config.format (a Claude Code
structured-output schema upstream Anthropic model groups reject) while keeping
output_config.effort."""

from switchyard.lib.endpoints.anthropic_messages_endpoint import (
    _strip_unsupported_output_config,
)


def test_strips_format_keeps_effort():
    body = {"model": "x", "output_config": {"effort": "high", "format": {"schema": {}}}}
    _strip_unsupported_output_config(body)
    assert body["output_config"] == {"effort": "high"}


def test_drops_output_config_when_only_format():
    body = {"model": "x", "output_config": {"format": {"schema": {}}}}
    _strip_unsupported_output_config(body)
    assert "output_config" not in body


def test_noop_without_output_config():
    body = {"model": "x", "messages": []}
    _strip_unsupported_output_config(body)
    assert body == {"model": "x", "messages": []}


def test_noop_when_no_format_key():
    body = {"model": "x", "output_config": {"effort": "high"}}
    _strip_unsupported_output_config(body)
    assert body["output_config"] == {"effort": "high"}
