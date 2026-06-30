# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_anthropic_messages_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests hermetic.

    ``format=auto`` resolution probes ``/v1/messages`` over the network at
    backend-build time, so presets that default a Claude tier to ``auto`` would
    otherwise make live calls from unit tests. Stub it to a no-network default;
    tests that exercise the probe set their own value, which overrides this.
    """
    monkeypatch.setattr(
        "switchyard.lib.backends.backend_format_resolver.probe_anthropic_messages_support_sync",
        lambda **_kwargs: False,
    )
