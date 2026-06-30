# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Markdown-docs fixture: alias passthrough profile builds to no-op profiles.

Mirrors ``tests/getting_started/conftest.py``: the README's "Use as a Python
library" snippet builds a passthrough profile and calls ``switchyard.call``,
which would otherwise hit a real backend. Gated on the ``--markdown-docs`` flag
so regular runs are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest


class _NoopProfileFixture:
    """No-op profile that preserves optional route-table decoration hooks."""

    def __init__(self) -> None:
        """Build the real no-op runtime used by README snippet execution."""
        from switchyard import NoopProfileConfig

        self._inner = NoopProfileConfig().build()

    def with_runtime_components(self, **_kwargs: Any) -> _NoopProfileFixture:
        """Accept route-table runtime components while staying hermetic."""
        return self

    async def process(self, input: Any) -> Any:
        """Delegate request-side profile work to the no-op runtime."""
        return await self._inner.process(input)

    async def rprocess(self, processed: Any, response: Any) -> Any:
        """Delegate response-side profile work to the no-op runtime."""
        return await self._inner.rprocess(processed, response)

    async def run(self, input: Any) -> Any:
        """Run the no-op runtime instead of making a real backend call."""
        return await self._inner.run(input)


def _markdown_docs_active(config: pytest.Config) -> bool:
    try:
        return bool(config.getoption("markdowndocs", default=False))
    except (KeyError, ValueError):
        return False


@pytest.fixture(autouse=True, scope="session")
def _markdown_docs_passthrough_to_noop(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    if not _markdown_docs_active(request.config):
        yield
        return

    from switchyard import PassthroughProfileConfig

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        PassthroughProfileConfig,
        "build",
        lambda _self: _NoopProfileFixture(),
    )
    try:
        yield
    finally:
        monkeypatch.undo()
