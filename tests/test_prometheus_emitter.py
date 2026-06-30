# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the module-level Prometheus emitter table."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from switchyard.lib.endpoints import prometheus_emitter


@pytest.fixture(autouse=True)
def _clean_table() -> Iterator[None]:
    """Isolate every test: emitters registered here must not leak."""
    prometheus_emitter._clear_for_tests()
    yield
    prometheus_emitter._clear_for_tests()


class TestTableLifecycle:
    def test_render_empty_returns_empty_string(self) -> None:
        """No emitters → empty string so callers can concat unconditionally."""
        assert prometheus_emitter.render() == ""

    def test_register_then_render_composes_lines(self) -> None:
        prometheus_emitter.register(lambda: ["foo 1", "bar 2"])
        rendered = prometheus_emitter.render()
        assert "foo 1" in rendered
        assert "bar 2" in rendered
        assert rendered.endswith("\n")

    def test_register_is_idempotent(self) -> None:
        """Re-registering the same callable must not double-count."""
        emitter = lambda: ["x 1"]  # noqa: E731
        prometheus_emitter.register(emitter)
        prometheus_emitter.register(emitter)
        assert prometheus_emitter.render().count("x 1") == 1

    def test_unregister_removes_emitter(self) -> None:
        emitter = lambda: ["x 1"]  # noqa: E731
        prometheus_emitter.register(emitter)
        prometheus_emitter.unregister(emitter)
        assert prometheus_emitter.render() == ""

    def test_unregister_unknown_is_noop(self) -> None:
        """Shutdown paths should not throw if registration was skipped."""
        prometheus_emitter.unregister(lambda: ["x"])

    def test_multiple_emitters_compose_in_registration_order(self) -> None:
        prometheus_emitter.register(lambda: ["a 1"])
        prometheus_emitter.register(lambda: ["b 2"])
        rendered = prometheus_emitter.render()
        assert rendered.index("a 1") < rendered.index("b 2")
