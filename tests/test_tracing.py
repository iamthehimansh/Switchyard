# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional ddtrace wrapper in :mod:`switchyard.lib.tracing`."""

from __future__ import annotations

from typing import Any

import pytest

from switchyard.lib import tracing


class _FakeSpan:
    def __init__(self) -> None:
        self.tags: dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


class _FakeTracer:
    def __init__(self) -> None:
        self.started: list[tuple[str, _FakeSpan]] = []

    def trace(self, name: str) -> _FakeSpan:
        span = _FakeSpan()
        self.started.append((name, span))
        return span


def test_routing_span_noops_without_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no tracer, the context manager still yields a usable no-op span."""
    monkeypatch.setattr(tracing, "_dd_tracer", None)
    with tracing.routing_span("switchyard.route_decision") as span:
        span.set_tag("switchyard.model", "m")  # must not raise


def test_routing_span_uses_tracer_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _FakeTracer()
    monkeypatch.setattr(tracing, "_dd_tracer", tracer)
    with tracing.routing_span("switchyard.upstream_attempt") as span:
        span.set_tag("switchyard.selected_endpoint", "model-A")
    assert tracer.started[0][0] == "switchyard.upstream_attempt"
    assert tracer.started[0][1].tags == {"switchyard.selected_endpoint": "model-A"}


def test_set_tags_skips_none() -> None:
    span = _FakeSpan()
    tracing.set_tags(span, {"a": 1, "b": None, "c": "x", "d": False})
    # ``None`` is dropped; falsey-but-meaningful values (0, False, "") are kept.
    assert span.tags == {"a": 1, "c": "x", "d": False}
