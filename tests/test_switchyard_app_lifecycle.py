# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for app-level component lifecycle wiring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from switchyard.server.switchyard_app import build_switchyard_app


class _LifecycleSwitchyard:
    state_key = "switchyard"

    def __init__(self) -> None:
        self.events: list[str] = []

    def iter_components(self) -> list[object]:
        return [_AsyncLifecycleComponent(self.events), _SyncLifecycleComponent(self.events)]


class _AsyncLifecycleComponent:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def startup(self) -> None:
        self._events.append("async-startup")

    async def shutdown(self) -> None:
        self._events.append("async-shutdown")


class _SyncLifecycleComponent:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def shutdown(self) -> None:
        self._events.append("sync-shutdown")


def test_build_switchyard_app_runs_component_lifecycle() -> None:
    switchyard = _LifecycleSwitchyard()
    app = build_switchyard_app(switchyard)  # type: ignore[arg-type]

    with TestClient(app):
        assert switchyard.events == ["async-startup"]

    assert switchyard.events == ["async-startup", "sync-shutdown", "async-shutdown"]
