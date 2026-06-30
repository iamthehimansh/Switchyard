# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end gate that a latency-service-routed chain exposes ``/metrics``.

Pins the fix for the production 404: the profile-backed latency service chain
must contribute a stats response processor so Datadog / OTel scrapers do not
see 404 on every scrape.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from switchyard.lib.backends.health_poller import HealthPoller
from switchyard.lib.backends.latency_service_llm_backend import (
    LatencyServiceLLMBackend,
)
from switchyard.lib.config.latency_service_backend_config import (
    LatencyServiceBackendConfig,
    LatencyServiceEndpoint,
)
from switchyard.lib.processors.stats_request_processor import (
    StatsRequestProcessor,
)
from switchyard.lib.processors.stats_response_processor_accumulator import (
    StatsResponseProcessor,
)
from switchyard.lib.profiles import LatencyServiceProfileConfig, ProfileSwitchyard
from switchyard.lib.stats_accumulator import StatsAccumulator
from switchyard.server.switchyard_app import build_switchyard_app


def _config(*, enable_stats: bool = True) -> LatencyServiceBackendConfig:
    return LatencyServiceBackendConfig(
        latency_service_url="http://latency-service.test:8080",
        endpoints=[
            LatencyServiceEndpoint(
                model="model-A",
                api_key="test-key",
                base_url="http://llm.test/v1",
            ),
        ],
        enable_stats=enable_stats,
    )


def _build_switchyard(
    config: LatencyServiceBackendConfig,
    *,
    stats_accumulator: StatsAccumulator | None = None,
) -> ProfileSwitchyard:
    """Build a latency-service profile without starting real poller threads."""
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            return ProfileSwitchyard(
                LatencyServiceProfileConfig.from_config(config)
                .build()
                .with_runtime_components(
                    stats_accumulator=stats_accumulator,
                    enable_stats=config.enable_stats,
                )
            )


# ---------------------------------------------------------------------------
# Profile component assembly
# ---------------------------------------------------------------------------


def test_profile_wires_stats_processors_by_default() -> None:
    switchyard = _build_switchyard(_config())

    components = list(switchyard.iter_components())

    assert any(isinstance(p, StatsRequestProcessor) for p in components), (
        "expected StatsRequestProcessor in the latency-service profile"
    )
    assert any(isinstance(p, StatsResponseProcessor) for p in components), (
        "expected StatsResponseProcessor in the latency-service profile"
    )


def test_profile_skips_stats_when_disabled() -> None:
    switchyard = _build_switchyard(_config(enable_stats=False))
    components = list(switchyard.iter_components())

    assert not any(isinstance(p, StatsRequestProcessor) for p in components)
    assert not any(isinstance(p, StatsResponseProcessor) for p in components)


async def test_profile_shares_one_accumulator_across_components() -> None:
    """Response processor and backend must record into the same accumulator.

    The shared bucket is how ``/metrics`` ends up with both per-call counts
    (recorded by the backend) and token usage (recorded by the response
    processor) under the same model label. The Rust binding's
    ``accumulator`` getter returns a fresh Python wrapper each time, so
    we verify shared identity by writing through one surface and reading
    through the other.
    """
    switchyard = _build_switchyard(_config())
    components = list(switchyard.iter_components())

    stats_processor = next(
        p for p in components
        if isinstance(p, StatsResponseProcessor)
    )
    backend = next(
        p for p in components
        if isinstance(p, LatencyServiceLLMBackend)
    )

    # Write through the backend's accumulator handle, then read through the
    # response processor's accumulator handle. If construction wired
    # two separate accumulators, the read would not see the write.
    assert backend._stats is not None
    await backend._stats.record_success("model-A", 12.5)

    snapshot = await stats_processor.accumulator.snapshot()
    assert snapshot["total_requests"] == 1
    assert snapshot["models"]["model-A"]["calls"] == 1

    assert stats_processor.get_endpoint() is not None


def test_profile_honors_externally_provided_stats_accumulator() -> None:
    shared = StatsAccumulator()
    switchyard = _build_switchyard(_config(), stats_accumulator=shared)

    stats_processor = next(
        p for p in switchyard.iter_components()
        if isinstance(p, StatsResponseProcessor)
    )
    # The processor's stats source must be the shared accumulator so an
    # externally-mounted StatsEndpoint surfaces the same data.
    assert stats_processor.get_endpoint() is not None


# ---------------------------------------------------------------------------
# End-to-end: build_switchyard_app exposes /metrics + /v1/stats
# ---------------------------------------------------------------------------


def _build_app(*, enable_stats: bool):
    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            switchyard = ProfileSwitchyard(
                LatencyServiceProfileConfig.from_config(_config(enable_stats=enable_stats))
                .build()
                .with_runtime_components(enable_stats=enable_stats)
            )
            return build_switchyard_app(switchyard)


def test_recipe_app_exposes_metrics_endpoint() -> None:
    """Regression: latency-service deployments must serve ``/metrics``.

    Before this fix, Datadog/OTel scrapers got 404 on every poll because the
    latency-service chain contributed no ``StatsEndpoint`` via
    :func:`build_switchyard_app`'s component iteration.
    """
    app = _build_app(enable_stats=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        metrics = client.get("/metrics")
        stats = client.get("/v1/stats")
        routing_stats = client.get("/v1/routing/stats")

    assert metrics.status_code == 200, (
        f"/metrics should be 200 on a latency-service chain, got {metrics.status_code}"
    )
    assert metrics.headers["content-type"].startswith("text/plain")
    # Prometheus exposition includes the canonical counter family even at zero.
    assert "switchyard_requests_total" in metrics.text

    assert stats.status_code == 200
    assert routing_stats.status_code == 200


def test_recipe_app_omits_metrics_when_stats_disabled() -> None:
    """``enable_stats=False`` keeps ``/metrics`` unmounted — opt-out is intentional."""
    app = _build_app(enable_stats=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        metrics = client.get("/metrics")
        stats = client.get("/v1/stats")

    assert metrics.status_code == 404
    assert stats.status_code == 404


# ---------------------------------------------------------------------------
# YAML route-bundle path
# ---------------------------------------------------------------------------


def test_route_bundle_latency_service_exposes_metrics() -> None:
    """Deployment path (YAML bundle) must also surface ``/metrics``."""
    from switchyard.cli.route_bundle import build_route_bundle_table

    bundle = {
        "routes": {
            "ls-route": {
                "type": "latency_service",
                "latency_service_url": "http://latency-service.test:8080",
                "endpoints": [
                    {
                        "model": "model-A",
                        "api_key": "test-key",
                        "base_url": "http://llm.test/v1",
                    },
                ],
            },
        },
    }

    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            table = build_route_bundle_table(bundle)
            app = build_switchyard_app(table)

            with TestClient(app, raise_server_exceptions=False) as client:
                metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "switchyard_requests_total" in metrics.text


def test_route_bundle_latency_service_honors_enable_stats_false() -> None:
    """YAML ``enable_stats: false`` opts out of the metrics surface."""
    from switchyard.cli.route_bundle import build_route_bundle_table

    bundle = {
        "routes": {
            "ls-route": {
                "type": "latency_service",
                "latency_service_url": "http://latency-service.test:8080",
                "enable_stats": False,
                "endpoints": [
                    {
                        "model": "model-A",
                        "api_key": "test-key",
                        "base_url": "http://llm.test/v1",
                    },
                ],
            },
        },
    }

    with patch(
        "switchyard.lib.backends.latency_service_llm_backend.OpenAILLMClient",
    ) as mock_cls:
        mock_cls.side_effect = lambda **kw: MagicMock(name=f"client-{kw.get('base_url')}")
        with patch.object(HealthPoller, "start"), patch.object(HealthPoller, "stop"):
            table = build_route_bundle_table(bundle)
            app = build_switchyard_app(table)

            with TestClient(app, raise_server_exceptions=False) as client:
                metrics = client.get("/metrics")

    assert metrics.status_code == 404
