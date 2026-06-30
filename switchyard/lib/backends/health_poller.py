# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Health enum and background poller for the Latency Service usage case.

The Latency Service exposes a bulk health endpoint that returns one of
three states per endpoint.  :class:`EndpointHealthStatus` is the shared
contract between Switchyard and the Latency Service — both sides must
agree on these string values.

:class:`HealthPoller` is a daemon thread that periodically pulls fresh
verdicts from the Latency Service and writes them into an in-memory
cache shared with :class:`LatencyServiceLLMBackend`.  The backend's
request hot path only ever *reads* the cache (under a lock), so health
polling adds zero per-request latency.

Each cache entry is an :class:`EndpointHealth` snapshot carrying both
the discrete ``status`` and the most recent ``last_latency_ms`` sample
reported by the Latency Service.  The status drives tier selection
(HEALTHY > UNKNOWN > DEGRADED); the latency drives inverse-latency
weighted selection *within* a tier.

Design properties:

- **Daemon thread, not asyncio task.** Simple lifecycle: starts when the
  backend is constructed, dies with the process.  The sync ``httpx.Client``
  used inside the thread avoids event-loop contention for other work.
- **Fallback to UNKNOWN on poll failure.** If the Latency Service is
  unreachable, all endpoints reset to ``UNKNOWN`` so the backend falls
  back to random routing rather than acting on stale health data.
- **Graceful stop via ``stop()``.** Tests and ``shutdown()`` paths set
  ``_stop_event`` to break the polling loop promptly.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import NamedTuple

import httpx

logger = logging.getLogger(__name__)


class EndpointHealthStatus(enum.Enum):
    """Shared contract between Switchyard and the Latency Service.

    Three states are sufficient for routing decisions — the Latency
    Service may track finer-grained failure reasons internally for
    observability but exposes only these values in the health API.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class EndpointHealth(NamedTuple):
    """Snapshot of one endpoint's health, written by :class:`HealthPoller`.

    ``status`` drives tier-based routing (HEALTHY > UNKNOWN > DEGRADED);
    ``last_latency_ms`` — when present — drives inverse-latency weighted
    selection within whichever tier is chosen.  ``last_latency_ms`` is
    ``None`` before the first successful poll and whenever the Latency
    Service reports it as null.
    """

    status: EndpointHealthStatus
    last_latency_ms: float | None = None


class HealthPoller(threading.Thread):
    """Background daemon thread that refreshes the backend's health cache.

    Polls ``{latency_service_url}/v1/endpoints/health`` every
    ``poll_interval_s`` seconds and writes verdicts into
    ``health_cache`` under ``cache_lock``.  Holds references to the
    same dict and lock the backend reads from — no message passing, no
    extra copies.
    """

    def __init__(
        self,
        latency_service_url: str,
        model_ids: list[str],
        health_cache: dict[str, EndpointHealth],
        cache_lock: threading.Lock,
        poll_interval_s: float,
        poll_timeout_s: float,
    ) -> None:
        super().__init__(daemon=True, name="latency-health-poller")
        self._url = latency_service_url.rstrip("/") + "/v1/endpoints/health"
        self._model_ids = model_ids
        self._health_cache = health_cache
        self._cache_lock = cache_lock
        self._poll_interval_s = poll_interval_s
        self._http_client = httpx.Client(timeout=poll_timeout_s)
        self._stop_event = threading.Event()
        self._poll_count = 0
        # Failure counter and last-success timestamp drive the /metrics
        # poll-health gauges; tracked here because the poll loop is the
        # single point where outcomes are observable.
        self._poll_failures = 0
        self._last_poll_ok = False
        self._last_success_at: float | None = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                resp = self._http_client.get(
                    self._url,
                    params=[("endpoint_ids", mid) for mid in self._model_ids],
                )
                resp.raise_for_status()
                data = resp.json()
                with self._cache_lock:
                    for mid, info in data["endpoint_health"].items():
                        if mid in self._health_cache:
                            self._health_cache[mid] = EndpointHealth(
                                status=EndpointHealthStatus(info["status"]),
                                last_latency_ms=info.get("last_latency_ms"),
                            )
                self._poll_count += 1
                self._last_poll_ok = True
                self._last_success_at = time.monotonic()
            except Exception:
                logger.warning(
                    "Health poller: failed to reach Latency Service, "
                    "resetting all endpoints to UNKNOWN (random routing)"
                )
                with self._cache_lock:
                    for mid in self._health_cache:
                        self._health_cache[mid] = EndpointHealth(
                            status=EndpointHealthStatus.UNKNOWN,
                        )
                self._poll_failures += 1
                self._last_poll_ok = False
            self._stop_event.wait(timeout=self._poll_interval_s)

    @property
    def has_polled(self) -> bool:
        """True once at least one poll has successfully updated the cache."""
        return self._poll_count > 0

    @property
    def poll_successes(self) -> int:
        """Total number of successful polls since the poller started."""
        return self._poll_count

    @property
    def poll_failures(self) -> int:
        """Total number of poll attempts that failed (including timeouts).

        On every failure, all cached endpoint verdicts are reset to UNKNOWN
        so routing falls back to random selection instead of acting on
        stale health data.
        """
        return self._poll_failures

    @property
    def last_poll_ok(self) -> bool:
        """True iff the most recent poll attempt succeeded."""
        return self._last_poll_ok

    @property
    def seconds_since_last_success(self) -> float | None:
        """Monotonic seconds since the last successful poll.

        ``None`` before the first success — useful as a "never polled"
        signal on /metrics scrapes during startup or sustained outage.
        """
        if self._last_success_at is None:
            return None
        return time.monotonic() - self._last_success_at

    def stop(self) -> None:
        """Signal the polling loop to exit at the next iteration boundary."""
        self._stop_event.set()
