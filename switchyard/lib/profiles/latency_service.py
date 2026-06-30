# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile-owned latency-service routing construction."""

from __future__ import annotations

from typing import Self

from switchyard.lib.config import LatencyServiceBackendConfig
from switchyard.lib.profiles.chain import ComponentChainProfile
from switchyard.lib.profiles.table import profile_config


@profile_config("latency_service")
class LatencyServiceProfileConfig:
    """Profile config wrapper for health-aware latency-service profiles."""

    config: LatencyServiceBackendConfig

    @classmethod
    def from_config(cls, config: LatencyServiceBackendConfig) -> Self:
        """Create a profile config from the validated parsing model."""
        return cls(config=config)

    def build(self) -> ComponentChainProfile:
        """Build the latency-service profile runtime."""
        from switchyard.lib.backends import LatencyServiceLLMBackend

        backend = LatencyServiceLLMBackend(self.config)
        return ComponentChainProfile(
            backend=backend,
        )


__all__ = ["LatencyServiceProfileConfig"]
