// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Health snapshots used by the components-v2 latency-service profile.

use serde::{Deserialize, Serialize};

/// Endpoint states returned by the external Latency Service.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointHealthStatus {
    /// Endpoint is eligible for preferred routing.
    Healthy,
    /// Endpoint can still receive traffic, but only after healthy and unknown pools.
    Degraded,
    /// Endpoint has no current health verdict.
    Unknown,
}

impl EndpointHealthStatus {
    /// Routing preference order for latency-service target selection.
    pub(crate) const ROUTING_ORDER: [Self; 3] = [Self::Healthy, Self::Unknown, Self::Degraded];

    /// Stable lowercase label used by routing metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unknown => "unknown",
        }
    }
}

/// Cached health snapshot for one configured target.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct EndpointHealth {
    /// Service-reported health status for the endpoint.
    pub status: EndpointHealthStatus,
    /// Optional last observed latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_latency_ms: Option<f64>,
}

impl EndpointHealth {
    /// Creates a health snapshot with no latency sample.
    pub fn new(status: EndpointHealthStatus) -> Self {
        Self {
            status,
            last_latency_ms: None,
        }
    }

    /// Creates a health snapshot with a latency sample.
    pub fn with_latency(status: EndpointHealthStatus, last_latency_ms: f64) -> Self {
        Self {
            status,
            last_latency_ms: Some(last_latency_ms),
        }
    }
}
