// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Health polling helpers for the components-v2 latency-service profile.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Deserialize;
use switchyard_core::{LlmTargetId, Result, SwitchyardError};

use super::{EndpointHealth, EndpointHealthStatus};

/// Polling client for one latency-service profile.
pub(crate) struct HealthPoller {
    url: String,
    target_ids: Vec<LlmTargetId>,
    client: reqwest::Client,
}

impl HealthPoller {
    /// Creates a poller for the profile's configured target IDs.
    pub(crate) fn new(
        latency_service_url: &str,
        target_ids: Vec<LlmTargetId>,
        poll_timeout: Duration,
    ) -> Result<Self> {
        let url = format!(
            "{}/v1/endpoints/health",
            latency_service_url.trim_end_matches('/')
        );
        let client = reqwest::Client::builder()
            .timeout(poll_timeout)
            .user_agent("SwitchyardLatencyServiceProfile")
            .build()
            .map_err(|error| {
                SwitchyardError::InvalidConfig(format!(
                    "latency_service failed to build HTTP client: {error}"
                ))
            })?;
        Ok(Self {
            url,
            target_ids,
            client,
        })
    }

    /// Returns the target IDs this poller sends to the latency service.
    pub(crate) fn target_ids(&self) -> &[LlmTargetId] {
        &self.target_ids
    }

    /// Fetches one health payload from the external latency service.
    pub(crate) async fn fetch_health(&self) -> Result<HealthResponse> {
        let params = self
            .target_ids
            .iter()
            .map(|target_id| ("endpoint_ids", target_id.as_str()))
            .collect::<Vec<_>>();
        let response = self
            .client
            .get(&self.url)
            .query(&params)
            .send()
            .await
            .map_err(|error| {
                SwitchyardError::Upstream(format!("Latency Service health request failed: {error}"))
            })?;
        let status = response.status();
        if !status.is_success() {
            return Err(SwitchyardError::Upstream(format!(
                "Latency Service health request returned HTTP {status}"
            )));
        }
        response
            .json::<RawHealthResponse>()
            .await
            .map(HealthResponse::from)
            .map_err(|error| {
                SwitchyardError::Upstream(format!(
                    "Latency Service health response was invalid JSON: {error}"
                ))
            })
    }
}

/// Parsed health response with public target health values.
pub(crate) struct HealthResponse {
    pub(crate) endpoint_health: BTreeMap<String, EndpointHealth>,
}

#[derive(Debug, Deserialize)]
struct RawHealthResponse {
    endpoint_health: BTreeMap<String, HealthEntry>,
}

#[derive(Debug, Deserialize)]
struct HealthEntry {
    status: EndpointHealthStatus,
    #[serde(default)]
    last_latency_ms: Option<f64>,
}

impl From<RawHealthResponse> for HealthResponse {
    fn from(response: RawHealthResponse) -> Self {
        Self {
            endpoint_health: response
                .endpoint_health
                .into_iter()
                .map(|(target_id, health)| (target_id, health.into()))
                .collect(),
        }
    }
}

impl From<HealthEntry> for EndpointHealth {
    fn from(entry: HealthEntry) -> Self {
        Self {
            status: entry.status,
            last_latency_ms: entry.last_latency_ms,
        }
    }
}

/// Converts positive finite seconds into a Rust duration.
pub(crate) fn duration_from_secs(value: f64, field: &'static str) -> Result<Duration> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SwitchyardError::InvalidConfig(format!(
            "latency_service {field} must be finite and positive, got {value:?}"
        )));
    }
    Duration::try_from_secs_f64(value).map_err(|error| {
        SwitchyardError::InvalidConfig(format!(
            "latency_service {field} is outside the supported duration range: {error}"
        ))
    })
}
