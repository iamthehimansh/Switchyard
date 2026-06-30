// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Target selection policy for the components-v2 latency-service profile.

use std::collections::{BTreeMap, BTreeSet};

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use switchyard_core::{LlmTargetId, Result, SwitchyardError};

use super::{EndpointHealth, EndpointHealthStatus};

/// Selected target ID emitted by the latency-service selector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectedTarget {
    /// Target ID selected from the health cache.
    pub target_id: LlmTargetId,
    /// Health tier that made the target selectable.
    pub health_status: EndpointHealthStatus,
}

/// Selects a target from the health cache while avoiding excluded targets when possible.
pub(crate) fn select_target(
    snapshot: &BTreeMap<LlmTargetId, EndpointHealth>,
    excluded: &BTreeSet<LlmTargetId>,
) -> Result<SelectedTarget> {
    for tier in EndpointHealthStatus::ROUTING_ORDER {
        let entries = candidates_for_tier(snapshot, excluded, tier);
        if !entries.is_empty() {
            return pick_from_tier(&entries, tier);
        }
    }

    Err(SwitchyardError::InvalidConfig(
        "latency_service profile has no selectable targets".to_string(),
    ))
}

fn candidates_for_tier<'a>(
    snapshot: &'a BTreeMap<LlmTargetId, EndpointHealth>,
    excluded: &BTreeSet<LlmTargetId>,
    tier: EndpointHealthStatus,
) -> Vec<(&'a LlmTargetId, EndpointHealth)> {
    snapshot
        .iter()
        .filter_map(|(target_id, health)| {
            (!excluded.contains(target_id) && health.status == tier).then_some((target_id, *health))
        })
        .collect()
}

fn pick_from_tier(
    entries: &[(&LlmTargetId, EndpointHealth)],
    health_status: EndpointHealthStatus,
) -> Result<SelectedTarget> {
    if entries.len() == 1 {
        return Ok(SelectedTarget {
            target_id: entries[0].0.clone(),
            health_status,
        });
    }

    let weights = entries
        .iter()
        .map(|(_, health)| {
            health
                .last_latency_ms
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|value| 1.0 / value)
        })
        .collect::<Option<Vec<_>>>();

    let index = if let Some(weights) = weights {
        weighted_index(&weights)?
    } else {
        rand::thread_rng().gen_range(0..entries.len())
    };

    Ok(SelectedTarget {
        target_id: entries[index].0.clone(),
        health_status,
    })
}

fn weighted_index(weights: &[f64]) -> Result<usize> {
    let distribution = WeightedIndex::new(weights).map_err(|error| {
        SwitchyardError::InvalidConfig(format!(
            "latency_service computed invalid target weights: {error}"
        ))
    })?;
    Ok(distribution.sample(&mut rand::thread_rng()))
}
