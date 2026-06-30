// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Intake sink configuration.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Behavior when the async intake queue is full.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntakeQueueFullPolicy {
    /// Drop the payload and keep serving the user response.
    #[default]
    Drop,
    /// Wait for queue capacity before returning from enqueue.
    Block,
}

/// Runtime configuration for the HTTP intake sink.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct IntakeSinkConfig {
    /// Base URL of the intake API service.
    pub intake_base_url: Option<String>,
    /// Workspace path segment for the intake API.
    pub workspace: Option<String>,
    /// User ID attached to every intake payload.
    pub user_id: String,
    /// Bearer token used when posting intake payloads.
    pub api_key: Option<String>,
    /// When set, post a flat NVDataflow telemetry document to this project's
    /// posting endpoint instead of the nemo-platform chat-completions ingest.
    pub nvdataflow_project: Option<String>,
    /// Maximum buffered payloads before applying `on_queue_full`.
    pub max_queue_size: usize,
    /// Per-request HTTP timeout in seconds.
    pub request_timeout_s: f64,
    /// Number of retry attempts after the first failed POST.
    pub max_retries: u32,
    /// Queue pressure behavior.
    pub on_queue_full: IntakeQueueFullPolicy,
    /// Capture prompt/response text. Off by default (metadata-only).
    #[serde(default)]
    pub capture_content: bool,
}

impl fmt::Debug for IntakeSinkConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntakeSinkConfig")
            .field("intake_base_url", &self.intake_base_url)
            .field("workspace", &self.workspace)
            .field("user_id", &self.user_id)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("nvdataflow_project", &self.nvdataflow_project)
            .field("max_queue_size", &self.max_queue_size)
            .field("request_timeout_s", &self.request_timeout_s)
            .field("max_retries", &self.max_retries)
            .field("on_queue_full", &self.on_queue_full)
            .field("capture_content", &self.capture_content)
            .finish()
    }
}

impl Default for IntakeSinkConfig {
    fn default() -> Self {
        Self {
            intake_base_url: None,
            workspace: None,
            user_id: "switchyard".to_string(),
            api_key: None,
            nvdataflow_project: None,
            max_queue_size: 1000,
            request_timeout_s: 10.0,
            max_retries: 2,
            on_queue_full: IntakeQueueFullPolicy::Drop,
            capture_content: false,
        }
    }
}

/// Default NVDataflow host used when `intake_base_url` is unset.
pub const DEFAULT_NVDATAFLOW_BASE_URL: &str = "https://nvdataflow.nvidia.com";

impl IntakeSinkConfig {
    /// Returns the configured workspace or the Python-compatible default.
    pub fn workspace_or_default(&self) -> &str {
        self.workspace.as_deref().unwrap_or("default")
    }

    /// Posting URL for the configured NVDataflow project, or `None` when the
    /// sink runs in chat-completions ingest mode.
    pub fn nvdataflow_posting_url(&self) -> Option<String> {
        self.nvdataflow_project.as_deref().map(|project| {
            let base = self
                .intake_base_url
                .as_deref()
                .unwrap_or(DEFAULT_NVDATAFLOW_BASE_URL)
                .trim_end_matches('/');
            format!("{base}/dataflow/{project}/posting")
        })
    }
}
