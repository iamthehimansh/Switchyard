// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No-op profile for local smoke tests and benchmark harnesses.

use async_trait::async_trait;
use serde_json::json;
use switchyard_core::{ChatResponse, Result};

use crate::{profile_config, Profile, ProfileConfig, ProfileHooks, ProfileInput, ProfileResponse};

/// Config for a no-op profile.
#[profile_config("noop")]
pub struct NoopProfileConfig {}

impl ProfileConfig for NoopProfileConfig {
    type Runtime = NoopProfile;

    /// Builds the no-op runtime profile.
    fn build(&self) -> Result<Self::Runtime> {
        Ok(NoopProfile {})
    }
}

/// Profile that returns a deterministic local response without calling an upstream model.
pub struct NoopProfile {}

#[async_trait]
impl ProfileHooks for NoopProfile {
    type ProcessedRequest = ProfileInput;

    /// Leaves the request unchanged for hook-level inspection.
    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest> {
        Ok(input)
    }

    /// Leaves the response unchanged after no-op response creation.
    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for NoopProfile {
    /// Returns a deterministic OpenAI-compatible chat completion.
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        let processed = self.process(input).await?;
        let model = processed.request.model().unwrap_or("switchyard/noop");
        let response = self
            .rprocess(
                &processed,
                ChatResponse::openai_completion(json!({
                    "id": "switchyard-noop",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })),
            )
            .await?;
        Ok(ProfileResponse::from(response))
    }
}
