// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-processor adapter for the context-signal extractor.
//!
//! Thin wrapper around [`crate::dimension_collector::extract_signals`].
//! The pure logic lives in [`crate::dimension_collector`]; this file is
//! the chain entry point — it pulls the user prompt out of `ChatRequest`,
//! lowercases it once, hands off to the extractor, and stamps the
//! resulting [`ContextSignals`] into `ProxyContext` via the typed
//! extensions bag.

use serde_json::Value;
use switchyard_core::{ChatRequest, ProxyContext, Result};

use crate::dimension_collector::{
    extract_signals, extract_tool_signals_with_window, ContextSignals, ScoringConfig,
    ToolResultSignal, DEFAULT_RECENT_WINDOW,
};

/// Populates `ProxyContext` with scored dimensions extracted from the prompt.
///
/// Read by downstream estimators (LLM classifier, future rules
/// estimator) via `ctx.get::<ContextSignals>()`.
#[derive(Clone, Debug)]
pub struct DimensionCollector {
    config: ScoringConfig,
    recent_window: usize,
}

impl Default for DimensionCollector {
    fn default() -> Self {
        Self {
            config: ScoringConfig::default(),
            recent_window: DEFAULT_RECENT_WINDOW,
        }
    }
}

impl DimensionCollector {
    /// Construct a collector with an explicit scoring config and the default
    /// `recent_*` window size ([`DEFAULT_RECENT_WINDOW`]).
    pub fn new(config: ScoringConfig) -> Self {
        Self::with_recent_window(config, DEFAULT_RECENT_WINDOW)
    }

    /// Construct a collector with a caller-supplied sliding-window size for
    /// `recent_*` signal counts. Smaller windows make the picker more
    /// reactive to the very last tool call; larger windows smooth over
    /// noisy turn-by-turn fluctuations.
    pub fn with_recent_window(config: ScoringConfig, recent_window: usize) -> Self {
        Self {
            config,
            recent_window,
        }
    }

    /// Returns the underlying scoring config (for tests / introspection).
    pub fn config(&self) -> &ScoringConfig {
        &self.config
    }

    /// Returns the configured `recent_*` sliding-window size.
    pub fn recent_window(&self) -> usize {
        self.recent_window
    }

    /// Extracts request-side signals and stores them on the request context.
    pub async fn process(
        &self,
        ctx: &mut ProxyContext,
        request: ChatRequest,
    ) -> Result<ChatRequest> {
        // Text-dimension pass (existing 15 scorers).
        let prompt = extract_user_prompt(&request).unwrap_or_default();
        let lower = prompt.to_lowercase();
        let signals = extract_signals(&prompt, &lower, &self.config);
        ctx.insert::<ContextSignals>(signals);
        // Tool-signal pass: walk the messages array for tool results and call names.
        let tool_signal = extract_tool_signals_with_window(&request, self.recent_window);
        ctx.insert::<ToolResultSignal>(tool_signal);
        Ok(request)
    }
}

/// Extract a single concatenated user-text blob from any inbound `ChatRequest`.
///
/// Looks at the request body's `messages` array (OpenAI chat / Anthropic
/// messages) or `input` field (OpenAI responses) and concatenates all
/// string user content. Tool calls, role-system messages, and structured
/// content blocks are intentionally ignored — they're either non-user
/// signal or handled by the upcoming dedicated scorers.
fn extract_user_prompt(request: &ChatRequest) -> Option<String> {
    let body = request.body();
    let object = body.as_object()?;

    if let Some(messages) = object.get("messages").and_then(Value::as_array) {
        let collected = collect_user_text_from_messages(messages);
        if !collected.is_empty() {
            return Some(collected);
        }
    }

    if let Some(input) = object.get("input") {
        match input {
            Value::String(text) => return Some(text.clone()),
            Value::Array(items) => {
                let collected = collect_user_text_from_messages(items);
                if !collected.is_empty() {
                    return Some(collected);
                }
            }
            _ => {}
        }
    }

    None
}

fn collect_user_text_from_messages(messages: &[Value]) -> String {
    let mut chunks: Vec<String> = Vec::new();
    for message in messages {
        let Some(object) = message.as_object() else {
            continue;
        };
        let role = object.get("role").and_then(Value::as_str).unwrap_or("");
        if role != "user" {
            continue;
        }
        match object.get("content") {
            Some(Value::String(text)) => chunks.push(text.clone()),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(text) = part
                        .as_object()
                        .and_then(|p| p.get("text"))
                        .and_then(Value::as_str)
                    {
                        chunks.push(text.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    chunks.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension_collector::Keywords;
    use serde_json::json;

    #[tokio::test]
    async fn stamps_context_signals_into_proxy_context() {
        let collector = DimensionCollector::new(ScoringConfig {
            code_keywords: Keywords::new(["def", "class"]),
            ..ScoringConfig::default()
        });

        let request = ChatRequest::openai_chat(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": "def hello(): class Foo: pass"},
            ],
        }));

        let mut ctx = ProxyContext::new();
        let _ = collector
            .process(&mut ctx, request)
            .await
            .expect("process ok");

        let signals = ctx
            .get::<ContextSignals>()
            .expect("ContextSignals stamped into context");
        assert_eq!(signals.dimensions.len(), 14);
        let code_presence = signals
            .dimensions
            .iter()
            .find(|dim| dim.name == "codePresence")
            .expect("codePresence dimension present");
        assert_eq!(code_presence.score, 1.0);
    }

    #[tokio::test]
    async fn handles_request_with_no_user_content_gracefully() {
        let collector = DimensionCollector::default();
        let request = ChatRequest::openai_chat(json!({
            "model": "test-model",
            "messages": [{"role": "system", "content": "only system"}],
        }));

        let mut ctx = ProxyContext::new();
        collector
            .process(&mut ctx, request)
            .await
            .expect("process ok");

        let signals = ctx.get::<ContextSignals>().expect("signals stamped");
        // Empty user prompt → 0 estimated tokens → tokenCount fires as "short".
        assert_eq!(signals.token_count_estimate, 0);
    }
}
