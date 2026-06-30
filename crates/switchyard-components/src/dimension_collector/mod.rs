// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context-signal extraction layer.
//!
//! This module owns the pure logic for turning a request's text content
//! into a tuple of scored dimensions plus a token-count estimate and an
//! agentic scalar. The [`crate::request_processors::dimension_collector`]
//! module wraps it as a request-side Switchyard component.
//!
//! Port of `switchyard.experimental.rules_routing.dimension_collector`
//! (deleted in e5f88be2). This layer fits into a broader context-signal
//! vs estimated-signal taxonomy.

pub mod config;
pub mod response;
pub mod scorers;
pub mod tool_signals;

pub use config::{DimensionScore, Keywords, ScoringConfig, TokenCountThresholds};
pub use response::{extract_response_signals, ResponseFlag, ResponseSignals};
pub use tool_signals::{
    extract_tool_signals, extract_tool_signals_with_window, ToolResultSignal, DEFAULT_RECENT_WINDOW,
};

/// Token-per-character heuristic from ClawRouter (`strategy.ts`):
/// `estimatedTokens = ceil(fullText.length / 4)`. Cheap and good enough
/// for routing decisions. A real tokenizer lands behind a feature flag
/// in future work.
pub const CHARS_PER_TOKEN: u32 = 4;

/// Estimate token count using the ClawRouter chars/4 heuristic.
pub fn estimate_token_count(text: &str) -> u32 {
    let chars = text.chars().count() as u32;
    chars.div_ceil(CHARS_PER_TOKEN)
}

/// Aggregate output of one dimension-collection pass.
///
/// Stamped into `ProxyContext` by the request-processor adapter so
/// downstream estimators (LLM classifier, rules estimator, …) can read
/// dimension scores without re-extracting them.
///
/// Phase D removed the previously-emitted `agentic_score` scalar after
/// Calibration showed the underlying signal could not be
/// stably extracted from real TBLite traffic. Downstream consumers that
/// previously read `agentic_score` should use a context-level signal
/// (tool-call count, scope-aware routing input) instead.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextSignals {
    pub dimensions: Vec<DimensionScore>,
    pub token_count_estimate: u32,
}

/// Run all 14 dimension scorers against a prompt.
///
/// `lower_text` MUST be the lowercased version of the prompt. The raw
/// `prompt` is also accepted so `score_question_complexity` can count
/// `?` against the case-insensitive original (case doesn't matter for
/// `?`, but the unfolded text avoids re-walking the lower_text).
///
/// Emits a `ContextSignals` carrying:
///
/// * the 14 dimension scores in the canonical order documented below,
/// * a `chars/4` token-count estimate.
pub fn extract_signals(prompt: &str, lower_text: &str, config: &ScoringConfig) -> ContextSignals {
    let token_count_estimate = estimate_token_count(lower_text);

    let dimensions = vec![
        scorers::score_token_count(
            token_count_estimate,
            config.token_count.short,
            config.token_count.long,
        ),
        scorers::score_code_presence(lower_text, config.code_keywords.as_slice()),
        scorers::score_reasoning_markers(lower_text, config.reasoning_keywords.as_slice()),
        scorers::score_technical_terms(lower_text, config.technical_keywords.as_slice()),
        scorers::score_creative_markers(lower_text, config.creative_keywords.as_slice()),
        scorers::score_simple_indicators(lower_text, config.simple_keywords.as_slice()),
        scorers::score_imperative_verbs(lower_text, config.imperative_verbs.as_slice()),
        scorers::score_constraint_count(lower_text, config.constraint_indicators.as_slice()),
        scorers::score_output_format(lower_text, config.output_format_keywords.as_slice()),
        scorers::score_reference_complexity(lower_text, config.reference_keywords.as_slice()),
        scorers::score_negation_complexity(lower_text, config.negation_keywords.as_slice()),
        scorers::score_domain_specificity(lower_text, config.domain_specific_keywords.as_slice()),
        scorers::score_multi_step_patterns(lower_text),
        scorers::score_question_complexity(prompt),
    ];

    ContextSignals {
        dimensions,
        token_count_estimate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_token_count_uses_chars_div_four_ceiling() {
        assert_eq!(estimate_token_count(""), 0);
        assert_eq!(estimate_token_count("a"), 1);
        assert_eq!(estimate_token_count("abcd"), 1);
        assert_eq!(estimate_token_count("abcde"), 2);
    }

    #[test]
    fn extract_signals_runs_all_fourteen_scorers_in_canonical_order() {
        let config = ScoringConfig::default();
        let prompt = "Hello world.";
        let signals = extract_signals(prompt, &prompt.to_lowercase(), &config);

        let names: Vec<&str> = signals.dimensions.iter().map(|dim| dim.name).collect();
        assert_eq!(
            names,
            vec![
                "tokenCount",
                "codePresence",
                "reasoningMarkers",
                "technicalTerms",
                "creativeMarkers",
                "simpleIndicators",
                "imperativeVerbs",
                "constraintCount",
                "outputFormat",
                "referenceComplexity",
                "negationComplexity",
                "domainSpecificity",
                "multiStepPatterns",
                "questionComplexity",
            ],
        );
    }
}
