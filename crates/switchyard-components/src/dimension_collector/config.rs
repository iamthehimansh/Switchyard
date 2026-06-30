// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Config types and scored-signal records for the dimension collector.
//!
//! Mirrors `ClawRouter/src/router/types.ts` (MIT). Only the fields the
//! collector itself consumes are reproduced here; tier maps, promotions,
//! and model pricing live with the eventual rules estimator, not the
//! context-signal layer.

/// One scorer's output: dimension name, signed score, optional human-readable signal.
///
/// Scores roughly in `[-1, 1]`. Individual scorers narrow that range
/// (`simple_indicators` is non-positive, `code_presence` is non-negative).
#[derive(Clone, Debug, PartialEq)]
pub struct DimensionScore {
    pub name: &'static str,
    pub score: f32,
    pub signal: Option<String>,
}

impl DimensionScore {
    /// Convenience constructor for the no-signal case (score 0.0).
    pub fn zero(name: &'static str) -> Self {
        Self {
            name,
            score: 0.0,
            signal: None,
        }
    }
}

/// Token-count thresholds for `score_token_count`.
///
/// Below `short` → score `-1.0` (looks SIMPLE). Above `long` → score
/// `+1.0` (looks COMPLEX). In-between → `0.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenCountThresholds {
    pub short: u32,
    pub long: u32,
}

impl Default for TokenCountThresholds {
    fn default() -> Self {
        Self {
            short: 50,
            long: 500,
        }
    }
}

/// Pre-lowercased keyword list for substring-match scorers.
///
/// Keyword lists are lowercased once at config construction so the
/// per-request hot path does only the prompt-side `to_lowercase()` plus
/// `O(len(keywords) × len(text))` substring scans.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Keywords(Vec<String>);

impl Keywords {
    /// Constructs a lowercased keyword set from any iterator of strings.
    pub fn new(items: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let lowered: Vec<String> = items
            .into_iter()
            .map(|item| item.as_ref().to_lowercase())
            .collect();
        Self(lowered)
    }

    /// Constructs a keyword set from a static list of already-lowercase
    /// `&'static str` entries — used by [`ScoringConfig::clawrouter_defaults`]
    /// so the per-construction lowercase pass is skipped.
    pub fn from_static(items: &[&'static str]) -> Self {
        Self(items.iter().map(|s| (*s).to_string()).collect())
    }

    /// Returns the underlying lowercased keyword slice.
    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    /// Returns whether the keyword set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Per-scorer configuration consumed by the dimension collector.
///
/// Only the subset of `ScoringConfig` the collector needs lives here.
/// Tier boundaries, dimension weights, confidence calibration, and
/// model-pricing knobs belong to a separate rules estimator — putting
/// them here would couple the context-signal layer to scoring policy.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoringConfig {
    pub token_count: TokenCountThresholds,
    pub code_keywords: Keywords,
    pub reasoning_keywords: Keywords,
    pub simple_keywords: Keywords,
    pub technical_keywords: Keywords,
    pub creative_keywords: Keywords,
    pub imperative_verbs: Keywords,
    pub constraint_indicators: Keywords,
    pub output_format_keywords: Keywords,
    pub reference_keywords: Keywords,
    pub negation_keywords: Keywords,
    pub domain_specific_keywords: Keywords,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self::clawrouter_defaults()
    }
}

impl ScoringConfig {
    /// Returns the ClawRouter-aligned default keyword set.
    ///
    /// Trailing-space suffixes (e.g. `"def "`, `"not "`) prevent false
    /// substring matches inside common longer words (`"definition"`,
    /// `"another"`). The collector lower-cases the prompt once per request
    /// and matches each keyword as a substring, so suffix discipline is
    /// the cheapest way to keep precision up.
    pub fn clawrouter_defaults() -> Self {
        Self {
            token_count: TokenCountThresholds::default(),
            code_keywords: Keywords::from_static(defaults::CODE),
            reasoning_keywords: Keywords::from_static(defaults::REASONING),
            simple_keywords: Keywords::from_static(defaults::SIMPLE),
            technical_keywords: Keywords::from_static(defaults::TECHNICAL),
            creative_keywords: Keywords::from_static(defaults::CREATIVE),
            imperative_verbs: Keywords::from_static(defaults::IMPERATIVE_VERBS),
            constraint_indicators: Keywords::from_static(defaults::CONSTRAINT_INDICATORS),
            output_format_keywords: Keywords::from_static(defaults::OUTPUT_FORMAT),
            reference_keywords: Keywords::from_static(defaults::REFERENCE),
            negation_keywords: Keywords::from_static(defaults::NEGATION),
            domain_specific_keywords: Keywords::from_static(defaults::DOMAIN_SPECIFIC),
        }
    }
}

/// Static keyword tables for [`ScoringConfig::clawrouter_defaults`].
/// All entries lowercase by construction; retuning here is a pure data diff.
mod defaults {
    pub static CODE: &[&str] = &[
        "def ",
        "class ",
        "function",
        "async ",
        "await ",
        "import ",
        "from ",
        "return ",
        "fn ",
        "struct ",
        "impl ",
        "trait ",
        "pub ",
        "mod ",
        "package ",
        "interface ",
        "namespace ",
        "let ",
        "const ",
        "```",
        "->",
        "=>",
        "{}",
        "[]",
    ];

    pub static REASONING: &[&str] = &[
        "prove",
        "theorem",
        "derive",
        "step by step",
        "let's think",
        "reasoning",
        "deduce",
        "induction",
        "lemma",
        "proof",
        "show that",
        "demonstrate",
        "given that",
        "therefore",
        "hence ",
        "thus ",
        "implies",
        "because of",
    ];

    // "no " and "ok " removed — collide with "no longer", "no need", "ok with".
    pub static SIMPLE: &[&str] = &[
        "hello",
        "hi ",
        "thanks",
        "thank you",
        "what is",
        "what's",
        "define ",
        "meaning of",
        "summary of",
        "translate ",
    ];

    pub static TECHNICAL: &[&str] = &[
        "distributed",
        "concurrent",
        "kubernetes",
        "container",
        "algorithm",
        "protocol",
        "throughput",
        "latency",
        "bandwidth",
        "consistency",
        "compiler",
        "interpreter",
        "garbage collection",
        "memory leak",
        "race condition",
        "deadlock",
        "cache",
        "transaction",
        "atomic",
        "encryption",
        "compression",
        "deserialize",
        "serialize",
    ];

    pub static CREATIVE: &[&str] = &[
        "story",
        "poem",
        "novel",
        "lyrics",
        "creative",
        "brainstorm",
        "imagine",
        "fictional",
        "narrative",
        "character",
        "plot",
        "metaphor",
        "analogy",
        "essay",
    ];

    // Over-common verbs ("add", "update", "change", "make", "write")
    // removed — they appear in nearly every request and gave the dimension
    // no discriminating power.
    pub static IMPERATIVE_VERBS: &[&str] = &[
        "implement ",
        "design ",
        "refactor ",
        "architect ",
        "engineer ",
        "compose ",
        "scaffold ",
        "bootstrap ",
    ];

    pub static CONSTRAINT_INDICATORS: &[&str] = &[
        "must ",
        "should ",
        "at most",
        "at least",
        "within",
        "no more than",
        "fewer than",
        "exactly",
        "o(n)",
        "o(1)",
        "o(log",
        "linear time",
        "constant time",
        "bounded by",
        "must not",
        "should not",
        "ensure ",
    ];

    pub static OUTPUT_FORMAT: &[&str] = &[
        "json",
        "yaml",
        "csv",
        "xml",
        "html",
        "markdown",
        "table",
        "format as",
        "in the format",
        "output as",
        "structure",
        "schema",
    ];

    pub static REFERENCE: &[&str] = &[
        "the code above",
        "the code below",
        "the file",
        "the function",
        "the api docs",
        "the previous",
        "earlier you said",
        "in the last",
        "as shown above",
        "above mentioned",
        "as discussed",
        "you mentioned",
    ];

    pub static NEGATION: &[&str] = &[
        "not ",
        "never ",
        "neither ",
        "except ",
        "unless ",
        "without ",
        "no longer",
        "cannot ",
        "couldn't ",
        "shouldn't ",
        "wouldn't ",
        "doesn't ",
        "don't ",
        "didn't ",
    ];

    pub static DOMAIN_SPECIFIC: &[&str] = &[
        "quantum",
        "fpga",
        "embedded",
        "kernel",
        "driver",
        "genomics",
        "blockchain",
        "smart contract",
        "calculus",
        "differential",
        "tensor",
        "linear algebra",
        "signal processing",
        "fourier",
        "cryptography",
        "graph theory",
        "topology",
        "category theory",
    ];
}
