// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure dimension scorers — `(text, keywords, knobs) -> DimensionScore`.
//!
//! Direct port of the 15 scorer functions in
//! `switchyard.experimental.rules_routing.scorers` (deleted in e5f88be2),
//! itself a port of `ClawRouter/src/router/rules.ts` (MIT).
//!
//! All functions are pure: no shared state, no I/O. The collector
//! lowercases the input text once per request and lowercases the
//! keyword sets once at construction (see [`crate::dimension_collector::Keywords`]),
//! so the per-request hot path is just integer arithmetic and substring scans.

use super::config::DimensionScore;

/// Score the `tokenCount` dimension.
///
/// Below `short` → `-1.0` (looks SIMPLE). Above `long` → `+1.0`
/// (looks COMPLEX). Otherwise `0.0`. Mirrors `scoreTokenCount` in
/// `rules.ts` verbatim.
pub fn score_token_count(estimated_tokens: u32, short: u32, long: u32) -> DimensionScore {
    if estimated_tokens < short {
        return DimensionScore {
            name: "tokenCount",
            score: -1.0,
            signal: Some(format!("short ({estimated_tokens} tokens)")),
        };
    }
    if estimated_tokens > long {
        return DimensionScore {
            name: "tokenCount",
            score: 1.0,
            signal: Some(format!("long ({estimated_tokens} tokens)")),
        };
    }
    DimensionScore::zero("tokenCount")
}

/// Generic keyword-match scorer used by most dimensions.
///
/// Counts how many `lower_keywords` appear as substrings in `lower_text`.
/// Returns `score_high` once the count reaches `high`, `score_low` once
/// it reaches `low`, otherwise `0.0`. Mirrors `scoreKeywordMatch` in
/// `rules.ts`.
///
/// Both inputs MUST already be lower-cased.
//
// The 8-arg signature is intentional: this is the shared kernel for 10
// of the 15 dimension scorers. Each caller is a 1-line specialization
// that pins the per-dimension knobs (name, label, thresholds, scores).
// Bundling into a config struct would force every call site through a
// constructor + clone with no readability win.
#[allow(clippy::too_many_arguments)]
pub fn score_keyword_match(
    lower_text: &str,
    lower_keywords: &[String],
    name: &'static str,
    signal_label: &str,
    low: usize,
    high: usize,
    score_low: f32,
    score_high: f32,
) -> DimensionScore {
    let matches: Vec<&str> = lower_keywords
        .iter()
        .filter_map(|kw| lower_text.contains(kw.as_str()).then_some(kw.as_str()))
        .collect();

    let count = matches.len();
    if count >= high {
        return DimensionScore {
            name,
            score: score_high,
            signal: Some(format!("{signal_label} ({})", preview_matches(&matches),)),
        };
    }
    if count >= low {
        return DimensionScore {
            name,
            score: score_low,
            signal: Some(format!("{signal_label} ({})", preview_matches(&matches),)),
        };
    }
    DimensionScore::zero(name)
}

/// Score `codePresence`: how code-shaped does the prompt look?
///
/// Thresholds `(low=1, high=2)`, scores `(0.5, 1.0)`.
pub fn score_code_presence(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "codePresence",
        "code",
        1,
        2,
        0.5,
        1.0,
    )
}

/// Score `reasoningMarkers`: how proof / chain-of-thought shaped?
///
/// Thresholds `(low=1, high=2)`, scores `(0.7, 1.0)`. `score_low` of
/// 0.7 is deliberately high — one reasoning marker is a strong signal.
pub fn score_reasoning_markers(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "reasoningMarkers",
        "reasoning",
        1,
        2,
        0.7,
        1.0,
    )
}

/// Score `technicalTerms` (kubernetes, distributed, algorithm, …).
///
/// Thresholds `(low=2, high=4)`, scores `(0.5, 1.0)` — needs more
/// matches than other keyword dimensions because technical jargon is
/// noisier and a single hit means less.
pub fn score_technical_terms(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "technicalTerms",
        "technical",
        2,
        4,
        0.5,
        1.0,
    )
}

/// Score `creativeMarkers` (story, poem, brainstorm, …).
///
/// Thresholds `(low=1, high=2)`, scores `(0.5, 0.7)` — capped below
/// code / reasoning ceilings because creative requests aren't always
/// complex.
pub fn score_creative_markers(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "creativeMarkers",
        "creative",
        1,
        2,
        0.5,
        0.7,
    )
}

/// Score `simpleIndicators` (what is, hello, define, …).
///
/// ALWAYS negative: matches push the request toward SIMPLE rather
/// than away from any other tier. Scores `(-1.0, -1.0)`.
pub fn score_simple_indicators(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "simpleIndicators",
        "simple",
        1,
        2,
        -1.0,
        -1.0,
    )
}

/// Score `imperativeVerbs` (implement, design, refactor, …).
///
/// Thresholds `(low=2, high=3)`, scores `(0.3, 0.5)` — mild signal;
/// imperative verbs are common across all tiers so weights are low.
/// **Phase B retune**: thresholds raised from `(1, 2)` to `(2, 3)` after
/// Calibration showed a 97% baseline fire rate. Multiple
/// strong verbs now required before the signal fires.
pub fn score_imperative_verbs(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "imperativeVerbs",
        "imperative",
        2,
        3,
        0.3,
        0.5,
    )
}

/// Score `constraintCount` (at most, within, O(), …).
///
/// Thresholds `(low=1, high=3)`, scores `(0.3, 0.7)` — high score
/// requires multiple constraints; prompts with many constraints tend
/// to be COMPLEX/REASONING-tier algorithmic work.
pub fn score_constraint_count(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "constraintCount",
        "constraints",
        1,
        3,
        0.3,
        0.7,
    )
}

/// Score `outputFormat` (json, yaml, table, csv, …).
///
/// Thresholds `(low=1, high=2)`, scores `(0.4, 0.7)` — structured
/// output correlates with structured tasks (parsing, transformation)
/// which skew MEDIUM/COMPLEX.
pub fn score_output_format(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "outputFormat",
        "format",
        1,
        2,
        0.4,
        0.7,
    )
}

/// Score `referenceComplexity` ("the code above", "the API docs", …).
///
/// Thresholds `(low=1, high=2)`, scores `(0.3, 0.5)` — mild signal
/// that the request depends on multi-turn context.
pub fn score_reference_complexity(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "referenceComplexity",
        "references",
        1,
        2,
        0.3,
        0.5,
    )
}

/// Score `negationComplexity` (not, never, except, unless, …).
///
/// Thresholds `(low=2, high=3)`, scores `(0.3, 0.5)` — needs multiple
/// negations to fire; many real prompts contain at least one negation
/// so a single hit doesn't move the needle.
pub fn score_negation_complexity(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "negationComplexity",
        "negation",
        2,
        3,
        0.3,
        0.5,
    )
}

/// Score `domainSpecificity` (quantum, FPGA, genomics, …).
///
/// Thresholds `(low=1, high=2)`, scores `(0.5, 0.8)` — even one
/// domain-specific term is a strong signal that the request needs a
/// capable model.
pub fn score_domain_specificity(lower_text: &str, lower_keywords: &[String]) -> DimensionScore {
    score_keyword_match(
        lower_text,
        lower_keywords,
        "domainSpecificity",
        "domain-specific",
        1,
        2,
        0.5,
        0.8,
    )
}

/// Score `multiStepPatterns` (`first…then`, `step N`, numbered lists).
///
/// Three byte-level patterns from `scoreMultiStep` in `rules.ts`,
/// inlined as manual scans to avoid the regex dependency and the
/// runtime fallibility of `Regex::new`. Any hit → score `0.5`; no
/// hit → `0.0`. Caller passes already-lower-cased text.
pub fn score_multi_step_patterns(lower_text: &str) -> DimensionScore {
    if has_first_then(lower_text) || has_step_digit(lower_text) || has_numbered_list(lower_text) {
        return DimensionScore {
            name: "multiStepPatterns",
            score: 0.5,
            signal: Some("multi-step".to_string()),
        };
    }
    DimensionScore::zero("multiStepPatterns")
}

// Match the regex `first.*then`: substring "first" appears somewhere
// before substring "then".
fn has_first_then(lower_text: &str) -> bool {
    match lower_text.find("first") {
        Some(idx) => lower_text[idx..].contains("then"),
        None => false,
    }
}

// Match the regex `step \d`: substring "step " followed by an ASCII digit.
fn has_step_digit(lower_text: &str) -> bool {
    let bytes = lower_text.as_bytes();
    bytes
        .windows(6)
        .any(|window| &window[..5] == b"step " && window[5].is_ascii_digit())
}

// Match a tightened numbered-list pattern: ASCII digit, '.', whitespace,
// then an ASCII letter. Phase C retune — the original `\d\.\s` window
// matched version strings ("python 3.11 install"), floating-point
// numbers ("0.5 "), and shell prompts, firing on ~85% of calibration
// prompts. Requiring an ASCII letter after the whitespace catches actual
// list items ("1. Install ...") while ignoring numeric contexts.
fn has_numbered_list(lower_text: &str) -> bool {
    let bytes = lower_text.as_bytes();
    bytes.windows(4).any(|window| {
        window[0].is_ascii_digit()
            && window[1] == b'.'
            && window[2].is_ascii_whitespace()
            && window[3].is_ascii_alphabetic()
    })
}

/// Score `questionComplexity` — count of `?` characters in the prompt.
///
/// `>3` questions → score `0.5` (compound multi-part request); else
/// `0.0`. Mirrors `scoreQuestionComplexity` in `rules.ts`. Caller
/// passes the raw prompt (case doesn't matter for `?`).
pub fn score_question_complexity(prompt: &str) -> DimensionScore {
    let count = prompt.matches('?').count();
    if count > 3 {
        return DimensionScore {
            name: "questionComplexity",
            score: 0.5,
            signal: Some(format!("{count} questions")),
        };
    }
    DimensionScore::zero("questionComplexity")
}

// Phase D: the `score_agentic_task` scorer and the
// `agentic_score` scalar were removed entirely after calibration showed
// the dimension saturating at ~72% even after retuning the keyword list
// to irreversibility markers. No keyword choice discriminated agentic
// turns from "request running inside an agent harness" on real TBLite
// traffic. Downstream estimators that needed an explicit agentic
// signal must rely on context (request shape, tool-call counts) instead.

/// Inputs to the architecture-settled scorer.
///
fn preview_matches(matches: &[&str]) -> String {
    matches
        .iter()
        .take(3)
        .copied()
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_count_below_short_scores_negative_one() {
        let result = score_token_count(20, 50, 500);
        assert_eq!(result.score, -1.0);
        assert_eq!(result.name, "tokenCount");
        assert!(result.signal.as_deref().unwrap().contains("short"));
    }

    #[test]
    fn token_count_above_long_scores_positive_one() {
        let result = score_token_count(900, 50, 500);
        assert_eq!(result.score, 1.0);
        assert!(result.signal.as_deref().unwrap().contains("long"));
    }

    #[test]
    fn token_count_in_band_scores_zero() {
        let result = score_token_count(200, 50, 500);
        assert_eq!(result.score, 0.0);
        assert!(result.signal.is_none());
    }

    #[test]
    fn keyword_match_counts_substrings_with_low_high_thresholds() {
        let keywords: Vec<String> = ["foo", "bar", "baz"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let none = score_keyword_match(
            "nothing relevant here",
            &keywords,
            "x",
            "lbl",
            1,
            2,
            0.5,
            1.0,
        );
        assert_eq!(none.score, 0.0);

        let one = score_keyword_match(
            "this mentions foo only",
            &keywords,
            "x",
            "lbl",
            1,
            2,
            0.5,
            1.0,
        );
        assert_eq!(one.score, 0.5);

        let three =
            score_keyword_match("foo and bar and baz", &keywords, "x", "lbl", 1, 2, 0.5, 1.0);
        assert_eq!(three.score, 1.0);
    }

    #[test]
    fn code_presence_fires_on_two_matches() {
        let kws: Vec<String> = ["def", "class", "function"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let result = score_code_presence("def foo():\n    class Bar: ...", &kws);
        assert_eq!(result.score, 1.0);
        assert_eq!(result.name, "codePresence");
    }

    fn lowered(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_lowercase()).collect()
    }

    #[test]
    fn reasoning_markers_use_high_low_score_band() {
        let kws = lowered(&["prove", "derive", "theorem"]);
        assert_eq!(score_reasoning_markers("nothing here", &kws).score, 0.0);
        assert_eq!(
            score_reasoning_markers("prove the theorem", &kws).score,
            1.0
        );
        assert_eq!(score_reasoning_markers("just prove it", &kws).score, 0.7);
    }

    #[test]
    fn technical_terms_require_two_matches_to_fire() {
        let kws = lowered(&["kubernetes", "distributed", "algorithm", "protocol"]);
        // single hit → 0 because low=2
        assert_eq!(
            score_technical_terms("a kubernetes question", &kws).score,
            0.0
        );
        assert_eq!(
            score_technical_terms("kubernetes and distributed systems", &kws).score,
            0.5
        );
        assert_eq!(
            score_technical_terms("kubernetes, distributed algorithm, protocol design", &kws).score,
            1.0
        );
    }

    #[test]
    fn creative_markers_capped_below_code_ceiling() {
        let kws = lowered(&["story", "poem", "brainstorm"]);
        assert_eq!(
            score_creative_markers("write a short story", &kws).score,
            0.5
        );
        assert_eq!(
            score_creative_markers("write a story and a poem", &kws).score,
            0.7
        );
    }

    #[test]
    fn simple_indicators_always_negative() {
        let kws = lowered(&["hello", "what is", "define"]);
        assert_eq!(score_simple_indicators("hello world", &kws).score, -1.0);
        assert_eq!(
            score_simple_indicators("hello, define quark", &kws).score,
            -1.0
        );
        assert_eq!(score_simple_indicators("hard problem", &kws).score, 0.0);
    }

    #[test]
    fn imperative_verbs_are_mild_signal() {
        let kws = lowered(&["implement", "design", "refactor", "architect"]);
        // Phase B thresholds: (low=2, high=3). One match no longer fires.
        assert_eq!(score_imperative_verbs("implement a tree", &kws).score, 0.0);
        assert_eq!(
            score_imperative_verbs("implement and design things", &kws).score,
            0.3
        );
        assert_eq!(
            score_imperative_verbs("implement, design, refactor", &kws).score,
            0.5
        );
    }

    #[test]
    fn constraint_count_high_requires_three_hits() {
        let kws = lowered(&["at most", "within", "no more than"]);
        assert_eq!(
            score_constraint_count("solve at most quickly", &kws).score,
            0.3
        );
        assert_eq!(
            score_constraint_count("at most within no more than", &kws).score,
            0.7
        );
    }

    #[test]
    fn output_format_signal_for_structured_keywords() {
        let kws = lowered(&["json", "yaml", "csv"]);
        assert_eq!(score_output_format("return json", &kws).score, 0.4);
        assert_eq!(score_output_format("return json or yaml", &kws).score, 0.7);
    }

    #[test]
    fn reference_complexity_picks_up_context_pointers() {
        let kws = lowered(&["the code above", "the api docs"]);
        assert_eq!(
            score_reference_complexity("use the code above", &kws).score,
            0.3
        );
        assert_eq!(
            score_reference_complexity("the code above and the api docs", &kws).score,
            0.5
        );
    }

    #[test]
    fn negation_complexity_requires_two_negations() {
        let kws = lowered(&["not", "never", "except", "unless"]);
        assert_eq!(
            score_negation_complexity("this is not relevant", &kws).score,
            0.0
        );
        assert_eq!(
            score_negation_complexity("not now, never tomorrow", &kws).score,
            0.3
        );
        assert_eq!(
            score_negation_complexity("not, never, except", &kws).score,
            0.5
        );
    }

    #[test]
    fn domain_specificity_strongest_single_keyword_signal() {
        let kws = lowered(&["quantum", "fpga", "genomics"]);
        assert_eq!(
            score_domain_specificity("quantum question", &kws).score,
            0.5
        );
        assert_eq!(
            score_domain_specificity("quantum and fpga design", &kws).score,
            0.8
        );
    }

    #[test]
    fn multi_step_patterns_match_first_then_step_or_numbered_list() {
        assert_eq!(
            score_multi_step_patterns("nothing structured here").score,
            0.0
        );
        assert_eq!(
            score_multi_step_patterns("first do x, then do y").score,
            0.5
        );
        assert_eq!(
            score_multi_step_patterns("follow step 1 carefully").score,
            0.5
        );
        assert_eq!(
            score_multi_step_patterns("1. install\n2. configure").score,
            0.5
        );
    }

    #[test]
    fn question_complexity_fires_above_three_questions() {
        assert_eq!(score_question_complexity("Is this fast?").score, 0.0);
        assert_eq!(score_question_complexity("A? B? C? D? E?").score, 0.5);
    }
}
