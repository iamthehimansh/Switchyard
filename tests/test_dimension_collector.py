# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python tests for the Rust-backed ``DimensionCollector``.

Mirrors the coverage of the deleted
``switchyard/experimental/rules_routing/tests/test_dimension_collector.py``
against the new Rust port exposed through ``switchyard_rust.components``.
"""

from __future__ import annotations

from typing import Any, cast

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.components import (
    ContextSignals,
    DimensionCollector,
    DimensionScore,
    ScoringConfig,
    get_context_signals,
)
from switchyard_rust.core import ChatRequest


def _request(prompt: str) -> ChatRequest:
    return ChatRequest.openai_chat(
        cast(
            Any,
            {
                "model": "client-model",
                "messages": [{"role": "user", "content": prompt}],
            },
        ),
    )


def _default_config() -> ScoringConfig:
    return ScoringConfig(
        code_keywords=["def", "class", "function"],
        reasoning_keywords=["prove", "theorem", "derive"],
        simple_keywords=["hello", "what is", "define"],
        technical_keywords=["kubernetes", "distributed", "algorithm", "protocol"],
        creative_keywords=["story", "poem", "brainstorm"],
        imperative_verbs=["build", "create", "implement"],
        constraint_indicators=["at most", "within", "no more than"],
        output_format_keywords=["json", "yaml", "csv"],
        reference_keywords=["the code above", "the api docs"],
        negation_keywords=["not", "never", "except", "unless"],
        domain_specific_keywords=["quantum", "fpga", "genomics"],
    )


async def test_dimension_collector_stamps_context_signals_into_proxy_context() -> None:
    """Hot-path assertion: ``get_context_signals`` returns the stamped record."""
    collector = DimensionCollector(_default_config())
    ctx = ProxyContext()

    await collector.process(ctx, _request("def hello(): class Foo: pass"))

    signals = get_context_signals(ctx)
    assert isinstance(signals, ContextSignals)
    assert len(signals.dimensions) == 14
    code_presence = next(d for d in signals.dimensions if d.name == "codePresence")
    assert isinstance(code_presence, DimensionScore)
    assert code_presence.score == 1.0


async def test_dimensions_are_emitted_in_canonical_order() -> None:
    """Order is a public contract for downstream estimators (e.g. weighted sums)."""
    collector = DimensionCollector(_default_config())
    ctx = ProxyContext()

    await collector.process(ctx, _request("Hello world."))

    signals = get_context_signals(ctx)
    assert signals is not None
    assert [d.name for d in signals.dimensions] == [
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
    ]


async def test_token_count_short_pushes_negative_score() -> None:
    """A bare ``hello`` should score short (chars/4 = 1 ≪ default short=50)."""
    collector = DimensionCollector(_default_config())
    ctx = ProxyContext()

    await collector.process(ctx, _request("hello"))

    signals = get_context_signals(ctx)
    assert signals is not None
    token_dim = next(d for d in signals.dimensions if d.name == "tokenCount")
    assert token_dim.score == -1.0


async def test_simple_indicators_pull_negative_for_chitchat() -> None:
    collector = DimensionCollector(_default_config())
    ctx = ProxyContext()

    await collector.process(ctx, _request("hello, what is the meaning of life?"))

    signals = get_context_signals(ctx)
    assert signals is not None
    simple_dim = next(d for d in signals.dimensions if d.name == "simpleIndicators")
    assert simple_dim.score == -1.0


async def test_multi_step_patterns_detect_first_then_chain() -> None:
    collector = DimensionCollector(_default_config())
    ctx = ProxyContext()

    await collector.process(ctx, _request("first do x then do y"))

    signals = get_context_signals(ctx)
    assert signals is not None
    multi = next(d for d in signals.dimensions if d.name == "multiStepPatterns")
    assert multi.score == 0.5


async def test_question_complexity_fires_above_three_questions() -> None:
    collector = DimensionCollector(_default_config())
    ctx = ProxyContext()

    await collector.process(ctx, _request("A? B? C? D? E?"))

    signals = get_context_signals(ctx)
    assert signals is not None
    questions = next(d for d in signals.dimensions if d.name == "questionComplexity")
    assert questions.score == 0.5


async def test_collector_without_args_uses_populated_rust_defaults() -> None:
    """No-arg `DimensionCollector()` picks up `ScoringConfig::default()`.

    The Rust-side `Default` ships populated keyword lists. A bland prompt like ``"just some text"`` should still
    emit the full 14-dimension vector with sensible zeros for
    non-matching scorers.
    """
    collector = DimensionCollector()
    ctx = ProxyContext()

    await collector.process(ctx, _request("just some text"))

    signals = get_context_signals(ctx)
    assert signals is not None
    assert len(signals.dimensions) == 14
    # `"just some text"` is short (< 50 token estimate) so `tokenCount`
    # fires negative; no keyword scorer should hit on this prompt.
    by_name = {d.name: d.score for d in signals.dimensions}
    assert by_name["tokenCount"] == -1.0
    keyword_dims = {
        "codePresence", "reasoningMarkers", "technicalTerms",
        "creativeMarkers", "simpleIndicators", "imperativeVerbs",
        "constraintCount", "outputFormat", "referenceComplexity",
        "negationComplexity", "domainSpecificity",
    }
    for name in keyword_dims:
        assert by_name[name] == 0.0, f"{name} unexpectedly fired"


async def test_get_context_signals_returns_none_when_not_stamped() -> None:
    """Reader must not fabricate a record when no collector has run."""
    ctx = ProxyContext()
    assert get_context_signals(ctx) is None
