# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified ``LiveStatsFooter``.

One layout for every routing strategy: an aggregate row + one indented row
per active outbound model tier.
"""

from __future__ import annotations

from switchyard.cli.launchers.live_stats_footer import FOOTER_ROWS, LiveStatsFooter
from switchyard.lib.stats_accumulator import StatsAccumulator


def _strip_ansi(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and not text[j].isalpha():
                j += 1
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


class _StubHealth:
    def poll(self) -> None:
        return None

    @property
    def indicator(self) -> tuple[str, int]:
        return ("●", 1)


async def _stats_with_model_call(
    model: str,
    *,
    prompt: int = 1234,
    completion: int = 567,
    cached: int = 200,
) -> StatsAccumulator:
    acc = StatsAccumulator()
    await acc.record_success(model=model)
    await acc.record_usage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
    )
    return acc


def _footer(stats: StatsAccumulator, *, model: str = "nvidia/some/default") -> LiveStatsFooter:
    return LiveStatsFooter(stats, model=model, health=_StubHealth())  # type: ignore[arg-type]


async def test_footer_height_at_zero_traffic() -> None:
    """Before any traffic: aggregate + 1 fallback row = 2 rows."""
    footer = _footer(StatsAccumulator())
    assert footer.height == FOOTER_ROWS == 2
    rows = footer.render(cols=80)
    assert len(rows) == 2


async def test_aggregate_row_shows_totals_without_model_name() -> None:
    stats = await _stats_with_model_call("vendor/some-model")
    rows = _footer(stats).render(cols=80)
    agg = _strip_ansi(rows[0][0])
    assert "switchyard" in agg
    assert "1 req" in agg
    assert "1,234 in" in agg
    assert "567 out" in agg
    # The aggregate row never names a model — that's the active row's job.
    assert "some-model" not in agg


async def test_active_row_shows_model_with_recent_traffic() -> None:
    stats = await _stats_with_model_call("vendor/winner-model")
    rows = _footer(stats).render(cols=80)
    active = _strip_ansi(rows[1][0])
    assert "winner-model" in active
    assert "1 req" in active
    assert "1,234 in" in active
    assert "567 out" in active
    assert "200 cached" in active


async def test_active_row_falls_back_to_default_when_no_traffic() -> None:
    """Before any backend call lands, the row labels with the launch default."""
    rows = _footer(StatsAccumulator(), model="vendor/launch-default").render(cols=80)
    active = _strip_ansi(rows[1][0])
    assert "launch-default" in active
    assert "0 req" in active


async def test_new_tier_adds_a_row_on_next_render() -> None:
    """A new model in traffic adds a row; height grows from 2 to 3."""
    stats = StatsAccumulator()
    await stats.record_success(model="vendor/first")
    await stats.record_usage(
        model="vendor/first", prompt_tokens=10, completion_tokens=20,
    )
    footer = _footer(stats)
    rows = footer.render(cols=80)
    assert len(rows) == 2
    assert footer.height == 2
    assert "first" in _strip_ansi(rows[1][0])

    await stats.record_success(model="vendor/second")
    await stats.record_usage(
        model="vendor/second", prompt_tokens=30, completion_tokens=40,
    )
    rows = footer.render(cols=80)
    assert len(rows) == 3
    assert footer.height == 3
    tier_texts = " ".join(_strip_ansi(r[0]) for r in rows[1:])
    assert "first" in tier_texts
    assert "second" in tier_texts
