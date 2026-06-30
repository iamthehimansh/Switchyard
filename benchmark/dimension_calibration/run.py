# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 calibration runner.

Walks ``corpus_manifest.yaml``, calls the per-extractor module for each run,
and scores every prompt twice (``full`` and ``latest`` scope) through the
PyO3-backed :class:`switchyard_rust.components.DimensionCollector`.

Emits under ``report/``:

* ``data.csv``           — every prompt × scope × dimension score, one row.
* ``tier2_overall.md``   — per-dimension headline numbers.
* ``tier2_by_source.md`` — per-dimension fire rate, sliced by manifest source.
* ``tier2_by_turn.md``   — bucketed turn-index growth slices.

Run with::

    uv run python benchmark/dimension_calibration/run.py

Paths in the manifest resolve relative to the repo root, so adding a new
Harbor run to the corpus is one YAML entry pointing at the run directory.
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from extractors import ScoredPrompt

from switchyard.lib.proxy_context import ProxyContext
from switchyard_rust.components import (
    DimensionCollector,
    get_context_signals,
)
from switchyard_rust.core import ChatRequest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "corpus_manifest.yaml"
REPORT_DIR = HERE / "report"


def load_manifest() -> list[dict[str, Any]]:
    """Load the manifest and resolve paths relative to the repo root."""
    raw = yaml.safe_load(MANIFEST_PATH.read_text())
    runs: list[dict[str, Any]] = []
    for entry in raw.get("runs", []):
        run = dict(entry)
        run["path"] = str((REPO_ROOT / entry["path"]).resolve())
        runs.append(run)
    return runs


def stream_prompts(runs: list[dict[str, Any]]) -> Iterator[ScoredPrompt]:
    """Walk all runs in the manifest and yield ScoredPrompts in order."""
    for run in runs:
        module = importlib.import_module(f"extractors.{run['extractor']}")
        path = Path(run["path"])
        if not path.exists():
            print(
                f"  skip {run['id']}: path does not exist ({path})",
                file=sys.stderr,
            )
            continue
        kwargs: dict[str, Any] = {}
        if "source" in run:
            kwargs["source"] = run["source"]
        yield from module.extract(run["id"], path, **kwargs)


def make_request(text: str) -> ChatRequest:
    """Wrap a plain text blob as the user message of an OpenAI-chat request.

    The DimensionCollector adapter extracts the user content from
    ``messages[]``, so we shape the request body to match what production
    traffic looks like at the proxy boundary.
    """
    return ChatRequest.openai_chat({
        "model": "calibration",
        "messages": [{"role": "user", "content": text}],
    })


async def score_one(collector: DimensionCollector, text: str) -> dict[str, Any]:
    """Score a single prompt and return the signals as a flat dict."""
    ctx = ProxyContext()
    await collector.process(ctx, make_request(text))
    signals = get_context_signals(ctx)
    if signals is None:
        return {"token_count_estimate": 0, "dims": {}}
    return {
        "token_count_estimate": signals.token_count_estimate,
        "dims": {dim.name: dim.score for dim in signals.dimensions},
    }


async def score_all(prompts: Iterable[ScoredPrompt]) -> list[dict[str, Any]]:
    """Score every prompt twice (full + latest scopes) and return flat rows."""
    # No-arg constructor picks up the Rust-side `ScoringConfig::default()` —
    # the populated keyword set. Passing `ScoringConfig()` from Python would
    # build an explicit-empty config and bypass the Rust default.
    collector = DimensionCollector()
    rows: list[dict[str, Any]] = []
    count = 0
    started_at = time.perf_counter()
    for prompt in prompts:
        for scope in ("full", "latest"):
            text = prompt.full if scope == "full" else prompt.latest
            result = await score_one(collector, text)
            row = {
                **asdict(prompt),
                "scope": scope,
                "char_len": len(text),
                "token_count_estimate": result["token_count_estimate"],
                **{f"dim:{name}": score for name, score in result["dims"].items()},
            }
            # Trim raw text out of the CSV — keep only metadata. The raw
            # prompts live in the source trajectories; replicating them
            # here would balloon the CSV and serve no analytical purpose.
            row.pop("full", None)
            row.pop("latest", None)
            rows.append(row)
        count += 1
        if count % 500 == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"  scored {count} prompts ({elapsed:.1f}s, "
                f"{count / elapsed:.0f}/s)",
                file=sys.stderr,
            )
    elapsed = time.perf_counter() - started_at
    print(
        f"  scored {count} prompts total in {elapsed:.1f}s "
        f"({count / max(elapsed, 1e-9):.0f}/s)",
        file=sys.stderr,
    )
    return rows


# ─── Report writers ──────────────────────────────────────────────────────────

# Canonical dimension order as the collector emits them, mirrored here so the
# report tables read consistently across runs.
DIMENSIONS = [
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


def write_csv(rows: list[dict[str, Any]]) -> Path:
    """Dump every (prompt, scope, dimension) row to CSV for offline analysis."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "data.csv"
    fieldnames = [
        "run_id", "source", "task", "turn_idx", "scope", "char_len",
        "token_count_estimate",
        *[f"dim:{d}" for d in DIMENSIONS],
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out


def fire_stats(rows: list[dict[str, Any]], dim: str) -> dict[str, Any]:
    """Compute fire-rate + nonzero-score summary for one dimension."""
    key = f"dim:{dim}"
    fires: list[float] = []
    total = 0
    for row in rows:
        total += 1
        score = row.get(key, 0.0) or 0.0
        if score != 0.0:
            fires.append(score)
    if total == 0:
        return {"n": 0, "fires": 0, "rate": 0.0,
                "nonzero_min": None, "nonzero_p50": None, "nonzero_max": None}
    return {
        "n": total,
        "fires": len(fires),
        "rate": len(fires) / total,
        "nonzero_min": min(fires) if fires else None,
        "nonzero_p50": statistics.median(fires) if fires else None,
        "nonzero_max": max(fires) if fires else None,
    }


def write_overall(rows: list[dict[str, Any]]) -> Path:
    """Per-dimension headline: fire-rate + nonzero score band, by scope."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tier2_overall.md"
    lines: list[str] = []
    lines.append("# Tier-2 calibration — overall per-dimension fire rates")
    lines.append("")
    lines.append(f"Total prompt observations: **{len(rows)}** "
                 f"(prompts × 2 scopes).")
    lines.append("")
    for scope in ("full", "latest"):
        scoped = [r for r in rows if r["scope"] == scope]
        lines.append(f"## scope = `{scope}`  ({len(scoped)} rows)")
        lines.append("")
        lines.append("| Dimension | Fire rate | Fires | Nonzero min / p50 / max |")
        lines.append("|---|---:|---:|---|")
        for dim in DIMENSIONS:
            s = fire_stats(scoped, dim)
            band = "—" if s["fires"] == 0 else \
                f"{s['nonzero_min']:.2f} / {s['nonzero_p50']:.2f} / {s['nonzero_max']:.2f}"
            lines.append(
                f"| `{dim}` | {s['rate']:.1%} | {s['fires']} | {band} |"
            )
        lines.append("")
        # Aggregate scalars on the same scope
        tc = [r["token_count_estimate"] for r in scoped]
        if tc:
            lines.append(
                f"`token_count_estimate`: min={min(tc)}, "
                f"p50={statistics.median(tc):.0f}, max={max(tc)}, "
                f"mean={statistics.fmean(tc):.0f}"
            )
        lines.append("")
    out.write_text("\n".join(lines))
    return out


def write_by_source(rows: list[dict[str, Any]]) -> Path:
    """Per-dimension fire-rate sliced by trajectory source (hermes vs harbor)."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tier2_by_source.md"
    sources = sorted({r["source"] for r in rows})
    lines: list[str] = []
    lines.append("# Tier-2 calibration — fire rate by source")
    lines.append("")
    lines.append("Restricted to `scope=full` (production-realistic blob).")
    lines.append("")
    header = "| Dimension | " + " | ".join(sources) + " |"
    sep = "|---|" + "|".join(["---:"] * len(sources)) + "|"
    lines.append(header)
    lines.append(sep)
    for dim in DIMENSIONS:
        per_source: list[str] = []
        for src in sources:
            scoped = [r for r in rows if r["source"] == src and r["scope"] == "full"]
            s = fire_stats(scoped, dim)
            per_source.append(f"{s['rate']:.1%}")
        lines.append(f"| `{dim}` | " + " | ".join(per_source) + " |")
    lines.append("")
    out.write_text("\n".join(lines))
    return out


def write_by_turn(rows: list[dict[str, Any]]) -> Path:
    """Fire-rate sliced by turn-position bucket — diagnoses trajectory growth."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tier2_by_turn.md"
    # Three buckets: turn 0 (task framing), early (1-4), mid+ (5+).
    def bucket(idx: int) -> str:
        if idx == 0:
            return "turn=0"
        if idx <= 4:
            return "1-4"
        return "5+"

    lines: list[str] = []
    lines.append("# Tier-2 calibration — fire rate by turn position")
    lines.append("")
    lines.append("Restricted to `scope=full`. Buckets: `turn=0` (task framing) / "
                 "`1-4` (early loop) / `5+` (deep loop).")
    lines.append("")
    buckets = ["turn=0", "1-4", "5+"]
    header = "| Dimension | " + " | ".join(buckets) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join(["---:"] * len(buckets)) + "|")
    for dim in DIMENSIONS:
        per_bucket: list[str] = []
        for b in buckets:
            scoped = [
                r for r in rows
                if r["scope"] == "full" and bucket(r["turn_idx"]) == b
            ]
            s = fire_stats(scoped, dim)
            per_bucket.append(f"{s['rate']:.1%}")
        lines.append(f"| `{dim}` | " + " | ".join(per_bucket) + " |")
    lines.append("")
    # Per-bucket row counts for context
    counts: Counter[str] = Counter(
        bucket(r["turn_idx"]) for r in rows if r["scope"] == "full"
    )
    lines.append(
        "Row counts per bucket (scope=full): "
        + ", ".join(f"{b}={counts.get(b, 0)}" for b in buckets)
    )
    lines.append("")
    out.write_text("\n".join(lines))
    return out


def write_per_run_breakdown(rows: list[dict[str, Any]]) -> Path:
    """Per-run row counts so reviewers see corpus composition."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "tier2_corpus_composition.md"
    by_run: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scope"] == "full":
            by_run[row["run_id"]].append(row)
    lines: list[str] = []
    lines.append("# Tier-2 calibration — corpus composition")
    lines.append("")
    lines.append("| Run | Prompts (scope=full) | Tasks | Avg turn idx |")
    lines.append("|---|---:|---:|---:|")
    for run_id in sorted(by_run):
        scoped = by_run[run_id]
        tasks = len({r["task"] for r in scoped})
        avg_turn = statistics.fmean(r["turn_idx"] for r in scoped)
        lines.append(f"| `{run_id}` | {len(scoped)} | {tasks} | {avg_turn:.1f} |")
    lines.append("")
    out.write_text("\n".join(lines))
    return out


async def main() -> int:
    print("loading manifest…", file=sys.stderr)
    runs = load_manifest()
    print(f"  {len(runs)} runs declared", file=sys.stderr)

    print("scoring prompts (this is one DimensionCollector pass per scope)…",
          file=sys.stderr)
    rows = await score_all(stream_prompts(runs))

    print("writing reports…", file=sys.stderr)
    csv_path = write_csv(rows)
    overall_path = write_overall(rows)
    by_source_path = write_by_source(rows)
    by_turn_path = write_by_turn(rows)
    composition_path = write_per_run_breakdown(rows)

    print("done.", file=sys.stderr)
    print()
    print(f"  csv:                {csv_path.relative_to(REPO_ROOT)}")
    print(f"  overall report:     {overall_path.relative_to(REPO_ROOT)}")
    print(f"  by-source report:   {by_source_path.relative_to(REPO_ROOT)}")
    print(f"  by-turn report:     {by_turn_path.relative_to(REPO_ROOT)}")
    print(f"  corpus composition: {composition_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
