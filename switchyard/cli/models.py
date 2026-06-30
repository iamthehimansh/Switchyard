# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model catalog listing helpers for the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from switchyard.cli.model_catalog.model_discovery import (
    claude_model_candidates,
    codex_model_candidates,
)

ModelListTarget = Literal["all", "claude", "codex"]


@dataclass(frozen=True)
class ModelListRequest:
    """Inputs for rendering ``switchyard models``."""

    target: ModelListTarget = "all"
    query: str | None = None
    limit: int = 50


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _rank_models(model_ids: list[str], target: ModelListTarget) -> list[str]:
    if target == "claude":
        return _dedupe([*claude_model_candidates(model_ids), *model_ids])
    if target == "codex":
        return _dedupe([*codex_model_candidates(model_ids), *model_ids])
    return sorted(model_ids)


def filter_and_rank_models(
    model_ids: list[str],
    request: ModelListRequest,
) -> list[str]:
    ranked = _rank_models(model_ids, request.target)
    if request.query:
        needle = request.query.lower()
        ranked = [model_id for model_id in ranked if needle in model_id.lower()]
    if request.limit > 0:
        return ranked[: request.limit]
    return ranked


def render_models(model_ids: list[str], request: ModelListRequest) -> str:
    """Render a model list with target-aware ranking."""

    visible = filter_and_rank_models(model_ids, request)
    lines = [
        f"Models ({request.target})",
        f"showing: {len(visible)} of {len(model_ids)}",
    ]
    if request.query:
        lines.append(f"query: {request.query}")
    lines.append("")
    lines.extend(visible or ["<none>"])
    return "\n".join(lines)


__all__ = [
    "ModelListRequest",
    "filter_and_rank_models",
    "render_models",
]
