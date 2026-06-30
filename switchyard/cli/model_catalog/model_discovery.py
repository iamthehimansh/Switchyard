# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model discovery and default model selection for launcher configuration."""

from typing import Literal

import httpx


class ModelDiscoveryError(RuntimeError):
    """Raised when the upstream model catalog cannot be fetched."""


def _models_url(base_url: str) -> str:
    """Input: a provider base URL. Output: the normalized ``/models`` URL."""

    return f"{base_url.rstrip('/')}/models"


def fetch_model_ids(
    base_url: str,
    api_key: str,
    timeout_s: float = 10.0,
) -> list[str]:
    """Input: endpoint credentials. Output: sorted unique model IDs.

    Accepts OpenAI-compatible ``GET /models`` responses where the body is
    either a list or a dict with ``data``. Rows may be raw strings or dicts
    carrying an ``id``, ``model``, or ``name`` string. Fetch/shape failures
    raise ``ModelDiscoveryError`` so command callers can decide whether to
    fail, warn, or fall back to manual model entry.
    """

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(_models_url(base_url), headers=headers)
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelDiscoveryError(str(exc)) from exc

    raw_items: object
    if isinstance(body, dict):
        raw_items = body.get("data", [])
    else:
        raw_items = body

    if not isinstance(raw_items, list):
        raise ModelDiscoveryError("GET /models response did not contain a model list")

    model_ids: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            model_ids.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("id", "model", "name"):
            value = item.get(key)
            if isinstance(value, str) and value:
                model_ids.append(value)
                break

    return sorted(set(model_ids))


def _ranked_model_candidates(
    model_ids: list[str],
    target: Literal["claude", "codex"],
) -> list[str]:
    """Input: discovered model IDs. Output: target candidates, best first.

    Model IDs are normalized into lowercase separator-delimited segments without
    regex. This keeps ``gpt`` distinct from ``egpt`` while still understanding
    provider paths and versions such as ``gpt-5.10`` or
    ``claude-sonnet-4-5@20251001``.

    Ranking policy:
    - Skip tool-only/non-chat model families such as embeddings and audio.
    - Prefer stable IDs over previews, nightlies, and other prereleases.
    - Claude defaults rank family first (Opus > Sonnet > Haiku), then version.
    - Codex defaults rank exact ``gpt`` models by version, then Codex/size tiers.
    - Snapshot dates are ignored as versions, so pinned dates do not outrank a
      newer actual model release.
    """

    separators = str.maketrans(dict.fromkeys("-_./:@", " "))
    denied_segments = {
        "audio",
        "classifier",
        "embedding",
        "embeddings",
        "embed",
        "guardrail",
        "image",
        "images",
        "moderation",
        "realtime",
        "rerank",
        "reranker",
        "stt",
        "test",
        "transcribe",
        "transcription",
        "tts",
    }
    unstable_segments = {
        "alpha",
        "beta",
        "dev",
        "experimental",
        "nightly",
        "preview",
    }
    claude_family_rank = {"opus": 3, "sonnet": 2, "haiku": 1}
    gpt_size_rank = {"max": 4, "pro": 4, "mini": 1, "nano": 0}

    scored: list[tuple[tuple[int, ...], str]] = []
    for model_id in model_ids:
        segments = tuple(model_id.strip().lower().translate(separators).split())
        segment_set = set(segments)
        if denied_segments & segment_set:
            continue

        stable_rank = int(not (unstable_segments & segment_set))
        alias_rank = int("latest" not in segment_set)

        if target == "claude":
            if "claude" not in segment_set:
                continue

            family_index = next(
                (
                    index
                    for index, segment in enumerate(segments)
                    if segment in claude_family_rank
                ),
                None,
            )
            family = segments[family_index] if family_index is not None else ""
            family_rank = claude_family_rank.get(family, 0)

            claude_index = segments.index("claude")
            version_starts = (
                [family_index + 1, claude_index + 1]
                if family_index is not None
                else [claude_index + 1]
            )
            provider_rank = int("anthropic" in segment_set) + int("azure" in segment_set)
        else:
            if "gpt" not in segment_set:
                continue

            version_starts = [segments.index("gpt") + 1]
            size_rank = 3
            for size_segment, rank in gpt_size_rank.items():
                if size_segment in segment_set:
                    size_rank = rank
                    break
            provider_rank = int("openai" in segment_set) + int("azure" in segment_set)

        version: list[int] = []
        for version_start in version_starts:
            version = []
            for segment in segments[version_start:]:
                if segment.isdigit():
                    if len(segment) == 8 or len(segment) > 3:
                        break
                    version.append(int(segment))
                elif segment.endswith("o") and segment[:-1].isdigit():
                    version.append(int(segment[:-1]))
                else:
                    break
                if len(version) == 3:
                    break
            if version:
                break
        version += [0] * (3 - len(version))

        rank_key: tuple[int, ...]
        if target == "claude":
            rank_key = (
                stable_rank,
                family_rank,
                version[0],
                version[1],
                version[2],
                provider_rank,
                alias_rank,
            )
        else:
            rank_key = (
                stable_rank,
                version[0],
                version[1],
                version[2],
                int("codex" in segment_set),
                size_rank,
                provider_rank,
                alias_rank,
            )
        scored.append((rank_key, model_id))

    scored.sort(key=lambda item: tuple(-part for part in item[0]) + (item[1],))
    return [model_id for _, model_id in scored]


def choose_default_claude_model(model_ids: list[str]) -> str | None:
    """Input: discovered model IDs. Output: best Claude default or ``None``."""

    candidates = claude_model_candidates(model_ids)
    return candidates[0] if candidates else None


def choose_default_codex_model(model_ids: list[str]) -> str | None:
    """Input: discovered model IDs. Output: best GPT/Codex default or ``None``."""

    candidates = codex_model_candidates(model_ids)
    return candidates[0] if candidates else None


def claude_model_candidates(model_ids: list[str]) -> list[str]:
    """Input: discovered model IDs. Output: Claude candidates, best first."""

    return _ranked_model_candidates(model_ids, "claude")


def codex_model_candidates(model_ids: list[str]) -> list[str]:
    """Input: discovered model IDs. Output: GPT candidates, best first."""

    return _ranked_model_candidates(model_ids, "codex")
