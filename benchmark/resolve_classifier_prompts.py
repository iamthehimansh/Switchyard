# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve deterministic-route classifier prompts for benchmark manifests."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from switchyard.lib.processors.llm_classifier import DEFAULT_MAX_REQUEST_CHARS
from switchyard.lib.processors.llm_classifier.presets import (
    classifier_prompt_sha256,
    profile_default_prompt,
)

_DETERMINISTIC_TYPES = {"deterministic", "llm_classifier", "llm_classifier_routing"}
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MIN_MAX_REQUEST_CHARS = 256


def _expand_env_string(value: str) -> str:
    missing = [name for name in _ENV_REF_RE.findall(value) if name not in os.environ]
    if missing:
        raise ValueError(
            f"missing environment variable(s): {', '.join(sorted(set(missing)))}"
        )
    return os.path.expandvars(value)


def _int_or_default(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _bounded_int_or_default(value: object, default: int, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if value < minimum:
        return default
    return value


def resolve_classifier_prompts(path: Path) -> dict[str, dict[str, Any]]:
    """Return effective classifier-prompt metadata for deterministic routes."""
    loaded = yaml.safe_load(path.read_text())
    routes = loaded.get("routes") if isinstance(loaded, dict) else None
    if not isinstance(routes, dict):
        return {}

    result: dict[str, dict[str, Any]] = {}
    for route_key, route in routes.items():
        if not isinstance(route_key, str) or not isinstance(route, dict):
            continue
        route_type = str(route.get("type", "")).lower().replace("-", "_")
        if route_type not in _DETERMINISTIC_TYPES:
            continue

        profile = route.get("profile") or "general"
        if not isinstance(profile, str):
            raise ValueError(f"route {route_key!r}: profile must be a string")

        classifier = route.get("classifier")
        classifier_map = classifier if isinstance(classifier, dict) else {}
        raw_prompt = classifier_map.get("prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            prompt = _expand_env_string(raw_prompt)
            source = "override"
        else:
            prompt = profile_default_prompt(profile)
            source = "profile_default"

        result[route_key] = {
            "profile": profile,
            "source": source,
            "classifier_prompt": prompt,
            "classifier_prompt_sha256": classifier_prompt_sha256(prompt),
            "max_request_chars": _bounded_int_or_default(
                classifier_map.get("max_request_chars"),
                DEFAULT_MAX_REQUEST_CHARS,
                minimum=_MIN_MAX_REQUEST_CHARS,
            ),
            "recent_turn_window": _int_or_default(
                classifier_map.get("recent_turn_window"),
                4,
            ),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    """Print classifier-prompt metadata JSON; never fail a benchmark run."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("WARNING: no routing-profiles path supplied", file=sys.stderr)
        print("{}")
        return 0
    try:
        print(json.dumps(resolve_classifier_prompts(Path(args[0]))))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to resolve classifier prompts: {exc}", file=sys.stderr)
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
