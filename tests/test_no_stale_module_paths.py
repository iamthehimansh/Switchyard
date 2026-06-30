# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard for stale Python module-path references in source code.

Runtime tests don't catch this class of bug — broken docstring paths
don't fail at import or at request time, only when someone follows the
reference (Sphinx build, IDE go-to-definition, copying the path into an
import). Hence this dedicated source-scan test.

Patterns checked:

* ``switchyard.core.`` — intermediate path that was flattened into
  ``switchyard.lib.`` / ``switchyard.cli.`` during the open-source
  cleanup. No file in the current tree should reference it.
* ``switchyard.foundation`` — pre-rename sub-package. Renamed to
  ``switchyard.lib`` in the same cleanup.
* ``nemo_switchyard.`` — original Python package name. Renamed to
  ``switchyard``. ``test_cli_stale_names.py`` covers user-facing CLI
  strings (``nemo-switchyard`` with a dash); this guards Python-import
  paths (``nemo_switchyard`` with an underscore).
* ``SwitchyardV2`` / ``switchyard_v2`` — V2 vs V1 was a transient
  migration concept that has since been collapsed to a single,
  unversioned ``Switchyard`` API. Make sure the V2 suffix does not
  creep back into module paths or symbols.
* ``switchyard_v2_cli`` / ``switchyard_v2_app`` / ``build_switchyard_v2_app``
  — old module/function names from the V2 era.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_STALE_PATH_PATTERNS = (
    "switchyard.core.",
    "switchyard.foundation",
    "nemo_switchyard.",
    "SwitchyardV2",
    "switchyard_v2",
    "build_switchyard_v2_app",
)

# Repo root: tests/<this file> → tests/ → repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "switchyard"


def _all_source_files() -> list[Path]:
    """Every ``.py`` file under the ``switchyard/`` package."""
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("pattern", _STALE_PATH_PATTERNS)
def test_no_stale_module_paths_in_package(pattern: str) -> None:
    """No source file under ``switchyard/`` may contain *pattern*.

    Failure message lists every offending ``file:line`` so the fix is a
    one-shot edit pass, not a repeated test-fail / fix / test-fail cycle.
    """
    offenders: list[str] = []
    for path in _all_source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        f"Found {len(offenders)} stale reference(s) to {pattern!r}:\n  "
        + "\n  ".join(offenders)
    )
