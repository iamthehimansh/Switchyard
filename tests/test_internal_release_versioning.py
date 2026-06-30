# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/release/set_internal_version.py"
_SPEC = importlib.util.spec_from_file_location("set_internal_version", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
set_internal_version = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = set_internal_version
_SPEC.loader.exec_module(set_internal_version)


@pytest.mark.parametrize(
    ("tag", "cargo", "python"),
    [
        ("internal/v0.1.1", "0.1.1", "0.1.1"),
        ("internal/v0.1.1-alpha.20260601", "0.1.1-alpha.20260601", "0.1.1a20260601"),
        ("internal/v0.1.1-beta.2", "0.1.1-beta.2", "0.1.1b2"),
        ("internal/v0.1.1-rc.3", "0.1.1-rc.3", "0.1.1rc3"),
    ],
)
def test_parse_internal_tag(tag: str, cargo: str, python: str) -> None:
    version = set_internal_version.parse_internal_tag(tag)

    assert version.tag == tag
    assert version.cargo == cargo
    assert version.python == python


@pytest.mark.parametrize(
    "tag",
    [
        "v0.1.1",
        "internal/0.1.1",
        "internal/v0.1",
        "internal/v0.1.1.dev1",
        "internal/v0.1.1-alpha",
        "internal/v0.1.1-nightly.20260601",
    ],
)
def test_parse_internal_tag_rejects_public_or_invalid_tags(tag: str) -> None:
    with pytest.raises(ValueError):
        set_internal_version.parse_internal_tag(tag)


def test_rewrite_internal_dependency_adds_publish_metadata() -> None:
    line = 'switchyard-core = { path = "../switchyard-core" }\n'

    rewritten = set_internal_version.rewrite_internal_dependency(line, "0.1.1-rc.1")

    assert (
        rewritten
        == 'switchyard-core = { path = "../switchyard-core", version = "0.1.1-rc.1", registry = "artifactory" }\n'
    )


def test_rewrite_internal_dependency_replaces_existing_publish_metadata() -> None:
    line = (
        'switchyard-components = { path = "../switchyard-components", '
        'version = "0.1.0", table = "old" }\n'
    )

    rewritten = set_internal_version.rewrite_internal_dependency(line, "0.1.1-alpha.20260601")

    assert (
        rewritten
        == 'switchyard-components = { path = "../switchyard-components", version = "0.1.1-alpha.20260601", registry = "artifactory" }\n'
    )


def test_rewrite_internal_dependency_leaves_external_dependencies_alone() -> None:
    line = 'serde = { version = "1", features = ["derive"] }\n'

    assert set_internal_version.rewrite_internal_dependency(line, "0.1.1") == line


def test_metadata_file_updates(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[build-system]\nrequires = []\n\n[project]\nname = "switchyard"\nversion = "0.1.0"\n')
    init = tmp_path / "__init__.py"
    init.write_text('__all__ = []\n\n__version__ = "0.1.0"\n')
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        '[package]\nname = "switchyard-components"\nversion = "0.1.0"\n\n'
        '[dependencies]\nswitchyard-core = { path = "../switchyard-core" }\n'
    )

    assert set_internal_version.update_pyproject(pyproject, "0.1.1rc1")
    assert set_internal_version.update_python_init(init, "0.1.1rc1")
    assert set_internal_version.update_cargo_package_version(
        cargo,
        "0.1.1-rc.1",
        cargo_artifactory=True,
    )

    assert 'version = "0.1.1rc1"' in pyproject.read_text()
    assert '__version__ = "0.1.1rc1"' in init.read_text()
    assert 'version = "0.1.1-rc.1"' in cargo.read_text()
    assert 'registry = "artifactory"' in cargo.read_text()
