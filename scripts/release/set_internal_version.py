#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply an internal release tag version to Switchyard package metadata."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

TAG_RE = re.compile(
    r"^internal/v(?P<release>\d+\.\d+\.\d+)"
    r"(?:-(?P<label>alpha|beta|rc)\.(?P<number>\d+))?$"
)
PACKAGE_VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(".*)$')
PYTHON_VERSION_RE = re.compile(r'^(__version__\s*=\s*")([^"]+)(".*)$', re.MULTILINE)
INLINE_TABLE_RE = re.compile(r"^(\s*(?P<name>[\w-]+)\s*=\s*\{)(?P<body>.*)(\}\s*)$")

RUST_MANIFESTS = (
    Path("crates/switchyard-core/Cargo.toml"),
    Path("crates/switchyard-translation/Cargo.toml"),
    Path("crates/switchyard-components/Cargo.toml"),
    Path("crates/switchyard-server/Cargo.toml"),
    Path("crates/switchyard-py/Cargo.toml"),
)
INTERNAL_RUST_DEPENDENCIES = {
    "switchyard-core",
    "switchyard-translation",
    "switchyard-components",
    "switchyard-server",
    "switchyard-py",
}


@dataclasses.dataclass(frozen=True)
class InternalReleaseVersion:
    """Resolved package versions derived from an internal GitLab tag."""

    tag: str
    cargo: str
    python: str


def parse_internal_tag(tag: str) -> InternalReleaseVersion:
    """Return Cargo and Python package versions for an internal release tag."""

    match = TAG_RE.fullmatch(tag)
    if match is None:
        raise ValueError(
            "internal release tags must look like "
            "internal/v0.1.0, internal/v0.1.0-alpha.20260601, or internal/v0.1.0-rc.1"
        )

    release = match.group("release")
    label = match.group("label")
    number = match.group("number")
    if label is None:
        return InternalReleaseVersion(tag=tag, cargo=release, python=release)

    cargo = f"{release}-{label}.{number}"
    pep440_label = {"alpha": "a", "beta": "b", "rc": "rc"}[label]
    python = f"{release}{pep440_label}{number}"
    return InternalReleaseVersion(tag=tag, cargo=cargo, python=python)


def update_pyproject(path: Path, version: str) -> bool:
    """Set the `[project]` version in `pyproject.toml`."""

    lines = path.read_text().splitlines(keepends=True)
    in_project = False
    changed = False
    found = False
    output: list[str] = []
    for line in lines:
        section = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?$", line)
        if section is not None:
            in_project = section.group(1) == "project"

        if in_project:
            updated, count = PACKAGE_VERSION_RE.subn(rf"\g<1>{version}\g<3>", line, count=1)
            if count:
                found = True
                changed = changed or updated != line
                output.append(updated)
                continue

        output.append(line)

    if not found:
        raise ValueError(f"{path}: missing [project] version")
    if changed:
        path.write_text("".join(output))
    return changed


def update_python_init(path: Path, version: str) -> bool:
    """Set `switchyard.__version__`."""

    text = path.read_text()
    updated, count = PYTHON_VERSION_RE.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise ValueError(f"{path}: missing __version__")
    if updated != text:
        path.write_text(updated)
        return True
    return False


def update_cargo_package_version(path: Path, version: str, *, cargo_artifactory: bool) -> bool:
    """Set a Cargo package version and optionally make local deps publishable."""

    lines = path.read_text().splitlines(keepends=True)
    section = ""
    found = False
    changed = False
    output: list[str] = []
    for line in lines:
        section_match = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?$", line)
        if section_match is not None:
            section = section_match.group(1)

        updated = line
        if section == "package":
            updated, count = PACKAGE_VERSION_RE.subn(rf"\g<1>{version}\g<3>", line, count=1)
            if count:
                found = True
        elif cargo_artifactory and section == "dependencies":
            updated = rewrite_internal_dependency(line, version)

        changed = changed or updated != line
        output.append(updated)

    if not found:
        raise ValueError(f"{path}: missing [package] version")
    if changed:
        path.write_text("".join(output))
    return changed


def rewrite_internal_dependency(line: str, version: str) -> str:
    """Add version and registry metadata to known local path dependencies."""

    newline = "\n" if line.endswith("\n") else ""
    body_line = line[:-1] if newline else line
    match = INLINE_TABLE_RE.match(body_line)
    if match is None or match.group("name") not in INTERNAL_RUST_DEPENDENCIES:
        return line

    fields = parse_inline_table_fields(match.group("body"))
    if "path" not in fields:
        return line

    fields["version"] = f'"{version}"'
    fields.pop("table", None)
    fields["registry"] = '"artifactory"'
    ordered_keys = [key for key in ("path", "version", "registry") if key in fields]
    ordered_keys.extend(key for key in fields if key not in ordered_keys)
    body = ", ".join(f"{key} = {fields[key]}" for key in ordered_keys)
    return f"{match.group(1)} {body} }}{newline}"


def parse_inline_table_fields(body: str) -> dict[str, str]:
    """Parse the simple inline tables used by Switchyard Cargo dependencies."""

    fields: dict[str, str] = {}
    for raw_part in body.split(","):
        part = raw_part.strip()
        if not part:
            continue
        key, separator, value = part.partition("=")
        if not separator:
            raise ValueError(f"unsupported inline table field: {part!r}")
        fields[key.strip()] = value.strip()
    return fields


def apply_versions(version: InternalReleaseVersion, *, cargo_artifactory: bool) -> None:
    """Update all package metadata files in the repository."""

    changes = [
        ("pyproject.toml", update_pyproject(Path("pyproject.toml"), version.python)),
        ("switchyard/__init__.py", update_python_init(Path("switchyard/__init__.py"), version.python)),
    ]
    for manifest in RUST_MANIFESTS:
        changes.append(
            (
                str(manifest),
                update_cargo_package_version(
                    manifest,
                    version.cargo,
                    cargo_artifactory=cargo_artifactory,
                ),
            )
        )

    changed = [path for path, did_change in changes if did_change]
    if changed:
        print(f"Set internal release {version.tag}:")
        print(f"  Python: {version.python}")
        print(f"  Cargo:  {version.cargo}")
        for path in changed:
            print(f"  updated {path}")
    else:
        print(f"Internal release metadata already set for {version.tag}")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Internal release tag, such as internal/v0.1.0-rc.1")
    parser.add_argument(
        "--cargo-artifactory",
        action="store_true",
        help="Rewrite local Cargo dependencies for publishing to the artifactory registry",
    )
    parser.add_argument(
        "--print-python-version",
        action="store_true",
        help="Print only the derived Python version",
    )
    parser.add_argument(
        "--print-cargo-version",
        action="store_true",
        help="Print only the derived Cargo version",
    )
    args = parser.parse_args(argv)

    try:
        version = parse_internal_tag(args.tag)
        if args.print_python_version:
            print(version.python)
            return 0
        if args.print_cargo_version:
            print(version.cargo)
            return 0
        apply_versions(version, cargo_artifactory=args.cargo_artifactory)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
