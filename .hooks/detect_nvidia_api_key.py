# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""detect-secrets plugin for literal NVIDIA_API_KEY assignments."""

from __future__ import annotations

import re
from collections.abc import Generator

from detect_secrets.plugins.base import BasePlugin


class NvidiaApiKeyDetector(BasePlugin):
    """Detects hard-coded NVIDIA_API_KEY values, including low-entropy literals."""

    secret_type = "NVIDIA API Key"  # pragma: allowlist secret

    _assignment = re.compile(
        r"""
        (?<![A-Z0-9_])
        NVIDIA_API_KEY
        (?![A-Z0-9_])
        \s*(?:=|:|:=|=>|::)\s*
        (?P<quote>["']?)
        (?P<secret>[^"'\s,;#()[\]{}]+)
        (?P=quote)
        (?=$|\s|[,;#])
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    def analyze_string(self, string: str) -> Generator[str]:
        """Yields literal NVIDIA_API_KEY values while allowing env/secret references."""
        for match in self._assignment.finditer(string):
            secret = match.group("secret")
            if secret.startswith(("$", "{", "<")):
                continue
            yield f"NVIDIA_API_KEY={secret}"
