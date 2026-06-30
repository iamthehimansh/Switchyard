# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for Switchyard examples.

This module provides common configuration loading for all examples.
"""

import json
import os
from pathlib import Path

# Repository root (parent of examples folder)
REPO_ROOT = Path(__file__).parent.parent

# Path to secrets file
SECRETS_FILE = REPO_ROOT / "secrets" / "secrets.json"

# Defaults for OpenAI API
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def load_secrets() -> dict:
    """
    Load secrets from secrets/secrets.json if it exists.

    Returns:
        Dictionary with secrets, or empty dict if file doesn't exist.
    """
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE) as f:
            return json.load(f)
    return {}


def get_config() -> dict:
    """
    Get API configuration from environment variables or secrets file.

    Priority:
        1. Environment variables (OPENAI_API_KEY, OPENAI_BASE_URL)
        2. Secrets file (secrets/secrets.json)
        3. Defaults for base_url and model

    Returns:
        Dictionary with api_key, base_url, and model.
    """
    secrets = load_secrets()
    openai_secrets = secrets.get("openai", {})

    return {
        "api_key": os.environ.get("OPENAI_API_KEY") or openai_secrets.get("api_key"),
        "base_url": (
            os.environ.get("OPENAI_BASE_URL")
            or openai_secrets.get("base_url")
            or DEFAULT_BASE_URL
        ),
        "model": (
            os.environ.get("OPENAI_MODEL")
            or openai_secrets.get("model")
            or DEFAULT_MODEL
        ),
    }


def print_config(config: dict) -> None:
    """Print configuration in a consistent format."""
    api_key = config["api_key"]
    print("Configuration:")
    if api_key:
        print(f"  API Key: {'*' * 8}...{api_key[-4:]}")
    else:
        print("  API Key: Not set")
    print(f"  Base URL: {config['base_url']}")
    print(f"  Model: {config['model']}")
    print()


def check_api_key(config: dict) -> bool:
    """
    Check if API key is available and print error message if not.

    Returns:
        True if API key is available, False otherwise.
    """
    if config["api_key"]:
        return True

    print("Error: API key not found.")
    print()
    print("Set it via environment variable:")
    print("    export OPENAI_API_KEY='sk-...'")
    print()
    print("Or create secrets/secrets.json (copy from secrets/secrets.template.json)")
    return False
