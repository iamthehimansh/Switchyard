# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified Python/Rust v2 profile loader.

The headline behavior: one config file builds and runs *both* a Rust-defined
profile and a Python-defined profile through the same loading path
(``load_profiles``), sharing one resolved set of targets.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from switchyard import load_profiles
from switchyard.lib.profiles import ProfileConfigError
from switchyard_rust.core import ChatRequest, SwitchyardConfigError
from switchyard_rust.profiles import ProfileInput, ProfileRequestMetadata


class _MockOpenAIServer(ThreadingHTTPServer):
    calls: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _MockOpenAIHandler)
        self.calls = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    server: _MockOpenAIServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(content_length)
        body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        self.server.calls.append({"path": self.path, "body": body})
        response = {
            "id": "chatcmpl-loader",
            "object": "chat.completion",
            "model": body.get("model"),
            "mock_path": self.path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.fixture
def mock_openai_server() -> _MockOpenAIServer:
    try:
        server = _MockOpenAIServer()
    except PermissionError as exc:
        pytest.skip(f"loopback socket binding is unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _mixed_config(base_url: str) -> str:
    """A config with a Rust passthrough and the Python header-routing profile."""
    return f"""
targets:
  strong:
    model: provider/strong
    format: openai
    base_url: "{base_url}/strong/v1"
    api_key: test-key
  weak:
    model: provider/weak
    format: openai
    base_url: "{base_url}/weak/v1"
    api_key: test-key

profiles:
  direct:
    type: passthrough
    target: weak
  smart:
    type: header-routing
    strong: strong
    weak: weak
"""


def _write(tmp_path: Path, text: str, name: str = "profiles.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _request(model: str) -> ChatRequest:
    return ChatRequest.openai_chat(
        {"model": model, "messages": [{"role": "user", "content": "hello"}]}
    )


def test_load_profiles_builds_rust_and_python_profiles(tmp_path: Path) -> None:
    path = _write(tmp_path, _mixed_config("http://127.0.0.1:9"))
    profiles = load_profiles(path)
    # One Rust-defined profile (passthrough) and one Python-defined profile
    # (header-routing) built through the same path.
    assert sorted(profiles) == ["direct", "smart"]
    assert all(hasattr(runner, "run") for runner in profiles.values())


async def test_loaded_rust_and_python_profiles_run_against_mock(
    mock_openai_server: _MockOpenAIServer,
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _mixed_config(mock_openai_server.base_url))
    profiles = load_profiles(path)

    # Rust passthrough -> weak target.
    direct = await profiles["direct"].run(ProfileInput(_request("client/x")))
    assert direct.body["model"] == "provider/weak"
    assert direct.body["mock_path"] == "/weak/v1/chat/completions"

    # Python header-routing: header selects strong, delegating to the Rust backend.
    strong_meta = ProfileRequestMetadata(headers={"x-switchyard-tier": "strong"})
    strong = await profiles["smart"].run(ProfileInput(_request("client/x"), strong_meta))
    assert strong.body["model"] == "provider/strong"
    assert strong.body["mock_path"] == "/strong/v1/chat/completions"

    # Missing/other header falls back to weak.
    weak = await profiles["smart"].run(ProfileInput(_request("client/x")))
    assert weak.body["model"] == "provider/weak"
    assert weak.body["mock_path"] == "/weak/v1/chat/completions"


def test_load_profiles_accepts_json(tmp_path: Path) -> None:
    document = {
        "targets": {
            "weak": {
                "model": "provider/weak",
                "format": "openai",
                "base_url": "http://127.0.0.1:9/weak/v1",
                "api_key": "test-key",
            }
        },
        "profiles": {"direct": {"type": "passthrough", "target": "weak"}},
    }
    path = _write(tmp_path, json.dumps(document), name="profiles.json")
    assert sorted(load_profiles(path)) == ["direct"]


def test_unknown_profile_type_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "profiles:\n  bad:\n    type: does-not-exist\n",
    )
    with pytest.raises(SwitchyardConfigError, match="does-not-exist"):
        load_profiles(path)


def test_python_profile_missing_target_is_rejected(tmp_path: Path) -> None:
    config = """
targets:
  weak:
    model: provider/weak
    format: openai
    base_url: http://127.0.0.1:9/weak/v1
    api_key: test-key

profiles:
  smart:
    type: header-routing
    strong: ghost
    weak: weak
"""
    path = _write(tmp_path, config)
    with pytest.raises(ProfileConfigError, match="unknown target 'ghost'"):
        load_profiles(path)


def test_python_profile_unknown_field_is_rejected(tmp_path: Path) -> None:
    config = """
targets:
  weak:
    model: provider/weak
    format: openai
    base_url: http://127.0.0.1:9/weak/v1
    api_key: test-key

profiles:
  smart:
    type: header-routing
    strong: weak
    weak: weak
    bogus: 1
"""
    path = _write(tmp_path, config)
    with pytest.raises(ProfileConfigError, match="unknown field"):
        load_profiles(path)


def test_unknown_format_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "profiles: {}\n", name="profiles.ini")
    with pytest.raises(SwitchyardConfigError, match="unsupported profile config extension"):
        load_profiles(path)
