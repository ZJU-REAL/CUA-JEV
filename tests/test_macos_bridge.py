"""Exercise the actual pipe transport without macOS permissions or a GUI."""

import base64
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cua_jev import macos_bridge
from cua_jev.errors import CapabilityUnavailable
from cua_jev.macos_bridge import MacBridge


@pytest.fixture
def helper(monkeypatch):
    if os.name == "nt":
        pytest.skip("native macOS transport uses POSIX selectable pipes")
    launch = subprocess.Popen
    bridges = []

    def create(program, *, timeout_s=2):
        def start(*args, **kwargs):
            return launch([sys.executable, "-u", "-c", program], **kwargs)

        monkeypatch.setattr(macos_bridge.subprocess, "Popen", start)
        bridge = MacBridge(binary=Path("unused-test-helper"), timeout_s=timeout_s)
        bridges.append(bridge)
        return bridge

    yield create
    for bridge in bridges:
        bridge.close()


def test_transport_handles_short_writes_and_fragmented_responses(helper, monkeypatch):
    write = os.write
    monkeypatch.setattr(macos_bridge.os, "write", lambda fd, data: write(fd, data[:7]))
    bridge = helper("""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    answer = json.dumps({"ok": True, "result": {"text": request["text"]}}) + "\\n"
    for character in answer:
        sys.stdout.write(character)
        sys.stdout.flush()
""")
    value = "A note 中文 🎉" * 20
    assert bridge.call("echo", text=value) == {"text": value}
    assert bridge.call("echo", text="second request") == {"text": "second request"}


@pytest.mark.parametrize("request_size", [1, 30_000])
def test_deadline_covers_stalled_reader_and_response(helper, request_size):
    bridge = helper("import time; time.sleep(60)", timeout_s=0.15)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        bridge.call("echo", text="x" * request_size)
    assert time.monotonic() - started < 3
    assert bridge.process is None


@pytest.mark.parametrize("reply", ["not json", "[]", '{"ok":true,"result":[]}'])
def test_malformed_response_closes_transport(helper, reply):
    bridge = helper(f"import sys; sys.stdin.readline(); print({reply!r}, flush=True)")
    with pytest.raises((ValueError, RuntimeError)):
        bridge.call("health")
    assert bridge.process is None


def test_response_size_limit_is_enforced_before_json_decoding(helper, monkeypatch):
    monkeypatch.setattr(macos_bridge, "_MAX_RESPONSE_BYTES", 128)
    bridge = helper("import sys; sys.stdin.readline(); print('x' * 1024, flush=True)")
    with pytest.raises(RuntimeError, match="size bound"):
        bridge.call("health")
    assert bridge.process is None


def test_helper_exit_has_a_bounded_failure(helper):
    bridge = helper("import sys; sys.stdin.readline()")
    with pytest.raises(RuntimeError, match="exited"):
        bridge.call("health")
    assert bridge.process is None


def test_native_command_error_does_not_desynchronize_next_response(helper):
    bridge = helper("""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    answer = ({"ok": False, "error": "permission denied"} if request["command"] == "select"
              else {"ok": True, "result": {"accessibility": False}})
    print(json.dumps(answer), flush=True)
""")
    with pytest.raises(CapabilityUnavailable, match="permission denied"):
        bridge.call("select")
    assert bridge.call("health") == {"accessibility": False}


@pytest.mark.parametrize("value", ["x" * 32_000, float("nan")])
def test_invalid_request_does_not_start_a_helper(value):
    bridge = MacBridge(binary=Path("must-not-run"))
    with pytest.raises(ValueError):
        bridge.call("echo", value=value)
    assert bridge.process is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match="timeout"):
        MacBridge(timeout_s=timeout)


def test_close_releases_the_child_and_is_idempotent(helper):
    bridge = helper("""
import json, sys
for line in sys.stdin:
    print(json.dumps({"ok": True, "result": {}}), flush=True)
""")
    assert bridge.call("health") == {}
    process = bridge.process
    bridge.close()
    bridge.close()
    assert bridge.process is None and process.poll() == 0
    assert process.stdin.closed and process.stdout.closed


def capture_payload():
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (8, 4), color="#f0f0f0").save(output, format="PNG")
    return {
        "png": base64.b64encode(output.getvalue()).decode(), "bounds": [-20, 10, 4, 2],
        "pixel_width": 8, "pixel_height": 4,
    }


@pytest.mark.parametrize("bounds", [[], [0, 0, 0, 2], [0, 0, -1, 2], [0, 0, float("nan"), 2],
                                    [True, 0, 4, 2], [0, 0, "4", 2]])
def test_capture_rejects_invalid_point_geometry(monkeypatch, bounds):
    bridge = MacBridge()
    data = {**capture_payload(), "bounds": bounds}
    monkeypatch.setattr(bridge, "call", lambda command: data)
    with pytest.raises(RuntimeError, match="invalid point bounds"):
        bridge.capture_image()


def test_capture_checks_decoded_dimensions_before_conversion(monkeypatch):
    bridge = MacBridge()
    data = capture_payload()
    monkeypatch.setattr(bridge, "call", lambda command: data)
    monkeypatch.setattr(macos_bridge, "_MAX_CAPTURE_PIXELS", 16)
    with pytest.raises(RuntimeError, match="pixel dimensions"):
        bridge.capture_image()
    monkeypatch.setattr(macos_bridge, "_MAX_CAPTURE_PIXELS", 32)
    data["pixel_width"] = 16
    with pytest.raises(RuntimeError, match="pixel dimensions"):
        bridge.capture_image()


def test_capture_preserves_point_coordinates_and_validates_scale(monkeypatch):
    bridge = MacBridge()
    data = capture_payload()
    monkeypatch.setattr(bridge, "call", lambda command: data)
    result = bridge.capture_image()
    assert result.region == (-20, 10, 4, 2)
    data["bounds"][2] = 8
    with pytest.raises(RuntimeError, match="pixel-to-point scale"):
        bridge.capture_image()
