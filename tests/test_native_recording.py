"""Exercise native recording deadlines and callbacks without capture permission."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift lifecycle tests require macOS")
def test_native_recording_lifecycle_is_bounded_and_preserves_failures(tmp_path):
    compiler = shutil.which("swiftc")
    if not compiler:
        pytest.skip("Apple Command Line Tools are unavailable")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "native-recording-tests"
    compilation = subprocess.run([
        compiler, "-parse-as-library", "-D", "CUA_JEV_NATIVE_TESTS",
        "-target", f"{platform.machine()}-apple-macosx14.0",
        "-module-cache-path", str(tmp_path / "module-cache"),
        str(root / "src/cua_jev/native/MacBridge.swift"),
        str(root / "tests/native/RecordingTests.swift"), "-o", str(executable),
    ], check=False, capture_output=True, text=True, timeout=180)
    assert compilation.returncode == 0, compilation.stderr
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
    assert "Native recording lifecycle cases passed" in result.stdout
