"""Compile and test the actual Swift occlusion classifier without AX or GUI."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="Swift native guard requires macOS")
def test_native_occlusion_guard_exempts_only_the_system_dock_display_plane(tmp_path):
    compiler = shutil.which("swiftc")
    if not compiler:
        pytest.skip("Apple Command Line Tools are unavailable")
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "native-guard-tests"
    compilation = subprocess.run([
        compiler, "-parse-as-library", "-D", "CUA_JEV_NATIVE_TESTS",
        "-target", f"{platform.machine()}-apple-macosx14.0",
        "-module-cache-path", str(tmp_path / "module-cache"),
        str(root / "src/cua_jev/native/MacBridge.swift"),
        str(root / "tests/native/GuardTests.swift"), "-o", str(executable),
    ], check=False, capture_output=True, text=True, timeout=180)
    assert compilation.returncode == 0, compilation.stderr
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
    assert "Native occlusion guard cases passed" in result.stdout
