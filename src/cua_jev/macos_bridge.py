"""Bounded JSON transport to the repository-owned native macOS helper.

Swift is compiled locally with Apple's Command Line Tools. The helper has a
small fixed command set; model text never becomes native code or shell input.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import platform
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .errors import CapabilityUnavailable
from .vision import WindowImage

_MAX_REQUEST_BYTES = 32_000
_MAX_RESPONSE_BYTES = 90_000_000
_MAX_CAPTURE_PIXELS = 16_000_000


def native_binary() -> Path:
    if sys.platform != "darwin":
        raise CapabilityUnavailable("the macOS native backend requires macOS")
    if int(platform.mac_ver()[0].split(".")[0]) < 14:
        raise CapabilityUnavailable("the native backend requires macOS 14 or newer")
    compiler = shutil.which("swiftc")
    if not compiler:
        raise CapabilityUnavailable(
            "Apple Command Line Tools are required (install with xcode-select --install)"
        )
    source = Path(__file__).parent / "native" / "MacBridge.swift"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path.home() / "Library" / "Caches" / "cua-jev" / "macos"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "cua-jev-mac"
    stamp = cache / "source.sha256"
    import fcntl

    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_file() and stamp.is_file() and stamp.read_text() == digest:
            return target
        descriptor, temporary = tempfile.mkstemp(prefix="bridge-", dir=cache)
        os.close(descriptor)
        try:
            result = subprocess.run(
                [compiler, "-parse-as-library", "-target", f"{platform.machine()}-apple-macosx14.0",
                 str(source), "-o", temporary],
                capture_output=True, text=True, timeout=180, check=False,
            )
            if result.returncode:
                raise CapabilityUnavailable(f"macOS helper compilation failed:\n{result.stderr[-6000:]}")
            os.chmod(temporary, 0o700)
            os.replace(temporary, target)
            stamp.write_text(digest)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target


class MacBridge:
    """One subprocess, one pinned window. Calls serialize capture and actions."""

    def __init__(self, *, binary: Path | None = None, timeout_s: float = 30) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("native request timeout must be positive and finite")
        self.binary = binary
        self.timeout_s = timeout_s
        self.process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._buffer = b""
        self.last_recording_diagnostics: dict[str, Any] | None = None

    def call(self, command: str, **arguments: Any) -> dict[str, Any]:
        with self._lock:
            self.last_recording_diagnostics = None
            payload = json.dumps(
                {"command": command, **arguments}, ensure_ascii=False, allow_nan=False,
            ).encode()
            if len(payload) > _MAX_REQUEST_BYTES:
                raise ValueError("native request exceeds its size bound")
            if self.process is None:
                self.process = subprocess.Popen(
                    [str(self.binary or native_binary())], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
                )
            process = self.process
            assert process.stdin is not None and process.stdout is not None
            try:
                deadline = time.monotonic() + self.timeout_s
                pending = memoryview(payload + b"\n")
                # A helper stalled before readLine must not block a full pipe
                # before the response timeout even starts. Short writes are legal.
                os.set_blocking(process.stdin.fileno(), False)
                os.set_blocking(process.stdout.fileno(), False)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while pending or b"\n" not in self._buffer:
                        remaining = deadline - time.monotonic()
                        ready = selector.select(max(0, remaining))
                        if remaining <= 0 or not ready:
                            raise RuntimeError("macOS native request timed out")
                        for key, events in ready:
                            try:
                                if events & selectors.EVENT_WRITE:
                                    written = os.write(key.fd, pending)
                                    if not written:
                                        raise RuntimeError("macOS helper stopped accepting requests")
                                    pending = pending[written:]
                                    if not pending:
                                        selector.unregister(process.stdin)
                                else:
                                    chunk = os.read(key.fd, 65536)
                                    if not chunk:
                                        raise RuntimeError("macOS helper exited before responding")
                                    self._buffer += chunk
                                    if len(self._buffer) > _MAX_RESPONSE_BYTES:
                                        raise RuntimeError("native response exceeds its size bound")
                            except BlockingIOError:
                                continue
                line, self._buffer = self._buffer.split(b"\n", 1)
                response = json.loads(line)
            except (OSError, ValueError, RuntimeError):
                self._close(graceful=False)
                raise
            if not isinstance(response, dict):
                self._close(graceful=False)
                raise RuntimeError("invalid macOS native response")
            if response.get("ok") is not True:
                diagnostics = response.get("recording_diagnostics")
                if isinstance(diagnostics, dict):
                    self.last_recording_diagnostics = diagnostics
                raise CapabilityUnavailable(str(response.get("error", "macOS native request failed")))
            if not isinstance(response.get("result"), dict):
                self._close(graceful=False)
                raise RuntimeError("invalid macOS native result")
            return response["result"]

    def capture_image(self) -> WindowImage:
        from PIL import Image

        data = self.call("capture")
        bounds = data.get("bounds")
        if (not isinstance(bounds, list) or len(bounds) != 4 or
                not all(type(value) in (int, float) and math.isfinite(value) for value in bounds) or
                bounds[2] < 1 or bounds[3] < 1):
            raise RuntimeError("window capture has invalid point bounds")
        raw = base64.b64decode(data["png"], validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if (source.format != "PNG" or source.width * source.height > _MAX_CAPTURE_PIXELS or
                    (source.width, source.height) != (data.get("pixel_width"), data.get("pixel_height"))):
                raise RuntimeError("window capture has invalid pixel dimensions or format")
            image = source.convert("RGB")
        width, height = bounds[2:]
        if abs(image.width / width - image.height / height) > 0.05:
            raise RuntimeError("window capture has inconsistent pixel-to-point scale")
        sample = image.convert("L").resize((64, 64)).tobytes()
        image.thumbnail((1280, 1280))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72)
        return WindowImage(buffer.getvalue(), sample, tuple(round(n) for n in bounds))

    def close(self) -> None:
        with self._lock:
            self._close(graceful=True)

    def _close(self, *, graceful: bool) -> None:
        process, self.process = self.process, None
        self._buffer = b""
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except OSError:
                pass
        if not graceful:
            process.terminate()
        try:
            process.wait(timeout=15 if graceful else 3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()
