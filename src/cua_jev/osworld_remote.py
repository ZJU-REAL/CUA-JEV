"""Bounded SSH JSONL transport for an owned OSWorld worker.

Only desktop observations and typed input actions cross this channel. Model
clients and credentials stay with the local caller, outside this transport.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import queue
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .osworld_step_observation import (
    IMAGE_ONLY_INFO_KEY,
    IMAGE_ONLY_VERSION,
    validate_intermediate_action,
)

MAX_LINE = 48_000_000


class RemoteOSWorldError(RuntimeError):
    """Static transport error; never contains worker output or remote exceptions."""

    def __init__(self, message: str, *, code: str = "remote_error"):
        super().__init__(message)
        self.code = code


def _observation(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {"screenshot", "accessibility_tree"}:
        raise RemoteOSWorldError("remote worker returned an invalid observation", code="invalid_observation")
    encoded = value["screenshot"]
    tree = value["accessibility_tree"]
    if not isinstance(encoded, str) or len(encoded) > 44_000_000:
        raise RemoteOSWorldError("remote screenshot is missing or too large", code="invalid_observation")
    if tree is not None and (not isinstance(tree, str) or len(tree) > 2_000_000):
        raise RemoteOSWorldError("remote accessibility observation is invalid", code="invalid_observation")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise RemoteOSWorldError(
            "remote screenshot encoding is invalid", code="invalid_observation",
        ) from None
    if (
        not image
        or len(image) > 32_000_000
        or not (image.startswith(b"\x89PNG\r\n\x1a\n") or image.startswith(b"\xff\xd8\xff"))
    ):
        raise RemoteOSWorldError("remote screenshot format is invalid", code="invalid_observation")
    return {"screenshot": image, "accessibility_tree": tree}


def _settling_diagnostic(value: Any, observation: dict) -> dict | None:
    """Optional bounded worker evidence, bound to the exact ready observation."""
    if value is None:
        return None  # Older workers remain identifiable as unknown, not enabled.
    fixed = {
        "timeout_s": 30.0, "max_samples": 10,
        "min_stable_samples": 3, "min_stable_span_s": 4.0, "interval_s": 1.0,
        "application_ready_proven": False, "future_loading_excluded": False,
    }
    fields = {"enabled", "status", "reason", "samples_observed", "elapsed_ms", "stable_samples",
              "stable_span_ms", "surface_stability", "last_screenshot_sha256", "last_accessibility_sha256"}

    def invalid():
        raise RemoteOSWorldError("invalid passive settling diagnostic", code="invalid_handshake")

    version = value.get("version") if isinstance(value, dict) else None
    if not isinstance(version, str) or version not in {
        "public_structure_v1", "public_structure_v2", "public_structure_v3",
    }:
        invalid()
    fixed["version"] = version
    reserve_version = version in {"public_structure_v2", "public_structure_v3"}
    if reserve_version:
        fields.add("observation_reserve_ms")
    if not isinstance(value, dict) or set(value) != set(fixed) | fields:
        invalid()
    for key, expected in fixed.items():
        if value[key] != expected or isinstance(value[key], bool) is not isinstance(expected, bool):
            invalid()
    if type(value["enabled"]) is not bool:
        invalid()
    for key in ("samples_observed", "stable_samples"):
        if type(value[key]) is not int or not 0 <= value[key] <= 10:
            invalid()
    for key in ("elapsed_ms", "stable_span_ms"):
        if (type(value[key]) not in (int, float) or not math.isfinite(value[key])
                or not 0 <= value[key] <= 600_000):
            invalid()
    if reserve_version and (
        type(value["observation_reserve_ms"]) not in (int, float)
        or not math.isfinite(value["observation_reserve_ms"])
        or not 0 <= value["observation_reserve_ms"] <= 600_000
    ):
        invalid()
    if value["stable_samples"] > value["samples_observed"] or value["stable_span_ms"] > value["elapsed_ms"]:
        invalid()
    if not value["enabled"]:
        valid_state = (value["status"], value["reason"], value["surface_stability"]) == (
            "disabled", "disabled", "unproven") and value["samples_observed"] == 0
        valid_state = valid_state and (not reserve_version or value["observation_reserve_ms"] == 0)
    elif value["status"] == "settled":
        valid_state = (value["reason"] == "public_structure_stable"
                       and value["surface_stability"] == "observed_quiet"
                       and value["stable_samples"] >= 3 and value["stable_span_ms"] >= 4000
                       and value["elapsed_ms"] < 30_000)
    else:
        admitted_exhaustion = (
            reserve_version and value["reason"] == "observation_budget_reserve"
            and value["observation_reserve_ms"] > 0 and value["samples_observed"] >= 2
            and value["elapsed_ms"] + value["observation_reserve_ms"] >= 30_000
        )
        exhausted = (
            (value["reason"] == "sample_limit" and value["samples_observed"] == 10)
            or (value["reason"] == "timeout" and value["elapsed_ms"] >= 30_000)
            or admitted_exhaustion
        )
        valid_state = (value["status"] == "budget_exhausted" and exhausted
                       and value["surface_stability"] == "unproven" and value["samples_observed"] > 0)
    tree = observation["accessibility_tree"]
    if (not valid_state
            or value["last_screenshot_sha256"] != hashlib.sha256(observation["screenshot"]).hexdigest()
            or value["last_accessibility_sha256"] != (hashlib.sha256(tree.encode()).hexdigest()
                                                       if isinstance(tree, str) else None)):
        invalid()
    return dict(value)


class RemoteOSWorldEnv:
    """DesktopEnv subset backed by one SSH child, with explicit cleanup status."""

    def __init__(
        self,
        host: str,
        command: list[str] | tuple[str, ...],
        *,
        log_path: str | Path,
        timeout_s: float = 120,
        startup_timeout_s: float = 300,
        close_timeout_s: float = 45,
        popen_factory=subprocess.Popen,
    ):
        if (
            not isinstance(host, str)
            or len(host) > 253
            or not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*", host)
        ):
            raise ValueError("remote SSH host must be a hostname or IPv4 address without shell options")
        if (
            not isinstance(command, (list, tuple))
            or not command
            or any(not isinstance(token, str) or not token or "\x00" in token for token in command)
        ):
            raise ValueError("remote worker command must be a nonempty argument sequence")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 600
            for value in (timeout_s, startup_timeout_s, close_timeout_s)
        ):
            raise ValueError("remote worker timeouts must be finite and between zero and 600 seconds")
        self.timeout_s, self.close_timeout_s = timeout_s, close_timeout_s
        self.instruction = ""
        self.initial_observation: dict = {}
        self.passive_settling: dict | None = None
        self.supports_image_only_steps = False
        self.supports_screenshot_only = False
        self.cleanup_status = "not_requested"
        self._closed = False
        self._evaluated = False
        self._counter = 0
        self._lock = threading.Lock()
        self._lines: queue.Queue = queue.Queue(maxsize=4)
        self._process = None
        self._log = None
        startup_started = time.perf_counter()
        wait_started = None
        self.startup_diagnostics = {
            "error_code": None,
            "phase": "process_launch",
            "startup_timeout_s": startup_timeout_s,
            "startup_elapsed_ms": None,
            "wait_elapsed_ms": None,
            "handshake_received_elapsed_ms": None,
            "handshake_accepted_elapsed_ms": None,
        }
        destination = Path(log_path)
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ConnectTimeout=15",
            host,
            shlex.join(command),
        ]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._log = destination.open("ab")
            self._process = popen_factory(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                shell=False,
            )
            self._reader = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader.start()
            self.startup_diagnostics["phase"] = "ready_wait"
            wait_started = time.perf_counter()
            ready = self._receive(startup_timeout_s)
            received = time.perf_counter()
            self.startup_diagnostics.update(
                wait_elapsed_ms=(received - wait_started) * 1000,
                handshake_received_elapsed_ms=(received - startup_started) * 1000,
                phase="handshake_validation",
            )
            if ready.get("event") == "fatal":
                raise RemoteOSWorldError(
                    "remote worker initialization failed; see worker log", code="worker_fatal",
                )
            if (
                ready.get("event") != "ready"
                or ready.get("protocol") != 1
                or not isinstance(ready.get("instruction"), str)
                or not ready["instruction"].strip()
            ):
                raise RemoteOSWorldError(
                    "remote worker sent an invalid startup handshake", code="invalid_handshake",
                )
            self.startup_diagnostics["phase"] = "observation_validation"
            self.initial_observation = _observation(ready.get("observation"))
            self.passive_settling = _settling_diagnostic(
                ready.get("passive_settling"), self.initial_observation,
            )
            if "capabilities" in ready:
                capabilities = ready["capabilities"]
                if (not isinstance(capabilities, dict)
                        or not capabilities
                        or set(capabilities) - {"intermediate_fill_observation", "screenshot_only"}
                        or ("intermediate_fill_observation" in capabilities and
                            capabilities["intermediate_fill_observation"] != IMAGE_ONLY_VERSION)
                        or ("screenshot_only" in capabilities and
                            capabilities["screenshot_only"] != "screenshot_v1")):
                    raise RemoteOSWorldError(
                        "remote worker sent invalid observation capabilities", code="invalid_handshake",
                    )
                self.supports_image_only_steps = "intermediate_fill_observation" in capabilities
                self.supports_screenshot_only = "screenshot_only" in capabilities
            self.instruction = ready["instruction"]
            self.controller = SimpleNamespace(get_screenshot=self._get_screenshot)
            accepted_ms = (time.perf_counter() - startup_started) * 1000
            self.startup_diagnostics.update(
                phase="ready", startup_elapsed_ms=accepted_ms, handshake_accepted_elapsed_ms=accepted_ms,
            )
        except Exception as error:
            failed_at = time.perf_counter()
            diagnostic = self.startup_diagnostics
            diagnostic["startup_elapsed_ms"] = (failed_at - startup_started) * 1000
            if wait_started is not None and diagnostic["wait_elapsed_ms"] is None:
                diagnostic["wait_elapsed_ms"] = (failed_at - wait_started) * 1000
            if not isinstance(error, RemoteOSWorldError):
                error = RemoteOSWorldError("remote worker startup failed", code="startup_process_error")
            diagnostic["error_code"] = error.code
            self._closed = True
            clean = False
            cleanup_started = time.perf_counter()
            try:
                clean = self._finish_process()
            except Exception:
                diagnostic["cleanup_error_code"] = "process_cleanup_error"
            # An SSH exit alone never proves that a remote VM was cleaned up.
            if self.cleanup_status != "unconfirmed_forced_termination":
                self.cleanup_status = "unconfirmed_startup"
            diagnostic.update(
                cleanup_elapsed_ms=(time.perf_counter() - cleanup_started) * 1000,
                ssh_returncode=self._process.returncode if self._process is not None else None,
                ssh_exit_confirmed=self._process is not None and self._process.returncode is not None,
                ssh_exit_success=clean,
                cleanup_status=self.cleanup_status,
                resource_cleanup_confirmed=False,
            )
            error.startup_diagnostics = dict(diagnostic)
            raise error from None

    def _read_stdout(self) -> None:
        try:
            while True:
                line = self._process.stdout.readline(MAX_LINE + 1)
                if not line:
                    self._lines.put(("eof", None))
                    return
                if len(line) > MAX_LINE or not line.endswith("\n"):
                    self._lines.put(("invalid", None))
                    return
                self._lines.put(("line", line))
        except Exception:
            self._lines.put(("invalid", None))

    def _receive(self, timeout: float) -> dict:
        try:
            kind, line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise RemoteOSWorldError("remote worker response timed out", code="response_timeout") from None
        if kind == "eof":
            raise RemoteOSWorldError(
                "remote worker closed its output unexpectedly; see worker log", code="unexpected_eof",
            )
        if kind != "line":
            raise RemoteOSWorldError("remote worker output is not valid bounded JSONL", code="invalid_jsonl")
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            raise RemoteOSWorldError("remote worker output is not valid JSON", code="invalid_json") from None
        if not isinstance(value, dict):
            raise RemoteOSWorldError("remote worker response must be an object", code="invalid_response")
        return value

    def _rpc(self, method: str, *, timeout: float | None = None, **parameters):
        with self._lock:
            if self._closed:
                raise RemoteOSWorldError("remote worker is closed")
            if self._evaluated and method != "close":
                raise RemoteOSWorldError("remote episode has already been evaluated")
            self._counter += 1
            request_id = self._counter
            request = {"id": request_id, "method": method, **parameters}
            if method == "evaluate":
                # An uncertain response must not cause a second evaluation.
                self._evaluated = True
            try:
                self._process.stdin.write(json.dumps(request, ensure_ascii=True, allow_nan=False) + "\n")
                self._process.stdin.flush()
            except (OSError, ValueError):
                raise RemoteOSWorldError("remote worker input is unavailable") from None
            response = self._receive(self.timeout_s if timeout is None else timeout)
            if type(response.get("id")) is not int or response["id"] != request_id:
                raise RemoteOSWorldError("remote worker response does not match its request")
            if response.get("ok") is not True:
                raise RemoteOSWorldError(f"remote worker rejected {method}; see worker log")
            if "result" not in response:
                raise RemoteOSWorldError("remote worker response has no result")
            return response["result"]

    def _get_obs(self) -> dict:
        return _observation(self._rpc("observe"))

    def _get_screenshot(self) -> bytes:
        if self.supports_screenshot_only:
            return _observation({"screenshot": self._rpc("screenshot"), "accessibility_tree": None})[
                "screenshot"
            ]
        return self._get_obs()["screenshot"]

    def step(self, action: dict, pause: float = 0.5) -> tuple:
        return self._step(action, pause=pause, image_only=False)

    def step_image_only(self, action: dict, pause: float = 0.5) -> tuple:
        """Request only negotiated intermediate fill screenshots; never retry input."""
        if self.supports_image_only_steps is not True:
            raise ValueError("remote worker does not support image-only fill steps")
        validate_intermediate_action(action)
        return self._step(action, pause=pause, image_only=True)

    def _step(self, action: dict, *, pause: float, image_only: bool) -> tuple:
        if not isinstance(action, dict) or set(action) != {"action_type", "parameters"}:
            raise ValueError("remote input requires one typed computer_13 action")
        if type(pause) not in (int, float) or not math.isfinite(pause) or not 0 <= pause <= 10:
            raise ValueError("remote action pause is outside its allowed bounds")
        options = {"observation_mode": IMAGE_ONLY_VERSION} if image_only else {}
        result = self._rpc("step", action=action, pause=pause, **options)
        if (
            not isinstance(result, dict)
            or set(result) != {"observation", "reward", "done", "info"}
            or type(result["reward"]) not in (int, float)
            or not math.isfinite(result["reward"])
            or type(result["done"]) is not bool
            or not isinstance(result["info"], dict)
        ):
            raise RemoteOSWorldError("remote worker returned an invalid step result")
        observation = _observation(result["observation"])
        if image_only:
            if (result["info"] != {IMAGE_ONLY_INFO_KEY: IMAGE_ONLY_VERSION}
                    or observation["accessibility_tree"] is not None):
                raise RemoteOSWorldError("remote worker returned an invalid image-only step result")
        elif IMAGE_ONLY_INFO_KEY in result["info"]:
            raise RemoteOSWorldError("remote worker returned an unexpected image-only step result")
        return observation, result["reward"], result["done"], result["info"]

    def evaluate(self) -> float:
        value = self._rpc("evaluate")
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RemoteOSWorldError("remote worker returned an invalid evaluation score")
        return float(value)

    def _finish_process(self) -> bool:
        clean = False
        process = self._process
        try:
            if process is not None:
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    clean = process.wait(timeout=self.close_timeout_s) == 0
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    self.cleanup_status = "unconfirmed_forced_termination"
                reader = getattr(self, "_reader", None)
                if reader is not None:
                    reader.join(timeout=0.2)
                if process.stdout is not None and (reader is None or not reader.is_alive()):
                    process.stdout.close()
        finally:
            if self._log is not None:
                self._log.close()
        return clean

    def close(self) -> None:
        if self._closed:
            return
        acknowledged = False
        try:
            self._rpc("close", timeout=self.close_timeout_s)
            acknowledged = True
        except RemoteOSWorldError:
            pass
        finally:
            self._closed = True
            clean = self._finish_process()
        if acknowledged and clean:
            self.cleanup_status = "confirmed"
        else:
            if self.cleanup_status != "unconfirmed_forced_termination":
                self.cleanup_status = "unconfirmed_worker_exit"
            raise RemoteOSWorldError(
                "remote worker cleanup was not confirmed; inspect worker log and owned VM"
            )
