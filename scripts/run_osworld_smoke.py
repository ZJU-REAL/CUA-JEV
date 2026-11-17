"""Run real VLM/Jev episodes in new, owned OSWorld Docker VMs.

The official evaluator runs once after the agent stops; its task configuration
and score are never inputs to either policy. Existing VMs and environments are
not modified. Docker writes go to a private, bounded /storage tmpfs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import (
    STALE_RECOVERY_PROGRESS_VERSION,
    STALE_RECOVERY_VERSION,
    AgentConfig,
    AgentRunner,
)
from cua_jev.config import load_local_env
from cua_jev.osworld_backend import OBSERVATION_BUDGET_VERSION
from cua_jev.osworld_remote import RemoteOSWorldEnv, RemoteOSWorldError
from cua_jev.policy import JevPolicy
from cua_jev.trace import JsonlTrace


class SetupFailed(RuntimeError):
    pass


class VMStartupError(RuntimeError):
    def __init__(self, diagnostics):
        super().__init__("owned OSWorld VM failed to start")
        self.diagnostics = diagnostics


class _ReadinessDeadline(BaseException):
    """Escape upstream observation retries, including ``except Exception``."""


class _ReadinessBudgetUnavailable(RuntimeError):
    pass


def _only_pointer_shape(foreground, bounds):
    """Recognize a small arrow, rather than reject a one-character terminal."""
    if bounds is None:
        return True
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if not (7 <= width <= 64 and 12 <= height <= 64 and 1.2 * width <= height <= 3 * width):
        return False
    crop = foreground.crop(bounds)
    rows = [crop.crop((0, y, width, y + 1)).getbbox() for y in range(min(width, height * 2 // 3))]
    if not rows or any(row is None for row in rows):
        return False
    widths = [row[2] - row[0] for row in rows]
    fixed_edge = any(
        max(row[edge] for row in rows) - min(row[edge] for row in rows) <= max(1, width // 12)
        for edge in (0, 2)
    )
    return (
        fixed_edge and widths[0] <= width / 4 and widths[-1] >= width * 0.8
        and all(later >= earlier - 1 for earlier, later in zip(widths, widths[1:], strict=False))
    )


def _dark_pointer_evidence(rgb):
    """Recognize an arrow and its tiny shadow on an otherwise uniform dark frame.

    A flat gray frame alone is deliberately insufficient. Exact differences
    outside the pointer's two-pixel halo preserve even faint terminal text.
    """
    from PIL import Image, ImageChops

    histogram = rgb.histogram()
    background = tuple(
        max(range(256), key=lambda value: histogram[channel * 256 + value])
        for channel in range(3)
    )
    if not 8 < max(background) <= 64:
        return None
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, background))
    red, green, blue = difference.split()
    contrast = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    pointer = contrast.point(lambda value: 255 if value > 8 else 0)
    bounds, all_bounds = pointer.getbbox(), contrast.getbbox()
    if bounds is None or not _only_pointer_shape(pointer, bounds):
        return None
    if not (
        bounds[0] - 2 <= all_bounds[0] and bounds[1] - 2 <= all_bounds[1]
        and all_bounds[2] <= bounds[2] + 2 and all_bounds[3] <= bounds[3] + 2
    ):
        return None
    # Geometry, rather than a global sparse-pixel percentage, admits the small
    # anti-aliased shadow of a real cursor without discarding a short prompt.
    return {
        "dark_background_rgb": list(background),
        "dark_background_max_channel": 64,
        "dark_nonbackground_pixels": rgb.width * rgb.height - contrast.histogram()[0],
        "dark_nonbackground_bounds": list(all_bounds),
        "dark_pointer_pixels": pointer.histogram()[255],
        "dark_pointer_bounds": list(bounds),
        "dark_pointer_min_contrast": 8,
    }


def desktop_observation_quality(observation):
    """Detect narrow boot placeholders, without treating AT-SPI as a screenshot."""
    from PIL import Image, ImageChops

    quality = {"reason": "observation_unavailable", "a11y_status": "unavailable", "a11y_nodes": 0}
    if not isinstance(observation, dict):
        return quality
    tree = observation.get("accessibility_tree")
    if isinstance(tree, str) and tree:
        if len(tree) > 2_000_000 or "<!DOCTYPE" in tree.upper() or "<!ENTITY" in tree.upper():
            quality["a11y_status"] = "unavailable"
        else:
            try:
                root = ElementTree.fromstring(tree)
                quality["a11y_nodes"] = sum(1 for _ in root.iter())
                quality["a11y_status"] = "populated" if len(root) else "empty"
            except (ElementTree.ParseError, ValueError, RecursionError):
                quality["a11y_status"] = "invalid"
    screenshot = observation.get("screenshot")
    if not isinstance(screenshot, bytes) or not 0 < len(screenshot) <= 20_000_000:
        return quality
    try:
        with Image.open(io.BytesIO(screenshot)) as image:
            width, height = image.size
            if image.format != "PNG" or not 0 < width * height <= 16_000_000:
                return quality
            rgb = image.convert("RGB")
            red, green, blue = rgb.split()
            # Full-resolution evidence retains small terminal text. A 99% or
            # thumbnail test would incorrectly discard legitimate dark UIs.
            brightest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
            foreground = brightest.point(lambda value: 255 if value > 8 else 0)
            count = foreground.histogram()[255]
            bounds = foreground.getbbox()
            dark_pointer = _dark_pointer_evidence(rgb)
    except (OSError, ValueError, Image.DecompressionBombError):
        return quality
    pixels = width * height
    # Extremely sparse content alone is insufficient: even a short shell prompt
    # is usable evidence. Only empty black or a recognizable tiny pointer waits.
    placeholder = count / pixels <= 0.0001 and _only_pointer_shape(foreground, bounds)
    quality.update(
        reason="black_placeholder" if placeholder else "visual_evidence_present",
        width=width, height=height, foreground_pixels=count,
        near_black_max_channel=8,
        near_black_fraction=(pixels - count) / pixels,
        foreground_bounds=list(bounds) if bounds is not None else None,
    )
    if not placeholder and dark_pointer is not None:
        quality.update(dark_pointer, reason="dark_pointer_placeholder")
    return quality


def _readiness_observe(env, remaining):
    """A bounded observation, with no background thread racing VM cleanup."""
    if hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread():
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise _ReadinessBudgetUnavailable("budget_conflict")
        previous = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise _ReadinessDeadline()

        signal.signal(signal.SIGALRM, expired)
        try:
            signal.setitimer(signal.ITIMER_REAL, remaining)
            return env._get_obs()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    # The Windows local driver normally receives an already-ready worker frame.
    # Only our known queue-bounded RPC transport may re-observe on this path.
    from cua_jev.osworld_remote import RemoteOSWorldEnv

    if not isinstance(env, RemoteOSWorldEnv):
        raise _ReadinessBudgetUnavailable("bounded_observer_unavailable")
    previous = env.timeout_s
    env.timeout_s = min(previous, remaining)
    try:
        return env._get_obs()
    finally:
        env.timeout_s = previous


def wait_for_desktop_ready(env, initial, *, output, report, timeout_s=30.0, max_attempts=8,
                           interval_s=1.0):
    """Reuse reset's frame; wait only for missing/placeholder visual evidence."""
    if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60
            or type(max_attempts) is not int or not 1 <= max_attempts <= 16
            or type(interval_s) not in (int, float) or not math.isfinite(interval_s)
            or not 0 <= interval_s <= 5):
        raise ValueError("invalid desktop readiness budget")
    started = time.monotonic()
    observation = initial
    diagnostic = {
        "status": "waiting", "reason": "", "attempts": 1, "elapsed_ms": 0.0,
        "timeout_s": timeout_s, "max_attempts": max_attempts, "interval_s": interval_s,
        "samples": [],
    }
    report["desktop_readiness"] = diagnostic

    def finish(status, reason):
        diagnostic.update(status=status, reason=reason, elapsed_ms=(time.monotonic() - started) * 1000)

    while True:
        quality = desktop_observation_quality(observation)
        diagnostic["samples"].append(quality)
        diagnostic["last_quality"] = quality
        if quality["reason"] == "visual_evidence_present":
            finish("ready", "visual_evidence_present")
            return observation
        if (hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
                and signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)):
            finish("failed", "budget_conflict")
            break
        if diagnostic["attempts"] == 1:
            save_observation_artifacts(output, "readiness-first", observation, report)
        if diagnostic["attempts"] >= max_attempts:
            finish("failed", "attempt_limit")
            break
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            finish("failed", "timeout")
            break
        time.sleep(min(interval_s, remaining))
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            finish("failed", "timeout")
            break
        diagnostic["attempts"] += 1
        try:
            observation = _readiness_observe(env, remaining)
        except _ReadinessDeadline:
            finish("failed", "timeout")
            break
        except _ReadinessBudgetUnavailable as error:
            finish("failed", str(error))
            break
        except Exception as error:
            diagnostic["observation_error_type"] = type(error).__name__
            finish("failed", "observation_error")
            break
        if time.monotonic() - started >= timeout_s:
            finish("failed", "timeout")
            break
    save_observation_artifacts(output, "readiness-last", observation, report)
    raise VMStartupError({"desktop_readiness": diagnostic})


PASSIVE_SETTLING_VERSION = "public_structure_v3"
PASSIVE_SETTLING_BOUNDS = {
    "timeout_s": 30.0, "max_samples": 10, "min_stable_samples": 3,
    "min_stable_span_s": 4.0, "interval_s": 1.0,
}


def _settling_structure(observation, quality):
    """A bounded public-surface signature, never a setup/application-ready test."""
    from cua_jev.osworld_backend import _accessibility_observation

    if quality["reason"] != "visual_evidence_present":
        return None, False, 0
    controls, metadata = [], {}
    title, _text, semantic = _accessibility_observation(
        observation.get("accessibility_tree"), screen_size=(quality["width"], quality["height"]),
        controls_out=controls, metadata=metadata,
    )
    if not metadata["complete"] or not controls:
        return None, metadata["complete"], len(controls)
    # Reuse only the parser's narrow clock normalization. Loading text, values,
    # focus, geometry and control topology remain significant; animation can
    # therefore exhaust this heuristic without making a usable desktop fail.
    body = [title, semantic, [control.to_dict() for control in controls]]
    signature = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return signature, True, len(controls)


def settling_summary(diagnostic):
    return {key: value for key, value in diagnostic.items() if key != "samples"}


def wait_for_desktop_settled(env, initial, *, output, report, enabled=True):
    """Passive, bounded quiet-surface heuristic; future loading is never excluded.

    Shell-only surfaces are valid. Missing/incomplete accessibility or moving
    content yields unproven stability, not application failure. Observation
    transport failures and a final placeholder still fail startup safely.
    """
    if type(enabled) is not bool:
        raise ValueError("passive settling must be a boolean")
    started = time.monotonic()
    observation = initial
    diagnostic = {
        "version": PASSIVE_SETTLING_VERSION, "enabled": enabled, **PASSIVE_SETTLING_BOUNDS,
        "status": "waiting", "reason": "", "samples_observed": 0, "elapsed_ms": 0.0,
        "stable_samples": 0, "stable_span_ms": 0.0, "surface_stability": "unproven",
        "application_ready_proven": False, "future_loading_excluded": False, "samples": [],
        "observation_reserve_ms": 0.0,
    }
    report["passive_settling"] = diagnostic

    def finish(status, reason):
        diagnostic.update(status=status, reason=reason, elapsed_ms=(time.monotonic() - started) * 1000)
        screenshot = observation.get("screenshot") if isinstance(observation, dict) else None
        tree = observation.get("accessibility_tree") if isinstance(observation, dict) else None
        diagnostic["last_screenshot_sha256"] = hashlib.sha256(screenshot).hexdigest() if isinstance(
            screenshot, bytes) else None
        diagnostic["last_accessibility_sha256"] = hashlib.sha256(tree.encode()).hexdigest() if isinstance(
            tree, str) else None
        return observation

    def fail(reason):
        finish("failed", reason)
        save_observation_artifacts(output, "settling-last", observation, report)
        raise VMStartupError({"passive_settling": diagnostic})

    if not enabled:
        return finish("disabled", "disabled")
    previous, stable_since = None, started
    max_observation_s = 0.0
    while True:
        quality = desktop_observation_quality(observation)
        signature, complete, control_count = _settling_structure(observation, quality)
        sampled = time.monotonic()
        diagnostic["samples_observed"] += 1
        if signature is not None and signature == previous:
            diagnostic["stable_samples"] += 1
        else:
            stable_since = sampled
            diagnostic["stable_samples"] = 1 if signature is not None else 0
        previous = signature
        diagnostic["stable_span_ms"] = (sampled - stable_since) * 1000
        diagnostic["samples"].append({
            "at_ms": (sampled - started) * 1000, "visual_reason": quality["reason"],
            "a11y_complete": complete, "control_count": control_count, "signature": signature,
        })
        remaining = PASSIVE_SETTLING_BOUNDS["timeout_s"] - (sampled - started)
        if (remaining > 0 and diagnostic["stable_samples"] >= 3
                and diagnostic["stable_span_ms"] >= 4000):
            diagnostic["surface_stability"] = "observed_quiet"
            return finish("settled", "public_structure_stable")
        limit = diagnostic["samples_observed"] >= PASSIVE_SETTLING_BOUNDS["max_samples"]
        if remaining <= 0 or limit:
            if quality["reason"] != "visual_evidence_present":
                fail("final_visual_evidence_unavailable")
            return finish("budget_exhausted", "timeout" if remaining <= 0 else "sample_limit")
        time.sleep(min(PASSIVE_SETTLING_BOUNDS["interval_s"], remaining))
        remaining = PASSIVE_SETTLING_BOUNDS["timeout_s"] - (time.monotonic() - started)
        if remaining <= 0:
            if quality["reason"] != "visual_evidence_present":
                fail("final_visual_evidence_unavailable")
            return finish("budget_exhausted", "timeout")
        if diagnostic["observation_reserve_ms"] > 0 and remaining * 1000 <= (
            diagnostic["observation_reserve_ms"]
        ):
            # Do not start another request when twice the prior maximum read
            # latency already consumes its remaining budget. This
            # is an admission heuristic, not a guarantee that future reads fit.
            # An actual in-flight error still fails startup below.
            if quality["reason"] != "visual_evidence_present":
                fail("final_visual_evidence_unavailable")
            return finish("budget_exhausted", "observation_budget_reserve")
        observed_at = time.monotonic()
        try:
            observation = _readiness_observe(env, remaining)
        except _ReadinessDeadline:
            fail("observation_timeout")
        except _ReadinessBudgetUnavailable as error:
            fail(str(error))
        except Exception:
            fail("observation_error")
        max_observation_s = max(max_observation_s, time.monotonic() - observed_at)
        diagnostic["observation_reserve_ms"] = (
            max_observation_s + max(0.25, max_observation_s)
        ) * 1000


def use_worker_settling(env, initial, *, output, report, enabled):
    """Copy the worker's checked result; never settle the SSH episode twice."""
    from cua_jev.osworld_remote import _settling_diagnostic

    diagnostic = getattr(env, "passive_settling", None)
    if (diagnostic is None or diagnostic["enabled"] is not enabled
            or diagnostic.get("version") != PASSIVE_SETTLING_VERSION):
        raise VMStartupError({"passive_settling": {"status": "failed", "reason": "worker_setting_mismatch"}})
    # The local visual gate must not silently replace the worker's ready frame
    # while retaining evidence about a different earlier observation.
    report["passive_settling"] = _settling_diagnostic(diagnostic, initial)
    return initial


def startup_diagnostics(provider, error):
    """Only boot logs from our fresh VM, which has never received model keys."""
    data = {"error_type": type(error).__name__}
    container = provider.container
    if container is not None:
        data["container_name"] = container.name
        try:
            container.reload()
            state = container.attrs.get("State", {})
            data.update(
                status=container.status, exit_code=state.get("ExitCode"), oom_killed=state.get("OOMKilled")
            )
            data["boot_log_tail"] = container.logs(tail=40).decode("utf-8", errors="replace")[-8000:]
        except Exception as diagnostic_error:
            data["diagnostic_error"] = type(diagnostic_error).__name__
    return data


def validate_vm_path(path: Path) -> Path:
    """A base owned by the invoking account is only ever mounted read-only."""
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or resolved.suffix != ".qcow2":
        raise ValueError("--vm-path must be an existing regular .qcow2 file")
    if not hasattr(os, "getuid") or info.st_uid != os.getuid():
        raise ValueError("the VM base must belong to the current Unix account")
    return resolved


def load_task(root: Path, selection: str) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9_]*/[a-zA-Z0-9_-]+", selection):
        raise ValueError("tasks must use domain/task-id without a .json suffix")
    examples = (root / "evaluation_examples" / "examples").resolve()
    source = (examples / f"{selection}.json").resolve(strict=True)
    if not source.is_relative_to(examples):
        raise ValueError("task path escapes the OSWorld example directory")
    task = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        raise ValueError("OSWorld task requires an instruction")
    return task


def source_metadata(root: Path, tasks: list[str]) -> dict:
    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()

    try:
        revision = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {
        "osworld_root": str(root),
        "osworld_revision": revision,
        "osworld_tracked_files_dirty": dirty,
        "evaluator_source": "existing_local_OSWorld_checkout",
        "task_sha256": {
            name: hashlib.sha256(
                (root / "evaluation_examples" / "examples" / f"{name}.json").read_bytes()
            ).hexdigest()
            for name in tasks
        },
    }


def _bounded_timeout(value, *, read_limit):
    def cap(item, limit):
        if item is None:
            return limit
        if isinstance(item, (int, float)) and not isinstance(item, bool) and item > 0:
            return min(item, limit)
        # Leave invalid values to requests' own validation.
        return item

    if isinstance(value, tuple) and len(value) == 2:
        return (cap(value[0], 10), cap(value[1], read_limit))
    if value is None or isinstance(value, (int, float)):
        return (cap(value, 10), cap(value, read_limit))
    # Requests also accepts urllib3 Timeout objects. Preserve smaller limits.
    if hasattr(value, "connect_timeout") and hasattr(value, "read_timeout"):
        return (cap(value.connect_timeout, 10), cap(value.read_timeout, read_limit))
    return value


@contextmanager
def bounded_vm_http(owned_ports: set[int]):
    """Bound only this process's requests to its currently owned VM services."""
    import requests

    original = requests.sessions.Session.request

    def request(session, method, url, *args, **kwargs):
        try:
            parsed = urlsplit(url)
            owned = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                and parsed.port in owned_ports
            )
        except (TypeError, ValueError):
            owned = False
        if owned:
            # Setup may upload task assets or launch an application. Downloads
            # from external asset servers keep their original timeout policy.
            read_limit = 120 if parsed.path.startswith("/setup/") else 60
            if len(args) >= 7:
                args = (*args[:6], _bounded_timeout(args[6], read_limit=read_limit), *args[7:])
            else:
                kwargs["timeout"] = _bounded_timeout(kwargs.get("timeout"), read_limit=read_limit)
        return original(session, method, url, *args, **kwargs)

    requests.sessions.Session.request = request
    try:
        yield
    finally:
        requests.sessions.Session.request = original


@contextmanager
def isolated_docker(
    provider_class, vm_path: Path, run_id: str, *, owner: str, boot_wait_budget=None,
):
    """Patch this process only; cleanup requires all ownership labels to match."""
    base = validate_vm_path(vm_path)
    labels = {"cua-jev.owner": owner, "cua-jev.purpose": "osworld-smoke", "cua-jev.run": run_id}
    original_start = provider_class.start_emulator
    original_stop = provider_class.stop_emulator
    active = []
    owned_ports: set[int] = set()

    def stop(provider, *_args, **_kwargs):
        container = provider.container
        if container is None:
            return
        container.reload()
        actual = container.attrs.get("Config", {}).get("Labels", {}) or {}
        if any(actual.get(key) != value for key, value in labels.items()):
            raise RuntimeError("refusing to stop a container without matching ownership labels")
        if not container.name.startswith(f"{owner}-cua-jev-"):
            raise RuntimeError("refusing to stop a container outside this run's name prefix")
        container.stop(timeout=10)
        container.remove()
        for attribute in ("server_port", "vnc_port", "chromium_port", "vlc_port"):
            owned_ports.discard(getattr(provider, attribute, None))
        provider.container = None
        provider.server_port = provider.vnc_port = provider.chromium_port = provider.vlc_port = None
        if provider in active:
            active.remove(provider)

    def start(provider, path_to_vm, _headless, _os_type):
        if Path(path_to_vm).resolve() != base:
            raise ValueError("provider received a different VM base")
        if boot_wait_budget is not None:
            boot_wait_budget.begin_start()
        # Do not pull images or alter a shared image registry on this host.
        provider.client.images.get("happysixd/osworld-docker")
        name = f"{owner}-cua-jev-{run_id[:8]}-{uuid.uuid4().hex[:8]}"
        environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "2", "ALLOCATE": "N"}
        if not Path("/dev/kvm").exists():
            raise RuntimeError("KVM is required for this bounded local experiment")
        try:
            provider.container = provider.client.containers.create(
                "happysixd/osworld-docker",
                # This image hardcodes /boot.qcow2 in install.sh. Redirect only
                # this new container's writable overlay into its private tmpfs.
                # findFile then discovers the regular /storage/boot.qcow2 file.
                entrypoint=["/usr/bin/tini", "-s", "--", "/bin/bash"],
                command=["-c", "set -eu; ln -s /storage/boot.qcow2 /boot.qcow2; exec /run/entry.sh"],
                name=name,
                labels=labels,
                environment=environment,
                cap_add=["NET_ADMIN"],
                devices=["/dev/kvm"],
                volumes={str(base): {"bind": "/System.qcow2", "mode": "ro"}},
                tmpfs={"/storage": "rw,size=2g"},
                ports={port: ("127.0.0.1", None) for port in (8006, 5000, 9222, 8080)},
                mem_limit="6g",
                nano_cpus=2_000_000_000,
                detach=True,
            )
            # Register the handle before start: Docker may create successfully
            # then fail to start, and containers.run would lose that handle.
            active.append(provider)
            provider.container.start()
            provider.container.reload()
            bindings = provider.container.attrs["NetworkSettings"]["Ports"]
            for port, attribute in (
                (8006, "vnc_port"),
                (5000, "server_port"),
                (9222, "chromium_port"),
                (8080, "vlc_port"),
            ):
                mappings = bindings[f"{port}/tcp"]
                if len(mappings) != 1 or mappings[0]["HostIp"] != "127.0.0.1":
                    raise RuntimeError("Docker did not bind the VM service only to loopback")
                setattr(provider, attribute, int(mappings[0]["HostPort"]))
                owned_ports.add(int(mappings[0]["HostPort"]))
            if boot_wait_budget is None:
                provider._wait_for_vm_ready(timeout=180)
            else:
                boot_wait_budget.wait_for_ready(provider._wait_for_vm_ready)
        except BaseException as error:
            diagnostic = startup_diagnostics(provider, error)
            if boot_wait_budget is not None:
                diagnostic["owned_boot_wait_budget"] = boot_wait_budget.report
            try:
                stop(provider)
            except Exception as cleanup_error:
                diagnostic["cleanup_error"] = type(cleanup_error).__name__
                # Keep the active handle for the outer context's cleanup retry.
            if isinstance(error, Exception):
                raise VMStartupError(diagnostic) from error
            raise

    provider_class.start_emulator = start
    provider_class.stop_emulator = stop
    with bounded_vm_http(owned_ports):
        try:
            yield
        finally:
            cleanup_errors = []
            try:
                for provider in list(active):
                    try:
                        stop(provider)
                    except Exception as error:
                        cleanup_errors.append(error)
            finally:
                provider_class.start_emulator = original_start
                provider_class.stop_emulator = original_stop
            if cleanup_errors:
                raise ExceptionGroup("owned OSWorld container cleanup failed", cleanup_errors)


def reset_checked(env, task):
    """Older OSWorld versions can return an observation after failed setup."""
    outcomes = []
    originals = []

    def attach():
        controller = env.setup_controller
        original = controller.setup
        originals.append((controller, original))

        def checked(*args, **kwargs):
            outcome = original(*args, **kwargs)
            outcomes.append(outcome)
            return outcome

        controller.setup = checked

    original_start = getattr(env, "_start_emulator", None)

    def restart(*args, **kwargs):
        result = original_start(*args, **kwargs)
        attach()
        return result

    attach()
    if original_start:
        env._start_emulator = restart
    try:
        observation = env.reset(task_config=task)
    finally:
        for controller, original in originals:
            controller.setup = original
        if original_start:
            env._start_emulator = original_start
    if not outcomes or outcomes[-1] is not True:
        raise SetupFailed("OSWorld task setup did not report success")
    return observation


def evaluate_once(env) -> dict:
    started = time.perf_counter()
    try:
        raw_score = env.evaluate()
        score = float(raw_score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("official score is not finite or is outside [0, 1]")
        return {
            "evaluation_status": "ok",
            "official_score": score,
            "success": score == 1.0,
            "evaluation_wall_ms": (time.perf_counter() - started) * 1000,
        }
    except Exception as error:
        return {
            "evaluation_status": "error",
            "official_score": None,
            "success": False,
            "evaluation_error": type(error).__name__,
            "evaluation_wall_ms": (time.perf_counter() - started) * 1000,
        }


def _plan_ttl(value):
    seconds = float(value)
    if not 0 < seconds <= 300:
        raise argparse.ArgumentTypeError("--plan-ttl must be greater than zero and at most 300 seconds")
    return seconds


def _model_request_timeout(value):
    seconds = float(value)
    if not 0 < seconds <= 300:
        raise argparse.ArgumentTypeError("--model-request-timeout must be greater than zero and at most 300")
    return seconds


def model_request_timeout_s(args):
    explicit = getattr(args, "model_request_timeout", None)
    return explicit if explicit is not None else (90.0 if getattr(args, "model_thinking", "default") == "on"
                                                 else 40.0)


def _model_thinking_budget(value):
    try:
        if type(value) not in {str, int}:
            raise ValueError
        tokens = int(value)
        if tokens <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise argparse.ArgumentTypeError("--model-thinking-budget must be a positive integer") from None
    return tokens


def model_thinking_budget_tokens(args):
    """Validate the declared request cap, without inferring provider support or limits."""
    budget = getattr(args, "model_thinking_budget", None)
    if budget is not None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("--model-thinking-budget must be a positive integer")
        if getattr(args, "model_thinking", "default") != "on":
            raise ValueError("--model-thinking-budget requires --model-thinking on")
    return budget


def add_plan_arguments(parser):
    parser.add_argument("--efficient-observations", action="store_true",
                        help="defer single-action post-plan reads to the backend's fresh input preflight")
    parser.add_argument(
        "--stale-recovery-on-progress", action="store_true",
        help="allow one fresh zero-input stale recovery between verified GUI inputs, within original budgets",
    )
    parser.add_argument("--native-menu-ocr-binary", type=Path, help="optional existing read-only OCR worker")
    parser.add_argument("--native-menu-ocr-sha256", help="required SHA256 pin for the optional worker")
    parser.add_argument("--native-menu-ocr-timeout", type=float, default=6.0,
                        help="local OCR frame budget in seconds, including tile preprocessing")
    parser.add_argument("--native-menu-ocr-batch", action="store_true",
                        help="use a pinned worker supporting one process per complete tiled frame")
    parser.add_argument(
        "--desktop-stage-contract",
        choices=("legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
                 "property_intent_v29", "property_intent_v30", "property_intent_v31", "property_intent_v33",
                 "property_intent_v49", "property_intent_v51", "property_intent_v52"),
        default="legacy_v23",
        help="desktop stage protocol; v33/v49/v51/v52 share complete immediate instructions for both modes; "
             "v52 also requires explicit semantic completion and property confirmation choices",
    )
    parser.add_argument(
        "--desktop-output-format", choices=("unconstrained_v1", "json_schema_v50", "tool_declaration_v52",
                                            "semantic_tools_v53", "semantic_tools_v54", "semantic_tools_v55",
                                            "semantic_tools_v56", "semantic_tools_v57", "semantic_tools_v58",
                                            "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64",
                                                "semantic_json_v66"),
        default="unconstrained_v1",
        help="optional structured model response format; "
             "json_schema_v50 requires property_intent_v49/v51/v52; "
             "tool_declaration_v52 and semantic_tools_v53/v54/v55/v56/v57/v58 require property_intent_v52 "
             "and full_multimodal_v1",
    )
    parser.add_argument(
        "--desktop-normal-stage-policy", choices=("program_optional", "atomic_stage_only"),
        default="program_optional",
        help="atomic_stage_only keeps reusable property/navigation/field stages and context "
             "handoff, while omitting optional multi-stage programs; requires semantic_tools_v64",
    )
    parser.add_argument(
        "--current-action-policy", dest="desktop_current_action_policy",
        choices=("all_families", "ref_visual_key_v1"), default="all_families",
        help="explicit capability-restricted ordinary/recovery H1 diagnostic: disable semantic-current; "
             "requires semantic_tools_v64; formal review, repair and normal H6 stay unchanged",
    )
    parser.add_argument(
        "--desktop-planning-observation", choices=("full_multimodal_v1", "native_first_v51"),
        default="full_multimodal_v1",
        help="native_first_v51 uses complete control text for multi-step planning with visual recovery; "
             "requires property_intent_v51/v52 and json_schema_v50",
    )
    parser.add_argument(
        "--visual-box-space", choices=("normalized_1000", "image_pixels"), default="normalized_1000",
        help="preferred VLM visual coordinates; image_pixels refers to the actual encoded JPEG",
    )
    parser.add_argument(
        "--pool-desktop-ancestors", action="store_true",
        help="losslessly share repeated VLM observation ancestry when saving over 4096 bytes, in both modes",
    )
    parser.add_argument(
        "--compact-desktop-controls", action="store_true",
        help="losslessly share VLM control capabilities and fields when saving over 4096 bytes, "
             "in both modes",
    )
    parser.add_argument(
        "--jev-request-layout",
        choices=("full_v1", "deduplicated_v1", "shared_fields_v2", "shared_fields_v3", "shared_fields_v4",
                 "shared_fields_v5", "shared_fields_v6", "shared_fields_v7"),
        default="full_v1",
        help="Jev payload: v2 identical defaults, v3 modal defaults, v4 shared ancestor paths, "
             "v5 exact argument/description sharing; v6 independently selects the smaller "
             "profitable lossless public representation; v7 optionally compacts full control descriptions "
             "and maps short wire choice handles back to the original offered IDs",
    )
    parser.add_argument(
        "--jev-capacity-recovery", action="store_true",
        help="allow a bounded fresh single-action VLM rescue after a known Jev capacity rejection; "
             "disabled by default and reported separately",
    )
    parser.add_argument(
        "--jev-capacity-recovery-policy", choices=("vlm_once_v1", "vlm_phase_v2", "vlm_phase_v3"),
        default="vlm_once_v1",
        help="capacity rescue policy; the separate --jev-capacity-recovery master is still required",
    )
    parser.add_argument(
        "--reuse-unexplained-context-planning-observation", action="store_true",
        help="reuse one fresh verified full observation for unexplained-context planning; "
             "disabled by default; preserves model-return and execution freshness checks",
    )
    parser.add_argument(
        "--navigation-continuation", action="store_true",
        help="bounded observed-loop continuation for hybrid desktop property52/semantic57; "
             "the same configured flag is recorded for the baseline",
    )
    parser.add_argument(
        "--jev-network-recovery", action="store_true",
        help="allow one fresh single-action VLM rescue after exhausted Jev connection/timeouts; "
             "disabled by default and reported separately from native hybrid execution",
    )
    parser.add_argument(
        "--no-passive-settling", dest="passive_settling", action="store_false",
        help="disable the bounded post-reset public-surface quiet heuristic in both modes",
    )
    parser.add_argument(
        "--no-reuse-verified-post-snapshot", dest="reuse_verified_post_snapshot", action="store_false",
        help="disable one-use verified desktop post-snapshot reuse in both modes (enabled by default)",
    )
    parser.add_argument(
        "--reuse-verified-pending-snapshot", action="store_true",
        help="reuse one fresh verified ordinary pending-effect full frame; requires post-snapshot reuse "
             "and preserves execution preflight checks in both modes",
    )
    parser.add_argument(
        "--reuse-whole-goal-planning-snapshot", action="store_true",
        help="reuse the just-checked completion frame for dedicated review planning only; "
             "requires whole-goal and verified-post reuse; final checks remain fresh",
    )
    parser.add_argument(
        "--inband-whole-goal-recheck", action="store_true",
        help="request whole-goal review inside an existing full-image planner request; "
             "requires --whole-goal-recheck and retains final fresh predicates",
    )
    parser.add_argument(
        "--whole-goal-recheck", action="store_true",
        help="recheck the complete instruction before a whole-goal completion claim, within the original "
             "planning budget; disabled by default in both modes",
    )
    parser.add_argument(
        "--satisfied-property-readonly-fast-path", action="store_true",
        help="use the existing read-only singleton proof for a unique already-selected literal "
             "property option; later confirmation and whole-goal checks remain unchanged",
    )
    parser.add_argument(
        "--image-only-fill-intermediates", action="store_true",
        help="retain screenshots but omit accessibility in fill's focus-click/select-all step responses; "
             "fresh input checks and the final typing observation remain unchanged in both modes",
    )
    parser.add_argument(
        "--model-thinking", choices=("default", "on", "off"), default="default",
        help="request provider enable_thinking for both modes; default omits this provider-specific option",
    )
    parser.add_argument(
        "--model-thinking-budget", type=_model_thinking_budget, default=None,
        help="optional positive reasoning-token request cap for both modes, requires --model-thinking on; "
             "provider support and upper bounds must be verified separately",
    )
    parser.add_argument(
        "--model-request-timeout", type=_model_request_timeout, default=None,
        help="VLM request wait for both modes: default90s with thinking on, otherwise40s; "
             "network phases capped by remaining episode time",
    )
    parser.add_argument(
        "--plan-horizon", type=int, choices=range(1, 13), default=6,
        help="maximum actions per hybrid plan (default: 6); vlm always uses 1",
    )
    parser.add_argument(
        "--plan-ttl", type=_plan_ttl, default=300.0,
        help="hybrid plan lifetime in seconds, greater than zero and at most 300 (default: 300)",
    )


def planning_settings(args, mode):
    """Record the settings actually passed to the runner, including baseline overrides."""
    return {
        "plan_horizon": getattr(args, "plan_horizon", 6) if mode == "hybrid" else 1,
        "plan_ttl_s": getattr(args, "plan_ttl", 300.0),
    }


def experiment_settings(args):
    provider = getattr(args, "vm_provider", "docker")
    capacity_policy = getattr(args, "jev_capacity_recovery_policy", "vlm_once_v1")
    if (type(capacity_policy) is not str
            or capacity_policy not in {"vlm_once_v1", "vlm_phase_v2", "vlm_phase_v3"}):
        raise ValueError("jev_capacity_recovery_policy must be vlm_once_v1, vlm_phase_v2 or vlm_phase_v3")
    for field in ("reuse_whole_goal_planning_snapshot", "inband_whole_goal_recheck",
                  "navigation_continuation", "satisfied_property_readonly_fast_path",
                  "reuse_unexplained_context_planning_observation"):
        if type(getattr(args, field, False)) is not bool:
            raise ValueError(f"{field} must be a boolean")
    normal_policy = getattr(args, "desktop_normal_stage_policy", "program_optional")
    if normal_policy not in {"program_optional", "atomic_stage_only"}:
        raise ValueError("invalid desktop normal-stage policy")
    if (normal_policy == "atomic_stage_only"
            and getattr(args, "desktop_output_format", "unconstrained_v1") != "semantic_tools_v64"):
        raise ValueError("atomic_stage_only requires semantic_tools_v64")
    observation_order = getattr(args, "observation_order", "original")
    if observation_order not in {"original", "ax_then_screenshot_v1"}:
        raise ValueError("invalid OSWorld observation order")
    current_policy = getattr(args, "desktop_current_action_policy", "all_families")
    if (type(current_policy) is not str or current_policy not in {"all_families", "ref_visual_key_v1"}
            or (current_policy != "all_families"
                and getattr(args, "desktop_output_format", "unconstrained_v1") != "semantic_tools_v64")):
        raise ValueError("ref_visual_key_v1 requires semantic_tools_v64; unsupported current action policy")
    return {
        "vm_provider": provider,
        "vm_network_mode": "user_restricted_loopback" if provider == "qemu" else "docker_bridge_loopback",
        "singleton_fast_path": True,
        **({"reuse_unexplained_context_planning_observation": True}
           if getattr(args, "reuse_unexplained_context_planning_observation", False) else {}),
        **({"satisfied_property_readonly_fast_path": True}
           if getattr(args, "satisfied_property_readonly_fast_path", False) else {}),
        "jev_request_layout": getattr(args, "jev_request_layout", "full_v1"),
        **({"efficient_observations": True} if getattr(args, "efficient_observations", False) else {}),
        **({"ax_timeout_s": args.ax_timeout} if getattr(args, "ax_timeout", None) is not None else {}),
        "desktop_stage_contract": getattr(args, "desktop_stage_contract", "legacy_v23"),
        "desktop_output_format": getattr(args, "desktop_output_format", "unconstrained_v1"),
        "desktop_normal_stage_policy": normal_policy,
        **({"desktop_current_action_policy": current_policy,
            "current_action_capability_restricted": True} if current_policy != "all_families" else {}),
        "observation_order": observation_order,
        "desktop_planning_observation": getattr(args, "desktop_planning_observation", "full_multimodal_v1"),
        "visual_box_space": getattr(args, "visual_box_space", "normalized_1000"),
        "desktop_snapshot_encoding": (
            ("shared_ancestors_and_controls_v1_if_saving_4096_bytes"
             if getattr(args, "pool_desktop_ancestors", False)
             else "shared_control_fields_v1_if_saving_4096_bytes")
            if getattr(args, "compact_desktop_controls", False) else
            ("shared_ancestor_paths_v1_if_saving_4096_bytes"
             if getattr(args, "pool_desktop_ancestors", False) else "control_defaults_v1")
        ),
        "stale_observation_recovery": (
            STALE_RECOVERY_PROGRESS_VERSION if getattr(args, "stale_recovery_on_progress", False)
            else STALE_RECOVERY_VERSION
        ),
        "desktop_observation_budget": OBSERVATION_BUDGET_VERSION,
        "menu_grounding": ("native_menu_text_v1" if getattr(args, "native_menu_ocr_binary", None)
                           else "disabled"),
        "menu_ocr_worker_sha256": (getattr(args, "native_menu_ocr_sha256", None)
                                   if getattr(args, "native_menu_ocr_binary", None) else "disabled"),
        "menu_ocr_timeout_s": (getattr(args, "native_menu_ocr_timeout", 6.0)
                               if getattr(args, "native_menu_ocr_binary", None) else "disabled"),
        "menu_ocr_process_layout": (
            ("frame_batch_v1" if getattr(args, "native_menu_ocr_batch", False) else "tile_process_v1")
            if getattr(args, "native_menu_ocr_binary", None) else "disabled"
        ),
        "jev_network_recovery": (
            "vlm_once_v1" if getattr(args, "jev_network_recovery", False) else "disabled"
        ),
        "jev_capacity_recovery": (
            capacity_policy if getattr(args, "jev_capacity_recovery", False) else "disabled"
        ),
        "jev_capacity_recovery_policy": capacity_policy,
        "navigation_continuation": getattr(args, "navigation_continuation", False),
        "navigation_continuation_version": (
            "observed_transition_v1" if getattr(args, "navigation_continuation", False) else "disabled"
        ),
        "reuse_verified_post_snapshot": getattr(args, "reuse_verified_post_snapshot", True),
        "reuse_verified_pending_snapshot": getattr(args, "reuse_verified_pending_snapshot", False),
        "whole_goal_recheck": (
            "current_goal_recheck_v1" if getattr(args, "whole_goal_recheck", False) else "disabled"
        ),
        "reuse_whole_goal_planning_snapshot": getattr(args, "reuse_whole_goal_planning_snapshot", False),
        "whole_goal_planning_snapshot_reuse_version": (
            "successful_completion_full_frame_v1"
            if (getattr(args, "whole_goal_recheck", False)
                and getattr(args, "reuse_verified_post_snapshot", True)
                and getattr(args, "reuse_whole_goal_planning_snapshot", False)) else "disabled"
        ),
        "inband_whole_goal_recheck": getattr(args, "inband_whole_goal_recheck", False),
        "inband_whole_goal_recheck_version": (
            "all_original_goal_review_v1"
            if (getattr(args, "whole_goal_recheck", False)
                and getattr(args, "inband_whole_goal_recheck", False)) else "disabled"
        ),
        "intermediate_fill_observation": (
            "image_only_v1" if getattr(args, "image_only_fill_intermediates", False) else "full_v1"
        ),
        "worker_instrumentation": (
            "fill_observation_rpc_v1" if getattr(args, "profile_observation_costs", False) else "disabled"
        ),
        "passive_settling": getattr(args, "passive_settling", True),
        "passive_settling_version": PASSIVE_SETTLING_VERSION,
        "model_thinking": getattr(args, "model_thinking", "default"),
        "model_thinking_budget_tokens": (
            model_thinking_budget_tokens(args) if getattr(args, "model_thinking_budget", None) is not None
            else "default"
        ),
        "model_request_timeout_s": model_request_timeout_s(args),
        "pause_s": getattr(args, "pause", None),
        "max_steps": getattr(args, "max_steps", None),
        "timeout_s": getattr(args, "timeout", None),
        "startup_timeout_s": getattr(args, "startup_timeout", None),
        "a11y": not args.no_a11y if hasattr(args, "no_a11y") else None,
    }


def save_observation_artifacts(output, stem, observation, report):
    """Save only already-observed public surfaces; artifact errors are nonfatal."""
    if observation is None:
        return
    if not isinstance(observation, dict):
        report.setdefault("artifact_errors", []).append({"artifact": stem, "error_type": "TypeError"})
        return
    for field, suffix in (("screenshot", ".png"), ("accessibility_tree", "-a11y.xml")):
        name = f"{stem}{suffix}"
        try:
            value = observation.get(field)
            if value is None:
                continue
            if field == "accessibility_tree":
                if not isinstance(value, str):
                    raise TypeError("accessibility artifact must be text")
                # Bound both allocation and the encoded file, including non-ASCII XML.
                if len(value) > 2_000_000:
                    raise ValueError("accessibility artifact exceeds the byte limit")
                value = value.encode("utf-8")
                if len(value) > 2_000_000:
                    raise ValueError("accessibility artifact exceeds the byte limit")
            elif not isinstance(value, bytes):
                raise TypeError("screenshot artifact must be bytes")
            elif len(value) > 20_000_000:
                raise ValueError("screenshot artifact exceeds the byte limit")
            elif not value:
                continue
            (output / name).write_bytes(value)
        except Exception as error:
            report.setdefault("artifact_errors", []).append(
                {"artifact": name, "error_type": type(error).__name__}
            )


def _menu_text_provider(args, output):
    binary = getattr(args, "native_menu_ocr_binary", None)
    digest = getattr(args, "native_menu_ocr_sha256", None)
    batch_mode = getattr(args, "native_menu_ocr_batch", False)
    if binary is None and digest is None and not batch_mode:
        return None
    if binary is None or digest is None or getattr(args, "no_a11y", False):
        raise ValueError("native menu OCR requires its worker, SHA256 pin and accessibility observations")
    from cua_jev.native_menu_ocr import NativeMenuOcr

    return NativeMenuOcr(binary, digest, output / "menu-ocr",
                         timeout_s=getattr(args, "native_menu_ocr_timeout", 6.0), batch_mode=batch_mode)


def run_episode(args, task_name, task, mode, *, env_factory, backend_factory, planner_factory,
                initialize=reset_checked, settle=None, policy_factory=None):
    output = args.output / task_name.replace("/", "-") / mode
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "task": task_name,
        "instruction": task.get("instruction", ""),
        "mode": mode,
        "model": args.model,
        "live_vlm": True,
        "live_jev": mode == "hybrid",
        "success": False,
        "official_score": None,
        "evaluation_status": "not_run",
        "runner": None,
        "experiment_settings": experiment_settings(args),
        **planning_settings(args, mode),
    }
    env = backend = planner = policy = None
    environment_attempted = False
    phase = "configuration"
    started = time.perf_counter()
    try:
        menu_provider = _menu_text_provider(args, output)
        planner = planner_factory(args)
        if mode == "hybrid":
            policy = (policy_factory() if policy_factory else
                      JevPolicy(retries=2, request_layout=getattr(args, "jev_request_layout", "full_v1")))
        report["configuration_wall_ms"] = (time.perf_counter() - started) * 1000
        phase = "startup"
        phase_started = time.perf_counter()
        environment_attempted = True
        env = env_factory(
            provider_name="docker",
            path_to_vm=str(args.vm_path),
            action_space="computer_13",
            cache_dir=str(output / "cache"),
            headless=True,
            require_a11y_tree=not args.no_a11y,
            require_terminal=False,
            enable_proxy=False,
        )
        # The remote factory receives the public instruction during its ready
        # handshake, after the report was created and before any planner call.
        report["instruction"] = task.get("instruction", "")
        if isinstance(env, RemoteOSWorldEnv):
            report["startup_diagnostics"] = dict(env.startup_diagnostics)
        report["startup_wall_ms"] = (time.perf_counter() - phase_started) * 1000
        phase = "setup"
        phase_started = time.perf_counter()
        initial = initialize(env, task)
        report["task_setup_wall_ms"] = (time.perf_counter() - phase_started) * 1000
        phase = "startup"
        initial = wait_for_desktop_ready(env, initial, output=output, report=report)
        initial = (settle or wait_for_desktop_settled)(
            env, initial, output=output, report=report,
            enabled=report["experiment_settings"]["passive_settling"],
        )
        save_observation_artifacts(output, "initial", initial, report)
        report["setup_wall_ms"] = (time.perf_counter() - started) * 1000
        phase = "agent"
        backend_options = {"menu_text_provider": menu_provider} if menu_provider is not None else {}
        if getattr(args, "image_only_fill_intermediates", False):
            backend_options["image_only_fill_intermediates"] = True
        if report["experiment_settings"].get("satisfied_property_readonly_fast_path") is True:
            backend_options["satisfied_property_readonly_fast_path"] = True
        backend = backend_factory(env, pause=args.pause, **backend_options)
        screenshot_index = 0

        def progress(data):
            nonlocal screenshot_index
            print(json.dumps(data, ensure_ascii=False), flush=True)
            if data.get("event") == "agent_step":
                screenshot_index += 1
                try:
                    save_observation_artifacts(
                        output, f"step-{screenshot_index:03}",
                        getattr(backend, "last_observation", {}), report,
                    )
                except Exception as error:
                    report.setdefault("artifact_errors", []).append(
                        {"artifact": f"step-{screenshot_index:03}", "error_type": type(error).__name__}
                    )

        try:
            runner = AgentRunner(
                backend,
                planner,
                policy=policy,
                config=AgentConfig(
                    mode=mode,
                    allow_actions=True,
                    reuse_verified_post_snapshot=(
                        report["experiment_settings"]["reuse_verified_post_snapshot"]
                    ),
                    stale_recovery_on_progress=getattr(args, "stale_recovery_on_progress", False),
                    jev_network_recovery=(
                        report["experiment_settings"]["jev_network_recovery"] == "vlm_once_v1"
                    ),
                    jev_capacity_recovery=(
                        report["experiment_settings"]["jev_capacity_recovery"] in {
                            "vlm_once_v1", "vlm_phase_v2", "vlm_phase_v3",
                        }
                    ),
                    navigation_continuation=report["experiment_settings"]["navigation_continuation"],
                    **({"reuse_unexplained_context_planning_observation": True}
                       if report["experiment_settings"].get(
                           "reuse_unexplained_context_planning_observation", False,
                       ) else {}),
                    jev_capacity_recovery_policy=(
                        report["experiment_settings"]["jev_capacity_recovery_policy"]
                    ),
                    reuse_whole_goal_planning_snapshot=(
                        report["experiment_settings"]["reuse_whole_goal_planning_snapshot"]
                    ),
                    inband_whole_goal_recheck=(
                        report["experiment_settings"]["inband_whole_goal_recheck"]
                    ),
                    reuse_verified_pending_snapshot=(
                        report["experiment_settings"]["reuse_verified_pending_snapshot"]
                    ),
                    whole_goal_recheck=(
                        report["experiment_settings"]["whole_goal_recheck"] == "current_goal_recheck_v1"
                    ),
                    plan_horizon=report["plan_horizon"],
                    plan_ttl_s=report["plan_ttl_s"],
                    max_steps=args.max_steps,
                    max_plans=args.max_steps + 5,
                    timeout_s=args.timeout,
                    defer_immediate_plan_refresh=getattr(args, "efficient_observations", False),
                ),
                trace=JsonlTrace(output / "trace.jsonl"),
                progress=progress,
            )
            # Only the natural-language instruction goes to the agent, never the evaluator.
            report["runner"] = runner.run(task["instruction"], success=None).to_dict()
        except Exception as error:
            report["agent_error"] = type(error).__name__
        observation_phase_metrics = getattr(backend, "observation_phase_metrics", None)
        if isinstance(observation_phase_metrics, dict):
            report["observation_phase_timing"] = observation_phase_metrics
        if getattr(args, "image_only_fill_intermediates", False):
            report["intermediate_observation"] = dict(backend.intermediate_observation_metrics)
        if menu_provider is not None:
            report["menu_grounding"] = {
                "backend": dict(getattr(backend, "menu_grounding_metrics", {})),
                "provider": dict(menu_provider.metrics),
            }
        try:
            screenshot = env.controller.get_screenshot()
            if screenshot:
                (output / "final.png").write_bytes(screenshot)
        except Exception as error:
            report["artifact_error"] = type(error).__name__
        phase = "evaluation"
        report.update(evaluate_once(env))
        report["episode_status"] = (
            "evaluation_error"
            if report["evaluation_status"] == "error"
            else "success"
            if report["success"]
            else "task_failure"
        )
    except Exception as error:
        report["episode_status"] = f"{phase}_error"
        report["error_type"] = type(error).__name__
        if isinstance(error, VMStartupError):
            report["startup_diagnostics"] = error.diagnostics
        elif isinstance(error, RemoteOSWorldError) and hasattr(error, "startup_diagnostics"):
            report["startup_diagnostics"] = dict(error.startup_diagnostics)
    finally:
        environment_close_failed = False
        for name, resource in (("backend", backend), ("environment", env), ("policy", policy),
                               ("planner", planner)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report.setdefault("cleanup_errors", []).append(type(error).__name__)
                    report.setdefault("cleanup_error_resources", []).append(name)
                    environment_close_failed |= name == "environment"
        if env is None:
            cleanup_status = "unconfirmed_startup" if environment_attempted else "not_started"
        else:
            # DesktopEnv has no separate acknowledgement field: a successful
            # close confirms cleanup. SSH transports expose a more specific
            # status, which must not be discarded or replaced with success.
            cleanup_status = getattr(env, "cleanup_status", None) or "confirmed"
            if environment_close_failed and cleanup_status == "confirmed":
                cleanup_status = "unconfirmed_close_error"
        report["cleanup_status"] = cleanup_status
        report["run_status"] = (
            "cleanup_error" if report.get("cleanup_errors")
            else "cleanup_unconfirmed" if cleanup_status not in {"confirmed", "not_started"}
            else report.get("episode_status", "interrupted")
        )
        report["episode_wall_ms"] = (time.perf_counter() - started) * 1000
        (output / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def run_exit_code(reports):
    """Task scores stay independent; a successful process also requires cleanup."""
    return 0 if reports and all(
        report.get("success") is True and report.get("run_status") == "success" for report in reports
    ) else 1


def make_desktop_planner(args):
    thinking_budget = model_thinking_budget_tokens(args)
    if not args.model_base_url or not args.model or not args.allow_screenshot_upload:
        raise ValueError("model URL/name and --allow-screenshot-upload are required")
    planner_options = {}
    normal_policy = getattr(args, "desktop_normal_stage_policy", "program_optional")
    if normal_policy != "program_optional":
        planner_options["desktop_normal_stage_policy"] = normal_policy
    current_policy = getattr(args, "desktop_current_action_policy", "all_families")
    if current_policy != "all_families":
        planner_options["desktop_current_action_policy"] = current_policy
    return MultimodalPlanner(
        args.model_base_url,
        args.model,
        profile="desktop",
        desktop_contract=getattr(args, "desktop_stage_contract", "legacy_v23"),
        desktop_output_format=getattr(args, "desktop_output_format", "unconstrained_v1"),
        **planner_options,
        desktop_planning_observation=getattr(args, "desktop_planning_observation", "full_multimodal_v1"),
        visual_box_space=getattr(args, "visual_box_space", "normalized_1000"),
        pool_desktop_ancestors=getattr(args, "pool_desktop_ancestors", False),
        compact_desktop_controls=getattr(args, "compact_desktop_controls", False),
        thinking_budget=thinking_budget,
        request_timeout_s=model_request_timeout_s(args),
        api_key=os.getenv("CUA_JEV_MODEL_API_KEY") or os.getenv("CUA_JEV_PLANNER_API_KEY"),
        allow_insecure_http=args.allow_insecure_model_http,
        use_env_proxy=args.use_env_proxy,
        enable_thinking={"default": None, "on": True, "off": False}[
            getattr(args, "model_thinking", "default")
        ],
    )


def main(argv=None):
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--vm-path", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True, metavar="DOMAIN/ID")
    parser.add_argument("--modes", nargs="+", choices=("hybrid", "vlm"), default=["hybrid"])
    parser.add_argument("--model-base-url", default=os.getenv("CUA_JEV_MODEL_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("CUA_JEV_VLM_MODEL"))
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--allow-screenshot-upload", action="store_true")
    parser.add_argument("--no-a11y", action="store_true")
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--max-steps", "--maxsteps", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=600)
    add_plan_arguments(parser)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/osworld") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args(argv)
    if args.no_a11y and args.image_only_fill_intermediates:
        parser.error("--image-only-fill-intermediates requires accessibility")
    args.osworld_root = args.osworld_root.resolve(strict=True)
    args.vm_path = validate_vm_path(args.vm_path)
    args.output = args.output.resolve()
    if not 0 <= args.pause <= 10:
        parser.error("--pause must be between 0 and 10 seconds")
    tasks = [(name, load_task(args.osworld_root, name)) for name in args.tasks]
    probe = make_desktop_planner(args)
    probe.close()
    # Import the existing checkout read-only, without creating/updating its pyc files.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.osworld_root))
    import pwd

    osworld_import_started = time.perf_counter()
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.docker.provider import DockerProvider

    from cua_jev.osworld_backend import OSWorldBackend

    osworld_import_wall_ms = (time.perf_counter() - osworld_import_started) * 1000
    owner = pwd.getpwuid(os.getuid()).pw_name
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("account name cannot safely form a Docker container name")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(args.osworld_root, args.tasks)
    metadata["osworld_import_wall_ms"] = osworld_import_wall_ms
    metadata["planning_by_mode"] = {mode: planning_settings(args, mode) for mode in args.modes}
    metadata["experiment_settings"] = experiment_settings(args)
    for name, _task in tasks:
        task_copy = args.output / "task_configs" / f"{name}.json"
        task_copy.parent.mkdir(parents=True, exist_ok=True)
        task_copy.write_bytes(
            (args.osworld_root / "evaluation_examples" / "examples" / f"{name}.json").read_bytes()
        )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reports = []
    run_id = uuid.uuid4().hex
    with isolated_docker(DockerProvider, args.vm_path, run_id, owner=owner):
        for name, task in tasks:
            for mode in args.modes:
                print(json.dumps({"event": "episode_start", "task": name, "mode": mode}), flush=True)
                report = run_episode(
                    args,
                    name,
                    task,
                    mode,
                    env_factory=DesktopEnv,
                    backend_factory=OSWorldBackend,
                    planner_factory=make_desktop_planner,
                )
                reports.append(report)
                payload = {
                    "kind": "osworld_integration_smoke",
                    "run_status": "success" if run_exit_code(reports) == 0 else "failed",
                    "run_id": run_id,
                    "official_evaluator": "once_after_agent",
                    **metadata,
                    "runs": reports,
                }
                (args.output / "summary.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(report, ensure_ascii=False), flush=True)
                if report.get("cleanup_status") != "confirmed":
                    return run_exit_code(reports)
    return run_exit_code(reports)


if __name__ == "__main__":
    raise SystemExit(main())
