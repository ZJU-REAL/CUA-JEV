"""Owned OSWorld VM worker over JSON-line stdin/stdout; no model/API proxy."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import posixpath
import re
import signal
import stat
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from sys import exc_info
from time import monotonic_ns

# This origin precedes project/OSWorld imports, but not interpreter/stdlib startup.
_WORKER_ENTRY = time.perf_counter()
_BOOT_BUDGET_ENTRY_NS = monotonic_ns()
if __name__ == "__main__":
    print('{"event":"worker_startup_phase","phase":"project_imports_started","elapsed_ms":0}',
          file=sys.stderr, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osworld_observation_profiler import timed_controller  # noqa: E402
from run_osworld_smoke import (  # noqa: E402
    bounded_vm_http,
    evaluate_once,
    isolated_docker,
    load_task,
    reset_checked,
    save_observation_artifacts,
    settling_summary,
    source_metadata,
    validate_vm_path,
    wait_for_desktop_ready,
    wait_for_desktop_settled,
)

from cua_jev.agent_planner import DESKTOP_KEYS, DESKTOP_MODIFIERS  # noqa: E402
from cua_jev.observation_budget import AccessibilityBudget  # noqa: E402
from cua_jev.osworld_backend import OSWorldBackend  # noqa: E402
from cua_jev.osworld_observation_order import (  # noqa: E402
    OBSERVATION_ORDERS,
    ORIGINAL_ORDER,
    OwnedObservationOrder,
)
from cua_jev.osworld_startup_budget import (  # noqa: E402
    BOOT_BUDGET_POLICIES,
    LEGACY_BOOT_BUDGET,
    SHARED_BOOT_BUDGET,
    OwnedBootWaitBudget,
)
from cua_jev.osworld_step_observation import (  # noqa: E402
    IMAGE_ONLY_INFO_KEY,
    IMAGE_ONLY_VERSION,
    native_image_only_step,
    validate_intermediate_action,
)

_WORKER_IMPORTS_DONE = time.perf_counter()

MAX_REQUEST = 65536


def record_startup_phase(timeline, phase):
    elapsed_ms = (time.perf_counter() - _WORKER_ENTRY) * 1000
    timeline["phases_ms"][phase] = elapsed_ms
    # Internal labels and numbers never contain a request or exception body.
    try:
        print(json.dumps({"event": "worker_startup_phase", "phase": phase, "elapsed_ms": elapsed_ms}),
              file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # A disconnected diagnostics pipe must never prevent owned-VM cleanup.
        pass


def emit_frame(stream, frame, *, startup_timeline=None):
    if startup_timeline is not None:
        record_startup_phase(startup_timeline, "ready_serialization_started")
    payload = json.dumps(frame, ensure_ascii=True, allow_nan=False) + "\n"
    if startup_timeline is not None:
        record_startup_phase(startup_timeline, "ready_write_started")
    stream.write(payload)
    if startup_timeline is not None:
        record_startup_phase(startup_timeline, "ready_write_completed")
    stream.flush()
    if startup_timeline is not None:
        record_startup_phase(startup_timeline, "ready_flush_completed")


class ProtocolError(ValueError):
    pass


class ActionError(RuntimeError):
    pass


class EvaluationError(RuntimeError):
    pass


def seed_asset_cache(task, asset_cache, output, *, owner_uid):
    """Copy only explicitly selected original assets into this run's new cache."""
    if asset_cache is None:
        return []
    source_root = Path(asset_cache)
    if source_root.is_symlink():
        raise PermissionError("asset cache must not be a symlink")
    source_root = source_root.resolve(strict=True)
    root_info = source_root.stat()
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != owner_uid:
        raise PermissionError("asset cache directory must belong to the current account")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id):
        raise ValueError("invalid task id for asset cache")
    records = []
    seen = set()
    for setup in task.get("config", []):
        if setup.get("type") != "download":
            continue
        for item in setup.get("parameters", {}).get("files", []):
            url, vm_path = item.get("url"), item.get("path")
            if not isinstance(url, str) or not url or not isinstance(vm_path, str):
                raise ValueError("invalid original download asset")
            basename = posixpath.basename(vm_path)
            if not basename or basename in {".", ".."} or "\\" in basename:
                raise ValueError("invalid original asset filename")
            filename = f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{basename}"
            if filename in seen:
                continue
            seen.add(filename)
            for candidate in (source_root / filename, source_root / task_id / filename):
                if not candidate.exists() and not candidate.is_symlink():
                    continue
                parent_info = candidate.parent.stat()
                if candidate.parent.is_symlink() or parent_info.st_uid != owner_uid:
                    raise PermissionError("asset cache parent must belong to the current account")
                if candidate.is_symlink():
                    raise PermissionError("cached asset must not be a symlink")
                source_info = candidate.stat()
                if not stat.S_ISREG(source_info.st_mode) or source_info.st_uid != owner_uid:
                    raise PermissionError("cached asset must be a regular file owned by the current account")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
                with os.fdopen(os.open(candidate, flags), "rb") as source:
                    opened = os.fstat(source.fileno())
                    if not stat.S_ISREG(opened.st_mode) or opened.st_uid != owner_uid:
                        raise PermissionError("cached asset ownership changed")
                    destination = Path(output) / "cache" / task_id / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    with destination.open("xb") as target:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            target.write(chunk)
                            size += len(chunk)
                records.append(
                    {
                        "source": str(candidate),
                        "destination": str(destination),
                        "sha256": digest.hexdigest(),
                        "size_bytes": size,
                    }
                )
                break
    return records


def public_observation(observation):
    if not isinstance(observation, dict):
        raise ProtocolError("invalid observation")
    screenshot = observation.get("screenshot")
    tree = observation.get("accessibility_tree")
    if not isinstance(screenshot, bytes) or not screenshot or len(screenshot) > 20_000_000:
        raise ProtocolError("invalid screenshot")
    if tree is not None and not isinstance(tree, str):
        raise ProtocolError("invalid accessibility tree")
    return {"screenshot": base64.b64encode(screenshot).decode("ascii"), "accessibility_tree": tree}


def validate_action(action, observation):
    from PIL import Image

    if not isinstance(action, dict) or set(action) != {"action_type", "parameters"}:
        raise ProtocolError("only typed computer_13 actions are accepted")
    kind, parameters = action["action_type"], action["parameters"]
    keys = {
        "CLICK": {"x", "y", "button", "num_clicks"},
        "TYPING": {"text"},
        "PRESS": {"key"},
        "HOTKEY": {"keys"},
        "SCROLL": {"dx", "dy"},
    }
    if not isinstance(kind, str) or kind not in keys or not isinstance(parameters, dict):
        raise ProtocolError("unsupported action")
    if set(parameters) != keys[kind]:
        raise ProtocolError("invalid action fields")
    valid_keys = DESKTOP_KEYS - {"'", "\\"}
    if kind == "CLICK":
        with Image.open(io.BytesIO(observation["screenshot"])) as image:
            width, height = image.size
        x, y = parameters["x"], parameters["y"]
        if not (
            type(x) is int
            and type(y) is int
            and 0 <= x < width
            and 0 <= y < height
            and isinstance(parameters["button"], str)
            and parameters["button"] in {"left", "right"}
            and type(parameters["num_clicks"]) is int
            and parameters["num_clicks"] in {1, 2}
            and (parameters["button"] == "left" or parameters["num_clicks"] == 1)
        ):
            raise ProtocolError("invalid click")
    elif kind == "TYPING":
        value = parameters["text"]
        if not isinstance(value, str) or not 0 < len(value) <= 2000 or not value.isascii() or "\x00" in value:
            raise ProtocolError("invalid text")
    elif kind == "PRESS":
        if not isinstance(parameters["key"], str) or parameters["key"] not in valid_keys:
            raise ProtocolError("invalid key")
    elif kind == "HOTKEY":
        value = parameters["keys"]
        if (
            not isinstance(value, list)
            or not 2 <= len(value) <= 4
            or not all(isinstance(key, str) for key in value)
            or len(set(value)) != len(value)
            or not all(key in DESKTOP_MODIFIERS for key in value[:-1])
            or value[-1] not in valid_keys
            or value[-1] in DESKTOP_MODIFIERS
        ):
            raise ProtocolError("invalid hotkey")
    elif kind == "SCROLL":
        dx, dy = parameters["dx"], parameters["dy"]
        if not (
            type(dx) is int and type(dy) is int and abs(dx) <= 10 and abs(dy) <= 10 and (dx == 0) != (dy == 0)
        ):
            raise ProtocolError("invalid scroll")
    return action


class ObservationCaptureTiming:
    """Time existing _get_obs/controller calls only; preserve their result/exception."""

    def __init__(self, env):
        self.env = env
        self.scope_name = "other"
        self.installed = False
        self.report = {
            "version": "owned_worker_original_controller_reads_v86",
            "clock": "monotonic_ns",
            "scope": "original _get_obs calls after DesktopEnv construction",
            "constructor_reads_covered": False,
            "component_failure_counts": "not observed by existing timed_controller",
            "HTTP_attempts_and_retries": "not observed by controller-method timing",
            "groups": {},
            "instrumentation_errors": {},
            "original_get_obs_restored": False,
        }
        try:
            self.previous = ("_get_obs" in vars(env), vars(env).get("_get_obs"))
            self.original = env._get_obs
            env._get_obs = self.read
            self.installed = True
        except Exception as error:
            self.note_error(self.report["instrumentation_errors"], error)
        self.report["installed"] = self.installed

    @staticmethod
    def note_error(errors, error):
        name = type(error).__name__
        errors[name] = errors.get(name, 0) + 1

    @contextmanager
    def scope(self, name):
        previous = self.scope_name
        self.scope_name = name
        try:
            yield
        finally:
            self.scope_name = previous

    def screenshot(self):
        # This read intentionally bypasses _get_obs; account for its cost too.
        timing = self.report.setdefault("standalone_screenshots", {
            "calls": 0, "failures": 0, "wall_ms": 0.0,
        })
        began = monotonic_ns()
        try:
            return self.env.controller.get_screenshot()
        except BaseException:
            timing["failures"] += 1
            raise
        finally:
            timing["calls"] += 1
            timing["wall_ms"] += max(0, monotonic_ns() - began) / 1_000_000

    def read(self, *args, **kwargs):
        group = self.report["groups"].setdefault(self.scope_name, {
            "get_obs_calls": 0, "get_obs_failures": 0, "get_obs_wall_ms": 0.0,
            "get_obs_error_types": {}, "controller_capture_failures": 0,
            "instrumentation_errors": {}, "controller_reads": {
                name: {"calls": 0, "wall_ms": 0.0}
                for name in ("screenshot", "accessibility_tree", "terminal")
            },
        })
        reads = {name: {"calls": 0, "wall_ms": 0.0}
                 for name in ("screenshot", "accessibility_tree", "terminal")}
        manager = None
        capture_failed = False
        try:
            manager = timed_controller(
                self.env.controller, reads, clock=lambda: monotonic_ns() / 1_000_000_000,
            )
            manager.__enter__()
        except Exception as error:
            manager = None
            capture_failed = True
            self.note_error(group["instrumentation_errors"], error)
        started = monotonic_ns()
        try:
            return self.original(*args, **kwargs)
        except BaseException as error:
            group["get_obs_failures"] += 1
            self.note_error(group["get_obs_error_types"], error)
            raise
        finally:
            # Preserve the same original exception even if diagnostic restoration fails.
            original_exception = exc_info()
            elapsed_ms = max(0, monotonic_ns() - started) / 1_000_000
            if manager is not None:
                try:
                    manager.__exit__(*original_exception)
                except Exception as error:
                    capture_failed = True
                    self.note_error(group["instrumentation_errors"], error)
            group["get_obs_calls"] += 1
            group["get_obs_wall_ms"] += elapsed_ms
            group["controller_capture_failures"] += int(capture_failed)
            for name, timing in reads.items():
                group["controller_reads"][name]["calls"] += timing["calls"]
                group["controller_reads"][name]["wall_ms"] += timing["wall_ms"]

    def restore(self):
        if not self.installed:
            return
        try:
            existed, previous = self.previous
            if existed:
                self.env._get_obs = previous
            else:
                del self.env._get_obs
            self.report["original_get_obs_restored"] = True
        except Exception as error:
            self.note_error(self.report["instrumentation_errors"], error)


class WorkerSession:
    def __init__(
        self, env, observation, output, *, guard_factory=OSWorldBackend, image_only_fill_intermediates=False,
        ax_timeout_s=None,
    ):
        if type(image_only_fill_intermediates) is not bool:
            raise ValueError("image-only fill option must be a boolean")
        if image_only_fill_intermediates and (
            type(getattr(env, "require_a11y_tree", None)) is not bool or not env.require_a11y_tree
        ):
            raise ValueError("image-only fill steps require enabled native accessibility")
        self.env, self.observation, self.output = env, observation, output
        self.capture_timing = None
        self.image_only_fill_intermediates = image_only_fill_intermediates
        self.guard = guard_factory(env)
        self.evaluated = self.closed = False
        self.steps = 0
        self.result = {"evaluation_status": "not_run", "official_score": None, "success": False}
        self.ax_budget = (AccessibilityBudget(env.controller, ax_timeout_s)
                          if ax_timeout_s is not None else None)

    def handle(self, request):
        if not isinstance(request, dict) or type(request.get("id")) is not int:
            raise ProtocolError("integer request id required")
        method = request.get("method")
        expected = {"id", "method", "action", "pause"} if method == "step" else {"id", "method"}
        if method == "step" and self.image_only_fill_intermediates:
            expected.add("observation_mode")
        if set(request) - expected or not {"id", "method"} <= set(request):
            raise ProtocolError("unknown request fields")
        if self.closed or (self.evaluated and method != "close"):
            raise ProtocolError("session is terminal")
        if method == "observe":
            with self.capture_scope("observe"):
                self.observation = self.env._get_obs()
            return public_observation(self.observation)
        if method == "screenshot":
            # Never pair these newer pixels with an older AX tree or mark them
            # as a complete observation. This read cannot validate semantics.
            with self.capture_scope("screenshot_only"):
                frame = (self.capture_timing.screenshot() if self.capture_timing is not None
                         else self.env.controller.get_screenshot())
            return public_observation({"screenshot": frame, "accessibility_tree": None})["screenshot"]
        if method == "step":
            pause = request.get("pause", 0.5)
            if type(pause) not in (int, float) or not math.isfinite(pause) or not 0 <= pause <= 10:
                raise ProtocolError("invalid pause")
            action = validate_action(request.get("action"), self.observation)
            image_only = "observation_mode" in request
            if image_only:
                if request["observation_mode"] != IMAGE_ONLY_VERSION:
                    raise ProtocolError("unknown step observation mode")
                validate_intermediate_action(action)
                with self.capture_scope("image_only_step"):
                    response = native_image_only_step(self.env, action, pause=pause)
            else:
                with self.capture_scope("full_step"):
                    response = self.env.step(action, pause=pause)
            if not isinstance(response, tuple) or len(response) != 4:
                raise ActionError("invalid step response")
            observation, _reward, _done, info = response
            if isinstance(info, dict) and (info.get("error") or info.get("success") is False):
                raise ActionError("remote action failed")
            self.steps += 1
            self.observation = observation
            (self.output / f"step-{self.steps:03}.png").write_bytes(observation["screenshot"])
            return {
                "observation": public_observation(observation), "reward": 0, "done": False,
                "info": {IMAGE_ONLY_INFO_KEY: IMAGE_ONLY_VERSION} if image_only else {},
            }
        if method == "evaluate":
            self.evaluated = True
            if self.ax_budget is not None:
                self.ax_budget.restore()
            (self.output / "final.png").write_bytes(self.observation["screenshot"])
            # Evaluation retains upstream behavior; the checked action wrapper
            # is only needed while accepting typed input calls.
            self.guard.close()
            with self.capture_scope("evaluation"):
                self.result = evaluate_once(self.env)
            self.save_result()
            if self.result["evaluation_status"] != "ok":
                raise EvaluationError("official evaluation failed")
            return self.result["official_score"]
        if method == "close":
            self.close()
            return None
        raise ProtocolError("unknown method")

    @contextmanager
    def capture_scope(self, name):
        if self.capture_timing is None:
            yield
        else:
            with self.capture_timing.scope(name):
                yield

    def save_result(self):
        if self.ax_budget is not None:
            (self.output / "observation-budget.json").write_text(
                json.dumps(self.ax_budget.report, indent=2), encoding="utf-8",
            )
        (self.output / "official-result.json").write_text(
            json.dumps({**self.result, "environment_steps": self.steps, "closed": self.closed}, indent=2),
            encoding="utf-8",
        )

    def close(self):
        if not self.closed:
            try:
                if self.ax_budget is not None:
                    self.ax_budget.restore()
                self.guard.close()
            finally:
                with self.capture_scope("close"):
                    self.env.close()
                self.closed = True
                self.save_result()


def serve(session, input_stream, output_stream):
    def send(frame):
        output_stream.write(json.dumps(frame, ensure_ascii=True, allow_nan=False) + "\n")
        output_stream.flush()

    while not session.closed:
        line = input_stream.readline(MAX_REQUEST + 1)
        if not line:
            break
        request = None
        try:
            if len(line) > MAX_REQUEST or not line.endswith("\n"):
                raise ProtocolError("request exceeds framing limit")
            request = json.loads(line)
            result = session.handle(request)
            send({"id": request["id"], "ok": True, "result": result})
        except Exception as error:
            identifier = request.get("id") if isinstance(request, dict) else None
            send(
                {
                    "id": identifier if type(identifier) is int else None,
                    "ok": False,
                    "error": type(error).__name__,
                }
            )
            if not isinstance(request, dict) or len(line) > MAX_REQUEST:
                break


def require_expected_user(expected_user, current_user):
    if not expected_user or current_user != expected_user:
        raise PermissionError("worker account does not match --expected-user")
    return current_user


@contextmanager
def owned_vm_context(args, owner):
    """Choose only the requested owned provider; never silently fall back."""
    if args.vm_provider == "qemu":
        from cua_jev.osworld_qemu import isolated_qemu

        with isolated_qemu(args.vm_path, args.output, owner) as runtime:
            if getattr(args, "guest_mcp_port", False):
                runtime.enable_mcp_port()
            if getattr(args, "qemu_boot_timeout", None) is not None:
                runtime.startup_timeout_s = args.qemu_boot_timeout
                runtime.metadata["qemu_boot_timeout_s"] = args.qemu_boot_timeout
            with bounded_vm_http(runtime.owned_ports):
                yield runtime.metadata
    elif args.vm_provider == "docker":
        from desktop_env.providers.docker.provider import DockerProvider

        boot_budget = None
        options = {}
        if getattr(args, "owned_boot_budget", LEGACY_BOOT_BUDGET) == SHARED_BOOT_BUDGET:
            boot_budget = OwnedBootWaitBudget(
                args.startup_timeout, worker_entry_ns=_BOOT_BUDGET_ENTRY_NS,
            )
            options["boot_wait_budget"] = boot_budget
        with isolated_docker(DockerProvider, args.vm_path, uuid.uuid4().hex, owner=owner, **options):
            runtime = {"actual_provider": "docker", "network_mode": "docker_bridge_loopback"}
            if boot_budget is not None:
                runtime["owned_boot_wait_budget"] = boot_budget.report
            yield runtime
    else:
        raise ValueError("unsupported VM provider")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-user", required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--vm-path", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lazy-evaluators", action="store_true",
                        help="load original evaluator exports only when used; fresh worker only")
    parser.add_argument("--guest-mcp-port", action="store_true",
                        help="QEMU only: forward guest MCP port 9292 to an owned loopback port")
    parser.add_argument("--asset-cache", type=Path)
    parser.add_argument("--owned-boot-budget", choices=BOOT_BUDGET_POLICIES, default=LEGACY_BOOT_BUDGET)
    parser.add_argument("--startup-timeout", type=float,
                        help="inherited existing outer ready budget; valid only with "
                             "explicit shared boot policy")
    parser.add_argument("--no-a11y", action="store_true")
    parser.add_argument("--ax-timeout", type=float, default=None,
                        help="agent-only AX budget in seconds; timeout opens a circuit until episode end")
    parser.add_argument("--observation-order", choices=OBSERVATION_ORDERS, default=ORIGINAL_ORDER,
                        help="explicit owned-instance acquisition-order experiment; default original")
    parser.add_argument("--image-only-fill-intermediates", action="store_true")
    parser.add_argument("--no-passive-settling", dest="passive_settling", action="store_false")
    parser.add_argument("--vm-provider", choices=("docker", "qemu"), default="docker")
    parser.add_argument("--qemu-boot-timeout", type=float, default=None,
                        help="explicit owned QEMU boot wait in seconds; default provider budget is 180")
    args = parser.parse_args(argv)
    if args.guest_mcp_port and args.vm_provider != "qemu":
        parser.error("--guest-mcp-port requires an owned QEMU instance")
    if args.qemu_boot_timeout is not None and (
        args.vm_provider != "qemu" or not math.isfinite(args.qemu_boot_timeout)
        or not 0 < args.qemu_boot_timeout <= 600
    ):
        parser.error("--qemu-boot-timeout requires QEMU and a finite budget in (0, 600]")
    if args.ax_timeout is not None and (
        not math.isfinite(args.ax_timeout) or not 0 < args.ax_timeout <= 60 or args.no_a11y
    ):
        parser.error("--ax-timeout requires accessibility and a finite budget in (0, 60]")
    if args.image_only_fill_intermediates and args.no_a11y:
        parser.error("--image-only-fill-intermediates requires accessibility")
    if args.owned_boot_budget == SHARED_BOOT_BUDGET:
        if (args.vm_provider != "docker" or args.startup_timeout is None
                or not math.isfinite(args.startup_timeout) or not 0 < args.startup_timeout <= 600):
            parser.error("shared boot budget requires Docker and inherited --startup-timeout in (0, 600]")
    elif args.startup_timeout is not None:
        parser.error("worker --startup-timeout requires explicit shared owned boot policy")
    return args


def main(argv=None):
    timeline = {
        "origin": "before_project_imports",
        "phases_ms": {
            "project_imports_started": 0,
            "project_imports_completed": (_WORKER_IMPORTS_DONE - _WORKER_ENTRY) * 1000,
        },
    }
    record_startup_phase(timeline, "main_entered")
    args = parse_args(argv)
    # Preserve the RPC pipe on a dedicated descriptor, then redirect both native
    # fd-1 writes and Python stdout from OSWorld/libraries to stderr.
    rpc = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def terminate(signum, _frame):
        # SSH disconnect/termination must unwind the owned-container contexts.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, terminate)

    env = session = capture_timing = observation_order = None
    evaluator_imports = ExitStack()
    metadata = {"startup_timeline": timeline}
    output_created = False
    try:
        import pwd

        owner = require_expected_user(args.expected_user, pwd.getpwuid(os.getuid()).pw_name)
        root = Path(__file__).resolve().parents[1]
        if root.stat().st_uid != os.getuid():
            raise PermissionError("worker project must belong to current account")
        args.output = args.output.resolve()
        if not args.output.is_relative_to(root) or args.output == root:
            raise PermissionError("output must be a new path within the worker project")
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        record_startup_phase(timeline, "output_created")
        args.osworld_root = args.osworld_root.resolve(strict=True)
        if args.osworld_root.stat().st_uid != os.getuid():
            raise PermissionError("OSWorld checkout must belong to current account")
        args.vm_path = validate_vm_path(args.vm_path)
        task = load_task(args.osworld_root, args.task)
        metadata.update(source_metadata(args.osworld_root, [args.task]))
        metadata["vm_provider"] = args.vm_provider
        metadata["preseeded_assets"] = seed_asset_cache(
            task, args.asset_cache, args.output, owner_uid=os.getuid()
        )
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (args.output / "task-config.json").write_bytes(
            (args.osworld_root / "evaluation_examples" / "examples" / f"{args.task}.json").read_bytes()
        )
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.osworld_root))
        if args.lazy_evaluators:
            from cua_jev.osworld_lazy_imports import lazy_evaluator_imports

            metadata["evaluator_imports"] = evaluator_imports.enter_context(
                lazy_evaluator_imports(args.osworld_root),
            )
        record_startup_phase(timeline, "osworld_import_started")
        import_started = time.perf_counter()
        from desktop_env.desktop_env import DesktopEnv

        metadata["osworld_import_wall_ms"] = (time.perf_counter() - import_started) * 1000
        record_startup_phase(timeline, "osworld_import_completed")
        with owned_vm_context(args, owner) as runtime_metadata:
            metadata["vm_runtime"] = runtime_metadata
            record_startup_phase(timeline, "provider_context_ready")
            try:
                started = time.perf_counter()
                env = DesktopEnv(
                    # Older harnesses whitelist this name for clean starts.
                    # The QEMU context replaces only this process's factory;
                    # actual provider/configuration is recorded separately.
                    provider_name="docker",
                    path_to_vm=str(args.vm_path),
                    action_space="computer_13",
                    cache_dir=str(args.output / "cache"),
                    headless=True,
                    require_a11y_tree=not args.no_a11y,
                    require_terminal=False,
                    enable_proxy=False,
                )
                metadata["startup_wall_ms"] = (time.perf_counter() - started) * 1000
                record_startup_phase(timeline, "environment_start_completed")
                observation_order = OwnedObservationOrder(
                    env, requested=getattr(args, "observation_order", ORIGINAL_ORDER),
                    osworld_root=args.osworld_root, owner_uid=os.getuid(),
                )
                metadata["observation_acquisition_order"] = observation_order.report
                capture_timing = ObservationCaptureTiming(env)
                metadata["original_observation_capture_timing"] = capture_timing.report
                started = time.perf_counter()
                with capture_timing.scope("startup_task_setup"):
                    observation = reset_checked(env, task)
                metadata["task_setup_wall_ms"] = (time.perf_counter() - started) * 1000
                record_startup_phase(timeline, "task_setup_completed")
                with capture_timing.scope("startup_desktop_readiness"):
                    observation = wait_for_desktop_ready(
                        env, observation, output=args.output, report=metadata,
                    )
                record_startup_phase(timeline, "desktop_readiness_completed")
                with capture_timing.scope("startup_passive_settling"):
                    observation = wait_for_desktop_settled(
                        env, observation, output=args.output, report=metadata,
                        enabled=getattr(args, "passive_settling", True),
                    )
                record_startup_phase(timeline, "passive_settling_completed")
                (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                save_observation_artifacts(args.output, "initial", observation, metadata)
                image_only = getattr(args, "image_only_fill_intermediates", False)
                session_options = {"image_only_fill_intermediates": True} if image_only else {}
                if getattr(args, "ax_timeout", None) is not None:
                    session_options["ax_timeout_s"] = args.ax_timeout
                session = WorkerSession(env, observation, args.output, **session_options)
                session.capture_timing = capture_timing
                if image_only:
                    metadata["intermediate_fill_observation"] = IMAGE_ONLY_VERSION
                record_startup_phase(timeline, "session_created")
                emit_frame(
                    rpc,
                    {
                        "event": "ready",
                        "protocol": 1,
                        "instruction": task["instruction"],
                        "observation": public_observation(observation),
                        "passive_settling": settling_summary(metadata["passive_settling"]),
                        "capabilities": {
                            "screenshot_only": "screenshot_v1",
                            **({"intermediate_fill_observation": IMAGE_ONLY_VERSION} if image_only else {}),
                        },
                    },
                    startup_timeline=timeline,
                )
                serve(session, sys.stdin, rpc)
            finally:
                record_startup_phase(timeline, "environment_close_started")
                try:
                    if session is not None:
                        session.close()
                    elif env is not None:
                        env.close()
                finally:
                    try:
                        if capture_timing is not None:
                            capture_timing.restore()
                    finally:
                        if observation_order is not None:
                            observation_order.restore()
                record_startup_phase(timeline, "environment_close_completed")
        return 0
    except Exception as error:
        diagnostic = {"error_type": type(error).__name__}
        if hasattr(error, "diagnostics"):
            diagnostic["startup_diagnostics"] = error.diagnostics
        if output_created:
            (args.output / "worker-error.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
        try:
            emit_frame(rpc, {"event": "fatal", "error": type(error).__name__})
        except (BrokenPipeError, OSError):
            pass
        return 1
    finally:
        evaluator_imports.close()
        record_startup_phase(timeline, "worker_finished")
        if output_created:
            try:
                (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            except OSError as error:
                print(json.dumps({"event": "metadata_write_error", "error_type": type(error).__name__}),
                      file=sys.stderr)
        rpc.close()


if __name__ == "__main__":
    raise SystemExit(main())
