"""Opt-in input observation diagnostics in an owned isolated worker.

This entry point preserves the normal worker RPC and typed-input path. It adds a
bounded sidecar log only; use its timings for mechanism diagnosis, separately
from uninstrumented benchmark acceptance. The ordinary worker never imports it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from osworld_observation_profiler import timed_controller

LOG_NAME = "fill-observation-components.jsonl"
MAX_RECORDS = 512
VERSION = "fill_observation_rpc_v1"
_KINDS = {"CLICK", "HOTKEY", "TYPING", "PRESS", "SCROLL"}


def _flag(env):
    try:
        value = getattr(env, "require_a11y_tree", None)
        return value if type(value) is bool else None
    except Exception:
        return None


def _error_kind(error):
    name = type(error).__name__
    return name if name.isascii() and name.isidentifier() and len(name) <= 64 else "Exception"


def _capture(result, method):
    observation = result.get("observation") if method == "step" and isinstance(result, dict) else result
    if not isinstance(observation, dict):
        return {"capture_status": "unavailable"}
    image = base64.b64decode(observation["screenshot"], validate=True)
    tree = observation["accessibility_tree"]
    tree_bytes = tree.encode("utf-8") if isinstance(tree, str) else None
    return {
        "capture_status": "image_only_returned" if tree is None else "full_returned",
        "screenshot_bytes": len(image),
        "screenshot_sha256": hashlib.sha256(image).hexdigest(),
        "accessibility_utf8_bytes": len(tree_bytes) if tree_bytes is not None else 0,
        "accessibility_sha256": hashlib.sha256(tree_bytes).hexdigest() if tree_bytes is not None else None,
        "accessibility_tree_present": isinstance(tree, str),
        "semantic_accessibility_completeness": "not_assessed",
    }


def profiling_session(worker, *, clock=time.perf_counter, max_records=MAX_RECORDS):
    if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
        raise ValueError("invalid bounded observation profile limit")

    class ProfileSession(worker.WorkerSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._profile_count = 0
            self._profile_truncated = False
            self._profile_warned = False
            self._profile_stream = None
            try:
                self._profile_stream = (self.output / LOG_NAME).open("x", encoding="utf-8")
            except Exception:
                self._profile_warning("log_unavailable")
            self._profile_write(
                {
                    "event": "profile_header",
                    "version": VERSION,
                    "max_records": max_records,
                    "diagnostic_only": True,
                    "counter_semantics": "worker_successful_step_responses",
                    "logs_input_values": False,
                    "phase_assignment": "requires_client_or_receipt_binding",
                }
            )

        def _profile_warning(self, code):
            if self._profile_warned:
                return
            self._profile_warned = True
            try:
                print(
                    json.dumps({"event": "observation_profile_error", "code": code}),
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass

        def _profile_write(self, record):
            if self._profile_stream is None:
                return
            try:
                self._profile_stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
                self._profile_stream.flush()
            except Exception:
                self._profile_warning("log_write_failed")

        def handle(self, request):
            method = request.get("method") if isinstance(request, dict) else None
            if not isinstance(method, str) or method not in {"observe", "step"}:
                return super().handle(request)
            if self._profile_count >= max_records:
                if not self._profile_truncated:
                    self._profile_truncated = True
                    self._profile_write(
                        {"event": "profile_truncated", "version": VERSION, "max_records": max_records}
                    )
                return super().handle(request)
            self._profile_count += 1
            started = clock()
            timing = {
                name: {"calls": 0, "wall_ms": 0.0}
                for name in ("screenshot", "accessibility_tree", "terminal")
            }
            request_id = request.get("id")
            action = request.get("action")
            kind = action.get("action_type") if isinstance(action, dict) else None
            kind = kind if isinstance(kind, str) and kind in _KINDS else None
            mode = (
                "full_v1"
                if "observation_mode" not in request
                else "image_only_v1"
                if request["observation_mode"] == "image_only_v1"
                else "invalid"
            )
            record = {
                "event": "rpc_observation",
                "version": VERSION,
                "index": self._profile_count,
                "request_id": request_id if type(request_id) is int else None,
                "method": method,
                "action_type": kind,
                "observation_mode": mode,
                "environment_steps_before": self.steps,
                "a11y_flag_before": _flag(self.env),
                "phase": "unassigned",
                "controller_reads": timing,
            }
            instrumentation = None
            installed = False
            try:
                instrumentation = timed_controller(self.env.controller, timing, clock=clock)
                instrumentation.__enter__()
                installed = True
            except Exception:
                self._profile_warning("instrumentation_unavailable")
            record["instrumentation"] = "active" if installed else "unavailable"
            result = None
            try:
                result = super().handle(request)
                record["outcome"] = "ok"
                return result
            except BaseException as error:
                record.update(outcome="error", error_type=_error_kind(error))
                raise
            finally:
                if installed:
                    try:
                        instrumentation.__exit__(None, None, None)
                    except Exception:
                        record["instrumentation"] = "restore_failed"
                        self._profile_warning("instrumentation_restore_failed")
                record["environment_steps_after"] = self.steps
                record["a11y_flag_after"] = _flag(self.env)
                elapsed = max(0, clock() - started) * 1000
                record["worker_handle_before_log_ms"] = elapsed if math.isfinite(elapsed) else None
                try:
                    record.update(_capture(result, method))
                except Exception:
                    record["capture_status"] = "diagnostic_unavailable"
                    self._profile_warning("capture_diagnostic_unavailable")
                self._profile_write(record)

        def close(self):
            try:
                return super().close()
            finally:
                if self._profile_stream is not None:
                    self._profile_write(
                        {
                            "event": "profile_footer",
                            "version": VERSION,
                            "records": self._profile_count,
                            "truncated": self._profile_truncated,
                            "diagnostic_error": self._profile_warned,
                            "worker_closed": self.closed,
                        }
                    )
                    try:
                        self._profile_stream.close()
                    except Exception:
                        self._profile_warning("log_close_failed")
                    self._profile_stream = None

    return ProfileSession


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "scripts"))
    import osworld_remote_worker as worker

    if Path(worker.__file__).resolve() != project / "scripts/osworld_remote_worker.py":
        raise RuntimeError("diagnostic must use its own project worker")
    original = worker.WorkerSession
    worker.WorkerSession = profiling_session(worker)
    try:
        return worker.main(argv)
    finally:
        worker.WorkerSession = original


if __name__ == "__main__":
    raise SystemExit(main())
