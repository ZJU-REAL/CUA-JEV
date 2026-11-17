"""Reusable original-controller timer and a separate read-only diagnostic CLI.

The context manager may be used by an owned benchmark worker to time its
existing reads. The diagnostic CLI runs in a separate owned project, accepts
three profile reads and close only, and accepts no step/evaluate RPC.
"""

from __future__ import annotations

import hashlib
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timed_controller(controller, timing, *, clock=time.perf_counter):
    """Restore exact instance attributes even when a read/patch raises."""
    names = {"get_screenshot": "screenshot", "get_accessibility_tree": "accessibility_tree",
             "get_terminal_output": "terminal"}
    installed = []
    try:
        own = vars(controller)
        for attribute, component in names.items():
            original = getattr(controller, attribute, None)
            if not callable(original):
                raise TypeError("controller read instrumentation unavailable")
            previous = (attribute in own, own.get(attribute))

            def read(*args, original=original, component=component, **kwargs):
                started = clock()
                try:
                    return original(*args, **kwargs)
                finally:
                    timing[component]["calls"] += 1
                    timing[component]["wall_ms"] += max(0, clock() - started) * 1000

            setattr(controller, attribute, read)
            installed.append((attribute, previous))
        yield
    finally:
        for attribute, (existed, previous) in reversed(installed):
            if existed:
                setattr(controller, attribute, previous)
            else:
                delattr(controller, attribute)


def profiling_session(worker, *, clock=time.perf_counter):
    """Keep the original setup/cleanup, replacing only this child's RPC session."""
    class ProfileSession(worker.WorkerSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.profile_count = 0

        def handle(self, request):
            if (not isinstance(request, dict) or set(request) != {"id", "method"}
                    or type(request.get("id")) is not int or not isinstance(request.get("method"), str)):
                raise worker.ProtocolError("invalid read-only profile request")
            if request["method"] == "close":
                return super().handle(request)
            if request["method"] != "profile_observe":
                raise worker.ProtocolError("diagnostic worker accepts no input or evaluation")
            if self.closed or self.evaluated or self.profile_count >= 3:
                raise worker.ProtocolError("diagnostic observation budget exhausted or session terminal")
            self.profile_count += 1  # Failed reads also consume the bounded budget.
            total_started = clock()
            timing = {name: {"calls": 0, "wall_ms": 0.0}
                      for name in ("screenshot", "accessibility_tree", "terminal")}
            observation_started = clock()
            with timed_controller(self.env.controller, timing, clock=clock):
                observation = self.env._get_obs()
            observation_ms = max(0, clock() - observation_started) * 1000
            encoding_started = clock()
            public = worker.public_observation(observation)
            encoding_ms = max(0, clock() - encoding_started) * 1000
            self.observation = observation
            tree = observation.get("accessibility_tree")
            tree_bytes = tree.encode("utf-8") if isinstance(tree, str) else None
            summary = {
                "version": "observation_components_v1", "index": self.profile_count,
                "controller_reads": timing, "worker_observation_ms": observation_ms,
                "worker_public_encode_ms": encoding_ms,
                "screenshot_bytes": len(observation["screenshot"]),
                "accessibility_utf8_bytes": len(tree_bytes) if tree_bytes is not None else 0,
                "screenshot_sha256": hashlib.sha256(observation["screenshot"]).hexdigest(),
                "accessibility_sha256": hashlib.sha256(tree_bytes).hexdigest()
                if tree_bytes is not None else None,
            }
            summary["worker_total_ms"] = max(0, clock() - total_started) * 1000
            return {"observation": public, "diagnostic": summary}

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
