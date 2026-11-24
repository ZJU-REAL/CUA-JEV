"""Timing preserves actual local backend calls, guards, failures and frames."""

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_osworld_driver import driver, ready_observation
from test_osworld_fill_observation import Environment, offered

from cua_jev import config as local_config
from cua_jev import osworld_backend as module
from cua_jev.osworld_step_observation import native_image_only_step


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def blocked(*_args, **_kwargs):
        pytest.fail("network, subprocess, .env and real OCR are forbidden")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(driver, "load_local_env", blocked)
    monkeypatch.setattr(local_config, "load_local_env", blocked)


@pytest.fixture
def original():
    path = Path(__file__).parent / "fixtures/references/osworld_backend_v73.py"
    spec = importlib.util.spec_from_file_location("cua_jev._source73_timing_reference", path)
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    return source.OSWorldBackend


@pytest.mark.parametrize("mode", ["full", "native_image_only", "remote_image_only"])
def test_real_fill_call_order_freshness_timestamps_and_fingerprints_equal_original(
    original, monkeypatch, mode,
):
    outputs = []
    for cls in (original, module.OSWorldBackend):
        log, perf_values = [], []

        class Env(Environment):
            timing_log = log
            supports_image_only_steps = mode == "remote_image_only"

            def _get_obs(self):
                self.timing_log.append(("fetch", self.require_a11y_tree))
                return super()._get_obs()

            def step(self, action, pause):
                self.timing_log.append(("step", action, pause, self.require_a11y_tree))
                return super().step(action, pause)

            def step_image_only(self, action, pause):
                self.timing_log.append(("remote_image_only", action, pause))
                return native_image_only_step(self, action, pause=pause)

        def perf(log=log, perf_values=perf_values):
            value = 100.0 + len(perf_values) / 1000
            perf_values.append(value)
            log.append(("original_perf_counter", value))
            return value

        with monkeypatch.context() as patch:
            patch.setattr(module.time, "perf_counter", perf)
            env = Env()
            backend = cls(env, pause=0, image_only_fill_intermediates=mode != "full")
            build = backend._snapshot

            def snapshot(raw, log=log, build=build):
                log.append(("snapshot_build", raw["accessibility_tree"]))
                return build(raw)

            patch.setattr(backend, "_snapshot", snapshot)
            candidate = offered(backend)
            initial_received = backend._last_observed_received_at
            receipt = backend.execute(candidate, "observation", "decision")
            outputs.append((log, perf_values, receipt.success, backend.verify(candidate, receipt).passed,
                receipt.output["before_fingerprint"], receipt.output["after_fingerprint"],
                backend._last_observed_received_at, backend._action_observation_received_at,
                initial_received, env.actions, env.reads, backend.environment_steps))
        if cls is module.OSWorldBackend:
            phases = backend.observation_phase_metrics["phases"]
            assert phases["full_observe"]["acquisition"]["calls"] == 4
            assert phases["full_observe"]["snapshot_build"]["calls"] == 4
            assert phases["full_step"]["acquisition"]["calls"] == (3 if mode == "full" else 1)
            assert phases["image_only_step"]["acquisition"]["calls"] == (0 if mode == "full" else 2)
            assert all(t["failures"] == 0 and t["duration_ns"] >= 0
                       for bucket in phases.values() for t in bucket.values())
            detached = backend.observation_phase_metrics
            detached["phases"]["full_observe"]["acquisition"]["calls"] = -100
            assert backend.observation_phase_metrics["phases"]["full_observe"]["acquisition"]["calls"] == 4
    assert outputs[0] == outputs[1]
    assert outputs[1][2:4] == (True, True)


@pytest.mark.parametrize("stage", ["observe_fetch", "observe_build", "step_fetch", "step_build"])
def test_original_failure_identity_order_and_received_time_are_preserved(original, monkeypatch, stage):
    outcomes = []
    failure = RuntimeError("synthetic untrusted failure, not metric content")
    for cls in (original, module.OSWorldBackend):
        log = []
        env = Environment()
        backend = cls(env, pause=0)

        def fail(*_args, log=log, **_kwargs):
            log.append("failure")
            raise failure

        def perf(log=log):
            log.append("original_received_at")
            return 100.0

        with monkeypatch.context() as patch:
            patch.setattr(module.time, "perf_counter", perf)
            if stage == "observe_fetch":
                patch.setattr(env, "_get_obs", fail)
            elif stage == "step_fetch":
                patch.setattr(env, "step", fail)
            else:
                patch.setattr(backend, "_snapshot", fail)
            with pytest.raises(RuntimeError) as caught:
                if stage.startswith("observe"):
                    backend.observe()
                else:
                    backend._step("PRESS", {"key": "enter"})
            assert caught.value is failure
            outcomes.append((log, backend.environment_steps, backend.fresh_observe_calls,
                             backend._last_observed_received_at, backend._action_observation_received_at))
        if cls is module.OSWorldBackend:
            bucket = "full_observe" if stage.startswith("observe") else "full_step"
            phases = backend.observation_phase_metrics["phases"][bucket]
            assert phases["acquisition"]["calls"] == 1
            assert phases["acquisition"]["failures"] == int(stage.endswith("fetch"))
            assert phases["snapshot_build"]["calls"] == int(stage.endswith("build"))
            assert phases["snapshot_build"]["failures"] == int(stage.endswith("build"))
            assert "synthetic" not in str(backend.observation_phase_metrics)
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("present", [False, True])
def test_actual_episode_reports_supplied_metrics_only_without_fabricated_fake_backend_zeroes(
    monkeypatch, tmp_path, present,
):
    backend = SimpleNamespace(close=lambda: None)
    metrics = {"version": "actual-fixture", "phases": {"full_observe": {"calls": 7}}}
    if present:
        backend.observation_phase_metrics = metrics

    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(driver, "AgentRunner", Runner)
    monkeypatch.setattr(driver, "wait_for_desktop_ready", lambda _env, initial, **_kwargs: initial)
    monkeypatch.setattr(driver, "wait_for_desktop_settled", lambda _env, initial, **_kwargs: initial)
    env = SimpleNamespace(controller=SimpleNamespace(get_screenshot=lambda: b""),
                          evaluate=lambda: 0.0, close=lambda: None)
    args = SimpleNamespace(output=tmp_path, model="offline", vm_path=tmp_path / "unused",
                           no_a11y=False, pause=0, max_steps=5, timeout=60)
    report = driver.run_episode(args, "generic/timing", {"instruction": "Public goal"}, "vlm",
        env_factory=lambda **_kwargs: env, backend_factory=lambda *_a, **_k: backend,
        planner_factory=lambda _: SimpleNamespace(close=lambda: None),
        initialize=lambda *_args: ready_observation())
    assert report["episode_status"] == "task_failure"
    if present:
        assert report["observation_phase_timing"] == metrics
    else:
        assert "observation_phase_timing" not in report
