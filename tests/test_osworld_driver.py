"""Lifecycle and evaluation isolation without Docker, VM, or model calls."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "osworld_driver", Path(__file__).parents[1] / "scripts" / "run_osworld_smoke.py"
)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


@pytest.fixture(autouse=True)
def isolate_settling(monkeypatch):
    # These lifecycle cases isolate reset/agent/evaluator handling; the real
    # bounded settling loop and ready-frame binding have independent tests.
    monkeypatch.setattr(driver, "wait_for_desktop_settled", lambda _env, initial, **_: initial)


def ready_observation():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(buffer, format="PNG")
    return {"screenshot": buffer.getvalue(), "accessibility_tree": "<desktop/>"}


def run_artifact_episode(monkeypatch, tmp_path, initial, steps):
    # These cases isolate artifact serialization (including deliberately invalid
    # image/object inputs); readiness has separate real-PNG lifecycle tests.
    monkeypatch.setattr(driver, "wait_for_desktop_ready", lambda _env, value, **_: value)
    events = []
    backend = SimpleNamespace(last_observation=None, close=lambda: events.append("backend_close"))

    def no_observation():
        pytest.fail("artifact recording must not fetch another observation")

    def final_screenshot():
        events.append("final_screenshot")
        return b"final PNG"

    env = SimpleNamespace(
        _get_obs=no_observation,
        controller=SimpleNamespace(get_screenshot=final_screenshot),
        evaluate=lambda: events.append("evaluate") or 1.0,
        close=lambda: events.append("environment_close"),
    )
    backend.observe = no_observation

    class Runner:
        def __init__(self, _backend, _planner, *, progress, **_):
            self.progress = progress

        def run(self, instruction, *, success):
            assert instruction == "Public goal" and success is None
            for observation in steps:
                backend.last_observation = observation
                self.progress({"event": "agent_step"})
            events.append("agent_finished")
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(driver, "AgentRunner", Runner)
    args = SimpleNamespace(
        output=tmp_path, model="test", vm_path=tmp_path / "base.qcow2", no_a11y=False,
        pause=0.5, max_steps=10, timeout=60,
    )
    report = driver.run_episode(
        args, "os/artifacts", {"instruction": "Public goal", "evaluator": {"private": "not evidence"}},
        "vlm", env_factory=lambda **_: env, backend_factory=lambda *_a, **_k: backend,
        planner_factory=lambda _: SimpleNamespace(close=lambda: events.append("planner_close")),
        initialize=lambda *_: initial,
    )
    assert report["run_status"] == "success" and report["cleanup_status"] == "confirmed"
    assert events == [
        "agent_finished", "final_screenshot", "evaluate",
        "backend_close", "environment_close", "planner_close",
    ]
    return report, tmp_path / "os-artifacts" / "vlm"


def test_existing_initial_and_step_observations_save_matching_raw_a11y_without_new_reads(
    monkeypatch, tmp_path,
):
    initial = {
        "screenshot": b"initial PNG",
        "accessibility_tree": '<desktop><label name="\u521d\u59cb"/></desktop>',
        "terminal": "private terminal",
        "instruction": "private setup information",
    }
    steps = [
        {"screenshot": b"first PNG", "accessibility_tree": '<desktop><check-box checked="false"/></desktop>'},
        {"screenshot": b"second PNG", "accessibility_tree": '<desktop><check-box checked="true"/></desktop>'},
    ]
    report, output = run_artifact_episode(monkeypatch, tmp_path, initial, steps)
    for stem, observation in zip(("initial", "step-001", "step-002"), (initial, *steps), strict=True):
        assert (output / f"{stem}.png").read_bytes() == observation["screenshot"]
        assert (output / f"{stem}-a11y.xml").read_bytes() == observation["accessibility_tree"].encode("utf-8")
    assert "artifact_errors" not in report
    assert not (output / "final-a11y.xml").exists()
    assert {path.name for path in output.iterdir()} == {
        "initial.png", "initial-a11y.xml", "step-001.png", "step-001-a11y.xml",
        "step-002.png", "step-002-a11y.xml", "final.png", "result.json",
    }


@pytest.mark.parametrize(
    ("tree", "error_type"),
    [
        (object(), "TypeError"),
        (b"<desktop/>", "TypeError"),
        ("a" * 2_000_001, "ValueError"),
        ("\u754c" * 666_667, "ValueError"),
        ("\ud800", "UnicodeEncodeError"),
    ],
    ids=("unknown-object", "binary-tree", "oversized-text", "oversized-utf8", "invalid-unicode"),
)
def test_invalid_a11y_artifacts_are_reported_without_stopping_episode(
    monkeypatch, tmp_path, tree, error_type,
):
    observation = {"screenshot": b"PNG", "accessibility_tree": tree}
    report, output = run_artifact_episode(monkeypatch, tmp_path, observation, [observation])
    assert report["artifact_errors"] == [
        {"artifact": "initial-a11y.xml", "error_type": error_type},
        {"artifact": "step-001-a11y.xml", "error_type": error_type},
    ]
    assert (output / "initial.png").read_bytes() == b"PNG"
    assert (output / "step-001.png").read_bytes() == b"PNG"
    assert not list(output.glob("*-a11y.xml"))
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["artifact_errors"] == report[
        "artifact_errors"
    ]


def test_a11y_artifact_size_boundary_counts_utf8_bytes(tmp_path):
    # An XML document that is exactly 2 MB in UTF-8, with fewer characters.
    prefix, suffix = "<desktop>", "</desktop>"
    content_bytes = 2_000_000 - len(prefix + suffix)
    tree = prefix + "\u754c" * (content_bytes // 3) + "x" * (content_bytes % 3) + suffix
    report = {}
    driver.save_observation_artifacts(tmp_path, "initial", {"accessibility_tree": tree}, report)
    assert (tmp_path / "initial-a11y.xml").stat().st_size == 2_000_000
    assert (tmp_path / "initial-a11y.xml").read_bytes() == tree.encode("utf-8")
    assert report == {}


def test_artifact_write_failure_preserves_other_surface_and_episode(monkeypatch, tmp_path):
    original = Path.write_bytes

    def fail_selected_artifact(path, value):
        if path.name in {"initial.png", "step-001-a11y.xml"}:
            raise OSError("private filesystem diagnostic")
        return original(path, value)

    monkeypatch.setattr(Path, "write_bytes", fail_selected_artifact)
    observation = {"screenshot": b"PNG", "accessibility_tree": "<desktop/>"}
    report, output = run_artifact_episode(monkeypatch, tmp_path, observation, [observation])
    assert report["artifact_errors"] == [
        {"artifact": "initial.png", "error_type": "OSError"},
        {"artifact": "step-001-a11y.xml", "error_type": "OSError"},
    ]
    assert (output / "initial-a11y.xml").read_text(encoding="utf-8") == "<desktop/>"
    assert (output / "step-001.png").read_bytes() == b"PNG"
    assert "private filesystem diagnostic" not in str(report)


def test_unknown_observation_and_screenshot_objects_do_not_break_recording(monkeypatch, tmp_path):
    report, output = run_artifact_episode(
        monkeypatch, tmp_path, object(), [{"screenshot": object(), "accessibility_tree": "<desktop/>"}]
    )
    assert report["artifact_errors"] == [
        {"artifact": "initial", "error_type": "TypeError"},
        {"artifact": "step-001.png", "error_type": "TypeError"},
    ]
    assert (output / "step-001-a11y.xml").read_text(encoding="utf-8") == "<desktop/>"


def test_missing_optional_a11y_is_not_an_artifact_error(monkeypatch, tmp_path):
    report, output = run_artifact_episode(
        monkeypatch, tmp_path, {"screenshot": b"initial PNG"},
        [{"screenshot": b"step PNG", "accessibility_tree": None}],
    )
    assert "artifact_errors" not in report
    assert not list(output.glob("*-a11y.xml"))


def test_late_public_instruction_is_saved_when_planning_fails_before_any_action(monkeypatch, tmp_path):
    task = {}
    args = SimpleNamespace(output=tmp_path, model="test", vm_path=tmp_path / "base.qcow2",
                           no_a11y=False, pause=0, max_steps=10, timeout=60)

    def factory(**kwargs):
        task.update(instruction="Change the requested setting", evaluator={"secret": "private-oracle"})
        return SimpleNamespace(close=lambda: None, evaluate=lambda: 0.0)

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, instruction, *, success):
            assert instruction == "Change the requested setting" and success is None
            return SimpleNamespace(to_dict=lambda: {"status": "planner_error", "steps": 0,
                                                    "planner_calls": 3, "success": False})

    monkeypatch.setattr(driver, "AgentRunner", Runner)
    report = driver.run_episode(
        args, "os/example", task, "vlm", env_factory=factory,
        backend_factory=lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
        planner_factory=lambda args: SimpleNamespace(close=lambda: None),
        initialize=lambda env, task: ready_observation(),
    )
    saved = json.loads((tmp_path / "os-example/vlm/result.json").read_bytes())
    assert report["instruction"] == saved["instruction"] == task["instruction"]
    assert saved["runner"]["steps"] == 0 and saved["evaluation_status"] == "ok"
    assert "private-oracle" not in json.dumps(saved) and "evaluator" not in saved


def test_injected_ablation_policy_uses_same_h6_runner_without_creating_jev(monkeypatch, tmp_path):
    events = []
    policy = SimpleNamespace(close=lambda: events.append("policy_closed"))
    monkeypatch.setattr(driver, "JevPolicy", lambda **_: pytest.fail("ablation must not construct Jev"))

    class Runner:
        def __init__(self, *args, **kwargs):
            assert kwargs["policy"] is policy
            assert kwargs["config"].mode == "hybrid" and kwargs["config"].plan_horizon == 6

        def run(self, instruction, success):
            assert success is None and instruction == "Public goal"
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(driver, "AgentRunner", Runner)
    env = SimpleNamespace(controller=SimpleNamespace(get_screenshot=lambda: b"PNG"),
                          evaluate=lambda: 0.0, close=lambda: None)
    args = SimpleNamespace(output=tmp_path, model="test", vm_path=tmp_path / "base.qcow2",
                           no_a11y=False, pause=0, max_steps=10, timeout=60)
    report = driver.run_episode(
        args, "os/ablation", {"instruction": "Public goal"}, "hybrid",
        env_factory=lambda **_: env, backend_factory=lambda *_a, **_k: SimpleNamespace(close=lambda: None),
        planner_factory=lambda _: SimpleNamespace(close=lambda: None),
        initialize=lambda *_: ready_observation(), policy_factory=lambda: policy,
    )
    assert report["official_score"] == 0.0 and events == ["policy_closed"]


@pytest.mark.parametrize(("mode", "settings", "horizon", "ttl"), [
    ("vlm", {}, 1, 300),
    ("hybrid", {}, 6, 300),
    ("hybrid", {"plan_horizon": 12, "plan_ttl": 0.5}, 12, 0.5),
    ("vlm", {"plan_horizon": 12, "plan_ttl": 0.5}, 1, 0.5),
    ("hybrid", {"reuse_verified_post_snapshot": False}, 6, 300),
    ("vlm", {"reuse_verified_post_snapshot": False}, 1, 300),
    ("hybrid", {"jev_network_recovery": True}, 6, 300),
    ("vlm", {"jev_network_recovery": True}, 1, 300),
    ("hybrid", {"jev_capacity_recovery": True, "reuse_verified_pending_snapshot": True,
                "whole_goal_recheck": True}, 6, 300),
    ("vlm", {"jev_capacity_recovery": True, "reuse_verified_pending_snapshot": True,
             "whole_goal_recheck": True}, 1, 300),
    ("hybrid", {"stale_recovery_on_progress": True}, 6, 300),
    ("vlm", {"stale_recovery_on_progress": True}, 1, 300),
])
def test_agent_never_receives_evaluator_and_score_is_independent(
    monkeypatch, tmp_path, mode, settings, horizon, ttl,
):
    events = []
    task = {"instruction": "Change theme", "evaluator": {"secret": "answer"}}

    class Env:
        setup_controller = SimpleNamespace(setup=lambda *_a, **_k: True)
        controller = SimpleNamespace(get_screenshot=lambda: b"PNG")

        def reset(self, *, task_config):
            assert task_config is task
            self.setup_controller.setup(task_config)
            events.append("setup")
            return ready_observation()

        def evaluate(self):
            assert events[-1] == "agent"
            events.append("evaluate")
            return 1.0

        def close(self):
            events.append("close")

    class Runner:
        def __init__(self, *_a, **_k):
            assert _k["config"].plan_horizon == horizon
            assert _k["config"].plan_ttl_s == ttl
            assert _k["config"].singleton_fast_path is True
            assert _k["config"].jev_network_recovery is settings.get("jev_network_recovery", False)
            assert _k["config"].jev_capacity_recovery is settings.get("jev_capacity_recovery", False)
            assert _k["config"].reuse_verified_pending_snapshot is settings.get(
                "reuse_verified_pending_snapshot", False,
            )
            assert _k["config"].whole_goal_recheck is settings.get("whole_goal_recheck", False)
            assert _k["config"].stale_recovery_on_progress is settings.get(
                "stale_recovery_on_progress", False,
            )
            assert _k["config"].reuse_verified_post_snapshot is settings.get(
                "reuse_verified_post_snapshot", True,
            )

        def run(self, instruction, *, success):
            assert instruction == "Change theme"
            assert success is None
            events.append("agent")
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(driver, "AgentRunner", Runner)
    args = SimpleNamespace(
        output=tmp_path,
        model="test",
        vm_path=tmp_path / "base.qcow2",
        no_a11y=False,
        pause=0.5,
        max_steps=10,
        timeout=60,
        **settings,
    )
    closeable = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(driver, "JevPolicy", lambda **_: closeable)
    report = driver.run_episode(
        args,
        "gimp/theme",
        task,
        mode,
        env_factory=lambda **_: Env(),
        backend_factory=lambda *_a, **_k: closeable,
        planner_factory=lambda _: closeable,
    )
    assert events == ["setup", "agent", "evaluate", "close"]
    assert report["success"] and report["official_score"] == 1.0
    assert report["runner"]["status"] == "unverified"
    assert not report["runner"]["success"]
    assert report["cleanup_status"] == "confirmed"
    assert report["run_status"] == "success"
    assert report["instruction"] == "Change theme" and "evaluator" not in report
    assert report["plan_horizon"] == horizon and report["plan_ttl_s"] == ttl
    assert report["experiment_settings"]["singleton_fast_path"] is True
    assert report["experiment_settings"]["desktop_snapshot_encoding"] == (
        "shared_ancestor_paths_v1_if_saving_4096_bytes" if settings.get("pool_desktop_ancestors", False)
        else "control_defaults_v1"
    )
    enabled = settings.get("reuse_verified_post_snapshot", True)
    assert report["experiment_settings"]["reuse_verified_post_snapshot"] is enabled
    assert report["experiment_settings"]["stale_observation_recovery"] == (
        "recompile_per_verified_input_v1" if settings.get("stale_recovery_on_progress", False)
        else "recompile_once_v2"
    )
    saved = json.loads((tmp_path / "gimp-theme" / mode / "result.json").read_text(encoding="utf-8"))
    assert saved["plan_horizon"] == horizon and saved["plan_ttl_s"] == ttl
    assert saved["experiment_settings"]["singleton_fast_path"] is True
    assert saved["experiment_settings"]["reuse_verified_post_snapshot"] is enabled
    assert driver.run_exit_code([report]) == 0


@pytest.mark.parametrize(("option", "value"), [
    ("--plan-horizon", "0"), ("--plan-horizon", "13"), ("--plan-horizon", "1.5"),
    ("--plan-ttl", "0"), ("--plan-ttl", "-1"), ("--plan-ttl", "301"),
    ("--plan-ttl", "nan"), ("--plan-ttl", "inf"),
])
def test_invalid_plan_budget_fails_before_environment_access(monkeypatch, tmp_path, option, value):
    monkeypatch.setattr(driver, "load_local_env", lambda: None)
    monkeypatch.setattr(driver, "validate_vm_path", lambda _: pytest.fail("no VM access"))
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("no model access"))
    with pytest.raises(SystemExit) as error:
        driver.main([
            "--osworld-root", str(tmp_path), "--vm-path", str(tmp_path / "base.qcow2"),
            "--tasks", "os/task", "--output", str(tmp_path / "run"), option, value,
        ])
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_failed_setup_never_runs_agent_or_evaluator(monkeypatch, tmp_path):
    events = []

    class Env:
        setup_controller = SimpleNamespace(setup=lambda *_a, **_k: False)

        def reset(self, **_):
            self.setup_controller.setup()

        def evaluate(self):
            pytest.fail("evaluation must not run after failed setup")

        def close(self):
            events.append("closed")

    args = SimpleNamespace(
        output=tmp_path,
        model="test",
        vm_path=tmp_path / "base.qcow2",
        no_a11y=True,
        pause=0.5,
        max_steps=10,
        timeout=60,
    )
    report = driver.run_episode(
        args,
        "os/example",
        {"instruction": "Goal"},
        "vlm",
        env_factory=lambda **_: Env(),
        backend_factory=lambda *_a, **_k: pytest.fail("no agent after failed setup"),
        planner_factory=lambda _: SimpleNamespace(close=lambda: None),
    )
    assert report["episode_status"] == "setup_error"
    assert report["evaluation_status"] == "not_run"
    assert report["official_score"] is None
    assert events == ["closed"]
    assert report["cleanup_status"] == "confirmed"
    assert report["run_status"] == "setup_error"
    assert driver.run_exit_code([report]) == 1


@pytest.mark.parametrize(("failed_resource", "reported_cleanup", "expected_cleanup", "run_status"), [
    ("environment", None, "unconfirmed_close_error", "cleanup_error"),
    ("environment", "unconfirmed_forced_termination", "unconfirmed_forced_termination", "cleanup_error"),
    ("planner", None, "confirmed", "cleanup_error"),
    (None, "unconfirmed_worker_exit", "unconfirmed_worker_exit", "cleanup_unconfirmed"),
])
def test_cleanup_failure_keeps_task_score_but_fails_run(
    monkeypatch, tmp_path, failed_resource, reported_cleanup, expected_cleanup, run_status,
):
    closed = []

    def close(name):
        closed.append(name)
        if name == failed_resource:
            raise RuntimeError("private cleanup diagnostic")

    env = SimpleNamespace(
        controller=SimpleNamespace(get_screenshot=lambda: b"PNG"),
        evaluate=lambda: 1.0,
        close=lambda: close("environment"),
    )
    if reported_cleanup is not None:
        env.cleanup_status = reported_cleanup
    monkeypatch.setattr(driver, "AgentRunner", lambda *_a, **_k: SimpleNamespace(
        run=lambda *_a, **_k: SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})
    ))
    args = SimpleNamespace(
        output=tmp_path, model="test", vm_path=tmp_path / "base.qcow2", no_a11y=False,
        pause=0.5, max_steps=10, timeout=60,
    )
    report = driver.run_episode(
        args, "os/cleanup", {"instruction": "Task"}, "vlm",
        env_factory=lambda **_: env,
        backend_factory=lambda *_a, **_k: SimpleNamespace(close=lambda: close("backend")),
        planner_factory=lambda _: SimpleNamespace(close=lambda: close("planner")),
        initialize=lambda *_: ready_observation(),
    )
    assert closed == ["backend", "environment", "planner"]
    assert report["evaluation_status"] == "ok" and report["official_score"] == 1.0
    assert report["success"] and report["episode_status"] == "success"
    assert report["cleanup_status"] == expected_cleanup
    assert report["run_status"] == run_status
    assert driver.run_exit_code([report]) == 1
    assert "private cleanup diagnostic" not in str(report)
    if failed_resource:
        assert report["cleanup_errors"] == ["RuntimeError"]
        assert report["cleanup_error_resources"] == [failed_resource]


def test_configuration_failure_does_not_claim_environment_was_started(tmp_path):
    def fail_configuration(_args):
        raise ValueError("invalid local model configuration")

    report = driver.run_episode(
        SimpleNamespace(output=tmp_path, model="test"), "os/task", {"instruction": "Task"}, "vlm",
        env_factory=lambda **_: pytest.fail("environment must not start"),
        backend_factory=lambda *_a, **_k: pytest.fail("backend must not start"),
        planner_factory=fail_configuration,
    )
    assert report["cleanup_status"] == "not_started"
    assert report["run_status"] == report["episode_status"] == "configuration_error"
    assert driver.run_exit_code([report]) == 1


def test_setup_retry_checks_recreated_controller_and_restores_both():
    def first_setup(*_a, **_k):
        return False

    def next_setup(*_a, **_k):
        return True

    first = SimpleNamespace(setup=first_setup)
    second = SimpleNamespace(setup=next_setup)

    class Env:
        setup_controller = first

        def _start_emulator(self):
            self.setup_controller = second

        def reset(self, **_):
            assert not self.setup_controller.setup()
            self._start_emulator()
            assert self.setup_controller.setup()
            return {"screenshot": b"test"}

    env = Env()
    original_start = env._start_emulator
    assert driver.reset_checked(env, {}) == {"screenshot": b"test"}
    assert first.setup is first_setup
    assert second.setup is next_setup
    assert env._start_emulator == original_start


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2, None])
def test_invalid_score_is_evaluation_error(score):
    calls = []
    env = SimpleNamespace(evaluate=lambda: calls.append(1) or score)
    report = driver.evaluate_once(env)
    assert calls == [1]
    assert report["evaluation_status"] == "error"
    assert report["official_score"] is None


def test_startup_error_preserves_bounded_owned_boot_diagnostics(tmp_path):
    log_calls = []
    container = SimpleNamespace(
        name="chenbofan-cua-jev-diag-test",
        status="exited",
        reload=lambda: None,
        attrs={"State": {"ExitCode": 1, "OOMKilled": False}},
        logs=lambda **kwargs: log_calls.append(kwargs) or b"boot failure\n" * 1000,
    )
    diagnostic = driver.startup_diagnostics(SimpleNamespace(container=container), RuntimeError("failure"))
    assert log_calls == [{"tail": 40}]
    assert len(diagnostic["boot_log_tail"]) == 8000
    assert diagnostic["container_name"].startswith("chenbofan-cua-jev-")
    assert diagnostic["status"] == "exited" and diagnostic["exit_code"] == 1

    def fail_start(**_):
        raise driver.VMStartupError(diagnostic)

    args = SimpleNamespace(
        output=tmp_path,
        model="test",
        vm_path=tmp_path / "base.qcow2",
        no_a11y=True,
        pause=0.5,
        max_steps=10,
        timeout=60,
    )
    report = driver.run_episode(
        args,
        "os/task",
        {"instruction": "Goal"},
        "vlm",
        env_factory=fail_start,
        backend_factory=lambda *_a, **_k: pytest.fail("no backend after startup failure"),
        planner_factory=lambda _: SimpleNamespace(close=lambda: None),
    )
    assert report["episode_status"] == "startup_error"
    assert report["startup_diagnostics"] == diagnostic
    assert report["evaluation_status"] == "not_run"


def test_container_scoping_limits_and_ownership_checked_cleanup(monkeypatch, tmp_path):
    base = tmp_path / "base.qcow2"
    base.write_bytes(b"base")
    monkeypatch.setattr(driver, "validate_vm_path", lambda _: base)
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: path.as_posix() == "/dev/kvm" or original_exists(path))
    calls = []

    class Container:
        def __init__(self, options):
            self.name = options["name"]
            self.attrs = {
                "Config": {"Labels": dict(options["labels"])},
                "NetworkSettings": {
                    "Ports": {
                        f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port + 20000)}]
                        for port in (5000, 8006, 8080, 9222)
                    }
                },
            }

        def reload(self):
            pass

        def start(self):
            pass

        def stop(self, **_):
            calls.append("stop")

        def remove(self):
            calls.append("remove")

    def create(image, **options):
        assert image == "happysixd/osworld-docker"
        assert options["mem_limit"] == "6g"
        assert options["nano_cpus"] == 2_000_000_000
        assert options["tmpfs"] == {"/storage": "rw,size=2g"}
        assert options["entrypoint"] == ["/usr/bin/tini", "-s", "--", "/bin/bash"]
        assert options["command"] == [
            "-c",
            "set -eu; ln -s /storage/boot.qcow2 /boot.qcow2; exec /run/entry.sh",
        ]
        assert options["volumes"] == {str(base): {"bind": "/System.qcow2", "mode": "ro"}}
        assert all(binding == ("127.0.0.1", None) for binding in options["ports"].values())
        return Container(options)

    class Provider:
        container = None
        client = SimpleNamespace(
            images=SimpleNamespace(get=lambda _: None), containers=SimpleNamespace(create=create)
        )

        def _wait_for_vm_ready(self, timeout):
            assert timeout == 180

        def start_emulator(self, *_a):
            pytest.fail("original provider must not start")

        def stop_emulator(self, *_a):
            pytest.fail("original provider must not stop")

    original = Provider.start_emulator
    with driver.isolated_docker(Provider, base, "run123", owner="chenbofan"):
        provider = Provider()
        provider.start_emulator(str(base), True, "Ubuntu")
        assert provider.server_port == 25000
        labels = provider.container.attrs["Config"]["Labels"]
        labels["cua-jev.owner"] = "another-user"
        with pytest.raises(RuntimeError, match="ownership"):
            provider.stop_emulator()
        assert calls == []
        labels["cua-jev.owner"] = "chenbofan"
        provider.stop_emulator()
    assert calls == ["stop", "remove"]
    assert Provider.start_emulator is original


@pytest.mark.parametrize("task", ["../secret", "os/../../secret", "/os/id", "os/id.json"])
def test_task_selection_cannot_escape_examples(tmp_path, task):
    with pytest.raises(ValueError):
        driver.load_task(tmp_path, task)


def test_created_container_is_registered_and_cleaned_when_start_fails(monkeypatch, tmp_path):
    base = tmp_path / "base.qcow2"
    base.write_bytes(b"base")
    monkeypatch.setattr(driver, "validate_vm_path", lambda _: base)
    monkeypatch.setattr(Path, "exists", lambda _: True)
    events = []

    class Container:
        status = "created"

        def __init__(self, options):
            self.name = options["name"]
            self.attrs = {"Config": {"Labels": options["labels"]}, "State": {"ExitCode": 0}}

        def start(self):
            assert provider.container is self
            events.append("start")
            raise RuntimeError("simulated Docker start failure")

        def reload(self):
            pass

        def logs(self, **_):
            return b""

        def stop(self, **_):
            events.append("stop")

        def remove(self):
            events.append("remove")

    class Provider:
        container = None
        client = SimpleNamespace(
            images=SimpleNamespace(get=lambda _: None),
            containers=SimpleNamespace(create=lambda _image, **opts: Container(opts)),
        )
        def start_emulator(self, *_):
            pass

        def stop_emulator(self, *_):
            pass

    with driver.isolated_docker(Provider, base, "start-failure", owner="chenbofan"):
        provider = Provider()
        with pytest.raises(driver.VMStartupError) as error:
            provider.start_emulator(str(base), True, "Ubuntu")
        assert error.value.diagnostics["status"] == "created"
        assert provider.container is None
    assert events == ["start", "stop", "remove"]


def test_outer_cleanup_tries_every_owned_container_when_one_fails(monkeypatch, tmp_path):
    base = tmp_path / "base.qcow2"
    base.write_bytes(b"base")
    monkeypatch.setattr(driver, "validate_vm_path", lambda _: base)
    monkeypatch.setattr(Path, "exists", lambda _: True)
    events, containers = [], []

    class Container:
        def __init__(self, options):
            self.index = len(containers)
            containers.append(self)
            self.name = options["name"]
            self.attrs = {
                "Config": {"Labels": options["labels"]},
                "NetworkSettings": {
                    "Ports": {
                        f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port + 20000 + self.index)}]
                        for port in (5000, 8006, 8080, 9222)
                    }
                },
            }

        def start(self):
            pass

        def reload(self):
            pass

        def stop(self, **_):
            events.append((self.index, "stop"))
            if self.index == 0:
                raise RuntimeError("simulated cleanup failure")

        def remove(self):
            events.append((self.index, "remove"))

    class Provider:
        container = None
        client = SimpleNamespace(
            images=SimpleNamespace(get=lambda _: None),
            containers=SimpleNamespace(create=lambda _image, **opts: Container(opts)),
        )
        def start_emulator(self, *_):
            pass

        def stop_emulator(self, *_):
            pass

        def _wait_for_vm_ready(self, *_a, **_k):
            pass

    original_start, original_stop = Provider.start_emulator, Provider.stop_emulator
    with pytest.raises(ExceptionGroup, match="cleanup failed") as error:
        with driver.isolated_docker(Provider, base, "cleanup-failure", owner="chenbofan"):
            Provider().start_emulator(str(base), True, "Ubuntu")
            Provider().start_emulator(str(base), True, "Ubuntu")
    assert len(error.value.exceptions) == 1
    assert events == [(0, "stop"), (1, "stop"), (1, "remove")]
    assert Provider.start_emulator is original_start and Provider.stop_emulator is original_stop


def test_http_timeout_only_applies_to_live_owned_loopback_ports(monkeypatch):
    import requests

    calls = []

    def original(session, method, url, *args, **kwargs):
        calls.append((url, args, kwargs))

    monkeypatch.setattr(requests.sessions.Session, "request", original)
    ports = set()
    session = requests.Session()
    with driver.bounded_vm_http(ports):
        session.get("http://127.0.0.1:35000/accessibility")
        assert calls[-1][2].get("timeout") is None
        ports.add(35000)
        session.get("http://127.0.0.1:35000/accessibility")
        assert calls[-1][2]["timeout"] == (10, 60)
        session.get("http://localhost:35000/terminal", timeout=300)
        assert calls[-1][2]["timeout"] == (10, 60)
        session.post("http://[::1]:35000/setup/upload", timeout=(None, 300))
        assert calls[-1][2]["timeout"] == (10, 120)
        session.get("http://127.0.0.1:35000/screenshot", timeout=(2, 5))
        assert calls[-1][2]["timeout"] == (2, 5)
        for url in (
            "http://localhost:35001/accessibility",
            "http://10.0.0.1:35000/v1/chat/completions",
            "https://assets.example.com/file",
            "http://localhost.evil.test:35000/",
        ):
            session.get(url, timeout=300)
            assert calls[-1][2]["timeout"] == 300
        ports.remove(35000)
        session.get("http://localhost:35000/terminal", timeout=300)
        assert calls[-1][2]["timeout"] == 300
    assert requests.sessions.Session.request is original


def test_http_timeout_restores_original_after_error_and_handles_positional_timeout(monkeypatch):
    import requests

    calls = []

    def original(session, method, url, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(requests.sessions.Session, "request", original)
    with pytest.raises(RuntimeError, match="test-error"):
        with driver.bounded_vm_http({35000}):
            requests.Session().request(
                "GET", "http://localhost:35000/accessibility", None, None, None, None, None, None, 300
            )
            assert calls[-1][0][6] == (10, 60)
            assert "timeout" not in calls[-1][1]
            raise RuntimeError("test-error")
    assert requests.sessions.Session.request is original
