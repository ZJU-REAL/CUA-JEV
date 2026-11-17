"""Local CLI wiring for SSH OSWorld episodes; no subprocess, VM, or model calls."""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cua_jev.osworld_remote import RemoteOSWorldError

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SETTINGS = {
    "TYPESAFE_API_KEY": "local-only-jev-key",
    "CUA_JEV_MODEL_API_KEY": "local-only-vlm-key",
    "CUA_JEV_MODEL_BASE_URL": "https://local-model.example.test/v1",
    "CUA_JEV_VLM_MODEL": "local-vision-model",
}


@pytest.fixture
def driver(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.chdir(tmp_path)
    for key in (*LOCAL_SETTINGS, "CUA_JEV_PLANNER_API_KEY", "CUA_JEV_MODEL", "CUA_JEV_API_URL"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(f"{key}={value}" for key, value in LOCAL_SETTINGS.items()), encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location(
        "remote_driver_test", ROOT / "scripts/run_osworld_remote.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RemoteOSWorldEnv", lambda **_: pytest.fail("unexpected SSH worker"))
    return module


def arguments(tmp_path, **overrides):
    options = {
        "host": "test-worker",
        "remote-user": "test-user",
        "remote-project": "/owned/CUA-JEV project",
        "osworld-root": "/owned/OSWorld",
        "vm-path": "/owned/OSWorld/docker_vm_data/Ubuntu.qcow2",
        "remote-output": "/owned/runs/current",
        "tasks": ["gimp/theme"],
        "modes": ["vlm"],
        "output": str(tmp_path / "run"),
    }
    options.update(overrides)
    result = []
    for name, value in options.items():
        result.append(f"--{name}")
        result.extend(value if isinstance(value, list) else [value])
    return [*result, "--allow-screenshot-upload"]


def install_episode_fakes(monkeypatch, driver, *, cleanup_status="confirmed", fail_cleanup=False):
    smoke = importlib.import_module("run_osworld_smoke")
    calls = {"planners": [], "policies": [], "environments": [], "goals": [], "configs": []}
    image = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(image, format="PNG")
    calls["initial_png"] = image.getvalue()

    class Planner:
        def __init__(self, base_url, model, **settings):
            self.base_url, self.model, self.settings = base_url, model, settings
            self.closed = False
            calls["planners"].append(self)

        def close(self):
            self.closed = True

    class Policy:
        def __init__(self, **settings):
            assert os.getenv("TYPESAFE_API_KEY") == LOCAL_SETTINGS["TYPESAFE_API_KEY"]
            self.settings = settings
            self.closed = False
            calls["policies"].append(self)

        def close(self):
            self.closed = True

    class Environment:
        def __init__(self, *, host, command, log_path, startup_timeout_s):
            self.host, self.command, self.log_path = host, command, log_path
            self.startup_timeout_s = startup_timeout_s
            selection = command[command.index("--task") + 1]
            self.instruction = f"Actual instruction returned by worker: {selection}"
            self.initial_observation = {"screenshot": image.getvalue(), "accessibility_tree": "<desktop/>"}
            evidence = {}
            smoke.wait_for_desktop_settled(self, self.initial_observation, output=log_path.parent,
                                          report=evidence, enabled=False)
            self.passive_settling = smoke.settling_summary(evidence["passive_settling"])
            if "--no-passive-settling" not in command:
                self.passive_settling.update(enabled=True, status="budget_exhausted", reason="sample_limit",
                                             elapsed_ms=9000, samples_observed=10)
            self.controller = SimpleNamespace(get_screenshot=lambda: b"final-test-image")
            self.evaluations = 0
            self.closed = False
            self.cleanup_status = "not_requested"
            calls["environments"].append(self)

        def reset(self, **_):
            pytest.fail("the remote worker already initialized its task")

        def evaluate(self):
            self.evaluations += 1
            return 1.0

        def close(self):
            self.closed = True
            self.cleanup_status = cleanup_status
            if fail_cleanup:
                raise RuntimeError("private SSH cleanup details")

    class Backend:
        def __init__(self, env, *, pause, menu_text_provider=None, image_only_fill_intermediates=False):
            self.env = env
            self.last_observation = env.initial_observation
            self.menu_grounding_metrics = {"attempts": 0}
            self.intermediate_observation_metrics = {"image_only_steps": 0}
            calls.setdefault("image_only_options", []).append(image_only_fill_intermediates)
            calls.setdefault("menu_providers", []).append(menu_text_provider)

        def close(self):
            pass

    class Runner:
        def __init__(self, backend, planner, *, policy, config, **_):
            self.env = backend.env
            calls["configs"].append(config)
            assert (policy is not None) == (config.mode == "hybrid")

        def run(self, goal, *, success):
            calls["goals"].append(goal)
            assert goal == self.env.instruction
            assert success is None
            assert self.env.evaluations == 0
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(smoke, "MultimodalPlanner", Planner)
    monkeypatch.setattr(smoke, "JevPolicy", Policy)
    monkeypatch.setattr(smoke, "AgentRunner", Runner)
    monkeypatch.setattr(driver, "RemoteOSWorldEnv", Environment)
    monkeypatch.setattr(driver, "OSWorldBackend", Backend)
    return calls


def test_local_config_stays_local_and_worker_instruction_reaches_goal(driver, monkeypatch, tmp_path, capsys):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(tmp_path, modes=["hybrid", "vlm"])) == 0
    assert len(calls["planners"]) == 3  # One local configuration probe and one planner per episode.
    for planner in calls["planners"]:
        assert planner.base_url == LOCAL_SETTINGS["CUA_JEV_MODEL_BASE_URL"]
        assert planner.model == LOCAL_SETTINGS["CUA_JEV_VLM_MODEL"]
        assert planner.settings["api_key"] == LOCAL_SETTINGS["CUA_JEV_MODEL_API_KEY"]
        assert planner.settings["profile"] == "desktop"
        assert planner.closed
    assert len(calls["policies"]) == 1 and calls["policies"][0].closed
    assert [(config.plan_horizon, config.plan_ttl_s) for config in calls["configs"]] == [
        (6, 300), (1, 300),
    ]
    assert calls["goals"] == ["Actual instruction returned by worker: gimp/theme"] * 2
    for env, mode in zip(calls["environments"], ("hybrid", "vlm"), strict=True):
        assert env.host == "test-worker"
        assert env.command == [
            "/owned/CUA-JEV project/.venv/bin/python", "-B", "-u",
            "/owned/CUA-JEV project/scripts/osworld_remote_worker.py",
            "--expected-user", "test-user",
            "--osworld-root", "/owned/OSWorld",
            "--vm-path", "/owned/OSWorld/docker_vm_data/Ubuntu.qcow2",
            "--vm-provider", "docker",
            "--task", "gimp/theme", "--output", f"/owned/runs/current/gimp-theme/{mode}",
        ]
        assert not any(value in " ".join(env.command) for value in LOCAL_SETTINGS.values())
        assert env.log_path == tmp_path / "run" / "gimp-theme" / mode / "worker.log"
        assert env.evaluations == 1 and env.closed
    output = capsys.readouterr().out
    assert LOCAL_SETTINGS["CUA_JEV_MODEL_API_KEY"] not in output
    assert LOCAL_SETTINGS["TYPESAFE_API_KEY"] not in output
    assert LOCAL_SETTINGS["CUA_JEV_MODEL_BASE_URL"] not in output


def test_image_only_fill_option_reaches_worker_and_backend_in_both_modes(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["hybrid", "vlm"]) + ["--image-only-fill-intermediates"]
    assert driver.main(argv) == 0
    assert calls["image_only_options"] == [True, True]
    assert all("--image-only-fill-intermediates" in env.command for env in calls["environments"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["intermediate_fill_observation"] == "image_only_v1"
    assert all(row["intermediate_observation"] == {"image_only_steps": 0} for row in summary["runs"])


def test_image_only_fill_without_a11y_stops_before_worker_or_model(driver, tmp_path):
    with pytest.raises(SystemExit):
        driver.main(arguments(tmp_path) + ["--image-only-fill-intermediates", "--no-a11y"])
    assert not (tmp_path / "run").exists()


def test_observation_profiler_has_explicit_worker_and_comparison_configuration(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["hybrid", "vlm"]) + ["--profile-observation-costs"]
    assert driver.main(argv) == 0
    assert calls["image_only_options"] == [False, False]
    assert all(env.command[3] == "/owned/CUA-JEV project/scripts/osworld_fill_observation_profiler.py"
               for env in calls["environments"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["worker_instrumentation"] == "fill_observation_rpc_v1"
    assert all(row["experiment_settings"]["worker_instrumentation"] == "fill_observation_rpc_v1"
               for row in summary["runs"])


def test_remote_episode_metadata_and_per_task_outputs(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, tasks=["gimp/theme", "os/quiet"])
    assert driver.main([*argv, "--no-a11y"]) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "osworld_integration_smoke"
    assert summary["run_status"] == "success"
    assert summary["decision_location"] == "local" and summary["vm_transport"] == "ssh_stdio"
    assert summary["planning_by_mode"] == {"vlm": {"plan_horizon": 1, "plan_ttl_s": 300}}
    metadata = json.loads((tmp_path / "run/metadata.json").read_text(encoding="utf-8"))
    assert metadata["planning_by_mode"] == summary["planning_by_mode"]
    assert [row["task"] for row in summary["runs"]] == ["gimp/theme", "os/quiet"]
    assert len(calls["environments"]) == 2
    for env, row in zip(calls["environments"], summary["runs"], strict=True):
        assert env.command[-1] == "--no-a11y"
        task_directory = row["task"].replace("/", "-")
        episode = tmp_path / "run" / task_directory / "vlm"
        assert row["decision_location"] == "local" and row["vm_transport"] == "ssh_stdio"
        assert row["remote_output"] == f"/owned/runs/current/{task_directory}/vlm"
        assert row["success"] and row["official_score"] == 1.0
        assert row["plan_horizon"] == 1 and row["plan_ttl_s"] == 300
        assert row["cleanup_status"] == "confirmed" and row["run_status"] == "success"
        assert not row["runner"]["success"]
        assert json.loads((episode / "result.json").read_text(encoding="utf-8")) == row
        assert (episode / "initial.png").read_bytes() == calls["initial_png"]


def test_plan_budgets_are_recorded_and_baseline_still_uses_one_action(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(
        tmp_path, modes=["hybrid", "vlm"], **{"plan-horizon": "12", "plan-ttl": "0.5"},
    )) == 0
    assert [(config.plan_horizon, config.plan_ttl_s) for config in calls["configs"]] == [
        (12, 0.5), (1, 0.5),
    ]
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["planning_by_mode"] == {
        "hybrid": {"plan_horizon": 12, "plan_ttl_s": 0.5},
        "vlm": {"plan_horizon": 1, "plan_ttl_s": 0.5},
    }
    for env, report in zip(calls["environments"], summary["runs"], strict=True):
        actual = summary["planning_by_mode"][report["mode"]]
        assert {key: report[key] for key in actual} == actual
        assert "--plan-horizon" not in env.command and "--plan-ttl" not in env.command


@pytest.mark.parametrize(("setting", "expected"), [("default", None), ("on", True), ("off", False)])
@pytest.mark.parametrize("pooling", [False, True])
@pytest.mark.parametrize("compaction", [False, True])
def test_model_thinking_applies_equally_to_modes_and_stays_out_of_worker(
    driver, monkeypatch, tmp_path, setting, expected, pooling, compaction,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["hybrid", "vlm"], **{"model-thinking": setting})
    if pooling:
        argv.append("--pool-desktop-ancestors")
    if compaction:
        argv.append("--compact-desktop-controls")
    assert driver.main(argv) == 0
    assert len(calls["planners"]) == 3
    assert all(planner.settings["enable_thinking"] is expected for planner in calls["planners"])
    assert all(planner.settings["request_timeout_s"] == (90.0 if setting == "on" else 40.0)
               for planner in calls["planners"])
    assert all(planner.settings["pool_desktop_ancestors"] is pooling for planner in calls["planners"])
    assert all(planner.settings["compact_desktop_controls"] is compaction for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    expected_settings = {
        "vm_provider": "docker", "vm_network_mode": "docker_bridge_loopback",
        "singleton_fast_path": True,
        "reuse_verified_post_snapshot": True,
        "intermediate_fill_observation": "full_v1",
        "worker_instrumentation": "disabled",
        "jev_network_recovery": "disabled",
        "jev_capacity_recovery": "disabled",
        # New opt-in metadata is recorded even when every feature stays off.
        "jev_capacity_recovery_policy": "vlm_once_v1",
        "navigation_continuation": False,
        "navigation_continuation_version": "disabled",
        "reuse_verified_pending_snapshot": False,
        "whole_goal_recheck": "disabled",
        "reuse_whole_goal_planning_snapshot": False,
        "whole_goal_planning_snapshot_reuse_version": "disabled",
        "inband_whole_goal_recheck": False,
        "inband_whole_goal_recheck_version": "disabled",
        "jev_request_layout": "full_v1",
        "desktop_stage_contract": "legacy_v23",
        "desktop_output_format": "unconstrained_v1",
        "desktop_normal_stage_policy": "program_optional",
        "observation_order": "original",
        "desktop_planning_observation": "full_multimodal_v1",
        "visual_box_space": "normalized_1000",
        "desktop_snapshot_encoding": (
            ("shared_ancestors_and_controls_v1_if_saving_4096_bytes" if pooling
             else "shared_control_fields_v1_if_saving_4096_bytes") if compaction else
            "shared_ancestor_paths_v1_if_saving_4096_bytes" if pooling else "control_defaults_v1"
        ),
        "stale_observation_recovery": "recompile_once_v2",
        "desktop_observation_budget": "names2k_text96k_v1",
        "menu_grounding": "disabled", "menu_ocr_worker_sha256": "disabled",
        "menu_ocr_timeout_s": "disabled",
        "menu_ocr_process_layout": "disabled",
        "passive_settling": True, "passive_settling_version": "public_structure_v3",
        "model_thinking": setting, "pause_s": 0.5, "max_steps": 25, "timeout_s": 420, "a11y": True,
        "model_thinking_budget_tokens": "default",
        "model_request_timeout_s": 90.0 if setting == "on" else 40.0,
        "startup_timeout_s": 300,
    }
    assert summary["experiment_settings"] == expected_settings
    for row, env in zip(summary["runs"], calls["environments"], strict=True):
        assert row["experiment_settings"] == expected_settings
        assert "--model-thinking" not in env.command
        assert "--pool-desktop-ancestors" not in env.command
        assert "--compact-desktop-controls" not in env.command


@pytest.mark.parametrize("timeout_s", [40.0, 90.0])
def test_model_wait_is_recorded_and_forwarded_equally_without_changing_episode_budgets(
    driver, monkeypatch, tmp_path, timeout_s,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"],
                     **{"model-request-timeout": str(timeout_s), "model-thinking": "on"})
    assert driver.main(argv) == 0
    assert len(calls["planners"]) == 3
    assert all(p.settings["request_timeout_s"] == timeout_s for p in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["model_request_timeout_s"] == timeout_s
    for row, env in zip(summary["runs"], calls["environments"], strict=True):
        assert row["experiment_settings"]["model_request_timeout_s"] == timeout_s
        assert row["experiment_settings"]["timeout_s"] == 420
        assert "--model-request-timeout" not in env.command
    assert all(c.timeout_s == 420 for c in calls["configs"])


@pytest.mark.parametrize("enabled", [True, False])
def test_post_snapshot_reuse_applies_equally_to_modes_and_all_experiment_records(
    driver, monkeypatch, tmp_path, enabled,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"])
    if not enabled:
        argv.append("--no-reuse-verified-post-snapshot")
    assert driver.main(argv) == 0
    assert [config.reuse_verified_post_snapshot for config in calls["configs"]] == [enabled, enabled]
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "run/metadata.json").read_text(encoding="utf-8"))
    key = "reuse_verified_post_snapshot"
    assert summary["experiment_settings"][key] is enabled
    assert metadata["experiment_settings"][key] is enabled
    for row, env in zip(summary["runs"], calls["environments"], strict=True):
        assert row["experiment_settings"][key] is enabled
        stored = tmp_path / "run" / row["task"].replace("/", "-") / row["mode"] / "result.json"
        assert json.loads(stored.read_text(encoding="utf-8"))["experiment_settings"][key] is enabled
        assert "--no-reuse-verified-post-snapshot" not in env.command


@pytest.mark.parametrize("enabled", [True, False])
def test_jev_network_recovery_is_explicit_local_and_recorded_for_both_modes(
    driver, monkeypatch, tmp_path, enabled,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"])
    if enabled:
        argv.append("--jev-network-recovery")
    assert driver.main(argv) == 0
    assert [config.jev_network_recovery for config in calls["configs"]] == [enabled, enabled]
    expected = "vlm_once_v1" if enabled else "disabled"
    for name in ("summary.json", "metadata.json"):
        saved = json.loads((tmp_path / "run" / name).read_text(encoding="utf-8"))
        assert saved["experiment_settings"]["jev_network_recovery"] == expected
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    for row, env in zip(summary["runs"], calls["environments"], strict=True):
        assert row["experiment_settings"]["jev_network_recovery"] == expected
        stored = tmp_path / "run" / row["task"].replace("/", "-") / row["mode"] / "result.json"
        settings = json.loads(stored.read_text(encoding="utf-8"))["experiment_settings"]
        assert settings["jev_network_recovery"] == expected
        assert "--jev-network-recovery" not in env.command


@pytest.mark.parametrize("enabled", [True, False])
def test_passive_settling_setting_reaches_worker_and_both_mode_reports_without_resampling(
    driver, monkeypatch, tmp_path, enabled,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"])
    if not enabled:
        argv.append("--no-passive-settling")
    assert driver.main(argv) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    for row, env in zip(summary["runs"], calls["environments"], strict=True):
        assert ("--no-passive-settling" not in env.command) is enabled
        assert row["passive_settling"] == env.passive_settling
        assert row["experiment_settings"]["passive_settling"] is enabled
        assert row["experiment_settings"]["passive_settling_version"] == "public_structure_v3"
        assert row["experiment_settings"]["timeout_s"] == 420
        assert row["experiment_settings"]["startup_timeout_s"] == 300
        assert "--no-reuse-verified-post-snapshot" not in env.command
    assert summary["experiment_settings"]["passive_settling"] is enabled


@pytest.mark.parametrize("provider,network", [
    ("docker", "docker_bridge_loopback"), ("qemu", "user_restricted_loopback"),
])
def test_vm_provider_is_explicit_in_worker_command_and_comparison_metadata(
    driver, monkeypatch, tmp_path, provider, network,
):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(tmp_path, modes=["vlm", "hybrid"], **{"vm-provider": provider})) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["vm_provider"] == provider
    assert summary["experiment_settings"]["vm_network_mode"] == network
    for env, report in zip(calls["environments"], summary["runs"], strict=True):
        assert env.command[env.command.index("--vm-provider") + 1] == provider
        assert report["experiment_settings"]["vm_provider"] == provider
        assert report["experiment_settings"]["vm_network_mode"] == network


def test_batch_does_not_start_another_vm_after_unconfirmed_cleanup(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver, cleanup_status="unconfirmed_forced_termination")
    assert driver.main(arguments(tmp_path, tasks=["os/one", "os/two"], modes=["vlm", "hybrid"])) == 1
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert len(calls["environments"]) == len(summary["runs"]) == 1
    assert summary["runs"][0]["cleanup_status"] == "unconfirmed_forced_termination"
    assert summary["runs"][0]["official_score"] == 1
    assert not (tmp_path / "run/os-two").exists()


def test_remote_startup_error_is_retained_with_local_transport_metadata(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)

    class StartupFailure(RuntimeError):
        pass

    def fail_worker(**_):
        raise StartupFailure("private remote diagnostic remains in worker log")

    monkeypatch.setattr(driver, "RemoteOSWorldEnv", fail_worker)
    assert driver.main(arguments(tmp_path)) == 1
    assert calls["goals"] == []
    assert all(planner.closed for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    report = summary["runs"][0]
    assert report["episode_status"] == "startup_error"
    assert report["cleanup_status"] == "unconfirmed_startup"
    assert report["run_status"] == "cleanup_unconfirmed"
    assert summary["run_status"] == "failed"
    assert report["error_type"] == "StartupFailure"
    assert report["evaluation_status"] == "not_run" and report["official_score"] is None
    assert report["runner"] is None and not report["success"]
    assert report["decision_location"] == "local" and report["vm_transport"] == "ssh_stdio"
    assert report["remote_output"] == "/owned/runs/current/gimp-theme/vlm"
    stored = (tmp_path / "run/gimp-theme/vlm/result.json").read_text(encoding="utf-8")
    assert json.loads(stored) == report
    assert "private remote diagnostic" not in stored


@pytest.mark.parametrize("budget", ["300", "600", "0.5"])
def test_startup_budget_is_independent_of_agent_budget_and_recorded(driver, monkeypatch, tmp_path, budget):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(tmp_path, **{"startup-timeout": budget}, modes=["vlm", "hybrid"])) == 0
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert all(env.startup_timeout_s == float(budget) for env in calls["environments"])
    assert all(config.timeout_s == 420 for config in calls["configs"])
    assert summary["experiment_settings"]["startup_timeout_s"] == float(budget)
    for row in summary["runs"]:
        assert row["experiment_settings"]["startup_timeout_s"] == float(budget)
        assert row["experiment_settings"]["timeout_s"] == 420


def test_startup_diagnostics_survive_constructor_failure_and_batch_stops(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)
    attempts = []
    diagnostic = {
        "error_code": "response_timeout", "phase": "ready_wait", "startup_timeout_s": 300,
        "wait_elapsed_ms": 300001.0, "startup_elapsed_ms": 300002.0,
        "cleanup_elapsed_ms": 9000.0, "ssh_returncode": 0, "ssh_exit_confirmed": True,
        "ssh_exit_success": True, "resource_cleanup_confirmed": False,
        "cleanup_status": "unconfirmed_startup",
    }

    def fail_worker(**_):
        attempts.append(True)
        error = RemoteOSWorldError("remote worker response timed out", code="response_timeout")
        error.startup_diagnostics = diagnostic
        raise error

    monkeypatch.setattr(driver, "RemoteOSWorldEnv", fail_worker)
    assert driver.main(arguments(tmp_path, tasks=["os/one", "os/two"], modes=["vlm", "hybrid"])) == 1
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert len(attempts) == len(summary["runs"]) == 1 and calls["goals"] == []
    report = summary["runs"][0]
    assert report["startup_diagnostics"] == diagnostic
    assert report["cleanup_status"] == "unconfirmed_startup"
    assert report["run_status"] == "cleanup_unconfirmed" and report["official_score"] is None
    assert not (tmp_path / "run/os-two").exists()


@pytest.mark.parametrize("fail_cleanup", [True, False])
def test_remote_cleanup_uncertainty_returns_nonzero_without_changing_score(
    driver, monkeypatch, tmp_path, fail_cleanup,
):
    calls = install_episode_fakes(
        monkeypatch, driver, cleanup_status="unconfirmed_forced_termination", fail_cleanup=fail_cleanup,
    )
    assert driver.main(arguments(tmp_path)) == 1
    assert calls["environments"][0].evaluations == 1
    assert all(planner.closed for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["run_status"] == "failed"
    report = summary["runs"][0]
    assert report["success"] and report["official_score"] == 1.0
    assert report["episode_status"] == "success"
    assert report["cleanup_status"] == "unconfirmed_forced_termination"
    assert report["run_status"] == ("cleanup_error" if fail_cleanup else "cleanup_unconfirmed")
    assert "private SSH cleanup details" not in str(report)
    assert json.loads((tmp_path / "run/gimp-theme/vlm/result.json").read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("task", ["../secret", "os/../../secret", "/os/id", "os/id.json", "os\\id"])
def test_cli_rejects_task_path_escape_before_model_or_worker(driver, monkeypatch, tmp_path, task):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, tasks=[task]))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("overrides", [
    {"tasks": ["os/quiet", "os/quiet"]}, {"modes": ["hybrid", "hybrid"]},
])
def test_cli_rejects_duplicate_tasks_and_modes_before_setup(driver, monkeypatch, tmp_path, overrides):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, **overrides))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("option", ["remote-project", "osworld-root", "vm-path", "remote-output"])
@pytest.mark.parametrize("value", ["relative/path", "~/not-expanded", "C:\\local\\path"])
def test_remote_paths_must_be_absolute_posix_before_setup(driver, monkeypatch, tmp_path, option, value):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, **{option: value}))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("value", ["/owned/invalid\npath", "/owned/invalid\x00path"])
def test_remote_paths_reject_control_characters(driver, monkeypatch, tmp_path, value):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, **{"remote-output": value}))
    assert error.value.code == 2


def test_remote_user_is_required_before_configuration(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    argv = arguments(tmp_path)
    index = argv.index("--remote-user")
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as error:
        driver.main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize("value", ["", "user\nother", "user/name"])
def test_remote_user_must_name_one_account(driver, monkeypatch, tmp_path, value):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, **{"remote-user": value}))
    assert error.value.code == 2


@pytest.mark.parametrize("layout", [
    "full_v1", "deduplicated_v1", "shared_fields_v2", "shared_fields_v3", "shared_fields_v4",
])
def test_jev_layout_is_applied_locally_and_recorded_for_both_modes(driver, monkeypatch, tmp_path, layout):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(tmp_path, modes=["vlm", "hybrid"], **{"jev-request-layout": layout})) == 0
    assert len(calls["policies"]) == 1
    assert calls["policies"][0].settings["request_layout"] == layout
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["jev_request_layout"] == layout
    assert all(row["experiment_settings"]["jev_request_layout"] == layout for row in summary["runs"])
    assert all("--jev-request-layout" not in env.command for env in calls["environments"])


@pytest.mark.parametrize("contract", [
    "legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
    "property_intent_v29", "property_intent_v49",
])
def test_desktop_stage_contract_applies_locally_and_is_recorded(driver, monkeypatch, tmp_path, contract):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"], **{"desktop-stage-contract": contract})
    assert driver.main(argv) == 0
    assert all(planner.settings["desktop_contract"] == contract for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["desktop_stage_contract"] == contract
    assert all(row["experiment_settings"]["desktop_stage_contract"] == contract for row in summary["runs"])
    assert all("--desktop-stage-contract" not in env.command for env in calls["environments"])


@pytest.mark.parametrize("output_format", ["unconstrained_v1", "json_schema_v50"])
def test_desktop_output_format_applies_and_is_recorded_equally_for_both_modes(
    driver, monkeypatch, tmp_path, output_format,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"], **{
        "desktop-stage-contract": "property_intent_v49", "desktop-output-format": output_format,
    })
    assert driver.main(argv) == 0
    assert all(planner.settings["desktop_output_format"] == output_format for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["desktop_output_format"] == output_format
    assert all(row["experiment_settings"]["desktop_output_format"] == output_format
               for row in summary["runs"])
    assert all("--desktop-output-format" not in env.command for env in calls["environments"])


@pytest.mark.parametrize("observation_mode", ["full_multimodal_v1", "native_first_v51"])
def test_planning_observation_is_requested_and_recorded_for_both_modes(
    driver, monkeypatch, tmp_path, observation_mode,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"], **{
        "desktop-stage-contract": "property_intent_v51", "desktop-output-format": "json_schema_v50",
        "desktop-planning-observation": observation_mode,
    })
    assert driver.main(argv) == 0
    assert all(planner.settings["desktop_planning_observation"] == observation_mode
               for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["desktop_planning_observation"] == observation_mode
    assert all(row["experiment_settings"]["desktop_planning_observation"] == observation_mode
               for row in summary["runs"])
    assert all("--desktop-planning-observation" not in env.command for env in calls["environments"])


@pytest.mark.parametrize("batch_mode", [False, True])
def test_optional_native_menu_provider_stays_local_and_configuration_applies_equally_to_both_modes(
    driver, monkeypatch, tmp_path, batch_mode,
):
    import hashlib

    calls = install_episode_fakes(monkeypatch, driver)
    binary = tmp_path / "worker.exe"
    binary.write_bytes(b"not executed by this lifecycle test")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    argv = arguments(tmp_path, modes=["vlm", "hybrid"], **{
        "native-menu-ocr-binary": str(binary), "native-menu-ocr-sha256": digest,
        "native-menu-ocr-timeout": "2.5",
    })
    if batch_mode:
        argv.append("--native-menu-ocr-batch")
    assert driver.main(argv) == 0
    assert len(calls["menu_providers"]) == 2
    assert all(p.binary_sha256 == digest and p.timeout_s == 2.5 for p in calls["menu_providers"])
    assert all(p.batch_mode is batch_mode for p in calls["menu_providers"])
    summary = json.loads((tmp_path / "run/summary.json").read_bytes())
    for row in summary["runs"]:
        assert row["experiment_settings"]["menu_grounding"] == "native_menu_text_v1"
        assert row["experiment_settings"]["menu_ocr_worker_sha256"] == digest
        assert row["experiment_settings"]["menu_ocr_process_layout"] == (
            "frame_batch_v1" if batch_mode else "tile_process_v1"
        )
        assert row["menu_grounding"]["provider"]["native_recognitions"] == 0
    assert all("--native-menu-ocr-binary" not in env.command for env in calls["environments"])


def test_invalid_native_worker_pin_fails_configuration_before_any_vm_start(driver, monkeypatch, tmp_path):
    calls = install_episode_fakes(monkeypatch, driver)
    binary = tmp_path / "worker.exe"
    binary.write_bytes(b"fixture")
    argv = arguments(tmp_path, modes=["vlm", "hybrid"], **{
        "native-menu-ocr-binary": str(binary), "native-menu-ocr-sha256": "0" * 64,
    })
    assert driver.main(argv) == 1
    assert calls["environments"] == []
    summary = json.loads((tmp_path / "run/summary.json").read_bytes())
    assert all(row["evaluation_status"] == "not_run" for row in summary["runs"])


@pytest.mark.parametrize("space", ["normalized_1000", "image_pixels"])
def test_visual_coordinate_preference_applies_to_both_local_planners_and_metadata(
    driver, monkeypatch, tmp_path, space,
):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(tmp_path, modes=["vlm", "hybrid"], **{"visual-box-space": space})) == 0
    assert all(planner.settings["visual_box_space"] == space for planner in calls["planners"])
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["visual_box_space"] == space
    assert all(row["experiment_settings"]["visual_box_space"] == space for row in summary["runs"])
    assert all("--visual-box-space" not in env.command for env in calls["environments"])


@pytest.mark.parametrize("overrides", [
    {"max-steps": "0"}, {"max-steps": "-1"},
    {"timeout": "0"}, {"timeout": "nan"}, {"timeout": "3601"},
    {"startup-timeout": "0"}, {"startup-timeout": "-1"}, {"startup-timeout": "601"},
    {"startup-timeout": "nan"}, {"startup-timeout": "inf"}, {"startup-timeout": "true"},
    {"plan-horizon": "0"}, {"plan-horizon": "13"}, {"plan-horizon": "1.5"},
    {"plan-ttl": "0"}, {"plan-ttl": "-1"}, {"plan-ttl": "301"},
    {"plan-ttl": "nan"}, {"plan-ttl": "inf"},
])
def test_invalid_episode_budget_does_not_start_remote_vm(driver, monkeypatch, tmp_path, overrides):
    monkeypatch.setattr(driver, "make_desktop_planner", lambda _: pytest.fail("validation must happen first"))
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, **overrides))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("policy", ["program_optional", "atomic_stage_only"])
def test_normal_stage_policy_cli_is_local_and_recorded(driver, monkeypatch, tmp_path, policy):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, modes=["hybrid", "vlm"], **{
        "desktop-stage-contract": "property_intent_v52",
        "desktop-output-format": "semantic_tools_v64",
        "desktop-normal-stage-policy": policy,
    })
    assert driver.main(argv) == 0
    assert len(calls["planners"]) == 3
    for planner in calls["planners"]:
        if policy == "program_optional":
            assert "desktop_normal_stage_policy" not in planner.settings
        else:
            assert planner.settings["desktop_normal_stage_policy"] == policy
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_settings"]["desktop_normal_stage_policy"] == policy
    for row, environment in zip(summary["runs"], calls["environments"], strict=True):
        assert row["experiment_settings"]["desktop_normal_stage_policy"] == policy
        assert "--desktop-normal-stage-policy" not in environment.command
    assert [config.plan_horizon for config in calls["configs"]] == [6, 1]


def test_atomic_policy_wrong_format_rejected_before_environment(driver, tmp_path):
    with pytest.raises(ValueError, match="semantic_tools_v64"):
        driver.main(arguments(tmp_path, **{"desktop-normal-stage-policy": "atomic_stage_only"}))
