"""Shared local/remote CLI and real AgentConfig construction with lifecycle fakes."""

import argparse
import importlib
import json
from types import SimpleNamespace

import pytest
from test_osworld_driver import driver as local_driver
from test_osworld_driver import ready_observation
from test_osworld_remote_driver import arguments, install_episode_fakes
from test_osworld_remote_driver import driver as driver

FIELD = "reuse_unexplained_context_planning_observation"
FLAG = "--reuse-unexplained-context-planning-observation"


@pytest.mark.parametrize("enabled", [False, True])
def test_remote_cli_reaches_real_config_in_both_modes_and_records_opt_in(
    driver, monkeypatch, tmp_path, enabled,
):
    calls = install_episode_fakes(monkeypatch, driver)
    smoke = importlib.import_module("run_osworld_smoke")
    original_config, kwargs_seen = smoke.AgentConfig, []

    def config_factory(**kwargs):
        kwargs_seen.append(dict(kwargs))
        return original_config(**kwargs)

    monkeypatch.setattr(smoke, "AgentConfig", config_factory)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"]) + ["--max-steps", "3", "--timeout", "10"]
    if enabled:
        argv.append(FLAG)
    assert driver.main(argv) == 0
    assert len(kwargs_seen) == len(calls["configs"]) == 2
    for kwargs, config in zip(kwargs_seen, calls["configs"], strict=True):
        assert isinstance(config, original_config)
        assert getattr(config, FIELD) is enabled
        if enabled:
            assert kwargs[FIELD] is True
        else:
            assert FIELD not in kwargs
        assert config.allow_actions is True
        assert config.max_steps == 3 and config.max_plans == 8 and config.timeout_s == 10
    assert [(config.mode, config.plan_horizon, config.plan_ttl_s) for config in calls["configs"]] == [
        ("vlm", 1, 300), ("hybrid", 6, 300),
    ]
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "run/metadata.json").read_text(encoding="utf-8"))
    settings = [summary["experiment_settings"], metadata["experiment_settings"],
                *(row["experiment_settings"] for row in summary["runs"])]
    for item in settings:
        if enabled:
            assert item[FIELD] is True
        else:
            assert FIELD not in item
    assert all(FLAG not in env.command for env in calls["environments"])
    assert all(env.closed and env.evaluations == 1 for env in calls["environments"])


@pytest.mark.parametrize("enabled", [False, True])
def test_local_shared_parser_creates_real_config_and_preserves_episode_limits(
    monkeypatch, tmp_path, enabled,
):
    parser = argparse.ArgumentParser()
    local_driver.add_plan_arguments(parser)
    args = parser.parse_args([FLAG] if enabled else [])
    assert getattr(args, FIELD) is enabled
    vars(args).update(output=tmp_path, model="offline", vm_path=tmp_path / "owned.qcow2",
                      no_a11y=False, pause=0.25, max_steps=3, timeout=10)
    original_config, kwargs_seen, configs = local_driver.AgentConfig, [], []
    initial = ready_observation()
    env = SimpleNamespace(controller=SimpleNamespace(get_screenshot=lambda: initial["screenshot"]),
                          evaluate=lambda: 1.0, close=lambda: None)

    def config_factory(**kwargs):
        kwargs_seen.append(dict(kwargs))
        return original_config(**kwargs)

    class Runner:
        def __init__(self, _backend, _planner, *, config, **_kwargs):
            assert isinstance(config, original_config)
            configs.append(config)

        def run(self, instruction, *, success):
            assert instruction == "Public goal" and success is None
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(local_driver, "AgentConfig", config_factory)
    monkeypatch.setattr(local_driver, "AgentRunner", Runner)
    report = local_driver.run_episode(
        args, "os/fixture", {"instruction": "Public goal"}, "vlm", env_factory=lambda **_: env,
        backend_factory=lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
        planner_factory=lambda _: SimpleNamespace(close=lambda: None),
        initialize=lambda *_: initial, settle=lambda _env, observed, **_: observed,
    )
    assert len(configs) == len(kwargs_seen) == 1
    assert getattr(configs[0], FIELD) is enabled
    if enabled:
        assert kwargs_seen[0][FIELD] is True
        assert report["experiment_settings"][FIELD] is True
    else:
        assert FIELD not in kwargs_seen[0]
        assert FIELD not in report["experiment_settings"]
    config = configs[0]
    assert config.plan_horizon == 1 and config.plan_ttl_s == 300
    assert config.max_steps == 3 and config.max_plans == 8 and config.timeout_s == 10
    assert report["cleanup_status"] == "confirmed" and report["evaluation_status"] == "ok"
    assert report["runner"] == {"status": "unverified", "success": False}


@pytest.mark.parametrize("bad", [1, "true", None])
def test_settings_rejects_nonboolean_planning_reuse_opt_in(bad):
    with pytest.raises(ValueError, match=f"{FIELD} must be a boolean"):
        local_driver.experiment_settings(SimpleNamespace(**{FIELD: bad}))


def test_cli_rejects_an_explicit_value_for_the_boolean_opt_in():
    parser = argparse.ArgumentParser()
    local_driver.add_plan_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args([FLAG, "false"])
    assert error.value.code == 2
