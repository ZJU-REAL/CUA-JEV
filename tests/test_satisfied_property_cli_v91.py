"""Shared real CLI/episode wiring with existing lifecycle fakes, no network."""

import argparse
import json
from types import SimpleNamespace

import pytest
from test_osworld_driver import driver as local_driver
from test_osworld_driver import ready_observation
from test_osworld_remote_driver import arguments, install_episode_fakes
from test_osworld_remote_driver import driver as driver


@pytest.mark.parametrize("enabled", [False, True])
def test_remote_cli_preserves_default_factory_shape_and_enables_both_modes(
    driver, monkeypatch, tmp_path, enabled,
):
    calls = install_episode_fakes(monkeypatch, driver)
    original_factory, kwargs_seen = driver.OSWorldBackend, []

    def factory(env, **kwargs):
        kwargs_seen.append(dict(kwargs))
        if enabled:
            assert kwargs.pop("satisfied_property_readonly_fast_path") is True
        else:
            assert "satisfied_property_readonly_fast_path" not in kwargs
        return original_factory(env, **kwargs)

    monkeypatch.setattr(driver, "OSWorldBackend", factory)
    argv = arguments(tmp_path, modes=["vlm", "hybrid"]) + ["--image-only-fill-intermediates"]
    if enabled:
        argv.append("--satisfied-property-readonly-fast-path")
    assert driver.main(argv) == 0
    expected = {"pause": 0.5, "image_only_fill_intermediates": True}
    if enabled:
        expected["satisfied_property_readonly_fast_path"] = True
    assert kwargs_seen == [expected, expected]
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "run/metadata.json").read_text(encoding="utf-8"))
    settings = [summary["experiment_settings"], metadata["experiment_settings"],
                *(row["experiment_settings"] for row in summary["runs"])]
    for item in settings:
        if enabled:
            assert item["satisfied_property_readonly_fast_path"] is True
        else:
            assert "satisfied_property_readonly_fast_path" not in item
    assert len(calls["environments"]) == 2
    assert all("--satisfied-property-readonly-fast-path" not in env.command
               for env in calls["environments"])


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_local_cli_parsing_reaches_real_episode_factory(monkeypatch, tmp_path, enabled):
    parser = argparse.ArgumentParser()
    local_driver.add_plan_arguments(parser)
    args = parser.parse_args(["--satisfied-property-readonly-fast-path"] if enabled else [])
    vars(args).update(output=tmp_path, model="offline", vm_path=tmp_path / "owned.qcow2",
                      no_a11y=False, pause=0.25, max_steps=3, timeout=10)
    kwargs_seen = []
    initial = ready_observation()
    env = SimpleNamespace(controller=SimpleNamespace(get_screenshot=lambda: initial["screenshot"]),
                          evaluate=lambda: 1.0, close=lambda: None)

    def factory(actual_env, **kwargs):
        assert actual_env is env
        kwargs_seen.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    class Runner:
        def __init__(self, _backend, _planner, **_kwargs):
            pass

        def run(self, instruction, *, success):
            assert instruction == "Public goal" and success is None
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    monkeypatch.setattr(local_driver, "AgentRunner", Runner)
    report = local_driver.run_episode(
        args, "os/fixture", {"instruction": "Public goal"}, "vlm", env_factory=lambda **_: env,
        backend_factory=factory, planner_factory=lambda _: SimpleNamespace(close=lambda: None),
        initialize=lambda *_: initial, settle=lambda _env, observed, **_: observed,
    )
    expected = {"pause": 0.25}
    if enabled:
        expected["satisfied_property_readonly_fast_path"] = True
    assert kwargs_seen == [expected]
    assert report["cleanup_status"] == "confirmed" and report["evaluation_status"] == "ok"
    assert report["experiment_settings"].get("satisfied_property_readonly_fast_path", False) is enabled
    if not enabled:
        assert "satisfied_property_readonly_fast_path" not in report["experiment_settings"]


@pytest.mark.parametrize("bad", [1, "true", None])
def test_shared_settings_rejects_nonboolean_opt_in_and_cli_rejects_a_value(bad):
    with pytest.raises(ValueError, match="satisfied_property_readonly_fast_path must be a boolean"):
        local_driver.experiment_settings(SimpleNamespace(satisfied_property_readonly_fast_path=bad))
    parser = argparse.ArgumentParser()
    local_driver.add_plan_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--satisfied-property-readonly-fast-path", "false"])
    assert error.value.code == 2
