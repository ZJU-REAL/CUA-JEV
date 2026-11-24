"""Thinking-budget CLI and experiment wiring without model, SSH, or VM calls."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_osworld_remote_driver import arguments, install_episode_fakes
from test_osworld_remote_driver import driver as driver


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "1e3", "nan", "inf", "true", "not-an-integer"])
def test_invalid_thinking_budget_is_rejected_before_planner_or_worker(
    driver, monkeypatch, tmp_path, value,
):
    monkeypatch.setattr(
        driver, "make_desktop_planner", lambda _: pytest.fail("invalid CLI must not construct a planner"),
    )
    with pytest.raises(SystemExit) as error:
        driver.main(arguments(tmp_path, **{"model-thinking": "on", "model-thinking-budget": value}))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "thinking_args", [[], ["--model-thinking", "default"], ["--model-thinking", "off"]],
    ids=["implicit-default", "explicit-default", "off"],
)
def test_thinking_budget_requires_explicit_on_before_model_construction(
    driver, monkeypatch, tmp_path, thinking_args,
):
    smoke = importlib.import_module("run_osworld_smoke")
    monkeypatch.setattr(
        smoke, "MultimodalPlanner", lambda *_a, **_k: pytest.fail("invalid combination constructed model"),
    )
    with pytest.raises(ValueError, match="thinking"):
        driver.main([*arguments(tmp_path), *thinking_args, "--model-thinking-budget", "512"])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("thinking_args", "budget", "enabled"),
    [
        ([], None, None),
        (["--model-thinking", "on"], None, True),
        (["--model-thinking", "off"], None, False),
        (["--model-thinking", "on", "--model-thinking-budget", "1"], 1, True),
        (["--model-thinking", "on", "--model-thinking-budget", str(2**40)], 2**40, True),
    ],
    ids=["provider-default", "on-with-provider-budget", "off", "one-token", "no-generic-model-cap"],
)
def test_budget_reaches_both_modes_and_all_records_but_stays_out_of_worker(
    driver, monkeypatch, tmp_path, thinking_args, budget, enabled,
):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main([*arguments(tmp_path, modes=["vlm", "hybrid"]), *thinking_args]) == 0
    assert len(calls["planners"]) == 3  # Configuration probe plus one planner for each mode.
    for planner in calls["planners"]:
        actual = planner.settings["thinking_budget"]
        assert actual == budget and type(actual) is type(budget)
        assert planner.settings["enable_thinking"] is enabled
        assert planner.closed
    assert [(config.mode, config.plan_horizon) for config in calls["configs"]] == [("vlm", 1), ("hybrid", 6)]

    output = tmp_path / "run"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    recorded = "default" if budget is None else budget
    for document in (summary, metadata):
        actual = document["experiment_settings"]["model_thinking_budget_tokens"]
        assert actual == recorded and type(actual) is type(recorded)
    assert [row["mode"] for row in summary["runs"]] == ["vlm", "hybrid"]
    for row, env in zip(summary["runs"], calls["environments"], strict=True):
        result = output / row["task"].replace("/", "-") / row["mode"] / "result.json"
        saved = json.loads(result.read_text(encoding="utf-8"))
        for document in (row, saved):
            actual = document["experiment_settings"]["model_thinking_budget_tokens"]
            assert actual == recorded and type(actual) is type(recorded)
        assert "--model-thinking-budget" not in env.command
        assert "--model-thinking" not in env.command


def test_programmatic_args_without_budget_keep_provider_default(driver, monkeypatch):
    smoke = importlib.import_module("run_osworld_smoke")
    planner = object()
    constructor = Mock(return_value=planner)
    monkeypatch.setattr(smoke, "MultimodalPlanner", constructor)
    args = SimpleNamespace(
        model_base_url="https://model.example.test/v1", model="offline-model",
        allow_screenshot_upload=True, allow_insecure_model_http=False, use_env_proxy=False,
    )
    assert driver.make_desktop_planner(args) is planner
    assert constructor.call_args.kwargs["thinking_budget"] is None
    assert driver.experiment_settings(args)["model_thinking_budget_tokens"] == "default"
