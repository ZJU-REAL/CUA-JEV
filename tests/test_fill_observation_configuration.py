"""Do not compare different intermediate observation policies as equal settings."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cli_policy_is_opt_in_and_recorded_identically_for_both_modes():
    driver = module("run_osworld_smoke")
    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    off, on = parser.parse_args([]), parser.parse_args(["--image-only-fill-intermediates"])
    assert driver.experiment_settings(off)["intermediate_fill_observation"] == "full_v1"
    assert driver.experiment_settings(on)["intermediate_fill_observation"] == "image_only_v1"
    assert driver.planning_settings(on, "hybrid")["plan_horizon"] == 6
    assert driver.planning_settings(on, "vlm")["plan_horizon"] == 1


@pytest.mark.parametrize("left,right,status", [
    ("full_v1", "image_only_v1", "mismatch"),
    ("image_only_v1", "image_only_v1", "no_observed_mismatch"),
    (True, True, "invalid_configuration"),
    ("skip", "skip", "invalid_configuration"),
    (None, "image_only_v1", "no_observed_mismatch"),
])
def test_comparison_keeps_policy_mismatch_invalid_and_unknown(left, right, status):
    comparer = module("summarize_osworld_comparison")
    result = comparer.compare_config(
        {"config": {"intermediate_fill_observation": left}},
        {"config": {"intermediate_fill_observation": right}},
    )
    assert result["status"] == status
    if left is None:
        assert result["unknown"]["intermediate_fill_observation"] == ["hybrid"]
        assert "intermediate_fill_observation" not in result["matched"]


@pytest.mark.parametrize("left,right,status", [
    ("disabled", "fill_observation_rpc_v1", "mismatch"),
    ("fill_observation_rpc_v1", "fill_observation_rpc_v1", "no_observed_mismatch"),
    (True, True, "invalid_configuration"),
    (None, "disabled", "no_observed_mismatch"),
])
def test_instrumented_and_ordinary_workers_cannot_be_pooled_as_equal(left, right, status):
    comparer = module("summarize_osworld_comparison")
    result = comparer.compare_config(
        {"config": {"worker_instrumentation": left}}, {"config": {"worker_instrumentation": right}},
    )
    assert result["status"] == status
    if left is None:
        assert result["unknown"]["worker_instrumentation"] == ["hybrid"]
