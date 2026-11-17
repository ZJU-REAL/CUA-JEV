"""New opt-ins are explicit in both modes; absent historical settings stay disabled."""

import argparse

import pytest
from test_osworld_comparison import comparison, row, stats, write_summary
from test_osworld_driver import driver


def test_cli56_layout_and_both_recovery_optins_are_explicit_and_default_off():
    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.jev_capacity_recovery is False
    assert defaults.reuse_verified_pending_snapshot is False
    assert defaults.whole_goal_recheck is False
    args = parser.parse_args([
        "--desktop-output-format", "semantic_tools_v56", "--desktop-stage-contract", "property_intent_v52",
        "--jev-request-layout", "shared_fields_v5", "--jev-capacity-recovery",
        "--reuse-verified-pending-snapshot",
        "--whole-goal-recheck",
    ])
    settings = driver.experiment_settings(args)
    assert settings["desktop_output_format"] == "semantic_tools_v56"
    assert settings["jev_request_layout"] == "shared_fields_v5"
    assert settings["jev_capacity_recovery"] == "vlm_once_v1"
    assert settings["reuse_verified_pending_snapshot"] is True
    assert settings["whole_goal_recheck"] == "current_goal_recheck_v1"
    assert driver.experiment_settings(defaults)["jev_capacity_recovery"] == "disabled"


def test_historical_new_settings_are_disabled_without_reclassifying_other_unknowns(tmp_path):
    report = comparison.summarize([("old", write_summary(tmp_path, [row("hybrid"), row("vlm")]))])
    config = report["pairs"][0]["configuration"]
    assert config["matched"]["jev_capacity_recovery"] == "disabled"
    assert config["matched"]["reuse_verified_pending_snapshot"] is False
    assert config["matched"]["whole_goal_recheck"] == "disabled"
    assert config["unknown"]["desktop_output_format"] == ["hybrid", "vlm"]


@pytest.mark.parametrize("key,enabled,disabled", [
    ("jev_capacity_recovery", "vlm_once_v1", "disabled"),
    ("reuse_verified_pending_snapshot", True, False),
    ("whole_goal_recheck", "current_goal_recheck_v1", "disabled"),
    ("desktop_output_format", "semantic_tools_v56", "semantic_tools_v55"),
    ("jev_request_layout", "shared_fields_v5", "shared_fields_v4"),
])
def test_different_optins_cannot_form_speed_pair(tmp_path, key, enabled, disabled):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", experiment_settings={key: enabled}),
        row("vlm", experiment_settings={key: disabled}),
    ]))])
    assert key in report["pairs"][0]["configuration"]["mismatches"]
    assert stats(report)["agent_ms"]["n"] == 0
    assert all(report["counts_all_records"][mode]["official_successes"] == 1 for mode in ("hybrid", "vlm"))


@pytest.mark.parametrize("key,enabled", [
    ("jev_capacity_recovery", "vlm_once_v1"), ("reuse_verified_pending_snapshot", True),
    ("whole_goal_recheck", "current_goal_recheck_v1"),
])
def test_explicit_unknown_new_setting_does_not_inherit_shared_enabled(tmp_path, key, enabled):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", experiment_settings={key: None}), row("vlm"),
    ], experiment_settings={key: enabled}))])
    config = report["pairs"][0]["configuration"]
    assert config["unknown"][key] == ["hybrid"]
    assert key not in config["matched"]


@pytest.mark.parametrize("key,bad", [
    ("jev_capacity_recovery", True), ("jev_capacity_recovery", "other"),
    ("reuse_verified_pending_snapshot", 0), ("reuse_verified_pending_snapshot", "disabled"),
    ("whole_goal_recheck", True), ("whole_goal_recheck", "other"),
])
def test_new_settings_validate_types_without_losing_successes(tmp_path, key, bad):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", experiment_settings={key: bad}), row("vlm", experiment_settings={key: bad}),
    ]))])
    assert key in report["pairs"][0]["configuration"]["invalid"]
    assert stats(report)["agent_ms"]["n"] == 0
