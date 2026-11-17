"""Regression tests for source57 cli and settings."""

import argparse
from types import SimpleNamespace

import pytest
from test_inband_whole_goal_review import InbandPlanner, already_complete
from test_osworld_comparison import comparison
from test_osworld_driver import driver, ready_observation
from test_whole_goal_recheck import finished
from test_whole_goal_recheck import run as run_original

from cua_jev.agent_runner import AgentConfig


def options(*flags):
    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    return parser.parse_args(flags)


def test_new_cli_defaults_retain_original_formats_and_disabled_optimizations():
    args = options()
    settings = driver.experiment_settings(args)
    config = AgentConfig(whole_goal_recheck=True)
    assert args.desktop_output_format == "unconstrained_v1" and args.jev_request_layout == "full_v1"
    assert config.jev_capacity_recovery is False and config.jev_capacity_recovery_policy == "vlm_once_v1"
    assert config.reuse_whole_goal_planning_snapshot is config.inband_whole_goal_recheck is False
    assert config.navigation_continuation is False and settings["navigation_continuation"] is False
    assert settings["navigation_continuation_version"] == "disabled"
    assert settings["jev_capacity_recovery"] == settings["whole_goal_recheck"] == "disabled"
    assert settings["jev_capacity_recovery_policy"] == "vlm_once_v1"
    assert settings["whole_goal_planning_snapshot_reuse_version"] == "disabled"
    assert settings["inband_whole_goal_recheck_version"] == "disabled"


def test_whole_goal_master_alone_keeps_the_original_dedicated_call_and_fresh_reads():
    planner = InbandPlanner([finished(), finished()])
    result, backend, planner, trace = run_original([], backend=already_complete(), planner=planner)
    assert result.status == "unverified" and len(planner.calls) == 2
    assert [call["review"] for call in planner.calls] == [False, True]
    assert all(call["inband_epoch"] is None for call in planner.calls)
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
    assert result.whole_goal_planning_snapshot_reuses == result.whole_goal_inband_review_calls == 0
    assert backend.fresh_observe_calls == 4
    assert not any(e["kind"] in {"agent_whole_goal_planning_observation_reuse",
                                 "agent_whole_goal_inband_review"} for e in trace.events)


def test_requested_opt_ins_without_the_master_do_not_claim_active_version_or_review():
    args = options("--reuse-whole-goal-planning-snapshot", "--inband-whole-goal-recheck",
                   "--jev-capacity-recovery-policy", "vlm_phase_v2")
    settings = driver.experiment_settings(args)
    assert settings["reuse_whole_goal_planning_snapshot"] is settings["inband_whole_goal_recheck"] is True
    assert settings["whole_goal_planning_snapshot_reuse_version"] == "disabled"
    assert settings["inband_whole_goal_recheck_version"] == "disabled"
    assert settings["jev_capacity_recovery_policy"] == "vlm_phase_v2"
    assert settings["jev_capacity_recovery"] == "disabled"
    result, _, planner, _ = run_original(
        [], enabled=False, backend=already_complete(), planner=InbandPlanner([finished()]),
        reuse_whole_goal_planning_snapshot=True, inband_whole_goal_recheck=True,
    )
    assert len(planner.calls) == 1 and planner.calls[0]["inband_epoch"] is None
    assert result.whole_goal_recheck_calls == result.whole_goal_inband_review_calls == 0


@pytest.mark.parametrize("field", ["reuse_whole_goal_planning_snapshot", "inband_whole_goal_recheck",
                                   "navigation_continuation"])
@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_configuration_opt_ins_require_exact_booleans(field, value):
    with pytest.raises(ValueError, match=field):
        AgentConfig(**{field: value})


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
def test_real_driver_wires_explicit_versions_to_both_modes_without_agent_or_VM_calls(
    monkeypatch, tmp_path, mode,
):
    args = options("--desktop-output-format", "semantic_tools_v57", "--desktop-stage-contract",
                   "property_intent_v52", "--jev-request-layout", "shared_fields_v6",
                   "--jev-capacity-recovery", "--jev-capacity-recovery-policy", "vlm_phase_v2",
                   "--whole-goal-recheck", "--reuse-whole-goal-planning-snapshot",
                   "--inband-whole-goal-recheck", "--navigation-continuation")
    args.output, args.model, args.vm_path = tmp_path, "offline", tmp_path / "owned-base.qcow2"
    args.no_a11y, args.pause, args.max_steps, args.timeout = False, 0.5, 25, 420
    captures = []
    events = []
    closed = SimpleNamespace(close=lambda: None)

    class Env:
        controller = SimpleNamespace(get_screenshot=lambda: ready_observation()["screenshot"])

        def evaluate(self):
            events.append("evaluate")
            return 1.0

        def close(self):
            events.append("close")

    class CaptureRunner:
        def __init__(self, _backend, _planner, *, config, **_kwargs):
            captures.append(config)

        def run(self, goal, *, success):
            assert goal == "Complete the public goal" and success is None
            events.append("agent")
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "success": False})

    def policy(**kwargs):
        assert kwargs["request_layout"] == "shared_fields_v6"
        return closed

    monkeypatch.setattr(driver, "AgentRunner", CaptureRunner)
    monkeypatch.setattr(driver, "JevPolicy", policy)
    monkeypatch.setattr(driver, "wait_for_desktop_ready", lambda _env, value, **_kwargs: value)
    monkeypatch.setattr(driver, "wait_for_desktop_settled", lambda _env, value, **_kwargs: value)
    report = driver.run_episode(
        args, "os/offline-case", {"instruction": "Complete the public goal", "evaluator": {"hidden": True}},
        mode, env_factory=lambda **_kwargs: Env(), backend_factory=lambda *_args, **_kwargs: closed,
        planner_factory=lambda _args: closed, initialize=lambda *_args: ready_observation(),
    )
    assert len(captures) == 1 and events == ["agent", "evaluate", "close"]
    config = captures[0]
    assert config.jev_capacity_recovery is True and config.jev_capacity_recovery_policy == "vlm_phase_v2"
    assert config.whole_goal_recheck is config.reuse_whole_goal_planning_snapshot is True
    assert config.inband_whole_goal_recheck is True
    assert config.navigation_continuation is True
    assert config.plan_horizon == (6 if mode == "hybrid" else 1)
    settings = report["experiment_settings"]
    assert settings["desktop_output_format"] == "semantic_tools_v57"
    assert settings["jev_request_layout"] == "shared_fields_v6"
    assert settings["jev_capacity_recovery"] == settings["jev_capacity_recovery_policy"] == "vlm_phase_v2"
    assert settings["whole_goal_planning_snapshot_reuse_version"] == "successful_completion_full_frame_v1"
    assert settings["inband_whole_goal_recheck_version"] == "all_original_goal_review_v1"
    assert settings["navigation_continuation"] is True
    assert settings["navigation_continuation_version"] == "observed_transition_v1"


def configured(**changes):
    config = driver.experiment_settings(options(
        "--desktop-output-format", "semantic_tools_v57", "--desktop-stage-contract", "property_intent_v52",
        "--jev-request-layout", "shared_fields_v6", "--jev-capacity-recovery",
        "--jev-capacity-recovery-policy", "vlm_phase_v2", "--whole-goal-recheck",
        "--reuse-whole-goal-planning-snapshot", "--inband-whole-goal-recheck", "--navigation-continuation",
    ))
    return {"config": {**config, **changes}}


def test_comparer_accepts_identical_new_versions_and_refuses_mixed_requested_protocols():
    same = comparison.compare_config(configured(), configured())
    assert not same["invalid"] and not same["mismatches"]
    assert same["matched"]["jev_request_layout"] == "shared_fields_v6"
    assert same["matched"]["desktop_output_format"] == "semantic_tools_v57"
    mixed = comparison.compare_config(configured(), configured(desktop_output_format="semantic_tools_v56"))
    assert "desktop_output_format" in mixed["mismatches"]
    mixed = comparison.compare_config(configured(), configured(jev_capacity_recovery="vlm_once_v1"))
    assert "jev_capacity_recovery" in mixed["mismatches"]


@pytest.mark.parametrize("field,value", [
    ("jev_request_layout", "shared_fields_v8"),
    ("desktop_output_format", "semantic_tools_unknown"),
    ("jev_capacity_recovery_policy", "vlm_phase_v4"),
    ("reuse_whole_goal_planning_snapshot", 1),
    ("inband_whole_goal_recheck", "true"),
    ("whole_goal_planning_snapshot_reuse_version", "guess"),
    ("inband_whole_goal_recheck_version", "guess"),
    ("navigation_continuation", 1),
    ("navigation_continuation_version", "guess"),
])
def test_comparer_does_not_treat_matching_malformed_new_settings_as_equal(field, value):
    report = comparison.compare_config(configured(**{field: value}), configured(**{field: value}))
    assert field in report["invalid"] and field not in report["matched"]


@pytest.mark.parametrize("field", ["reuse_whole_goal_planning_snapshot", "inband_whole_goal_recheck",
                                   "whole_goal_planning_snapshot_reuse_version",
                                   "inband_whole_goal_recheck_version", "jev_capacity_recovery_policy",
                                   "navigation_continuation", "navigation_continuation_version"])
def test_explicit_null_new_settings_remain_unknown_while_historical_absence_is_disabled(field):
    row = {"task": "os/offline", "mode": "hybrid", "experiment_settings": {field: None}}
    effective = comparison.effective_config(row, {})
    assert effective[field] is None
    historic = comparison.effective_config({"task": "os/offline", "mode": "hybrid"}, {})
    assert historic[field] == ({"jev_capacity_recovery_policy": "vlm_once_v1",
                               "reuse_whole_goal_planning_snapshot": False,
                               "inband_whole_goal_recheck": False,
                               "navigation_continuation": False}.get(field, "disabled"))
