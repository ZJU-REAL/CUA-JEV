"""Offline paired analyses must preserve failures and refuse misleading speedups."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "osworld_comparison", Path(__file__).parents[1] / "scripts" / "summarize_osworld_comparison.py"
)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


def row(mode, *, task="os/task", agent=100, episode=1000, success=True, **extra):
    return {
        "task": task, "mode": mode, "model": "test-vlm", "success": success,
        "evaluation_status": "ok", "official_score": 1.0 if success else 0.0,
        "episode_status": "success" if success else "task_failure",
        "run_status": "success" if success else "task_failure", "cleanup_status": "confirmed",
        "runner": {"status": "unverified", "duration_ms": agent,
                   "planner_calls": 2 if mode == "hybrid" else 4,
                   "jev_calls": 2 if mode == "hybrid" else 0},
        "episode_wall_ms": episode, **extra,
    }


def write_summary(tmp_path, rows, *, name="summary.json", **extra):
    path = tmp_path / name
    path.write_text(json.dumps({"runs": rows, **extra}), encoding="utf-8")
    return path


def stats(report, label="trial"):
    return report["by_run"][label]["speedup_baseline_divided_by_hybrid"]


def test_agent_and_episode_have_independent_paired_ratios_and_statistics(tmp_path):
    path = write_summary(tmp_path, [
        row("hybrid", task="os/a", agent=100, episode=500),
        row("vlm", task="os/a", agent=200, episode=750),
        row("hybrid", task="gimp/b", agent=100, episode=1000),
        row("vlm", task="gimp/b", agent=800, episode=2000),
    ])
    result = comparison.summarize([("trial", path)])
    assert stats(result)["agent_ms"] == {"n": 2, "geometric_mean": 4.0, "median": 5.0}
    assert stats(result)["episode_ms"]["geometric_mean"] == pytest.approx(math.sqrt(3))
    assert stats(result)["episode_ms"]["median"] == 1.75
    assert result["by_run"]["trial"]["successful_pair_calls"]["hybrid"]["jev_calls"]["sum"] == 4
    assert result["pairs"][0]["configuration"]["unknown"]["pause_s"] == ["hybrid", "vlm"]
    assert result["policy"]["full_configuration_equality_verified"] is False


def test_fast_failed_run_never_creates_speedup_or_disappears_from_denominator(tmp_path):
    path = write_summary(tmp_path, [row("hybrid", agent=1, episode=2, success=False),
                                    row("vlm", agent=1000, episode=10000)])
    result = comparison.summarize([("trial", path)])
    assert stats(result)["agent_ms"]["n"] == 0
    assert stats(result)["episode_ms"]["geometric_mean"] is None
    assert result["counts_all_records"]["hybrid"]["official_successes"] == 0
    assert result["counts_all_records"]["hybrid"]["records"] == 1
    assert result["counts_all_records"]["vlm"]["official_successes"] == 1
    assert result["records"][0]["failures"]
    assert result["integrity_status"] == "no_detected_issues"


def test_interrupted_agent_without_report_remains_unknown_and_cannot_be_a_fast_failure(tmp_path):
    interrupted = row("hybrid", success=False, runner=None, official_score=None,
                      evaluation_status="not_run", episode_status="interrupted", run_status="interrupted",
                      cleanup_status="unknown_local_interruption", agent_execution_status="unknown",
                      episode_wall_ms=None)
    result = comparison.summarize([("trial", write_summary(tmp_path, [interrupted, row("vlm")]))])
    counts = result["counts_all_records"]["hybrid"]
    assert counts["records"] == 1 and counts["not_successful"] == 1
    assert counts["execution_started"] == counts["execution_not_started"] == 0
    assert counts["execution_unknown"] == counts["evaluation_not_run"] == 1
    assert result["records"][0]["execution_started"] is None
    assert result["records"][0]["timing"]["agent_ms"] is None
    assert stats(result)["agent_ms"]["n"] == 0


def test_valid_runner_report_takes_precedence_over_unknown_interruption_marker(tmp_path):
    result = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", agent_execution_status="unknown"), row("vlm"),
    ]))])
    counts = result["counts_all_records"]["hybrid"]
    assert counts["execution_started"] == 1 and counts["execution_unknown"] == 0


def test_execution_and_evaluation_counts_separate_startup_task_and_cleanup_failures(tmp_path):
    rows = [
        row("hybrid", task="os/startup", success=False, runner=None, official_score=None,
            evaluation_status="not_run", episode_status="startup_error",
            run_status="cleanup_unconfirmed", cleanup_status="unconfirmed_startup"),
        row("hybrid", task="os/task_failure", success=False),
        row("hybrid", task="os/cleanup_failure", run_status="cleanup_error",
            cleanup_status="unconfirmed_close_error"),
        row("hybrid", task="os/success"),
    ]
    result = comparison.summarize([("trial", write_summary(tmp_path, rows))])
    counts = result["counts_all_records"]["hybrid"]
    assert counts["records"] == 4 and counts["official_successes"] == 2
    assert counts["not_successful"] == 2 and counts["success_rate"] == 0.5
    assert counts["execution_started"] == 3 and counts["execution_not_started"] == 1
    assert counts["evaluation_completed"] == 3 and counts["evaluation_not_completed"] == 1
    assert counts["evaluation_not_run"] == 1 and counts["startup_not_run"] == 1
    assert counts["evaluated_successes"] == 2 and counts["evaluated_not_successful"] == 1
    assert counts["evaluated_success_rate"] == pytest.approx(2 / 3)
    assert counts["episode_status_counts"] == {"startup_error": 1, "task_failure": 1, "success": 2}
    assert counts["status_counts"]["cleanup_error"] == 1
    assert counts["status_counts"]["cleanup_unconfirmed"] == 1
    assert result["records"][2]["official_success"] is True
    assert result["records"][2]["evaluation_completed"] is True
    assert result["records"][2]["episode_status"] == "success"
    assert result["by_run"]["trial"]["counts"]["hybrid"] == counts


@pytest.mark.parametrize("score", [None, True, False, -0.01, 1.01, "1", {}, math.inf, math.nan])
def test_invalid_scores_do_not_enter_evaluated_denominator(tmp_path, score):
    path = write_summary(tmp_path, [row("hybrid", official_score=score)])
    result = comparison.summarize([("trial", path)])
    counts = result["counts_all_records"]["hybrid"]
    assert counts["records"] == 1 and counts["execution_started"] == 1
    assert counts["evaluation_completed"] == 0 and counts["evaluation_not_completed"] == 1
    assert counts["evaluation_status_counts"] == {"ok": 1}
    assert counts["evaluated_successes"] == 0 and counts["evaluated_success_rate"] is None
    assert counts["success_rate"] == 0


@pytest.mark.parametrize("score", [0, 0.5, 1.0])
def test_all_legal_completed_scores_enter_evaluated_denominator(tmp_path, score):
    path = write_summary(tmp_path, [row("hybrid", official_score=score, success=score == 1)])
    counts = comparison.summarize([("trial", path)])["counts_all_records"]["hybrid"]
    assert counts["evaluation_completed"] == 1
    assert counts["evaluated_success_rate"] == int(score == 1)


@pytest.mark.parametrize("runner", [None, {}, "failed", {"status": ""}, {"status": 1}])
def test_cleanup_status_alone_does_not_imply_startup_or_task_failure(tmp_path, runner):
    path = write_summary(tmp_path, [row(
        "hybrid", success=False, runner=runner, official_score=None,
        evaluation_status="not_run", episode_status=None,
        run_status="cleanup_unconfirmed", cleanup_status="unconfirmed_startup",
    )])
    counts = comparison.summarize([("trial", path)])["counts_all_records"]["hybrid"]
    assert counts["execution_started"] == 0 and counts["execution_not_started"] == 1
    assert counts["startup_not_run"] == 0
    assert counts["episode_status_counts"] == {"unknown": 1}
    assert counts["evaluation_completed"] == 0 and counts["evaluated_success_rate"] is None


def test_evaluation_error_with_numeric_score_does_not_count_as_completed(tmp_path):
    path = write_summary(tmp_path, [row("hybrid", evaluation_status="error", official_score=1)])
    counts = comparison.summarize([("trial", path)])["counts_all_records"]["hybrid"]
    assert counts["evaluation_completed"] == 0
    assert counts["evaluated_success_rate"] is None and counts["official_successes"] == 0


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["official_score", "episode_wall_ms", "duration_ms", "planner_calls",
                                   "jev_calls"])
def test_cli_writes_json_safe_nonfinite_numbers_without_losing_records_or_input_hash(tmp_path, number, field):
    rows = [row("hybrid"), row("vlm")]
    container = rows[0] if field in {"official_score", "episode_wall_ms"} else rows[0]["runner"]
    container[field] = number
    path = write_summary(tmp_path, rows)
    original = path.read_bytes()
    output = tmp_path / "report.json"
    assert comparison.main(["--run", f"trial={path}", "--output", str(output)]) == 1

    def reject_nonfinite(value):
        pytest.fail(f"report contained non-JSON numeric constant: {value}")

    result = json.loads(output.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    assert len(result["records"]) == 2 and result["counts_all_records"]["hybrid"]["records"] == 1
    assert path.read_bytes() == original
    assert result["inputs"][0]["sha256"] == hashlib.sha256(original).hexdigest()
    expected_path = ["runs", 0, field] if container is rows[0] else ["runs", 0, "runner", field]
    invalid = result["records"][0]["invalid_numbers"][0]
    assert invalid["path"] == expected_path
    assert invalid["reason"] == ("NaN" if math.isnan(number) else "Infinity" if number > 0 else "-Infinity")
    assert invalid["replacement"] is None
    if field == "official_score":
        assert result["records"][0]["official_score"] is None
        assert result["records"][0]["official_success"] is False
        assert result["counts_all_records"]["hybrid"]["evaluation_completed"] == 0
    else:
        target = result["records"][0]["calls"] if field.endswith("calls") else result["records"][0]["timing"]
        key = {"episode_wall_ms": "episode_ms", "duration_ms": "agent_ms"}.get(field, field)
        assert target[key] is None
        assert result["records"][0]["official_success"] is True
        assert result["counts_all_records"]["hybrid"]["evaluation_completed"] == 1
    assert not result["pairs"][0]["eligible"]
    assert stats(result)["agent_ms"]["n"] == 0 and stats(result)["episode_ms"]["n"] == 0


def test_nonfinite_shared_configuration_and_source_metadata_are_json_safe_and_exclude_pairs(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"files": {"src/a.py": "hash"}, "measurement": math.inf}))
    summary = write_summary(tmp_path, [row("hybrid"), row("vlm")],
                            experiment_settings={"pause_s": math.nan})
    output = tmp_path / "report.json"
    assert comparison.main(["--run", f"trial={summary}", "--source", f"trial={source}",
                            "--output", str(output)]) == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["records"]) == 2
    assert all(len(record["invalid_numbers"]) == 2 for record in result["records"])
    assert all(record["official_success"] for record in result["records"])
    assert not result["pairs"][0]["eligible"]
    assert result["pairs"][0]["configuration"]["unknown"]["pause_s"] == ["hybrid", "vlm"]


@pytest.mark.parametrize("change", [
    {"evaluation_status": "error"}, {"official_score": 0.5}, {"success": False},
    {"cleanup_status": "unconfirmed_close_error"}, {"run_status": "cleanup_error"},
])
def test_independent_evaluation_and_cleanup_cannot_be_replaced_by_completion_claim(tmp_path, change):
    hybrid = row("hybrid", **change)
    hybrid["runner"]["status"] = "completed"
    result = comparison.summarize([("trial", write_summary(tmp_path, [hybrid, row("vlm")]))])
    assert not result["pairs"][0]["eligible"]
    assert stats(result)["agent_ms"]["n"] == 0


def test_duplicate_mode_is_not_silently_last_wins_and_missing_suite_tasks_remain_visible(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), row("hybrid"), row("vlm", task="gimp/b")])
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([{"task": "os/task"}, {"task": "gimp/b"}, {"task": "vs_code/c"}]))
    result = comparison.summarize([("trial", path)], suite_path=suite)
    assert len(result["records"]) == 3
    assert len(result["pairs"]) == 3
    assert result["counts_all_records"]["hybrid"]["records"] == 2
    assert sum(issue["kind"] == "duplicate_mode" for issue in result["issues"]) == 1
    assert sum(issue["kind"] == "missing_mode" for issue in result["issues"]) == 4
    assert not any(pair["eligible"] for pair in result["pairs"])


@pytest.mark.parametrize(("field", "left", "right"), [
    ("model", "model-a", "model-b"), ("pause", 0.5, 1.0),
    ("source_sha256", "aaa", "bbb"), ("max_steps", 10, 20),
    ("no_a11y", True, False), ("task_sha256", "task-a", "task-b"),
    ("singleton_fast_path", True, False),
    ("jev_request_layout", "full_v1", "deduplicated_v1"),
    ("jev_request_layout", "deduplicated_v1", "shared_fields_v2"),
    ("desktop_stage_contract", "legacy_v23", "stage_intent_v24"),
    ("desktop_stage_contract", "stage_intent_v24", "property_intent_v25"),
    ("desktop_stage_contract", "property_intent_v51", "property_intent_v52"),
    ("visual_box_space", "normalized_1000", "image_pixels"),
    ("desktop_snapshot_encoding", "control_defaults_v1", "shared_ancestor_paths_v1_if_saving_4096_bytes"),
    ("model_request_timeout_s", 40.0, 90.0),
    ("model_thinking_budget_tokens", "default", 512),
    ("model_thinking_budget_tokens", 512, 1024),
    ("stale_observation_recovery", "recompile_once_v1", "recompile_once_v2"),
    ("desktop_observation_budget", "names1k_text24k", "names2k_text96k_v1"),
    ("reuse_verified_post_snapshot", True, False),
    ("passive_settling", True, False),
    ("passive_settling_version", "public_structure_v1", "future_v2"),
    ("startup_timeout_s", 300, 600),
    ("vm_provider", "qemu", "docker"),
    ("vm_network_mode", "user_restricted_loopback", "docker_bridge_loopback"),
])
def test_known_configuration_conflicts_exclude_successful_pair(tmp_path, field, left, right):
    path = write_summary(tmp_path, [row("hybrid", **{field: left}), row("vlm", **{field: right})])
    result = comparison.summarize([("trial", path)])
    assert result["issues"][0]["kind"] == "configuration_mismatch"
    assert stats(result)["agent_ms"]["n"] == 0
    assert result["counts_all_records"]["hybrid"]["official_successes"] == 1


def test_intended_mode_difference_not_marked_configuration_conflict(tmp_path):
    path = write_summary(tmp_path, [row("hybrid", plan_horizon=6, live_jev=True),
                                    row("vlm", plan_horizon=1, live_jev=False)])
    result = comparison.summarize([("trial", path)])
    assert result["pairs"][0]["eligible"]
    assert not result["issues"]


def test_row_configuration_overrides_summary_and_shared_missing_is_not_verified(tmp_path):
    path = write_summary(tmp_path, [row("hybrid", config={"pause_s": 0.5}),
                                    row("vlm", config={"pause": 0.5})], config={"pause": 1})
    result = comparison.summarize([("trial", path)])
    config = result["pairs"][0]["configuration"]
    assert config["matched"]["pause_s"] == 0.5
    assert config["unknown"]["source_sha256"] == ["hybrid", "vlm"]
    assert config["complete_configuration_equality_verified"] is False


def test_task_hash_maps_and_mode_planning_metadata_compare_only_relevant_task(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")],
                         task_sha256={"os/task": "abc", "unrelated/task": "def"},
                         osworld_tracked_files_dirty=True,
                         planning_by_mode={"hybrid": {"plan_ttl_s": 300}, "vlm": {"plan_ttl_s": 300}})
    result = comparison.summarize([("trial", path)])
    matched = result["pairs"][0]["configuration"]["matched"]
    assert matched["task_sha256"] == "abc"
    assert matched["plan_ttl_s"] == 300
    assert matched["osworld_dirty"] is True


def test_new_experiment_settings_are_compared_and_old_missing_values_stay_unknown(tmp_path):
    settings = {"model_thinking": "off", "singleton_fast_path": True, "reuse_verified_post_snapshot": True,
                "passive_settling": True, "passive_settling_version": "public_structure_v1",
                "stale_observation_recovery": "recompile_once_v2",
                "desktop_observation_budget": "names2k_text96k_v1",
                "pause_s": 0.5, "max_steps": 30,
                "timeout_s": 420, "startup_timeout_s": 600, "a11y": True}
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")], experiment_settings=settings)
    result = comparison.summarize([("trial", path)])
    matched = result["pairs"][0]["configuration"]["matched"]
    assert all(matched[key] == value for key, value in settings.items())
    old_path = write_summary(tmp_path, [row("hybrid"), row("vlm")], name="old.json")
    old_result = comparison.summarize([("old", old_path)])
    unknown = old_result["pairs"][0]["configuration"]["unknown"]
    assert all(unknown[key] == ["hybrid", "vlm"] for key in settings)


def test_row_experiment_settings_can_reveal_mismatch_hidden_by_shared_metadata(tmp_path):
    path = write_summary(tmp_path, [row("hybrid", experiment_settings={"model_thinking": "on"}),
                                    row("vlm")], experiment_settings={"model_thinking": "off"})
    result = comparison.summarize([("trial", path)])
    conflict = result["pairs"][0]["configuration"]["mismatches"]["model_thinking"]
    assert conflict == {"hybrid": "on", "vlm": "off"}
    assert stats(result)["agent_ms"]["n"] == 0


def test_empty_summary_does_not_look_like_completed_successful_experiment(tmp_path):
    result = comparison.summarize([("trial", write_summary(tmp_path, []))])
    assert result["integrity_status"] == "issues_found"
    assert result["issues"][0]["kind"] == "empty_summary"
    assert stats(result)["agent_ms"]["geometric_mean"] is None


def test_repeats_pair_by_label_and_without_sources_are_not_pooled(tmp_path):
    first = write_summary(tmp_path, [row("hybrid", agent=100), row("vlm", agent=200)], name="a.json")
    second = write_summary(tmp_path, [row("hybrid", agent=200), row("vlm", agent=100)], name="b.json")
    result = comparison.summarize([("first", first), ("second", second)])
    assert stats(result, "first")["agent_ms"]["median"] == 2
    assert stats(result, "second")["agent_ms"]["median"] == 0.5
    assert len(result["by_source_and_configuration"]) == 2
    assert all(len(group["labels"]) == 1 for group in result["by_source_and_configuration"])


def test_source_manifest_changes_split_versions_but_identical_file_hashes_can_group_repeats(tmp_path):
    summary = write_summary(tmp_path, [row("hybrid"), row("vlm")])
    source_a, source_b, source_c = (tmp_path / name for name in ("a-source.json", "b-source.json", "c.json"))
    source_a.write_text(json.dumps({"files": {"src/agent.py": "aaa"}, "archive_sha256": "first"}))
    source_b.write_text(json.dumps({"files": {"src/agent.py": "aaa"}, "archive_sha256": "second"}))
    source_c.write_text(json.dumps({"files": {"src/agent.py": "changed"}}))
    result = comparison.summarize(
        [(label, summary) for label in ("first", "repeat", "changed")],
        source_specs=[("first", source_a), ("repeat", source_b), ("changed", source_c)],
    )
    groups = result["by_source_and_configuration"]
    assert len(groups) == 2
    assert sorted(group["labels"] for group in groups) == [["changed"], ["first", "repeat"]]
    assert all(group["configuration_equality_verified"] is False for group in groups)


@pytest.mark.parametrize("field,left,right", [
    ("pause", 0.5, 1),
    ("desktop_snapshot_encoding", "control_defaults_v1", "shared_ancestor_paths_v1_if_saving_4096_bytes"),
    ("model_request_timeout_s", 40.0, 90.0),
    ("model_thinking_budget_tokens", "default", 512),
    ("model_thinking_budget_tokens", 512, 1024),
])
def test_shared_source_still_separates_reported_configuration_changes(tmp_path, field, left, right):
    first = write_summary(tmp_path, [row("hybrid", **{field: left}), row("vlm", **{field: left})],
                          name="a.json")
    second = write_summary(tmp_path, [row("hybrid", **{field: right}), row("vlm", **{field: right})],
                           name="b.json")
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    result = comparison.summarize([("a", first), ("b", second)],
                                  source_specs=[("a", source), ("b", source)])
    assert len(result["by_source_and_configuration"]) == 2


def test_missing_snapshot_encoding_remains_unknown_instead_of_assuming_pooling_disabled(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")])
    result = comparison.summarize([("trial", path)])
    facts = result["pairs"][0]["configuration"]
    assert facts["unknown"]["desktop_snapshot_encoding"] == ["hybrid", "vlm"]
    assert not facts["complete_configuration_equality_verified"]


def test_network_rescue_success_is_counted_but_not_pooled_as_pure_jev_speedup(tmp_path):
    hybrid, vlm = row("hybrid", agent=50), row("vlm", agent=100)
    hybrid["runner"].update(network_recovery_attempts=1, network_recovery_actions=1,
                             network_recovery_verified=1, jev_request_attempts=3, jev_network_failures=1)
    vlm["runner"]["network_recovery_attempts"] = 0
    path = write_summary(tmp_path, [hybrid, vlm],
                         experiment_settings={"jev_network_recovery": "vlm_once_v1"})
    report = comparison.summarize([("trial", path)])
    assert report["counts_all_records"]["hybrid"]["official_successes"] == 1
    assert stats(report)["agent_ms"]["n"] == 0
    rescue = report["by_run"]["trial"]["network_recovery_analysis"]
    assert rescue["eligible_pairs"] == 1
    assert rescue["speedup_baseline_divided_by_hybrid"]["agent_ms"]["median"] == 2
    assert rescue["successful_pair_calls"]["hybrid"]["jev_request_attempts"]["sum"] == 3
    assert report["pairs"][0]["execution_policy"] == "network_recovery_used"


def test_failed_short_rescue_never_creates_any_speedup(tmp_path):
    hybrid = row("hybrid", agent=1, success=False)
    hybrid["runner"]["network_recovery_attempts"] = 1
    path = write_summary(tmp_path, [hybrid, row("vlm", agent=1000)])
    report = comparison.summarize([("trial", path)])
    assert stats(report)["agent_ms"]["n"] == 0
    assert report["by_run"]["trial"]["network_recovery_analysis"]["eligible_pairs"] == 0
    assert report["counts_all_records"]["hybrid"]["records"] == 1


def test_network_rescue_setting_and_actual_use_each_separate_repeat_pools(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    runs = []
    for label, setting, used in (("disabled", "disabled", 0), ("enabled", "vlm_once_v1", 0),
                                  ("rescued", "vlm_once_v1", 1), ("legacy", None, None)):
        rows = [row("hybrid"), row("vlm")]
        if used is not None:
            rows[0]["runner"]["network_recovery_attempts"] = used
            rows[1]["runner"]["network_recovery_attempts"] = 0
        settings = {} if setting is None else {"jev_network_recovery": setting}
        runs.append((label, write_summary(tmp_path, rows, name=f"{label}.json",
                                           experiment_settings=settings)))
    report = comparison.summarize(runs, source_specs=[(label, source) for label, _ in runs])
    assert len(report["by_source_and_configuration"]) == 4
    legacy = next(pair for pair in report["pairs"] if pair["label"] == "legacy")
    assert legacy["configuration"]["unknown"]["jev_network_recovery"] == ["hybrid", "vlm"]
    assert legacy["execution_policy"] == "unknown"


def test_known_recovery_configuration_mismatch_excludes_paired_ratio(tmp_path):
    path = write_summary(tmp_path, [
        row("hybrid", experiment_settings={"jev_network_recovery": "vlm_once_v1"}),
        row("vlm", experiment_settings={"jev_network_recovery": "disabled"}),
    ])
    report = comparison.summarize([("trial", path)])
    assert not report["pairs"][0]["eligible"]
    assert "jev_network_recovery" in report["pairs"][0]["configuration"]["mismatches"]


def test_settling_versions_and_missing_version_never_pool_even_with_same_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    summaries = []
    for label, version in (("v1", "public_structure_v1"), ("v2", "future_v2"), ("unknown", None)):
        settings = {"passive_settling": True}
        if version is not None:
            settings["passive_settling_version"] = version
        summary = write_summary(tmp_path, [row("vlm"), row("hybrid")],
                                name=f"{label}.json", experiment_settings=settings)
        summaries.append((label, summary))
    result = comparison.summarize(summaries, source_specs=[(label, source) for label, _ in summaries])
    assert len(result["by_source_and_configuration"]) == 3


@pytest.mark.parametrize("field", ["singleton_fast_path", "reuse_verified_post_snapshot", "passive_settling"])
def test_boolean_optimization_changes_and_unknown_legacy_runs_never_pool(tmp_path, field):
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    summaries = []
    for label, enabled in (("enabled", True), ("repeat", True), ("disabled", False), ("legacy", None)):
        settings = {} if enabled is None else {field: enabled}
        path = write_summary(tmp_path, [row("hybrid"), row("vlm")], name=f"{label}.json",
                             experiment_settings=settings)
        summaries.append((label, path))
    result = comparison.summarize(summaries, source_specs=[(label, source) for label, _ in summaries])
    groups = result["by_source_and_configuration"]
    assert len(groups) == 3
    assert sorted(group["labels"] for group in groups) == [["disabled"], ["enabled", "repeat"], ["legacy"]]
    for pair in result["pairs"]:
        config = pair["configuration"]
        assert pair["eligible"]
        if pair["label"] == "legacy":
            assert config["unknown"][field] == ["hybrid", "vlm"]
            assert field not in config["matched"]
        else:
            assert config["matched"][field] is (pair["label"] != "disabled")


@pytest.mark.parametrize(("field", "qemu_value", "docker_value"), [
    ("vm_provider", "qemu", "docker"),
    ("vm_network_mode", "user_restricted_loopback", "docker_bridge_loopback"),
    ("startup_timeout_s", 600, 300),
])
def test_vm_environment_and_startup_budget_changes_with_unknown_legacy_runs_never_pool(
    tmp_path, field, qemu_value, docker_value,
):
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    summaries = []
    for label, value in (("qemu", qemu_value), ("repeat", qemu_value),
                         ("docker", docker_value), ("legacy", None)):
        settings = {} if value is None else {field: value}
        path = write_summary(tmp_path, [row("hybrid"), row("vlm")], name=f"{label}.json",
                             experiment_settings=settings)
        summaries.append((label, path))
    result = comparison.summarize(summaries, source_specs=[(label, source) for label, _ in summaries])
    groups = result["by_source_and_configuration"]
    assert sorted(group["labels"] for group in groups) == [["docker"], ["legacy"], ["qemu", "repeat"]]
    for pair in result["pairs"]:
        config = pair["configuration"]
        assert pair["eligible"]
        if pair["label"] == "legacy":
            assert config["unknown"][field] == ["hybrid", "vlm"]
            assert field not in config["matched"]
        else:
            assert config["matched"][field] == (docker_value if pair["label"] == "docker" else qemu_value)


@pytest.mark.parametrize(("first_horizons", "second_horizons"), [
    ((2, 1), (6, 1)), ((6, 1), (6, 2)), ((6, 1), (None, 1)),
])
def test_shared_source_does_not_pool_changed_or_unknown_per_mode_horizons(
    tmp_path, first_horizons, second_horizons,
):
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    summaries = []
    for label, horizons in (("first", first_horizons), ("second", second_horizons)):
        path = write_summary(tmp_path, [row(mode, plan_horizon=horizon)
                                        for mode, horizon in zip(comparison.MODES, horizons, strict=True)],
                             name=f"{label}.json")
        summaries.append((label, path))
    result = comparison.summarize(summaries, source_specs=[(label, source) for label, _ in summaries])
    groups = result["by_source_and_configuration"]
    assert len(groups) == 2
    assert sorted(group["labels"] for group in groups) == [["first"], ["second"]]
    assert all(group["eligible_pairs"] == 1 for group in groups)
    assert all(not pair["configuration"]["mismatches"] for pair in result["pairs"])
    for group in groups:
        expected = first_horizons if group["labels"] == ["first"] else second_horizons
        assert tuple(group["mode_settings"][mode]["plan_horizon"] for mode in comparison.MODES) == expected


def test_top_level_horizons_are_available_but_actual_row_horizon_takes_precedence(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    first = write_summary(tmp_path, [row("hybrid"), row("vlm")], name="first.json",
                          planning_by_mode={"hybrid": {"plan_horizon": 6}, "vlm": {"plan_horizon": 1}})
    second = write_summary(tmp_path, [row("hybrid", plan_horizon=6), row("vlm", plan_horizon=1)],
                           name="second.json", planning_by_mode={"hybrid": {"plan_horizon": 2}})
    result = comparison.summarize([("first", first), ("second", second)],
                                  source_specs=[("first", source), ("second", source)])
    groups = result["by_source_and_configuration"]
    assert len(groups) == 1
    assert groups[0]["labels"] == ["first", "second"]
    assert groups[0]["mode_settings"]["hybrid"]["plan_horizon"] == 6
    assert groups[0]["mode_settings"]["vlm"]["plan_horizon"] == 1


@pytest.mark.parametrize("bad_time", [0, -1, None, "10", True])
def test_bad_agent_time_does_not_discard_valid_episode_time(tmp_path, bad_time):
    path = write_summary(tmp_path, [row("hybrid", agent=bad_time), row("vlm")])
    result = comparison.summarize([("trial", path)])
    assert stats(result)["agent_ms"]["n"] == 0
    assert stats(result)["episode_ms"]["n"] == 1
    assert result["issues"][0]["kind"] == "invalid_timing"


def test_every_input_is_hashed_as_read_and_no_task_or_manifest_path_is_opened(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")])
    source = tmp_path / "source.json"
    source.write_text('{"files": {"/nonexistent/private/task.json": "not-opened"}}')
    before = {item: item.read_bytes() for item in (path, source)}
    result = comparison.summarize([("trial", path)], source_specs=[("trial", source)])
    for item in result["inputs"]:
        original = before[Path(item["path"])]
        assert item["sha256"] == hashlib.sha256(original).hexdigest()
        assert Path(item["path"]).read_bytes() == original
    assert all(row["input_index"] == 1 for row in result["records"])


def test_multiple_files_for_same_label_report_overlaps_instead_of_overwriting(tmp_path):
    first = write_summary(tmp_path, [row("hybrid")], name="a.json")
    second = write_summary(tmp_path, [row("hybrid"), row("vlm")], name="b.json")
    result = comparison.summarize([("trial", first), ("trial", second)])
    assert len(result["inputs"]) == 2
    assert result["pairs"][0]["exclusions"] == ["duplicate_mode:hybrid"]


def test_invalid_rows_are_counted_as_issues_without_losing_other_rows(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), None, {"task": "os/task", "mode": "wrong"}])
    result = comparison.summarize([("trial", path)])
    assert len(result["records"]) == 1
    assert sum(issue["kind"] == "invalid_record" for issue in result["issues"]) == 2


def test_cli_output_never_overwrites_summary_and_success_is_official_not_runner(tmp_path, capsys):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")])
    original = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        comparison.main(["--run", f"trial={path}", "--output", str(path)])
    assert error.value.code == 2
    assert path.read_bytes() == original
    assert comparison.main(["--run", f"trial={path}"]) == 0
    assert json.loads(capsys.readouterr().out)["by_run"]["trial"]["eligible_pairs"] == 1


def test_fixed_suite_contains_exactly_two_per_requested_domain_and_no_task_internals():
    suite = json.loads((Path(__file__).parents[1] / "scripts" / "osworld_fixed8.json").read_text())
    assert len(suite) == 8
    assert len({item["task"] for item in suite}) == 8
    assert all(set(item) == {"task", "label"} for item in suite)
    for domain in ("os", "vs_code", "gimp", "thunderbird"):
        assert sum(item["task"].startswith(domain + "/") for item in suite) == 2


def test_missing_request_timeout_remains_unknown_instead_of_assuming_the_legacy_default(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")])
    report = comparison.summarize([("trial", path)])
    facts = report["pairs"][0]["configuration"]
    assert facts["unknown"]["model_request_timeout_s"] == ["hybrid", "vlm"]
    assert not facts["complete_configuration_equality_verified"]


@pytest.mark.parametrize("budget", ["default", 1, 512, 2**40])
def test_declared_thinking_budget_is_compared_without_inferring_provider_limits(tmp_path, budget):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")],
                         experiment_settings={"model_thinking_budget_tokens": budget})
    report = comparison.summarize([("trial", path)])
    pair = report["pairs"][0]
    assert pair["configuration"]["matched"]["model_thinking_budget_tokens"] == budget
    assert not pair["configuration"]["invalid"] and pair["eligible"]
    assert not pair["configuration"]["complete_configuration_equality_verified"]
    assert "provider honoring is unverified" in report["policy"]["model_thinking_budget"]


@pytest.mark.parametrize("value", [None, "absent"])
def test_historical_missing_or_null_thinking_budget_stays_unknown(tmp_path, value):
    settings = {} if value == "absent" else {"model_thinking_budget_tokens": None}
    path = write_summary(tmp_path, [row("hybrid", **settings), row("vlm", **settings)])
    report = comparison.summarize([("trial", path)])
    facts = report["pairs"][0]["configuration"]
    assert facts["unknown"]["model_thinking_budget_tokens"] == ["hybrid", "vlm"]
    assert "model_thinking_budget_tokens" not in facts["matched"]
    assert not facts["invalid"]


@pytest.mark.parametrize("container", [None, "config", "experiment_settings"])
def test_explicit_unknown_row_budget_does_not_inherit_shared_budget(tmp_path, container):
    unknown = {"model_thinking_budget_tokens": None}
    hybrid = row("hybrid", **(unknown if container is None else {container: unknown}))
    path = write_summary(tmp_path, [hybrid, row("vlm")],
                         experiment_settings={"model_thinking_budget_tokens": 512})
    report = comparison.summarize([("trial", path)])
    facts = report["pairs"][0]["configuration"]
    assert facts["unknown"]["model_thinking_budget_tokens"] == ["hybrid"]
    assert "model_thinking_budget_tokens" not in facts["matched"]
    assert report["records"][0]["config"]["model_thinking_budget_tokens"] is None


@pytest.mark.parametrize("left,right", [
    (True, 1), (False, 0), (0, 0), (-1, -1), (1.0, 1), ("512", 512),
    ("automatic", "automatic"), ("DEFAULT", "DEFAULT"), ([], []), ({}, {}),
])
def test_invalid_thinking_budget_never_certifies_equal_config_or_speedup(tmp_path, left, right):
    path = write_summary(tmp_path, [row("hybrid", model_thinking_budget_tokens=left),
                                    row("vlm", model_thinking_budget_tokens=right)])
    report = comparison.summarize([("trial", path)])
    pair = report["pairs"][0]
    assert "model_thinking_budget_tokens" not in pair["configuration"]["matched"]
    assert pair["configuration"]["invalid"]["model_thinking_budget_tokens"]["hybrid"] == left
    assert "invalid_configuration" in pair["exclusions"] and not pair["eligible"]
    assert stats(report)["agent_ms"]["n"] == 0
    assert all(report["counts_all_records"][mode]["official_successes"] == 1
               for mode in ("hybrid", "vlm"))


@pytest.mark.parametrize("thinking", ["off", "default"])
@pytest.mark.parametrize("both_modes", [False, True])
def test_positive_budget_with_explicitly_non_on_thinking_is_invalid(tmp_path, thinking, both_modes):
    path = write_summary(tmp_path, [
        row("hybrid", model_thinking=thinking, model_thinking_budget_tokens=512),
        row("vlm", model_thinking=thinking if both_modes else "on", model_thinking_budget_tokens=512),
    ])
    report = comparison.summarize([("trial", path)])
    pair = report["pairs"][0]
    expected = {"hybrid": 512, "vlm": 512} if both_modes else {"hybrid": 512}
    assert pair["configuration"]["invalid"]["model_thinking_budget_tokens"] == expected
    assert "model_thinking_budget_tokens" not in pair["configuration"]["matched"]
    assert "invalid_configuration" in pair["exclusions"] and not pair["eligible"]
    assert stats(report)["agent_ms"]["n"] == 0
    assert report["counts_all_records"]["hybrid"]["official_successes"] == 1


@pytest.mark.parametrize("thinking", [None, "absent"])
def test_positive_budget_does_not_invent_missing_historical_thinking_setting(tmp_path, thinking):
    setting = {} if thinking == "absent" else {"model_thinking": None}
    path = write_summary(tmp_path, [
        row("hybrid", **setting, model_thinking_budget_tokens=512),
        row("vlm", **setting, model_thinking_budget_tokens=512),
    ])
    report = comparison.summarize([("trial", path)])
    facts = report["pairs"][0]["configuration"]
    assert facts["unknown"]["model_thinking"] == ["hybrid", "vlm"]
    assert facts["matched"]["model_thinking_budget_tokens"] == 512
    assert not facts["invalid"] and not facts["complete_configuration_equality_verified"]


@pytest.mark.parametrize("hybrid_budget", [None, 256])
def test_repeat_identity_preserves_actual_per_mode_budget_when_not_matched(tmp_path, hybrid_budget):
    paths = []
    for label, budget in (("a", 512), ("b", 1024)):
        paths.append((label, write_summary(
            tmp_path, [row("hybrid", model_thinking_budget_tokens=hybrid_budget),
                       row("vlm", model_thinking_budget_tokens=budget)], name=f"{label}.json",
        )))
    source = tmp_path / "source.json"
    source.write_text('{"files": {"src/a.py": "same"}}')
    report = comparison.summarize(paths, source_specs=[("a", source), ("b", source)])
    assert len(report["by_source_and_configuration"]) == 2
    assert all(len(group["labels"]) == 1 for group in report["by_source_and_configuration"])
