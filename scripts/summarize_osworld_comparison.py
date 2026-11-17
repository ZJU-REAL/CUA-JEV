"""Compare immutable OSWorld summary snapshots without opening tasks or running agents.

Example:
    python scripts/summarize_osworld_comparison.py \
        --run trial1=runs/trial1/summary.json --source trial1=runs/trial1-source.json \
        --suite scripts/osworld_fixed8.json --output runs/comparison.json

A label identifies a run/repeat, not a claim that two labels used the same code.
Source manifests are explicit caller declarations; their files are never opened.
Only labels with the same declared source, available configuration, and per-mode
settings are pooled.
Missing configuration remains unknown, including inside such a pool. Exit code 1
means integrity/completeness issues; recorded task failures alone do not cause it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("hybrid", "vlm")
# Plan horizon and live_jev intentionally differ between the two modes.
CONFIG_FIELDS = {
    "model": ("model",),
    "model_thinking": ("model_thinking",),
    "model_thinking_budget_tokens": ("model_thinking_budget_tokens",),
    "model_request_timeout_s": ("model_request_timeout_s",),
    "singleton_fast_path": ("singleton_fast_path",),
    "reuse_verified_post_snapshot": ("reuse_verified_post_snapshot",),
    "reuse_verified_pending_snapshot": ("reuse_verified_pending_snapshot",),
    "whole_goal_recheck": ("whole_goal_recheck",),
    "reuse_whole_goal_planning_snapshot": ("reuse_whole_goal_planning_snapshot",),
    "whole_goal_planning_snapshot_reuse_version": ("whole_goal_planning_snapshot_reuse_version",),
    "inband_whole_goal_recheck": ("inband_whole_goal_recheck",),
    "inband_whole_goal_recheck_version": ("inband_whole_goal_recheck_version",),
    "jev_capacity_recovery_policy": ("jev_capacity_recovery_policy",),
    "navigation_continuation": ("navigation_continuation",),
    "navigation_continuation_version": ("navigation_continuation_version",),
    "intermediate_fill_observation": ("intermediate_fill_observation",),
    "worker_instrumentation": ("worker_instrumentation",),
    "jev_network_recovery": ("jev_network_recovery",),
    "jev_capacity_recovery": ("jev_capacity_recovery",),
    "jev_request_layout": ("jev_request_layout",),
    "desktop_stage_contract": ("desktop_stage_contract",),
    "desktop_output_format": ("desktop_output_format",),
    "desktop_planning_observation": ("desktop_planning_observation",),
    "visual_box_space": ("visual_box_space",),
    "desktop_snapshot_encoding": ("desktop_snapshot_encoding",),
    "stale_observation_recovery": ("stale_observation_recovery",),
    "desktop_observation_budget": ("desktop_observation_budget",),
    "menu_grounding": ("menu_grounding",),
    "menu_ocr_worker_sha256": ("menu_ocr_worker_sha256",),
    "menu_ocr_timeout_s": ("menu_ocr_timeout_s",),
    "menu_ocr_process_layout": ("menu_ocr_process_layout",),
    "passive_settling": ("passive_settling",),
    "passive_settling_version": ("passive_settling_version",),
    "pause_s": ("pause", "pause_s"),
    "max_steps": ("max_steps", "maxsteps"),
    "timeout_s": ("timeout", "timeout_s"),
    "startup_timeout_s": ("startup_timeout", "startup_timeout_s"),
    "a11y": ("a11y",),
    "plan_ttl_s": ("plan_ttl_s",),
    "live_vlm": ("live_vlm",),
    "decision_location": ("decision_location",),
    "vm_transport": ("vm_transport",),
    "vm_provider": ("vm_provider",),
    "vm_network_mode": ("vm_network_mode",),
    "vm_image_sha256": ("vm_image_sha256", "base_image_sha256"),
    "osworld_revision": ("osworld_revision", "osworld_commit", "osworld_git_head"),
    "osworld_dirty": ("osworld_dirty", "osworld_tracked_files_dirty"),
    "task_sha256": ("task_sha256",),
    "source_sha256": ("source_sha256", "agent_source_sha256", "source_fingerprint"),
}
COUNT_FIELDS = (
    "planner_calls", "jev_calls", "jev_request_attempts", "jev_network_failures",
    "network_recovery_attempts", "network_recovery_actions", "network_recovery_verified",
    "stale_recovery_attempts",
    "stale_snapshot_reuses",
)
TIMING_FIELDS = ("agent_ms", "episode_ms")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def json_safe_numbers(value, path=()):
    """Retain invalid numeric provenance while making every parsed input JSON-safe."""
    if isinstance(value, float) and not math.isfinite(value):
        reason = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return None, [{"path": list(path), "reason": reason, "replacement": None}]
    invalid = []
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            clean[key], found = json_safe_numbers(item, (*path, key))
            invalid.extend(found)
        return clean, invalid
    if isinstance(value, list):
        clean = []
        for index, item in enumerate(value):
            normalized, found = json_safe_numbers(item, (*path, index))
            clean.append(normalized)
            invalid.extend(found)
        return clean, invalid
    return value, invalid


def read_json(path, kind, label=None):
    """Hash exactly the bytes parsed; never modify or follow paths inside JSON."""
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    value, invalid = json_safe_numbers(json.loads(content.decode("utf-8-sig")))
    record = {
        "kind": kind, "path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }
    if label is not None:
        record["label"] = label
    if invalid:
        record["nonfinite_numbers"] = invalid
    return value, record


def finite_number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    return math.isfinite(value) and (value > 0 if positive else value >= 0)


def effective_config(row, summary):
    """Only compare allowlisted reported fields, not model credentials or task JSON."""
    # These opt-ins did not exist in older reports. Their historical absence
    # means disabled; an explicitly recorded null remains unknown below.
    config = {"jev_capacity_recovery": "disabled", "reuse_verified_pending_snapshot": False,
              "whole_goal_recheck": "disabled", "jev_capacity_recovery_policy": "vlm_once_v1",
              "reuse_whole_goal_planning_snapshot": False,
              "whole_goal_planning_snapshot_reuse_version": "disabled",
              "inband_whole_goal_recheck": False, "inband_whole_goal_recheck_version": "disabled",
              "navigation_continuation": False, "navigation_continuation_version": "disabled"}
    for source in (summary, row):
        containers = [source]
        containers += [source[key] for key in ("config", "configuration", "metadata", "experiment_settings")
                       if isinstance(source.get(key), dict)]
        for container in containers:
            for field, aliases in CONFIG_FIELDS.items():
                for alias in aliases:
                    # Explicit unknown settings do not inherit a known shared
                    # cap or output format. Missing history remains unknown.
                    if (field in {"model_thinking_budget_tokens", "desktop_output_format",
                                  "desktop_planning_observation", "jev_capacity_recovery",
                                  "reuse_verified_pending_snapshot", "whole_goal_recheck",
                                  "reuse_whole_goal_planning_snapshot", "inband_whole_goal_recheck",
                                  "whole_goal_planning_snapshot_reuse_version",
                                  "inband_whole_goal_recheck_version", "jev_capacity_recovery_policy",
                                  "navigation_continuation", "navigation_continuation_version"}
                            and alias in container):
                        config[field] = container[alias]
                        break
                    if alias in container and container[alias] is not None:
                        value = container[alias]
                        if field == "task_sha256" and isinstance(value, dict):
                            value = value.get(row["task"])
                        if value is not None:
                            config[field] = value
                        break
            if "a11y" not in container and isinstance(container.get("no_a11y"), bool):
                config["a11y"] = not container["no_a11y"]
        planning_by_mode = source.get("planning_by_mode")
        planning = planning_by_mode.get(row["mode"], {}) if isinstance(planning_by_mode, dict) else {}
        if "plan_ttl_s" not in config and isinstance(planning, dict) and "plan_ttl_s" in planning:
            config["plan_ttl_s"] = planning["plan_ttl_s"]
    return config


def normalize_row(row, summary, label, input_index, row_index):
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    score = row.get("official_score")
    execution_started = isinstance(runner.get("status"), str) and bool(runner["status"].strip())
    if not execution_started and row.get("agent_execution_status") == "unknown":
        execution_started = None
    evaluation_completed = row.get("evaluation_status") == "ok" and finite_number(score) and score <= 1
    official_success = (
        row.get("success") is True and evaluation_completed and score == 1
    )
    failures = []
    if not official_success:
        failures.append("official_success_not_confirmed")
    if row.get("success") is True and not official_success:
        failures.append("inconsistent_success_evidence")
    for field in ("episode_status", "run_status"):
        if row.get(field) is not None and row[field] != "success":
            failures.append(f"{field}:{row[field]}")
    if row.get("cleanup_status") not in (None, "confirmed"):
        failures.append(f"cleanup_status:{row['cleanup_status']}")
    counts = {key: runner.get(key) for key in COUNT_FIELDS}
    horizon = row.get("plan_horizon")
    planning_by_mode = summary.get("planning_by_mode")
    if horizon is None and isinstance(planning_by_mode, dict):
        planning = planning_by_mode.get(row["mode"])
        if isinstance(planning, dict):
            horizon = planning.get("plan_horizon")
    config = effective_config(row, summary)
    return {
        "label": label, "task": row["task"], "mode": row["mode"],
        "input_index": input_index, "row_index": row_index,
        "reported_success": row.get("success"), "official_success": official_success,
        "execution_started": execution_started, "evaluation_completed": evaluation_completed,
        "startup_not_run": row.get("episode_status") == "startup_error" and execution_started is False,
        "official_score": score, "evaluation_status": row.get("evaluation_status"),
        "episode_status": row.get("episode_status"), "run_status": row.get("run_status"),
        "cleanup_status": row.get("cleanup_status"), "runner_status": runner.get("status"),
        "failures": failures, "config": config,
        "mode_settings": {"plan_horizon": horizon, "live_jev": row.get("live_jev"),
                          "model_thinking_budget_tokens": config.get("model_thinking_budget_tokens"),
                          "network_recovery_use": (
                              "used" if runner["network_recovery_attempts"] > 0 else "not_used"
                          ) if type(runner.get("network_recovery_attempts")) is int
                          and runner["network_recovery_attempts"] >= 0 else "unknown"},
        "timing": {"agent_ms": runner.get("duration_ms"), "episode_ms": row.get("episode_wall_ms")},
        "calls": counts,
    }


def compare_config(hybrid, vlm):
    matched, mismatches, unknown, invalid = {}, {}, {}, {}
    for key in CONFIG_FIELDS:
        left, right = hybrid["config"].get(key), vlm["config"].get(key)
        if key in {"reuse_verified_pending_snapshot", "reuse_whole_goal_planning_snapshot",
                   "inband_whole_goal_recheck", "navigation_continuation"}:
            invalid_values = {mode: value for mode, value in (("hybrid", left), ("vlm", right))
                              if value is not None and type(value) is not bool}
            if invalid_values:
                invalid[key] = invalid_values
                if left is None or right is None:
                    unknown[key] = [mode for mode, value in (("hybrid", left), ("vlm", right))
                                    if value is None]
                continue
        allowed = {
            "intermediate_fill_observation": {"full_v1", "image_only_v1"},
            "worker_instrumentation": {"disabled", "fill_observation_rpc_v1"},
            "jev_capacity_recovery": {"disabled", "vlm_once_v1", "vlm_phase_v2", "vlm_phase_v3"},
            "jev_capacity_recovery_policy": {"vlm_once_v1", "vlm_phase_v2", "vlm_phase_v3"},
            "whole_goal_planning_snapshot_reuse_version": {"disabled", "successful_completion_full_frame_v1"},
            "inband_whole_goal_recheck_version": {"disabled", "all_original_goal_review_v1"},
            "navigation_continuation_version": {"disabled", "observed_transition_v1"},
            "jev_request_layout": {"full_v1", "deduplicated_v1", "shared_fields_v2",
                                   "shared_fields_v3", "shared_fields_v4", "shared_fields_v5",
                                   "shared_fields_v6", "shared_fields_v7"},
            "whole_goal_recheck": {"disabled", "current_goal_recheck_v1"},
            "desktop_output_format": {"unconstrained_v1", "json_schema_v50", "tool_declaration_v52",
                                      "semantic_tools_v53", "semantic_tools_v54", "semantic_tools_v55",
                                      "semantic_tools_v56", "semantic_tools_v57", "semantic_tools_v58",
                                      "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64",
                                          "semantic_json_v66"},
            "desktop_planning_observation": {"full_multimodal_v1", "native_first_v51"},
        }.get(key)
        if allowed is not None:
            invalid_values = {
                mode: value for mode, value in (("hybrid", left), ("vlm", right))
                if value is not None and (type(value) is not str or value not in allowed)
            }
            if invalid_values:
                invalid[key] = invalid_values
                if left is None or right is None:
                    unknown[key] = [mode for mode, value in (("hybrid", left), ("vlm", right))
                                    if value is None]
                continue
        if key == "model_thinking_budget_tokens":
            invalid_values = {}
            for mode, record in (("hybrid", hybrid), ("vlm", vlm)):
                value = record["config"].get(key)
                valid_value = (
                    (type(value) is int and value > 0) or (type(value) is str and value == "default")
                )
                explicit_thinking_conflict = (
                    type(value) is int and value > 0
                    and record["config"].get("model_thinking") in ("off", "default")
                )
                if value is not None and (not valid_value or explicit_thinking_conflict):
                    invalid_values[mode] = value
            if invalid_values:
                invalid[key] = invalid_values
                if left is None or right is None:
                    unknown[key] = [mode for mode, value in (("hybrid", left), ("vlm", right))
                                    if value is None]
                continue
        if left is None or right is None:
            unknown[key] = [mode for mode, value in (("hybrid", left), ("vlm", right)) if value is None]
        elif left != right:
            mismatches[key] = {"hybrid": left, "vlm": right}
        else:
            matched[key] = left
    return {
        "status": ("mismatch" if mismatches else "invalid_configuration" if invalid
                   else "no_observed_mismatch"),
        "matched": matched, "mismatches": mismatches, "unknown": unknown,
        "invalid": invalid,
        "complete_configuration_equality_verified": False,
    }


def ratio_summary(values):
    if not values:
        return {"n": 0, "geometric_mean": None, "median": None}
    return {
        "n": len(values),
        "geometric_mean": math.exp(statistics.mean(math.log(value) for value in values)),
        "median": statistics.median(values),
    }


def aggregate_pairs(pairs, *, rescue=False):
    eligible = [pair for pair in pairs if pair["eligible"]
                and (rescue or pair.get("execution_policy") != "network_recovery_used")]
    timings = {}
    for metric in TIMING_FIELDS:
        values = [pair["speedup"][metric] for pair in eligible if pair["speedup"].get(metric) is not None]
        timings[metric] = ratio_summary(values)
    counts = {}
    for mode in MODES:
        counts[mode] = {}
        for metric in COUNT_FIELDS:
            values = [pair["calls"][mode][metric] for pair in eligible
                      if finite_number(pair["calls"][mode].get(metric))]
            counts[mode][metric] = {
                "n": len(values), "sum": sum(values),
                "median": statistics.median(values) if values else None,
            }
    result = {
        "pairs": len(pairs), "eligible_pairs": len(eligible), "excluded_pairs": len(pairs) - len(eligible),
        "speedup_baseline_divided_by_hybrid": timings, "successful_pair_calls": counts,
    }
    if not rescue:
        result["network_recovery_analysis"] = aggregate_pairs(
            [pair for pair in pairs if pair.get("execution_policy") == "network_recovery_used"], rescue=True,
        )
    return result


def mode_counts(records):
    result = {}
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        successes = sum(row["official_success"] for row in rows)
        executed = sum(row["execution_started"] is True for row in rows)
        evaluated = sum(row["evaluation_completed"] for row in rows)
        evaluated_successes = sum(row["official_success"] for row in rows if row["evaluation_completed"])
        counts = {}
        for metric in COUNT_FIELDS:
            values = [row["calls"][metric] for row in rows if finite_number(row["calls"].get(metric))]
            counts[metric] = {"recorded": len(values), "missing_or_invalid": len(rows) - len(values),
                              "sum": sum(values)}
        result[mode] = {
            "records": len(rows), "official_successes": successes,
            "not_successful": len(rows) - successes,
            "success_rate": successes / len(rows) if rows else None,
            "execution_started": executed,
            "execution_not_started": sum(row["execution_started"] is False for row in rows),
            "execution_unknown": sum(row["execution_started"] is None for row in rows),
            "evaluation_completed": evaluated,
            "evaluation_not_completed": len(rows) - evaluated,
            "evaluation_not_run": sum(row["evaluation_status"] == "not_run" for row in rows),
            "startup_not_run": sum(row["startup_not_run"] for row in rows),
            "evaluated_successes": evaluated_successes,
            "evaluated_not_successful": evaluated - evaluated_successes,
            "evaluated_success_rate": evaluated_successes / evaluated if evaluated else None,
            "episode_status_counts": dict(Counter(row["episode_status"] or "unknown" for row in rows)),
            "evaluation_status_counts": dict(Counter(row["evaluation_status"] or "unknown" for row in rows)),
            "status_counts": dict(Counter(row["run_status"] or row["episode_status"] or "unknown"
                                          for row in rows)),
            "all_record_calls": counts,
        }
    return result


def source_identity(manifest):
    # A manifest's archive digest can change merely from packaging timestamps.
    # Canonical file hashes identify its contents; no listed file is opened.
    if isinstance(manifest, dict) and isinstance(manifest.get("files"), dict) and manifest["files"]:
        return "files-sha256:" + hashlib.sha256(canonical(manifest["files"]).encode()).hexdigest()
    return "manifest-sha256:" + hashlib.sha256(canonical(manifest).encode()).hexdigest()


def summarize(run_specs, *, source_specs=(), suite_path=None):
    inputs, records, issues = [], [], []
    labels, tasks, declared_sources = set(), set(), {}
    source_invalid = {}

    def register_input(info):
        input_index = len(inputs)
        inputs.append(info)
        invalid = [{"input_index": input_index, **item} for item in info.get("nonfinite_numbers", [])]
        if invalid:
            issues.append({"kind": "nonfinite_input_numbers", "input_index": input_index,
                           "label": info.get("label"), "numbers": invalid})
        return input_index, invalid

    for label, path in source_specs:
        if label in declared_sources:
            raise ValueError(f"duplicate source label: {label}")
        manifest, info = read_json(path, "source", label)
        _, source_invalid[label] = register_input(info)
        declared_sources[label] = source_identity(manifest)
    suite_labels = {}
    if suite_path is not None:
        suite, info = read_json(suite_path, "suite")
        register_input(info)
        if not isinstance(suite, list) or not suite:
            raise ValueError("suite must be a nonempty list of task/label objects")
        for item in suite:
            if not isinstance(item, dict) or not isinstance(item.get("task"), str):
                raise ValueError("suite entries require a task string")
            if item["task"] in suite_labels:
                raise ValueError("suite contains a duplicate task")
            suite_labels[item["task"]] = item.get("label", item["task"])
        tasks.update(suite_labels)
    for label, path in run_specs:
        if not label.strip():
            raise ValueError("run labels must be nonempty")
        labels.add(label)
        summary, info = read_json(path, "summary", label)
        input_index, invalid = register_input(info)
        if not isinstance(summary, dict) or not isinstance(summary.get("runs"), list):
            raise ValueError("summary must contain a runs list")
        if not summary["runs"]:
            issues.append({"kind": "empty_summary", "label": label, "input_index": input_index})
        for row_index, row in enumerate(summary["runs"]):
            if (not isinstance(row, dict) or not isinstance(row.get("task"), str)
                    or not row["task"] or row.get("mode") not in MODES):
                issues.append({"kind": "invalid_record", "label": label,
                               "input_index": input_index, "row_index": row_index})
                continue
            tasks.add(row["task"])
            normalized = normalize_row(row, summary, label, input_index, row_index)
            relevant_invalid = source_invalid.get(label, []) + [
                item for item in invalid
                if len(item["path"]) < 2 or item["path"][0] != "runs" or item["path"][1] == row_index
            ]
            if relevant_invalid:
                normalized["invalid_numbers"] = relevant_invalid
                normalized["failures"].append("nonfinite_input_numbers")
            records.append(normalized)
            if suite_path and row["task"] not in suite_labels:
                issues.append({"kind": "outside_suite", "label": label, "task": row["task"]})
    if not labels:
        raise ValueError("at least one labeled run is required")
    if set(declared_sources) - labels:
        raise ValueError("source labels must also appear in --run")
    grouped = defaultdict(list)
    for index, row in enumerate(records):
        grouped[(row["label"], row["task"], row["mode"])].append(index)
    pairs = []
    for label in sorted(labels):
        for task in sorted(tasks):
            members = {mode: grouped[(label, task, mode)] for mode in MODES}
            pair = {"label": label, "task": task, "records": members, "eligible": False,
                    "exclusions": [], "speedup": {}, "calls": {}}
            for mode, indexes in members.items():
                if len(indexes) != 1:
                    kind = "missing_mode" if not indexes else "duplicate_mode"
                    issues.append({"kind": kind, "label": label, "task": task, "mode": mode,
                                   "record_count": len(indexes)})
                    pair["exclusions"].append(f"{kind}:{mode}")
            if not pair["exclusions"]:
                hybrid, vlm = (records[members[mode][0]] for mode in MODES)
                uses = [row["mode_settings"]["network_recovery_use"] for row in (hybrid, vlm)]
                pair["execution_policy"] = (
                    "network_recovery_used" if "used" in uses else
                    "unknown" if "unknown" in uses else "no_network_recovery"
                )
                pair["configuration"] = compare_config(hybrid, vlm)
                if pair["configuration"]["mismatches"]:
                    issues.append({"kind": "configuration_mismatch", "label": label, "task": task,
                                   "fields": pair["configuration"]["mismatches"]})
                    pair["exclusions"].append("configuration_mismatch")
                if pair["configuration"]["invalid"]:
                    issues.append({"kind": "invalid_configuration", "label": label, "task": task,
                                   "fields": pair["configuration"]["invalid"]})
                    pair["exclusions"].append("invalid_configuration")
                for row in (hybrid, vlm):
                    pair["exclusions"].extend(f"{row['mode']}:{failure}" for failure in row["failures"])
                    pair["calls"][row["mode"]] = row["calls"]
                pair["eligible"] = not pair["exclusions"]
                for metric in TIMING_FIELDS:
                    left, right = hybrid["timing"][metric], vlm["timing"][metric]
                    valid_times = finite_number(left, positive=True) and finite_number(right, positive=True)
                    if pair["eligible"] and valid_times:
                        ratio = right / left
                        if math.isfinite(ratio) and ratio > 0:
                            pair["speedup"][metric] = ratio
                            continue
                    pair["speedup"][metric] = None
                    if pair["eligible"]:
                        pair.setdefault("timing_exclusions", []).append(metric)
                        issues.append({"kind": "invalid_timing", "label": label, "task": task,
                                       "metric": metric})
            pairs.append(pair)
    by_run = {}
    for label in sorted(labels):
        selected = [pair for pair in pairs if pair["label"] == label]
        by_run[label] = {
            "declared_source_identity": declared_sources.get(label),
            "expected_mode_slots": len(tasks) * len(MODES),
            "missing_mode_slots": sum(not members for pair in selected
                                      for members in pair["records"].values()),
            "duplicate_mode_slots": sum(len(members) > 1 for pair in selected
                                        for members in pair["records"].values()),
            "counts": mode_counts([row for row in records if row["label"] == label]),
            **aggregate_pairs(selected),
        }
    # Never pool labels just because their names look like repeat numbers. A
    # shared declared source is required; known configuration differences split
    # groups as well. A mode's horizon may differ from the other mode's horizon,
    # but changing that horizon between repeats changes the experiment. Unknown
    # metadata is reported, not silently certified equal.
    pools = defaultdict(list)
    for pair in pairs:
        matched = pair.get("configuration", {}).get("matched", {})
        shared = {key: value for key, value in matched.items() if key != "task_sha256"}
        mode_settings = {
            mode: records[indexes[0]]["mode_settings"] if len(indexes) == 1 else None
            for mode, indexes in pair["records"].items()
        }
        identity = declared_sources.get(pair["label"], "unverified-label:" + pair["label"])
        pools[(identity, canonical(shared), canonical(mode_settings))].append(pair)
    source_groups = [{
        "source_identity": identity, "matched_configuration": json.loads(configuration),
        "mode_settings": json.loads(mode_settings),
        "labels": sorted({pair["label"] for pair in selected}),
        "configuration_equality_verified": False,
        **aggregate_pairs(selected),
    } for (identity, configuration, mode_settings), selected in sorted(pools.items())]
    return {
        "kind": "osworld_offline_comparison", "schema_version": 1, "inputs": inputs,
        "policy": {
            "success": "success=true AND evaluation_status=ok AND official_score=1",
            "execution_started": "runner object with a nonempty status string; does not imply GUI input",
            "execution_unknown": "explicit agent_execution_status=unknown without a valid runner report",
            "evaluation_completed": "evaluation_status=ok AND finite non-boolean score in [0,1]",
            "startup_not_run": "episode_status=startup_error AND no valid runner; not inferred from cleanup",
            "evaluated_success_rate": "official successes / evaluation_completed, or null if none",
            "nonfinite_numbers": "replaced with null and paths/reasons retained; affected pairs excluded",
            "speedup": "vlm / hybrid time, conditioned on unique successful pairs without known conflicts",
            "causal_speedup_established": False,
            "timings": {"agent_ms": "runner.duration_ms", "episode_ms": "episode_wall_ms"},
            "failure_times_count_as_speedup": False,
            "network_recovery": "actual recovery pairs have a separate network_recovery_analysis; "
                                "excluded from primary speedup statistics, never pure Jev benefit; "
                                "unknown historical recovery remains unknown",
            "full_configuration_equality_verified": False,
            "model_thinking_budget": "reported positive integer or explicit default is a request setting; "
                                     "missing/null remains unknown; provider honoring is unverified; "
                                     "invalid values exclude speedup pairs",
            "source_manifests": "caller-declared provenance, not independent proof of runtime identity",
            "pooling": "same declared source, matched metadata and per-mode settings; unknowns unverified",
            "denominators": "all valid records include duplicates/failures; missing slots listed separately",
            "scope": "explicit suite plus observed tasks" if suite_path else "union of observed tasks only",
        },
        "expected_tasks": [{"task": task, "label": suite_labels.get(task)} for task in sorted(tasks)],
        "counts_all_records": mode_counts(records), "issues": issues,
        "integrity_status": "issues_found" if issues else "no_detected_issues",
        "records": records, "pairs": pairs, "by_run": by_run, "by_source_and_configuration": source_groups,
    }


def labeled_path(value):
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", action="append", required=True, type=labeled_path, metavar="LABEL=SUMMARY")
    parser.add_argument("--source", action="append", default=[], type=labeled_path, metavar="LABEL=MANIFEST")
    parser.add_argument("--suite", type=Path, help="Explicit list of expected task/label objects")
    parser.add_argument("--output", type=Path, help="Create a new JSON report; never overwrite a file")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.run, source_specs=args.source, suite_path=args.suite)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.exit(2, f"Cannot summarize inputs ({type(error).__name__}). No input files were modified.\n")
    return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
