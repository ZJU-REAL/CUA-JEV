"""Summarize all attempts and explicitly identified paired successful cases."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def mean_agent_seconds(rows):
    if any("duration_ms" not in row for row in rows):
        return None
    return statistics.mean(row["duration_ms"] for row in rows) / 1000


def summarize(document, modes=("vlm", "plan_order", "hybrid")):
    rows = document["runs"]
    if not modes or len(set(modes)) != len(modes) or any(r["mode"] not in modes for r in rows):
        raise ValueError("provide the full predeclared, nonduplicated mode set")
    groups = defaultdict(dict)
    task_groups = defaultdict(list)
    for row in rows:
        key = row["task"], row["seed"]
        if row["mode"] in groups[key]:
            raise ValueError("duplicate case/mode")
        groups[key][row["mode"]] = row
        task_groups[row["task"]].append(row)
    result = {"planned": document["planned"], "completed": len(rows), "by_mode": {}, "by_task": {}}
    for mode in modes:
        items = [r for r in rows if r["mode"] == mode]
        if not items:
            continue
        jev_usage = [r.get("reported_tokens_by_role", {}).get("jev", {}) for r in items]
        result["by_mode"][mode] = {
            "episodes": len(items), "successes": sum(r["benchmark_success"] for r in items),
            "valid_evaluations": sum(r.get("official") is not None for r in items),
            "agent_mean_s_all": mean_agent_seconds(items),
            "episode_mean_s_all": statistics.mean(r["episode_wall_ms"] for r in items) / 1000,
            "total_mean_s_all": (statistics.mean(r["total_wall_ms"] for r in items) / 1000
                                 if all("total_wall_ms" in r for r in items) else None),
            "planner_calls": sum(r.get("planner_calls", 0) for r in items),
            "jev_calls": sum(r.get("jev_calls", 0) for r in items),
            "vlm_reported_total_tokens": sum(r.get("model_usage", {}).get("total_tokens", 0) for r in items),
            "jev_reported_input_tokens": sum(u.get("prompt_tokens", 0) for u in jev_usage),
            "jev_reported_output_tokens": sum(u.get("completion_tokens", 0) for u in jev_usage),
            "episodes_missing_jev_usage": sum(r.get("jev_calls", 0) > 0 and not u
                                               for r, u in zip(items, jev_usage, strict=True)),
            "failure_reasons": dict(Counter(r.get("reason", r.get("infrastructure_error_type", "unknown"))
                                            for r in items if not r["benchmark_success"])),
            "full_observations": sum(r.get("observation_metrics", {}).get("full", 0) for r in items),
            "structured_only_observations": sum(r.get("observation_metrics", {}).get("structured_only", 0)
                                                for r in items),
            "post_snapshot_reuses": sum(r.get("post_snapshot_reuses", 0) for r in items),
            "dispatch_refreshes": sum(r.get("dispatch_refreshes", 0) for r in items),
            "dispatch_s2_replans": sum(r.get("dispatch_s2_replans", 0) for r in items),
        }
    for task, items in task_groups.items():
        result["by_task"][task] = {
            mode: {"n": len(part), "successes": sum(r["benchmark_success"] for r in part),
                   "agent_mean_s_all": mean_agent_seconds(part)}
            for mode in modes if (part := [r for r in items if r["mode"] == mode])
        }
    common = []
    excluded = []
    for key, group in groups.items():
        if set(group) != set(modes):
            excluded.append({"task": key[0], "seed": key[1], "reason": "incomplete_mode_block"})
        elif not all(r["benchmark_success"] for r in group.values()):
            excluded.append({"task": key[0], "seed": key[1], "reason": "not_successful_in_every_mode"})
        elif (None in {r.get("initial_observation_sha256") for r in group.values()} or
              len({r.get("initial_observation_sha256") for r in group.values()}) != 1):
            excluded.append({"task": key[0], "seed": key[1], "reason": "initial_observations_not_matched"})
        elif (None in {r.get("goal_sha256") for r in group.values()} or
              len({r.get("goal_sha256") for r in group.values()}) != 1):
            excluded.append({"task": key[0], "seed": key[1], "reason": "goals_not_matched"})
        else:
            common.append(group)
    times = {mode: sum(g[mode]["duration_ms"] for g in common) for mode in modes}
    result["paired_successful_in_all_modes"] = {
        "n": len(common), "total_case_blocks": len(groups), "excluded": excluded,
        "agent_mean_s": {mode: times[mode] / len(common) / 1000 if common else None for mode in modes},
        "vlm_to_hybrid_ratio_of_total_agent_time": (
            times["vlm"] / times["hybrid"] if times.get("hybrid") and "vlm" in times else None),
        "plan_order_to_hybrid_ratio_of_total_agent_time": (
            times["plan_order"] / times["hybrid"]
            if times.get("hybrid") and "plan_order" in times else None),
        "vlm_to_mode_agent_time_ratios": {
            mode: times["vlm"] / times[mode] if times[mode] and "vlm" in times else None
            for mode in modes if mode != "vlm"
        },
        "interpretation": "Conditional descriptive subset, not an all-task speedup or significance claim.",
    }
    result["paired_outcomes_against_vlm"] = {}
    for mode in modes:
        if mode == "vlm" or "vlm" not in modes:
            continue
        counts = Counter()
        valid = 0
        for group in groups.values():
            if "vlm" not in group or mode not in group:
                continue
            baseline, comparison = group["vlm"], group[mode]
            if any(baseline.get(k) is None or baseline.get(k) != comparison.get(k)
                   for k in ("initial_observation_sha256", "goal_sha256")):
                continue
            if baseline.get("official") is None or comparison.get("official") is None:
                continue
            valid += 1
            counts[(bool(baseline["benchmark_success"]), bool(comparison["benchmark_success"]))] += 1
        result["paired_outcomes_against_vlm"][mode] = {
            "valid_matched_pairs": valid, "total_case_blocks": len(groups),
            "both_success": counts[True, True], "baseline_only_success": counts[True, False],
            "comparison_only_success": counts[False, True], "both_fail": counts[False, False],
            "interpretation": ("Observed paired outcomes only; "
                               "small-sample ties do not establish noninferiority."),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["vlm", "plan_order", "hybrid"])
    args = parser.parse_args()
    document = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    result = summarize(document, tuple(args.modes))
    (args.output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = ["task", "seed", "mode", "benchmark_success", "status", "reason", "steps",
              "planner_calls", "jev_calls", "duration_ms", "episode_wall_ms", "total_wall_ms",
              "planner_wall_ms", "jev_wall_ms"]
    with (args.output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in document["runs"])
    print(json.dumps({"completed": result["completed"], "by_mode": result["by_mode"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
