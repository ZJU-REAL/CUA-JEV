"""Compare complete attempts without treating infrastructure failures as scores."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def official_valid(row):
    return row.get("official") is not None or (
        row.get("evaluation_status") == "valid" and row.get("official_score") is not None
    )


def mean_seconds(rows, field):
    if not rows or any(type(row.get(field)) not in (float, int) for row in rows):
        return None
    return statistics.mean(row[field] for row in rows) / 1000


def vlm_tokens(rows):
    if any(row.get("planner_calls", 0) and "total_tokens" not in row.get("model_usage", {}) for row in rows):
        return None
    return sum(row.get("model_usage", {}).get("total_tokens", 0) for row in rows)


def summarize(document, modes=("vlm", "adaptive_rule", "adaptive")):
    rows = document["runs"]
    if not modes or len(set(modes)) != len(modes) or any(row["mode"] not in modes for row in rows):
        raise ValueError("the complete predeclared mode set is required")
    blocks = defaultdict(dict)
    for row in rows:
        key = str(row["task"]), row.get("seed", 0)
        if row["mode"] in blocks[key]:
            raise ValueError("duplicate task/seed/mode attempt")
        blocks[key][row["mode"]] = row
    matched = [
        group
        for group in blocks.values()
        if set(group) == set(modes) and all(official_valid(row) for row in group.values())
    ]
    result = {
        "planned": document["planned"],
        "completed": len(rows),
        "valid_complete_blocks": len(matched),
        "observed_blocks": len(blocks),
        "modes": {},
    }
    for mode in modes:
        attempts = [row for row in rows if row["mode"] == mode]
        valid = [row for row in attempts if official_valid(row)]
        paired = [group[mode] for group in matched]
        result["modes"][mode] = {
            "attempts": len(attempts),
            "valid_evaluations": len(valid),
            "unevaluated_attempts": len(attempts) - len(valid),
            "successes": sum(row["benchmark_success"] for row in valid),
            "success_rate_valid": sum(row["benchmark_success"] for row in valid) / len(valid)
            if valid
            else None,
            "total_wall_mean_s_all_attempts": mean_seconds(attempts, "total_wall_ms"),
            "agent_mean_s_all_attempts": mean_seconds(attempts, "duration_ms"),
            "total_wall_mean_s_valid_matched": mean_seconds(paired, "total_wall_ms"),
            "agent_mean_s_valid_matched": mean_seconds(paired, "duration_ms"),
            "vlm_calls": sum(row.get("planner_calls", 0) for row in attempts),
            "vlm_reported_tokens": vlm_tokens(attempts),
            "jev_calls": sum(row.get("jev_calls", 0) for row in attempts),
            "failure_reasons": dict(
                Counter(
                    row.get(
                        "reason", row.get("error_code") or row.get("infrastructure_error_type", "unknown")
                    )
                    for row in attempts
                    if not row.get("benchmark_success")
                )
            ),
        }
    result["paired_against_vlm"] = {}
    if "vlm" in modes:
        for mode in modes:
            if mode == "vlm":
                continue
            counts = Counter(
                (bool(g["vlm"]["benchmark_success"]), bool(g[mode]["benchmark_success"])) for g in matched
            )
            result["paired_against_vlm"][mode] = {
                "both_success": counts[True, True],
                "baseline_only": counts[True, False],
                "comparison_only": counts[False, True],
                "both_fail": counts[False, False],
            }
    result["interpretation"] = (
        "Descriptive paired subsets, not statistical noninferiority or full benchmark scores. "
        "Unevaluated attempts remain explicit; timing alone does not establish a quality-preserving speedup."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=/absolute/path/to/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["vlm", "adaptive_rule", "adaptive"])
    args = parser.parse_args()
    result = {}
    for entry in args.run:
        label, separator, path = entry.partition("=")
        if not separator or not label or label in result:
            parser.error("unique label=summary.json entries are required")
        result[label] = summarize(json.loads(Path(path).read_text()), args.modes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
