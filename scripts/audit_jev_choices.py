"""Audit whether Jev actually differs from the deterministic first-candidate ablation."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit_trace(events, *, model):
    observation_id = None
    candidates = None
    rows = []
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if kind == "observation":
            observation_id = payload["observation_id"]
            candidates = None
        elif kind == "candidates":
            candidates = payload["items"]
        elif kind == "decision" and payload.get("model") == model:
            if candidates is None or payload["observation_id"] != observation_id:
                raise ValueError("model decision has no matching observed candidate set")
            ids = [c["id"] for c in candidates]
            if len(set(ids)) != len(ids) or payload["candidate_id"] not in ids:
                raise ValueError("model decision does not bind to unique offered candidate ids")
            rank = ids.index(payload["candidate_id"])
            rows.append({
                "decision_id": payload["decision_id"], "selected_id": payload["candidate_id"],
                "first_candidate_id": ids[0], "selected_rank_zero_based": rank,
                "matches_first": rank == 0, "candidate_count": len(candidates),
                "distinct_intents": len({c["intent"] for c in candidates}),
                "selected_route": candidates[rank].get("arguments", {}).get("route"),
                "raw_confidence": payload["confidence"], "latency_ms": payload["latency_ms"],
            })
    return rows


def audit_run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    model = manifest["models"]["jev"]
    rows, episodes, missing = [], [], []
    for result_file in sorted(root.glob("*/result.json")):
        result = json.loads(result_file.read_text())
        key = {k: result[k] for k in ("task", "seed", "mode", "benchmark_success")}
        reported = result.get("jev_calls", 0)
        trace = result_file.parent / "trace.jsonl"
        if not trace.exists():
            missing.append({**key, "reported_jev_calls": reported})
            continue
        choices = audit_trace([json.loads(line) for line in trace.read_text().splitlines()], model=model)
        rows.extend({**key, "trace": str(trace.relative_to(root)), **choice} for choice in choices)
        episodes.append({**key, "reported_jev_calls": reported, "recorded_jev_decisions": len(choices)})
    return {
        "jev_model": model, "recorded_decisions": len(rows),
        "matches_first": sum(r["matches_first"] for r in rows),
        "differs_from_first": sum(not r["matches_first"] for r in rows),
        "multiple_intents": sum(r["distinct_intents"] > 1 for r in rows),
        "one_intent_multiple_routes": sum(r["distinct_intents"] == 1 and r["candidate_count"] > 1
                                           for r in rows),
        "selected_routes": dict(Counter(r["selected_route"] for r in rows)),
        "episodes": episodes, "missing_traces": missing, "decisions": rows,
        "interpretation": "A different choice is not evidence of a better choice. Raw confidence is "
                          "uncalibrated. Compare matched adaptive_rule outcomes and all-attempt costs "
                          "separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"decisions", "episodes", "missing_traces"}}))


if __name__ == "__main__":
    main()
