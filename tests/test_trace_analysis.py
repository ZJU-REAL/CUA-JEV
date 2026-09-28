import json

import pytest

from cua_jev.trace import JsonlTrace
from cua_jev.trace_analysis import analyze_traces


def test_trace_analysis_aggregates_channel_probability_not_sensitive_state(tmp_path):
    path = tmp_path / "runs.jsonl"
    trace = JsonlTrace(path, run_id="run-1")
    trace.append("observation", {"text": "private task content"})
    trace.append("candidates", {"items": [
        {"id": "gui-a", "channel": "gui", "capability": "browser.click"},
        {"id": "cli-a", "channel": "cli", "capability": "tool.run"},
        {"id": "cli-b", "channel": "cli", "capability": "tool.run"},
    ]})
    trace.append("decision", {
        "model": "jev-1.13.0",
        "candidate_id": "cli-a", "probabilities": {
            "gui-a": 0.2, "cli-a": 0.5, "cli-b": 0.3,
        }, "confidence": 0.7, "latency_ms": 43,
    })
    trace.append("model_usage", {
        "role": "vision", "model": "example-vlm",
        "usage": {"prompt_tokens": 286, "completion_tokens": 65},
    })
    trace.append("episode", {"status": "success", "duration_ms": 1200})
    result = analyze_traces([path])
    assert result["success_rate"] == 1
    assert result["wall_ms_median"] == 1200
    assert result["decision_ms_median"] == 43
    assert result["selected_channels"] == {"cli": 1}
    assert result["selected_routes"] == {"cli:tool.run": 1}
    assert result["mean_probability_by_available_channel"] == {"cli": 0.8, "gui": 0.2}
    assert result["mean_probability_by_available_route"] == {
        "cli:tool.run": 0.8, "gui:browser.click": 0.2,
    }
    assert result["cost_usd"] is None
    assert result["reported_model_tokens"]["vision"]["prompt_tokens"] == 286
    assert "private task content" not in str(result)


def test_trace_analysis_rejects_empty_or_invalid_files(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        analyze_traces([])
    path = tmp_path / "broken.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        analyze_traces([path])


def test_overlapping_trace_exports_do_not_inflate_outcomes_or_usage(tmp_path):
    first = tmp_path / "first.jsonl"
    trace = JsonlTrace(first, run_id="same-run")
    trace.append("model_usage", {
        "role": "planner", "requests": 2, "usage": {"total_tokens": 100},
    })
    trace.append("episode", {"status": "success", "duration_ms": 1200})
    second = tmp_path / "overlap.jsonl"
    # One event was copied into another export, whose input may also be repeated.
    second.write_text(json.dumps(trace.events[-1]) + "\n", encoding="utf-8")
    result = analyze_traces([first, second, first])
    assert result["runs_seen"] == 1
    assert result["completed_episodes"] == 1
    assert result["successes"] == 1
    assert result["model_requests"] == {"planner": 2}
    assert result["reported_model_tokens"] == {"planner": {"total_tokens": 100}}


def test_conflicting_trace_identity_is_rejected_instead_of_counted(tmp_path):
    path = tmp_path / "conflict.jsonl"
    trace = JsonlTrace(path, run_id="same-run")
    trace.append("episode", {"status": "success", "duration_ms": 1200})
    conflict = {**trace.events[0], "payload": {"status": "task_failure"}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(conflict) + "\n")
    with pytest.raises(ValueError, match="conflicting trace event"):
        analyze_traces([path])
