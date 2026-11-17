import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "adaptive_comparison", Path(__file__).parents[1] / "scripts/summarize_adaptive_comparison.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unevaluated_attempts_and_missing_tokens_cannot_be_hidden():
    rows = [
        {
            "task": "case",
            "mode": "vlm",
            "official": {"reward": 1},
            "benchmark_success": True,
            "total_wall_ms": 1000,
            "duration_ms": 700,
            "planner_calls": 1,
            "model_usage": {"total_tokens": 5},
        },
        {
            "task": "case",
            "mode": "adaptive",
            "official": None,
            "benchmark_success": False,
            "total_wall_ms": 100,
            "planner_calls": 1,
        },
    ]
    result = module.summarize({"planned": 2, "runs": rows}, ("vlm", "adaptive"))
    assert result["valid_complete_blocks"] == 0
    assert result["modes"]["adaptive"]["unevaluated_attempts"] == 1
    assert result["modes"]["adaptive"]["vlm_reported_tokens"] is None
    assert result["modes"]["adaptive"]["success_rate_valid"] is None
    assert result["modes"]["adaptive"]["total_wall_mean_s_valid_matched"] is None
    with pytest.raises(ValueError, match="duplicate"):
        module.summarize({"planned": 3, "runs": rows + rows[:1]})


def test_valid_osworld_zero_score_counts_as_failure_and_retains_its_time():
    rows = [
        {
            "task": "case",
            "mode": mode,
            "official_score": score,
            "evaluation_status": "valid",
            "benchmark_success": bool(score),
            "total_wall_ms": 1000,
            "duration_ms": 700,
        }
        for mode, score in (("vlm", 1), ("adaptive", 0))
    ]
    result = module.summarize({"planned": 2, "runs": rows}, ("vlm", "adaptive"))
    assert result["valid_complete_blocks"] == 1
    assert result["modes"]["adaptive"]["total_wall_mean_s_valid_matched"] == 1
    assert result["paired_against_vlm"]["adaptive"]["baseline_only"] == 1


def test_partial_run_does_not_infer_a_smaller_mode_set():
    rows = [{"task": "case", "mode": "vlm", "official": {"reward": 1},
             "benchmark_success": True, "total_wall_ms": 1000, "duration_ms": 700}]
    result = module.summarize({"planned": 3, "runs": rows})
    assert result["valid_complete_blocks"] == 0
    assert result["modes"]["adaptive"]["attempts"] == 0
    assert result["modes"]["vlm"]["total_wall_mean_s_valid_matched"] is None
