import pytest

from scripts.summarize_miniwob_pilot import summarize


def test_conditional_speed_comparison_does_not_hide_failures_from_main_results():
    rows = []
    for seed in (1, 2):
        for mode, duration in (("vlm", 100), ("plan_order", 80), ("hybrid", 90)):
            rows.append({"task": "fixture", "seed": seed, "mode": mode, "duration_ms": duration,
                         "episode_wall_ms": duration + 20, "official": {"reward": 1},
                         "benchmark_success": (seed, mode) != (2, "hybrid"),
                         "initial_observation_sha256": f"state-{seed}", "goal_sha256": "goal"})
    result = summarize({"planned": 6, "runs": rows})
    assert result["by_mode"]["hybrid"]["episodes"] == 2
    assert result["by_mode"]["hybrid"]["successes"] == 1
    common = result["paired_successful_in_all_modes"]
    assert common["n"] == 1 and common["total_case_blocks"] == 2
    assert common["excluded"] == [{"task": "fixture", "seed": 2,
                                    "reason": "not_successful_in_every_mode"}]
    assert common["vlm_to_hybrid_ratio_of_total_agent_time"] == 100 / 90
    rows[0].pop("duration_ms")
    rows[0]["benchmark_success"] = False
    assert summarize({"planned": 6, "runs": rows})["by_mode"]["vlm"]["agent_mean_s_all"] is None


def test_adaptive_modes_require_explicit_protocol_and_retain_paired_failures():
    rows = [{"task": "fixture", "seed": seed, "mode": mode, "duration_ms": 100,
             "episode_wall_ms": 120, "total_wall_ms": 140, "official": {"reward": 1},
             "benchmark_success": (seed, mode) != (2, "adaptive"),
             "initial_observation_sha256": f"state-{seed}", "goal_sha256": "goal"}
            for seed in (1, 2) for mode in ("vlm", "adaptive_rule", "adaptive")]
    document = {"planned": 6, "runs": rows}
    with pytest.raises(ValueError, match="predeclared"):
        summarize(document)
    result = summarize(document, ("vlm", "adaptive_rule", "adaptive"))
    assert result["by_mode"]["adaptive"]["successes"] == 1
    assert result["by_mode"]["adaptive"]["total_mean_s_all"] == 0.14
    paired = result["paired_outcomes_against_vlm"]["adaptive"]
    assert paired["valid_matched_pairs"] == 2 and paired["baseline_only_success"] == 1
    assert result["paired_successful_in_all_modes"]["n"] == 1
    rows[2]["goal_sha256"] = "mismatch"
    result = summarize(document, ("vlm", "adaptive_rule", "adaptive"))
    assert result["paired_outcomes_against_vlm"]["adaptive"]["valid_matched_pairs"] == 1
    assert result["paired_successful_in_all_modes"]["n"] == 0
