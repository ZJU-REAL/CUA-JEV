from types import SimpleNamespace

import pytest

from cua_jev.benchmark_miniwob import (
    BenchmarkTerminalFailure,
    Evaluator,
    PlanOrderPolicy,
    aggregate,
    matching_initial_states,
    schedule,
)


def test_missing_initial_evidence_does_not_count_as_matched():
    rows = [{"task": "a", "seed": 1, "mode": mode} for mode in ("vlm", "hybrid")]
    result = matching_initial_states(rows)[0]
    assert not result["all_initial_observations_match"]
    assert not result["all_goals_match"]
    for row in rows:
        row.update(initial_observation_sha256="same", goal_sha256="same")
    assert matching_initial_states(rows)[0]["all_initial_observations_match"]


def test_matched_schedule_is_complete_and_rotates_order():
    modes = ["vlm", "plan_order", "hybrid"]
    jobs = list(schedule(["a", "b"], [100, 101, 102], modes))
    assert len(jobs) == len(set(jobs)) == 18
    assert [jobs[i][2] for i in (0, 3, 6)] == modes
    for task in ("a", "b"):
        for seed in (100, 101, 102):
            assert {m for t, s, m in jobs if (t, s) == (task, seed)} == set(modes)


def test_evaluator_accepts_only_terminal_upstream_success_and_stops_failure():
    state = [1.0, False, "do not give this message to the agent", {"answer": "PRIVATE"}]
    task = SimpleNamespace(validate=lambda page, chat: tuple(state))
    evaluator = Evaluator(task, object())
    assert evaluator.accept(object()) is False
    state[1] = True
    assert evaluator.accept(object()) is True
    state[0] = -1.0
    with pytest.raises(BenchmarkTerminalFailure):
        evaluator.accept(object())
    assert "PRIVATE" not in str(evaluator.last)


def test_ablation_uses_compiler_order_without_task_answers_or_model():
    candidates = [SimpleNamespace(id="second"), SimpleNamespace(id="first")]
    decision = PlanOrderPolicy().choose(SimpleNamespace(observation_id="obs"), candidates)
    assert decision.candidate_id == "second"
    assert decision.model == "deterministic-plan-order"


def test_metrics_keep_failures_and_invalid_evaluations_in_denominator():
    rows = [{"mode": "vlm", "episode_wall_ms": t, "benchmark_success": success,
             "official": official, "planner_calls": 1, "model_usage": usage}
            for t, success, official, usage in [
                (10, True, {"reward": 1}, {"total_tokens": 20}),
                (100, False, {"reward": 0}, {"total_tokens": 40}),
                (5, False, None, {}),
            ]]
    stats = aggregate(rows)["vlm"]
    assert stats["attempted"] == 3 and stats["valid_evaluations"] == 2
    assert stats["success_rate_all_attempts"] == 1 / 3
    assert stats["episode_wall_ms_mean_all"] == 115 / 3
    assert stats["episode_wall_ms_mean_success_only"] == 10
    assert stats["runs_missing_vlm_token_usage"] == 1
