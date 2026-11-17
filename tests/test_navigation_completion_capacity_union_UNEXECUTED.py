"""Regression tests for navigation completion capacity union."""

import pytest
from test_actual_policy_v6_UNEXECUTED import capture, restore_selected, wire
from test_navigation_continuation_runner_draft import (
    MenuEnvironment,
    ObservedBackend,
    Planner,
    ScriptedPolicy,
)

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentTarget
from cua_jev.trace import JsonlTrace


class WholeGoalPlanner(Planner):
    supports_inband_whole_goal_recheck = True

    def __init__(self):
        super().__init__()
        self.full_requests = []
        self.last_request_provenance = None

    def plan(self, goal, snapshot, history, *, max_actions=6, whole_goal_recheck=False,
             whole_goal_inband_epoch=None):
        self.full_requests.append((max_actions, whole_goal_recheck, whole_goal_inband_epoch, snapshot))
        self.last_request_provenance = {
            "image_uploaded": snapshot.image is not None, "a11y_complete": snapshot.a11y_complete,
            "planning_observation": "full_multimodal_v1", "max_actions": max_actions,
        }
        if whole_goal_inband_epoch is not None:
            self.last_request_provenance["whole_goal_inband_review"] = {
                "version": "all_original_goal_review_v1", "input_epoch": whole_goal_inband_epoch,
            }
        if whole_goal_recheck:
            effect = AgentCondition("exists", AgentTarget(name="Arrival", app="Demo"))
            return AgentPlan("Original goal observed", (), done=True, reusable=True,
                             completion_conditions=(effect,))
        return super().plan(goal, snapshot, history, max_actions=max_actions)


class TrackedBackend(ObservedBackend):
    def __init__(self, environment):
        self.evidence_changes = []
        self.clear_observe_counts = []
        super().__init__(environment)

    def set_input_observation_evidence_enabled(self, enabled):
        previous = super().set_input_observation_evidence_enabled(enabled)
        self.evidence_changes.append((enabled, previous))
        return previous

    def clear_navigation_memory_sources(self):
        self.clear_observe_counts.append(self.fresh_observe_calls)
        return super().clear_navigation_memory_sources()


class ObservedPolicy(ScriptedPolicy):
    def __init__(self, labels, backend):
        super().__init__(labels)
        self.backend = backend
        self.read_counts = []

    def choose(self, observation, candidates):
        self.read_counts.append(self.backend.fresh_observe_calls)
        return super().choose(observation, candidates)


def exercise_combined(*, decline=False):
    environment = MenuEnvironment()
    backend = TrackedBackend(environment)
    planner = WholeGoalPlanner()
    policy = ObservedPolicy(("Parent", "Parent", "Parent" if decline else "Destination"), backend)
    trace = JsonlTrace()
    result = AgentRunner(
        backend, planner, policy=policy, trace=trace,
        config=AgentConfig(
            allow_actions=True, navigation_continuation=True,
            reuse_verified_post_snapshot=True, reuse_verified_pending_snapshot=True,
            whole_goal_recheck=True, reuse_whole_goal_planning_snapshot=True,
            inband_whole_goal_recheck=True, jev_network_recovery=True,
            jev_capacity_recovery=True, jev_capacity_recovery_policy="vlm_phase_v2",
        ),
    ).run("Reach the requested Destination")
    return result, environment, backend, planner, policy, trace


@pytest.mark.parametrize("decline", [False, True])
def test_single_evidence_enable_restore_and_real_fresh_navigation_preserve_whole_goal_checks(decline):
    result, environment, backend, planner, policy, trace = exercise_combined(decline=decline)
    try:
        assert environment.arrived and len(environment.inputs) == result.steps == 3
        assert backend.evidence_changes == [(True, False), (False, True)]
        assert backend._input_evidence_enabled is False
        steps = [e["payload"] for e in trace.events if e["kind"] == "agent_step"]
        assert [step["verified"] for step in steps] == [True, False, True]
        assert policy.read_counts[2] > backend.clear_observe_counts[0]
        assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
        assert result.whole_goal_inband_review_done == 0
        assert result.whole_goal_planning_snapshot_reuses == 1
        assert result.capacity_recovery_attempts == result.network_recovery_attempts == 0
        assert planner.full_requests[-1][1] is True
        assert backend._last_observed_snapshot is not planner.full_requests[-1][3]
        assert not any(e["kind"] in {"policy_failure", "agent_capacity_recovery",
                                     "agent_network_recovery"} for e in trace.events)
        rejected = [e for e in trace.events if e["kind"] == "decision_admission"
                    and not e["payload"]["approved"]]
        assert len(rejected) == int(decline)
        if decline:
            assert [request[0] for request in planner.full_requests] == [6, 1, 1]
            assert result.plan_reuses == 0
            decision_count = sum(e["kind"] == "decision" for e in trace.events)
            receipt_count = sum(e["kind"] == "receipt" for e in trace.events)
            assert decision_count == receipt_count + 1
        else:
            assert [request[0] for request in planner.full_requests] == [6, 1]
            assert result.plan_reuses == 1
    finally:
        backend.close()


def test_v6_request_keeps_complete_navigation_memory_full_pool_and_original_decision_options():
    _result, _environment, backend, _planner, policy, _trace = exercise_combined()
    try:
        observation, choices = policy.requests[2]
        assert observation.state["navigation_memory"]["known_repeat_used"] is True
        baseline, _, _, _ = capture("shared_fields_v4", observation, choices)
        encoded, _, exchange, _ = capture("shared_fields_v6", observation, choices)
        assert encoded["state"]["state"]["navigation_memory"] == observation.state["navigation_memory"]
        assert list(encoded["questions"]["action"]["criteria"]) == [choice.id for choice in choices]
        assert wire(restore_selected(encoded)) == wire(baseline)
        assert exchange["request_encoding"]["requested_layout"] == "shared_fields_v6"
        assert "Navigation memory is observed history" in encoded["questions"]["action"]["instructions"]
    finally:
        backend.close()


def test_same_navigation_flag_on_baseline_keeps_controller_and_admission_inactive():
    environment = MenuEnvironment()
    backend, planner, trace = TrackedBackend(environment), Planner(), JsonlTrace()
    result = AgentRunner(
        backend, planner, policy=ScriptedPolicy(["Destination"]), trace=trace,
        config=AgentConfig(mode="vlm", allow_actions=True, navigation_continuation=True,
                           reuse_verified_post_snapshot=False),
    ).run("Reach the requested Destination")
    try:
        assert environment.arrived and result.steps == 1
        assert backend.evidence_changes == []
        assert not any(e["kind"] in {"decision_admission", "agent_navigation_transition",
                                     "agent_navigation_continuation"} for e in trace.events)
    finally:
        backend.close()
