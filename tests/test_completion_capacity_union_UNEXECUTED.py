"""Regression tests for completion capacity union."""

import hashlib
import time

import pytest
from test_capacity_phase_v2_draft import wrong_arrival
from test_desktop_invalid_plan_recovery import check_step, plan
from test_inband_whole_goal_review import EpochBackend, InbandPlanner, already_complete
from test_jev_capacity_recovery import run_capacity
from test_jev_network_recovery import initial_plan
from test_whole_goal_recheck import finished, run, stage

from cua_jev.agent_runner import _whole_goal_full_observation_proof

_run_with_original_defaults = run


def run(*args, **kwargs):
    kwargs.setdefault("reuse_whole_goal_planning_snapshot", True)
    kwargs.setdefault("inband_whole_goal_recheck", True)
    return _run_with_original_defaults(*args, **kwargs)


def marker(epoch=0):
    return {
        "image_uploaded": True,
        "a11y_complete": True,
        "whole_goal_inband_review": {
            "version": "all_original_goal_review_v1", "input_epoch": epoch,
        },
    }


def test_pre_call_repair_setter_failure_cannot_count_previous_same_epoch_marker():
    class SetterFailure(InbandPlanner):
        def __init__(self):
            super().__init__([finished()])
            self.last_request_provenance = marker()
            self.setter_calls = 0

        def set_prepare_repair_context(self, _context):
            self.setter_calls += 1
            if self.setter_calls == 1:
                raise ValueError("bounded fixture failure before planner invocation")

    planner = SetterFailure()
    result, _backend, planner, trace = run(
        [], backend=already_complete(), planner=planner, max_plans=3, mode="vlm",
    )
    assert result.status == "unverified"
    assert result.planner_calls == 2 and len(planner.calls) == 1
    assert result.whole_goal_inband_review_calls == result.whole_goal_inband_review_done == 1
    requested = [e["payload"] for e in trace.events
                 if e["kind"] == "agent_whole_goal_inband_review"
                 and e["payload"]["stage"] == "requested"]
    assert len(requested) == 1 and requested[0]["call"] == 2


def test_actual_invocation_reusing_old_marker_object_keeps_dedicated_fresh_path():
    class OldMarker(InbandPlanner):
        def __init__(self):
            super().__init__([finished()])
            self.last_request_provenance = marker()

        def plan(self, goal, snapshot, history, *, max_actions=6,
                 whole_goal_recheck=False, whole_goal_inband_epoch=None):
            if not self.calls:
                self.calls.append({"goal": goal, "limit": max_actions,
                                   "review": whole_goal_recheck,
                                   "inband_epoch": whole_goal_inband_epoch})
                # Deliberately do not mint a current-request provenance dict.
                return finished()
            return super().plan(goal, snapshot, history, max_actions=max_actions,
                                whole_goal_recheck=whole_goal_recheck,
                                whole_goal_inband_epoch=whole_goal_inband_epoch)

    result, _backend, planner, _trace = run(
        [], backend=already_complete(), planner=OldMarker(), reuse_verified_post_snapshot=True,
    )
    assert len(planner.calls) == 2 and planner.calls[1]["review"] is True
    assert result.whole_goal_inband_review_calls == result.whole_goal_inband_review_done == 0
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
    assert result.whole_goal_planning_snapshot_reuses == 1


def test_action_bearing_request_cannot_certify_after_state_but_completion_read_can_plan_review():
    class Recording(InbandPlanner):
        def __init__(self):
            super().__init__([stage("Name", "Ada"), finished()])
            self.frames = []

        def plan(self, goal, snapshot, history, **options):
            self.frames.append(snapshot)
            return super().plan(goal, snapshot, history, **options)

    backend = EpochBackend()
    backend.values = {"City": "Hangzhou"}
    result, backend, planner, trace = run(
        [], backend=backend, planner=Recording(), reuse_verified_post_snapshot=True,
    )
    assert result.status == "unverified" and backend.executed == ["Name"]
    assert result.steps == 1 and result.planner_calls == 2
    assert [call["review"] for call in planner.calls] == [False, True]
    assert result.whole_goal_inband_review_calls == 1 and result.whole_goal_inband_review_done == 0
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
    assert result.whole_goal_planning_snapshot_reuses == 1
    assert backend._last_observed_snapshot is not planner.frames[1]
    assert sum(e["kind"] == "agent_whole_goal_planning_observation_reuse"
               for e in trace.events) == 1


@pytest.mark.parametrize("input_count,observe_count", [(-1, 1), (False, 1), (0, 0), (0, True)])
def test_union_proof_rejects_malformed_counters_even_if_optional_callback_echoes_them(
    input_count, observe_count,
):
    backend = already_complete()
    frame = backend.observe()
    backend.environment_steps = input_count
    backend.fresh_observe_calls = observe_count

    def echo(_snapshot, *, expected=None):
        return (time.perf_counter(), input_count, observe_count,
                frame.fingerprint(), hashlib.sha256(frame.image.jpeg).hexdigest())

    backend.current_full_observation_proof = echo
    assert _whole_goal_full_observation_proof(backend, frame) is None


@pytest.mark.parametrize("bad_part", ["native", "jpeg", "extra_observe"])
def test_union_proof_keeps_direct_hash_binding_and_pure_query_boundary(bad_part):
    backend = already_complete()
    frame = backend.observe()
    original = backend.current_full_observation_proof

    def altered(snapshot, *, expected=None):
        proof = original(snapshot, expected=expected)
        assert proof is not None
        if bad_part == "extra_observe":
            backend.observe()
            return proof
        values = list(proof)
        values[3 if bad_part == "native" else 4] = "0" * 64
        return tuple(values)

    backend.current_full_observation_proof = altered
    assert _whole_goal_full_observation_proof(backend, frame) is None


@pytest.mark.parametrize("whole_goal", [False, True])
def test_capacity_phase_and_completion_options_share_original_single_ticket(whole_goal):
    result, trace, planner, desktop = run_capacity(
        [initial_plan(), wrong_arrival(), plan(check_step("Beta"))],
        jev_capacity_recovery_policy="vlm_phase_v2", whole_goal_recheck=whole_goal,
        reuse_verified_post_snapshot=True, reuse_verified_pending_snapshot=True,
    )
    assert result.success and desktop.alpha and desktop.beta
    assert result.capacity_recovery_attempts == result.capacity_recovery_verified == 1
    assert result.capacity_recovery_actions == result.steps == len(desktop.inputs) == 2
    assert result.jev_calls == result.jev_request_attempts == 1
    assert planner.limits == [6, 1, 1]
    events = [e["payload"] for e in trace.events if e["kind"] == "agent_capacity_recovery"]
    assert sum(e["stage"] == "requested" for e in events) == 1
    continuation = next(e for e in events if e["stage"] == "continuation_requested")
    assert continuation["action_failures"] == 1 and continuation["verified"] is False
    assert continuation["shared_ticket_used"] is True
    assert result.whole_goal_planning_snapshot_reuses == 0
