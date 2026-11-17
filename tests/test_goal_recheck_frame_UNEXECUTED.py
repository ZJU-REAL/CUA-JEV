"""Regression tests for goal recheck frame."""

from dataclasses import replace

import pytest
from test_whole_goal_recheck import InputBackend, SequencePlanner, finished, run, stage

from cua_jev import agent_runner
from cua_jev.agent_runner import (
    AgentConfig,
    AgentRunner,
    DirectPlanPolicy,
    _whole_goal_full_observation_proof,
)
from cua_jev.agent_types import AgentPlan
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace

_run_with_original_defaults = run


def run(*args, **kwargs):
    kwargs.setdefault("reuse_whole_goal_planning_snapshot", True)
    return _run_with_original_defaults(*args, **kwargs)


class ProofBackend(InputBackend):
    """Memory environment exposing the real backend's pure proof method."""

    def __init__(self, *, drift=None, now=None):
        super().__init__()
        self._closed = False
        self._last_observed_snapshot = self._last_observed_received_at = None
        self.environment_steps = self.fresh_observe_calls = 0
        self.offered = []
        self.frames = []
        self.drift, self.now = drift, now
        self.proof_checks = 0

    def observe(self):
        snapshot = super().observe()
        self.fresh_observe_calls += 1
        self._last_observed_snapshot = snapshot
        self._last_observed_received_at = agent_runner.time.perf_counter()
        self.frames.append(snapshot)
        return snapshot

    def execute(self, *args):
        self.environment_steps += 1
        return super().execute(*args)

    def current_full_observation_proof(self, snapshot, *, expected=None):
        if expected is None:
            self.offered.append(snapshot)
        else:
            self.proof_checks += 1
            if self.drift == "late_observe" and self.proof_checks == 3:
                self.observe()
        return OSWorldBackend.current_full_observation_proof(self, snapshot, expected=expected)

    def clear_input_observation_evidence(self):
        if self.drift == "observe":
            self.observe()
        elif self.drift == "input_counter":
            self.environment_steps += 1
        elif self.drift == "age":
            self.now[0] += 1.01
        elif self.drift == "wrong_object":
            self._last_observed_snapshot = replace(self._last_observed_snapshot)
        elif self.drift == "fingerprint":
            object.__setattr__(self._last_observed_snapshot, "text", "changed after completion")
        elif self.drift == "image_hash":
            snapshot = self._last_observed_snapshot
            object.__setattr__(snapshot, "image", replace(snapshot.image, jpeg=snapshot.image.jpeg + b"x"))
        elif self.drift == "partial":
            object.__setattr__(self._last_observed_snapshot, "a11y_complete", False)


class RecordingPlanner(SequencePlanner):
    def __init__(self, answers):
        super().__init__(answers)
        self.frames = []

    def plan(self, goal, snapshot, history, *, max_actions=6, whole_goal_recheck=False):
        self.frames.append(snapshot)
        return super().plan(goal, snapshot, history, max_actions=max_actions,
                            whole_goal_recheck=whole_goal_recheck)


@pytest.fixture
def now(monkeypatch):
    value = [10.0]
    # Both modules share the stdlib time module; no real wait is used.
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: value[0])
    return value


def episode(*, reuse=True, drift=None, now=None, generic=False):
    backend = InputBackend() if generic else ProofBackend(drift=drift, now=now)
    backend.values = {"City": "Hangzhou"}
    planner = RecordingPlanner([stage("Name", "Ada"), finished()])
    result, backend, planner, trace = run([], backend=backend, planner=planner,
                                          reuse_verified_post_snapshot=reuse)
    return result, backend, planner, trace


def test_one_real_completion_frame_supplies_only_next_whole_goal_request(now):
    result, backend, planner, trace = episode(now=now)
    old, old_backend, old_planner, _ = episode(now=now, reuse=False)
    assert result.planner_calls == old.planner_calls == len(planner.calls) == len(old_planner.calls) == 2
    assert result.steps == old.steps == 1 and backend.executed == old_backend.executed == ["Name"]
    assert result.plan_reuses == old.plan_reuses
    assert result.post_snapshot_reuses == old.post_snapshot_reuses == 0
    assert result.whole_goal_recheck_calls == old.whole_goal_recheck_calls == 1
    assert result.whole_goal_recheck_done == old.whole_goal_recheck_done == 1
    assert result.whole_goal_planning_snapshot_reuses == 1
    assert old_backend.observations == backend.observations + 1
    assert planner.frames[1] is backend.offered[0]
    # Reviewed empty-done still receives a different, real final fresh observation.
    assert backend.frames[-1] is not planner.frames[1]
    assert result.status == old.status == "unverified" and not result.success
    events = [e for e in trace.events if e["kind"] == "agent_whole_goal_planning_observation_reuse"]
    assert len(events) == 1 and events[0]["payload"]["planning_only"] is True


@pytest.mark.parametrize("drift", [
    "observe", "late_observe", "input_counter", "age", "wrong_object", "fingerprint", "image_hash", "partial",
])
def test_invalidated_or_incomplete_frame_falls_back_without_an_extra_model_call(now, drift):
    result, backend, planner, trace = episode(drift=drift, now=now)
    assert result.planner_calls == len(planner.calls) == 2 and backend.executed == ["Name"]
    assert result.whole_goal_recheck_done == 1 and result.steps == 1
    assert planner.frames[1] is not backend.offered[0]
    assert result.whole_goal_planning_snapshot_reuses == 0
    assert "whole_goal_planning_snapshot_reuses" not in result.to_dict()
    assert not any(e["kind"] == "agent_whole_goal_planning_observation_reuse" for e in trace.events)


def test_disabled_reuse_and_generic_backend_keep_fresh_path_and_public_zero_shape(now):
    for kwargs in ({"reuse": False}, {"generic": True}):
        result, _, planner, _ = episode(now=now, **kwargs)
        assert result.planner_calls == len(planner.calls) == 2
        assert "whole_goal_planning_snapshot_reuses" not in result.to_dict()


def test_unconditional_empty_done_is_not_a_successful_completion_offer(now):
    backend = ProofBackend(now=now)
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    planner = RecordingPlanner([AgentPlan("Unproven done", (), done=True), finished()])
    result, _, _, _ = run([], backend=backend, planner=planner)
    assert backend.offered == [] and result.whole_goal_planning_snapshot_reuses == 0
    assert result.planner_calls == 2 and result.whole_goal_recheck_done == 1


def test_normal_stage_planning_is_not_reused_and_model_return_reads_remain(now):
    backend = ProofBackend(now=now)
    planner = RecordingPlanner([stage("Name", "Ada", completion=False),
                                stage("City", "Hangzhou"), finished()])
    result, _, _, _ = run([], backend=backend, planner=planner)
    assert [item["review"] for item in planner.calls] == [False, False, True]
    assert result.planner_calls == 3 and backend.executed == ["Name", "City"]
    assert len(backend.offered) == result.whole_goal_planning_snapshot_reuses == 1
    assert planner.frames[2] is backend.offered[0]
    assert backend.frames[-1] is not planner.frames[2]


def test_default_whole_goal_off_does_not_offer_or_change_metric_shape(now):
    backend = ProofBackend(now=now)
    result, _, planner, _ = run([stage("Name", "Ada")], enabled=False, backend=backend)
    assert len(planner.calls) == 1 and backend.offered == []
    assert "whole_goal_planning_snapshot_reuses" not in result.to_dict()


def test_proof_is_pure_and_not_a_source_of_new_freshness(now):
    backend = ProofBackend(now=now)
    frame = backend.observe()
    reads, inputs = backend.fresh_observe_calls, backend.environment_steps
    proof = _whole_goal_full_observation_proof(backend, frame)
    assert proof is not None
    assert _whole_goal_full_observation_proof(backend, frame, expected=proof) == proof
    assert (backend.fresh_observe_calls, backend.environment_steps) == (reads, inputs)
    assert _whole_goal_full_observation_proof(backend, replace(frame), expected=proof) is None
    backend.observe()
    assert _whole_goal_full_observation_proof(backend, frame, expected=proof) is None


def test_real_osworld_callback_rejects_step_response_without_observe_identity(now):
    # This checks the real optional method without allocating any VM/environment.
    from types import SimpleNamespace

    backend = ProofBackend(now=now)
    observed = backend.observe()
    raw = SimpleNamespace(_closed=False, _last_observed_snapshot=observed,
                          _last_observed_received_at=now[0], environment_steps=0,
                          fresh_observe_calls=1)
    proof = OSWorldBackend.current_full_observation_proof(raw, observed)
    assert proof is not None
    assert OSWorldBackend.current_full_observation_proof(raw, replace(observed)) is None
    raw.environment_steps += 1
    assert OSWorldBackend.current_full_observation_proof(raw, observed, expected=proof) is None
    raw.environment_steps -= 1
    raw._closed = True
    assert OSWorldBackend.current_full_observation_proof(raw, observed, expected=proof) is None


def test_no_extra_plan_budget_is_created(now):
    backend = ProofBackend(now=now)
    backend.values = {"City": "Hangzhou"}
    result, _, planner, _ = run([stage("Name", "Ada")], backend=backend, max_plans=1)
    assert result.status == "max_plans" and len(planner.calls) == 1
    assert result.whole_goal_planning_snapshot_reuses == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_input_between_fresh_completion_read_and_gate_cannot_mint_current_proof(now, enabled):
    class RecordingProofBackend(ProofBackend):
        def __init__(self):
            super().__init__(now=now)
            self.proof_records = []

        def current_full_observation_proof(self, snapshot, *, expected=None):
            counters = self.environment_steps, self.fresh_observe_calls
            proof = super().current_full_observation_proof(snapshot, expected=expected)
            self.proof_records.append((snapshot, expected, counters, proof))
            return proof

    backend = RecordingProofBackend()
    backend.values = {"City": "Hangzhou"}
    planner = RecordingPlanner([stage("Name", "Ada"), finished()])
    injected = []
    completion_reads = []

    def progress(event):
        if event["event"] == "agent_plan_completion" and not injected:
            # The proof must have captured the counter before this callback.
            completion_reads.append((backend.frames[-1], backend.environment_steps,
                                     backend.fresh_observe_calls))
            injected.append(True)
            backend.environment_steps += 1

    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=JsonlTrace(),
                         progress=progress, config=AgentConfig(
                             allow_actions=True, whole_goal_recheck=True,
                             reuse_whole_goal_planning_snapshot=enabled,
                         )).run("Fill Name with Ada AND City with Hangzhou")
    assert result.planner_calls == 2 and injected == [True]
    assert result.whole_goal_planning_snapshot_reuses == 0
    assert len(completion_reads) == 1
    completion_frame, input_count, observe_count = completion_reads[0]
    assert planner.frames[1] is not completion_frame
    assert backend.frames[-1] is not planner.frames[1]
    assert result.whole_goal_recheck_done == 1 and not result.success
    if not enabled:
        assert backend.offered == backend.proof_records == []
        return
    assert planner.frames[1] is not backend.offered[0]
    assert len(backend.offered) == len(completion_reads) == 1
    assert backend.offered[0] is completion_frame
    assert len(backend.proof_records) == 2
    initial, gate = backend.proof_records
    assert initial[0] is gate[0] is completion_frame
    assert initial[1] is None and initial[2] == (input_count, observe_count)
    assert initial[3] is not None and initial[3][1:3] == (input_count, observe_count)
    assert gate[1] == initial[3] and gate[2] == (input_count + 1, observe_count)
    assert gate[3] is None  # Same native/image state cannot hide intervening input.


def test_expiry_during_late_proof_does_not_call_model_or_change_recovery_budget(now):
    class Expiring(ProofBackend):
        def current_full_observation_proof(self, snapshot, *, expected=None):
            if expected is not None and self.proof_checks == 2:
                now[0] += 3
            return super().current_full_observation_proof(snapshot, expected=expected)

    # Gate validation is check1, next-loop selection check2, late API validation check3.
    backend = Expiring(now=now)
    backend.values = {"City": "Hangzhou"}
    result, _, planner, _ = run([stage("Name", "Ada")], backend=backend, timeout_s=2)
    assert result.status == "timeout" and len(planner.calls) == 1
    assert result.whole_goal_planning_snapshot_reuses == 0
    assert result.network_recovery_attempts == result.capacity_recovery_attempts == 0
