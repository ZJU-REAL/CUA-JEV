"""Actual pending full-frame handoff; no model, VM, API or input guard bypass."""
import json
import socket
from dataclasses import replace

import httpx
import pytest
from test_osworld_navigation_context_signal import _execute, _observation
from test_semantic_tools_v57 import packet
from test_verified_recovery_planning_snapshot import clock as clock

import cua_jev.agent_runner as runner
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_navigation_policy import enable_full_legal_navigation_gestures
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import RulePolicy
from cua_jev.trace import JsonlTrace


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("Only local MockTransport is permitted")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def bound_frame():
    backend, env, candidate, receipt = _execute(deferred=True)
    captured = runner._capture_unexplained_pending_planning_frame(backend, candidate, receipt, 1)
    assert captured is not None
    signal = backend.take_unexplained_navigation_context(candidate, receipt)
    frame = runner._bind_unexplained_planning_signal(captured, signal)
    assert frame is not None and signal["arrival_evidence"][0]["outcome"] == "unknown"
    return backend, env, frame, signal


def test_actual_pending_frame_remains_planning_only_after_original_evidence_consumption(clock):
    backend, env, frame, _signal = bound_frame()
    try:
        before = env.reads, len(env.inputs), backend.fresh_observe_calls
        assert backend._input_evidence is None
        assert frame.receipt.output["postcondition_evidence"][0]["outcome"] == "unknown"
        assert runner._unexplained_planning_frame_is_current(backend, frame, 1)
        assert (env.reads, len(env.inputs), backend.fresh_observe_calls) == before
        assert frame.receipt.output["environment_steps"] == 1
    finally:
        backend.close()


@pytest.mark.parametrize("invalid", ["expired", "future", "observe", "input_counter", "epoch",
                                    "receipt_copy", "receipt_data", "candidate", "partial", "image",
                                    "owner", "verification"])
def test_pending_planning_frame_declines_every_source_drift_without_new_io(clock, invalid):
    backend, env, frame, _signal = bound_frame()
    epoch = 1
    try:
        if invalid == "expired":
            clock.now = frame.proof[0] + 1.001
        elif invalid == "future":
            clock.now = frame.proof[0] - 0.001
        elif invalid == "observe":
            backend.observe()
        elif invalid == "input_counter":
            backend.environment_steps += 1
        elif invalid == "epoch":
            epoch += 1
        elif invalid == "receipt_copy":
            frame = replace(frame, receipt=replace(frame.receipt))
        elif invalid == "receipt_data":
            frame.receipt.output["environment_steps"] = 0
        elif invalid == "candidate":
            frame = replace(frame, candidate=replace(frame.candidate, description="altered"))
        elif invalid == "partial":
            frame = replace(frame, snapshot=replace(frame.snapshot, a11y_complete=False))
        elif invalid == "image":
            frame = replace(frame, snapshot=replace(frame.snapshot, image=None))
        elif invalid == "owner":
            frame = replace(frame, owner_app="Other")
        elif invalid == "verification":
            old = backend._verified[frame.receipt.receipt_id]
            backend._verified[frame.receipt.receipt_id] = (*old[:2], replace(old[2], passed=False))
        before = env.reads, len(env.inputs), backend.fresh_observe_calls
        assert not runner._unexplained_planning_frame_is_current(backend, frame, epoch)
        assert (env.reads, len(env.inputs), backend.fresh_observe_calls) == before
    finally:
        backend.close()


@pytest.mark.parametrize("invalid", ["source", "receipt", "candidate", "fingerprint", "arrival", "authority"])
def test_original_detector_signal_cannot_be_replaced_by_a_synthetic_permission(clock, invalid):
    backend, _env, frame, signal = bound_frame()
    try:
        changed = dict(signal)
        if invalid == "source":
            changed["evidence_source"] = "sdk_after_input"
        elif invalid == "receipt":
            changed["receipt_id"] = "other"
        elif invalid == "candidate":
            changed["candidate_id"] = "other"
        elif invalid == "fingerprint":
            changed["evidence_after_fingerprint"] = "0" * 64
        elif invalid == "arrival":
            changed["arrival_evidence"] = [{"outcome": "true"}]
        elif invalid == "authority":
            changed["input_authority"] = True
        assert runner._bind_unexplained_planning_signal(frame, changed) is None
    finally:
        backend.close()


def test_sdk_response_is_not_promoted_to_the_pending_only_first_handoff(clock):
    backend, env, candidate, receipt = _execute(deferred=False)
    try:
        before = env.reads, len(env.inputs)
        assert runner._capture_unexplained_pending_planning_frame(backend, candidate, receipt, 1) is None
        assert (env.reads, len(env.inputs)) == before
    finally:
        backend.close()


class Environment:
    def __init__(self, *, deferred):
        self.inputs, self.reads, self.deferred = [], 0, deferred

    def _get_obs(self):
        self.reads += 1
        return _observation(len(self.inputs) == 1)

    def step(self, action, pause=0):
        self.inputs.append(action)
        opened = len(self.inputs) == 1 and not self.deferred
        return _observation(opened), 0, False, {}


class Backend(OSWorldBackend):
    def __init__(self, env):
        super().__init__(env, pause=0)
        self.execute_fresh_reads = []

    def execute(self, *args):
        before = self.fresh_observe_calls
        receipt = super().execute(*args)
        self.execute_fresh_reads.append(self.fresh_observe_calls - before)
        return receipt


class Planner:
    profile = "desktop"
    desktop_contract = "property_intent_v52"
    desktop_output_format = "semantic_tools_v64"
    desktop_planning_observation = "full_multimodal_v1"

    def __init__(self, *, provenance=True):
        self.requests, self.provenance = [], provenance
        self.last_request_provenance = None

    def plan(self, goal, snapshot, history, *, max_actions):
        self.requests.append((max_actions, snapshot, list(history)))
        self.last_request_provenance = {"planning_observation": "full_multimodal_v1",
            "image_uploaded": True, "a11y_complete": True,
            "max_actions": max_actions if self.provenance else 6}
        if len(self.requests) == 1:
            return enable_full_legal_navigation_gestures(AgentPlan("Reach Options", (AgentStep(
                "reach", "click", target=AgentTarget(app="Demo"), navigation=True,
                native_navigation_gestures=True, label="Reach Options", recovery="observe",
                postconditions=(AgentCondition("exists", AgentTarget(name="Options", role="frame")),),
            ),), reusable=True))
        if len(self.requests) == 2:
            return AgentPlan("Accept current Options", (AgentStep(
                "accept", "click", target=AgentTarget(name="Accept", role="button", app="Demo"),
                postconditions=(AgentCondition("absent", AgentTarget(name="Options", role="dialog")),),
            ),), reusable=True)
        return AgentPlan("Original goal now back at Main", (), done=True,
                         completion_conditions=(AgentCondition("window_contains", value="Main"),))


def exercise(*, enabled, deferred, steps=6, provenance=True, progress=None, post_reuse=True):
    env, trace = Environment(deferred=deferred), JsonlTrace()
    backend, planner = Backend(env), Planner(provenance=provenance)
    result = runner.AgentRunner(backend, planner, policy=RulePolicy(), trace=trace,
        progress=progress,
        config=runner.AgentConfig(allow_actions=True, max_steps=steps, max_plans=3, timeout_s=300,
            plan_ttl_s=300, reuse_verified_post_snapshot=post_reuse, reuse_verified_pending_snapshot=True,
            reuse_unexplained_context_planning_observation=enabled)).run("Use Options then return to Main")
    return result, env, backend, planner, trace.events


def test_first_pending_context_handoff_saves_only_one_planning_read_and_keeps_input_preflights(clock):
    enabled = exercise(enabled=True, deferred=True, steps=2)
    disabled = exercise(enabled=False, deferred=True, steps=2)
    try:
        result, env, backend, planner, events = enabled
        assert result.planner_calls == disabled[0].planner_calls == 2
        assert len(env.inputs) == len(disabled[1].inputs) == 2
        assert backend.execute_fresh_reads == disabled[2].execute_fresh_reads == [1, 1]
        assert result.runner_observe_calls == disabled[0].runner_observe_calls - 1
        assert result.fresh_observe_calls == disabled[0].fresh_observe_calls - 1
        assert [row[0] for row in planner.requests] == [6, 1]
        assert sum(event["kind"] == "agent_unexplained_context_planning_observation_reuse"
                   for event in events) == 1
        assert result.recovery_planning_snapshot_reuses == 0  # Second ticket never consumed at max_steps.
        assert any(event["kind"] == "agent_unexplained_navigation_context"
                   and event["payload"]["arrival_evidence"][0]["outcome"] == "unknown" for event in events)
    finally:
        enabled[2].close()
        disabled[2].close()


def test_verified_recovery_reuses_existing_ticket_without_early_phase_flag_reset(clock):
    enabled = exercise(enabled=True, deferred=False)
    disabled = exercise(enabled=False, deferred=False)
    try:
        result, env, backend, planner, events = enabled
        assert result.planner_calls == disabled[0].planner_calls == 3
        assert [row[0] for row in planner.requests] == [6, 1, 6]
        assert len(env.inputs) == len(disabled[1].inputs) == 2
        assert backend.execute_fresh_reads == disabled[2].execute_fresh_reads == [1, 1]
        assert result.runner_observe_calls == disabled[0].runner_observe_calls - 1
        assert result.recovery_planning_snapshot_reuses == 1
        assert not any(event["kind"] == "agent_unexplained_context_planning_observation_reuse"
                       for event in events)  # First frame came from SDK, not pending.
        assert result.status == "unverified" and not result.success
        assert any(event["kind"] == "agent_plan_completion" for event in events)
    finally:
        enabled[2].close()
        disabled[2].close()


def test_context_callback_age_expiry_falls_back_to_original_fresh_planning(clock):
    def progress(event):
        if event["event"] == "agent_unexplained_navigation_context":
            clock.advance(1.001)
    result, _env, backend, _planner, events = exercise(
        enabled=True, deferred=True, steps=2, progress=progress,
    )
    try:
        assert result.planner_calls == 2
        assert not any(event["kind"] == "agent_unexplained_context_planning_observation_reuse"
                       for event in events)
        assert backend.execute_fresh_reads == [1, 1]
    finally:
        backend.close()


def test_second_ticket_still_requires_original_h1_full_image_request_provenance(clock):
    result, _env, backend, _planner, _events = exercise(enabled=True, deferred=False, provenance=False)
    try:
        assert result.recovery_planning_snapshot_reuses == 0
    finally:
        backend.close()


@pytest.mark.parametrize("deferred", [False, True])
def test_master_post_snapshot_reuse_off_disables_both_new_paths(clock, deferred):
    enabled = exercise(enabled=True, deferred=deferred, post_reuse=False)
    disabled = exercise(enabled=False, deferred=deferred, post_reuse=False)
    try:
        assert enabled[0].runner_observe_calls == disabled[0].runner_observe_calls
        assert enabled[0].fresh_observe_calls == disabled[0].fresh_observe_calls
        assert enabled[0].recovery_planning_snapshot_reuses == 0
        assert not any(row["kind"] == "agent_unexplained_context_planning_observation_reuse"
                       for row in enabled[4])
        assert enabled[2].execute_fresh_reads == disabled[2].execute_fresh_reads == [1, 1]
    finally:
        enabled[2].close()
        disabled[2].close()


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_explicit_optin_requires_strict_boolean(invalid):
    with pytest.raises(ValueError, match="reuse_unexplained_context_planning_observation"):
        runner.AgentConfig(reuse_unexplained_context_planning_observation=invalid)


def wire_exercise(*, enabled, completion=False, second_ticket=False):
    env, trace, bodies = Environment(deferred=not second_ticket), JsonlTrace(), []
    backend = Backend(env)
    first = {"label": "Reach Options", "scope": {"app": "Demo"}, "completion": "stage",
             "arrival": [{"kind": "exists", "target": {"name": "Options", "role": "frame"}}]}
    second = {"summary": "Accept observed Options", "id": "accept", "done": False,
              "operation": "click", "target": {"name": "Accept", "role": "button", "app": "Demo"},
              "postconditions": [{"kind": "absent", "target": {"name": "Options", "role": "dialog"}}]}
    replies = [packet(first, "declare_navigation_stage", content=None),
               packet({"label": "Current Options", "conditions": [
                   {"kind": "window_contains", "value": "Options"}]} if completion else second,
                   "declare_completion" if completion else "declare_semantic_current_action", content=None)]
    if second_ticket:
        done = packet({"label": "Main is current", "conditions": [
            {"kind": "window_contains", "value": "Main"}]}, "declare_completion", content=None)
        replies.extend([done, done])

    def answer(request):
        assert request.url.host == "offline.invalid"
        assert request.extensions["timeout"]["read"] == 90
        bodies.append(json.loads(request.content))
        assert len(bodies) <= len(replies)
        return httpx.Response(200, json=replies[len(bodies) - 1])

    with httpx.Client(transport=httpx.MockTransport(answer), trust_env=False) as client:
        planner = MultimodalPlanner("https://offline.invalid", "same-model", client=client, api_key=None,
            profile="desktop", desktop_contract="property_intent_v52",
            desktop_output_format="semantic_tools_v64", desktop_normal_stage_policy="atomic_stage_only",
            desktop_planning_observation="full_multimodal_v1",
            enable_thinking=True, thinking_budget=None, request_timeout_s=90)
        result = runner.AgentRunner(backend, planner, policy=RulePolicy(), trace=trace,
            config=runner.AgentConfig(allow_actions=True, max_steps=6 if completion or second_ticket else 2,
                max_plans=4 if second_ticket else 2,
                timeout_s=300, plan_ttl_s=300, reuse_verified_post_snapshot=True,
                reuse_verified_pending_snapshot=True, whole_goal_recheck=True,
                inband_whole_goal_recheck=True, reuse_whole_goal_planning_snapshot=True,
                reuse_unexplained_context_planning_observation=enabled)).run(
                    "Use Options then return to Main")
        provenance = dict(planner.last_request_provenance)
    return result, env, backend, trace.events, bodies, provenance


@pytest.mark.parametrize("late_age", [False, True])
def test_actual_mock_requests_keep_complete_goal_review_marker_and_budgets(clock, monkeypatch, late_age):
    original = runner._unexplained_planning_frame_is_current
    admitted = []

    def after_admission(*args):
        accepted = original(*args)
        if accepted and not admitted:
            admitted.append(True)
            if late_age:
                clock.advance(1.001)
        return accepted

    monkeypatch.setattr(runner, "_unexplained_planning_frame_is_current", after_admission)
    enabled = wire_exercise(enabled=True)
    disabled = wire_exercise(enabled=False)
    try:
        on, off = enabled[0], disabled[0]
        assert on.planner_calls == off.planner_calls == 2
        assert enabled[2].execute_fresh_reads == disabled[2].execute_fresh_reads == [1, 1]
        assert len(enabled[1].inputs) == len(disabled[1].inputs) == 2
        assert on.runner_observe_calls == off.runner_observe_calls - (0 if late_age else 1)
        assert enabled[5]["whole_goal_inband_review"] == disabled[5]["whole_goal_inband_review"]
        assert enabled[5]["whole_goal_inband_review"]["input_epoch"] == 1
        assert on.whole_goal_inband_review_calls == off.whole_goal_inband_review_calls == 2
        assert enabled[4][1]["messages"][0]["content"] == disabled[4][1]["messages"][0]["content"]
        assert enabled[4][1]["tools"] == disabled[4][1]["tools"]
        for bodies in (enabled[4], disabled[4]):
            assert all(body["enable_thinking"] is True and "thinking_budget" not in body for body in bodies)
            assert all(json.loads(body["messages"][1]["content"][0]["text"])["goal"]
                       == "Use Options then return to Main" for body in bodies)
    finally:
        enabled[2].close()
        disabled[2].close()


def test_planning_only_cached_empty_done_cannot_receive_inband_completion_credit(clock):
    result, env, backend, events, bodies, provenance = wire_exercise(enabled=True, completion=True)
    try:
        assert result.planner_calls == len(bodies) == 2 and len(env.inputs) == 1
        assert provenance["whole_goal_inband_review"]["input_epoch"] == 1
        assert result.status == "max_plans" and not result.success
        assert any(event["kind"] == "agent_whole_goal_recheck"
                   and event["payload"]["stage"] == "required" for event in events)
        assert result.whole_goal_recheck_done == 0
        assert not any(event["kind"] == "agent_whole_goal_inband_review"
                       and event["payload"]["stage"] == "empty_done_fresh_checked" for event in events)
        assert any(event["kind"] == "agent_plan_completion" for event in events)
    finally:
        backend.close()


def test_new_unexplained_recovery_ticket_cannot_grant_existing_inband_done_credit(clock):
    enabled = wire_exercise(enabled=True, second_ticket=True)
    disabled = wire_exercise(enabled=False, second_ticket=True)
    try:
        on, off = enabled[0], disabled[0]
        assert on.recovery_planning_snapshot_reuses == 1 and off.recovery_planning_snapshot_reuses == 0
        assert enabled[2].execute_fresh_reads == disabled[2].execute_fresh_reads == [1, 1]
        assert len(enabled[1].inputs) == len(disabled[1].inputs) == 2
        assert on.planner_calls == 4 and off.planner_calls == 3
        assert on.whole_goal_inband_review_calls == off.whole_goal_inband_review_calls == 3
        assert on.whole_goal_recheck_done == 1
        assert any(row["kind"] == "agent_whole_goal_recheck"
                   and row["payload"]["stage"] == "requested" for row in enabled[3])
        assert not any(row["kind"] == "agent_whole_goal_inband_review"
                       and row["payload"]["stage"] == "empty_done_fresh_checked" for row in enabled[3])
        assert any(row["kind"] == "agent_whole_goal_inband_review"
                   and row["payload"]["stage"] == "empty_done_fresh_checked" for row in disabled[3])
        # The new source is not used by either final independent empty-done check.
        assert on.runner_observe_calls >= off.runner_observe_calls
        assert not on.success and not off.success
    finally:
        enabled[2].close()
        disabled[2].close()
