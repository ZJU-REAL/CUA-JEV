"""Bounded capacity rescue through real guards, with local HTTP and desktop mocks."""

import json
from dataclasses import replace

import httpx
import pytest
from test_desktop_invalid_plan_recovery import Desktop, check_step, condition, plan
from test_jev_network_recovery import initial_plan, run

from cua_jev.agent_runner import AgentConfig, _is_jev_capacity_failure
from cua_jev.models import ActionReceipt, Verification
from cua_jev.policy import JevFailure


def capacity(_request):
    return httpx.Response(400, json={"error": {"code": "max_tokens_exceeded"}})


def run_capacity(replies, *, handler=capacity, **settings):
    return run(replies, handler=handler, **{"jev_capacity_recovery": True, **settings})


def retry_plan():
    uncheck = {**check_step("Alpha"), "operation": "uncheck",
               "postconditions": [condition("unchecked", "Alpha")]}
    return plan(uncheck, check_step("Beta"))


def test_exact_capacity_failure_is_not_retried_and_gets_one_fresh_h1():
    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], max_steps=1,
        reuse_verified_pending_snapshot=True,
    )
    assert result.capacity_recovery_attempts == result.capacity_recovery_verified == 1
    assert result.capacity_recovery_actions == result.steps == len(desktop.inputs) == 1
    assert result.jev_calls == result.jev_request_attempts == result.jev_capacity_failures == 1
    assert result.network_recovery_attempts == result.jev_network_failures == 0
    assert planner.limits == [6, 1] and result.jev_wall_ms > 0
    failure = next(e["payload"] for e in trace.events if e["kind"] == "policy_failure")
    assert failure["http_status"] == 400 and failure["service_code"] == "max_tokens_exceeded"
    assert failure["retryable"] is False and failure["attempts"] == len(failure["retry_history"]) == 1
    stages = [e["payload"]["stage"] for e in trace.events if e["kind"] == "agent_capacity_recovery"]
    assert stages == ["requested", "fresh_observed", "verified"]
    decisions = [e["payload"]["model"] for e in trace.events if e["kind"] == "decision"]
    assert decisions == ["vlm-capacity-recovery"]
    assert result.verified_pending_snapshot_reuses == result.recovery_planning_snapshot_reuses == 0


def test_recovery_sees_real_state_and_invalidates_prior_input_evidence():
    requests, clear_sources = [], []

    def handler(request):
        requests.append(request)
        if len(requests) > 1:
            return capacity(request)
        ids = list(json.loads(request.content)["questions"]["action"]["criteria"])
        return httpx.Response(200, json={"answers": {"action": {
            "choice": ids[0], "probabilities": {key: float(key == ids[0]) for key in ids},
            "confidence": 1.0,
        }}})

    def setup(backend, _planner, _desktop):
        clear = backend.clear_input_observation_evidence

        def record_clear():
            clear_sources.append(backend._input_evidence)
            clear()

        backend.clear_input_observation_evidence = record_clear

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Beta"))], handler=handler, setup=setup,
        singleton_fast_path=False, reuse_verified_pending_snapshot=True,
    )
    assert result.success and desktop.alpha and desktop.beta
    assert len(requests) == 2 and planner.limits == [6, 1]
    assert clear_sources[0] is not None and clear_sources[0].input_steps > 0
    state = json.loads(planner.requests[1]["messages"][1]["content"][0]["text"])
    alpha = next(c for c in state["snapshot"]["controls"] if c["label"] == "Alpha")
    assert alpha["checked"] is True
    assert planner.history(1)[-1]["reason"] == "jev_capacity_recovery"
    assert result.capacity_recovery_verified == 1
    assert not any(e["kind"] == "agent_verified_pending_observation_reuse" for e in trace.events)


@pytest.mark.parametrize("first", ["capacity", "network"])
def test_capacity_and_network_share_one_episode_ticket(first):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        network = attempts > 1 if first == "capacity" else attempts <= 3
        if network:
            raise httpx.ConnectTimeout("not logged provider text", request=request)
        return capacity(request)

    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha")), retry_plan()], handler=handler,
        reuse_verified_pending_snapshot=True,
    )
    assert result.status == "error" and result.reason == "PolicyError"
    assert planner.limits == [6, 1, 6] and len(desktop.inputs) == result.steps == 1
    assert result.capacity_recovery_attempts + result.network_recovery_attempts == 1
    assert result.capacity_recovery_verified + result.network_recovery_verified == 1
    assert result.jev_request_attempts == attempts == 4


def test_second_capacity_refusal_stops_without_second_h1_or_resend():
    result, _trace, planner, desktop = run_capacity([
        initial_plan(), plan(check_step("Alpha")), retry_plan(),
    ])
    assert result.status == "error" and planner.limits == [6, 1, 6]
    assert result.jev_calls == result.jev_capacity_failures == result.jev_request_attempts == 2
    assert result.capacity_recovery_attempts == result.capacity_recovery_verified == len(desktop.inputs) == 1


@pytest.mark.parametrize("changes", [
    {"service_code": None}, {"service_code": "other"}, {"http_status": 401},
    {"http_status": "400"}, {"http_status": True}, {"retryable": True}, {"code": "timeout"},
])
def test_only_typed_capacity_refusal_is_eligible(changes):
    failure = JevFailure("observation", "a" * 64, "http_status", 1, False, 400, 1,
                         ((1, "http_status", 400, 1),), service_code="max_tokens_exceeded")
    assert _is_jev_capacity_failure(failure)
    assert not _is_jev_capacity_failure(replace(failure, **changes))
    assert not _is_jev_capacity_failure({"http_status": 400, "service_code": "max_tokens_exceeded"})


def test_default_off_keeps_public_shape_and_stops_at_original_400():
    assert AgentConfig().jev_capacity_recovery is False
    assert AgentConfig().reuse_verified_pending_snapshot is False
    result, trace, planner, desktop = run([initial_plan()], handler=capacity)
    assert result.status == "error" and planner.limits == [6] and desktop.inputs == []
    assert result.jev_request_attempts == 1
    assert not any("capacity" in key or key == "verified_pending_snapshot_reuses" for key in result.to_dict())
    assert not any(e["kind"] == "agent_capacity_recovery" for e in trace.events)


def test_capacity_recovery_does_not_grant_planning_budget():
    result, _trace, planner, desktop = run_capacity([initial_plan()], max_plans=1)
    assert result.status == "max_plans" and result.capacity_recovery_attempts == 0
    assert planner.limits == [6] and desktop.inputs == []


@pytest.mark.parametrize("reply", ["bad model JSON", "http_error", initial_plan()])
def test_bad_capacity_recovery_plan_spends_only_one_call(reply):
    result, _trace, planner, desktop = run_capacity([initial_plan(), reply])
    assert result.status == "planner_error" and result.capacity_recovery_attempts == 1
    assert planner.limits == [6, 1] and desktop.inputs == []


def test_verified_zero_input_cannot_complete_capacity_rescue():
    def setup(backend, _planner, _desktop):
        def execute(candidate, observation_id, decision_id):
            return ActionReceipt(observation_id, decision_id, candidate.id, candidate.channel,
                                 candidate.capability, True, 0, 0, {"environment_steps": 0})

        backend.execute = execute
        backend.verify = lambda *_: Verification(True, "agent.effect", {"postconditions_met": True})

    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], setup=setup,
    )
    assert result.status == "stuck" and "no GUI input" in result.reason
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0
    assert planner.limits == [6, 1] and desktop.inputs == []


def test_unconfirmed_positive_input_stops_without_pending_cache_or_new_recovery():
    desktop = Desktop(opened=True)

    def step(action, pause=0):
        desktop.inputs.append(action)
        return desktop._get_obs(), 0, False, {}

    desktop.step = step
    result, trace, planner, _ = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], desktop=desktop,
        reuse_verified_pending_snapshot=True,
    )
    assert result.status == "stuck" and result.capacity_recovery_actions == len(desktop.inputs) == 1
    assert result.capacity_recovery_verified == result.context_recoveries == 0
    assert planner.limits == [6, 1]
    assert not any(e["kind"] == "agent_verified_pending_observation_reuse" for e in trace.events)


def test_slow_actual_recovery_observation_cannot_extend_episode(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def setup(backend, _planner, _desktop):
        original = backend.observe
        reads = 0

        def observe():
            nonlocal reads
            reads += 1
            if reads == 3:
                clock[0] = 21
            return original()

        backend.observe = observe

    result, _trace, planner, desktop = run_capacity([initial_plan()], setup=setup, timeout_s=20)
    assert result.status == "timeout" and result.capacity_recovery_attempts == 1
    assert planner.limits == [6] and desktop.inputs == []


@pytest.mark.parametrize("field", ["jev_capacity_recovery", "reuse_verified_pending_snapshot"])
@pytest.mark.parametrize("value", [1, None, "true"])
def test_new_switches_require_exact_bool(field, value):
    with pytest.raises(ValueError, match=field):
        AgentConfig(**{field: value})
