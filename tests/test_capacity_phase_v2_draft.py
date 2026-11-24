"""Regression tests for capacity phase v2."""

import json

import httpx
import pytest
from test_desktop_invalid_plan_recovery import Desktop, check_step, condition, plan
from test_jev_capacity_recovery import capacity, retry_plan, run_capacity
from test_jev_network_recovery import initial_plan

from cua_jev.agent_runner import AgentConfig
from cua_jev.models import ActionReceipt, Verification

POLICY = "vlm_phase_v2"


@pytest.fixture(autouse=True)
def no_poll_sleep(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _seconds: None)


def wrong_arrival():
    # A real positive Alpha input changes state; this future name does not exist.
    return plan(check_step("Alpha", postconditions=[condition("exists", "Unknown destination")]))


def capacity_events(trace):
    return [e["payload"] for e in trace.events if e["kind"] == "agent_capacity_recovery"]


@pytest.mark.parametrize("singleton", [True, False])
def test_positive_unknown_then_real_fresh_h1_can_verify_without_second_ticket(singleton):
    requests = []

    def handler(request):
        requests.append(request)
        return capacity(request)

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), wrong_arrival(), plan(check_step("Beta"))],
        handler=handler, jev_capacity_recovery_policy=POLICY,
        singleton_fast_path=singleton, reuse_verified_pending_snapshot=True,
    )
    assert result.success and desktop.alpha and desktop.beta
    assert planner.limits == [6, 1, 1] and len(planner.requests) == result.planner_calls == 3
    assert result.jev_calls == result.jev_request_attempts == len(requests) == 1
    assert result.capacity_recovery_attempts == result.capacity_recovery_verified == 1
    assert result.capacity_recovery_actions == result.steps == len(desktop.inputs) == 2
    assert result.network_recovery_attempts == result.jev_network_failures == 0
    stages = capacity_events(trace)
    assert [e["stage"] for e in stages] == [
        "requested", "fresh_observed", "continuation_requested", "fresh_observed", "verified",
    ]
    continuation = stages[2]
    assert continuation["verified"] is False and continuation["task_success"] is False
    assert continuation["shared_ticket_used"] is True and continuation["action_failures"] == 1
    final_steps = [e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step"]
    assert final_steps == [False, True]
    state = json.loads(planner.requests[2]["messages"][1]["content"][0]["text"])
    assert next(c for c in state["snapshot"]["controls"] if c["label"] == "Alpha")["checked"] is True
    assert any(c["type"] == "image_url" for c in planner.requests[2]["messages"][1]["content"])
    assert result.verified_pending_snapshot_reuses == result.recovery_planning_snapshot_reuses == 0
    failure = next(e["payload"] for e in trace.events if e["kind"] == "policy_failure")
    assert (failure["http_status"], failure["service_code"], failure["retryable"]) == (
        400, "max_tokens_exceeded", False,
    )
    assert failure["attempts"] == len(failure["retry_history"]) == 1


@pytest.mark.parametrize("settings", [{}, {"jev_capacity_recovery_policy": "vlm_once_v1"}])
def test_default_and_explicit_v1_keep_the_existing_one_action_stop(settings):
    result, trace, planner, desktop = run_capacity([initial_plan(), wrong_arrival()], **settings)
    assert AgentConfig().jev_capacity_recovery_policy == "vlm_once_v1"
    assert result.status == "stuck" and result.reason == "capacity recovery effect unconfirmed"
    assert planner.limits == [6, 1] and len(desktop.inputs) == 1
    assert result.capacity_recovery_verified == 0
    assert not any(e["stage"] == "continuation_requested" for e in capacity_events(trace))
    assert all("policy" not in e for e in capacity_events(trace))


def test_accepted_new_h1_does_not_reset_the_action_failure_streak():
    desktop = Desktop(opened=True)

    def no_effect(action, pause=0):
        desktop.inputs.append(action)
        return desktop._get_obs(), 0, False, {}

    desktop.step = no_effect
    result, trace, planner, _ = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], desktop=desktop,
        jev_capacity_recovery_policy=POLICY, max_failures=2,
    )
    assert result.status == "stuck" and planner.limits == [6, 1, 1]
    assert result.steps == result.capacity_recovery_actions == len(desktop.inputs) == 2
    assert result.capacity_recovery_attempts == 1 and result.capacity_recovery_verified == 0
    continuation = [e for e in capacity_events(trace) if e["stage"] == "continuation_requested"]
    assert len(continuation) == 1 and continuation[0]["action_failures"] == 1


@pytest.mark.parametrize("budget,status", [({"max_plans": 2}, "max_plans"), ({"max_steps": 1}, "max_steps")])
def test_continuation_requested_grants_no_plan_or_input_budget(budget, status):
    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), wrong_arrival(), plan(check_step("Beta"))],
        jev_capacity_recovery_policy=POLICY, **budget,
    )
    assert result.status == status and planner.limits == [6, 1]
    assert len(desktop.inputs) == result.capacity_recovery_actions == 1
    assert result.capacity_recovery_verified == 0 and result.jev_request_attempts == 1


@pytest.mark.parametrize("reply", ["bad model JSON", "http_error", initial_plan(), plan(check_step("Ghost"))])
def test_invalid_multi_action_or_missing_current_target_keeps_the_old_bounded_stop(reply):
    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), reply], jev_capacity_recovery_policy=POLICY,
    )
    assert result.status == "planner_error" and planner.limits == [6, 1]
    assert result.capacity_recovery_attempts == 1 and desktop.inputs == []
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0


def test_verified_zero_input_is_not_phase_progress_or_permission_for_more_plans():
    def setup(backend, _planner, _desktop):
        backend.execute = lambda candidate, observation_id, decision_id: ActionReceipt(
            observation_id, decision_id, candidate.id, candidate.channel,
            candidate.capability, True, 0, 0, {"environment_steps": 0},
        )
        backend.verify = lambda *_: Verification(True, "agent.effect", {"postconditions_met": True})

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], setup=setup,
        jev_capacity_recovery_policy=POLICY,
    )
    assert result.status == "stuck" and "no GUI input" in result.reason
    assert planner.limits == [6, 1] and desktop.inputs == []
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0
    assert not any(e["stage"] == "continuation_requested" for e in capacity_events(trace))


@pytest.mark.parametrize("later", ["capacity", "network"])
def test_later_jev_failure_cannot_get_a_second_shared_entrance(later):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) > 1 and later == "network":
            raise httpx.ConnectTimeout("private provider data", request=request)
        return capacity(request)

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), wrong_arrival(), plan(check_step("Beta")), retry_plan()],
        handler=handler, accept=False, singleton_fast_path=False,
        jev_capacity_recovery_policy=POLICY,
    )
    assert result.status == "error" and result.reason == "PolicyError"
    assert planner.limits == [6, 1, 1, 6] and len(desktop.inputs) == 2
    assert result.jev_calls == 2 and result.jev_request_attempts == len(requests)
    assert len(requests) == (2 if later == "capacity" else 4)
    assert result.capacity_recovery_attempts == result.capacity_recovery_verified == 1
    assert result.network_recovery_attempts == 0
    assert sum(e["stage"] == "requested" for e in capacity_events(trace)) == 1
    assert "private provider data" not in str(trace.events)


def test_next_real_fresh_observation_cannot_extend_the_original_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def setup(backend, _planner, desktop):
        original = backend.observe

        def observe():
            if desktop.inputs:
                clock[0] = 21.0
            return original()

        backend.observe = observe

    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), wrong_arrival(), plan(check_step("Beta"))], setup=setup,
        jev_capacity_recovery_policy=POLICY, timeout_s=20,
    )
    assert result.status == "timeout" and planner.limits == [6, 1]
    assert len(desktop.inputs) == 1 and result.capacity_recovery_verified == 0


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "phase", "vlm_phase_v4"])
def test_policy_selector_is_a_bounded_exact_string(value):
    with pytest.raises(ValueError, match="jev_capacity_recovery_policy"):
        AgentConfig(jev_capacity_recovery_policy=value)
