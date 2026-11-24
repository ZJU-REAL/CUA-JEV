"""Opt-in capacity H1 bad-plan recovery, using only local transport/GUI mocks."""

from dataclasses import replace

import pytest
from test_capacity_phase_v2_draft import capacity_events, wrong_arrival
from test_desktop_invalid_plan_recovery import check_step, plan
from test_jev_capacity_recovery import retry_plan, run_capacity
from test_jev_network_recovery import initial_plan
from test_whole_goal_recheck import (
    InputBackend,
    SequencePlanner,
    finished,
    stage,
)
from test_whole_goal_recheck import (
    run as run_goal,
)

from cua_jev.agent_runner import AgentConfig, DirectPlanPolicy
from cua_jev.models import ActionReceipt, Verification
from cua_jev.policy import JevFailure, JevRequestError

POLICY = "vlm_phase_v3"


@pytest.fixture(autouse=True)
def no_poll_sleep(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _seconds: None)


def retries(trace):
    return [e for e in capacity_events(trace) if e["stage"] == "invalid_plan_retry_requested"]


def test_positive_unknown_then_observed_validation_rejection_gets_real_new_h1():
    result, trace, planner, desktop = run_capacity(
        [initial_plan(), wrong_arrival(), plan(check_step("Ghost")), plan(check_step("Beta"))],
        jev_capacity_recovery_policy=POLICY, reuse_verified_pending_snapshot=True,
    )
    assert result.success and desktop.alpha and desktop.beta
    assert planner.limits == [6, 1, 1, 1]
    assert result.jev_calls == result.jev_request_attempts == result.capacity_recovery_attempts == 1
    assert result.capacity_recovery_actions == 2 and result.capacity_recovery_verified == 1
    retry = retries(trace)[0]
    assert retry["planning_failures"] == retry["action_failures"] == 1
    assert retry["cause"] == "planning_validation" and not retry["accepted_plan_retired"]
    assert retry["shared_ticket_used"] and not retry["verified"]
    assert result.plan_reuses == 0  # semantic_steps is a different original counter, not strict reuse.
    stages = [e["stage"] for e in capacity_events(trace)]
    assert stages.count("requested") == 1 and stages.count("fresh_observed") == 3


def test_rejected_plans_spend_original_consecutive_planning_limit():
    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Ghost"))], jev_capacity_recovery_policy=POLICY,
        max_failures=3,
    )
    assert result.status == "planner_error" and result.reason == "consecutive planning failures"
    assert planner.limits == [6, 1, 1, 1] and desktop.inputs == []
    assert [e["planning_failures"] for e in retries(trace)] == [1, 2]
    assert result.capacity_recovery_attempts == result.jev_request_attempts == 1
    assert result.capacity_recovery_verified == result.capacity_recovery_actions == 0


def test_transport_runtime_error_is_not_a_new_vlm_retry_policy():
    result, trace, planner, desktop = run_capacity(
        [initial_plan(), "http_error"], jev_capacity_recovery_policy=POLICY,
    )
    assert result.status == "planner_error" and result.reason == "capacity recovery planning failed"
    assert planner.limits == [6, 1] and desktop.inputs == [] and not retries(trace)


@pytest.mark.parametrize("unusable", ["zero", "ambiguous"])
def test_unusable_accepted_h1_charges_both_failures_and_reobserves(unusable):
    fresh_before_calls = []

    def setup(backend, planner, _desktop):
        compile_original, plan_original = backend.compile, planner.plan

        def compile(*args):
            choices = compile_original(*args)
            if planner.calls == 2:
                return () if unusable == "zero" else (*choices, replace(choices[0], id="second-choice"))
            return choices

        def planning(*args, **kwargs):
            fresh_before_calls.append(backend.fresh_observe_calls)
            answer = plan_original(*args, **kwargs)
            return replace(answer, reusable=True) if unusable == "zero" and planner.calls == 2 else answer

        backend.compile, planner.plan = compile, planning

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha", recovery="observe")), plan(check_step("Alpha"))],
        setup=setup, jev_capacity_recovery_policy=POLICY, max_steps=1,
    )
    assert planner.limits == [6, 1, 1] and len(desktop.inputs) == result.steps == 1
    assert fresh_before_calls[2] > fresh_before_calls[1]
    assert result.jev_request_attempts == result.capacity_recovery_attempts == 1
    assert result.capacity_recovery_verified == 1
    retry = retries(trace)[0]
    assert retry["planning_failures"] == retry["action_failures"] == 1
    assert retry["accepted_plan_retired"] and retry["input_sent"] is False
    if unusable == "zero":
        assert result.recovery_observations == 2


@pytest.mark.parametrize("budget,status", [({"max_plans": 2}, "max_plans"),
                                            ({"max_failures": 1}, "stuck")])
def test_ambiguous_plan_gets_no_extra_original_budget(budget, status):
    def setup(backend, planner, _desktop):
        original = backend.compile

        def compile(*args):
            choices = original(*args)
            return (*choices, replace(choices[0], id="other")) if planner.calls == 2 else choices

        backend.compile = compile

    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], setup=setup,
        jev_capacity_recovery_policy=POLICY, **budget,
    )
    assert result.status == status and planner.limits == [6, 1] and desktop.inputs == []
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0


def test_bad_new_response_does_not_renew_plan_clock_or_add_total_phase_ttl(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def setup(_backend, planner, _desktop):
        original = planner.plan

        def planning(*args, **kwargs):
            if planner.calls == 1:
                clock[0] = 5.0
            return original(*args, **kwargs)

        planner.plan = planning

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Ghost")), plan(check_step("Alpha"))],
        setup=setup, jev_capacity_recovery_policy=POLICY, plan_ttl_s=1, max_steps=1,
    )
    assert planner.limits == [6, 1, 1] and len(desktop.inputs) == 1
    assert retries(trace)[0]["previous_plan_at_unchanged"] == 0.0
    assert result.capacity_recovery_verified == 1


def test_expired_accepted_unavailable_h1_cannot_be_renewed_as_retry(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def setup(backend, planner, _desktop):
        original = backend.compile

        def compile(*args):
            if planner.calls == 2:
                clock[0] = 2.0
                return ()
            return original(*args)

        backend.compile = compile

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], setup=setup,
        jev_capacity_recovery_policy=POLICY, plan_ttl_s=1,
    )
    assert result.status == "stuck" and planner.limits == [6, 1]
    assert desktop.inputs == [] and not retries(trace)


def test_real_fresh_retry_read_cannot_extend_global_episode_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def setup(backend, planner, _desktop):
        original = backend.observe

        def observe():
            if planner.calls == 2:
                clock[0] = 421.0
            return original()

        backend.observe = observe

    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Ghost")), plan(check_step("Alpha"))], setup=setup,
        jev_capacity_recovery_policy=POLICY, timeout_s=420,
    )
    assert result.status == "timeout" and planner.limits == [6, 1] and desktop.inputs == []


def test_valid_but_unavailable_replan_does_not_reset_planning_failures():
    def setup(backend, planner, _desktop):
        original = backend.compile
        backend.compile = lambda *args: () if planner.calls == 3 else original(*args)

    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Ghost")), plan(check_step("Alpha")), plan(check_step("Ghost"))],
        setup=setup, jev_capacity_recovery_policy=POLICY, max_failures=3,
    )
    assert result.reason == "consecutive planning failures" and planner.limits == [6, 1, 1, 1]
    assert [e["planning_failures"] for e in retries(trace)] == [1, 2] and desktop.inputs == []


def test_spent_ticket_survives_bad_plan_retry_and_verified_input():
    result, trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Ghost")), plan(check_step("Alpha")), retry_plan()],
        jev_capacity_recovery_policy=POLICY, singleton_fast_path=False, accept=False,
    )
    assert result.status == "error" and result.reason == "PolicyError"
    assert planner.limits == [6, 1, 1, 6] and len(desktop.inputs) == 1
    assert result.capacity_recovery_attempts == result.capacity_recovery_verified == 1
    assert result.jev_calls == result.jev_request_attempts == result.jev_capacity_failures == 2
    assert sum(e["stage"] == "requested" for e in capacity_events(trace)) == 1


class CapacityOnce:
    def __init__(self):
        self.calls = 0

    def choose(self, observation, candidates):
        self.calls += 1
        if self.calls == 1:
            failure = JevFailure(observation.observation_id, "a" * 64, "http_status", 1,
                                 False, 400, 1, ((1, "http_status", 400, 1),),
                                 service_code="max_tokens_exceeded")
            raise JevRequestError("fixture capacity refusal", failure)
        return DirectPlanPolicy().choose(observation, candidates)


def goal_run(answers, backend):
    planner = SequencePlanner(answers)
    planner.profile = "desktop"
    return run_goal([], planner=planner, backend=backend, policy=CapacityOnce(),
                    singleton_fast_path=False, jev_capacity_recovery=True,
                    jev_capacity_recovery_policy=POLICY)


def test_whole_goal_review_stays_h1_until_verified_positive_input_then_h6():
    backend = InputBackend()
    backend.values = {"Name": "Ada"}
    partial = replace(finished(), completion_conditions=(finished().completion_conditions[0],))
    result, backend, planner, _trace = goal_run(
        [stage("Name", "Other"), partial, stage("City", "Hangzhou"), finished()], backend,
    )
    assert backend.executed == ["City"]
    assert [c["limit"] for c in planner.calls] == [6, 1, 1, 1]
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 1
    assert result.whole_goal_recheck_done == 1


def test_legal_empty_done_fresh_goal_review_stops_without_forced_input_or_credit():
    backend = InputBackend()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    result, backend, planner, _trace = goal_run([stage("Name", "Other"), finished(), finished()], backend)
    assert backend.executed == [] and [c["limit"] for c in planner.calls] == [6, 1, 1]
    assert result.status == "unverified" and result.whole_goal_recheck_done == 1
    assert result.capacity_recovery_attempts == 1
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0


def test_bad_whole_goal_review_can_replan_h1_and_still_requires_fresh_conditions():
    def rejected():
        raise ValueError("invalid current target")

    backend = InputBackend()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    result, backend, planner, trace = goal_run(
        [stage("Name", "Other"), finished(), rejected, finished()], backend,
    )
    assert backend.executed == [] and [c["limit"] for c in planner.calls] == [6, 1, 1, 1]
    assert result.whole_goal_recheck_calls == 2 and result.whole_goal_recheck_done == 1
    assert result.capacity_recovery_verified == 0 and len(retries(trace)) == 1


def test_explicit_selector_still_needs_existing_master_enable():
    assert AgentConfig().jev_capacity_recovery_policy == "vlm_once_v1"
    result, trace, planner, desktop = run_capacity(
        [initial_plan()], jev_capacity_recovery=False, jev_capacity_recovery_policy=POLICY,
    )
    assert result.status == "error" and planner.limits == [6] and desktop.inputs == []
    assert result.capacity_recovery_attempts == 0 and not capacity_events(trace)


def test_nonempty_zero_input_receipt_cannot_verify_recovery_or_resume_h6():
    def setup(backend, _planner, _desktop):
        backend.execute = lambda candidate, observation_id, decision_id: ActionReceipt(
            observation_id, decision_id, candidate.id, candidate.channel,
            candidate.capability, True, 0, 0, {"environment_steps": 0},
        )
        backend.verify = lambda *_: Verification(True, "agent.effect", {"postconditions_met": True})

    result, _trace, planner, desktop = run_capacity(
        [initial_plan(), plan(check_step("Alpha"))], setup=setup,
        jev_capacity_recovery_policy=POLICY,
    )
    assert result.status == "stuck" and "no GUI input" in result.reason
    assert planner.limits == [6, 1] and desktop.inputs == []
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0


def test_bad_plan_retry_restores_input_evidence_mode_and_planner_deadline_before_close():
    changes, deadlines, state_before_close = [], [], []

    def setup(backend, planner, _desktop):
        configure, close = backend.set_input_observation_evidence_enabled, backend.close

        def configured(value):
            changes.append(value)
            return configure(value)

        def closed():
            state_before_close.append((backend._input_evidence_enabled, backend._input_evidence))
            close()

        backend.set_input_observation_evidence_enabled, backend.close = configured, closed
        planner.set_episode_deadline = deadlines.append

    result, _trace, _planner, _desktop = run_capacity(
        [initial_plan(), plan(check_step("Ghost"))], setup=setup,
        jev_capacity_recovery_policy=POLICY, reuse_verified_pending_snapshot=True,
    )
    assert result.status == "planner_error"
    assert changes == [True, False] and state_before_close == [(False, None)]
    assert deadlines[0] is not None and deadlines[-1] is None
