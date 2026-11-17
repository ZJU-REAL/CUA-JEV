"""One explicitly enabled network rescue, using real guards and mocked transports."""

import json
from dataclasses import replace

import httpx
import pytest
from test_desktop_invalid_plan_recovery import Desktop, RecordedPlanner, check_step, condition, plan

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import JevFailure, JevPolicy, JevRequestError
from cua_jev.trace import JsonlTrace


def initial_plan():
    return plan(check_step("Alpha"), check_step("Beta"))


def run(replies, *, handler=None, desktop=None, accept=True, setup=None, **settings):
    desktop = desktop or Desktop(opened=True)
    planner = RecordedPlanner(replies)
    backend, trace = OSWorldBackend(desktop, pause=0), JsonlTrace()

    def timeout(request):
        raise httpx.ConnectTimeout("secret endpoint and authorization", request=request)

    policy = JevPolicy(api_key="private-key", retries=2, sleep=lambda _: None,
                       client=httpx.Client(transport=httpx.MockTransport(handler or timeout)))
    if setup:
        setup(backend, planner, desktop)
    try:
        result = AgentRunner(
            backend, planner, policy=policy, trace=trace,
            config=AgentConfig(allow_actions=True, **{"jev_network_recovery": True, **settings}),
        ).run("Enable Alpha and Beta", success=(lambda _: desktop.alpha and desktop.beta) if accept else None)
    finally:
        backend.close()
        planner.close()
        policy._client.close()
    return result, trace, planner, desktop


@pytest.mark.parametrize("singleton", [True, False])
def test_one_fresh_rescue_keeps_failed_jev_attempts_and_returns_to_normal_hybrid(singleton):
    result, trace, planner, desktop = run(
        [initial_plan(), plan(check_step("Alpha")), plan(check_step("Beta"))],
        singleton_fast_path=singleton,
        # Stop after verified rescue so singleton=False cannot trigger a second network failure.
        max_steps=1,
    )
    assert result.steps == result.network_recovery_actions == result.network_recovery_verified == 1
    assert result.jev_calls == result.jev_network_failures == result.network_recovery_attempts == 1
    assert result.jev_request_attempts == 3 and result.jev_wall_ms > 0
    assert result.planner_calls == 2 and planner.limits == [6, 1]
    assert result.candidate_counts == (2, 1) and desktop.alpha
    failures = [e["payload"] for e in trace.events if e["kind"] == "policy_failure"]
    assert len(failures) == 1 and len(failures[0]["retry_history"]) == 3
    assert all(item["code"] == "connect_timeout" for item in failures[0]["retry_history"])
    decisions = [e["payload"] for e in trace.events if e["kind"] == "decision"]
    assert [d["model"] for d in decisions] == ["vlm-network-recovery"]
    observations = [e["payload"]["observation_id"] for e in trace.events if e["kind"] == "observation"]
    assert len(set(observations)) == 2
    assert "secret endpoint" not in str(trace.events) and "private-key" not in str(trace.events)


def test_verified_rescue_resumes_hybrid_horizon_without_hiding_extra_vlm_call():
    result, _trace, planner, desktop = run([
        initial_plan(), plan(check_step("Alpha")), plan(check_step("Beta")),
    ])
    assert result.success and desktop.alpha and desktop.beta
    assert planner.limits == [6, 1, 6] and result.planner_calls == 3
    assert result.jev_calls == 1 and result.network_recovery_verified == 1


def test_state_changes_during_network_failure_are_seen_by_recovery_planner():
    desktop = Desktop(opened=True)
    attempts = 0

    def failure(request):
        nonlocal attempts
        attempts += 1
        desktop.alpha = True
        raise httpx.ConnectError("private outage", request=request)

    result, _trace, planner, _ = run([initial_plan(), plan(check_step("Beta"))],
                                    desktop=desktop, handler=failure)
    assert result.success and result.steps == 1 and attempts == 3
    state = json.loads(planner.requests[1]["messages"][1]["content"][0]["text"])
    alpha = next(c for c in state["snapshot"]["controls"] if c["label"] == "Alpha")
    assert alpha["checked"] is True
    assert planner.history(1)[-1]["reason"] == "jev_network_recovery"


@pytest.mark.parametrize("failure", [401, 403, 451, 429, 503, "invalid_response", "request_error"])
def test_permanent_http_or_invalid_response_never_enters_network_rescue(failure):
    def handler(request):
        if failure == "invalid_response":
            return httpx.Response(200, json={"answers": {"action": {"choice": "not-offered"}}})
        if failure == "request_error":
            raise httpx.ProtocolError("private provider detail", request=request)
        return httpx.Response(failure, text="private provider body")

    result, trace, planner, desktop = run([initial_plan()], handler=handler)
    assert result.status == "error" and result.reason == "PolicyError"
    assert result.network_recovery_attempts == result.steps == 0 and desktop.inputs == []
    assert planner.limits == [6] and "private provider" not in str(trace.events)


def test_prior_invalid_response_cannot_be_hidden_by_final_timeout():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json=[])
        raise httpx.ConnectTimeout("private endpoint", request=request)

    result, _trace, planner, _ = run([initial_plan()], handler=handler)
    assert result.status == "error" and result.network_recovery_attempts == 0
    assert attempts == 3 and planner.limits == [6]


def test_default_and_disabled_preserve_stop_after_network_failure():
    options = {"jev_network_recovery": False}
    assert AgentConfig().jev_network_recovery is False
    result, _trace, planner, desktop = run([initial_plan()], **options)
    assert result.status == "error" and result.jev_network_failures == 1
    assert result.jev_request_attempts == 3 and result.network_recovery_attempts == 0
    assert planner.limits == [6] and desktop.inputs == []


def test_second_network_failure_stops_without_a_second_rescue():
    uncheck = {**check_step("Alpha"), "operation": "uncheck",
               "postconditions": [condition("unchecked", "Alpha")]}
    result, _trace, planner, desktop = run([
        initial_plan(), plan(check_step("Alpha")), plan(uncheck, check_step("Beta")),
    ])
    assert result.status == "error" and result.jev_calls == result.jev_network_failures == 2
    assert result.jev_request_attempts == 6 and result.network_recovery_attempts == 1
    assert result.network_recovery_verified == 1 and result.steps == len(desktop.inputs) == 1
    assert planner.limits == [6, 1, 6]


@pytest.mark.parametrize("reply", ["private bad model JSON", "http_error", plan(check_step("Alpha"),
                                                                          check_step("Beta"))])
def test_failed_recovery_planning_consumes_only_one_call_and_stops(reply):
    result, _trace, planner, desktop = run([initial_plan(), reply])
    assert result.status == "planner_error" and result.network_recovery_attempts == 1
    assert planner.limits == [6, 1] and result.steps == 0 and desktop.inputs == []


def test_fresh_compile_ambiguity_cannot_select_a_default_candidate():
    def setup(backend, planner, _desktop):
        original = backend.compile

        def compile(*args):
            candidates = original(*args)
            if planner.calls == 2 and candidates:
                return (*candidates, replace(candidates[0], id="another-route"))
            return candidates

        backend.compile = compile

    result, _trace, planner, desktop = run([initial_plan(), plan(check_step("Alpha"))], setup=setup)
    assert result.status == "stuck" and result.network_recovery_actions == 0
    assert planner.limits == [6, 1] and desktop.inputs == []


def test_state_changed_after_recovery_plan_is_not_sent_from_old_observation():
    def setup(_backend, planner, desktop):
        original = planner.plan

        def get_plan(*args, **kwargs):
            value = original(*args, **kwargs)
            if planner.calls == 2:
                desktop.alpha = True
            return value

        planner.plan = get_plan

    result, _trace, planner, desktop = run([initial_plan(), plan(check_step("Alpha"))],
                                          setup=setup, accept=False)
    assert result.status == "stuck" and result.network_recovery_actions == 0
    assert result.condition_skips == 1 and desktop.inputs == [] and planner.limits == [6, 1]


def test_unverified_recovery_input_stops_without_another_plan_or_context_allowance():
    desktop = Desktop(opened=True)

    def step(action, pause=0):
        desktop.inputs.append(action)
        return desktop._get_obs(), 0, False, {}

    desktop.step = step
    result, _trace, planner, _ = run([initial_plan(), plan(check_step("Alpha"))], desktop=desktop)
    assert result.status == "stuck" and result.network_recovery_actions == 1
    assert result.network_recovery_verified == result.context_recoveries == 0
    assert result.steps == len(desktop.inputs) == 1 and planner.limits == [6, 1]


def test_no_extra_planner_budget_is_granted():
    result, _trace, planner, desktop = run([initial_plan()], max_plans=1)
    assert result.status == "max_plans" and result.network_recovery_attempts == 0
    assert planner.limits == [6] and desktop.inputs == []


def test_elapsed_network_failure_cannot_start_recovery_beyond_episode_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def handler(request):
        clock[0] += 10
        raise httpx.ConnectTimeout("private endpoint", request=request)

    result, _trace, planner, desktop = run([initial_plan()], handler=handler, timeout_s=20)
    assert result.status == "timeout" and result.network_recovery_attempts == 0
    assert result.jev_wall_ms == 30000 and planner.limits == [6] and desktop.inputs == []


def test_backend_raised_transport_error_is_not_a_preexecution_policy_failure():
    def handler(request):
        ids = list(json.loads(request.content)["questions"]["action"]["criteria"])
        return httpx.Response(200, json={"answers": {"action": {
            "choice": ids[0], "probabilities": {key: float(key == ids[0]) for key in ids},
            "confidence": 1.0,
        }}})

    def setup(backend, _planner, _desktop):
        def execute(_candidate, observation_id, _decision_id):
            raise JevRequestError("private backend detail", JevFailure(
                observation_id, "a" * 64, "connect_timeout", 1, True, None, 1,
                ((1, "connect_timeout", None, 1),),
            ))

        backend.execute = execute

    result, trace, planner, desktop = run([initial_plan()], handler=handler, setup=setup)
    assert result.status == "error" and result.network_recovery_attempts == 0
    assert planner.limits == [6] and desktop.inputs == []
    assert not any(e["kind"] == "policy_failure" for e in trace.events)
    assert "private backend" not in str(trace.events)


def test_failed_fresh_observation_consumes_recovery_and_does_not_replan_again():
    def setup(backend, _planner, _desktop):
        original = backend.observe
        observations = 0

        def observe():
            nonlocal observations
            observations += 1
            if observations == 3:
                raise RuntimeError("private observation detail")
            return original()

        backend.observe = observe

    result, trace, planner, desktop = run([initial_plan()], setup=setup)
    assert result.status == "error" and result.network_recovery_attempts == 1
    assert result.network_recovery_actions == 0 and planner.limits == [6] and desktop.inputs == []
    final_error = next(e["payload"] for e in reversed(trace.events) if e["kind"] == "agent_error")
    assert "policy_failure" not in final_error  # Do not relabel a later error from stale last_failure.
    assert "private observation" not in str(trace.events)


def test_recovery_observation_time_is_inside_original_episode_budget(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    def setup(backend, _planner, _desktop):
        original = backend.observe
        observations = 0

        def observe():
            nonlocal observations
            observations += 1
            if observations == 3:
                clock[0] = 21
            return original()

        backend.observe = observe

    result, _trace, planner, desktop = run([initial_plan()], setup=setup, timeout_s=20)
    assert result.status == "timeout" and result.network_recovery_attempts == 1
    assert planner.limits == [6] and desktop.inputs == []


@pytest.mark.parametrize("value", [1, None, "true"])
def test_recovery_flag_must_be_a_boolean(value):
    with pytest.raises(ValueError, match="jev_network_recovery"):
        AgentConfig(jev_network_recovery=value)
