"""Actual MockTransport failures exercise the original bounded H1 path."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import packet as function_packet
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_runner_v60 import events
from test_structured_declaration_planner_runner_v66 import ProductionMemoryFields, packet

from cua_jev import agent_runner
from cua_jev.agent_planner import MultimodalPlanner, _consume_local_read_timeout
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.trace import JsonlTrace

FORMAT = "semantic_json_v66"
GOAL = "Fill the requested Name field with Ada"
TARGET = {"name": "Name", "role": "textbox", "app": "Editor", "window": "Main"}


def fill():
    return {
        "summary": "Fill the requested Name",
        "id": "only",
        "done": False,
        "operation": "fill",
        "target": TARGET,
        "value": "Ada",
        "postconditions": [{"kind": "value_equals", "target": TARGET, "value": "Ada"}],
    }


class RecordingPlanner(MultimodalPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invocations, self.deadlines = [], []

    def set_episode_deadline(self, value):
        self.deadlines.append(value)
        return super().set_episode_deadline(value)

    def plan(self, goal, snapshot, history, **options):
        self.invocations.append(
            {"goal": goal, "snapshot": snapshot, "history": copy.deepcopy(history), "options": dict(options)}
        )
        return super().plan(goal, snapshot, history, **options)


def run(answer, *, backend=None, goal=GOAL, **settings):
    backend = backend or ProductionMemoryFields()
    requests, trace = [], JsonlTrace()

    def respond(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return answer(request, len(requests), backend)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = make_planner(client, fmt=FORMAT, cls=RecordingPlanner)
        cfg = {
            "allow_actions": True,
            "mode": "hybrid",
            "plan_horizon": 6,
            "max_plans": 2,
            "max_failures": 3,
            "timeout_s": 420,
            **settings,
        }
        result = AgentRunner(
            backend, planner, policy=DirectPlanPolicy(), trace=trace, config=AgentConfig(**cfg)
        ).run(goal)
    return result, backend, planner, trace, requests


def test_actual_read_timeout_6_to_fresh_1_and_original58_entire_body():
    def answer(request, call, _backend):
        if call == 1:
            raise httpx.ReadTimeout("PRIVATE ReadTimeout body URL", request=request)
        return httpx.Response(200, json=function_packet(fill(), "declare_semantic_current_action"))

    result, backend, planner, trace, requests = run(answer)
    assert [item["options"]["max_actions"] for item in planner.invocations] == [6, 1]
    assert planner.invocations[0]["snapshot"] is not planner.invocations[1]["snapshot"]
    assert result.planner_calls == planner.calls == len(requests) == 2
    assert result.steps == backend.environment_steps == 1 and backend.executed == ["Name"]
    second = planner.invocations[1]
    expected_requests = []

    def original_answer(request):
        expected_requests.append(request)
        return httpx.Response(200, json=function_packet(fill(), "declare_semantic_current_action"))

    with httpx.Client(transport=httpx.MockTransport(original_answer)) as client:
        original = make_planner(client, fmt="semantic_tools_v58")
        original.plan(second["goal"], second["snapshot"], second["history"], **second["options"])
    assert requests[1].content == expected_requests[0].content
    body = json.loads(requests[1].content)
    assert body["enable_thinking"] is True and "thinking_budget" not in body
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    recovery = events(trace, "agent_planner_transport_recovery")
    assert len(recovery) == 1 and recovery[0]["planning_failures"] == 1
    assert recovery[0]["original_plan_at"] == 0 and recovery[0]["original_plan_actions"] == 0
    assert recovery[0]["selection_recovery_ticket_used"] is False
    assert (
        recovery[0]["input_sent"]
        is recovery[0]["positive_input_credit"]
        is recovery[0]["task_success"]
        is False
    )
    assert recovery[0]["provider_completion_and_cost_unknown"] is True
    assert not events(trace, "agent_high_level_context_requested")
    assert "PRIVATE" not in json.dumps(trace.events)
    assert planner._local_request_failure is None


def test_failed_H1_still_stops_at_original_three_failures_without_free_retry():
    def answer(request, _call, _backend):
        raise httpx.ReadTimeout("timeout", request=request)

    result, backend, planner, trace, requests = run(answer, max_plans=20)
    assert [item["options"]["max_actions"] for item in planner.invocations] == [6, 1, 1]
    assert len(requests) == result.planner_calls == 3 and result.status == "planner_error"
    assert result.steps == backend.environment_steps == 0 and backend.executed == []
    assert len(events(trace, "agent_planner_transport_recovery")) == 1
    assert not events(trace, "agent_high_level_context_requested")
    assert len(events(trace, "agent_recovery")) == 3


@pytest.mark.parametrize("kind", ["connect", "write", "pool", "protocol", "runtime", 401, 429, 503, "json"])
def test_other_errors_do_not_enter_new_timeout_route(kind):
    def answer(request, _call, _backend):
        classes = {
            "connect": httpx.ConnectTimeout,
            "write": httpx.WriteTimeout,
            "pool": httpx.PoolTimeout,
            "protocol": httpx.RemoteProtocolError,
        }
        if kind in classes:
            raise classes[kind]("ReadTimeout in an untrusted-looking message", request=request)
        if kind == "runtime":
            raise RuntimeError("ReadTimeout")
        if kind == "json":
            return httpx.Response(200, content=b"not json ReadTimeout")
        return httpx.Response(kind, json={"error": {"message": "ReadTimeout"}})

    result, backend, planner, trace, requests = run(answer)
    assert [item["options"]["max_actions"] for item in planner.invocations] == [6, 6]
    assert len(requests) == result.planner_calls == 2 and backend.environment_steps == 0
    assert not events(trace, "agent_planner_transport_recovery")


def test_existing_ValueError_H1_route_survives_without_new_timeout_event():
    def answer(_request, _call, _backend):
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ReadTimeout"}}]}
        )

    _result, _backend, planner, trace, _requests = run(answer)
    assert [item["options"]["max_actions"] for item in planner.invocations] == [6, 1]
    assert not events(trace, "agent_planner_transport_recovery")


@pytest.mark.parametrize("change", ["owner", "token", "provenance", "error", "consume_twice"])
def test_capsule_requires_current_local_identity_and_consumes_once(observed, change):
    def answer(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT)
        with pytest.raises(RuntimeError) as caught:
            planner.plan("Inspect the current desktop", observed, [], max_actions=6)
        marker, error = planner._local_request_failure, caught.value
        if change == "owner":
            planner._local_request_failure = replace(marker, owner=object())
        elif change == "token":
            planner._request_call_token = object()
        elif change == "provenance":
            planner.last_request_provenance = dict(planner.last_request_provenance)
        elif change == "error":
            error = RuntimeError(str(error))
        else:
            assert _consume_local_read_timeout(planner, error)["kind"] == "local_post_read_timeout"
        assert _consume_local_read_timeout(planner, error) is None
        assert planner._local_request_failure is None


def test_new_call_clears_previous_marker_before_early_validation(observed):
    def answer(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT)
        with pytest.raises(RuntimeError) as caught:
            planner.plan("Inspect the current desktop", observed, [], max_actions=6)
        marker = planner._local_request_failure
        with pytest.raises(ValueError, match="whole_goal_recheck"):
            planner.plan("Inspect", observed, [], whole_goal_recheck="invalid")
        assert planner._local_request_failure is None and planner.calls == 1
        planner._local_request_failure = marker
        assert _consume_local_read_timeout(planner, caught.value) is None


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"max_plans": 1}, [6]),
        ({"max_failures": 1}, [6]),
        ({"mode": "vlm"}, [1, 1]),
        ({"plan_horizon": 1}, [1, 1]),
    ],
)
def test_original_budget_boundary_or_H1_never_gets_new_timeout_fallback(settings, expected):
    def answer(request, _call, _backend):
        raise httpx.ReadTimeout("timeout", request=request)

    result, backend, planner, trace, _requests = run(answer, **settings)
    assert [item["options"]["max_actions"] for item in planner.invocations] == expected
    assert result.steps == backend.environment_steps == 0
    assert not events(trace, "agent_planner_transport_recovery")


def test_timeout_does_not_extend_episode_deadline(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])

    def answer(request, _call, _backend):
        now[0] = 431.0
        raise httpx.ReadTimeout("timeout", request=request)

    result, backend, planner, trace, _requests = run(answer)
    assert result.status == "timeout" and result.planner_calls == 1
    assert planner.deadlines == [430.0, None]
    assert backend.environment_steps == 0 and not events(trace, "agent_planner_transport_recovery")


def test_fresh_recovery_frame_keeps_original_target_guard():
    def answer(request, call, backend):
        if call == 1:
            backend.names = ("Changed",)
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, json=function_packet(fill(), "declare_semantic_current_action"))

    result, backend, planner, trace, _requests = run(answer)
    assert [item["options"]["max_actions"] for item in planner.invocations] == [6, 1]
    assert "Name" not in {control.label for control in planner.invocations[1]["snapshot"].controls}
    assert result.steps == backend.environment_steps == 0 and backend.executed == []
    assert not events(trace, "agent_step")


def test_formal_review_timeout_is_not_a_normal_high_level_trigger():
    def answer(request, call, _backend):
        if call == 1:
            return httpx.Response(
                200,
                json=packet(
                    {"label": "Observed Name", "conditions": [{"kind": "exists", "target": TARGET}]},
                    "declare_completion",
                ),
            )
        raise httpx.ReadTimeout("timeout", request=request)

    result, backend, planner, trace, _requests = run(answer, whole_goal_recheck=True)
    assert result.steps == backend.environment_steps == 0
    assert len(planner.invocations) == 2
    assert planner.invocations[1]["options"].get("whole_goal_recheck") is True
    assert not events(trace, "agent_planner_transport_recovery")


def city_fill():
    target = {**TARGET, "name": "City"}
    return {
        **fill(),
        "summary": "Fill the requested City",
        "target": target,
        "value": "Hangzhou",
        "postconditions": [{"kind": "value_equals", "target": target, "value": "Hangzhou"}],
    }


def test_second_normal_H6_timeout_after_verified_H1_gets_fresh_original_H1(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = ProductionMemoryFields(reveal_after_first=True)
    answers = {2: fill(), 4: city_fill()}

    def answer(request, call, current):
        assert call <= 4
        now[0] = 10.0 + call * 10.0
        if call == 3:
            assert current.executed == ["Name"] and current.values["Name"] == "Ada"
        if call in {1, 3}:
            raise httpx.ReadTimeout("local response read timeout", request=request)
        return httpx.Response(
            200, json=function_packet(answers[call], "declare_semantic_current_action")
        )

    result, backend, planner, trace, requests = run(
        answer, backend=backend, max_plans=4, max_steps=2,
        goal="Fill Name with Ada AND City with Hangzhou; both values must be present.",
    )
    assert [item["options"]["max_actions"] for item in planner.invocations] == [6, 1, 6, 1]
    assert result.planner_calls == planner.calls == len(requests) == 4
    assert result.steps == backend.environment_steps == 2 and backend.executed == ["Name", "City"]
    assert backend.values == {"Name": "Ada", "City": "Hangzhou"}
    assert all(item["verified"] is True for item in events(trace, "agent_step"))
    assert [item["planner_calls"] for item in events(trace, "agent_step")] == [2, 4]
    assert planner.deadlines == [430.0, None]
    assert not result.success  # Memory receipts never supply official evaluation.

    for previous, fallback in ((0, 1), (2, 3)):
        assert planner.invocations[previous]["snapshot"] is not planner.invocations[fallback]["snapshot"]
    assert "City" not in {control.label for control in planner.invocations[1]["snapshot"].controls}
    assert "City" in {control.label for control in planner.invocations[3]["snapshot"].controls}

    # Both fallback request bodies must match the unchanged original58 encoder,
    # with their actual respective goal, fresh frame, history and call options.
    for index, call in ((1, 2), (3, 4)):
        invocation, expected_requests = planner.invocations[index], []

        def original_answer(request, *, captured=expected_requests, raw=answers[call]):
            captured.append(request)
            return httpx.Response(
                200, json=function_packet(raw, "declare_semantic_current_action")
            )

        with httpx.Client(transport=httpx.MockTransport(original_answer)) as client:
            original = make_planner(client, fmt="semantic_tools_v58")
            original.plan(
                invocation["goal"], invocation["snapshot"], invocation["history"],
                **invocation["options"],
            )
        assert requests[index].content == expected_requests[0].content
        body = json.loads(requests[index].content)
        assert body["enable_thinking"] is True and "thinking_budget" not in body
        assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False

    recovery = events(trace, "agent_planner_transport_recovery")
    assert [item["call"] for item in recovery] == [1, 3]
    # The original accepted H1 resets the planning-error streak; the new route
    # neither resets it itself nor erases the earlier actual failure/input cost.
    assert [item["planning_failures"] for item in recovery] == [1, 1]
    assert [item["action_failures"] for item in recovery] == [0, 0]
    assert [item["original_plan_actions"] for item in recovery] == [0, 1]
    assert [item["original_plan_at"] for item in recovery] == [0, 30.0]
    assert all(item["original_episode_deadline"] == 430.0 for item in recovery)
    assert all(item["selection_recovery_ticket_used"] is False for item in recovery)
    assert all(item["input_sent"] is item["positive_input_credit"] is item["task_success"] is False
               for item in recovery)
    assert len(events(trace, "agent_recovery")) == 2
    assert not events(trace, "agent_high_level_context_requested")
    assert planner._local_request_failure is None


@pytest.mark.parametrize("boundary", ["plan_budget", "deadline", "target_changed"])
def test_second_timeout_preserves_remaining_budget_deadline_and_fresh_target_guard(monkeypatch, boundary):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = ProductionMemoryFields(reveal_after_first=True)

    def answer(request, call, current):
        assert call <= 4
        if call in {1, 3}:
            if call == 3:
                assert current.executed == ["Name"] and current.values["Name"] == "Ada"
                if boundary == "deadline":
                    now[0] = 431.0
                if boundary == "target_changed":
                    current.names = ("Name",)
            raise httpx.ReadTimeout("local response read timeout", request=request)
        raw = fill() if call == 2 else city_fill()
        return httpx.Response(200, json=function_packet(raw, "declare_semantic_current_action"))

    result, backend, planner, trace, requests = run(
        answer, backend=backend, max_plans=3 if boundary == "plan_budget" else 4,
        goal="Fill Name with Ada AND City with Hangzhou; both values must be present.",
    )
    expected_limits = [6, 1, 6, 1] if boundary == "target_changed" else [6, 1, 6]
    assert [item["options"]["max_actions"] for item in planner.invocations] == expected_limits
    assert len(requests) == result.planner_calls == len(expected_limits)
    assert result.steps == backend.environment_steps == 1 and backend.executed == ["Name"]
    assert backend.values == {"Name": "Ada"}
    assert len(events(trace, "agent_step")) == 1 and events(trace, "agent_step")[0]["verified"] is True
    assert planner.deadlines == [430.0, None]
    assert not result.success
    if boundary == "target_changed":
        assert "City" not in {control.label for control in planner.invocations[3]["snapshot"].controls}
        assert planner.invocations[2]["snapshot"] is not planner.invocations[3]["snapshot"]
        assert len(events(trace, "agent_planner_transport_recovery")) == 2
    else:
        assert len(events(trace, "agent_planner_transport_recovery")) == 1
    if boundary == "deadline":
        assert result.status == "timeout"
