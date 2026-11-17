"""Capability-restricted H1: real MockTransport, original decoder and backend guards."""

import copy
import json
import socket
from dataclasses import replace

import httpx
import pytest
from test_atomic_stage_policy_v89 import complete
from test_native_current_example_v89 import Backend, Environment
from test_osworld_remote_driver import arguments, install_episode_fakes
from test_osworld_remote_driver import driver as driver
from test_property_preparation import initial as initial
from test_semantic_tools_planner import model
from test_semantic_tools_v57 import current, packet
from test_semantic_tools_v57 import observed as observed

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.trace import JsonlTrace

FOUR = ["declare_native_current_action", "declare_visual_current_action",
        "declare_key_page_action", "declare_completion"]
GOAL = "Complete every original requested result and verify it in the current desktop."
HISTORY = [{"feedback": "Retain every original goal requirement and every declared guard."}]
LIMITED = "ref_visual_key_v1"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("Only local MockTransport is permitted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def capture(snapshot, raw, *, tool="declare_native_current_action", policy=None, limit=1,
            formal=False, repair=None, history=None):
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=packet(raw, tool, content=None))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        options = {} if policy is None else {"desktop_current_action_policy": policy}
        planner = model(client, desktop_output_format="semantic_tools_v64", **options)
        planner.set_prepare_repair_context(repair)
        plan = planner.plan(GOAL, snapshot, HISTORY if history is None else history,
                            max_actions=limit, whole_goal_recheck=formal)
    assert planner.calls == len(requests) == 1
    return plan, requests[0], planner


def test_default_is_identical_and_restricted_request_keeps_exact_four_schemas_and_full_context(observed):
    raw = current(1)
    original_plan, original_request, _ = capture(observed, raw)
    explicit_plan, explicit_request, _ = capture(observed, raw, policy="all_families")
    assert original_request.content == explicit_request.content and original_plan == explicit_plan
    limited_plan, request, planner = capture(observed, raw, policy=LIMITED)
    old, body = json.loads(original_request.content), json.loads(request.content)
    assert [tool["function"]["name"] for tool in old["tools"]] == [
        "declare_native_current_action", "declare_semantic_current_action", "declare_visual_current_action",
        "declare_key_page_action", "declare_completion"]
    assert [tool["function"]["name"] for tool in body["tools"]] == FOUR
    assert body["tools"] == [tool for tool in old["tools"]
                             if tool["function"]["name"] != "declare_semantic_current_action"]
    restored = copy.deepcopy(body)
    restored["tools"] = old["tools"]
    restored["messages"][0]["content"] = old["messages"][0]["content"]
    assert restored == old and limited_plan == original_plan
    assert body["enable_thinking"] is True and request.extensions["timeout"]["read"] == 90
    provenance = planner.last_request_provenance
    assert provenance["current_action_policy"] == LIMITED
    assert provenance["requested_current_action_policy"] == LIMITED
    assert provenance["current_action_policy_applied"] is True
    assert provenance["semantic_current_disabled"] is True and provenance["capability_restricted"] is True
    assert provenance["semantic_pointer_effect_expressibility_preserved"] is False
    assert provenance["offered_tool_names"] == FOUR and provenance["declaration_tool_count"] == 4
    assert provenance["request_body_sha256"] == _digest(body)
    assert provenance["declaration_schema_sha256"] == _digest(body["tools"])
    assert provenance["parameter_schema_sha256"] == {
        tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in body["tools"]}
    assert planner.last_response_transport["offered_tool_names"] == FOUR
    assert "declare_semantic_current_action is disabled" in body["messages"][0]["content"]
    assert "Never delete or weaken a genuinely required effect" in body["messages"][0]["content"]


@pytest.mark.parametrize("case", ["normalH6", "formalH1", "formalH6", "repair"])
def test_policy_does_not_change_normal_H6_formal_or_repair_body_or_plan(observed, case):
    formal = case.startswith("formal")
    limit = 6 if case in {"normalH6", "formalH6"} else 1
    raw, tool, repair = complete(), "declare_completion", None
    if case == "repair":
        token = "ab" * 16
        raw = {"summary": "Repair current preparation", "id": "current", "done": False,
               "operation": "visual_fill", "action_label": "Visible editable field", "value": "query",
               "box": [100, 100, 300, 200], "box_space": "normalized_1000", "prepare_repair": token}
        tool = "declare_visual_current_action"
        repair = {"token": token, "remaining_stage": {"scope": {"app": "Editor"}}}
    expected, old, _ = capture(observed, raw, tool=tool, limit=limit, formal=formal, repair=repair)
    actual, request, planner = capture(observed, raw, tool=tool, limit=limit, formal=formal,
                                      repair=repair, policy=LIMITED)
    assert actual == expected and request.content == old.content
    assert "semantic_current_disabled" not in planner.last_request_provenance
    assert planner.last_request_provenance["current_action_policy_applied"] is False


@pytest.mark.parametrize("bad", ["semantic", "native_conditions", "unknown_attribute"])
def test_unoffered_semantic_and_incompatible_conditions_are_rejected_without_rewrite_or_retry(observed, bad):
    tool = "declare_semantic_current_action"
    raw = current(1, tool)
    if bad == "native_conditions":
        tool, raw = "declare_native_current_action", current(1)
        raw["postconditions"] = [{"kind": "exists", "target": {"name": "Unobserved result"}}]
    elif bad == "unknown_attribute":
        tool, raw = "declare_native_current_action", {**current(1), "extra_authority": True}
    original, requests = copy.deepcopy(raw), []
    with httpx.Client(transport=httpx.MockTransport(lambda request: (
        requests.append(request) or httpx.Response(200, json=packet(raw, tool, content=None))
    ))) as client:
        planner = model(client, desktop_output_format="semantic_tools_v64",
                        desktop_current_action_policy=LIMITED)
        with pytest.raises(ValueError):
            planner.plan(GOAL, observed, HISTORY, max_actions=1)
    assert raw == original and planner.calls == len(requests) == 1
    assert [tool["function"]["name"] for tool in json.loads(requests[0].content)["tools"]] == FOUR


def test_original_native_backend_executes_one_input_and_stale_frame_still_refuses(initial):
    environment = Environment(initial)
    backend = Backend(environment, pause=0)
    try:
        before = backend.observe()
        raw = {"summary": "Activate current control", "id": "current", "done": False,
               "operation": "click", "ref": "help"}
        plan, _, _ = capture(before, raw, policy=LIMITED)
        assert not plan.steps[0].postconditions and not plan.steps[0].preconditions
        candidate, = backend.compile(plan.steps[0], before, before)
        receipt = backend.execute(candidate, "observation", "decision")
        assert receipt.success and receipt.output["environment_steps"] == len(environment.actions) == 1
        assert backend.verify(candidate, receipt).passed
        assert not plan.done and not plan.completion_conditions  # UI progress is not a goal claim.
        assert backend.compile(plan.steps[0], before, backend.observe()) == ()
        assert len(environment.actions) == 1
    finally:
        backend.close()


def test_native_fill_keeps_original_internal_focus_guard_and_sends_no_text_after_focus_loss(observed):
    class LostFocus(Environment):
        def step(self, action, pause):
            super().step(action, pause)
            self.snapshot = replace(self.snapshot, controls=tuple(
                replace(control, focused=False) if control.ref == "field" else control
                for control in self.snapshot.controls))
            return self.snapshot, 0, False, {}

    environment = LostFocus(observed)
    backend = Backend(environment, pause=0)
    try:
        before = backend.observe()
        plan, _, _ = capture(before, {"summary": "Fill current input", "id": "current", "done": False,
            "operation": "fill", "ref": "field", "value": "new"}, policy=LIMITED)
        candidate, = backend.compile(plan.steps[0], before, before)
        receipt = backend.execute(candidate, "observation", "decision")
        assert not receipt.success and receipt.output["failure_code"] == "text_focus_guard_rejected"
        assert [action["action_type"] for action in environment.actions] == ["CLICK"]
        assert not backend.verify(candidate, receipt).passed
    finally:
        backend.close()


def test_real_runner_typed_context_recovery_uses_fresh_restricted_H1_once(initial):
    environment = Environment(initial)
    backend = Backend(environment, pause=0)
    responses = iter([
        packet({"reason": "missing_future_semantics"}, "declare_current_context_needed", content=None),
        packet({"summary": "Activate current control", "id": "current", "done": False,
                "operation": "click", "ref": "help"}, "declare_native_current_action", content=None),
    ])
    requests, trace = [], JsonlTrace()
    with httpx.Client(transport=httpx.MockTransport(lambda request: (
        requests.append(request) or httpx.Response(200, json=next(responses))
    ))) as client:
        planner = model(client, desktop_output_format="semantic_tools_v64",
            desktop_normal_stage_policy="atomic_stage_only", desktop_current_action_policy=LIMITED)
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(mode="hybrid", allow_actions=True, plan_horizon=6,
                max_plans=2, max_steps=4, max_failures=3, whole_goal_recheck=True)).run(GOAL)
    try:
        assert result.planner_calls == planner.calls == len(requests) == 2, result.to_dict()
        assert result.steps == 1 and len(environment.actions) == 1
        assert result.status == "max_plans" and not result.success
        bodies = [json.loads(request.content) for request in requests]
        assert len(bodies[0]["tools"]) == 5
        assert [tool["function"]["name"] for tool in bodies[1]["tools"]] == FOUR
        states = [json.loads(body["messages"][1]["content"][0]["text"]) for body in bodies]
        assert all(state["goal"] == GOAL for state in states)
        plans = [event["payload"] for event in trace.events if event["kind"] == "agent_plan"]
        assert len(plans) == 1 and plans[0]["max_actions"] == 1 and plans[0]["recovery"] is True
    finally:
        backend.close()


@pytest.mark.parametrize("policy", ["all_families", LIMITED])
def test_cli_only_forwards_and_records_the_explicit_opt_in_locally(driver, monkeypatch, tmp_path, policy):
    calls = install_episode_fakes(monkeypatch, driver)
    assert driver.main(arguments(tmp_path, modes=["hybrid", "vlm"], **{
        "desktop-stage-contract": "property_intent_v52", "desktop-output-format": "semantic_tools_v64",
        "current-action-policy": policy,
    })) == 0
    assert len(calls["planners"]) == 3
    for planner in calls["planners"]:
        if policy == "all_families":
            assert "desktop_current_action_policy" not in planner.settings
        else:
            assert planner.settings["desktop_current_action_policy"] == LIMITED
    summary = json.loads((tmp_path / "run/summary.json").read_text(encoding="utf-8"))
    settings = summary["experiment_settings"]
    if policy == "all_families":
        assert "desktop_current_action_policy" not in settings
    else:
        assert settings["desktop_current_action_policy"] == LIMITED
        assert settings["current_action_capability_restricted"] is True
    assert all("--current-action-policy" not in environment.command for environment in calls["environments"])
    assert [config.plan_horizon for config in calls["configs"]] == [6, 1]


@pytest.mark.parametrize("policy,fmt", [(None, "semantic_tools_v64"), (True, "semantic_tools_v64"),
    ("unknown", "semantic_tools_v64"), (LIMITED, "semantic_tools_v58"), (LIMITED, "semantic_json_v66")])
def test_invalid_or_other_format_opt_in_fails_before_HTTP(policy, fmt):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request))) as client:
        with pytest.raises(ValueError):
            model(client, desktop_output_format=fmt, desktop_current_action_policy=policy)
    assert requests == []
