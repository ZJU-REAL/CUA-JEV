"""Explicit five-tool H6 policy; MockTransport and original guarded local backend."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from test_program_wire_v64 import wire_declaration
from test_property_preparation import Backend, Environment, field
from test_property_preparation import initial as initial
from test_semantic_tools_planner import model
from test_semantic_tools_v57 import current, packet
from test_semantic_tools_v57 import observed as observed

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.desktop_phase_handoff_v62 import HighLevelContextRequest
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_stage_program import StageProgramPlan
from cua_jev.models import Decision
from cua_jev.trace import JsonlTrace

FIVE = ["declare_property_stage", "declare_navigation_stage", "declare_field_edits_stage",
        "declare_completion", "declare_current_context_needed"]
GOAL = "Set First result to Alpha AND Second result to Beta; both values must be present."
HISTORY = [{"feedback": "Keep every requirement of the original two-result goal."}]


def complete():
    return {"label": "Current main view", "conditions": [{"kind": "window_contains", "value": "Main"}]}


def capture(snapshot, raw, *, tool="declare_completion", policy=None, limit=6, formal=False,
            repair=None):
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=packet(raw, tool, content=None))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        options = {} if policy is None else {"desktop_normal_stage_policy": policy}
        planner = model(client, desktop_output_format="semantic_tools_v64", **options)
        planner.set_prepare_repair_context(repair)
        plan = planner.plan(GOAL, snapshot, HISTORY, max_actions=limit, whole_goal_recheck=formal)
    assert planner.calls == len(requests) == 1
    return plan, requests[0], planner


def test_default_optional_program_retains_six_tools_original_wire_and_program_decoder(observed):
    wire = wire_declaration()
    first, request, planner = capture(observed, wire, tool="declare_stage_program")
    second, explicit, _ = capture(observed, wire, tool="declare_stage_program", policy="program_optional")
    assert isinstance(first, StageProgramPlan) and isinstance(second, StageProgramPlan)
    assert first == second and request.content == explicit.content
    assert len(json.loads(request.content)["tools"]) == 6
    assert planner.last_request_provenance["program_tool_added"] is True


def test_atomic_H6_keeps_exact_five_schemas_full_goal_image_history_and_actual_hashes(observed):
    _, old_request, _ = capture(observed, complete())
    plan, request, planner = capture(observed, complete(), policy="atomic_stage_only")
    old, body = json.loads(old_request.content), json.loads(request.content)
    assert plan.done and plan.completion_conditions
    assert [tool["function"]["name"] for tool in body["tools"]] == FIVE
    assert body["tools"] == old["tools"][1:]
    restored = copy.deepcopy(body)
    restored["tools"] = old["tools"]
    restored["messages"][0]["content"] = old["messages"][0]["content"]
    assert restored == old
    system = body["messages"][0]["content"]
    assert "declare_stage_program" not in system and "EVERY member" not in system
    provenance = planner.last_request_provenance
    assert provenance["normal_high_level_phase"] is True and provenance["normal_program_eligible"] is True
    assert provenance["normal_stage_policy"] == "atomic_stage_only"
    assert provenance["program_tool_added"] is False and provenance["declaration_tool_count"] == 5
    assert provenance["offered_tool_names"] == FIVE
    assert provenance["request_body_sha256"] == _digest(body)
    assert provenance["declaration_schema_sha256"] == _digest(body["tools"])
    assert provenance["parameter_schema_sha256"] == {
        tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in body["tools"]}
    assert body["enable_thinking"] is True and request.extensions["timeout"]["read"] == 90
    assert planner.last_response_transport["normal_stage_policy"] == "atomic_stage_only"


@pytest.mark.parametrize("case", ["H1", "formalH1", "formalH6", "repair"])
def test_non_normal_body_plan_and_original_guard_shape_are_unchanged(observed, case):
    formal = case.startswith("formal")
    limit = 6 if case == "formalH6" else 1
    raw, tool, repair = complete(), "declare_completion", None
    if case == "H1":
        raw, tool = current(1), "declare_native_current_action"
    elif case == "repair":
        token = "ab" * 16
        raw = {"summary": "Repair current preparation", "id": "only", "done": False,
               "operation": "visual_fill", "action_label": "Visible current field", "value": "query",
               "box": [100, 100, 300, 200], "box_space": "normalized_1000", "prepare_repair": token}
        tool = "declare_visual_current_action"
        repair = {"token": token, "remaining_stage": {"scope": {"app": "Editor"}}}
    expected, old, _ = capture(observed, raw, tool=tool, limit=limit, formal=formal, repair=repair)
    actual, request, _ = capture(observed, raw, tool=tool, policy="atomic_stage_only",
                                 limit=limit, formal=formal, repair=repair)
    assert expected == actual and request.content == old.content


@pytest.mark.parametrize("bad", ["program", "current_escape", "unknown_attribute"])
def test_unoffered_or_extra_output_is_rejected_after_exactly_one_POST(observed, bad):
    tool, raw = "declare_stage_program", wire_declaration()
    if bad == "current_escape":
        tool, raw = "declare_native_current_action", current(6)
    elif bad == "unknown_attribute":
        tool, raw = "declare_completion", {**complete(), "extra_authority": True}
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: (
        requests.append(request) or httpx.Response(200, json=packet(raw, tool, content=None))
    ))) as client:
        planner = model(client, desktop_output_format="semantic_tools_v64",
                        desktop_normal_stage_policy="atomic_stage_only")
        with pytest.raises(ValueError):
            planner.plan(GOAL, observed, HISTORY, max_actions=6)
    assert planner.calls == len(requests) == 1
    assert [tool["function"]["name"] for tool in json.loads(requests[0].content)["tools"]] == FIVE


def test_atomic_context_handoff_still_uses_the_original_typed_zero_input_result(observed):
    plan, _, planner = capture(observed, {"reason": "missing_future_semantics"},
        tool="declare_current_context_needed", policy="atomic_stage_only")
    assert isinstance(plan, HighLevelContextRequest)
    assert plan.reason == "missing_future_semantics" and planner.calls == 1


def test_two_complete_properties_use_two_H6_declarations_then_original_full_goal_H1(initial):
    environment = Environment(initial)
    environment.opened = True
    environment.focus = "first"
    environment.snapshot = replace(initial, controls=(initial.controls[0],
        field("first", "First result", value="old-first", focused=True, box=(100, 100, 400, 180)),
        field("second", "Second result", value="old-second", focused=False, box=(100, 300, 400, 380))))
    final_conditions = [{"kind": "value_equals", "target": {"name": name, "role": "textbox",
        "app": "Editor", "window": "Main"}, "value": value}
        for name, value in [("First result", "Alpha"), ("Second result", "Beta")]]

    def property_result(label, value, completion):
        result = {"label": label, "scope": {"app": "Editor"}, "edit_operation": "fill",
                  "edit_value": value, "numeric_finalize_key": "none",
                  "confirmation_label": "none", "completion": completion}
        if completion == "goal":
            result["goal_conditions"] = final_conditions
        return result

    replies = iter([
        packet(property_result("Set First result to Alpha", "Alpha", "stage"), "declare_property_stage"),
        packet(property_result("Set Second result to Beta", "Beta", "goal"), "declare_property_stage"),
        packet({"label": "Both original requested values observed", "conditions": final_conditions},
               "declare_completion"),
    ])
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=next(replies))

    class ObservedFieldChoice:
        def __init__(self):
            self.selected = []

        def choose(self, observation, candidates):
            candidates = list(candidates)
            name, subgoal = [("First result", "Set First result to Alpha"),
                             ("Second result", "Set Second result to Beta")][len(self.selected)]
            # The real policy receives wire-encoded candidates, not internal
            # bound_control/value arguments. Select using its visible description.
            matches = [candidate for candidate in candidates
                       if candidate.description.startswith("fill " + name + " ")
                       and "toward property goal: " + subgoal in candidate.description]
            assert len(matches) == 1, [candidate.description for candidate in candidates]
            chosen = matches[0]
            self.selected.append(name)
            return Decision(observation.observation_id, chosen.id,
                {candidate.id: float(candidate.id == chosen.id) for candidate in candidates},
                1.0, "offline-observed-choice", 0)

    trace = JsonlTrace()
    choice = ObservedFieldChoice()
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v64",
                        desktop_normal_stage_policy="atomic_stage_only")
        result = AgentRunner(Backend(environment, pause=0), planner, policy=choice, trace=trace,
            config=AgentConfig(mode="hybrid", allow_actions=True, plan_horizon=6, max_plans=3,
                max_steps=4, max_failures=1, whole_goal_recheck=True)).run(GOAL)
    assert result.planner_calls == planner.calls == len(requests) == 3, result.to_dict()
    assert choice.selected == ["First result", "Second result"]
    assert result.status == "unverified" and not result.success
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
    assert result.steps == 2 and result.property_edits == 2
    assert [action["parameters"]["text"] for action in environment.actions
            if action["action_type"] == "TYPING"] == ["Alpha", "Beta"]
    assert {control.label: control.value for control in environment.snapshot.controls
            if control.editable} == {"First result": "Alpha", "Second result": "Beta"}
    plans = [event["payload"] for event in trace.events if event["kind"] == "agent_plan"]
    assert [plan["max_actions"] for plan in plans] == [6, 6, 1]
    assert [plan["reason"] for plan in plans] == ["initial", "plan_exhausted", "whole_goal_recheck"]
    assert not any(event["kind"] == "agent_program_declared" for event in trace.events)
    assert all(json.loads(request.content)["messages"][1]["content"][0]["type"] == "text"
               for request in requests)
    states = [json.loads(json.loads(request.content)["messages"][1]["content"][0]["text"])
              for request in requests]
    assert all(state["goal"] == GOAL for state in states)


@pytest.mark.parametrize("policy,fmt", [(None, "semantic_tools_v64"), (True, "semantic_tools_v64"),
    ("unknown", "semantic_tools_v64"), ("atomic_stage_only", "semantic_tools_v62"),
    ("atomic_stage_only", "semantic_json_v66")])
def test_unsupported_policy_configuration_fails_before_any_HTTP(policy, fmt):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request))) as client:
        with pytest.raises(ValueError):
            model(client, desktop_output_format=fmt, desktop_normal_stage_policy=policy)
    assert requests == []
