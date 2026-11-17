"""Other flat declarations traverse the real planner and original desktop parser."""

import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_desktop_tool_declaration_planner import Backend, Environment
from test_semantic_tools_planner import initial as initial
from test_semantic_tools_planner import model, packet
from test_semantic_tools_planner import snapshot as snapshot

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.desktop_semantic_tools import FORMAT
from cua_jev.desktop_tool_declaration import FORMAT as ORIGINAL_FORMAT
from cua_jev.trace import JsonlTrace

SEARCH = {"name": "Search", "role": "textbox", "app": "Editor", "window": "Main"}
VARIANTS = [
    (
        "declare_navigation_stage", "navigate",
        {"label": "Reach the requested view", "scope": {"app": "Editor"},
         "arrival": [{"kind": "exists", "target": {"name": "Requested view", "app": "Editor"}}],
         "completion": "stage"},
    ),
    (
        "declare_field_edits_stage", "edit_fields",
        {"label": "Change the current field", "completion": "goal",
         "edits": [{"operation": "fill", "target": SEARCH, "value": "requested text"}]},
    ),
    (
        "declare_current_action", "act_current",
        {"label": "Open current Help", "action": {"operation": "click", "ref": "help"}},
    ),
    (
        "declare_completion", "complete",
        {"label": "Requested field state already holds",
         "conditions": [{"kind": "value_equals", "target": SEARCH, "value": "old query"}]},
    ),
    ("declare_completion", "complete", {"label": "Advisory completion only"}),
]


def capture_declaration(snapshot, raw, name, *, output_format=FORMAT, limit=6, history=None):
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=output_format)
        plan = planner.plan("Follow the requested current desktop stage", snapshot,
                            [] if history is None else history, max_actions=limit)
    assert len(requests) == planner.calls == 1
    return plan, requests[0], planner


@pytest.mark.parametrize("name,intent,raw", VARIANTS)
def test_other_flat_functions_equal_original_52_plans(snapshot, name, intent, raw):
    plan, request, planner = capture_declaration(snapshot, raw, name)
    original, _request, _planner = capture_declaration(
        snapshot, {"intent": intent, **raw}, "declare_desktop_stage", output_format=ORIGINAL_FORMAT,
    )
    body = json.loads(request.content)
    declaration = next(tool["function"] for tool in body["tools"] if tool["function"]["name"] == name)
    assert Draft202012Validator(declaration["parameters"]).is_valid(raw)
    assert plan == original
    assert planner.last_response_transport["tool_name"] == name
    assert planner.last_response_transport["mapped_intent"] == intent
    assert planner.last_request_provenance["response_transport"] == FORMAT
    assert planner.last_request_provenance["stage_mapping_applied"] is True
    assert [part["type"] for part in body["messages"][1]["content"]] == ["text", "image_url"]
    if intent == "navigate":
        assert plan.reusable and plan.steps[0].native_navigation_gestures
        assert plan.steps[0].target.app == "Editor" and not plan.steps[0].target.name
        assert plan.steps[0].postconditions[0].target.name == "Requested view"
        assert not plan.completion_conditions
    elif intent == "edit_fields":
        assert plan.reusable and len(plan.steps) == 1
        assert plan.steps[0].value == "requested text" and plan.steps[0].target.name == "Search"
        assert [condition.kind for condition in plan.steps[0].postconditions] == [
            "value_equals", "ready_to_fill",
        ]
        assert plan.completion_conditions == plan.steps[0].postconditions
    elif intent == "act_current":
        assert not plan.reusable and len(plan.steps) == 1
        assert plan.steps[0].operation == "click" and plan.steps[0].ref == "help"
    else:
        assert plan.done and not plan.steps
        assert bool(plan.completion_conditions) == ("conditions" in raw)
        if "conditions" in raw:
            assert plan.completion_conditions[0].value == "old query"


def test_history_effective_H1_keeps_original_body_and_actual_transport_provenance(snapshot):
    raw = {"summary": "Open current Help", "done": False,
           "steps": [{"id": "s1", "operation": "click", "ref": "help"}]}
    history = [{"recovery_required": True, "reason": "Existing single-action recovery"}]
    plan, request, planner = capture_declaration(
        snapshot, raw, "declare_desktop_stage", history=history,
    )
    original, original_request, _original_planner = capture_declaration(
        snapshot, raw, "declare_desktop_stage", output_format=ORIGINAL_FORMAT, history=history,
    )
    assert request.content == original_request.content and plan == original
    provenance = planner.last_request_provenance
    assert provenance["max_actions"] == 1 and provenance["image_uploaded"] is True
    assert provenance["requested_format"] == FORMAT
    assert provenance["response_transport"] == ORIGINAL_FORMAT
    assert provenance["stage_mapping_applied"] is False
    assert planner.last_response_transport["response_transport"] == ORIGINAL_FORMAT


@pytest.mark.parametrize("limit", [1, 6])
def test_runner_emits_actual_H1_or_H6_transport_from_the_real_request(snapshot, limit):
    env, trace, requests = Environment(snapshot), JsonlTrace(), []
    raw = ({"summary": "Open current Help", "done": False,
            "steps": [{"id": "s1", "operation": "click", "ref": "help"}]} if limit == 1 else
           {"label": "Open current Help", "action": {"operation": "click", "ref": "help"}})
    name = "declare_desktop_stage" if limit == 1 else "declare_current_action"

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client)
        result = AgentRunner(
            Backend(env, pause=0), planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, max_steps=1, max_plans=1,
                               max_failures=2, plan_horizon=limit, timeout_s=30),
        ).run("Use current Help")
    assert len(requests) == result.planner_calls == planner.calls == len(env.actions) == 1
    events = [event["payload"] for event in trace.events
              if event["kind"] == "agent_model_output_transport"]
    assert len(events) == 1
    assert events[0]["requested_format"] == FORMAT
    expected_transport = ORIGINAL_FORMAT if limit == 1 else FORMAT
    assert events[0]["response_transport"] == expected_transport
    assert events[0]["declaration"]["response_transport"] == expected_transport
    assert events[0]["declaration"]["stage_mapping_applied"] is (limit > 1)
    assert events[0]["max_actions"] == limit and events[0]["image_uploaded"] is True
    assert events[0]["http_wall_ms"] == planner.last_request_provenance["http_wall_ms"]
