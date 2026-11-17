"""Flat55 real-request regressions against the original parser and input guards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_desktop_output_schema_contract import CURRENT_ACTIONS, existing_oracle
from test_desktop_output_schema_contract import observed as observed
from test_desktop_tool_declaration_planner import Backend, Environment
from test_observed_dialog_commit import observed as _dialog_fixture
from test_osworld_comparison import comparison, row, stats, write_summary
from test_osworld_driver import driver
from test_semantic_tools_planner import model, packet

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_semantic_tools_v55 import FORMAT, decode_semantic_declaration
from cua_jev.trace import JsonlTrace

TOKEN = "ab" * 16
GOAL = "Use the requested desktop state"
FAMILIES = {
    "declare_native_current_action", "declare_semantic_current_action",
    "declare_visual_current_action", "declare_key_page_action",
}
STAGES = {
    "declare_property_stage", "declare_navigation_stage", "declare_field_edits_stage", "declare_completion",
}
FIELD = {"name": "Field 0", "role": "textbox", "app": "Editor", "window": "Main"}
HELP = {"name": "Help", "role": "button", "app": "Editor", "window": "Main"}
CHECK = {"name": "Wrap", "role": "checkbox", "app": "Editor", "window": "Main"}


@pytest.fixture
def dialog_observed():
    # An explicit wrapper keeps this second "observed" fixture discoverable
    # without a Ruff-removable import used only through a pytest parameter.
    return _dialog_fixture.__wrapped__()


def current(limit, *, ref="help"):
    header = {"summary": "Current Help", "id": "explicit-id", "done": False} if limit == 1 else {
        "label": "Current Help",
    }
    return {**header, "operation": "click", "ref": ref}, "declare_native_current_action"


def capture(observed, raw=None, name=None, *, limit=6, output_format=FORMAT, context=None, history=None):
    requests = []
    if raw is None:
        raw, name = current(limit)

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name=name, content="Ignored private narration"))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=output_format)
        if context is not None:
            planner.set_prepare_repair_context(context)
        plan = planner.plan(GOAL, observed, history or [], max_actions=limit)
    assert len(requests) == planner.calls == 1
    return plan, json.loads(requests[0].content), planner


def parameters(body, name):
    return next(tool["function"]["parameters"] for tool in body["tools"] if tool["function"]["name"] == name)


def original_current(action, limit, *, summary="Current action", identifier="explicit-id", **header):
    if limit == 1:
        return {"summary": summary, "id": identifier, "done": False, **header}, {
            "summary": summary, "steps": [{"id": identifier, **copy.deepcopy(action)}],
            "done": False, **header,
        }
    return {"label": summary}, {"intent": "act_current", "label": summary, "action": copy.deepcopy(action)}


def wire_current(action, limit):
    """Test fixtures supply an explicit alternate spelling of known oracle inputs."""
    action = copy.deepcopy(action)
    header, original = original_current(action, limit)
    if "ref" in action:
        name = "declare_native_current_action"
    elif action["operation"].startswith("visual_"):
        name = "declare_visual_current_action"
        action.setdefault("box_space", "normalized_1000")
    elif "target" in action:
        name = "declare_semantic_current_action"
    else:
        name = "declare_key_page_action"
        if action["operation"] == "wait":
            action.setdefault("value", "")
    if "label" in action:
        action["action_label"] = action.pop("label")
    return {**header, **action}, name, original


GUARDED_ACTIONS = [{
    "operation": operation, "value": value,
    "preconditions": [{"kind": "focused", "target": FIELD}],
    "postconditions": [{"kind": "value_equals", "target": FIELD, "value": "new"}],
    "recovery": "observe",
} for operation, value in (
    ("type_text", "literal text"), ("hotkey", "ctrl+shift+p"), ("key_press", "ENTER"),
    ("press", "Enter"), ("scroll", "down"), ("wait", "250"),
)]
SEMANTIC_ACTIONS = [
    {"operation": "click", "target": HELP,
     "postconditions": [{"kind": "focused", "target": HELP}]},
    {"operation": "fill", "target": FIELD, "value": "new",
     "postconditions": [{"kind": "value_equals", "target": FIELD, "value": "new"}]},
    {"operation": "check", "target": CHECK,
     "postconditions": [{"kind": "checked", "target": CHECK}]},
    {"operation": "uncheck", "target": CHECK,
     "postconditions": [{"kind": "unchecked", "target": CHECK}]},
]
ACTION_ORACLES = [*CURRENT_ACTIONS, {"operation": "press", "ref": "field0", "value": "Enter"},
    {"operation": "visual_click", "label": "Current target", "box": [100, 100, 300, 200],
     "button": "right", "click_count": 1},
    {"operation": "visual_click", "label": "Current target", "box": [100, 100, 300, 200],
     "button": "left", "click_count": 2},
    *GUARDED_ACTIONS, *SEMANTIC_ACTIONS,
]


@pytest.mark.parametrize("limit", [1, 6])
@pytest.mark.parametrize("action", ACTION_ORACLES, ids=lambda action: action["operation"])
def test_every_current_family_operation_preserves_original_parse_oracle_and_actual_wire(
    observed, action, limit,
):
    wire, name, original = wire_current(action, limit)
    expected = existing_oracle(original, observed, limit=limit)
    before = copy.deepcopy(wire)
    plan, body, planner = capture(observed, wire, name, limit=limit)
    assert wire == before and plan == expected
    schema = parameters(body, name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(wire)
    assert all(not step.full_legal_navigation_gestures for step in plan.steps)
    assert planner.last_response_transport["input_authority"] is False
    assert planner.last_response_transport["tool_name"] == name


@pytest.mark.parametrize("limit", [1, 6])
def test_actual55_ancestor54_tool_parameters_system_and_http_metadata_are_separate(observed, limit):
    _plan, body, planner = capture(observed, limit=limit)
    old_raw = ({"summary": "Current Help", "done": False, "steps": [
        {"id": "explicit-id", "operation": "click", "ref": "help"},
    ]} if limit == 1 else {"label": "Current Help", "action": {"operation": "click", "ref": "help"}})
    old_name = "declare_desktop_stage" if limit == 1 else "declare_current_action"
    _old_plan, ancestor, _old_planner = capture(
        observed, old_raw, old_name, limit=limit, output_format="semantic_tools_v54",
    )
    names = {tool["function"]["name"] for tool in body["tools"]}
    assert names == FAMILIES | ({"declare_current_completion"} if limit == 1 else STAGES)
    assert len(body["tools"]) == (5 if limit == 1 else 8)
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["enable_thinking"] is True and "thinking_budget" not in body
    assert "response_format" not in body
    assert all("strict" not in tool["function"] for tool in body["tools"])
    assert [part["type"] for part in body["messages"][1]["content"]] == ["text", "image_url"]
    restored = copy.deepcopy(body)
    restored["tools"] = ancestor["tools"]
    restored["messages"][0]["content"] = ancestor["messages"][0]["content"]
    assert restored == ancestor
    metadata = planner.last_request_provenance
    assert metadata["response_transport"] == metadata["requested_format"] == FORMAT
    assert metadata["ancestor_response_transport"] == "semantic_tools_v54"
    assert (
        metadata["declaration_schema_sha256"] == metadata["declaration_tools_sha256"]
        == _digest(body["tools"])
    )
    assert metadata["parameter_schema_sha256"] == {
        tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in body["tools"]
    }
    assert metadata["ancestor_declaration_schema_sha256"] == _digest(ancestor["tools"])
    assert metadata["ancestor_parameter_schema_sha256"] == {
        tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in ancestor["tools"]
    }
    assert metadata["ancestor_system_sha256"] == _digest(ancestor["messages"][0]["content"])
    assert metadata["projected_system_sha256"] == _digest(body["messages"][0]["content"])
    assert metadata["ancestor_request_sha256"] == _digest(ancestor)
    assert metadata["request_body_sha256"] == _digest(body)
    assert metadata["http_wall_ms"] == planner.last_http_wall_ms >= 0
    assert planner.last_response_transport["response_transport"] == FORMAT
    assert planner.last_response_transport["ancestor_response_transport"] == "semantic_tools_v54"
    assert planner.last_response_transport["parameter_schema_sha256"] == metadata["parameter_schema_sha256"]
    assert planner.last_response_transport["stage_mapping_applied"] is True
    assert planner.last_response_transport["accompanying_content_present"] is True
    assert planner.last_usage == planner.usage_totals
    assert planner.last_usage["completion_details_reasoning_tokens"] == 4
    assert "Ignored private narration" not in json.dumps(metadata)


@pytest.mark.parametrize("identifier", ["owned-id", 42])
def test_H1_explicit_headers_and_semantic_whole_goal_conditions_survive(observed, identifier):
    action = SEMANTIC_ACTIONS[1]
    conditions = [{"kind": "value_equals", "target": FIELD, "value": "new"}]
    header, original = original_current(action, 1, identifier=identifier, completion_conditions=conditions)
    wire = {**header, **copy.deepcopy(action), "depends_on": [], "done": True}
    original["steps"][0]["depends_on"] = []
    original["done"] = True
    expected = existing_oracle(original, observed, limit=1)
    plan, body, _planner = capture(observed, wire, "declare_semantic_current_action", limit=1)
    assert plan == expected and plan.steps[0].id == str(identifier)
    assert not plan.done and plan.completion_conditions[0].value == "new"
    Draft202012Validator(parameters(body, "declare_semantic_current_action")).validate(wire)


@pytest.mark.parametrize("conditions", [None, [], [{"kind": "window_contains", "value": "Main"}]])
def test_H1_empty_completion_remains_advisory_and_never_creates_an_action(observed, conditions):
    wire = {"summary": "Already complete", "done": True}
    original = {**wire, "steps": []}
    if conditions is not None:
        wire["completion_conditions"] = original["completion_conditions"] = conditions
    plan, body, _planner = capture(observed, wire, "declare_current_completion", limit=1)
    assert plan == existing_oracle(original, observed, limit=1)
    assert plan.done and not plan.steps
    Draft202012Validator(parameters(body, "declare_current_completion")).validate(wire)


@pytest.mark.parametrize("limit", [1, 6])
@pytest.mark.parametrize("name,extra", [
    ("declare_native_current_action", {"postconditions": []}),
    ("declare_native_current_action", {"target": None}),
    ("declare_semantic_current_action", {"ref": ""}),
    ("declare_visual_current_action", {"postconditions": [{"kind": "exists", "target": HELP}]}),
    ("declare_visual_current_action", {"target": None}),
    ("declare_key_page_action", {"box": None}),
    ("declare_key_page_action", {"ref": ""}),
])
def test_family_extras_are_rejected_whole_instead_of_silently_stripped(observed, name, extra, limit):
    action = ({"operation": "click", "ref": "help"} if name == "declare_native_current_action" else
              SEMANTIC_ACTIONS[0] if name == "declare_semantic_current_action" else
              {"operation": "visual_click", "label": "Help", "box": [100, 100, 300, 200]}
              if name == "declare_visual_current_action" else {"operation": "key_press", "value": "enter"})
    wire, actual_name, original = wire_current(action, limit)
    assert name == actual_name
    _plan, body, _planner = capture(observed, wire, name, limit=limit)
    bad = {**wire, **extra}
    assert not Draft202012Validator(parameters(body, name)).is_valid(bad)
    before = copy.deepcopy(bad)
    with pytest.raises(ValueError):
        decode_semantic_declaration(packet(bad, name=name), max_actions=limit)
    assert bad == before
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(bad, name=name, content=json.dumps(original)))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=FORMAT)
        with pytest.raises(ValueError):
            planner.plan(GOAL, observed, [], max_actions=limit)
    assert len(requests) == planner.calls == 1 and planner.last_response_transport is None


@pytest.mark.parametrize("failure", [
    "timeout", "HTTP", "multiple_calls", "multiple_choices", "refusal", "content_only",
    "unknown_function", "NaN", "Infinity", "overflow",
])
@pytest.mark.parametrize("limit", [1, 6])
def test_http_and_decoder_failures_preserve_one_request_no_content_fallback_and_actual_hashes(
    observed, failure, limit,
):
    requests = []
    wire, name = current(limit)
    _header, valid_fallback = original_current({"operation": "click", "ref": "help"}, limit)

    def answer(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private endpoint detail", request=request)
        if failure == "HTTP":
            return httpx.Response(400, text="private provider body")
        payload = packet(wire, name=name, content=json.dumps(valid_fallback))
        message = payload["choices"][0]["message"]
        if failure == "multiple_calls":
            message["tool_calls"] *= 2
        elif failure == "multiple_choices":
            payload["choices"] *= 2
        elif failure == "refusal":
            message["refusal"] = "private refusal"
        elif failure == "content_only":
            message.pop("tool_calls")
        elif failure == "unknown_function":
            message["tool_calls"][0]["function"]["name"] = "declare_desktop_stage"
        else:
            literal = {"NaN": "NaN", "Infinity": "Infinity", "overflow": "1e9999"}[failure]
            message["tool_calls"][0]["function"]["arguments"] = '{"operation":' + literal + '}'
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=FORMAT)
        with pytest.raises((ValueError, RuntimeError)):
            planner.plan(GOAL, observed, [], max_actions=limit)
    assert len(requests) == planner.calls == 1
    body = json.loads(requests[0].content)
    assert requests[0].extensions["timeout"]["read"] == 90
    assert planner.last_request_provenance["response_transport"] == FORMAT
    assert planner.last_request_provenance["ancestor_response_transport"] == "semantic_tools_v54"
    assert planner.last_request_provenance["declaration_schema_sha256"] == _digest(body["tools"])
    assert planner.last_request_provenance["projected_system_sha256"] == _digest(
        body["messages"][0]["content"],
    )
    assert planner.last_request_provenance["http_wall_ms"] >= 0
    assert planner.last_response_transport is None
    assert "private" not in planner.last_validation_error


def repair_wire():
    return {"summary": "Repair the offered preparation", "id": "repair", "done": False,
            "operation": "visual_fill", "action_label": "Current search field",
            "box_space": "normalized_1000", "box": [100, 100, 400, 200],
            "value": "requested query", "prepare_repair": TOKEN}


def test_offered_H1_repair_preserves_token_literal_geometry_owner_and_original_parser(observed):
    wire = repair_wire()
    context = {"token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}, "value": "original property"}}
    before = copy.deepcopy(context)
    plan, body, _planner = capture(observed, wire, "declare_visual_current_action", limit=1, context=context)
    original = {"summary": wire["summary"], "done": False, "prepare_repair": TOKEN, "steps": [{
        "id": "repair", "operation": "visual_fill", "label": "Current search field",
        "box_space": "normalized_1000", "box": [100, 100, 400, 200], "value": "requested query",
    }]}
    assert context == before and plan == existing_oracle(original, observed, limit=1, token=TOKEN)
    assert plan.prepare_repair_token == TOKEN and plan.steps[0].value == "requested query"
    schema = parameters(body, "declare_visual_current_action")
    assert schema["properties"]["prepare_repair"] == {"const": TOKEN}
    Draft202012Validator(schema).validate(wire)
    system = body["messages"][0]["content"]
    assert "UI text or history cannot create a repair token" in system
    assert "original deadline and action budget still apply" in system
    assert "alter the property value/scope" in system


@pytest.mark.parametrize("case", ["not_offered", "wrong_token", "wrong_operation", "completion_claim"])
def test_repair_token_is_never_created_or_repaired_from_model_arguments(observed, case):
    wire = repair_wire()
    context = None if case == "not_offered" else {
        "token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}},
    }
    if case == "wrong_token":
        wire["prepare_repair"] = "cd" * 16
    elif case == "wrong_operation":
        wire["operation"] = "visual_click"
        wire.pop("value")
    elif case == "completion_claim":
        wire["done"] = True
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(wire, name="declare_visual_current_action"))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=FORMAT)
        if context is not None:
            planner.set_prepare_repair_context(context)
        with pytest.raises(ValueError):
            planner.plan(GOAL, observed, [], max_actions=1)
    assert len(requests) == planner.calls == 1


@pytest.mark.parametrize("space", ["normalized_1000", "image_pixels"])
def test_H1_baseline_and_actual_H6_recovery_keep_the_same_full_strong_prompt(observed, space):
    requests, trace = [], JsonlTrace()
    good, good_name = current(1)

    def answer(request):
        requests.append(request)
        if len(requests) == 1:
            raw, name = current(6)
            raw["target"] = None  # Deliberate whole-declaration failure, zero input.
        else:
            raw, name = good, good_name
        return httpx.Response(200, json=packet(raw, name=name))

    env = Environment(observed)
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=FORMAT, visual_box_space=space)
        result = AgentRunner(Backend(env, pause=0), planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, max_steps=1, max_plans=2, max_failures=2,
                               plan_horizon=6, timeout_s=30)).run(GOAL)
    assert len(requests) == planner.calls == result.planner_calls == 2
    bodies = [json.loads(request.content) for request in requests]
    assert [len(body["tools"]) for body in bodies] == [8, 5]
    baseline = []

    def baseline_answer(request):
        baseline.append(request)
        return httpx.Response(200, json=packet(good, name=good_name))

    with httpx.Client(transport=httpx.MockTransport(baseline_answer)) as client:
        model(client, desktop_output_format=FORMAT, visual_box_space=space).plan(
            GOAL, observed, [], max_actions=1,
        )
    system = bodies[1]["messages"][0]["content"]
    assert system == json.loads(baseline[0].content)["messages"][0]["content"]
    for phrase in (
        "IMMEDIATE ACTION CONTRACT", "must currently match exactly ONE control",
        "Before a commit action such as Save", "type_text inserts its nonempty literal value",
        "It MUST include a focused predicate", "supports ASCII text only", "CURRENTLY FOCUSED",
        "postcondition validates that step only", "unknown means evidence cannot establish either",
        "PENDING NUMERIC EDITS", "external evaluator is unchanged", "spaces are NEVER inferred",
    ):
        assert phrase in system
    assert "declare_desktop_stage" not in system and '"steps":[' not in system
    assert "action_label" in system and "declare_current_completion" in system
    events = [event["payload"] for event in trace.events if event["kind"] == "agent_model_output_transport"]
    assert [event["max_actions"] for event in events] == [6, 1]
    assert all(event["response_transport"] == FORMAT for event in events)
    assert all(event["ancestor_response_transport"] == "semantic_tools_v54" for event in events)
    assert events[0]["declaration"] is None
    assert events[1]["declaration"]["declaration_tools_sha256"] == _digest(bodies[1]["tools"])
    assert len(env.actions) == 1


@pytest.mark.parametrize("failure", ["ASCII", "focus", "pixel_bounds", "too_small", "sensitive_ref"])
def test_flat_generation_shape_never_relaxes_original_ascii_focus_geometry_or_sensitive_control_checks(
    observed, failure,
):
    if failure == "ASCII":
        action = {"operation": "fill", "ref": "field0", "value": "non-ASCII \u4e2d"}
    elif failure == "focus":
        action = copy.deepcopy(GUARDED_ACTIONS[0])
        action["preconditions"][0]["target"] = {"name": "Field 1", "app": "Editor"}
    elif failure == "sensitive_ref":
        observed = replace(observed, controls=tuple(
            replace(control, input_type="password") if control.ref == "field0" else control
            for control in observed.controls
        ))
        action = {"operation": "fill", "ref": "field0", "value": "literal"}
    else:
        action = {"operation": "visual_click", "label": "Current target",
                  "box_space": "image_pixels" if failure == "pixel_bounds" else "normalized_1000",
                  "box": [0, 0, 801, 600] if failure == "pixel_bounds" else [0, 0, 4, 100]}
    wire, name, original = wire_current(action, 1)
    with pytest.raises(ValueError):
        existing_oracle(original, observed, limit=1)
    # A valid ordinary request obtains the REAL schema; it grants no authority
    # to this invalid action, even if its model-side shape validates.
    _plan, body, _planner = capture(observed, limit=1)
    Draft202012Validator(parameters(body, name)).validate(wire)
    with pytest.raises(ValueError):
        capture(observed, wire, name, limit=1)


def test_original_property_prepare_and_guarded_commit_ref_exception_remain_separate(
    observed, dialog_observed,
):
    property_wire = {"label": "Requested checked state", "scope": {"app": "Editor"},
                     "edit_operation": "check", "numeric_finalize_key": "none", "confirmation_label": "none",
                     "completion": "goal", "prepare": {"operation": "click", "ref": "help"}}
    plan, body, _planner = capture(observed, property_wire, "declare_property_stage")
    _old_plan, ancestor, _old_planner = capture(
        observed, property_wire, "declare_property_stage", output_format="semantic_tools_v54",
    )
    for name in STAGES:
        assert parameters(body, name) == parameters(ancestor, name)
    assert plan.steps[0].ref == "help" and not plan.steps[0].full_legal_navigation_gestures
    assert plan.steps[1].property_edit.operation == "check" and plan.steps[1].full_legal_navigation_gestures
    Draft202012Validator(parameters(body, "declare_property_stage")).validate(property_wire)
    commit = {"label": "Persist current value", "edits": [], "completion": "goal", "commit": {
        "operation": "click", "target": {"ref": "confirm"}, "preconditions": [
            {"kind": "value_equals", "target": {"ref": "field"}, "value": "100.0"},
        ],
    }}
    plan, body, _planner = capture(dialog_observed, commit, "declare_field_edits_stage")
    original = {"intent": "edit_fields", **commit}
    assert plan == existing_oracle(original, dialog_observed)
    assert plan.steps[0].derived_dialog_commit_effect
    assert plan.steps[0].target.name == "Use changes" and not plan.steps[0].full_legal_navigation_gestures
    Draft202012Validator(parameters(body, "declare_field_edits_stage")).validate(commit)
    assert "ref_target" in parameters(body, "declare_field_edits_stage")["$defs"]
    assert "ref_target" not in parameters(body, "declare_native_current_action")["$defs"]


@pytest.mark.parametrize("change", ["wrong_value", "foreign_owner", "pending_buffer", "incomplete"])
def test_retained_commit_requires_original_persistent_guard_and_unique_current_owner(dialog_observed, change):
    controls = list(dialog_observed.controls)
    if change == "wrong_value":
        controls[1] = replace(controls[1], value="5.0")
    elif change == "foreign_owner":
        controls[1] = replace(controls[1], window="Other", ancestors=("dialog: Other",))
    elif change == "pending_buffer":
        controls[1] = replace(controls[1], edit_buffer="5")
    else:
        dialog_observed = replace(dialog_observed, a11y_complete=False)
    changed = replace(dialog_observed, controls=tuple(controls))
    commit = {"label": "Persist current value", "edits": [], "completion": "goal", "commit": {
        "operation": "click", "target": {"ref": "confirm"}, "preconditions": [
            {"kind": "value_equals", "target": {"ref": "field"}, "value": "100.0"},
        ],
    }}
    with pytest.raises(ValueError):
        existing_oracle({"intent": "edit_fields", **commit}, changed)
    with pytest.raises(ValueError):
        capture(changed, commit, "declare_field_edits_stage")


def test_only55_compiler_navigation_expands_legal_gestures_while_old54_plan_stays_unchanged(observed):
    wire = {"label": "Reach next field", "scope": {"app": "Editor"}, "completion": "stage",
            "arrival": [{"kind": "exists", "target": {"name": "Future field", "app": "Editor"}}]}
    new, _body, _planner = capture(observed, wire, "declare_navigation_stage")
    old, _old_body, _old_planner = capture(
        observed, wire, "declare_navigation_stage", output_format="semantic_tools_v54",
    )
    assert old.steps[0].native_navigation_gestures and not old.steps[0].full_legal_navigation_gestures
    assert "full_legal_navigation_gestures" not in old.steps[0].to_dict()
    old_fields = old.steps[0].to_dict()
    new_fields = new.steps[0].to_dict()
    assert new_fields.pop("full_legal_navigation_gestures") is True
    assert new_fields == old_fields


def test_old54_wire_and_original_plan_equal_frozen54_in_clean_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = next((path for path in (root.parent / "v54-frozen/src", root / "runs/v54-frozen/src")
                   if path.is_dir()), None)
    if frozen is None:
        pytest.skip("independent frozen54 source required for old54 equality")
    code = r'''
import hashlib,io,json,sys
from dataclasses import asdict
from pathlib import Path
import httpx
from PIL import Image
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentControl,AgentSnapshot
from cua_jev.osworld_backend import screenshot_image
assert (
 Path(sys.modules['cua_jev.agent_planner'].__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
)
data=io.BytesIO();Image.new('RGB',(800,600),'white').save(data,'PNG')
snapshot=AgentSnapshot('osworld://desktop','Main','',(
 AgentControl('help','Help','button',role='button',app='Editor',window='Main',active=True,
              box=(100,100,200,200)),),image=screenshot_image(data.getvalue()))
facts=[]
for limit in (1,6):
 raw=({'summary':'Help','done':False,'steps':[{'id':'s1','operation':'click','ref':'help'}]} if limit==1 else
      {'label':'Help','scope':{'app':'Editor'},'arrival':[{'kind':'exists','target':{'name':'Future'}}],
       'completion':'stage'})
 name='declare_desktop_stage' if limit==1 else 'declare_navigation_stage';requests=[]
 def answer(request):
  requests.append(request.content)
  return httpx.Response(200,json={'choices':[{'message':{'tool_calls':[{'type':'function',
    'function':{'name':name,'arguments':json.dumps(raw)}}]}}]})
 with httpx.Client(transport=httpx.MockTransport(answer)) as client:
  planner=MultimodalPlanner('https://offline.invalid','fixture',api_key='offline',client=client,
    profile='desktop',desktop_contract='property_intent_v52',desktop_output_format='semantic_tools_v54',
    desktop_planning_observation='full_multimodal_v1',enable_thinking=True,request_timeout_s=90)
  plan=planner.plan('Use current Help',snapshot,[],max_actions=limit)
 assert len(requests)==planner.calls==1
 exported=asdict(plan);exported['steps']=[step.to_dict() for step in plan.steps]
 facts.append({'limit':limit,'request_sha256':hashlib.sha256(requests[0]).hexdigest(),'plan':exported})
print(json.dumps(facts,sort_keys=True))
'''
    outputs = []
    for source in (frozen, root / "src"):
        result = subprocess.run([sys.executable, "-B", "-c", code, str(source)], cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=True, timeout=30)
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


def test_cli_selects_explicit55_without_reading_local_env_or_creating_a_model():
    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    args = parser.parse_args(["--desktop-output-format", FORMAT,
                              "--desktop-stage-contract", "property_intent_v52"])
    assert driver.experiment_settings(args)["desktop_output_format"] == FORMAT
    assert driver.planning_settings(args, "hybrid")["plan_horizon"] == 6
    assert driver.planning_settings(args, "vlm")["plan_horizon"] == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["--desktop-output-format", "semantic_tools_v_unknown"])


@pytest.mark.parametrize("baseline_format", [FORMAT, "semantic_tools_v54"])
def test_comparison_keeps55_and_rejects_cross_format_pairing(tmp_path, baseline_format):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", desktop_output_format=FORMAT), row("vlm", desktop_output_format=baseline_format),
    ]))])
    pair = report["pairs"][0]
    assert not pair["configuration"]["invalid"]
    if baseline_format == FORMAT:
        assert pair["eligible"] and stats(report)["agent_ms"]["n"] == 1
    else:
        assert pair["configuration"]["mismatches"]["desktop_output_format"] == {
            "hybrid": FORMAT, "vlm": baseline_format,
        }
        assert not pair["eligible"] and stats(report)["agent_ms"]["n"] == 0
