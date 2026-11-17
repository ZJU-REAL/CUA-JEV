"""An opted-in root may declare one conditional key without changing old input grammar."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner, _validate_desktop_execution_plan, parse_plan
from cua_jev.desktop_property import property_field_steps, valid_property_stage
from cua_jev.desktop_property_protocol import (
    numeric_property_protocol_instructions,
    property_protocol_instructions,
)


def response(*, key="enter", prepare=True, confirm=True):
    data = {
        "intent": "pursue_property", "label": "Set the requested numeric property",
        "scope": {"app": "Editor"}, "edit": {"operation": "set", "value": "42"},
        "numeric_finalize": {"key": key}, "completion": "goal",
        "goal_conditions": [{"kind": "window_contains", "value": "Main"}],
    }
    if prepare:
        data["prepare"] = {"operation": "click", "ref": "help"}
    if confirm:
        data["confirm"] = {"label": "Submit the requested property"}
    return data


def compile(data, initial, *, limit=6, enabled=True, **overrides):
    settings = {
        "max_actions": limit, "profile": "desktop", "allow_property_goal_conditions": True,
        "allow_property_preparation": True, "allow_property_value_binding": True,
        "allow_observed_property_goal": True, "allow_property_numeric_finalization": enabled,
    }
    return parse_plan(data, initial, **(settings | overrides))


@pytest.mark.parametrize("key", ["enter", "tab"])
@pytest.mark.parametrize("operation", ["fill", "set"])
def test_strategy_keeps_original_stage_preparation_confirm_and_goal_metadata(initial, key, operation):
    data = response(key=key)
    data["edit"]["operation"] = operation
    original = copy.deepcopy(data)
    plan = compile(data, initial, limit=4)
    _validate_desktop_execution_plan(plan, initial, 4)
    assert [step.id for step in plan.steps] == ["prepare", "property", "confirm"]
    assert plan.steps[0].ref == "help"
    stage = plan.steps[1]
    assert stage.depends_on == ("prepare",) and valid_property_stage(stage)
    assert stage.property_edit.numeric_finalize_key == key
    assert stage.property_edit.value == "42" and not stage.target.name and not stage.target.role
    assert stage.to_dict()["property_edit"]["numeric_finalize_key"] == key
    assert plan.steps[2].property_confirmation_of == "property" and plan.complete_property_goal
    assert plan.completion_conditions[0].value == "Main" and data == original


def test_absent_strategy_has_no_new_serialized_field_or_key_on_actual_field_steps(initial):
    data = response(prepare=False, confirm=False)
    del data["numeric_finalize"]
    old, opted_in = compile(data, initial, enabled=False), compile(data, initial)
    assert old == opted_in
    assert old.steps[0].property_edit.numeric_finalize_key == ""
    assert "numeric_finalize_key" not in old.steps[0].to_dict()["property_edit"]
    original = old.steps[0]
    declared = replace(original, property_edit=replace(original.property_edit, numeric_finalize_key="tab"))
    observed = replace(initial, controls=tuple(
        replace(control, box=(100, 100, 300, 200)) if control.editable else control
        for control in initial.controls
    ))
    fields = property_field_steps(declared, observed, observed.controls)
    assert fields and all(actual.property_edit is None for _control, actual in fields)
    assert all(actual.operation == "fill" for _control, actual in fields)


@pytest.mark.parametrize("strategy", [
    None, [], "enter", {}, {"key": ""}, {"key": "Enter"}, {"key": "return"}, {"key": True},
    {"key": ["tab", "enter"]}, {"key": "enter", "repeat": 2},
    {"key": "enter", "target": {"name": "Future"}}, {"key": "tab", "fallback": "enter"},
])
def test_only_one_explicit_allowed_key_is_accepted(initial, strategy):
    with pytest.raises(ValueError, match="numeric_finalize"):
        compile(response() | {"numeric_finalize": strategy}, initial)


@pytest.mark.parametrize("value", ["on", "draft.txt", "NaN", "Infinity", " 42", "42px", "1e1001", "1" * 129])
def test_strategy_cannot_authorize_nonfinite_or_nonnumeric_user_data(initial, value):
    with pytest.raises(ValueError, match="numeric_finalize"):
        compile(response() | {"edit": {"operation": "set", "value": value}}, initial)


@pytest.mark.parametrize("operation", ["check", "uncheck"])
def test_boolean_property_cannot_declare_numeric_finalization(initial, operation):
    with pytest.raises(ValueError, match="numeric_finalize"):
        compile(response() | {"edit": {"operation": operation}}, initial)


@pytest.mark.parametrize("flag", [False, None, 1, "true"])
def test_strategy_requires_strict_opt_in_without_relaxing_old_protocol(initial, flag):
    with pytest.raises(ValueError):
        compile(response(), initial, enabled=flag)


@pytest.mark.parametrize("profile,limit", [("desktop", 1), ("browser", 6)])
def test_strategy_never_enables_single_action_or_browser_planning(initial, profile, limit):
    with pytest.raises(ValueError, match="desktop multi-action"):
        compile(response(), initial, limit=limit, profile=profile)


@pytest.mark.parametrize("prepare,confirm,limit", [(False, False, 2), (True, False, 2), (False, True, 2),
                                                   (True, True, 3)])
def test_strategy_reserves_one_original_step_instead_of_gaining_an_action(initial, prepare, confirm, limit):
    data = response(prepare=prepare, confirm=confirm)
    actual_size = 1 + prepare + confirm
    if actual_size + 1 <= limit:
        assert len(compile(data, initial, limit=limit).steps) == actual_size
    else:
        with pytest.raises(ValueError, match="reserved original logical step"):
            compile(data, initial, limit=limit)
        plan = compile(data, initial, limit=actual_size + 1)
        with pytest.raises(ValueError, match="reserved original logical steps"):
            _validate_desktop_execution_plan(plan, initial, limit)


@pytest.mark.parametrize("key,value", [("space", "42"), (False, "42"), ([], "42"), ("tab", "on")])
def test_invalid_compiler_metadata_is_not_a_legal_property_stage(initial, key, value):
    stage = compile(response(), initial).steps[1]
    altered = replace(stage, property_edit=replace(
        stage.property_edit, numeric_finalize_key=key, value=value,
    ))
    assert not valid_property_stage(altered)


def request(initial, raw, *, contract="property_intent_v49", limit=6, history=None, fenced=False,
            repair_context=None):
    calls = []

    def handle(incoming):
        calls.append(incoming.content)
        answer = json.dumps(raw)
        if fenced:
            answer = "Selected one stage.\n```json\n" + answer + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        planner = MultimodalPlanner(
            "https://offline.invalid", "fixture", api_key="offline", profile="desktop", client=client,
            desktop_contract=contract, visual_box_space="image_pixels", enable_thinking=True,
            request_timeout_s=90,
        )
        if repair_context is not None:
            planner.set_prepare_repair_context(repair_context)
        plan = planner.plan("Apply the user-requested value", initial, history or [], max_actions=limit)
    return plan, calls[0]


def test_new_contract_accepts_explicit_strategy_and_retains_v33_observation_encoding(initial):
    plan, wire = request(initial, response(), fenced=True)
    body = json.loads(wire)
    state = json.loads(body["messages"][1]["content"][0]["text"])
    assert plan.steps[1].property_edit.numeric_finalize_key == "enter"
    assert '"numeric_finalize":{"key":"enter"}' in body["messages"][0]["content"]
    assert '"prepare":{"operation":"click","ref":"CURRENT_REF"}' in body["messages"][0]["content"]
    assert state["screenshot_geometry"]["control_box_space"] == "normalized_1000"
    assert all("box_1000" in control for control in state["snapshot"]["controls"] if "box" in control)
    assert body["enable_thinking"] is True and "thinking_budget" not in body


def test_v33_request_stays_strict_for_new_field(initial):
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(initial, response(), contract="property_intent_v33")


@pytest.mark.parametrize("contract", ["property_intent_v33", "property_intent_v49"])
def test_even_new_contract_single_action_reply_cannot_declare_a_property_strategy(initial, contract):
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(initial, response(), contract=contract, limit=1)


@pytest.mark.parametrize("history", [[], [{"recovery_required": True, "feedback": "Unconfirmed input"}]])
def test_v49_single_action_HTTP_body_is_identical_to_v33(initial, history):
    raw = {"summary": "Use current control", "done": False,
           "steps": [{"id": "s1", "operation": "click", "ref": "help"}]}
    legacy, old_wire = request(initial, raw, contract="property_intent_v33", limit=1, history=history)
    current, new_wire = request(initial, raw, limit=1, history=history)
    assert old_wire == new_wire and legacy == current
    assert "numeric_finalize" not in json.loads(new_wire)["messages"][0]["content"]


def test_v49_prepare_repair_offer_uses_same_single_action_wire_as_v33(initial):
    token = "ab" * 16
    context = {"token": token, "remaining_stage": {"scope": {"app": "Editor"}}}
    raw = {"summary": "Repair current visible input", "done": False, "prepare_repair": token,
           "steps": [{"id": "s1", "operation": "visual_fill", "label": "Current input",
                      "box_space": "normalized_1000", "box": [100, 100, 300, 200], "value": "query"}]}
    old, old_wire = request(initial, raw, contract="property_intent_v33", limit=1, repair_context=context)
    new, new_wire = request(initial, raw, limit=1, repair_context=context)
    assert new.prepare_repair_token == old.prepare_repair_token == token and old_wire == new_wire


@pytest.mark.parametrize("space", ["normalized_1000", "image_pixels"])
@pytest.mark.parametrize("limit", [2, 6, 12])
def test_v49_prompt_stays_bounded_and_preserves_the_direct_preparation_schema(space, limit):
    prompt = numeric_property_protocol_instructions(limit, visual_box_space=space)
    assert len(prompt) < 7000
    offset = prompt.index('"prepare":') + len('"prepare":')
    example, _end = json.JSONDecoder().raw_decode(prompt[offset:])
    assert example == {"operation": "click", "ref": "CURRENT_REF"}
    assert "numeric_finalize" not in property_protocol_instructions(limit, visual_box_space=space)
