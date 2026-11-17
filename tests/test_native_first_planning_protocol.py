"""Explicit native planning preserves the full-image single-action recovery path."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner, VisualInspectionRequested, parse_plan
from cua_jev.desktop_output_schema import desktop_response_format
from cua_jev.desktop_property_protocol import explicit_strategy_property_protocol_instructions

OMITTED = object()


def property_response(key="none", *, operation="set", value="42", prepare=False, confirm=True):
    raw = {
        "intent": "pursue_property",
        "label": "Requested property",
        "scope": {"app": "Editor"},
        "edit": {"operation": operation},
        "numeric_finalize": {"key": key},
        "completion": "goal",
    }
    if operation in {"fill", "set"}:
        raw["edit"]["value"] = value
    if prepare:
        raw["prepare"] = {"operation": "click", "ref": "help"}
    if confirm:
        raw["confirm"] = {"label": "Submit requested property"}
    return raw


def current():
    return {
        "summary": "Use observed control",
        "done": False,
        "steps": [{"id": "s1", "operation": "click", "ref": "help"}],
    }


def planner(client, **options):
    return MultimodalPlanner(
        "https://offline.invalid",
        "fixture",
        api_key="offline",
        client=client,
        profile="desktop",
        desktop_contract=options.pop("desktop_contract", "property_intent_v51"),
        desktop_output_format="json_schema_v50",
        enable_thinking=True,
        request_timeout_s=90,
        desktop_planning_observation=options.pop("desktop_planning_observation", "native_first_v51"),
        **options,
    )


def request(snapshot, raw, *, limit=6, history=None, **options):
    bodies = []

    def answer(req):
        bodies.append(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        model = planner(client, **options)
        try:
            result = model.plan("Apply requested property", snapshot, history or [], max_actions=limit)
        except (ValueError, VisualInspectionRequested) as error:
            result = error
    assert len(bodies) == model.calls == 1
    return result, json.loads(bodies[0]), bodies[0], model


def parse_property(raw, snapshot, *, require=True, limit=6, **overrides):
    return parse_plan(
        raw,
        snapshot,
        profile="desktop",
        max_actions=limit,
        allow_property_goal_conditions=True,
        allow_property_preparation=True,
        allow_property_value_binding=True,
        allow_observed_property_goal=True,
        allow_property_numeric_finalization=True,
        require_numeric_finalize_choice=require,
        **overrides,
    )


@pytest.mark.parametrize("key", ["none", "enter", "tab"])
def test_native_root_sends_complete_AX_without_image_and_keeps_explicit_strategy(initial, key):
    plan, body, _wire, model = request(initial, property_response(key))
    assert not isinstance(plan, Exception) and plan.reusable and plan.complete_property_goal
    assert plan.steps[0].property_edit.numeric_finalize_key == ("" if key == "none" else key)
    assert plan.steps[1].property_confirmation_of == "property"
    content = body["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text"]
    state = json.loads(content[0]["text"])
    assert len(state["snapshot"]["controls"]) == len(initial.controls)
    assert state["planning_input"]["image_uploaded"] is False and initial.image is not None
    assert model.last_request_image_uploaded is False
    assert model.last_request_provenance == {
        "image_uploaded": False,
        "planning_observation": "native_first_v51",
        "max_actions": 6,
        "a11y_complete": True,
    }
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object" and Draft202012Validator(schema).is_valid(property_response(key))


@pytest.mark.parametrize(
    "operation,value", [("fill", "notes.txt"), ("set", "on"), ("check", ""), ("uncheck", "")]
)
def test_none_is_required_but_does_not_change_old_wire_or_authorize_a_key(initial, operation, value):
    explicit = property_response(operation=operation, value=value)
    old = copy.deepcopy(explicit)
    old.pop("numeric_finalize")
    original = parse_property(old, initial, require=False)
    chosen = parse_property(explicit, initial)
    assert chosen == original
    assert [s.to_dict() for s in chosen.steps] == [s.to_dict() for s in original.steps]
    assert all("numeric_finalize_key" not in s.to_dict().get("property_edit", {}) for s in chosen.steps)


def test_old_protocol_stays_optional_and_rejects_none_while_new_requires_a_choice(initial):
    absent = property_response()
    absent.pop("numeric_finalize")
    assert parse_property(absent, initial, require=False).steps[0].property_edit.numeric_finalize_key == ""
    with pytest.raises(ValueError, match="pursue_property requires"):
        parse_property(absent, initial)
    with pytest.raises(ValueError, match="numeric_finalize"):
        parse_property(property_response("none"), initial, require=False)
    for strategy in (None, {"key": None}, {"key": "None"}, {"key": ""}, {"key": "enter", "fallback": "tab"}):
        with pytest.raises(ValueError, match="numeric_finalize"):
            parse_property(property_response() | {"numeric_finalize": strategy}, initial)


@pytest.mark.parametrize("key", ["enter", "tab"])
@pytest.mark.parametrize(
    "operation,value", [("set", "on"), ("fill", "NaN"), ("fill", "1e1001"), ("check", ""), ("uncheck", "")]
)
def test_declared_key_still_requires_original_bounded_numeric_operation(initial, key, operation, value):
    with pytest.raises(ValueError, match="numeric_finalize"):
        parse_property(property_response(key, operation=operation, value=value), initial)


def test_none_does_not_reserve_a_key_slot_but_enter_and_tab_do(initial):
    assert len(parse_property(property_response("none"), initial, limit=2).steps) == 2
    for key in ("enter", "tab"):
        with pytest.raises(ValueError, match="reserved original logical step"):
            parse_property(property_response(key), initial, limit=2)
        assert len(parse_property(property_response(key, prepare=True), initial, limit=4).steps) == 3


@pytest.mark.parametrize(
    "history,limit,incomplete", [([], 1, False), ([{"recovery_required": True}], 6, False), ([], 6, True)]
)
def test_native_single_action_is_the_old_full_image_request_exactly(initial, history, limit, incomplete):
    observed = replace(initial, a11y_complete=False) if incomplete else initial
    new, _new_body, wire, model = request(observed, current(), limit=limit, history=history)
    old, _old_body, old_wire, _ = request(
        observed,
        current(),
        limit=1,
        history=history,
        desktop_contract="property_intent_v49",
        desktop_planning_observation="full_multimodal_v1",
    )
    assert wire == old_wire and new == old and model.last_request_image_uploaded is True
    assert model.last_request_provenance["max_actions"] == 1


def test_pure_effective_horizon_never_calls_the_transport_or_mutates_counters(initial):
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("no planning call"))) as client:
        model = planner(client)
        assert model.effective_max_actions(initial, 6) == 6
        assert model.effective_max_actions(replace(initial, a11y_complete=False), 6) == 1
        assert model.effective_max_actions(replace(initial, a11y_complete=False), 1) == 1
        assert model.calls == 0 and model.last_request_provenance is None


def test_exact_inspect_is_typed_zero_input_and_does_not_make_an_internal_second_request(initial):
    result, body, _wire, model = request(
        initial, {"intent": "inspect_visual", "label": "Need visual context"}
    )
    assert isinstance(result, VisualInspectionRequested) and not isinstance(result, ValueError)
    assert model.calls == 1 and not model.last_validation_error and model.last_request_image_uploaded is False
    assert Draft202012Validator(body["response_format"]["json_schema"]["schema"]).is_valid(
        {"intent": "inspect_visual", "label": "Need visual context"}
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"intent": "inspect_visual"},
        {"intent": "inspect_visual", "label": ""},
        {"intent": "inspect_visual", "label": None},
        {"intent": "inspect_visual", "label": "Need image", "done": True},
        {"intent": "inspect_visual", "label": "Need image", "target": {"ref": "help"}},
    ],
)
def test_inspect_shape_cannot_carry_input_or_completion_claims(initial, raw):
    result, _body, _wire, _ = request(initial, raw)
    assert isinstance(result, ValueError) and not isinstance(result, VisualInspectionRequested)


@pytest.mark.parametrize("limit,mode", [(1, "native_first_v51"), (6, "full_multimodal_v1")])
def test_inspect_is_not_available_in_any_call_with_an_uploaded_image(initial, limit, mode):
    result, body, _wire, model = request(
        initial,
        {"intent": "inspect_visual", "label": "Need image"},
        limit=limit,
        desktop_planning_observation=mode,
    )
    assert isinstance(result, ValueError) and model.last_request_image_uploaded is True
    assert "inspect_visual" not in json.dumps(body["response_format"])


@pytest.mark.parametrize("prepare", [False, True])
def test_text_only_visual_actions_are_rejected_even_with_a_real_local_snapshot_image(initial, prepare):
    action = {"operation": "visual_click", "box": [100, 100, 300, 200], "label": "Visible object"}
    raw = (
        property_response() | {"prepare": action}
        if prepare
        else {"intent": "act_current", "label": "Current object", "action": action}
    )
    result, body, _wire, model = request(initial, raw)
    assert isinstance(result, ValueError)
    assert model.last_validation_error == "visual actions require an image uploaded in this planning request"
    assert initial.image is not None and model.last_request_image_uploaded is False
    assert not Draft202012Validator(body["response_format"]["json_schema"]["schema"]).is_valid(raw)


@pytest.mark.parametrize("space", ["normalized_1000", "image_pixels"])
@pytest.mark.parametrize("limit", [2, 6, 12])
@pytest.mark.parametrize("uploaded", [False, True])
def test_new_root_prompt_is_bounded_and_does_not_infer_a_key(space, limit, uploaded):
    prompt = explicit_strategy_property_protocol_instructions(
        limit, visual_box_space=space, image_uploaded=uploaded
    )
    assert len(prompt) <= 7000 and "enter|tab|none" in prompt and "NO default key" in prompt
    assert ('"prepare":{"operation":"click","ref":"CURRENT_REF"}') in prompt
    assert ("inspect_visual" in prompt) is not uploaded


@pytest.mark.parametrize("value", [None, True, [], {}, "native_first_v50"])
def test_unknown_observation_modes_are_rejected(value):
    with pytest.raises(ValueError, match="native_first_v51"):
        MultimodalPlanner("https://offline.invalid", "fixture", desktop_planning_observation=value)


@pytest.mark.parametrize(
    "contract,output,profile",
    [
        ("property_intent_v49", "json_schema_v50", "desktop"),
        ("property_intent_v51", "unconstrained_v1", "desktop"),
        ("legacy_v23", "unconstrained_v1", "browser"),
    ],
)
def test_native_mode_requires_explicit_new_desktop_contract_and_schema(contract, output, profile):
    with pytest.raises(ValueError, match="native_first_v51"):
        MultimodalPlanner(
            "https://offline.invalid",
            "fixture",
            profile=profile,
            desktop_contract=contract,
            desktop_output_format=output,
            desktop_planning_observation="native_first_v51",
        )


def test_new_schema_does_not_mutate_the_old_root_or_single_action_grammar():
    old = desktop_response_format(6)
    before = copy.deepcopy(old)
    native = desktop_response_format(6, property_contract="property_intent_v51", image_uploaded=False)
    Draft202012Validator.check_schema(native["json_schema"]["schema"])
    assert old == before == desktop_response_format(6)
    assert desktop_response_format(1) == desktop_response_format(1, property_contract="property_intent_v51")
