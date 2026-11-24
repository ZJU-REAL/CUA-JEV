"""Schema shape must retain existing v49 native, visual and max-one grammar.

These offline oracle cases first pass the unchanged parser and execution-plan
validator. Schema acceptance supplies no input authority or success evidence.
"""

from __future__ import annotations

import copy
import io
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.osworld_backend import screenshot_image

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture
def observed():
    data = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(data, "PNG")
    controls = [
        AgentControl("root", "Main", "container", role="frame", app="Editor", window="Main",
                     active=True, box=(0, 0, 1000, 1000)),
        AgentControl("help", "Help", "button", role="button", app="Editor", window="Main",
                     active=True, box=(800, 500, 900, 600)),
        AgentControl("check", "Wrap", "input", role="checkbox", input_type="checkbox",
                     app="Editor", window="Main", active=True, box=(600, 100, 700, 200)),
    ]
    controls.extend(AgentControl(
        f"field{i}", f"Field {i}", "input", role="textbox", input_type="text", value="old",
        editable=True, focused=i == 0, app="Editor", window="Main", active=True,
        box=(100, 50 + i * 60, 400, 100 + i * 60),
    ) for i in range(8))
    return AgentSnapshot("osworld://desktop", "Main", "Editor Main", tuple(controls),
                         image=screenshot_image(data.getvalue()))


def production_schema(limit=6, *, token=None):
    from cua_jev.desktop_output_schema import desktop_response_format

    return desktop_response_format(limit, prepare_repair_token=token)["json_schema"]["schema"]


def existing_oracle(raw, observed, *, limit=6, token=None):
    options = {"profile": "desktop", "max_actions": limit}
    if limit == 1:
        options["expected_prepare_repair_token"] = token
    else:
        options.update(
            allow_property_goal_conditions=True, allow_advisory_completion=True,
            inherit_current_action_label=True, allow_guarded_commit_only=True,
            allow_property_preparation=True, allow_property_value_binding=True,
            allow_dialog_commit_effect=True, allow_observed_property_goal=True,
            allow_current_commit_refs=True, allow_property_numeric_finalization=True,
        )
    plan = parse_plan(copy.deepcopy(raw), observed, **options)
    _validate_desktop_execution_plan(plan, observed, limit)
    return plan


def retained(raw, observed, *, limit=6, token=None):
    plan = existing_oracle(raw, observed, limit=limit, token=token)
    schema = production_schema(limit, token=token)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(raw)
    return plan


CURRENT_ACTIONS = [
    {"operation": "click", "ref": "help"},
    {"operation": "click", "ref": "help", "button": "right", "click_count": 1},
    {"operation": "click", "ref": "help", "click_count": 2},
    {"operation": "fill", "ref": "field0", "value": "ASCII text"},
    {"operation": "check", "ref": "check"},
    {"operation": "uncheck", "ref": "check"},
    {"operation": "key_press", "value": "enter"},
    {"operation": "key_press", "value": "ENTER"},
    {"operation": "hotkey", "value": "ctrl+shift+s"},
    {"operation": "type_text", "value": "typed text"},
    {"operation": "press", "value": "Enter"},
    {"operation": "scroll", "value": "down"},
    {"operation": "back"},
    {"operation": "wait"},
    {"operation": "wait", "value": "1000"},
    {"operation": "visual_click", "label": "Current control", "box": [100, 100, 300, 200]},
    {"operation": "visual_fill", "label": "Current field", "box": [100, 100, 300, 200], "value": "x"},
    {"operation": "visual_click", "label": "Current control", "box_space": "image_pixels",
     "box": [80, 60, 240, 120]},
]


@pytest.mark.parametrize("action", CURRENT_ACTIONS, ids=lambda x: x["operation"])
def test_all_legal_current_native_visual_key_and_page_actions_remain_in_v49_root(observed, action):
    retained({"intent": "act_current", "label": "One grounded current action", "action": action}, observed)


@pytest.mark.parametrize("action", CURRENT_ACTIONS, ids=lambda x: x["operation"])
def test_max_one_keeps_complete_baseline_actions_without_forced_semantic_guards(observed, action):
    retained({"summary": "One grounded action", "done": False, "steps": [{"id": "s1", **action}]},
             observed, limit=1)


@pytest.mark.parametrize("limit", [1, 6])
def test_native_explicit_null_defaults_are_legal_for_both_output_forms(observed, limit):
    action = {"operation": "click", "ref": "help", "target": None, "box": None,
              "value": "", "label": "", "preconditions": [], "postconditions": [], "recovery": "replan"}
    raw = ({"summary": "Current", "done": False, "steps": [{"id": 0, "navigation": False,
                                                             "depends_on": [], **action}]}
           if limit == 1 else {"intent": "act_current", "label": "Current", "action": action})
    retained(raw, observed, limit=limit)


@pytest.mark.parametrize("limit", [1, 6])
def test_guarded_keyboard_and_all_condition_shapes_remain_available(observed, limit):
    field = {"name": "Field 0", "role": "textbox", "app": "Editor", "window": "Main"}
    action = {"operation": "key_press", "value": "enter",
              "preconditions": [{"kind": "focused", "target": field}],
              "postconditions": [{"kind": "value_equals", "target": field, "value": "new"},
                                 {"kind": "text_contains", "value": "Saved", "target": None}],
              "recovery": "observe"}
    raw = ({"summary": "Guarded Enter", "done": False, "steps": [{"id": "enter", **action}],
            "completion_conditions": [{"kind": "window_contains", "value": "Main"}]}
           if limit == 1 else {"intent": "act_current", "label": "Guarded Enter", "action": action})
    retained(raw, observed, limit=limit)


@pytest.mark.parametrize("conditions", [None, [], [{"kind": "window_contains", "value": "Main"}]])
def test_max_one_advisory_done_preserves_absent_empty_or_explicit_conditions(observed, conditions):
    raw = {"summary": "Completion hint", "done": True, "steps": []}
    if conditions is not None:
        raw["completion_conditions"] = conditions
    plan = retained(raw, observed, limit=1)
    assert plan.done and not plan.steps


def test_max_one_done_with_an_action_is_still_legal_and_cannot_suppress_that_action(observed):
    plan = retained({"summary": "Current input", "done": True,
                     "steps": [{"id": 42, "operation": "click", "ref": "help"}]}, observed, limit=1)
    assert not plan.done and len(plan.steps) == 1


def test_dynamic_prepare_repair_token_is_exact_and_only_available_in_its_current_context(observed):
    token = "a" * 32
    raw = {"summary": "Repair current search", "done": False, "prepare_repair": token,
           "steps": [{"id": "repair", "operation": "visual_fill", "label": "Current search field",
                      "box": [100, 100, 400, 200], "value": "specific setting"}]}
    retained(raw, observed, limit=1, token=token)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(production_schema(1)).validate(raw)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(production_schema(1, token="b" * 32)).validate(raw)
    with pytest.raises(ValueError):
        existing_oracle(raw, observed, limit=1, token="b" * 32)


def test_all_five_v49_root_intents_and_numeric_policy_are_retained(observed):
    cases = [
        {"intent": "pursue_property", "label": "Set requested value", "scope": {"app": "Editor"},
         "edit": {"operation": "set", "value": "100"}, "numeric_finalize": {"key": "enter"},
         "confirm": {"label": "Submit requested property"}, "completion": "goal"},
        {"intent": "navigate", "label": "Open requested settings", "scope": {"app": "Editor"},
         "arrival": [{"kind": "exists", "target": {"role": "dialog", "name": "Settings"}}]},
        {"intent": "edit_fields", "label": "Edit current field", "edits": [{"operation": "fill",
         "target": {"name": "Field 0", "app": "Editor"}, "value": "new"}]},
        {"intent": "act_current", "label": "Click current Help",
         "action": {"operation": "click", "ref": "help"}},
        {"intent": "complete", "label": "Completion hint"},
        {"intent": "complete", "label": "Observed completion",
         "conditions": [{"kind": "window_contains", "value": "Main"}]},
    ]
    for raw in cases:
        retained(raw, observed)


def test_larger_existing_horizon_can_still_edit_more_than_six_current_fields(observed):
    raw = {"intent": "edit_fields", "label": "Edit seven current fields", "edits": [
        {"operation": "fill", "target": {"name": f"Field {i}", "app": "Editor"}, "value": f"new{i}"}
        for i in range(7)
    ]}
    assert len(retained(raw, observed, limit=12).steps) == 7


def test_property_explicit_empty_goal_conditions_remain_rejected_by_original_parser_and_schema(observed):
    raw = {"intent": "pursue_property", "label": "Set requested value", "scope": {"app": "Editor"},
           "edit": {"operation": "fill", "value": "100"}, "goal_conditions": []}
    with pytest.raises(ValueError):
        existing_oracle(raw, observed)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(production_schema()).validate(raw)


def test_explicit_guarded_commit_keyboard_nullable_target_is_still_legal(observed):
    retained({
        "intent": "edit_fields", "label": "Edit and save current field",
        "edits": [{"operation": "fill", "target": {"name": "Field 0", "app": "Editor"}, "value": "new"}],
        "commit": {"operation": "key_press", "value": "enter", "target": None,
                   "postconditions": [{"kind": "window_contains", "value": "Saved"}]},
    }, observed)


def test_current_commit_refs_and_derived_dialog_effect_remain_legal(observed):
    owner = {"app": "Editor", "window": "Options", "active": True}
    observed = replace(observed, title="Options", controls=(
        AgentControl("dialog", "Options", "container", role="dialog", box=(0, 0, 1000, 1000), **owner),
        AgentControl("field", "Kept revisions", "input", role="spinbutton", input_type="number",
                     value="100.0", edit_buffer="100", focused=True, editable=True,
                     box=(100, 100, 300, 200), ancestors=("dialog: Options",), **owner),
        AgentControl("ok", "Use changes", "button", role="button", box=(600, 800, 800, 900),
                     ancestors=("dialog: Options",), **owner),
    ))
    raw = {"intent": "edit_fields", "label": "Submit observed setting", "edits": [], "completion": "goal",
           "commit": {"operation": "click", "target": {"ref": "ok"},
                      "preconditions": [{"kind": "value_equals", "target": {"ref": "field"},
                                         "value": "100.0"}]}}
    plan = retained(raw, observed)
    assert plan.steps[0].derived_dialog_commit_effect
    assert plan.steps[0].target.name == "Use changes"


@pytest.mark.parametrize("action", CURRENT_ACTIONS[:15], ids=lambda x: x["operation"])
def test_property_preparation_preserves_same_current_native_and_keyboard_forms(observed, action):
    retained({"intent": "pursue_property", "label": "Set requested setting", "scope": {"app": "Editor"},
              "edit": {"operation": "set", "value": "100"}, "prepare": action}, observed)
