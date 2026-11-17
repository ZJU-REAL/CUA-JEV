"""Property annotations match existing live binding without changing authority."""

import copy
import importlib
import json
from dataclasses import replace
from uuid import UUID

import httpx
import pytest
import test_program_wire_planner_runner_v64 as repair_checks
from test_clean_high_level_phase_v65 import property_result
from test_prepare_repair_runner import png as png
from test_program_wire_v64 import actual_input
from test_semantic_tools_v57 import current
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_planning_preference_v61 import requested
from test_structured_high_level_declaration_v66 import response

from cua_jev import desktop_plan_repair
from cua_jev import desktop_semantic_tools_v58 as original58
from cua_jev import desktop_semantic_tools_v64 as annotation
from cua_jev.agent_types import AgentControl
from cua_jev.desktop_property import property_field_steps

LEGACY_OPERATIONS = (
    "fill is only for editable text or numeric fields, including creation/renaming with literal text. "
    "set is for a real selectable noneditable option, not typing into an editable field. "
    "check/uncheck are only for actual checkbox/radio controls. Use current evidence; "
    "do not invent a role or force a property declaration when it does not fit."
)
FORMATS = (
    "desktop_semantic_tools_v64", "desktop_semantic_json_v66",
)


def without_descriptions(value):
    if isinstance(value, dict):
        return {key: without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


def descriptions(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "description" and isinstance(item, str):
                yield item
            else:
                yield from descriptions(item)
    elif isinstance(value, list):
        for item in value:
            yield from descriptions(item)


def test_only_actual_schema_annotations_change_and_literal_payloads_stay_opaque():
    literal = {"description": LEGACY_OPERATIONS, "properties": {"x": {"description": LEGACY_OPERATIONS}}}
    schema = {
        "type": "object", "description": "Before " + LEGACY_OPERATIONS + " After",
        "properties": {"description": {"type": "string", "description": LEGACY_OPERATIONS}},
        "$defs": {"nested": {"description": LEGACY_OPERATIONS}},
        "const": copy.deepcopy(literal), "enum": [copy.deepcopy(literal)],
        "default": copy.deepcopy(literal), "examples": [copy.deepcopy(literal)],
        "unknown_data": copy.deepcopy(literal),
    }
    tools = [{"function": {"name": "declare_property_stage", "description": LEGACY_OPERATIONS,
                           "parameters": copy.deepcopy(schema)}}]
    annotation._normal_property_operation_annotations(tools)
    function = tools[0]["function"]
    assert function["description"] == annotation.NORMAL_PROPERTY_OPERATIONS
    actual = function["parameters"]
    assert actual["description"] == "Before " + annotation.NORMAL_PROPERTY_OPERATIONS + " After"
    assert actual["properties"]["description"]["description"] == annotation.NORMAL_PROPERTY_OPERATIONS
    assert actual["$defs"]["nested"]["description"] == annotation.NORMAL_PROPERTY_OPERATIONS
    for key in ("const", "enum", "default", "examples", "unknown_data"):
        assert actual[key] == schema[key]
    assert actual["type"] == schema["type"]


@pytest.mark.parametrize("module_name", FORMATS)
def test_every_actual_H6_property_description_agrees_and_constraints_stay_exact(
    observed, monkeypatch, module_name,
):
    body, options = actual_input(observed, monkeypatch)
    codec = importlib.import_module("cua_jev." + module_name)
    actual, provenance = codec.encode_semantic_declaration(body, **options)
    revised = annotation.NORMAL_PROPERTY_OPERATIONS
    with monkeypatch.context() as patch:
        patch.setattr(annotation, "_normal_property_operation_annotations", lambda _tools: None)
        old, _ = codec.encode_semantic_declaration(body, **options)

    schema = (actual["response_format"] if "response_format" in actual else actual["tools"])
    annotated = [text for text in descriptions(schema)
                 if "fill is only for editable text or numeric fields" in text]
    assert len(annotated) >= 2
    assert all(revised in text for text in annotated)
    assert "not typing into an editable field" not in json.dumps(actual)
    for phrase in (
        "fills an editable text or numeric field", "selectable noneditable option",
        "binds on/off", "under the original guards", "widget type is unknown",
        "check/uncheck are only for actual checkbox/radio controls",
    ):
        assert phrase in revised

    # Complete bodies match after removing schema annotations alone, including
    # all original system/user text, literals, images and request settings.
    restored = without_descriptions(copy.deepcopy(actual))
    assert restored == without_descriptions(old)
    assert original58.PROPERTY_OPERATIONS == LEGACY_OPERATIONS
    metadata = provenance.get("internal_template", {}).get("provenance", provenance)
    assert metadata["property_operation_annotation_variant"] == "v68_live_adaptive_set_documentation"
    if "response_format" in actual:
        assert len(actual["response_format"]["json_schema"]["schema"]["oneOf"]) == 6


@pytest.fixture
def widgets(observed):
    owner = {"app": "Editor", "window": "Main", "active": True}
    option = AgentControl(
        "option", "Requested option", "button", role="cell", selectable=True,
        box=(100, 300, 400, 380), **owner,
    )
    return replace(observed, controls=(
        observed.controls[0], observed.controls[-1], observed.controls[-2], option,
    ))


def structured_property(snapshot, value):
    raw = property_result(edit_value=value)
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=response({"tool": "declare_property_stage", "arguments": raw}))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt="semantic_json_v66")
        plan = planner.plan("Set the requested property", snapshot, [], max_actions=6)
    assert len(requests) == planner.calls == 1
    assert plan.steps[0].property_edit.operation == "set"
    assert plan.steps[0].property_edit.bind_boolean_states
    return plan.steps[0]


@pytest.mark.parametrize("value,ref,operation,effect", [
    ("requested literal", "field", "fill", "value_equals"),
    ("Requested option", "option", "click", "selected"),
    ("on", "check", "check", "checked"),
    ("off", "check", "uncheck", "unchecked"),
])
def test_actual_structured_set_binds_existing_editor_option_and_boolean_paths(
    widgets, value, ref, operation, effect,
):
    if value == "off":
        widgets = replace(widgets, controls=tuple(
            replace(control, checked=True) if control.ref == "check" else control
            for control in widgets.controls
        ))
    stage = structured_property(widgets, value)
    choices = property_field_steps(stage, widgets, widgets.controls)
    bound, = [step for control, step in choices if control.ref == ref]
    assert bound.operation == operation and bound.postconditions[0].kind == effect
    assert bound.preconditions and bound.recovery == "observe"
    assert bound.value == (value if operation == "fill" else "")
    assert stage.property_edit.numeric_finalize_key == ""


def test_adaptive_annotation_does_not_create_ambiguous_option_or_unknown_toggle_input(widgets):
    option = widgets.controls[-1]
    ambiguous = replace(widgets, controls=(*widgets.controls, replace(option, ref="duplicate")))
    stage = structured_property(ambiguous, "Requested option")
    choices = property_field_steps(stage, ambiguous, ambiguous.controls)
    assert not any(control.ref in {"option", "duplicate"} for control, _step in choices)
    unknown = replace(widgets.controls[-2], role="menuitem", tag="button", input_type="")
    frame = replace(widgets, controls=(unknown,))
    stage = structured_property(frame, "off")
    assert property_field_steps(stage, frame, frame.controls) == ()


@pytest.mark.parametrize("fmt", ["semantic_tools_v58", "semantic_json_v66"])
@pytest.mark.parametrize("formal", [False, True])
def test_actual_H1_bytes_are_identical_to_legacy_annotation(observed, monkeypatch, fmt, formal):
    raw = ({"label": "Observed editor", "conditions": [{"kind": "text_contains", "value": "Editor"}]}
           if formal else current(1))
    tool = "declare_completion" if formal else "declare_native_current_action"
    options = {"whole_goal_recheck": True} if formal else {}
    actual, request, _ = requested(observed, raw, tool, fmt=fmt, limit=1, planning_options=options)
    with monkeypatch.context() as patch:
        patch.setattr(annotation, "_normal_property_operation_annotations", lambda _tools: None)
        old, old_request, _ = requested(
            observed, raw, tool, fmt=fmt, limit=1, planning_options=options,
        )
    assert request.content == old_request.content and actual == old


@pytest.mark.parametrize("limit", [1, 6])
def test_actual_formal_request_is_byte_exact_original58_and_source67(observed, monkeypatch, limit):
    raw = {"label": "Observed editor", "conditions": [{"kind": "text_contains", "value": "Editor"}]}
    captures = [requested(
        observed, raw, "declare_completion", fmt=fmt, limit=limit,
        planning_options={"whole_goal_recheck": True},
    ) for fmt in ("semantic_tools_v58", "semantic_json_v66")]
    assert captures[0][1].content == captures[1][1].content and captures[0][0] == captures[1][0]
    with monkeypatch.context() as patch:
        patch.setattr(annotation, "_normal_property_operation_annotations", lambda _tools: None)
        old = requested(observed, raw, "declare_completion", fmt="semantic_json_v66", limit=limit,
                        planning_options={"whole_goal_recheck": True})
    assert old[1].content == captures[1][1].content and old[0] == captures[1][0]


def test_actual_prepare_repair_H1_body_is_exact58_and_unchanged_by_annotation(png, monkeypatch):
    captured = []
    original_factory = repair_checks.make_planner

    def factory(client, *, fmt):
        client.event_hooks["request"].append(lambda request: captured.append(request.content))
        return original_factory(client, fmt=fmt)

    monkeypatch.setattr(repair_checks, "make_planner", factory)
    monkeypatch.setattr(repair_checks, "FORMAT", "semantic_json_v66")
    fixed_token = UUID("12345678123456781234567812345678")
    monkeypatch.setattr(desktop_plan_repair.uuid, "uuid4", lambda: fixed_token)
    repair_checks.test_real_prepare_repair_H1_actual_request_is_exact58(png)
    with monkeypatch.context() as patch:
        patch.setattr(annotation, "_normal_property_operation_annotations", lambda _tools: None)
        repair_checks.test_real_prepare_repair_H1_actual_request_is_exact58(png)
    assert len(captured) == 4 and captured[0] == captured[1] == captured[2] == captured[3]
