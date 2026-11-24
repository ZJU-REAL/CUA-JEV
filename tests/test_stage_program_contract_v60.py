'Regression tests for test stage program contract v60.'

import copy
import hashlib
import json
from dataclasses import replace

import pytest
from test_deferred_property_confirmation import current_dialog
from test_explicit_stage_completion_protocol import parse as original_stage_parse
from test_semantic_tools_v57 import observed as observed

from cua_jev.agent_planner import _validate_desktop_execution_plan
from cua_jev.desktop_stage_program import (
    StageProgramPlan,
    activate_program_stage,
    make_program_plan,
    parse_stage_program,
)

FUTURE = {"name": "Requested query", "role": "textbox", "app": "Editor", "window": "Query panel"}


def navigation(label="Reach requested query", target=None):
    return {"id": "reach", "tool": "declare_navigation_stage", "arguments": {
        "label": label, "scope": {"app": "Editor"}, "completion": "stage",
        "arrival": [{"kind": "exists", "target": target or FUTURE}],
    }}


def fields():
    return {"id": "edit", "tool": "declare_field_edits_stage", "arguments": {
        "label": "Enter the requested query", "completion": "stage",
        "edits": [{"operation": "fill", "target": FUTURE, "value": "requested literal"}],
    }}


def declaration():
    return {"label": "Reach and edit the requested query", "stages": [navigation(), fields()]}


def parse_original(raw, snapshot, horizon):
    parsed = original_stage_parse(raw, snapshot, policy=True, limit=horizon)
    _validate_desktop_execution_plan(parsed, snapshot, horizon)
    return parsed


def query_panel(observed):
    owner = replace(observed.controls[0], ref="query-root", label="Query panel", window="Query panel")
    field = replace(observed.controls[-1], ref="new-query", label="Requested query", window="Query panel",
                    ancestors=("frame: Query panel",), value="old", focused=True)
    return replace(observed, title="Query panel", controls=(owner, field))


def test_future_field_absent_initially_is_not_compiled_until_actual_new_panel(observed):
    raw = declaration()
    before = copy.deepcopy(raw)
    program = parse_stage_program(raw, max_actions=6)
    calls = []

    def original(raw, snapshot, horizon):
        calls.append((copy.deepcopy(raw), snapshot, horizon))
        return parse_original(raw, snapshot, horizon)

    wrapped = make_program_plan(program, observed, parse_stage=original)
    assert isinstance(wrapped, StageProgramPlan) and len(calls) == 1
    assert calls[0][0]["intent"] == "navigate" and calls[0][1] is observed
    assert not any(control.label == FUTURE["name"] for control in observed.controls)
    current = query_panel(observed)
    next_plan = activate_program_stage(program, 1, current, remaining_actions=1, parse_stage=original)
    assert len(calls) == 2 and calls[1][1] is current
    assert calls[1][2] == 6  # remaining1 must not turn the parser into strong H1.
    assert next_plan == parse_original(program.stages[1].parser_object(), current, 6)
    assert next_plan.reusable and next_plan.steps[0].value == "requested literal"
    assert next_plan.steps[0].target.name == FUTURE["name"]
    assert not next_plan.done and not next_plan.completion_conditions
    assert raw == before


@pytest.mark.parametrize("drift", ["missing", "duplicate", "app", "window", "incomplete"])
def test_future_field_binding_reuses_original_native_validation_without_fallback(observed, drift):
    program = parse_stage_program(declaration(), max_actions=6)
    current = query_panel(observed)
    if drift == "missing":
        current = replace(current, controls=current.controls[:1])
    elif drift == "duplicate":
        current = replace(current, controls=(
            *current.controls, replace(current.controls[-1], ref="duplicate")))
    elif drift in {"app", "window"}:
        current = replace(current, controls=(
            current.controls[0], replace(current.controls[-1], **{drift: "Elsewhere"})))
    else:
        current = replace(current, a11y_complete=False)
    with pytest.raises(ValueError):
        activate_program_stage(program, 1, current, remaining_actions=5, parse_stage=parse_original)


@pytest.mark.parametrize("drift", ["missing", "app", "window", "incomplete"])
def test_false_or_unknown_entry_does_not_even_call_current_parser(observed, drift):
    raw = declaration()
    raw["stages"][1]["entry_conditions"] = [{"kind": "exists", "target": FUTURE}]
    program = parse_stage_program(raw, max_actions=6)
    current = query_panel(observed)
    if drift == "missing":
        current = observed
    elif drift == "incomplete":
        current = replace(current, a11y_complete=False)
    else:
        current = replace(current, controls=(
            current.controls[0], replace(current.controls[-1], **{drift: "Elsewhere"})))
    calls = []
    with pytest.raises(ValueError):
        activate_program_stage(program, 1, current, remaining_actions=5,
                               parse_stage=lambda *args: calls.append(args))
    assert calls == []


def test_property_and_guarded_confirmation_are_compiled_only_in_actual_dialog(observed):
    raw = declaration()
    raw["stages"][1] = {"id": "set-value", "tool": "declare_property_stage", "arguments": {
        "label": "Set requested limit", "scope": {"app": "Editor", "window": "Preferences"},
        "edit_operation": "set", "edit_value": "50", "numeric_finalize_key": "none",
        "confirmation_label": "Confirm requested property", "completion": "stage",
    }}
    program = parse_stage_program(raw, max_actions=6)
    current = current_dialog(observed)
    plan = activate_program_stage(program, 1, current, remaining_actions=2, parse_stage=parse_original)
    assert plan == parse_original(program.stages[1].parser_object(), current, 6)
    assert plan.steps[0].property_edit.value == "50"
    assert plan.steps[1].property_confirmation_of == plan.steps[0].id
    assert not plan.complete_property_goal and not plan.completion_conditions
    remaining_one = activate_program_stage(program, 1, current, remaining_actions=1,
                                           parse_stage=parse_original)
    assert remaining_one == plan  # Do not reserve an optional/unneeded commit as certain input.


@pytest.mark.parametrize("field,value", [
    ("ref", "old-query"), ("box", [1, 2, 3, 4]), ("box_space", "image_pixels"),
    ("prepare", {"operation": "key_press", "value": "ENTER"}),
    ("action", {"operation": "click", "ref": "old"}),
])
def test_future_stage_current_authority_is_rejected_recursively(field, value):
    raw = declaration()
    raw["stages"][1]["arguments"]["edits"][0][field] = value
    with pytest.raises(ValueError):
        parse_stage_program(raw, max_actions=6)


@pytest.mark.parametrize("change", [
    "goal", "goal_conditions", "current_tool", "duplicate_id", "one_stage", "too_many",
    "supplied_slots", "empty_arrival",
])
def test_declarations_cannot_bypass_stage_order_scope_or_compiler_budget(change):
    raw = declaration()
    if change == "goal":
        raw["stages"][1]["arguments"]["completion"] = "goal"
    elif change == "goal_conditions":
        raw["stages"][1]["arguments"]["goal_conditions"] = [{"kind": "exists", "target": FUTURE}]
    elif change == "current_tool":
        raw["stages"][1]["tool"] = "declare_native_current_action"
    elif change == "duplicate_id":
        raw["stages"][1]["id"] = raw["stages"][0]["id"]
    elif change == "one_stage":
        raw["stages"] = raw["stages"][:1]
    elif change == "too_many":
        raw["stages"] = [dict(navigation(), id=f"stage{i}") for i in range(7)]
    elif change == "empty_arrival":
        raw["stages"][0]["arguments"]["arrival"] = []
    else:
        raw["stages"][1]["declared_slots"] = 0
    with pytest.raises(ValueError):
        parse_stage_program(raw, max_actions=6)


@pytest.mark.parametrize("remaining", [-1, True, "6", 7])
def test_invalid_remaining_never_creates_a_new_horizon(observed, remaining):
    program = parse_stage_program(declaration(), max_actions=6)
    called = []
    with pytest.raises(ValueError):
        activate_program_stage(program, 1, query_panel(observed), remaining_actions=remaining,
                               parse_stage=lambda *args: called.append(args))
    assert called == []


def test_optional_numeric_and_confirmation_declarations_do_not_reserve_real_attempts():
    stages = [{"id": f"property-{index}", "tool": "declare_property_stage", "arguments": {
        "label": f"Requested property {index}", "scope": {"app": "Editor"},
        "edit_operation": "set", "edit_value": "50", "numeric_finalize_key": "enter",
        "confirmation_label": "Confirm requested property", "completion": "stage",
    }} for index in range(3)]
    program = parse_stage_program({"label": "Requested independent properties", "stages": stages},
                                  max_actions=6)
    assert len(program.stages) == 3 and program.max_actions == 6
    assert sum(stage.declared_slots for stage in program.stages) > 6
    # This is inert metadata, not a proof of attempts: fields may already be
    # committed and require neither the numeric contingency nor a GUI commit.


def test_original_declaration_is_immutable_and_independently_recomputable():
    raw = declaration()
    before = copy.deepcopy(raw)
    program = parse_stage_program(raw, max_actions=6)
    canonical = json.dumps(before, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert program.declaration_json == canonical
    assert program.declaration_sha256 == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert program.to_dict()["source_declaration"] == before
    raw["stages"][1]["arguments"]["edits"][0]["value"] = "later mutation"
    extracted = program.stages[1].arguments()
    extracted["edits"][0]["value"] = "caller mutation"
    assert program.stages[1].arguments()["edits"][0]["value"] == "requested literal"
    assert program.to_dict()["source_declaration"] == before


def test_remaining_zero_still_allows_original_parser_for_readonly_terminal_state(observed):
    raw = declaration()
    program = parse_stage_program(raw, max_actions=6)
    current = query_panel(observed)
    current = replace(current, controls=(
        current.controls[0], replace(current.controls[1], value="requested literal")))
    calls = []

    def original(raw, snapshot, horizon):
        calls.append((snapshot, horizon))
        return parse_original(raw, snapshot, horizon)

    plan = activate_program_stage(program, 1, current, remaining_actions=0, parse_stage=original)
    assert calls == [(current, 6)] and plan.reusable
    # Runner must prove actual read-only terminal facts here; compilation itself
    # sends no input and grants no seventh attempt or fresh horizon.
