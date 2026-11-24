'Regression tests for test stage program recovery contract v60.'

import copy
import json
from dataclasses import replace

import pytest
from test_deferred_property_confirmation import current_dialog
from test_semantic_tools_v57 import current
from test_semantic_tools_v57 import observed as observed
from test_stage_program_contract_v60 import FUTURE, declaration, parse_original, query_panel
from test_stage_program_planner_v60 import captured

from cua_jev.desktop_semantic_tools_v60 import FORMAT, PROGRAM_TOOL
from cua_jev.desktop_semantics import conditions_hold
from cua_jev.desktop_stage_program import activate_program_stage, parse_stage_program


def test_omission_inherits_original_observe_permission_without_changing_original_source():
    raw = declaration()
    original = copy.deepcopy(raw)
    program = parse_stage_program(raw, max_actions=6)
    assert all(stage.recovery == "observe" for stage in program.stages)
    assert all(stage.to_dict()["recovery"] == "observe" for stage in program.stages)
    assert program.to_dict()["source_declaration"] == original == raw
    assert all("recovery" not in stage for stage in program.to_dict()["source_declaration"]["stages"])


def test_wire_schema_default_is_inherit_observe_on_initial_and_future_member_branches(observed):
    _, request, _ = captured(observed, fmt=FORMAT, limit=6,
                             raw=current(6), tool="declare_native_current_action")
    body = json.loads(request.content)
    schema = next(tool["function"]["parameters"] for tool in body["tools"]
                  if tool["function"]["name"] == PROGRAM_TOOL)
    stages = schema["properties"]["stages"]
    branches = [*stages["prefixItems"][0]["anyOf"], *stages["items"]["anyOf"]]
    assert len(branches) == 6
    for branch in branches:
        policy = branch["properties"]["recovery"]
        assert policy["enum"] == ["observe", "replan"] and policy["default"] == "observe"
        assert "recovery" not in branch["required"]


@pytest.mark.parametrize("policy", ["observe", "replan"])
def test_outer_recovery_never_rewrites_original_property_and_confirmation_templates(observed, policy):
    raw = declaration()
    raw["stages"][1] = {
        "id": "set-limit", "tool": "declare_property_stage", "recovery": policy,
        "arguments": {
            "label": "Set requested limit", "scope": {"app": "Editor", "window": "Preferences"},
            "edit_operation": "set", "edit_value": "50", "numeric_finalize_key": "none",
            "confirmation_label": "Confirm requested property", "completion": "stage",
        },
    }
    program = parse_stage_program(raw, max_actions=6)
    stage = program.stages[1]
    snapshot = current_dialog(observed)
    compiled = activate_program_stage(program, 1, snapshot, remaining_actions=2,
                                      parse_stage=parse_original)
    assert stage.recovery == policy
    assert compiled == parse_original(stage.parser_object(), snapshot, 6)
    assert compiled.steps[0].property_edit and compiled.steps[0].recovery == "observe"
    assert compiled.steps[1].property_confirmation_of == compiled.steps[0].id
    assert compiled.steps[1].recovery == "observe"
    # Explicit outer replan suppresses an optional runner polling branch. It
    # must not replace these original parser-required template fields.


@pytest.mark.parametrize("policy", ["observe", "replan"])
@pytest.mark.parametrize("entry_result", ["false", "unknown"])
def test_entry_recovery_never_grants_a_parser_call_or_observation_ticket(observed, policy, entry_result):
    raw = declaration()
    raw["stages"][1]["recovery"] = policy
    raw["stages"][1]["entry_conditions"] = [
        {"kind": "value_equals", "target": FUTURE, "value": "required admission value"},
    ]
    program = parse_stage_program(raw, max_actions=6)
    snapshot = query_panel(observed) if entry_result == "false" else observed
    called = []
    with pytest.raises(ValueError, match="program_entry_conditions_not_true"):
        activate_program_stage(program, 1, snapshot, remaining_actions=5,
                               parse_stage=lambda *args: called.append(args))
    assert called == []
    # There is no observer/retry callback. Runner must discard the program to
    # its original H1 route; stage.observe is not permission to retry admission.


def test_admission_does_not_replace_original_persistent_action_preconditions(observed):
    raw = declaration()
    raw["stages"][1]["entry_conditions"] = [{"kind": "exists", "target": FUTURE}]
    guard = {"kind": "value_equals", "target": FUTURE, "value": "old"}
    raw["stages"][1]["arguments"]["edits"][0]["preconditions"] = [guard]
    program = parse_stage_program(raw, max_actions=6)
    snapshot = query_panel(observed)
    seen = []

    def original(mapped, current_snapshot, horizon):
        seen.append(copy.deepcopy(mapped))
        return parse_original(mapped, current_snapshot, horizon)

    plan = activate_program_stage(program, 1, snapshot, remaining_actions=5, parse_stage=original)
    assert plan == parse_original(program.stages[1].parser_object(), snapshot, 6)
    assert "entry_conditions" not in seen[0] and "recovery" not in seen[0]
    assert seen[0]["edits"][0]["preconditions"] == [guard]
    assert conditions_hold(plan.steps[0].preconditions, snapshot)
    drifted = replace(snapshot, controls=(
        snapshot.controls[0], replace(snapshot.controls[1], value="changed")))
    assert not conditions_hold(plan.steps[0].preconditions, drifted)


@pytest.mark.parametrize("policy", [None, True, "", "inherit", "Observe", {}])
def test_unknown_recovery_policy_is_not_silently_normalized(policy):
    raw = declaration()
    raw["stages"][1]["recovery"] = policy
    with pytest.raises(ValueError, match="program_recovery_policy"):
        parse_stage_program(raw, max_actions=6)
