import copy
from dataclasses import replace

import pytest

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.desktop_semantics import conditions_hold


@pytest.fixture
def snapshot():
    return AgentSnapshot(
        "osworld://desktop",
        "Preferences",
        "",
        (
            AgentControl(
                "one",
                "",
                "input",
                role="textbox",
                input_type="text",
                editable=True,
                focused=True,
                value="old",
                box=(10, 10, 100, 40),
                app="Editor",
                window="Preferences",
            ),
            AgentControl(
                "two",
                "Alias",
                "input",
                role="textbox",
                input_type="text",
                editable=True,
                value="old2",
                box=(10, 50, 100, 80),
                app="Editor",
                window="Preferences",
            ),
            AgentControl(
                "label",
                "Nearby label",
                "text",
                role="label",
                box=(110, 10, 200, 40),
                app="Editor",
                window="Preferences",
            ),
            AgentControl(
                "save",
                "Save",
                "button",
                role="button",
                box=(10, 90, 100, 120),
                app="Editor",
                window="Preferences",
            ),
        ),
    )


def current_edits():
    return {
        "intent": "edit_fields",
        "label": "Set the alias",
        "edits": [{"operation": "fill", "target": {"name": "Alias", "app": "Editor"}, "value": "new"}],
        "commit": {
            "operation": "click",
            "target": {"name": "Save", "app": "Editor"},
            "postconditions": [{"kind": "absent", "target": {"name": "Save", "app": "Editor"}}],
        },
        "completion": "goal",
    }


def edits():
    return current_edits()


def parse(raw, snapshot, limit=6):
    result = parse_plan(raw, snapshot, profile="desktop", max_actions=limit)
    _validate_desktop_execution_plan(result, snapshot, limit)
    return result


def test_current_edit_and_explicit_commit_keep_persistent_guard_dependency_and_readiness(snapshot):
    result = parse(edits(), snapshot)
    field, commit = result.steps
    assert field.operation == "fill" and field.value == "new"
    assert field.preconditions[-1].kind == field.postconditions[-1].kind == "ready_to_fill"
    assert commit.depends_on == (field.id,) and commit.preconditions[0].kind == "value_equals"
    assert commit.preconditions[0].target == field.target
    assert not conditions_hold(commit.preconditions, snapshot)
    assert result.completion_conditions == commit.postconditions


def test_anonymous_current_field_uses_real_role_context_and_focus_without_copying_label(snapshot):
    raw = edits()
    raw["edits"] = [
        {
            "operation": "fill",
            "target": {
                "role": "textbox",
                "app": "Editor",
                "window": "Preferences",
                "ancestor": "anonymous context",
            },
            "value": "new",
        }
    ]
    # Narrow the observed role query with actual focus evidence by using the
    # one-field current view; no invented accessible name is required.
    snapshot = replace(snapshot, controls=(snapshot.controls[0], snapshot.controls[2], snapshot.controls[3]))
    raw["edits"][0]["target"].pop("ancestor")
    raw["edits"][0]["preconditions"] = [{"kind": "focused", "target": raw["edits"][0]["target"]}]
    result = parse(raw, snapshot)
    assert result.steps[0].target.name == "" and result.steps[0].target.role == "textbox"
    assert [c.kind for c in result.steps[0].preconditions] == ["focused", "ready_to_fill"]


def test_independent_current_fields_create_legal_choice_without_requiring_edit_order(snapshot):
    raw = edits()
    raw["edits"].append(
        {
            "operation": "fill",
            "target": {"name": "", "role": "textbox", "app": "Editor", "window": "Preferences"},
            "value": "first",
        }
    )
    # Add observed disambiguating ancestor, not a name derived from current text.
    snapshot = replace(
        snapshot, controls=(replace(snapshot.controls[0], ancestors=("panel: Main",)), *snapshot.controls[1:])
    )
    raw["edits"][1]["target"]["ancestor"] = "Main"
    plan = parse(raw, snapshot)
    assert len(plan.steps) == 3 and not plan.steps[0].depends_on and not plan.steps[1].depends_on
    assert plan.steps[2].depends_on == (plan.steps[0].id, plan.steps[1].id)
    assert len(plan.steps[2].preconditions) == 2


def test_goal_without_commit_keeps_full_readiness_and_does_not_trust_truncation(snapshot):
    raw = edits()
    raw.pop("commit")
    plan = parse(raw, snapshot)
    filled = replace(
        snapshot, controls=tuple(replace(c, value="new") if c.ref == "two" else c for c in snapshot.controls)
    )
    assert conditions_hold(plan.completion_conditions, filled)
    assert not conditions_hold(plan.completion_conditions, replace(filled, a11y_complete=False))


def test_navigate_describes_arrival_and_exposes_native_gestures_without_invented_clicks(snapshot):
    raw = {
        "intent": "navigate",
        "label": "Reach another field",
        "scope": {"app": "Editor"},
        "arrival": [{"kind": "exists", "target": {"name": "Later field", "app": "Editor"}}],
        "completion": "stage",
    }
    plan = parse(raw, snapshot)
    assert len(plan.steps) == 1 and plan.steps[0].navigation and plan.steps[0].native_navigation_gestures
    assert plan.steps[0].target.app == "Editor" and not plan.steps[0].target.name
    assert not plan.completion_conditions


def test_explicit_future_named_edit_reuses_existing_reach_edit_compiler(snapshot):
    raw = {
        "intent": "navigate_edit",
        "label": "Set later property",
        "scope": {"app": "Editor"},
        "edit": {"operation": "fill", "target": {"name": "Later field", "app": "Editor"}, "value": "new"},
    }
    plan = parse(raw, snapshot)
    assert len(plan.steps) == 2 and plan.steps[0].native_navigation_gestures
    assert plan.steps[0].postconditions[0].kind == "ready_to_fill"
    assert plan.steps[1].depends_on == (plan.steps[0].id,)
    for name in ("Nearby label", "Alias"):
        raw["edit"]["target"]["name"] = name
        with pytest.raises(ValueError, match="navigate_edit target"):
            parse(raw, snapshot)


def test_act_current_is_one_current_grounded_input_and_keeps_normal_hybrid_click_rules(snapshot):
    raw = {
        "intent": "act_current",
        "label": "Change alias",
        "action": {
            "operation": "fill",
            "target": {"name": "Alias", "app": "Editor"},
            "value": "new",
            "postconditions": [
                {"kind": "value_equals", "target": {"name": "Alias", "app": "Editor"}, "value": "new"}
            ],
        },
    }
    assert len(parse(raw, snapshot).steps) == 1
    raw["action"]["target"]["name"] = "Later field"
    with pytest.raises(ValueError, match="no current executable"):
        parse(raw, snapshot)
    raw["action"] = {
        "operation": "click",
        "target": {"name": "Save", "app": "Editor"},
        "postconditions": [{"kind": "absent", "target": {"name": "Save", "app": "Editor"}}],
    }
    with pytest.raises(ValueError, match="persistent-state-guarded"):
        parse(raw, snapshot)


def test_complete_is_advisory_and_requires_explicit_whole_goal_evidence(snapshot):
    plan = parse(
        {
            "intent": "complete",
            "label": "Finished",
            "conditions": [
                {"kind": "value_equals", "target": {"name": "Alias", "app": "Editor"}, "value": "old2"}
            ],
        },
        snapshot,
    )
    assert not plan.steps and plan.done and plan.completion_conditions
    with pytest.raises(ValueError, match="explicit whole-goal"):
        parse({"intent": "complete", "label": "Finished", "conditions": []}, snapshot)


@pytest.mark.parametrize(
    "change",
    [
        {"intent": []},
        {"intent": "unknown"},
        {"steps": []},
        {"completion": []},
        {"completion": "maybe"},
        {"edits": []},
        {"edits": "invalid"},
        {"edits": [{"operation": "SCRIPT", "target": {}}]},
        {"edits": [{"operation": "fill", "target": {"name": "Alias", "app": "Editor"}}]},
        {"edits": [{"operation": "check", "target": {"name": "Alias", "app": "Editor"}, "value": "true"}]},
        {"commit": {"operation": [], "postconditions": []}},
        {"commit": {"operation": "click", "postconditions": []}},
        {"commit": {"operation": "SCRIPT", "postconditions": [{"kind": "text_contains", "value": "done"}]}},
    ],
)
def test_malformed_or_mixed_intents_are_rejected_without_target_rewriting(snapshot, change):
    raw = edits() | change
    with pytest.raises(ValueError):
        parse(raw, snapshot)


def test_empty_edits_feedback_explains_how_to_guard_a_commit_without_repeating_satisfied_input(snapshot):
    raw = edits()
    raw["edits"] = []
    with pytest.raises(ValueError, match="already-correct field.*guard an explicit commit"):
        parse(raw, snapshot)
    raw["edits"] = [{"operation": "fill", "target": {"name": "Alias", "app": "Editor"}, "value": "old2"}]
    result = parse(raw, snapshot)
    assert conditions_hold(result.steps[0].postconditions, snapshot)
    assert result.steps[1].depends_on == ("edit1",)
    assert any(c.kind == "value_equals" and c.value == "old2" for c in result.steps[1].preconditions)


@pytest.mark.parametrize("issue,expected", [
    ("array", "edits must be a JSON array"),
    ("count", "at most 6 current fields"),
    ("incomplete", "complete observation"),
    ("missing", "missing required stage fields: edits"),
    ("extra", "unsupported stage fields; allowed fields:"),
])
def test_form_shape_feedback_identifies_distinct_repair_without_echoing_private_values(
    snapshot, issue, expected,
):
    raw = edits()
    if issue == "array":
        raw["edits"] = "private_value"
    elif issue == "count":
        raw["edits"] *= 7
    elif issue == "incomplete":
        snapshot = replace(snapshot, a11y_complete=False)
    elif issue == "missing":
        raw.pop("edits")
    else:
        raw["private_field"] = "private_value"
    with pytest.raises(ValueError, match=expected) as error:
        parse(raw, snapshot)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("name", ["Nearby label", "Missing"])
def test_static_labels_and_future_controls_cannot_masquerade_as_current_edits(snapshot, name):
    raw = edits()
    raw["edits"][0]["target"]["name"] = name
    with pytest.raises(ValueError, match="operation-capable control"):
        parse(raw, snapshot)


def test_ambiguous_context_duplicate_field_and_incomplete_view_fail_closed(snapshot):
    raw = edits()
    raw["edits"][0]["target"] = {"role": "textbox", "app": "Editor"}
    with pytest.raises(ValueError, match="one current"):
        parse(raw, snapshot)
    raw = edits()
    raw["edits"].append(copy.deepcopy(raw["edits"][0]))
    with pytest.raises(ValueError, match="only once"):
        parse(raw, snapshot)
    with pytest.raises(ValueError, match="complete observation"):
        parse(edits(), replace(snapshot, a11y_complete=False))


def test_existing_action_and_model_authored_guard_bounds_are_not_expanded(snapshot):
    raw = edits()
    other = replace(snapshot.controls[1], ref="other", label="Other")
    snapshot = replace(snapshot, controls=(*snapshot.controls, other))
    raw["edits"].append({"operation": "fill", "target": {"name": "Other", "app": "Editor"}, "value": "new"})
    with pytest.raises(ValueError, match="bounded steps"):
        parse(raw, snapshot, limit=2)
    raw = edits()
    raw["commit"]["preconditions"] = [{"kind": "exists", "target": {"name": "Alias"}}] * 8
    with pytest.raises(ValueError, match="at most eight"):
        parse(raw, snapshot)


@pytest.mark.parametrize("profile,limit", [("desktop", 1), ("browser", 6)])
def test_stage_intents_do_not_change_single_action_recovery_or_browser_protocol(snapshot, profile, limit):
    with pytest.raises(ValueError, match="desktop multi-action"):
        parse_plan(edits(), snapshot, profile=profile, max_actions=limit)
