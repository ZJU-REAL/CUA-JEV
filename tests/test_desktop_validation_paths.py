"""Recovery feedback distinguishes condition targets from navigation scopes."""

import copy

import pytest

from cua_jev.agent_planner import parse_plan
from cua_jev.agent_types import AgentSnapshot


def navigation_plan():
    return {"summary": "Reach Options", "done": False, "steps": [{
        "id": "step_private_name", "operation": "click", "navigation": True,
        "target": {"app": "Editor"}, "label": "Reach Options",
        "postconditions": [{"kind": "exists", "target": {"name": "Options", "app": "Editor"}}],
    }]}


@pytest.mark.parametrize("field", ["preconditions", "postconditions", "completion_conditions"])
def test_missing_condition_name_reports_exact_location_without_values(field):
    plan = navigation_plan()
    invalid = {"kind": "exists", "target": {"app": "private_application_value"}}
    if field == "completion_conditions":
        plan[field] = [invalid]
        expected = f"{field}[0].target"
    else:
        plan["steps"][0][field] = [invalid]
        expected = f"steps[0].{field}[0].target"
    snapshot = AgentSnapshot("osworld://desktop", "Desktop", "", ())
    with pytest.raises(ValueError) as error:
        parse_plan(plan, snapshot, profile="desktop")
    assert str(error.value).startswith(expected + ":")
    assert "not a condition target" in str(error.value)
    assert "private" not in str(error.value)


def test_navigation_scope_is_valid_after_fixing_only_condition_target():
    snapshot = AgentSnapshot("osworld://desktop", "Desktop", "", ())
    valid = navigation_plan()
    invalid = copy.deepcopy(valid)
    invalid["steps"][0]["postconditions"][0]["target"].pop("name")
    with pytest.raises(ValueError, match=r"steps\[0\].postconditions\[0\].target"):
        parse_plan(invalid, snapshot, profile="desktop")
    result = parse_plan(valid, snapshot, profile="desktop")
    assert result.steps[0].navigation and result.steps[0].target.app == "Editor"


def test_action_target_failure_is_not_reported_as_a_condition_failure():
    plan = navigation_plan()
    plan["steps"][0]["navigation"] = False
    snapshot = AgentSnapshot("osworld://desktop", "Desktop", "", ())
    with pytest.raises(ValueError, match=r"^steps\[0\].target:"):
        parse_plan(plan, snapshot, profile="desktop")


@pytest.mark.parametrize("field", ["postconditions", "completion_conditions"])
def test_unsupported_target_fields_report_legal_shape_and_location_without_raw_data(field):
    plan = navigation_plan()
    condition = {"kind": "exists", "target": {"name": "Options", "app": "Editor",
                                              "private_field": "private_value"}}
    if field == "completion_conditions":
        plan[field] = [condition]
        path = "completion_conditions[0].target"
    else:
        plan["steps"][0][field] = [condition]
        path = "steps[0].postconditions[0].target"
    snapshot = AgentSnapshot("osworld://desktop", "Desktop", "", ())
    with pytest.raises(ValueError) as error:
        parse_plan(plan, snapshot, profile="desktop")
    message = str(error.value)
    assert message.startswith(path + ":") and "only name, role, app, window, ancestor, match" in message
    assert "not inside target" in message and "private" not in message
