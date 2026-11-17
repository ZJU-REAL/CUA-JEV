"""A per-step VLM action must identify its control, not select by tree order."""

from dataclasses import replace

import pytest

from cua_jev.agent_planner import _validate_desktop_execution_plan
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget


def inputs():
    first = AgentControl("d1", "", "input", "spinbutton", input_type="number", editable=True,
                         box=(10, 10, 100, 30), app="Editor", ancestors=("Memory",))
    second = replace(first, ref="d2", box=(10, 50, 100, 70), ancestors=("History",))
    return AgentSnapshot("osworld://desktop", "Preferences", "Options", (first, second))


@pytest.mark.parametrize("operation,value", [("fill", "100"), ("click", "")])
def test_immediate_ambiguous_query_is_rejected_before_direct_policy(operation, value):
    step = AgentStep("s1", operation, value=value, target=AgentTarget(role="spinbutton", app="Editor"))
    with pytest.raises(ValueError, match="multiple executable controls"):
        _validate_desktop_execution_plan(AgentPlan("Set option", (step,)), inputs(), 1)


def test_observed_context_disambiguates_immediate_field():
    step = AgentStep("s1", "fill", value="100",
                     target=AgentTarget(role="spinbutton", app="Editor", ancestor="History"))
    _validate_desktop_execution_plan(AgentPlan("Set option", (step,)), inputs(), 1)


def test_screenshot_fallback_remains_available_for_anonymous_inputs():
    step = AgentStep("s1", "visual_fill", value="100", label="History", box=(10, 50, 100, 70))
    _validate_desktop_execution_plan(AgentPlan("Set option", (step,)), inputs(), 1)


def test_hybrid_navigation_still_offers_multiple_real_controls():
    step = AgentStep("s1", "click", navigation=True, label="Reach History",
                     target=AgentTarget(app="Editor"),
                     postconditions=(AgentCondition("exists", AgentTarget(name="History")),))
    _validate_desktop_execution_plan(AgentPlan("Navigate", (step,), reusable=True), inputs(), 6)


def test_unique_executable_control_ignores_readonly_duplicate_label():
    snapshot = inputs()
    first = replace(snapshot.controls[0], label="History")
    readonly = replace(first, ref="label", tag="text", role="label", editable=False)
    snapshot = replace(snapshot, controls=(first, readonly))
    step = AgentStep("s1", "fill", value="100", target=AgentTarget(name="History"))
    _validate_desktop_execution_plan(AgentPlan("Set option", (step,)), snapshot, 1)
