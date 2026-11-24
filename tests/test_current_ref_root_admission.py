"""Current references remain one fresh action under a multi-action planning horizon."""

import io
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image


@pytest.fixture
def snapshot():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, "PNG")
    return AgentSnapshot("osworld://desktop", "Main", "", (
        AgentControl("button", "Create document Ctrl+N", "button", role="menuitem", selectable=True,
                     app="Editor", window="Main", box=(100, 100, 300, 150)),
        AgentControl("field", "Name", "input", role="textbox", input_type="text", editable=True,
                     app="Editor", window="Main", box=(100, 200, 300, 250)),
        AgentControl("toggle", "Backup", "input", role="checkbox", input_type="checkbox",
                     app="Editor", window="Main", box=(100, 300, 300, 350)),
    ), image=screenshot_image(buffer.getvalue()))


def raw(operation="click", ref="button", **extra):
    return {"intent": "act_current", "label": "Use the actual current control", "action": {
        "operation": operation, "ref": ref, **extra,
    }}


@pytest.mark.parametrize("operation,ref,extra", [("click", "button", {}),
                                               ("fill", "field", {"value": "report"}),
                                               ("check", "toggle", {}), ("uncheck", "toggle", {})])
def test_root_horizon_does_not_reclassify_a_current_ref_as_a_semantic_chain(snapshot, operation, ref, extra):
    plan = parse_plan(raw(operation, ref, **extra), snapshot, profile="desktop", max_actions=6)
    assert not plan.reusable and len(plan.steps) == 1
    _validate_desktop_execution_plan(plan, snapshot, 1)
    _validate_desktop_execution_plan(plan, snapshot, 6)
    backend = OSWorldBackend(object(), pause=0)
    try:
        choices = backend.compile(plan.steps[0], snapshot, snapshot)
        assert len(choices) == 1 and choices[0].arguments["bound_control"]["ref"] == ref
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["moved", "disabled", "removed", "state"])
def test_ref_is_not_reused_after_the_planned_state_changes(snapshot, change):
    plan = parse_plan(raw(), snapshot, profile="desktop", max_actions=6)
    _validate_desktop_execution_plan(plan, snapshot, 6)
    first, *others = snapshot.controls
    updated = {
        "moved": replace(first, box=(400, 100, 600, 150)),
        "disabled": replace(first, disabled=True), "state": replace(first, selected=True),
        "removed": None,
    }[change]
    changed = replace(snapshot, controls=tuple(([updated] if updated else []) + others))
    backend = OSWorldBackend(object(), pause=0)
    try:
        assert backend.compile(plan.steps[0], snapshot, changed) == ()
    finally:
        backend.close()


def test_unavailable_current_ref_is_still_rejected_before_any_input(snapshot):
    hidden = replace(snapshot, controls=(replace(snapshot.controls[0], disabled=True),))
    with pytest.raises(ValueError):
        parse_plan(raw(), hidden, profile="desktop", max_actions=6)
    static = replace(snapshot, controls=(replace(snapshot.controls[0], tag="label", role="label"),))
    with pytest.raises(ValueError, match="no current executable control"):
        parse_plan(raw(), static, profile="desktop", max_actions=6)


def test_current_ref_cannot_carry_future_semantic_conditions_or_fake_reusability(snapshot):
    bad = raw(postconditions=[{"kind": "exists", "target": {"name": "Later", "app": "Editor"}}])
    with pytest.raises(ValueError, match="cannot mix refs"):
        parse_plan(bad, snapshot, profile="desktop", max_actions=6)
    bad = raw() | {"reusable": False}
    with pytest.raises(ValueError):
        parse_plan(bad, snapshot, profile="desktop", max_actions=6)
