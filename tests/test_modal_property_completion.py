"""Native modal ancestry defers a field-only stop without inferring a commit."""

import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest
from test_native_toggle_actions import FOREIGN, STATE, Desktop

from cua_jev.agent_types import AgentCondition, AgentPlan, AgentTarget
from cua_jev.desktop_property import pending_dialog_field_completion
from cua_jev.desktop_semantics import executable_controls
from cua_jev.osworld_backend import OSWorldBackend


def observed(*, modal=True, depth=1, foreign=False, conflict=False, hidden=False):
    root = ET.Element("desktop")
    app = ET.SubElement(root, "application", name="Editor")
    common = {STATE + k: "true" for k in ("showing", "visible", "enabled", "sensitive")}
    frame = ET.SubElement(app, "frame", {"name": "Main", **common})
    panel = ET.SubElement(frame, "panel", common)
    if modal is not None:
        panel.set((FOREIGN if foreign else STATE) + "modal", str(modal).lower())
    if conflict:
        panel.set(FOREIGN + "modal", "false")
    if hidden:
        panel.set(STATE + "showing", "false")
    parent = panel
    for _ in range(depth):
        parent = ET.SubElement(parent, "panel", common)
    field = ET.SubElement(parent, "text", {
        "name": "Value", STATE + "editable": "true", STATE + "focused": "true", **common,
    })
    field.text = "new value"
    ET.SubElement(parent, "push-button", {"name": "Submit", **common})
    outside = ET.SubElement(frame, "text", {"name": "Other value", STATE + "editable": "true", **common})
    outside.text = "new value"
    env = Desktop(root)
    return OSWorldBackend(env)._snapshot(env._get_obs())


def plan(name="Value"):
    return AgentPlan("Set the requested value", (), completion_conditions=(
        AgentCondition("value_equals", AgentTarget(name=name, app="Editor"), "new value"),
    ), defer_dialog_field_completion=True)


@pytest.mark.parametrize("depth", [0, 1, 6, 12])
def test_unnamed_native_modal_panel_survives_deep_ancestry_without_dialog_role(depth):
    current = observed(depth=depth)
    field, = [c for c in current.controls if c.label == "Value"]
    assert field.modal_context and field.role == "textbox"
    assert all(c.role != "dialog" for c in current.controls)
    pending = pending_dialog_field_completion(plan(), current)
    assert pending["reason"] == "field_effect_in_open_modal" and not pending["task_success"]
    assert not next(c for c in current.controls if c.label == "Other value").modal_context
    assert pending_dialog_field_completion(plan("Other value"), current) is None


@pytest.mark.parametrize("change", [
    {"modal": None}, {"modal": False}, {"foreign": True}, {"conflict": True}, {"hidden": True},
])
def test_missing_untrusted_negative_or_hidden_modal_evidence_is_not_manufactured(change):
    current = observed(**change)
    assert not any(c.modal_context for c in current.controls)
    assert pending_dialog_field_completion(plan(), current) is None


def test_modal_evidence_changes_no_input_capability_and_does_not_relax_false_or_unknown_conditions():
    current, ordinary = observed(), observed(modal=False)
    scope = AgentTarget(app="Editor")
    for operation in ("click", "fill", "check", "uncheck"):
        choices = executable_controls(current, operation, scope, allow_scope=True)
        previous = executable_controls(ordinary, operation, scope, allow_scope=True)
        assert [c.ref for c in choices] == [c.ref for c in previous]
    false = replace(plan(), completion_conditions=(
        AgentCondition("value_equals", AgentTarget(name="Value", app="Editor"), "different value"),
    ))
    unknown = replace(plan(), completion_conditions=plan().completion_conditions + (
        AgentCondition("exists", AgentTarget(name="Unobserved fact", app="Editor")),
    ))
    assert pending_dialog_field_completion(false, current) is None
    assert pending_dialog_field_completion(unknown, current) is None
    legacy = replace(plan(), defer_dialog_field_completion=False)
    assert pending_dialog_field_completion(legacy, current) is None
