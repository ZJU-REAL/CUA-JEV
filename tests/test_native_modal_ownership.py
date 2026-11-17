"""Observed modal-container ancestry supports anonymous and non-dialog forms."""

import xml.etree.ElementTree as ET

import pytest
from test_native_toggle_actions import COMPONENT, FOREIGN, STATE, Desktop
from test_observed_property_goal import compile

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.desktop_property_confirmation import property_confirmation_steps
from cua_jev.desktop_semantics import matching_controls
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend

VISIBLE = {STATE + k: "true" for k in ("enabled", "sensitive", "showing", "visible")}


def observed_form(*, role="panel", depth=1, modal=True, foreign=False, conflict=False, hidden=False):
    root = ET.Element("desktop-frame")
    app = ET.SubElement(root, "application", name="Shell")
    frame = ET.SubElement(app, "frame", {"name": "Desktop", **VISIBLE})
    owner = ET.SubElement(frame, role, {
        "name": "", **VISIBLE, COMPONENT + "screencoord": "(100, 100)", COMPONENT + "size": "(400, 100)",
    })
    if modal is not None:
        owner.set((FOREIGN if foreign else STATE) + "modal", str(modal).lower())
    if conflict:
        owner.set(FOREIGN + "modal", "false")
    if hidden:
        owner.set(STATE + "showing", "false")
    parent = owner
    for _ in range(depth):
        parent = ET.SubElement(parent, "panel", VISIBLE)
    field = ET.SubElement(parent, "text", {
        "name": "Value", **VISIBLE, STATE + "editable": "true", STATE + "focused": "true",
        COMPONENT + "screencoord": "(120, 140)", COMPONENT + "size": "(200, 30)",
    })
    field.text = "wanted"
    ET.SubElement(parent, "push-button", {
        "name": "Submit", **VISIBLE, COMPONENT + "screencoord": "(340, 140)",
        COMPONENT + "size": "(100, 30)",
    })
    ET.SubElement(frame, "push-button", {
        "name": "Outside", **VISIBLE, COMPONENT + "screencoord": "(500, 400)",
        COMPONENT + "size": "(100, 30)",
    })
    env = Desktop(root)
    backend = OSWorldBackend(env, pause=0)
    return env, backend, owner, backend.observe()


def template():
    return AgentStep("confirm", "confirm_property", label="Submit the requested change and close its form",
                     depends_on=("property",), property_confirmation_of="property", recovery="observe")


def selected_field():
    target = AgentTarget(name="Value", role="textbox", app="Shell", window="Desktop")
    return AgentStep("property", "fill", target=target, value="wanted", postconditions=(
        AgentCondition("value_equals", target, "wanted"), AgentCondition("ready_to_fill", target, "wanted"),
    ))


@pytest.mark.parametrize("role", ["panel", "frame", "file-chooser"])
@pytest.mark.parametrize("depth", [0, 6, 12])
def test_anonymous_container_is_retained_readonly_and_linked_through_actual_deep_ancestry(role, depth):
    env, backend, _owner, current = observed_form(role=role, depth=depth)
    owner, = [c for c in current.controls if c.modal]
    field, = matching_controls(current, selected_field().target)
    button, = [c for c in current.controls if c.label == "Submit"]
    outside, = [c for c in current.controls if c.label == "Outside"]
    assert owner.role == role and owner.tag == "container" and owner.modal_owner_ref == owner.ref
    assert field.modal_owner_ref == button.modal_owner_ref == owner.ref
    assert not outside.modal_owner_ref and not owner.label
    assert not owner.active and field.focused
    choices = property_confirmation_steps(template(), selected_field(), current)
    assert len(choices) == 1 and choices[0].target.name == "Submit"
    assert choices[0].postconditions[0].target.role == role
    assert backend.compile(AgentStep("bad", "click", target=AgentTarget(role=role, app="Shell")),
                           current, current) == ()
    assert not env.calls


@pytest.mark.parametrize("change", [
    {"modal": None}, {"modal": False}, {"foreign": True}, {"conflict": True}, {"hidden": True},
])
def test_absent_foreign_negative_or_hidden_modal_state_creates_no_owner(change):
    _env, _backend, _owner, current = observed_form(**change)
    assert not any(c.modal or c.modal_owner_ref for c in current.controls)
    assert property_confirmation_steps(template(), selected_field(), current) == ()


def test_current_owner_relation_and_its_observed_identity_must_both_be_unique():
    env, backend, owner, current = observed_form()
    parent = next(p for p in env.root.iter() if owner in list(p))
    duplicate = ET.SubElement(parent, "panel", {"name": "", **VISIBLE, STATE + "modal": "true"})
    assert duplicate is not owner
    current = backend.observe()
    assert len([c for c in current.controls if c.modal]) == 2
    assert property_confirmation_steps(template(), selected_field(), current) == ()


def test_receipt_verified_value_and_native_modal_ownership_finish_without_another_planner_call():
    env, backend, owner, current = observed_form()
    parent = next(p for p in env.root.iter() if owner in list(p))
    env.after = lambda _root: parent.remove(owner)
    policies = []

    class Planner:
        calls = 0

        def plan(self, _goal, observed, _history, **_kwargs):
            self.calls += 1
            return compile({"intent": "pursue_property", "label": "Set requested field value",
                            "scope": {"app": "Shell"}, "edit": {"operation": "set", "value": "wanted"},
                            "completion": "goal", "confirm": {"label": "Submit and close its form"}},
                           observed)

    class Policy:
        def choose(self, observation, candidates):
            explicit = observation.state.get("explicit_property_confirmation") is True
            policies.append(explicit)
            chosen = (next(c for c in candidates if c.description.startswith("click Submit ")) if explicit
                      else next(c for c in candidates if c.capability == "agent.osworld.fill"))
            return Decision(observation.observation_id, chosen.id,
                            {c.id: float(c is chosen) for c in candidates}, 1, "offline-policy", 0)

    planner, events = Planner(), []
    result = AgentRunner(backend, planner, policy=Policy(), progress=events.append,
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Set and confirm the field")
    assert planner.calls == 1 and result.planner_calls == 1 and result.plan_completion_claims == 1
    assert result.status == "unverified" and not result.success and policies[-1] is True
    assert len(env.calls) == 1 and env.calls[0]["action_type"] == "CLICK"
    assert any(e["event"] == "agent_property_commit" for e in events)
    assert not any(c.modal for c in backend.observe().controls)
