"""Direct native toggle evidence grants pointer state edits, not guessed focus."""

from __future__ import annotations

import copy
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.desktop_semantics import (
    condition_evidence,
    desktop_click_modes,
    executable_control,
    executable_controls,
    matching_controls,
    navigation_control_evidence,
    observable_condition_kinds,
    observed_checked_state,
)
from cua_jev.osworld_backend import OSWorldBackend

FIXTURE = json.loads((Path(__file__).parent / "fixtures/osworld_native_toggle.json").read_text())
STATE = "{https://accessibility.ubuntu.example.org/ns/state}"
ACTION = "{https://accessibility.ubuntu.example.org/ns/action}"
COMPONENT = "{https://accessibility.ubuntu.example.org/ns/component}"
FOREIGN = "{https://unrelated.invalid/ns/state}"
TARGET = AgentTarget(name="Do Not Disturb", app="gnome-control-center", window="Settings")


def tree(variant="gtk_settings"):
    return ET.fromstring(FIXTURE["variants"][variant]["xml"])


def switch(root):
    return next(root.iter("toggle-button"))


def parent_of(root, child):
    return next(node for node in root.iter() if child in list(node))


class Desktop:
    def __init__(self, root=None, after=None):
        self.root = tree() if root is None else root
        self.after = after
        self.calls = []
        frame = io.BytesIO()
        Image.new("RGB", (1920, 1080), "white").save(frame, format="PNG")
        self.png = frame.getvalue()

    def _get_obs(self):
        return {"screenshot": self.png, "accessibility_tree": ET.tostring(self.root, encoding="unicode")}

    def step(self, action, pause=2):
        self.calls.append(action)
        if self.after:
            self.after(self.root)
        return self._get_obs(), 0, False, {}


@pytest.fixture
def desktop():
    env = Desktop()
    backend = OSWorldBackend(env, pause=0)
    try:
        yield env, backend
    finally:
        backend.close()


def target_control(snapshot):
    return next(c for c in matching_controls(snapshot, TARGET) if c.input_type == "checkbox")


def step(operation="check", target=TARGET):
    condition = AgentCondition("checked" if operation == "check" else "unchecked", target)
    return AgentStep("edit-switch", operation, target=target, recovery="observe",
                     postconditions=(condition,))


def test_recorded_native_action_is_a_pointer_target_without_inventing_focus(desktop):
    env, backend = desktop
    raw = switch(env.root)
    snapshot = backend.observe()
    control = target_control(snapshot)
    assert FIXTURE["variants"]["gtk_settings"]["source_sha256"] == (
        "0962b4c478fa503019916ad5367f6c11dae0ff790e11cc4d5ca055d009957012"
    )
    assert STATE + "focusable" not in raw.attrib and STATE + "checked" not in raw.attrib
    assert snapshot.a11y_complete and control.tag == "container" and control.role == "checkbox"
    assert control.box == (698, 219, 724, 243)
    assert control.native.actions == ("toggle",) and control.native.state_set_complete
    assert observed_checked_state(control) is False and not control.focused
    assert "focused" not in observable_condition_kinds(control)
    assert executable_controls(snapshot, "check", TARGET) == (control,)
    assert executable_controls(snapshot, "uncheck", TARGET) == (control,)
    assert desktop_click_modes(control) == () and not executable_control(control, "fill")
    assert condition_evidence(AgentCondition("ready_to_check", TARGET), snapshot)["outcome"] == "true"
    raw_plan = {"summary": "Set observed switch", "done": False, "steps": [step().to_dict()]}
    plan = parse_plan(raw_plan, snapshot, max_actions=1, profile="desktop")
    _validate_desktop_execution_plan(plan, snapshot, 1)
    assert len(backend.compile(plan.steps[0], snapshot, snapshot)) == 1
    assert env.calls == []


def test_wrong_current_ref_operation_reports_native_capabilities_without_granting_click(desktop):
    env, backend = desktop
    observed = backend.observe()
    actual = target_control(observed)
    plan = parse_plan({"summary": "Use observed switch", "done": False, "steps": [
        {"id": "s1", "operation": "click", "ref": actual.ref},
    ]}, observed, profile="desktop", max_actions=1)
    with pytest.raises(ValueError) as error:
        _validate_desktop_execution_plan(plan, observed, 1)
    message = str(error.value)
    assert "Requested operation=click; matched current controls=1" in message
    assert "observed operation capabilities=check,uncheck" in message
    assert actual.label not in message and actual.app not in message and actual.ref not in message
    assert backend.compile(plan.steps[0], observed, observed) == () and env.calls == []

    typed = parse_plan({"summary": "Enable observed switch", "done": False, "steps": [
        {"id": "s1", "operation": "check", "ref": actual.ref},
    ]}, observed, profile="desktop", max_actions=1)
    _validate_desktop_execution_plan(typed, observed, 1)
    assert len(backend.compile(typed.steps[0], observed, observed)) == 1 and env.calls == []


@pytest.mark.parametrize("operation", ["check", "uncheck"])
def test_actual_backend_sends_one_pointer_input_then_verifies_actual_state(desktop, operation):
    env, backend = desktop
    if operation == "uncheck":
        switch(env.root).set(STATE + "checked", "true")

    def change(root):
        if operation == "check":
            switch(root).set(STATE + "checked", "true")
        else:
            switch(root).attrib.pop(STATE + "checked")

    env.after = change
    before = backend.observe()
    candidate, = backend.compile(step(operation), before, before)
    assert candidate.arguments["bound_control"]["native"]["checked_state"] is (operation == "uncheck")
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and backend.verify(candidate, receipt).passed
    assert env.calls == [{"action_type": "CLICK", "parameters": {
        "x": 1365, "y": 249, "button": "left", "num_clicks": 1,
    }}]
    assert backend.environment_steps == 1
    assert condition_evidence(step(operation).postconditions[0], backend.observe())["outcome"] == "true"


@pytest.mark.parametrize("operation", ["check", "uncheck"])
def test_known_satisfied_state_is_idempotent_not_an_automatic_toggle(desktop, operation):
    env, backend = desktop
    if operation == "check":
        switch(env.root).set(STATE + "checked", "true")
    before = backend.observe()
    candidate, = backend.compile(step(operation), before, before)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and backend.verify(candidate, receipt).passed
    assert env.calls == [] and backend.environment_steps == 0


def test_submission_without_state_change_is_not_success_and_pending_only_observes(desktop):
    env, backend = desktop
    before = backend.observe()
    candidate, = backend.compile(step(), before, before)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    switch(env.root).set(STATE + "checked", "true")
    assert backend.verify_pending(candidate, receipt).passed
    assert len(env.calls) == 1


@pytest.mark.parametrize("description", ["", "Changer le commutateur", "not an action name"])
def test_native_action_name_comes_from_key_not_description(desktop, description):
    env, backend = desktop
    switch(env.root).set(ACTION + "toggle_desc", description)
    before = backend.observe()
    assert len(backend.compile(step(), before, before)) == 1


@pytest.mark.parametrize("defect", [
    "foreign", "missing_kb", "description_only", "parent_action", "child_action",
])
def test_actionless_or_nonprotocol_copies_cannot_borrow_toggle_authority(desktop, defect):
    env, backend = desktop
    node = switch(env.root)
    fields = {key: node.attrib.pop(key) for key in (ACTION + "toggle_desc", ACTION + "toggle_kb")}
    if defect == "foreign":
        node.attrib.update({key.replace("accessibility.ubuntu.example.org", "unrelated.invalid"): value
                            for key, value in fields.items()})
    elif defect == "missing_kb":
        node.set(ACTION + "toggle_desc", "Toggles the switch")
    elif defect == "description_only":
        node.set(ACTION + "press_desc", "toggle")
        node.set(ACTION + "press_kb", "")
    elif defect == "parent_action":
        parent_of(env.root, node).attrib.update(fields)
    else:
        ET.SubElement(node, "check-box", {**fields, STATE + "visible": "true"})
    snapshot = backend.observe()
    assert backend.compile(step(), snapshot, snapshot) == ()
    assert env.calls == []


@pytest.mark.parametrize("defect", [
    "checked_invalid", "checked_foreign_true", "checked_foreign_false", "indeterminate",
    "defunct", "missing_visible", "foreign_visible", "state_invalid",
])
def test_unknown_or_foreign_state_is_not_unchecked_or_an_input_permission(desktop, defect):
    env, backend = desktop
    node = switch(env.root)
    if defect == "checked_invalid":
        node.set(STATE + "checked", "perhaps")
    elif defect.startswith("checked_foreign_"):
        node.set(FOREIGN + "checked", defect.removeprefix("checked_foreign_"))
    elif defect in {"indeterminate", "defunct"}:
        node.set(STATE + defect, "true")
    elif defect in {"missing_visible", "foreign_visible"}:
        node.attrib.pop(STATE + "visible")
        if defect == "foreign_visible":
            node.set(FOREIGN + "visible", "true")
    else:
        node.set(STATE + "busy", "perhaps")
    snapshot = backend.observe()
    control = target_control(snapshot)
    assert observed_checked_state(control) is None
    for kind in ("checked", "unchecked", "ready_to_check", "ready_to_uncheck"):
        assert condition_evidence(AgentCondition(kind, TARGET), snapshot)["outcome"] == "unknown"
    for operation in ("check", "uncheck"):
        assert backend.compile(step(operation), snapshot, snapshot) == ()
    assert env.calls == []


def test_foreign_focus_cannot_promote_native_state_copy_to_ordinary_input(desktop):
    env, backend = desktop
    switch(env.root).set(FOREIGN + "focusable", "true")
    snapshot = backend.observe()
    control = target_control(snapshot)
    assert control.tag == "container" and control.native.focusable is None
    assert "focused" not in observable_condition_kinds(control)
    assert desktop_click_modes(control) == ()
    assert len(backend.compile(step(), snapshot, snapshot)) == 1


@pytest.mark.parametrize("location", ["node", "parent", "frame", "application"])
@pytest.mark.parametrize("state", ["enabled", "sensitive"])
def test_explicit_disabled_evidence_anywhere_in_chain_blocks_pointer_edit(desktop, location, state):
    env, backend = desktop
    node = switch(env.root)
    changed = (node if location == "node" else parent_of(env.root, node) if location == "parent"
               else next(env.root.iter(location)))
    changed.set(STATE + state, "false")
    snapshot = backend.observe()
    assert backend.compile(step(), snapshot, snapshot) == ()


@pytest.mark.parametrize("defect", ["hidden", "offscreen", "no_geometry", "foreign_geometry", "nonleaf"])
def test_native_switch_requires_its_own_visible_leaf_rectangle(desktop, defect):
    env, backend = desktop
    node = switch(env.root)
    if defect == "hidden":
        node.set(STATE + "showing", "false")
    elif defect == "offscreen":
        node.set(COMPONENT + "screencoord", "(-50, 236)")
    elif defect in {"no_geometry", "foreign_geometry"}:
        value = node.attrib.pop(COMPONENT + "screencoord")
        if defect == "foreign_geometry":
            node.set("{https://unrelated.invalid/ns/component}screencoord", value)
    else:
        ET.SubElement(node, "label", {"name": "presentation", STATE + "visible": "true"})
    snapshot = backend.observe()
    assert backend.compile(step(), snapshot, snapshot) == ()


@pytest.mark.parametrize("changed", ["checked", "disabled", "indeterminate", "foreign_focus", "duplicate"])
def test_fresh_preflight_rejects_changes_after_binding_without_input(desktop, changed):
    env, backend = desktop
    before = backend.observe()
    candidate, = backend.compile(step(), before, before)
    node = switch(env.root)
    if changed == "checked":
        node.set(STATE + "checked", "true")
    elif changed == "disabled":
        node.set(STATE + "enabled", "false")
    elif changed == "indeterminate":
        node.set(STATE + "indeterminate", "true")
    elif changed == "foreign_focus":
        node.set(STATE + "focusable", "false")
        node.set(FOREIGN + "focusable", "true")
    else:
        row = parent_of(env.root, parent_of(env.root, node))
        parent_of(env.root, row).append(copy.deepcopy(row))
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert env.calls == []


@pytest.mark.parametrize("query", [TARGET, AgentTarget(role="checkbox", app=TARGET.app)])
def test_multiple_real_controls_are_ambiguous_before_desired_state_filtering(desktop, query):
    env, backend = desktop
    row = parent_of(env.root, parent_of(env.root, switch(env.root)))
    duplicate = copy.deepcopy(row)
    switch(duplicate).set(STATE + "checked", "true")
    parent_of(env.root, row).append(duplicate)
    snapshot = backend.observe()
    assert condition_evidence(AgentCondition("checked", query), snapshot)["reason"] == "ambiguous_target"
    assert backend.compile(step(target=query), snapshot, snapshot) == ()


def test_partial_tree_cannot_prove_unique_native_target_or_completion(desktop):
    env, backend = desktop
    switch(env.root).set(STATE + "checked", "true")
    snapshot = replace(backend.observe(), a11y_complete=False)
    for kind in ("checked", "unchecked", "ready_to_check"):
        evidence = condition_evidence(AgentCondition(kind, TARGET), snapshot)
        assert evidence["reason"] == "incomplete_observation"
    assert backend.compile(step(), snapshot, snapshot) == ()
    control = target_control(snapshot)
    assert backend.compile(AgentStep("ref", "uncheck", ref=control.ref), snapshot, snapshot) == ()


@pytest.mark.parametrize("after", ["indeterminate", "foreign_checked", "lost_native_action"])
def test_after_input_unknown_state_cannot_pass_immediate_or_pending_verification(desktop, after):
    env, backend = desktop

    def change(root):
        node = switch(root)
        node.set(STATE + "checked", "true")
        if after == "indeterminate":
            node.set(STATE + "indeterminate", "true")
        elif after == "foreign_checked":
            node.set(FOREIGN + "checked", "false")
        else:
            node.attrib.pop(ACTION + "toggle_desc")
            node.attrib.pop(ACTION + "toggle_kb")

    env.after = change
    before = backend.observe()
    candidate, = backend.compile(step(), before, before)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert len(env.calls) == 1


def test_truncated_after_observation_cannot_claim_completion_even_with_checked_true(desktop):
    env, backend = desktop

    def change(root):
        switch(root).set(STATE + "checked", "true")
        ET.SubElement(parent_of(root, switch(root)), "label", {
            "name": "x" * 2001, STATE + "showing": "true",
        })

    env.after = change
    before = backend.observe()
    candidate, = backend.compile(step(), before, before)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert not backend.observe().a11y_complete and len(env.calls) == 1


@pytest.mark.parametrize("variant", ["shell_shown_unchecked", "shell_shown_checked"])
def test_existing_shell_toggle_keeps_its_route_and_child_state_copy_stays_readonly(variant):
    env = Desktop(tree(variant))
    with_backend = OSWorldBackend(env, pause=0)
    try:
        snapshot = with_backend.observe()
        checks = [c for c in snapshot.controls if c.input_type == "checkbox"]
        actual = next(c for c in checks if c.label == "Do Not Disturb")
        assert actual.tag == "input" and actual.native is None
        assert executable_control(actual, "check")
        copies = [c for c in checks if c is not actual]
        assert copies and all(c.tag == "container" and c.native is None for c in copies)
        assert all(not executable_control(c, "check") for c in copies)
    finally:
        with_backend.close()


def test_checked_and_native_capability_changes_are_fingerprint_and_progress_evidence(desktop):
    env, backend = desktop
    before = backend.observe()
    switch(env.root).set(STATE + "checked", "true")
    after = backend.observe()
    assert before.fingerprint() != after.fingerprint()
    assert navigation_control_evidence(target_control(before)) != navigation_control_evidence(
        target_control(after)
    )
    assert navigation_control_evidence(target_control(before))["operations"] == ("check", "uncheck")
