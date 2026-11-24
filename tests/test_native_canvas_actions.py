"""Native item evidence grants explicit mouse gestures, never invented UI effects."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from cua_jev.agent_planner import (
    _desktop_instructions,
    _desktop_snapshot_payload,
    _validate_desktop_execution_plan,
    parse_plan,
)
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_semantics import (
    condition_evidence,
    desktop_click_modes,
    executable_control,
    executable_controls,
    observable_condition_kinds,
)
from cua_jev.osworld_backend import OSWorldBackend, _navigation_scope

FIXTURE = json.loads((Path(__file__).parent / "fixtures/osworld_native_canvas.json").read_text())
STATE = "{https://accessibility.ubuntu.example.org/ns/state}"
ACTION = "{https://accessibility.ubuntu.example.org/ns/action}"
COMPONENT = "{https://accessibility.ubuntu.example.org/ns/component}"
TARGET = AgentTarget(name="todo_list_Jan_1", role="canvas", app="org.gnome.Nautilus")


def fixture_tree():
    return ET.fromstring(FIXTURE["xml"])


def canvas(root):
    return next(node for node in root.iter() if node.tag == "canvas")


def parent_of(root, child):
    return next(node for node in root.iter() if child in node)


def image():
    buffer = io.BytesIO()
    Image.new("RGB", tuple(FIXTURE["screen_size"]), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class Desktop:
    def __init__(self, root=None, *, after=None):
        self.root = fixture_tree() if root is None else root
        self.after = after
        self.calls = []
        self.png = image()

    def _get_obs(self):
        return {"screenshot": self.png, "accessibility_tree": ET.tostring(self.root, encoding="unicode")}

    def step(self, action, pause=2):
        self.calls.append(action)
        if self.after is not None:
            self.after(self.root)
        return self._get_obs(), 0, False, {}


def step(*, button="left", count=1, post=None, navigation=False):
    return AgentStep(
        "native-item", "click", button=button, click_count=count,
        target=AgentTarget(app=TARGET.app) if navigation else TARGET,
        label="Inspect the current native item", navigation=navigation,
        postconditions=post or (AgentCondition("focused", TARGET),),
        recovery="observe",
    )


def observe(root):
    backend = OSWorldBackend(Desktop(root), pause=0)
    try:
        return backend.observe()
    finally:
        backend.close()


def native_control(snapshot):
    return next(control for control in snapshot.controls if control.role == "canvas")


def test_recorded_leaf_retains_unknown_states_without_losing_three_real_mouse_choices():
    assert FIXTURE["source"]["sha256"] == "71a6ca2823f73c6b3debb1f83e1f7b5d7afef058c39178cb6af938756fed1afc"
    assert hashlib.sha256(FIXTURE["xml"].encode()).hexdigest() == FIXTURE["xml_sha256"]
    snapshot = observe(fixture_tree())
    control = native_control(snapshot)
    assert control.role == "canvas" and control.tag == "container"
    assert control.native.enabled is None and control.native.sensitive is None
    assert control.native.actions == ("menu", "open") and control.native.complete
    assert control.native.parent_enabled and control.native.parent_sensitive
    assert control.box == (174, 229, 215, 308)
    assert desktop_click_modes(control) == (("left", 1), ("right", 1), ("left", 2))
    assert "focused" in observable_condition_kinds(control)
    assert "selected" not in observable_condition_kinds(control)
    assert condition_evidence(AgentCondition("selected", TARGET), snapshot) == {
        "outcome": "unknown", "reason": "unsupported_state",
    }
    assert not control.editable and not executable_control(control, "fill")


@pytest.mark.parametrize("button,count", [("left", 1), ("right", 1), ("left", 2)])
def test_full_gesture_parses_binds_and_sends_exact_typed_mouse_input(button, count):
    desktop = Desktop(after=lambda root: canvas(root).set(STATE + "focused", "true"))
    backend = OSWorldBackend(desktop, pause=0)
    try:
        snapshot = backend.observe()
        raw = {"summary": "Explicit mouse gesture", "done": False,
               "steps": [step(button=button, count=count).to_dict()]}
        plan = parse_plan(raw, snapshot, max_actions=1, profile="desktop")
        _validate_desktop_execution_plan(plan, snapshot, 1)
        candidates = backend.compile(plan.steps[0], snapshot, snapshot)
        assert len(candidates) == 1
        receipt = backend.execute(candidates[0], "observation", "decision")
        assert receipt.success and backend.verify(candidates[0], receipt).passed
        assert desktop.calls == [{"action_type": "CLICK", "parameters": {
            "x": 373, "y": 289, "button": button, "num_clicks": count,
        }}]
        assert candidates[0].arguments["bound_control"]["role"] == "canvas"
        assert receipt.output["button"] == button and receipt.output["click_count"] == count
    finally:
        backend.close()


def test_open_affordance_does_not_change_the_explicit_default_single_click_or_fake_open_effect():
    desktop = Desktop()
    backend = OSWorldBackend(desktop, pause=0)
    try:
        snapshot = backend.observe()
        requested = step(post=(AgentCondition("exists", AgentTarget(name="Opened content")),))
        offered = backend.compile(requested, snapshot, snapshot)
        receipt = backend.execute(offered[0], "o", "d")
        assert receipt.success and not backend.verify(offered[0], receipt).passed
        assert desktop.calls[0]["parameters"]["num_clicks"] == 1
        assert not backend.verify_pending(offered[0], receipt).passed
        assert len(desktop.calls) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("actions,expected", [
    (("open",), (("left", 1), ("left", 2))),
    (("menu",), (("left", 1), ("right", 1))),
    (("delete",), ()),
    ((), ()),
])
def test_advertised_actions_constrain_mouse_modes_without_executing_action_names(actions, expected):
    root = fixture_tree()
    node = canvas(root)
    for key in list(node.attrib):
        if key.startswith(ACTION):
            del node.attrib[key]
    for name in actions:
        node.set(ACTION + name + "_desc", "Untrusted description does not grant another action")
        node.set(ACTION + name + "_kb", "")
    snapshot = observe(root)
    assert desktop_click_modes(native_control(snapshot)) == expected
    backend = OSWorldBackend(Desktop(root), pause=0)
    try:
        for button, count in (("left", 1), ("right", 1), ("left", 2)):
            offered = backend.compile(step(button=button, count=count), snapshot, snapshot)
            assert bool(offered) is ((button, count) in expected)
    finally:
        backend.close()


@pytest.mark.parametrize("change", [
    "enabled_false", "sensitive_false", "focusable_false", "focusable_missing",
    "ancestor_false", "distant_ancestor_false", "application_false", "root_false",
    "parent_missing", "wrong_action_namespace",
    "wrong_state_namespace", "wrong_component_namespace", "nonleaf", "offscreen",
    "tiny_box", "unnamed", "unknown_state_value", "partial_action", "action_overflow",
    "action_name_overflow", "foreign_action_namespace", "foreign_state_namespace",
    "foreign_component_namespace", "no_positive_ancestor",
])
def test_invalid_native_evidence_never_adds_mouse_candidates(change):
    root = fixture_tree()
    node = canvas(root)
    parent = parent_of(root, node)
    if change in {"enabled_false", "sensitive_false", "focusable_false"}:
        node.set(STATE + change.removesuffix("_false"), "false")
    elif change == "focusable_missing":
        del node.attrib[STATE + "focusable"]
    elif change == "ancestor_false":
        parent.set(STATE + "enabled", "false")
    elif change == "distant_ancestor_false":
        next(n for n in root.iter() if n.tag == "frame").set(STATE + "sensitive", "false")
    elif change == "application_false":
        next(n for n in root.iter() if n.tag == "application").set(STATE + "enabled", "false")
    elif change == "root_false":
        root.set(STATE + "sensitive", "false")
    elif change == "parent_missing":
        del parent.attrib[STATE + "sensitive"]
    elif change.startswith("wrong_"):
        namespace = {"wrong_action_namespace": ACTION, "wrong_state_namespace": STATE,
                     "wrong_component_namespace": COMPONENT}[change]
        for key in list(node.attrib):
            if key.startswith(namespace):
                node.set("{urn:lookalike}" + key.split("}", 1)[1], node.attrib.pop(key))
    elif change.startswith("foreign_"):
        namespace = {"foreign_action_namespace": ACTION, "foreign_state_namespace": STATE,
                     "foreign_component_namespace": COMPONENT}[change]
        for key in list(node.attrib):
            if key.startswith(namespace):
                node.set(key.replace("accessibility.ubuntu.example.org", "unrelated.invalid"),
                         node.attrib.pop(key))
    elif change == "nonleaf":
        ET.SubElement(node, "label", {"name": "Child", STATE + "showing": "true"})
    elif change == "offscreen":
        node.set(COMPONENT + "screencoord", "(1900, 247)")
    elif change == "tiny_box":
        node.set(COMPONENT + "size", "(2, 2)")
    elif change == "unnamed":
        node.set("name", "")
    elif change == "unknown_state_value":
        node.set(STATE + "enabled", "sometimes")
    elif change == "partial_action":
        del node.attrib[ACTION + "open_kb"]
    elif change == "action_overflow":
        for i in range(17):
            node.set(ACTION + f"extra{i}_desc", "")
            node.set(ACTION + f"extra{i}_kb", "")
    elif change == "action_name_overflow":
        node.set(ACTION + "x" * 65 + "_desc", "")
    elif change == "no_positive_ancestor":
        for ancestor in root.iter():
            if ancestor is not node:
                ancestor.attrib.pop(STATE + "enabled", None)
                ancestor.attrib.pop(STATE + "sensitive", None)
    snapshot = observe(root)
    assert not executable_controls(snapshot, "click", AgentTarget(role="canvas"))
    control = next(c for c in snapshot.controls if c.role == "canvas")
    assert desktop_click_modes(control) == ()
    backend = OSWorldBackend(Desktop(root), pause=0)
    try:
        assert backend.compile(step(), snapshot, snapshot) == ()
        assert backend.env.calls == []
    finally:
        backend.close()


@pytest.mark.parametrize("flag,value", [("showing", None), ("showing", "false"), ("visible", "false")])
def test_hidden_native_items_are_not_exposed_at_all(flag, value):
    root = fixture_tree()
    node = canvas(root)
    node.attrib.pop(STATE + flag, None)
    if value is not None:
        node.set(STATE + flag, value)
    snapshot = observe(root)
    assert not any(control.role == "canvas" for control in snapshot.controls)


def test_native_payload_exposes_modes_and_nullable_facts_without_inventing_selectability():
    snapshot = observe(fixture_tree())
    payload = _desktop_snapshot_payload(snapshot)
    control = next(c for c in payload["controls"] if c.get("role") == "canvas")
    assert control["native"]["enabled"] is None and control["native"]["sensitive"] is None
    assert control["click_modes"] == [{"button": "left", "click_count": 1},
                                      {"button": "right", "click_count": 1},
                                      {"button": "left", "click_count": 2}]
    assert control["operations"] == ["click"]
    assert control["condition_kinds"] == ["exists", "absent", "focused"]
    instructions = _desktop_instructions(6, snapshot)
    assert "open/menu action names do not specify mouse gestures" in instructions
    assert "Do not infer checked/selected from focusable" in instructions
    assert "condition_kinds lists observable state capabilities" in instructions


@pytest.mark.parametrize("remove,button,count", [("menu", "right", 1), ("open", "left", 2)])
def test_immediate_planner_rejects_the_same_unsupported_gesture_as_the_backend(remove, button, count):
    root = fixture_tree()
    node = canvas(root)
    del node.attrib[ACTION + remove + "_desc"]
    del node.attrib[ACTION + remove + "_kb"]
    snapshot = observe(root)
    requested = step(button=button, count=count)
    raw = {"summary": "Unsupported gesture", "done": False, "steps": [requested.to_dict()]}
    plan = parse_plan(raw, snapshot, max_actions=1, profile="desktop")
    with pytest.raises(ValueError, match="no current executable control"):
        _validate_desktop_execution_plan(plan, snapshot, 1)
    backend = OSWorldBackend(Desktop(root), pause=0)
    try:
        assert backend.compile(requested, snapshot, snapshot) == ()
        assert backend.env.calls == []
    finally:
        backend.close()


def test_real_duplicate_native_targets_are_not_resolved_by_candidate_order():
    root = fixture_tree()
    node = canvas(root)
    duplicate = copy.deepcopy(node)
    duplicate.set(COMPONENT + "screencoord", "(450, 247)")
    parent_of(root, node).append(duplicate)
    desktop = Desktop(root)
    backend = OSWorldBackend(desktop, pause=0)
    try:
        snapshot = backend.observe()
        raw = {"summary": "Resolve ambiguity", "done": False, "steps": [step().to_dict()]}
        plan = parse_plan(raw, snapshot, max_actions=1, profile="desktop")
        with pytest.raises(ValueError, match="multiple executable controls"):
            _validate_desktop_execution_plan(plan, snapshot, 1)
        offered = backend.compile(step(), snapshot, snapshot)
        assert len(offered) == 2 and offered[0].arguments["box"] != offered[1].arguments["box"]
        assert desktop.calls == []
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["menu_removed", "enabled_reported", "moved", "parent_disabled"])
def test_bound_native_evidence_is_rechecked_before_any_input(change):
    desktop = Desktop()
    backend = OSWorldBackend(desktop, pause=0)
    try:
        snapshot = backend.observe()
        offered = backend.compile(step(button="right"), snapshot, snapshot)
        node = canvas(desktop.root)
        if change == "menu_removed":
            del node.attrib[ACTION + "menu_desc"]
            del node.attrib[ACTION + "menu_kb"]
        elif change == "enabled_reported":
            node.set(STATE + "enabled", "true")
        elif change == "moved":
            node.set(COMPONENT + "screencoord", "(450, 247)")
        else:
            parent_of(desktop.root, node).set(STATE + "sensitive", "false")
        after = backend.observe()
        assert after.fingerprint() != snapshot.fingerprint()
        receipt = backend.execute(offered[0], "o", "d")
        assert not receipt.success and desktop.calls == []
    finally:
        backend.close()


def test_native_navigation_counts_real_items_once_and_tracks_lost_mouse_affordances():
    desktop = Desktop()
    backend = OSWorldBackend(desktop, pause=0)
    try:
        before = backend.observe()
        navigation = step(navigation=True)
        offered = backend.compile(navigation, before, before)
        native_offered = [c for c in offered if c.arguments["bound_control"]["role"] == "canvas"]
        assert len(native_offered) == 1 and native_offered[0].arguments["click_count"] == 1
        node = canvas(desktop.root)
        del node.attrib[ACTION + "menu_desc"]
        del node.attrib[ACTION + "menu_kb"]
        after = backend.observe()
        assert _navigation_scope(before, navigation) != _navigation_scope(after, navigation)
        focused = copy.deepcopy(desktop.root)
        canvas(focused).set(STATE + "focused", "true")
        assert _navigation_scope(after, navigation) == _navigation_scope(observe(focused), navigation)
        assert desktop.calls == []
    finally:
        backend.close()


def test_native_evidence_cannot_turn_the_item_into_a_text_input_or_change_browser_identity():
    control = native_control(observe(fixture_tree()))
    plain = replace(control, native=None)
    assert control.identity() == plain.identity()
    for operation in ("fill", "check", "uncheck"):
        assert not executable_control(control, operation)
    assert not executable_control(control, "click", button="right", click_count=2)
    assert not executable_control(control, "click", button="left", click_count=True)


@pytest.mark.parametrize("field", ["focused", "selected", "selectable", "expanded", "expandable"])
@pytest.mark.parametrize("trusted_value", [None, "false", "unknown"])
def test_foreign_positive_native_states_cannot_manufacture_effects_or_capabilities(field, trusted_value):
    root = fixture_tree()
    node = canvas(root)
    if trusted_value is not None:
        node.set(STATE + field, trusted_value)
    node.set("{https://unrelated.invalid/ns/state}" + field, "true")
    snapshot = observe(root)
    control = native_control(snapshot)
    assert getattr(control, field) is False
    assert "selected" not in observable_condition_kinds(control)
    assert "expanded" not in observable_condition_kinds(control)
    predicate = AgentCondition("focused" if field == "focused" else (
        "selected" if field in {"selected", "selectable"} else "expanded"
    ), TARGET)
    evidence = condition_evidence(predicate, snapshot)
    assert evidence["outcome"] == ("false" if field == "focused" else "unknown")


@pytest.mark.parametrize("condition", ["focused", "selected"])
def test_foreign_post_input_state_does_not_pass_immediate_or_pending_verification(condition):
    def conflicting_after(root):
        node = canvas(root)
        node.set(STATE + condition, "false")
        node.set("{https://unrelated.invalid/ns/state}" + condition, "true")

    desktop = Desktop(after=conflicting_after)
    backend = OSWorldBackend(desktop, pause=0)
    try:
        snapshot = backend.observe()
        requested = step(post=(AgentCondition(condition, TARGET),))
        candidate = backend.compile(requested, snapshot, snapshot)[0]
        receipt = backend.execute(candidate, "o", "d")
        assert receipt.success and len(desktop.calls) == 1
        immediate = backend.verify(candidate, receipt)
        assert not immediate.passed and not immediate.details["postconditions_met"]
        pending = backend.verify_pending(candidate, receipt)
        assert not pending.passed and not pending.details["postconditions_met"]
        assert len(desktop.calls) == 1
    finally:
        backend.close()


def test_trusted_native_positive_state_is_observable_but_explicit_false_wins():
    root = fixture_tree()
    node = canvas(root)
    node.set(STATE + "selected", "true")
    selected = observe(root)
    assert condition_evidence(AgentCondition("selected", TARGET), selected)["outcome"] == "true"
    node.set("{https://unrelated.invalid/ns/state}selected", "false")
    assert condition_evidence(AgentCondition("selected", TARGET), observe(root))["outcome"] != "true"


@pytest.mark.parametrize("namespace", [
    "https://unrelated.invalid/ns/", "https://accessibility.ubuntu.example.org/ns/",
])
@pytest.mark.parametrize("expected", ["fabricated", ""])
def test_native_pointer_item_never_exposes_text_value_capability(namespace, expected):
    root = fixture_tree()
    node = canvas(root)
    node.set("{" + namespace + "state}editable", "true")
    node.set("{" + namespace + "value}value", "fabricated")
    node.text = "fabricated"
    snapshot = observe(root)
    control = native_control(snapshot)
    assert control.editable is False and control.value == ""
    assert not executable_control(control, "fill")
    assert "value_equals" not in observable_condition_kinds(control)
    assert condition_evidence(AgentCondition("value_equals", TARGET, expected), snapshot) == {
        "outcome": "unknown", "reason": "unsupported_state",
    }
    # The supported pointer gesture remains available; this is a capability
    # boundary, not a reason to mark the entire native item unusable.
    assert desktop_click_modes(control) == (("left", 1), ("right", 1), ("left", 2))
    payload = _desktop_snapshot_payload(snapshot)
    public = next(c for c in payload["controls"] if c.get("role") == "canvas")
    assert "value_equals" not in public["condition_kinds"]


@pytest.mark.parametrize("namespace", [
    "https://unrelated.invalid/ns/", "https://accessibility.ubuntu.example.org/ns/",
])
def test_native_canvas_value_attribute_cannot_confirm_a_whole_goal_without_input(namespace):
    root = fixture_tree()
    node = canvas(root)
    node.set("{" + namespace + "state}editable", "true")
    node.set("{" + namespace + "value}value", "fabricated")
    desktop = Desktop(root)
    backend = OSWorldBackend(desktop, pause=0)

    class ClaimPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Claim a field value", (), done=True, completion_conditions=(
                AgentCondition("value_equals", TARGET, "fabricated"),
            ))

    try:
        result = AgentRunner(backend, ClaimPlanner(), config=AgentConfig(
            mode="vlm", allow_actions=True, max_plans=2, max_failures=1,
        )).run("Set the field to the requested value")
        assert not result.success and result.plan_completion_claims == 0
        assert result.status == "stuck" and result.reason == "goal completion conditions remain unconfirmed"
        assert result.steps == 0 and desktop.calls == []
    finally:
        backend.close()
