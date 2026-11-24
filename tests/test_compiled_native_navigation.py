"""Compiler navigation exposes real native gestures without cloning logical steps."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from cua_jev.agent_planner import parse_plan
from cua_jev.agent_types import AgentCondition, AgentControl, AgentStep, AgentTarget
from cua_jev.desktop_semantics import native_navigation_click_modes
from cua_jev.osworld_backend import OSWorldBackend


@pytest.fixture
def recorded():
    fixture = json.loads((Path(__file__).parent / "fixtures/osworld_native_canvas.json").read_text())
    buffer = io.BytesIO()
    Image.new("RGB", tuple(fixture["screen_size"]), "white").save(buffer, "PNG")
    # The actual source parser supplies all native evidence; the tests below
    # replay only public snapshots through real compile/preflight/execute code.
    backend = OSWorldBackend(object(), pause=0)
    snapshot = backend._snapshot({"screenshot": buffer.getvalue(), "accessibility_tree": fixture["xml"]})
    native = next(control for control in snapshot.controls if control.native is not None)
    ordinary = AgentControl("ordinary", "Toolbar", "button", role="button", app=native.app,
                            window=native.window, box=(500, 100, 600, 200))
    return replace(snapshot, controls=(native, ordinary))


class Desktop:
    def __init__(self, snapshot, after=None):
        self.snapshot = snapshot
        self.after = after or snapshot
        self.actions = []

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause=0):
        self.actions.append(action)
        self.snapshot = self.after
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def intent(app="org.gnome.Nautilus"):
    return {"intent": "reach_edit", "label": "Reach and edit the requested field",
            "scope": {"app": app},
            "edit": {"operation": "fill", "target": {"name": "Requested field", "app": app},
                     "value": "requested value"}}


def compiled(snapshot):
    return parse_plan(intent(), snapshot, profile="desktop", max_actions=6).steps[0]


def gestures(candidates, ref):
    return [(c.arguments["button"], c.arguments["click_count"]) for c in candidates
            if c.arguments["bound_control"]["ref"] == ref]


def test_compiler_expands_only_recorded_native_item_and_preserves_logical_step(recorded):
    native, ordinary = recorded.controls
    step = compiled(recorded)
    backend = Backend(Desktop(recorded), pause=0)
    offered = backend.compile(step, recorded, recorded)
    assert step.native_navigation_gestures and step.navigation
    assert gestures(offered, native.ref) == [("left", 1), ("right", 1), ("left", 2)]
    assert gestures(offered, ordinary.ref) == [("left", 1)]
    assert len({c.arguments["bound_control"]["ref"] for c in offered}) == 2
    assert len(offered) == 4 and len({c.id for c in offered}) == 4
    assert all(c.intent == step.id and backend._issued[c.id].step is step for c in offered)
    assert all(c.arguments["bound_control"]["role"] == "canvas" for c in offered[:3])


@pytest.mark.parametrize("button,count", [("left", 1), ("right", 1), ("left", 2)])
def test_selected_gesture_binding_input_and_receipt_agree_without_faking_effect(recorded, button, count):
    native = recorded.controls[0]
    desktop, step = Desktop(recorded), compiled(recorded)
    backend = Backend(desktop, pause=0)
    offered = backend.compile(step, recorded, recorded)
    candidate = next(c for c in offered if c.arguments["bound_control"]["ref"] == native.ref
                     and (c.arguments["button"], c.arguments["click_count"]) == (button, count))
    binding = backend._issued[candidate.id]
    assert (binding.button, binding.click_count) == (button, count)
    assert binding.step is step and (step.button, step.click_count) == ("left", 1)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert (receipt.output["button"], receipt.output["click_count"]) == (button, count)
    assert desktop.actions == [{"action_type": "CLICK", "parameters": {
        "x": 373, "y": 289, "button": button, "num_clicks": count,
    }}]
    assert not backend.verify_pending(candidate, receipt).passed
    assert len(desktop.actions) == 1


@pytest.mark.parametrize("actions,modes", [
    (("menu",), [("left", 1), ("right", 1)]),
    (("open",), [("left", 1), ("left", 2)]),
    ((), []),
    (("unknown",), []),
])
def test_gestures_require_the_corresponding_observed_native_action(recorded, actions, modes):
    native = recorded.controls[0]
    native = replace(native, native=replace(native.native, actions=actions))
    state = replace(recorded, controls=(native, recorded.controls[1]))
    backend = Backend(Desktop(state), pause=0)
    offered = backend.compile(compiled(state), state, state)
    assert gestures(offered, native.ref) == modes
    assert gestures(offered, state.controls[1].ref) == [("left", 1)]


@pytest.mark.parametrize("change", [
    {"complete": False}, {"showing": False}, {"visible": False}, {"enabled": False},
    {"sensitive": False}, {"focusable": None}, {"leaf": False},
    {"parent_enabled": None}, {"ancestors_sensitive": False},
])
def test_incomplete_hidden_or_negative_native_evidence_does_not_expand(recorded, change):
    native = recorded.controls[0]
    native = replace(native, native=replace(native.native, **change))
    assert native_navigation_click_modes(native) == ()
    state = replace(recorded, controls=(native, recorded.controls[1]))
    backend = Backend(Desktop(state), pause=0)
    offered = backend.compile(compiled(state), state, state)
    assert len(offered) == 1 and offered[0].arguments["bound_control"]["ref"] == "ordinary"


@pytest.mark.parametrize("navigation", [False, True])
@pytest.mark.parametrize("button,count", [("left", 1), ("right", 1), ("left", 2)])
def test_legacy_explicit_gestures_are_not_expanded_or_reinterpreted(recorded, navigation, button, count):
    native = recorded.controls[0]
    condition = AgentCondition("focused", AgentTarget(name=native.label, app=native.app))
    step = AgentStep("legacy", "click", label="Inspect item", navigation=navigation,
                     target=AgentTarget(app=native.app) if navigation else AgentTarget(
                         name=native.label, app=native.app,
                     ), button=button, click_count=count,
                     postconditions=(condition,))
    backend = Backend(Desktop(recorded), pause=0)
    offered = backend.compile(step, recorded, recorded)
    assert gestures(offered, native.ref) == [(button, count)]
    assert not step.native_navigation_gestures
    assert all((c.arguments["button"], c.arguments["click_count"]) == (button, count) for c in offered)


@pytest.mark.parametrize("placement", ["step", "envelope", "edit", "scope"])
def test_raw_model_wire_cannot_inject_internal_expansion_marker(recorded, placement):
    raw = intent()
    if placement == "step":
        raw = {"summary": "Inspect", "done": False, "steps": [{
            "id": "nav", "operation": "click", "navigation": True, "label": "Inspect",
            "target": raw["scope"], "native_navigation_gestures": True,
            "postconditions": [{"kind": "exists", "target": {"name": "Field", "app": "Files"}}],
        }]}
    elif placement == "envelope":
        raw["native_navigation_gestures"] = True
    else:
        raw[placement]["native_navigation_gestures"] = True
    with pytest.raises(ValueError):
        parse_plan(raw, recorded, profile="desktop", max_actions=6)


@pytest.mark.parametrize("limit,profile", [(1, "desktop"), (6, "browser")])
def test_baseline_recovery_and_browser_cannot_get_compiler_navigation(recorded, limit, profile):
    with pytest.raises(ValueError):
        parse_plan(intent(), recorded, profile=profile, max_actions=limit)


@pytest.mark.parametrize("change", ["capability", "geometry", "gesture_argument"])
def test_changed_capability_or_binding_is_rejected_before_any_input(recorded, change):
    native = recorded.controls[0]
    desktop, step = Desktop(recorded), compiled(recorded)
    backend = Backend(desktop, pause=0)
    candidate = next(c for c in backend.compile(step, recorded, recorded)
                     if c.arguments["button"] == "right")
    if change == "gesture_argument":
        arguments = copy.deepcopy(candidate.arguments)
        arguments["button"] = "left"
        candidate = replace(candidate, arguments=arguments)
    else:
        updated = replace(native, box=(300, 400, 400, 500)) if change == "geometry" else replace(
            native, native=replace(native.native, actions=("open",)),
        )
        desktop.snapshot = replace(recorded, controls=(updated, recorded.controls[1]))
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and desktop.actions == []
    assert not backend.verify(candidate, receipt).passed


@pytest.mark.parametrize("target_count,expected_actions", [(85, 255), (86, 0)])
def test_budget_is_checked_after_expansion_without_registering_a_prefix(
    recorded, target_count, expected_actions,
):
    native = recorded.controls[0]
    controls = tuple(replace(native, ref=f"native-{index}", label=f"Item {index}",
                             box=(index * 5, 100, index * 5 + 5, 200))
                     for index in range(target_count))
    state = replace(recorded, controls=controls)
    backend = Backend(Desktop(state), pause=0)
    offered = backend.compile(compiled(state), state, state)
    assert len(offered) == len(backend._issued) == expected_actions
    if offered:
        assert len({c.arguments["bound_control"]["ref"] for c in offered}) == target_count


def test_marked_logical_step_keeps_receipt_authorized_handoff_across_gesture_choices(recorded):
    native, ordinary = recorded.controls
    launcher = replace(ordinary, ref="launcher", label="Files", app="Shell", window="")
    before = replace(recorded, controls=(launcher,))
    frame = AgentControl("frame", "Files", "container", role="frame", app=native.app,
                         window=native.window, active=True)
    after = replace(recorded, controls=(frame, replace(native, active=True), replace(ordinary, active=True)))
    desktop, backend = Desktop(before, after), None
    backend = Backend(desktop, pause=0)
    raw = intent()
    raw["scope"] = {"app": "Shell"}
    step = parse_plan(raw, before, profile="desktop", max_actions=6).steps[0]
    launcher_candidate = backend.compile(step, before, before)[0]
    receipt = backend.execute(launcher_candidate, "observation", "decision")
    assert backend.verify(launcher_candidate, receipt).passed
    assert receipt.output["navigation_handoff"] and receipt.output["navigation_app"] == native.app
    offered = backend.compile(step, before, after)
    assert len(offered) == 4 and gestures(offered, native.ref) == [("left", 1), ("right", 1), ("left", 2)]
    assert all(backend._issued[c.id].step is step for c in offered)
    assert all(c.arguments["navigation_app"] == native.app for c in offered)
    assert all(c.arguments["bound_control"]["app"] == native.app for c in offered)


def test_gesture_expansion_does_not_waive_incomplete_snapshot(recorded):
    state = replace(recorded, a11y_complete=False)
    backend = Backend(Desktop(state), pause=0)
    assert backend.compile(compiled(state), state, state) == ()


def test_duplicate_public_binding_cannot_inflate_expanded_choices(recorded):
    native, ordinary = recorded.controls
    duplicate = replace(recorded, controls=(native, native, ordinary))
    backend = Backend(Desktop(duplicate), pause=0)
    offered = backend.compile(compiled(recorded), recorded, duplicate)
    assert len(offered) == 4
    assert len({json.dumps(c.arguments["bound_control"], sort_keys=True) for c in offered}) == 2


@pytest.mark.parametrize("button,count", [("right", 1), ("left", 2)])
def test_actual_new_controls_verify_progress_and_preserve_one_shot_cache(recorded, button, count):
    item = replace(recorded.controls[1], ref="new-item", label="Observed menu item", role="menuitem")
    after = replace(recorded, controls=(*recorded.controls, item))
    desktop, step = Desktop(recorded, after), compiled(recorded)
    backend = Backend(desktop, pause=0)
    candidate = next(c for c in backend.compile(step, recorded, recorded)
                     if (c.arguments["button"], c.arguments["click_count"]) == (button, count))
    receipt = backend.execute(candidate, "observation", "decision")
    verification = backend.verify(candidate, receipt)
    assert verification.passed and verification.details["accessibility_progress"]
    assert not verification.details["postconditions_met"]
    assert not verification.details["task_success"]
    assert backend.take_verified_post_snapshot(candidate, receipt)[0] is after
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    new_choices = backend.compile(step, recorded, after)
    assert len(new_choices) == 5
    assert all(backend._issued[c.id].step is step for c in new_choices)
