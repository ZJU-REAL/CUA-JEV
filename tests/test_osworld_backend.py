"""OSWorld protocol checks without a VM, network model or task evaluator."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import (
    condition_evidence,
    condition_holds,
    conditions_hold,
    executable_controls,
    matching_controls,
    observable_condition_kinds,
)
from cua_jev.models import Channel
from cua_jev.osworld_backend import OSWorldBackend, _navigation_scope, accessibility_context, screenshot_image


def png(color="white", *, cursor=False, size=(800, 600)):
    image = Image.new("RGB", size, color)
    if cursor:
        ImageDraw.Draw(image).rectangle((40, 40, 41, 48), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class Environment:
    def __init__(self):
        self.frame = png()
        self.xml = '<desktop><frame name="Editor" active="true"><button name="Save"/></frame></desktop>'
        self.calls = []
        self.change_after = True
        self.controller = SimpleNamespace(execute_python_command=lambda command: {"status": "success"})

    def _get_obs(self):
        return {
            "screenshot": self.frame,
            "accessibility_tree": self.xml,
            "instruction": "PRIVATE TEST INSTRUCTION",
            "terminal": "PRIVATE TERMINAL",
        }

    def step(self, action, pause=2):
        self.calls.append((action, pause))
        self.controller.execute_python_command("trusted OSWorld generated command")
        if self.change_after:
            self.frame = png("gray")
        return self._get_obs(), 999, True, {}


@pytest.fixture
def env():
    return Environment()


@pytest.fixture
def backend(env):
    value = OSWorldBackend(env, pause=0)
    yield value
    value.close()


def candidate(backend, op="visual_click", value=""):
    snapshot = backend.observe()
    visual = op.startswith("visual_")
    step = AgentStep(
        "s1", op, value=value, box=(100, 100, 300, 300) if visual else None, label="Save" if visual else ""
    )
    return backend.compile(step, snapshot, snapshot)[0]


def test_screenshot_and_context_do_not_include_task_metadata(backend):
    snapshot = backend.observe()
    assert snapshot.image.region == (0, 0, 800, 600)
    assert snapshot.image.jpeg.startswith(b"\xff\xd8")
    assert snapshot.title == "Editor"
    assert "Save" in snapshot.text
    assert {control.label for control in snapshot.controls} == {"Editor", "Save"}
    assert all(control.box is None for control in snapshot.controls)
    assert "PRIVATE" not in str(snapshot.to_dict())
    assert set(backend.last_observation) == {"screenshot", "accessibility_tree"}


def test_visibility_focus_and_malformed_accessibility():
    xml = """<desktop xmlns:s="urn:state"><frame name="Editor" s:active="true">
      <entry name="Find" s:focused="true">needle</entry>
      <frame name="Hidden" s:showing="false"><button name="Secret"/></frame>
      <password-text name="Password">sensitive</password-text>
    </frame></desktop>"""
    title, context = accessibility_context(xml)
    assert title == "Editor"
    assert "needle [focused]" in context
    assert "Secret" not in context and "sensitive" not in context
    assert accessibility_context("<bad") == ("OSWorld desktop", "")
    assert accessibility_context('<!DOCTYPE a [<!ENTITY b "secret">]><a>&b;</a>')[1] == ""


def atspi_desktop(
    *, clock="02∶31", reverse_apps=False, hidden="Not shown", buttons=None, focus=False, editor_clock="03:45"
):
    # Minimal real OSWorld AT-SPI shape: omitted state entries mean false;
    # applications are containers without showing/visible flags of their own.
    state = 's:showing="true" s:visible="true"'
    clock_label = (
        f'<label name="Oct 23 {clock}" {state} a:toolkit="clutter" '
        'c:screencoord="(937, 0)" c:size="(116, 27)">'
        '<text s:showing="true" a:toolkit="clutter" c:screencoord="(952, 4)" c:size="(86, 18)">'
        f"Oct 23 {clock}</text></label>"
    )
    shell = (
        f'<application name="gnome-shell"><window {state} s:focused="true">'
        f'<panel {state}>{clock_label}<menu name="System" {state}/></panel>'
        f'<panel s:visible="true"><label name="{hidden}" {state}/></panel>'
        "</window></application>"
    )
    buttons = buttons or ["Save", "Cancel"]
    controls = "".join(f'<push-button name="{name}" {state}/>' for name in buttons)
    controls += f"<entry {state}" + (' s:focused="true"' if focus else "") + "/>"
    editor = (
        f'<application name="Editor"><frame name="Document" {state} s:active="true">'
        f'{controls}<label name="{editor_clock}" {state} a:toolkit="clutter" '
        'c:screencoord="(937, 0)" c:size="(116, 27)"/></frame></application>'
    )
    noise = (
        '<application name="gsd-power"/><application name="hidden-app"><frame name="Hidden"/></application>'
    )
    apps = [shell, editor, noise]
    if reverse_apps:
        apps.reverse()
    return (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:a="https://accessibility.ubuntu.example.org/ns/attributes" '
        'xmlns:c="https://accessibility.ubuntu.example.org/ns/component">'
        + "".join(apps)
        + "</desktop-frame>"
    )


def test_real_atspi_showing_semantics_drop_hidden_subtrees_and_empty_apps():
    title, context = accessibility_context(atspi_desktop())
    assert title == "Document"
    assert "Not shown" not in context and "hidden-app" not in context and "gsd-power" not in context
    assert "Save" in context and "System" in context
    assert "window:  [focused]" in context


def test_application_enumeration_and_hidden_changes_do_not_invalidate_scope(backend, env):
    env.xml = atspi_desktop()
    before = backend.observe()
    env.xml = atspi_desktop(reverse_apps=True, hidden="Random changing hidden menu")
    after = backend.observe()
    assert before.to_dict() == after.to_dict()
    assert before.scope_key() == after.scope_key()


@pytest.mark.parametrize(
    "change",
    [
        {"buttons": ["Cancel", "Save"]},
        {"focus": True},
        {"editor_clock": "03:46"},
    ],
)
def test_visible_internal_order_focus_and_application_clock_remain_significant(backend, env, change):
    env.xml = atspi_desktop()
    before = backend.observe()
    env.xml = atspi_desktop(**change)
    after = backend.observe()
    assert before.scope_key() != after.scope_key()
    assert backend.compile(AgentStep("stale", "hotkey", value="ctrl+s"), before, after) == ()


def test_only_gnome_topbar_clock_tick_is_ignored_for_scope(backend, env):
    env.xml = atspi_desktop()
    before = backend.observe()
    env.xml = atspi_desktop(clock="02∶32")
    after = backend.observe()
    assert before.text != after.text
    assert "02∶32" in after.text
    assert before.scope_key() == after.scope_key()
    assert len(backend.compile(AgentStep("fresh", "hotkey", value="ctrl+s"), before, after)) == 1


def test_clock_only_tick_is_not_action_effect(backend, env):
    env.xml = atspi_desktop()
    env.change_after = False
    original = env.step

    def step(action, pause):
        result = original(action, pause)
        env.xml = atspi_desktop(clock="02∶32")
        return env._get_obs(), *result[1:]

    env.step = step
    action = candidate(backend, "hotkey", "ctrl+s")
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert not backend.verify(action, receipt).passed


def test_clock_outside_topbar_is_significant(backend, env):
    env.xml = atspi_desktop().replace("(937, 0)", "(937, 150)")
    before = backend.observe()
    env.xml = atspi_desktop(clock="02∶32").replace("(937, 0)", "(937, 150)")
    assert before.scope_key() != backend.observe().scope_key()


def test_a11y_coordinates_use_original_screen_dimensions(backend, env):
    env.frame = png(size=(1920, 1080))
    env.xml = atspi_desktop().replace('name="Save"', 'name="Save" c:screencoord="(96, 54)" c:size="(96, 54)"')
    snapshot = backend.observe()
    # JPEG was resized for the VLM, but normalized coordinates cover the full
    # original screen and are converted back by the same backend at execution.
    assert snapshot.image.region == (0, 0, 1920, 1080)
    assert "push-button: Save box_1000=[50, 50, 100, 100]" in snapshot.text
    assert any("bounds_px=[96, 54, 192, 108]" in line for line in snapshot.semantic_text)


@pytest.mark.parametrize(
    "attributes",
    [
        'c:screencoord="(-1, 54)" c:size="(96, 54)"',
        'c:screencoord="(1919, 54)" c:size="(96, 54)"',
        'c:screencoord="(96, 1080)" c:size="(96, 54)"',
        'c:screencoord="(96, 54)" c:size="(0, 54)"',
        'c:screencoord="(96, 54)" c:size="(-10, 54)"',
        'c:screencoord="(96, 54)"',
        'c:size="(96, 54)"',
        'c:screencoord="(nan, 54)" c:size="(96, 54)"',
        'c:screencoord="(96, 54)" c:size="(999999999999999999999, 54)"',
    ],
)
def test_invalid_or_offscreen_bounds_never_reach_model_as_coordinates(backend, env, attributes):
    env.frame = png(size=(1920, 1080))
    env.xml = atspi_desktop().replace('name="Save"', f'name="Save" {attributes}')
    snapshot = backend.observe()
    line = next(line for line in snapshot.text.splitlines() if line.startswith("push-button: Save"))
    assert "box_1000" not in line


def test_even_sub_normalization_target_movement_invalidates_plan(backend, env):
    env.frame = png(size=(1920, 1080))
    env.xml = atspi_desktop().replace('name="Save"', 'name="Save" c:screencoord="(97, 54)" c:size="(96, 54)"')
    before = backend.observe()
    env.xml = env.xml.replace("(97, 54)", "(98, 54)")
    after = backend.observe()
    assert before.text == after.text  # Same rounded normalized box.
    assert before.scope_key() != after.scope_key()
    assert backend.compile(AgentStep("stale", "hotkey", value="ctrl+s"), before, after) == ()


@pytest.mark.parametrize("change", ["move", "focus"])
def test_normalized_clock_retains_geometry_and_focus(backend, env, change):
    env.frame = png(size=(1920, 1080))
    env.xml = atspi_desktop()
    before = backend.observe()
    if change == "move":
        env.xml = env.xml.replace("(937, 0)", "(938, 0)", 1)
    else:
        env.xml = env.xml.replace('name="Oct 23 02∶31"', 'name="Oct 23 02∶31" s:focused="true"')
    after = backend.observe()
    assert before.scope_key() != after.scope_key()
    clock = next(line for line in after.semantic_text if "GNOME top-bar clock" in line)
    assert "box_1000=" in clock and "bounds_px=" in clock
    if change == "focus":
        assert "[focused]" in clock


def test_click_uses_typed_coordinates_and_effect_evidence(backend, env):
    action = candidate(backend)
    receipt = backend.execute(action, "obs", "decision")
    assert action.channel == Channel.GUI
    assert receipt.success
    assert env.calls == [
        (
            {
                "action_type": "CLICK",
                "parameters": {
                    "x": 160,
                    "y": 120,
                    "button": "left",
                    "num_clicks": 1,
                },
            },
            0,
        )
    ]
    verification = backend.verify(action, receipt)
    assert verification.passed
    assert verification.details["effects"] == ["image"]
    assert verification.details["task_success"] is False
    assert receipt.output["environment_steps"] == 1
    assert set(receipt.output["timings_ms"]) == {"preflight", "action", "effect_check"}


def test_successful_submission_without_observed_effect_is_not_verified(backend, env):
    env.change_after = False
    action = candidate(backend)
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert not backend.verify(action, receipt).passed


def test_a11y_change_is_effect_even_without_image_change(backend, env):
    env.change_after = False
    original = env.step

    def step(action, pause):
        result = original(action, pause)
        env.xml = env.xml.replace('name="Save"', 'name="Saved"')
        return env._get_obs(), *result[1:]

    env.step = step
    action = candidate(backend)
    receipt = backend.execute(action, "obs", "decision")
    assert backend.verify(action, receipt).details["effects"] == ["accessibility"]


@pytest.mark.parametrize("mutation", ["pixels", "a11y", "size"])
def test_stale_candidate_never_sends_input(backend, env, mutation):
    action = candidate(backend)
    if mutation == "pixels":
        env.frame = png("black")
    elif mutation == "a11y":
        env.xml = env.xml.replace("Save", "Discard")
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (900, 700), "white").save(buffer, format="PNG")
        env.frame = buffer.getvalue()
    receipt = backend.execute(action, "obs", "decision")
    assert not receipt.success
    assert env.calls == []


def test_tiny_cursor_change_does_not_invalidate_decision(backend, env):
    action = candidate(backend)
    env.frame = png(cursor=True)
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert len(env.calls) == 1


@pytest.mark.parametrize(
    "operation,value",
    [
        ("type_text", "replacement"),
        ("hotkey", "ctrl+a"),
        ("key_press", "enter"),
        ("press", "Enter"),
        ("scroll", "down"),
        ("back", ""),
        ("wait", "250"),
    ],
)
def test_changed_screen_during_planning_rejects_nonvisual_actions_without_a11y(
    backend,
    env,
    operation,
    value,
):
    env.xml = None
    planned = backend.observe()
    env.frame = png("black")
    current = backend.observe()
    # This is precisely the case that textual runner freshness cannot detect.
    assert planned.fingerprint() == current.fingerprint()
    assert backend.compile(AgentStep("stale", operation, value=value), planned, current) == ()
    assert env.calls == []


def test_cursor_change_during_planning_still_allows_keyboard(backend, env):
    env.xml = None
    planned = backend.observe()
    env.frame = png(cursor=True)
    current = backend.observe()
    offered = backend.compile(AgentStep("current", "hotkey", value="ctrl+a"), planned, current)
    assert len(offered) == 1
    assert backend.execute(offered[0], "obs", "decision").success


def test_tampering_and_replay_are_rejected(backend, env):
    action = candidate(backend)
    forged = replace(action, arguments={**action.arguments, "value": "arbitrary"})
    assert not backend.execute(forged, "obs", "decision").success
    assert env.calls == []
    action = candidate(backend)
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert not backend.execute(action, "obs2", "decision2").success
    assert len(env.calls) == 1
    assert not backend.verify(forged, receipt).passed


@pytest.mark.parametrize(
    ("op", "value", "kind", "parameters"),
    [
        ("hotkey", "ctrl+shift+p", "HOTKEY", {"keys": ["ctrl", "shift", "p"]}),
        ("type_text", "font size", "TYPING", {"text": "font size"}),
        ("key_press", "f1", "PRESS", {"key": "f1"}),
        ("press", "ArrowDown", "PRESS", {"key": "down"}),
        ("scroll", "up", "SCROLL", {"dx": 0, "dy": 3}),
        ("back", "", "HOTKEY", {"keys": ["alt", "left"]}),
    ],
)
def test_keyboard_and_scroll_actions_are_structured(backend, env, op, value, kind, parameters):
    action = candidate(backend, op, value)
    assert backend.execute(action, "obs", "decision").success
    assert env.calls == [({"action_type": kind, "parameters": parameters}, 0)]


def test_fill_counts_all_environment_steps(backend, env):
    action = candidate(backend, "visual_fill", "replacement")
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert [a[0]["action_type"] for a in env.calls] == ["CLICK", "HOTKEY", "TYPING"]
    assert receipt.output["environment_steps"] == 3


@pytest.mark.parametrize(
    "step",
    [
        AgentStep("s1", "hotkey", value="ctrl+a+delete"),
        AgentStep("s1", "hotkey", value="ctrl+ctrl+p"),
        AgentStep("s1", "key_press", value="os.system('x')"),
        AgentStep("s1", "type_text", value=""),
        AgentStep("s1", "type_text", ref="invented", value="value"),
        AgentStep("s1", "type_text", value="a\x00b"),
        AgentStep("s1", "visual_click", box=(-1, 0, 100, 100), label="Bad"),
        AgentStep("s1", "click", ref="e0"),
        AgentStep("s1", "wait", value="100000"),
    ],
)
def test_unsupported_or_malformed_actions_never_compile(backend, step):
    snapshot = backend.observe()
    assert backend.compile(step, snapshot, snapshot) == ()


def test_controller_error_is_not_silently_accepted(env):
    def original(command):
        return {"status": "error", "message": "PRIVATE RESPONSE"}

    env.controller.execute_python_command = original
    backend = OSWorldBackend(env, pause=0)
    action = candidate(backend)
    receipt = backend.execute(action, "obs", "decision")
    assert not receipt.success
    assert "PRIVATE" not in str(receipt.to_dict())
    assert not backend.verify(action, receipt).passed
    backend.close()
    assert env.controller.execute_python_command is original


def test_exhausted_controller_transport_is_failure(env):
    def original(command):
        return None

    env.controller.execute_python_command = original
    backend = OSWorldBackend(env, pause=0)
    action = candidate(backend)
    receipt = backend.execute(action, "obs", "decision")
    assert not receipt.success
    assert not backend.verify(action, receipt).passed
    assert receipt.output["environment_steps"] == 1
    backend.close()
    assert env.controller.execute_python_command is original


@pytest.mark.parametrize("operation", ["type_text", "visual_fill"])
def test_non_ascii_typing_rejected_before_any_input(backend, env, operation):
    snapshot = backend.observe()
    step = AgentStep(
        "unicode",
        operation,
        value="中文 café",
        box=(100, 100, 300, 300) if operation == "visual_fill" else None,
        label="Field",
    )
    assert backend.compile(step, snapshot, snapshot) == ()
    assert env.calls == []


@pytest.mark.parametrize(
    "operation,value",
    [
        ("key_press", "'"),
        ("key_press", "\\"),
        ("hotkey", "ctrl+'"),
        ("hotkey", "alt+\\"),
    ],
)
def test_unescaped_upstream_key_names_are_rejected(backend, env, operation, value):
    snapshot = backend.observe()
    assert backend.compile(AgentStep("bad-key", operation, value=value), snapshot, snapshot) == ()
    assert env.calls == []


def test_wait_is_bounded_without_osworld_wait_sentinel(backend, env, monkeypatch):
    slept = []
    monkeypatch.setattr("cua_jev.osworld_backend.time.sleep", slept.append)
    action = candidate(backend, "wait", "250")
    backend.execute(action, "obs", "decision")
    assert slept == [0.25]
    assert env.calls == []


def test_close_does_not_close_vm(backend, env):
    backend.close()
    backend.close()
    with pytest.raises(RuntimeError):
        backend.observe()
    assert env._get_obs()["screenshot"]


def semantic_ui(items, *, window="Preferences", app="Demo"):
    return (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:c="https://accessibility.ubuntu.example.org/ns/component">'
        f'<application name="{app}"><frame name="{window}" s:showing="true" s:active="true">'
        '<panel name="General" s:showing="true">' + items + "</panel></frame></application></desktop-frame>"
    )


def semantic_control(role="push-button", name="Save", *, x=80, enabled=True, states="", value="", child=""):
    enabled_flags = 's:enabled="true" s:sensitive="true"' if enabled else ""
    return (
        f'<{role} name="{name}" s:showing="true" s:visible="true" {enabled_flags} {states} '
        f'c:screencoord="({x}, 120)" c:size="(120, 40)">{value}{child}</{role}>'
    )


def test_semantic_controls_contain_live_context_and_true_state(backend, env):
    env.xml = semantic_ui(
        semantic_control("entry", "Search", states='s:editable="true" s:focused="true"', value="abc")
        + semantic_control("check-box", "Follow up", x=240, states='s:checked="true"')
        + semantic_control(name="Unavailable", enabled=False)
    )
    controls = tuple(control for control in backend.observe().controls if control.tag != "container")
    assert len(controls) == 3
    entry, checkbox, disabled = controls
    assert (entry.role, entry.value, entry.editable, entry.focused) == ("textbox", "abc", True, True)
    assert (entry.app, entry.window, entry.active) == ("Demo", "Preferences", True)
    assert entry.ancestors == ("frame: Preferences", "panel: General")
    assert entry.box == (100, 200, 250, 267)
    assert checkbox.checked and checkbox.input_type == "checkbox"
    assert disabled.disabled


def test_unnamed_control_uses_visible_label_without_child_menu_text(backend, env):
    label = '<panel s:showing="true"><label s:showing="true" name="Activities"/></panel>'
    menu = semantic_control("menu", "", child=semantic_control("menu-item", "Delete"))
    env.xml = semantic_ui(
        semantic_control("toggle-button", "", states='s:focusable="true"', child=label) + menu
    )
    labels = [control.label for control in backend.observe().controls if control.tag in {"button", "input"}]
    assert labels == ["Activities", "Delete"]


def test_semantic_target_rebinds_to_new_screen_coordinates(backend, env):
    env.xml = semantic_ui(semantic_control(name="Open"), window="Main")
    planned = backend.observe()
    env.xml = semantic_ui(semantic_control(name="Save", x=400))
    current = backend.observe()
    step = AgentStep(
        "save", "click", target=AgentTarget(name="Save", role="push-button", window="Preferences")
    )
    offered = backend.compile(step, planned, current)
    assert len(offered) == 1
    receipt = backend.execute(offered[0], "obs", "decision")
    assert receipt.success
    assert env.calls[0][0]["parameters"]["x"] == 460
    assert offered[0].arguments["route"] == "osworld-a11y"


def test_multiple_real_matches_produce_distinct_candidates_but_disabled_filtered(backend, env):
    env.xml = semantic_ui(
        semantic_control(x=80) + semantic_control(x=240) + semantic_control(x=400, enabled=False)
    )
    snapshot = backend.observe()
    target = AgentTarget(name="save", role="button", app="demo", ancestor="general")
    assert len(matching_controls(snapshot, target)) == 3
    offered = backend.compile(AgentStep("save", "click", target=target), snapshot, snapshot)
    assert len(offered) == 2
    assert len({action.arguments["box"] for action in offered}) == 2
    assert len({action.description for action in offered}) == 2


def test_single_semantic_step_candidates_are_bounded(backend, env):
    env.xml = semantic_ui("".join(semantic_control(x=80 + i % 4 * 120) for i in range(80)))
    snapshot = backend.observe()
    offered = backend.compile(AgentStep("save", "click", target=AgentTarget(name="Save")), snapshot, snapshot)
    assert len(offered) == 64


def test_rebound_candidate_still_rejects_new_movement_before_execution(backend, env):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    offered = backend.compile(AgentStep("save", "click", target=AgentTarget(name="Save")), snapshot, snapshot)
    env.xml = semantic_ui(semantic_control(x=240))
    assert not backend.execute(offered[0], "obs", "decision").success
    assert env.calls == []


def test_fill_checks_focus_and_verifies_actual_text_without_leaking_typed_value(backend, env):
    def tree(value="old", focused=False):
        flags = 's:editable="true"' + (' s:focused="true"' if focused else "")
        return semantic_ui(semantic_control("entry", "Search", states=flags, value=value))

    env.xml = tree()
    original = env.step

    def step(action, pause):
        response = original(action, pause)
        if action["action_type"] == "CLICK":
            env.xml = tree(focused=True)
        elif action["action_type"] == "TYPING":
            env.xml = tree(action["parameters"]["text"], focused=True)
        return env._get_obs(), *response[1:]

    env.step = step
    snapshot = backend.observe()
    target = AgentTarget(name="Search", role="textbox")
    plan_step = AgentStep(
        "fill",
        "fill",
        value="private literal",
        target=target,
        postconditions=(AgentCondition("value_equals", target, "private literal"),),
    )
    action = backend.compile(plan_step, snapshot, snapshot)[0]
    assert "private literal" not in action.description
    assert "15 characters" in action.description
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert backend.verify(action, receipt).passed
    assert [item[0]["action_type"] for item in env.calls] == ["CLICK", "HOTKEY", "TYPING"]


@pytest.mark.parametrize("lose_focus_after", ["CLICK", "HOTKEY"])
def test_fill_does_not_type_into_a_new_dialog(backend, env, lose_focus_after):
    entry = semantic_control("entry", "Search", states='s:editable="true" s:focused="true"')
    env.xml = semantic_ui(entry)
    original = env.step

    def step(action, pause):
        response = original(action, pause)
        if action["action_type"] == lose_focus_after:
            env.xml = semantic_ui(entry, window="Unrelated dialog")
        return env._get_obs(), *response[1:]

    env.step = step
    snapshot = backend.observe()
    action = backend.compile(
        AgentStep("fill", "fill", value="never typed", target=AgentTarget(name="Search")), snapshot, snapshot
    )[0]
    assert not backend.execute(action, "obs", "decision").success
    assert all(item[0]["action_type"] != "TYPING" for item in env.calls)


@pytest.mark.parametrize("operation", ["fill", "check"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_post_input_target_motion_uses_unique_semantic_identity(backend, env, operation, duplicate):
    role = "entry" if operation == "fill" else "check-box"
    states = 's:editable="true" s:focused="true"' if operation == "fill" else ""
    env.xml = semantic_ui(semantic_control(role, "Target", states=states))
    original = env.step

    def step(action, pause):
        result = original(action, pause)
        if action["action_type"] == ("TYPING" if operation == "fill" else "CLICK"):
            changed = semantic_control(
                role,
                "Target",
                x=240,
                states=states if operation == "fill" else 's:checked="true"',
                value="updated" if operation == "fill" else "",
            )
            if duplicate:
                changed += semantic_control(
                    role,
                    "Target",
                    x=400,
                    states=states if operation == "fill" else 's:checked="true"',
                    value="updated" if operation == "fill" else "",
                )
            env.xml = semantic_ui(changed)
        return env._get_obs(), *result[1:]

    env.step = step
    snapshot = backend.observe()
    target = AgentTarget(name="Target")
    # Use a general postcondition so this tests direct target verification;
    # it cannot accidentally pass only because a matching state exists.
    step = AgentStep(
        "change",
        operation,
        value="updated" if operation == "fill" else "",
        target=target,
        postconditions=(AgentCondition("window_contains", value="Preferences"),),
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success
    assert backend.verify(action, receipt).passed is not duplicate


def test_fill_keeps_strict_geometry_before_typing(backend, env):
    states = 's:editable="true" s:focused="true"'
    env.xml = semantic_ui(semantic_control("entry", "Search", states=states))
    original = env.step

    def click_moves_input(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(semantic_control("entry", "Search", x=240, states=states))
        return env._get_obs(), *result[1:]

    env.step = click_moves_input
    snapshot = backend.observe()
    action = backend.compile(
        AgentStep("fill", "fill", value="never typed", target=AgentTarget(name="Search")), snapshot, snapshot
    )[0]
    receipt = backend.execute(action, "obs", "decision")
    assert not receipt.success
    assert [call[0]["action_type"] for call in env.calls] == ["CLICK"]


def test_postconditions_cannot_be_replaced_by_any_image_change(backend, env):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    step = AgentStep(
        "save",
        "click",
        target=AgentTarget(name="Save"),
        postconditions=(AgentCondition("exists", AgentTarget(name="Success")),),
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and "image" in receipt.output["effects"]
    assert not backend.verify(action, receipt).passed
    assert not receipt.output["postconditions_met"]


def test_check_already_satisfied_has_observed_state_evidence(backend, env):
    env.xml = semantic_ui(semantic_control("check-box", "Follow up", states='s:checked="true"'))
    snapshot = backend.observe()
    target = AgentTarget(name="Follow up")
    action = backend.compile(
        AgentStep("check", "check", target=target, postconditions=(AgentCondition("checked", target),)),
        snapshot,
        snapshot,
    )[0]
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and backend.verify(action, receipt).passed
    assert env.calls == []


def test_guarded_keyboard_rebinds_only_when_current_focus_is_uniquely_observed(backend, env):
    planned = backend.observe()
    env.xml = semantic_ui(semantic_control("entry", "Search", states='s:editable="true" s:focused="true"'))
    current = backend.observe()
    target = AgentTarget(name="Search")
    step = AgentStep(
        "type",
        "type_text",
        value="secret phrase",
        preconditions=(AgentCondition("focused", target),),
        postconditions=(AgentCondition("value_equals", target, "secret phrase"),),
    )
    offered = backend.compile(step, planned, current)
    assert len(offered) == 1 and "secret phrase" not in offered[0].description
    env.xml = semantic_ui(semantic_control("entry", "Search", states='s:editable="true"'))
    assert not backend.execute(offered[0], "obs", "decision").success
    assert env.calls == []
    wrong_focus = replace(step, preconditions=(AgentCondition("exists", target),))
    assert backend.compile(wrong_focus, planned, current) == ()


@pytest.mark.parametrize(
    ("role", "states", "enabled"),
    [
        ("push-button", 's:focused="true"', True),
        ("entry", 's:focused="true"', True),
        ("entry", 's:focused="true" s:editable="true"', False),
        ("check-box", 's:focused="true" s:editable="true"', True),
    ],
)
def test_guarded_typing_rejects_focus_without_an_enabled_editable_text_input(
    backend, env, role, states, enabled
):
    env.xml = semantic_ui(semantic_control(role, "Target", states=states, enabled=enabled))
    snapshot = backend.observe()
    focus = AgentCondition("focused", AgentTarget(name="Target"))
    assert condition_holds(focus, snapshot)
    step = AgentStep(
        "type",
        "type_text",
        value="payload",
        preconditions=(focus,),
        postconditions=(AgentCondition("window_contains", value="Preferences"),),
    )
    assert backend.compile(step, snapshot, snapshot) == ()
    assert env.calls == []
    # Other keyboard commands may legitimately act on a focused button.
    assert len(backend.compile(replace(step, operation="hotkey", value="ctrl+a"), snapshot, snapshot)) == 1


def test_guarded_typing_binds_text_focus_and_types_without_clicking_or_clearing(backend, env):
    planned = backend.observe()
    label = '<label name="Search" s:showing="true"/>'
    entry = semantic_control("entry", "Search", states='s:editable="true" s:focused="true"')
    env.xml = semantic_ui(label + entry)
    current = backend.observe()
    target = AgentTarget(name="Search")
    step = AgentStep(
        "type",
        "type_text",
        value="query",
        preconditions=(AgentCondition("focused", target),),
        postconditions=(AgentCondition("value_equals", target, "query"),),
    )
    candidate = backend.compile(step, planned, current)[0]
    original = env.step

    def type_into_focused_input(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(
            label
            + semantic_control("entry", "Search", states='s:editable="true" s:focused="true"', value="query")
        )
        return env._get_obs(), *result[1:]

    env.step = type_into_focused_input
    receipt = backend.execute(candidate, "obs", "decision")
    assert receipt.success and backend.verify(candidate, receipt).passed
    assert env.calls == [({"action_type": "TYPING", "parameters": {"text": "query"}}, 0)]


@pytest.mark.parametrize("mutation", ["disabled", "readonly", "focus_shift"])
def test_guarded_typing_rechecks_bound_text_input_before_dispatch(backend, env, mutation):
    entry = semantic_control("entry", "Search", states='s:editable="true" s:focused="true"')
    env.xml = semantic_ui(entry)
    snapshot = backend.observe()
    target = AgentTarget(name="Search")
    step = AgentStep(
        "type",
        "type_text",
        value="query",
        preconditions=(AgentCondition("focused", target),),
        postconditions=(AgentCondition("value_equals", target, "query"),),
    )
    candidate = backend.compile(step, snapshot, snapshot)[0]
    if mutation == "disabled":
        env.xml = semantic_ui(
            semantic_control("entry", "Search", states='s:editable="true" s:focused="true"', enabled=False)
        )
    elif mutation == "readonly":
        env.xml = semantic_ui(semantic_control("entry", "Search", states='s:focused="true"'))
    else:
        env.xml = semantic_ui(semantic_control("push-button", "Search", states='s:focused="true"'))
    receipt = backend.execute(candidate, "obs", "decision")
    assert not receipt.success and env.calls == []


def test_conditions_fail_closed_on_ambiguous_state_and_allow_empty_expected_values():
    first = AgentControl(
        "d0",
        "Save item",
        "input",
        role="entry",
        value="",
        checked=True,
        focused=True,
        app="Example App",
        window="Settings",
        ancestors=("panel: General",),
    )
    snapshot = AgentSnapshot("osworld://desktop", "Settings", "Ready for input", (first,))
    target = AgentTarget(name="SAVE", role="textbox", match="contains", app="example", ancestor="GENERAL")
    assert matching_controls(snapshot, target) == (first,)
    assert conditions_hold(
        (AgentCondition("focused", target), AgentCondition("value_equals", target, "")), snapshot
    )
    assert condition_holds(AgentCondition("text_contains", value="ready"), snapshot)
    assert condition_holds(AgentCondition("window_contains", value="SETTINGS"), snapshot)
    ambiguous = replace(snapshot, controls=(first, replace(first, ref="d1")))
    assert condition_holds(AgentCondition("exists", target), ambiguous)
    assert not condition_holds(AgentCondition("focused", target), ambiguous)
    assert not condition_holds(AgentCondition("checked", target), ambiguous)
    assert not condition_holds(AgentCondition("absent", AgentTarget()), snapshot)


def test_observable_dialog_and_label_satisfy_conditions_but_never_compile_click(backend, env):
    env.xml = semantic_ui(
        '<dialog name="Confirmation" s:showing="true" s:active="true">'
        '<label name="Saved successfully" s:showing="true"/></dialog>'
    )
    snapshot = backend.observe()
    for role, name in [("dialog", "Confirmation"), ("label", "Saved successfully")]:
        target = AgentTarget(name=name, role=role)
        assert condition_holds(AgentCondition("exists", target), snapshot)
        assert not condition_holds(AgentCondition("absent", target), snapshot)
        assert backend.compile(AgentStep("no-click", "click", target=target), snapshot, snapshot) == ()


@pytest.mark.parametrize("xml", [None, "", "not XML", "<broken>", "<desktop/>"])
def test_missing_or_invalid_a11y_cannot_establish_absence(backend, env, xml):
    env.xml = xml
    snapshot = backend.observe()
    assert not condition_holds(AgentCondition("absent", AgentTarget(name="Dialog")), snapshot)
    if xml != "<desktop/>":
        assert not snapshot.a11y_complete


@pytest.mark.parametrize("mode", ["controls", "text"])
def test_truncated_observation_cannot_establish_absence(backend, env, mode):
    if mode == "controls":
        items = "".join(f'<label name="item {i}" s:showing="true"/>' for i in range(513))
    else:
        items = '<label name="Long text" s:showing="true">' + "x" * 2500 + "</label>"
    env.xml = semantic_ui(items)
    snapshot = backend.observe()
    assert not snapshot.a11y_complete
    assert not condition_holds(AgentCondition("absent", AgentTarget(name="Later unseen item")), snapshot)


def test_unchecked_and_empty_value_cannot_be_claimed_from_plain_button_defaults():
    control = AgentControl("d0", "Save", "button", role="button")
    snapshot = AgentSnapshot("osworld://desktop", "Main", "Save", (control,))
    target = AgentTarget(name="Save")
    assert not condition_holds(AgentCondition("unchecked", target), snapshot)
    assert not condition_holds(AgentCondition("value_equals", target, ""), snapshot)


def test_background_window_does_not_satisfy_current_window_guard():
    control = AgentControl("d0", "Save", "button", role="button", window="Preferences", active=False)
    snapshot = AgentSnapshot("osworld://desktop", "Unrelated dialog", "", (control,))
    assert not condition_holds(AgentCondition("window_contains", value="Preferences"), snapshot)
    assert condition_holds(AgentCondition("window_contains", value="Unrelated"), snapshot)


@pytest.mark.parametrize("state", ["", 's:active="false"', 's:focused="false"'])
def test_visible_atspi_window_without_active_evidence_cannot_guard_keyboard(backend, env, state):
    env.xml = (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state">'
        '<application name="Editor"><frame name="Preferences" s:showing="true" '
        f's:visible="true" {state}>'
        '<entry name="Search" s:showing="true" s:editable="true" s:enabled="true" s:sensitive="true"/>'
        "</frame></application></desktop-frame>"
    )
    snapshot = backend.observe()
    assert snapshot.title == "OSWorld desktop"
    assert any(control.window == "Preferences" for control in snapshot.controls)
    assert not any(control.active for control in snapshot.controls)
    guard = AgentCondition("window_contains", value="Preferences")
    assert not condition_holds(guard, snapshot)
    step = AgentStep(
        "hotkey",
        "hotkey",
        value="ctrl+a",
        preconditions=(guard,),
        postconditions=(AgentCondition("text_contains", value="Search"),),
    )
    assert backend.compile(step, snapshot, snapshot) == ()
    assert env.calls == []


@pytest.mark.parametrize("state", ['s:active="true"', 's:focused="true"'])
def test_atspi_window_active_or_focus_evidence_supports_current_window_guard(backend, env, state):
    env.xml = (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state">'
        '<application name="A background app"><frame name="Unrelated" s:showing="true"/></application>'
        f'<application name="Editor"><frame name="Preferences" s:showing="true" {state}/></application>'
        "</desktop-frame>"
    )
    snapshot = backend.observe()
    assert snapshot.title == "Preferences"
    assert condition_holds(AgentCondition("window_contains", value="Preferences"), snapshot)
    assert not condition_holds(AgentCondition("window_contains", value="Unrelated"), snapshot)


def test_pending_effect_recovery_observes_without_repeating_input(backend, env):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    step = AgentStep(
        "save",
        "click",
        target=AgentTarget(name="Save"),
        recovery="observe",
        postconditions=(AgentCondition("exists", AgentTarget(name="Saved", role="label")),),
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and not backend.verify(action, receipt).passed
    assert not backend.verify_pending(action, receipt).passed
    env.xml = semantic_ui(semantic_control() + '<label name="Saved" s:showing="true"/>')
    assert backend.verify_pending(action, receipt).passed
    assert backend.verify(action, receipt).passed
    assert len(env.calls) == 1


def test_pending_recovery_is_bounded_and_rejects_forged_receipts(backend, env):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    step = AgentStep(
        "save",
        "click",
        target=AgentTarget(name="Save"),
        recovery="observe",
        postconditions=(AgentCondition("exists", AgentTarget(name="Saved", role="label")),),
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(action, "obs", "decision")
    assert not backend.verify_pending(action, replace(receipt, success=False)).passed
    assert not backend.verify_pending(replace(action, description="forged"), receipt).passed
    assert not backend.verify_pending(action, receipt).passed
    assert not backend.verify_pending(action, receipt).passed
    env.xml = semantic_ui('<label name="Saved" s:showing="true"/>')
    result = backend.verify_pending(action, receipt)
    assert not result.passed and "budget exhausted" in result.details["reason"]
    assert len(env.calls) == 1


def test_pending_fill_still_requires_target_value_not_just_general_postcondition(backend, env):
    entry = semantic_control("entry", "Search", states='s:editable="true" s:focused="true"', value="wrong")
    env.xml = semantic_ui(entry)
    snapshot = backend.observe()
    step = AgentStep(
        "fill",
        "fill",
        value="expected",
        target=AgentTarget(name="Search"),
        recovery="observe",
        postconditions=(AgentCondition("window_contains", value="Preferences"),),
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and not backend.verify(action, receipt).passed
    assert not backend.verify_pending(action, receipt).passed
    env.xml = env.xml.replace(">wrong<", ">expected<")
    assert backend.verify_pending(action, receipt).passed
    assert len(env.calls) == 3


def clock_menu_xml(time="02∶31"):
    return (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:a="https://accessibility.ubuntu.example.org/ns/attributes" '
        'xmlns:c="https://accessibility.ubuntu.example.org/ns/component">'
        '<application name="gnome-shell"><window s:showing="true" s:focused="true">'
        '<menu name="" s:showing="true" s:enabled="true" s:sensitive="true" s:focusable="true" '
        'c:screencoord="(907, 0)" c:size="(176, 27)"><panel s:showing="true">'
        f'<label name="Oct 23 {time}" s:showing="true" a:toolkit="clutter" '
        'c:screencoord="(937, 0)" c:size="(116, 27)">'
        '<text s:showing="true" a:toolkit="clutter" c:screencoord="(952, 4)" c:size="(86, 18)">'
        f"Oct 23 {time}</text></label>"
        "</panel></menu></window></application></desktop-frame>"
    )


def test_clock_menu_inherits_observed_stable_label_and_fresh_bounds(backend, env):
    env.frame = png(size=(1920, 1080))
    env.xml = clock_menu_xml()
    before = backend.observe()
    target = AgentTarget(name="GNOME top-bar clock", role="menuitem")
    matches = matching_controls(before, target)
    assert len(matches) == 1
    assert matches[0].box == (472, 0, 564, 25)
    assert matches[0].tag == "button" and not matches[0].disabled
    assert len(matching_controls(before, AgentTarget(name="GNOME top-bar clock", role="label"))) == 1
    env.xml = clock_menu_xml("02∶32")
    after = backend.observe()
    assert before.scope_key() == after.scope_key()
    assert before.controls == after.controls
    action = backend.compile(AgentStep("clock", "click", target=target), before, after)[0]
    assert backend.execute(action, "obs", "decision").success
    # Coordinates use the actual menu bounds, not its smaller label rectangle.
    assert env.calls[0][0]["parameters"]["x"] == 994


@pytest.mark.parametrize("raw_role", ["toggle-button", "toggle", "switch", "check-box"])
@pytest.mark.parametrize("query_role", ["toggle-button", "toggle", "switch", "checkbox"])
def test_toggle_aliases_match_only_actual_checkable_controls(backend, env, raw_role, query_role):
    env.xml = semantic_ui(
        semantic_control(raw_role, "Notifications") + semantic_control("push-button", "Notifications", x=240)
    )
    snapshot = backend.observe()
    target = AgentTarget(name="Notifications", role=query_role)
    matches = matching_controls(snapshot, target)
    assert len(matches) == 1
    assert matches[0].input_type == "checkbox"
    assert condition_holds(AgentCondition("unchecked", target), snapshot)
    offered = backend.compile(AgentStep("enable", "check", target=target), snapshot, snapshot)
    assert len(offered) == 1 and offered[0].arguments["bound_control"]["input_type"] == "checkbox"


def test_plain_button_does_not_match_switch_alias(backend, env):
    env.xml = semantic_ui(semantic_control("push-button", "Notifications"))
    snapshot = backend.observe()
    target = AgentTarget(name="Notifications", role="switch")
    assert matching_controls(snapshot, target) == ()
    assert backend.compile(AgentStep("bad", "check", target=target), snapshot, snapshot) == ()


@pytest.mark.parametrize(
    ("kind", "role", "states", "value"),
    [
        ("checked", "check-box", 's:checked="true"', ""),
        ("unchecked", "check-box", "", ""),
        ("value_equals", "entry", 's:editable="true"', "query"),
        ("value_equals", "entry", 's:editable="true"', ""),
        ("focused", "entry", 's:editable="true" s:focused="true"', ""),
    ],
)
def test_state_conditions_ignore_same_named_static_labels_but_not_real_controls(
    backend, env, kind, role, states, value
):
    label = '<label name="Setting" s:showing="true"/>'
    control = semantic_control(role, "Setting", states=states, value=value)
    env.xml = semantic_ui(label + control)
    target = AgentTarget(name="Setting")
    condition = AgentCondition(kind, target, value if kind == "value_equals" else "")
    snapshot = backend.observe()
    assert len(matching_controls(snapshot, target)) == 2
    assert condition_holds(condition, snapshot)
    # An additional real input remains ambiguous even when only one has focus
    # or the expected value/state. Never pick the convenient matching state.
    other = semantic_control(role, "Setting", x=240, states='s:editable="true"', value="other")
    env.xml = semantic_ui(label + control + other)
    assert not condition_holds(condition, backend.observe())


def test_static_label_cannot_claim_focus_or_checked_state(backend, env):
    env.xml = semantic_ui('<label name="Setting" s:showing="true" s:focused="true" s:checked="true"/>')
    snapshot = backend.observe()
    for kind in ("focused", "checked", "unchecked"):
        assert not condition_holds(AgentCondition(kind, AgentTarget(name="Setting")), snapshot)


def sibling_toggle_ui(*, extra="", label_x=650, toggle_x=757, container_size="(425, 42)"):
    return semantic_ui(
        '<panel s:showing="true" c:screencoord="(644, 432)" '
        f'c:size="{container_size}">'
        f'<label name="Example setting" s:showing="true" c:screencoord="({label_x}, 444)" '
        'c:size="(101, 17)"><text s:showing="true"/></label>'
        '<toggle-button s:showing="true" s:enabled="true" s:sensitive="true" s:focusable="true" '
        f'c:screencoord="({toggle_x}, 438)" c:size="(52, 30)">'
        '<check-box s:showing="true" c:screencoord="(759, 440)" c:size="(48, 26)"/>'
        "</toggle-button>"
        '<push-button name="Clear" s:showing="true" s:enabled="true" s:sensitive="true" '
        'c:screencoord="(978, 438)" c:size="(85, 30)"/>' + extra + "</panel>"
    )


def test_unique_visible_sibling_label_names_actual_checkable_control(backend, env):
    env.frame = png(size=(1920, 1080))
    env.xml = sibling_toggle_ui()
    snapshot = backend.observe()
    target = AgentTarget(name="Example setting", role="switch")
    matches = matching_controls(snapshot, target)
    assert len(matches) == 1
    assert matches[0].box == (394, 406, 421, 433)
    assert matches[0].tag == "input" and matches[0].input_type == "checkbox"
    assert condition_holds(AgentCondition("unchecked", target), snapshot)
    offered = backend.compile(AgentStep("enable", "check", target=target), snapshot, snapshot)
    assert len(offered) == 1
    assert offered[0].arguments["box"] == matches[0].box
    # Static observed label retains its independent role; no fake duplicate
    # actionable candidate is manufactured from the label or nested checkbox.
    assert (
        len(
            backend.compile(
                AgentStep("enable", "check", target=AgentTarget(name="Example setting")), snapshot, snapshot
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "extra": '<label name="Other setting" s:showing="true" '
            'c:screencoord="(850, 444)" c:size="(80, 17)"/>'
        },
        {
            "extra": '<check-box s:showing="true" s:focusable="true" '
            'c:screencoord="(850, 438)" c:size="(52, 30)"/>'
        },
        {"label_x": -5},
        {"toggle_x": 1900},
        {"container_size": "(50, 42)"},
        {"container_size": "(0, 0)"},
    ],
)
def test_sibling_label_association_rejects_ambiguity_and_invalid_geometry(backend, env, kwargs):
    env.frame = png(size=(1920, 1080))
    env.xml = sibling_toggle_ui(**kwargs)
    snapshot = backend.observe()
    target = AgentTarget(name="Example setting", role="checkbox")
    assert matching_controls(snapshot, target) == ()
    assert backend.compile(AgentStep("enable", "check", target=target), snapshot, snapshot) == ()


def test_hidden_sibling_label_cannot_name_a_visible_checkable(backend, env):
    env.frame = png(size=(1920, 1080))
    env.xml = sibling_toggle_ui().replace('name="Example setting" s:showing="true"', 'name="Example setting"')
    snapshot = backend.observe()
    assert matching_controls(snapshot, AgentTarget(name="Example setting", role="checkbox")) == ()


@pytest.mark.parametrize("role", ["check-box", "toggle-button", "switch", "radio-button"])
@pytest.mark.parametrize("checked", [False, True])
def test_anonymous_nonfocusable_checkable_state_is_preserved_but_never_clickable(backend, env, role, checked):
    states = 's:checked="true"' if checked else ""
    env.xml = semantic_ui(semantic_control(role, "", states=states))
    snapshot = backend.observe()
    target = AgentTarget(role="radio" if role == "radio-button" else "checkbox", app="Demo")
    controls = matching_controls(snapshot, target)
    assert len(controls) == 1
    control = controls[0]
    assert control.label == "" and control.tag == "container"
    assert control.checked is checked and control.input_type == target.role
    assert {"checked", "unchecked"} <= set(observable_condition_kinds(control))
    assert "focused" not in observable_condition_kinds(control)
    assert condition_holds(AgentCondition("checked" if checked else "unchecked", target), snapshot)
    for operation in ("click", "check", "uncheck"):
        assert backend.compile(AgentStep("s", operation, target=target), snapshot, snapshot) == ()
    assert env.calls == []


def test_selection_and_expansion_capabilities_require_observed_atspi_support(backend, env):
    env.xml = semantic_ui(
        semantic_control("page-tab", "Tab", states='s:selectable="true"')
        + semantic_control("tree-item", "Branch", states='s:focusable="true" s:expandable="true"')
        + semantic_control("page-tab", "Chosen", states='s:selected="true"')
        + semantic_control("tree-item", "Open branch", states='s:focusable="true" s:expanded="true"')
        + '<label name="Static" s:showing="true"/>'
    )
    snapshot = backend.observe()
    by_name = {control.label: control for control in snapshot.controls}
    assert by_name["Tab"].selectable and not by_name["Tab"].selected
    assert by_name["Branch"].expandable and not by_name["Branch"].expanded
    assert by_name["Chosen"].selectable and by_name["Chosen"].selected
    assert by_name["Open branch"].expandable and by_name["Open branch"].expanded
    assert not by_name["Static"].selectable and not by_name["Static"].expandable
    for kind, name, expected in (
        ("selected", "Tab", "false"), ("expanded", "Branch", "false"),
        ("selected", "Chosen", "true"), ("expanded", "Open branch", "true"),
        ("selected", "Static", "unknown"), ("expanded", "Static", "unknown"),
    ):
        evidence = condition_evidence(AgentCondition(kind, AgentTarget(name=name)), snapshot)
        assert evidence["outcome"] == expected


def settings_row_ui(*, checked=False, extra="", label_x=842, wrapper_states="", label_states=""):
    # Minimal structure observed in v8 Settings XML: a focusable unnamed row,
    # a visible label inside a layout wrapper, and one nonfocusable switch.
    return semantic_ui(
        '<list-item s:showing="true" s:enabled="true" s:sensitive="true" s:focusable="true" '
        'c:screencoord="(827, 221)" c:size="(578, 55)">'
        '<filler s:showing="true" c:screencoord="(842, 236)" c:size="(548, 26)">'
        f'<filler s:showing="true" {wrapper_states} c:screencoord="(842, 240)" c:size="(486, 17)">'
        f'<label name="Example option" s:showing="true" {label_states} '
        f'c:screencoord="({label_x}, 240)" c:size="(486, 17)"/>'
        '<label name="" s:visible="true"/></filler>'
        '<label name="" s:visible="true"/><icon s:visible="true"/>'
        '<toggle-button s:showing="true" s:enabled="true" s:sensitive="true" '
        + ('s:checked="true" ' if checked else "")
        + 'c:screencoord="(1340, 236)" c:size="(50, 26)"/>'
        + extra + '</filler></list-item>'
    )


def test_nested_label_identifies_readonly_switch_and_clickable_row_separately(backend, env):
    env.frame = png(size=(1920, 1080))
    for checked in (False, True):
        env.xml = settings_row_ui(checked=checked)
        snapshot = backend.observe()
        target = AgentTarget(name="Example option", app="Demo")
        matches = matching_controls(snapshot, target)
        assert {control.role for control in matches} == {"listitem", "label", "checkbox"}
        switch = next(control for control in matches if control.role == "checkbox")
        assert switch.tag == "container" and switch.checked is checked
        assert condition_holds(AgentCondition("checked" if checked else "unchecked", target), snapshot)
        candidates = backend.compile(AgentStep("click", "click", target=target), snapshot, snapshot)
        assert len(candidates) == 1
        assert candidates[0].arguments["bound_control"]["role"] == "listitem"
        assert backend.compile(AgentStep("check", "check", target=target), snapshot, snapshot) == ()


@pytest.mark.parametrize("changes", [
    {"extra": '<label name="Other" s:showing="true" c:screencoord="(900, 240)" c:size="(40, 17)"/>'},
    {"extra": '<check-box s:showing="true" c:screencoord="(1000, 240)" c:size="(40, 17)"/>'},
    {"label_x": 1900},
    {"wrapper_states": 's:focusable="true"'},
    {"label_states": 's:visible="false"'},
])
def test_nested_checkable_label_association_rejects_ambiguity_and_unreliable_geometry(backend, env, changes):
    env.frame = png(size=(1920, 1080))
    env.xml = settings_row_ui(**changes)
    snapshot = backend.observe()
    assert not matching_controls(snapshot, AgentTarget(name="Example option", role="checkbox"))
    assert all(control.label == "" for control in snapshot.controls if control.input_type == "checkbox")


def test_postcondition_evidence_uses_after_state_and_pending_updates_without_input(backend, env):
    env.xml = semantic_ui(semantic_control(name="Open"))
    target = AgentTarget(name="Option", role="checkbox", app="Demo")
    step = AgentStep(
        "open", "click", target=AgentTarget(name="Open"),
        postconditions=(AgentCondition("checked", target),), recovery="observe",
    )
    snapshot = backend.observe()
    candidate = backend.compile(step, snapshot, snapshot)[0]
    original = env.step

    def show_option(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(semantic_control("check-box", "Option"))
        return env._get_obs(), *result[1:]

    env.step = show_option
    receipt = backend.execute(candidate, "obs", "decision")
    actual = backend.verify(candidate, receipt)
    assert receipt.success and not actual.passed
    assert actual.details["postcondition_evidence"][0] == {
        "kind": "checked", "target": {"name": "Option", "role": "checkbox", "app": "Demo",
                                         "window": "", "ancestor": "", "match": "exact"},
        "outcome": "false", "reason": "state_mismatch", "satisfied": False,
    }
    # The missing pre-action target must not replace actual post-action evidence.
    assert condition_evidence(step.postconditions[0], snapshot)["reason"] == "target_missing"
    env.xml = semantic_ui(semantic_control(name="Open"))
    pending = backend.verify_pending(candidate, receipt)
    assert pending.details["postcondition_evidence"][0]["outcome"] == "unknown"
    assert pending.details["postcondition_evidence"][0]["reason"] == "target_missing"
    env.xml = semantic_ui(semantic_control("check-box", "Option", states='s:checked="true"'))
    pending = backend.verify_pending(candidate, receipt)
    assert pending.passed and pending.details["postcondition_evidence"][0]["outcome"] == "true"
    assert backend.verify(candidate, receipt) == pending
    assert len(env.calls) == 1


def test_postcondition_evidence_does_not_include_expected_or_observed_field_values(backend, env):
    env.xml = semantic_ui(
        semantic_control(name="Save") + semantic_control("entry", "Field", states='s:editable="true"',
                                                         value="private observed content")
    )
    step = AgentStep(
        "save", "click", target=AgentTarget(name="Save"),
        postconditions=(
            AgentCondition("value_equals", AgentTarget(name="Field"), "private expected content"),
        ),
        recovery="observe",
    )
    snapshot = backend.observe()
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    for verification in (backend.verify(candidate, receipt), backend.verify_pending(candidate, receipt)):
        facts = verification.details["postcondition_evidence"]
        assert facts[0]["outcome"] == "false"
        assert "value" not in facts[0]
        assert "private" not in str(facts)


def navigation_step(**changes):
    step = AgentStep(
        "navigate",
        "click",
        navigation=True,
        label="Open the advanced settings page",
        target=AgentTarget(app="Demo", window="Preferences", ancestor="General"),
        postconditions=(AgentCondition("exists", AgentTarget(name="Destination", role="label", app="Demo")),),
        recovery="observe",
    )
    return replace(step, **changes)


def test_navigation_offers_distinct_current_controls_with_scope_and_goal(backend, env):
    env.xml = semantic_ui(
        semantic_control(name="General")
        + semantic_control("page-tab", "Advanced", x=240)
        + semantic_control("entry", "Search", x=400, states='s:editable="true"', value="PRIVATE FIELD VALUE")
        + semantic_control(name="Unavailable", x=560, enabled=False)
        + '<label name="Not clickable" s:showing="true"/>'
    )
    planned = backend.observe()
    current = backend.observe()
    scope = navigation_step().target
    assert matching_controls(current, scope) == ()
    assert matching_controls(current, scope, allow_scope=True)
    assert not condition_holds(AgentCondition("exists", scope), current)
    assert not condition_holds(AgentCondition("absent", scope), current)
    candidates = backend.compile(navigation_step(), planned, current)
    assert len(candidates) == 3
    assert len({tuple(candidate.arguments["box"]) for candidate in candidates}) == 3
    assert {candidate.arguments["bound_control"]["label"] for candidate in candidates} == {
        "General",
        "Advanced",
        "Search",
    }
    for candidate in candidates:
        assert candidate.arguments["navigation"] is True
        assert "Open the advanced settings page" in candidate.description
        assert "Preferences" in candidate.description and "Demo" in candidate.description
        assert "PRIVATE FIELD VALUE" not in candidate.description


def test_navigation_rebinds_all_current_controls_without_old_label_or_coordinate_pins(backend, env):
    env.xml = semantic_ui(semantic_control(name="First page"))
    planned = backend.observe()
    env.xml = semantic_ui(semantic_control(name="Next page", x=400))
    current = backend.observe()
    candidates = backend.compile(navigation_step(), planned, current)
    assert len(candidates) == 1
    assert candidates[0].arguments["bound_control"]["label"] == "Next page"
    assert candidates[0].arguments["box"] == (500, 200, 650, 267)
    env.xml = semantic_ui(semantic_control(name="Next page", x=240))
    assert not backend.execute(candidates[0], "obs", "decision").success
    assert env.calls == []


def test_navigation_does_not_silently_truncate_at_64_or_exceed_transport_limit(backend, env):
    env.xml = semantic_ui(
        "".join(semantic_control(name=f"Choice {i}", x=80 + i % 4 * 120) for i in range(80))
    )
    snapshot = backend.observe()
    assert len(backend.compile(navigation_step(), snapshot, snapshot)) == 80
    assert backend.compile(navigation_step(), snapshot, replace(snapshot, a11y_complete=False)) == ()
    control = next(control for control in snapshot.controls if control.role == "button")
    oversized = replace(
        snapshot,
        controls=tuple(replace(control, ref=f"d{i}", label=f"Choice {i}") for i in range(256)),
    )
    assert backend.compile(navigation_step(), oversized, oversized) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"target": AgentTarget(name="Specific", app="Demo")},
        {"target": AgentTarget(role="button", app="Demo")},
        {"target": AgentTarget(ancestor="General")},
        {"target": AgentTarget(app="Demo", match="contains")},
        {"target": AgentTarget(app="Different app")},
        {"target": AgentTarget(app="Demo", window="Other window")},
        {"target": AgentTarget(app="Demo", ancestor="Other parent")},
        {"label": ""},
        {"operation": "fill", "value": "private text"},
        {"postconditions": ()},
        {"ref": "d0"},
        {"navigation": 1},
    ],
)
def test_navigation_contract_and_scope_fail_closed(backend, env, changes):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    assert backend.compile(navigation_step(**changes), snapshot, snapshot) == ()


def test_navigation_image_only_change_does_not_establish_progress(backend, env):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and "image" in receipt.output["effects"]
    assert not verification.passed
    assert receipt.output["navigation"] is True
    assert receipt.output["before_scope"] == receipt.output["after_scope"]
    assert not receipt.output["postconditions_met"]
    assert not backend.verify_pending(candidate, receipt).passed
    assert len(env.calls) == 1


@pytest.fixture
def recorded_gimp_navigation(backend, env, monkeypatch):
    # Real public controls from v7, augmented only with their recorded bound
    # candidate fields. No invented popup or reconstructed raw XML is used.
    data = json.loads((Path(__file__).parent / "fixtures" / "osworld_gimp_navigation_v7.json").read_text(
        encoding="utf-8"
    ))
    snapshots = []
    for index, captured in enumerate(data["snapshots"]):
        controls = []
        for raw in captured["controls"]:
            item = dict(raw)
            for key in ("box", "ancestors", "options"):
                if key in item:
                    item[key] = tuple(item[key])
            controls.append(AgentControl(**item))
        snapshots.append(AgentSnapshot(
            captured["url"], captured["title"], "\n".join(control.label for control in controls),
            tuple(controls),
            image=screenshot_image(png("white" if index == 0 else "gray", size=(1920, 1080))),
            semantic_text=tuple(str(control.to_dict()) for control in controls),
            a11y_complete=captured["a11y_complete"],
        ))
    current = [snapshots[0]]
    monkeypatch.setattr(backend, "observe", lambda: current[0])
    monkeypatch.setattr(backend, "_snapshot", lambda observation: current[0])
    original = env.step

    def open_recorded_menu(action, pause):
        result = original(action, pause)
        current[0] = snapshots[1]
        return result

    monkeypatch.setattr(env, "step", open_recorded_menu)
    step = navigation_step(
        target=AgentTarget(app="gimp"), label="Open the Preferences dialog",
        postconditions=(AgentCondition(
            "exists", AgentTarget(name="Preferences", role="dialog", app="gimp")
        ),),
    )
    return snapshots[0], snapshots[1], step


def test_recorded_anonymous_panel_churn_does_not_authorize_another_menu_toggle(
    backend, env, recorded_gimp_navigation
):
    before, after, step = recorded_gimp_navigation
    assert len(before.controls) == 95 and len(after.controls) == 94
    before_actions = backend.compile(step, before, before)
    assert len(before_actions) == 42
    assert len(backend.compile(step, before, after)) == 42
    assert any(control.role == "panel" and not control.label and control.focused
               for control in before.controls)
    assert _navigation_scope(before, step) == _navigation_scope(after, step)
    # Recompile because issuing the after-state choices intentionally changes
    # the backend's live candidate registrations.
    chosen = next(action for action in backend.compile(step, before, before)
                  if action.arguments["bound_control"]["label"] == "Edit")
    receipt = backend.execute(chosen, "observation", "decision")
    assert receipt.success and receipt.output["before_fingerprint"] != receipt.output["after_fingerprint"]
    assert not backend.verify(chosen, receipt).passed
    assert receipt.output["before_scope"] == receipt.output["after_scope"]
    assert not receipt.output["postconditions_met"]
    for _ in range(2):
        pending = backend.verify_pending(chosen, receipt)
        assert not pending.passed and not pending.details["accessibility_progress"]
    exhausted = backend.verify_pending(chosen, receipt)
    assert exhausted.details["reason"] == "observation recovery budget exhausted"
    assert len(env.calls) == 1


def test_recorded_navigation_reaches_single_action_vlm_recovery_before_second_toggle(
    backend, env, recorded_gimp_navigation
):
    from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy

    _, after, step = recorded_gimp_navigation

    class RecoveryReached(Exception):
        pass

    class Planner:
        calls = 0
        def plan(self, goal, snapshot, history, *, max_actions=6):
            self.calls += 1
            if self.calls == 1:
                return AgentPlan("Open application settings", (step,), reusable=True)
            assert max_actions == 1 and snapshot is after
            assert history[-1]["recovery_required"] and not history[-1]["verified"]
            assert "do not blindly repeat" in history[-1]["feedback"]
            raise RecoveryReached()

    class ChooseEdit(DirectPlanPolicy):
        calls = 0
        def choose(self, observation, candidates):
            self.calls += 1
            assert len(candidates) == 42
            ordered = sorted(candidates, key=lambda action: not action.description.startswith("click Edit ["))
            return super().choose(observation, ordered)

    planner, policy = Planner(), ChooseEdit()
    result = AgentRunner(
        backend, planner, policy=policy,
        config=AgentConfig(allow_actions=True, max_steps=3, max_plans=3),
    ).run("Open application settings")
    assert result.status == "error" and "RecoveryReached" in result.reason
    assert result.recovery_observations == 2
    assert planner.calls == 2 and policy.calls == 1
    assert len(env.calls) == result.steps == 1


@pytest.mark.parametrize("explicit_text_condition", [False, True])
def test_static_text_churn_needs_explicit_condition_not_blind_navigation(
    backend, env, explicit_text_condition
):
    env.xml = semantic_ui(semantic_control(name="Next") + '<label name="First page" s:showing="true"/>')
    original = env.step
    def change_text(action, pause):
        result = original(action, pause)
        env.xml = env.xml.replace("First page", "Second page")
        return env._get_obs(), *result[1:]

    env.step = change_text
    step = navigation_step(**({"postconditions": (AgentCondition("text_contains", value="Second page"),)}
                              if explicit_text_condition else {}))
    before = backend.observe()
    candidate = backend.compile(step, before, before)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and verification.passed is explicit_text_condition
    assert not verification.details["accessibility_progress"]
    assert verification.details["postconditions_met"] is explicit_text_condition
    assert receipt.output["before_scope"] == receipt.output["after_scope"]


def test_new_visible_menu_choices_allow_fast_navigation(backend, env):
    env.xml = semantic_ui(semantic_control("menu", "Edit"))
    original = env.step
    def expand_menu(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(semantic_control("menu", "Edit", child=(
            semantic_control("menu-item", "Preferences", x=240)
            + semantic_control("menu-item", "Shortcuts", x=400)
        )))
        return env._get_obs(), *result[1:]

    env.step = expand_menu
    before = backend.observe()
    step = navigation_step()
    candidate = backend.compile(step, before, before)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    after = backend.observe()
    assert receipt.success and verification.passed
    assert verification.details["accessibility_progress"] and not verification.details["postconditions_met"]
    choices = backend.compile(step, before, after)
    assert {choice.arguments["bound_control"]["label"] for choice in choices} == {
        "Edit", "Preferences", "Shortcuts"
    }


def test_navigation_multiset_preserves_duplicates_but_ignores_enumeration(backend, env):
    env.xml = semantic_ui(semantic_control(name="Open") + semantic_control(name="Open", x=240))
    before = backend.observe()
    step = navigation_step()
    reordered = replace(before, controls=tuple(replace(control, ref=f"new-{index}")
                                              for index, control in enumerate(reversed(before.controls))))
    assert _navigation_scope(before, step) == _navigation_scope(reordered, step)
    removed = replace(before, controls=before.controls[:-1])
    assert _navigation_scope(before, step) != _navigation_scope(removed, step)


def test_navigation_preserves_actual_anonymous_readonly_checkable_state(backend, env):
    state_node = '<toggle-button s:showing="true" s:enabled="true" s:sensitive="true"/>'
    env.xml = semantic_ui(semantic_control(name="Open") + state_node)
    original = env.step

    def state_changes(action, pause):
        result = original(action, pause)
        env.xml = env.xml.replace(state_node, state_node.replace('/>', ' s:checked="true"/>'))
        return env._get_obs(), *result[1:]

    env.step = state_changes
    before = backend.observe()
    step = navigation_step()
    before_choices = backend.compile(step, before, before)
    assert len(before_choices) == 1
    state = next(control for control in before.controls if control.input_type == "checkbox")
    assert state.tag == "container" and state.box is None and not state.label and not state.checked
    receipt = backend.execute(before_choices[0], "obs", "decision")
    verification = backend.verify(before_choices[0], receipt)
    after = backend.observe()
    assert receipt.success and verification.passed
    assert verification.details["accessibility_progress"] and not verification.details["postconditions_met"]
    assert len(backend.compile(step, before, after)) == 1


@pytest.mark.parametrize("change", ["focused", "active", "geometry"])
def test_navigation_incidental_changes_are_not_progress_or_pending_effects(backend, env, change):
    env.xml = semantic_ui(semantic_control())
    updated = semantic_ui(
        semantic_control(
            x=240 if change == "geometry" else 80,
            states='s:focused="true"' if change == "focused" else "",
        )
    )
    if change == "active":
        updated = updated.replace('s:active="true"', 's:active="false"')
    original = env.step

    def incidental_change(action, pause):
        result = original(action, pause)
        env.xml = updated
        return env._get_obs(), *result[1:]

    env.step = incidental_change
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    assert receipt.success
    assert receipt.output["before_fingerprint"] != receipt.output["after_fingerprint"]
    assert receipt.output["before_scope"] == receipt.output["after_scope"]
    assert not receipt.output["accessibility_progress"]
    assert not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert len(env.calls) == 1


@pytest.mark.parametrize("change", ["focused", "active", "geometry"])
def test_navigation_preflight_still_rejects_incidental_changes(backend, env, change):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    env.xml = semantic_ui(
        semantic_control(
            x=240 if change == "geometry" else 80,
            states='s:focused="true"' if change == "focused" else "",
        )
    )
    if change == "active":
        env.xml = env.xml.replace('s:active="true"', 's:active="false"')
    assert not backend.execute(candidate, "obs", "decision").success
    assert env.calls == []


@pytest.mark.parametrize(
    ("role", "initial_states", "updated_states", "value"),
    [
        ("tree-item", 's:focusable="true"', 's:focusable="true" s:expanded="true"', ""),
        ("page-tab", "", 's:selected="true"', ""),
        ("check-box", "", 's:checked="true"', ""),
        ("entry", 's:editable="true"', 's:editable="true"', "Changed value"),
    ],
)
def test_navigation_preserves_meaningful_state_progress(
    backend, env, role, initial_states, updated_states, value
):
    env.xml = semantic_ui(semantic_control(role, "Settings", states=initial_states))
    original = env.step

    def state_change(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(semantic_control(role, "Settings", states=updated_states, value=value))
        return env._get_obs(), *result[1:]

    env.step = state_change
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and verification.passed
    assert verification.details["accessibility_progress"]
    assert not verification.details["postconditions_met"]


def test_navigation_returned_state_cannot_evade_cycles_with_focus_or_geometry(backend, env):
    env.xml = semantic_ui(semantic_control(name="General"))
    states = iter([
        semantic_ui(semantic_control(name="Advanced", x=240)),
        semantic_ui(semantic_control(name="General", x=400, states='s:focused="true"')).replace(
            's:active="true"', 's:active="false"'
        ),
    ])
    original = env.step

    def navigate(action, pause):
        result = original(action, pause)
        env.xml = next(states)
        return env._get_obs(), *result[1:]

    env.step = navigate
    snapshot = backend.observe()
    receipts = []
    for index in range(2):
        current = backend.observe()
        candidate = backend.compile(navigation_step(), snapshot, current)[0]
        receipt = backend.execute(candidate, f"obs-{index}", f"decision-{index}")
        assert backend.verify(candidate, receipt).passed
        receipts.append(receipt)
    assert receipts[0].output["before_scope"] == receipts[1].output["after_scope"]
    assert receipts[0].output["before_fingerprint"] != receipts[1].output["after_fingerprint"]
    assert len(env.calls) == 2


def test_navigation_can_complete_explicit_focus_condition_without_claiming_progress(backend, env):
    env.xml = semantic_ui(semantic_control())
    original = env.step

    def focus(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(semantic_control(states='s:focused="true"'))
        return env._get_obs(), *result[1:]

    env.step = focus
    step = navigation_step(postconditions=(AgentCondition("focused", AgentTarget(name="Save", app="Demo")),))
    snapshot = backend.observe()
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and verification.passed
    assert verification.details["postconditions_met"]
    assert not verification.details["accessibility_progress"]


def test_navigation_scoped_accessibility_progress_is_distinct_from_completion(backend, env):
    env.xml = semantic_ui(semantic_control(name="General"))
    original = env.step

    def open_new_page(action, pause):
        result = original(action, pause)
        env.xml = semantic_ui(semantic_control(name="Advanced", x=240))
        return env._get_obs(), *result[1:]

    env.step = open_new_page
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and verification.passed
    assert verification.details["accessibility_progress"]
    assert not verification.details["postconditions_met"]
    assert receipt.output["before_scope"] != receipt.output["after_scope"]
    # Observing partial progress again must not mark the navigation goal done.
    pending = backend.verify_pending(candidate, receipt)
    assert pending.passed and not pending.details["postconditions_met"]
    env.xml = semantic_ui(semantic_control(name="Advanced") + '<label name="Destination" s:showing="true"/>')
    completed = backend.verify_pending(candidate, receipt)
    assert completed.passed and completed.details["postconditions_met"]
    assert not completed.details["task_success"]
    assert len(env.calls) == 1


def test_navigation_ignores_other_application_changes_and_ref_renumbering(backend, env):
    env.xml = semantic_ui(semantic_control(name="General"))
    original = env.step

    def background_changes(action, pause):
        result = original(action, pause)
        env.xml = env.xml.replace(
            '<application name="Demo">',
            '<application name="Background"><frame name="Background window" s:showing="true">'
            '<label name="New notification" s:showing="true"/></frame></application>'
            '<application name="Demo">',
        )
        return env._get_obs(), *result[1:]

    env.step = background_changes
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    assert receipt.success and "accessibility" in receipt.output["effects"]
    assert receipt.output["before_scope"] == receipt.output["after_scope"]
    assert not backend.verify(candidate, receipt).passed


def test_navigation_missing_a11y_is_not_progress(backend, env):
    env.xml = semantic_ui(semantic_control())
    original = env.step

    def lose_accessibility(action, pause):
        result = original(action, pause)
        env.xml = None
        return env._get_obs(), *result[1:]

    env.step = lose_accessibility
    snapshot = backend.observe()
    candidate = backend.compile(navigation_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert not receipt.output["accessibility_progress"]


def test_navigation_preconditions_are_required_again_immediately_before_click(backend, env):
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    step = navigation_step(preconditions=(AgentCondition("exists", AgentTarget(name="Missing")),))
    assert backend.compile(step, snapshot, snapshot) == ()
    step = navigation_step(preconditions=(AgentCondition("window_contains", value="Preferences"),))
    candidate = backend.compile(step, snapshot, snapshot)[0]
    env.xml = semantic_ui(semantic_control(), window="Other window")
    assert not backend.execute(candidate, "obs", "decision").success
    assert env.calls == []


@pytest.mark.parametrize("operation", ["click", "fill", "check", "uncheck"])
def test_shared_executable_controls_match_backend_action_legality_without_binding(backend, env, operation):
    env.xml = semantic_ui(
        semantic_control("push-button", "Target")
        + semantic_control("entry", "Target", x=240, states='s:editable="true"')
        + semantic_control("entry", "Target", x=400)
        + semantic_control("check-box", "Target")
        + semantic_control("radio-button", "Target", x=240)
        + semantic_control("push-button", "Target", enabled=False)
        + '<label name="Target" s:showing="true" c:screencoord="(80, 120)" c:size="(120, 40)"/>'
        + '<push-button name="Target" s:showing="true" s:enabled="true" s:sensitive="true"/>'
    )
    snapshot = backend.observe()
    target = AgentTarget(name="Target")
    matches = executable_controls(snapshot, operation, target)
    assert backend._issued == {} and env.calls == []
    action = AgentStep("shared", operation, target=target, value="text" if operation == "fill" else "")
    candidates = backend.compile(action, snapshot, snapshot)
    assert [item.arguments["bound_control"] for item in candidates] == [item.to_dict() for item in matches]
    expected = {"click": 5, "fill": 1, "check": 2, "uncheck": 1}
    assert len(matches) == expected[operation]


@pytest.mark.parametrize("input_type", ["password", "Password", "file", "hidden"])
def test_sensitive_controls_rejected_identically_by_helper_and_semantic_or_ref_backend(backend, input_type):
    snapshot = backend.observe()
    control = AgentControl(
        "secret",
        "Target",
        "input",
        role="textbox",
        input_type=input_type,
        editable=True,
        box=(100, 100, 200, 200),
        app="Demo",
        window="Preferences",
    )
    snapshot = replace(snapshot, controls=(control,))
    for operation in ("click", "fill"):
        assert executable_controls(snapshot, operation, AgentTarget(name="Target")) == ()
        for target in ({"target": AgentTarget(name="Target")}, {"ref": "secret"}):
            step = AgentStep("sensitive", operation, value="text" if operation == "fill" else "", **target)
            assert backend.compile(step, snapshot, snapshot) == ()
    assert backend._issued == {}


@pytest.mark.parametrize(
    "box", [None, (1, 1, 3, 3), (-1, 10, 100, 100), (1, 1, 1001, 100), (True, 1, 10, 10)]
)
def test_shared_executable_controls_require_actual_valid_geometry(backend, box):
    snapshot = backend.observe()
    control = AgentControl("item", "Target", "button", role="button", box=box)
    snapshot = replace(snapshot, controls=(control,))
    assert executable_controls(snapshot, "click", AgentTarget(name="Target")) == ()
    assert (
        backend.compile(AgentStep("click", "click", target=AgentTarget(name="Target")), snapshot, snapshot)
        == ()
    )


def test_shared_executable_scope_is_navigation_click_only_and_preserves_all_matches(backend, env):
    env.xml = semantic_ui(
        "".join(semantic_control(name=f"Choice {i}", x=80 + i % 4 * 120) for i in range(80))
    )
    snapshot = backend.observe()
    target = AgentTarget(app="Demo")
    assert len(executable_controls(snapshot, "click", target, allow_scope=True)) == 80
    assert executable_controls(snapshot, "click", target) == ()
    for operation in ("fill", "check", "uncheck", "select"):
        assert executable_controls(snapshot, operation, target, allow_scope=True) == ()
    assert backend._issued == {} and env.calls == []


def test_typed_step_verifies_its_returned_observation_without_redundant_fetch(backend, env):
    action = candidate(backend)
    original = env._get_obs
    observed = []

    def observe():
        observed.append(True)
        return original()

    env._get_obs = observe

    def step(action, pause):
        env.calls.append((action, pause))
        # Only the response contains the actual effect. An unrelated additional
        # read would return the earlier simulated frame and lose this evidence.
        return {"screenshot": png("black"), "accessibility_tree": env.xml}, 0, False, {}

    env.step = step
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and backend.verify(action, receipt).passed
    assert receipt.output["effects"] == ["image"]
    assert len(observed) == 1  # Mandatory preflight, no extra post-step fetch.
    assert backend.last_observation["screenshot"] == png("black")
    assert backend._action_observation is None


def test_typed_step_response_from_an_earlier_execution_is_never_reused(backend, env):
    first = candidate(backend)

    def step(action, pause):
        env.calls.append((action, pause))
        frame = png("black") if len(env.calls) == 1 else env.frame
        return {"screenshot": frame, "accessibility_tree": env.xml}, 0, False, {}

    env.step = step
    receipt = backend.execute(first, "obs1", "decision1")
    assert backend.verify(first, receipt).passed
    second = candidate(backend)
    receipt = backend.execute(second, "obs2", "decision2")
    assert receipt.success and not backend.verify(second, receipt).passed
    assert backend._action_observation is None


@pytest.mark.parametrize("operation", ["wait", "check"])
def test_wait_and_already_satisfied_check_fetch_fresh_post_action_observation(backend, env, operation):
    env.xml = semantic_ui(semantic_control("check-box", "Enabled", states='s:checked="true"'))
    snapshot = backend.observe()
    step = (
        AgentStep("wait", "wait", value="250")
        if operation == "wait"
        else AgentStep("check", "check", target=AgentTarget(name="Enabled"))
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    original = env._get_obs
    observed = []

    def observe():
        observed.append(True)
        return original()

    env._get_obs = observe
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and backend.verify(action, receipt).passed
    assert len(observed) == 2 and env.calls == []


@pytest.mark.parametrize("recovery", ["observe", "replan"])
def test_step_response_before_async_completion_requires_fresh_pending_observation(backend, env, recovery):
    env.change_after = False
    env.xml = semantic_ui(semantic_control())
    snapshot = backend.observe()
    step = AgentStep(
        "save",
        "click",
        target=AgentTarget(name="Save"),
        recovery=recovery,
        postconditions=(AgentCondition("exists", AgentTarget(name="Done", role="label")),),
    )
    action = backend.compile(step, snapshot, snapshot)[0]
    original = env.step

    def delayed_step(action, pause):
        response = original(action, pause)
        # The effect settles after the snapshot returned by the typed input.
        env.xml = semantic_ui(semantic_control() + '<label name="Done" s:showing="true"/>')
        return response

    env.step = delayed_step
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and not backend.verify(action, receipt).passed
    assert backend.verify_pending(action, receipt).passed is (recovery == "observe")
    assert len(env.calls) == 1


def test_failed_multiaction_does_not_retain_earlier_step_response(backend, env):
    original = env.step

    def reject_typing(action, pause):
        if action["action_type"] == "TYPING":
            raise RuntimeError("simulated input failure")
        return original(action, pause)

    env.step = reject_typing
    action = candidate(backend, "visual_fill", "query")
    receipt = backend.execute(action, "obs", "decision")
    assert not receipt.success and not backend.verify_pending(action, receipt).passed
    assert backend._action_observation is None
