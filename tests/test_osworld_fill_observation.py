"""Intermediate AX omission must preserve real input and effect observations."""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.osworld_step_observation import (
    IMAGE_ONLY_INFO_KEY,
    IMAGE_ONLY_VERSION,
    native_image_only_step,
)


class Environment:
    require_a11y_tree = True

    def __init__(self):
        self.value = "old"
        self.focused = True
        self.window = "Editor"
        self.actions = []
        self.reads = []
        self.change_after = None
        self.fail_after = None
        self.on_read = lambda: None

    def _get_obs(self):
        self.reads.append(self.require_a11y_tree)
        self.on_read()
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white" if self.value == "old" else "gray").save(buffer, "PNG")
        focused = 's:focused="true"' if self.focused else ""
        tree = (f'<desktop xmlns:s="urn:state" xmlns:cp="urn:component">'
                f'<application name="editor"><frame name="{self.window}" s:active="true" '
                's:visible="true" s:showing="true" cp:screencoord="(0, 0)" cp:size="(800, 600)">'
                f'<entry name="Field" s:editable="true" s:visible="true" s:showing="true" {focused} '
                f'cp:screencoord="(100, 100)" cp:size="(200, 30)">{self.value}</entry>'
                '</frame></application></desktop>')
        return {"screenshot": buffer.getvalue(),
                "accessibility_tree": tree if self.require_a11y_tree else None}

    def step(self, action, pause):
        self.actions.append(action)
        kind = action["action_type"]
        if self.change_after == kind:
            self.window = "Other dialog"
        if self.fail_after == kind:
            raise RuntimeError("uncertain response")
        if kind == "TYPING":
            self.value = action["parameters"]["text"]
        return self._get_obs(), 0, False, {}


def offered(backend, operation="fill"):
    snapshot = backend.observe()
    target = AgentTarget(name="Field", role="textbox", app="editor")
    step = AgentStep("fill", operation, value="new", target=target if operation == "fill" else None,
                     box=(125, 166, 375, 216) if operation == "visual_fill" else None,
                     label="Fill Field", postconditions=(AgentCondition("value_equals", target, "new"),))
    return backend.compile(step, snapshot, snapshot)[0]


@pytest.mark.parametrize("operation,full", [("fill", 4), ("visual_fill", 2)])
@pytest.mark.parametrize("enabled", [False, True])
def test_fill_only_omits_two_intermediate_ax_reads(operation, full, enabled):
    env = Environment()
    backend = OSWorldBackend(env, pause=0, image_only_fill_intermediates=enabled)
    action = offered(backend, operation)
    env.reads.clear()
    receipt = backend.execute(action, "observation", "decision")
    assert receipt.success and backend.verify(action, receipt).passed
    assert [x["action_type"] for x in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert env.reads.count(True) == full + (0 if enabled else 2)
    assert env.reads.count(False) == (2 if enabled else 0)
    assert env.require_a11y_tree is True
    assert backend.last_observation["accessibility_tree"] is not None
    assert backend._action_observation is None
    assert backend._post_snapshot[3].a11y_complete
    assert backend.intermediate_observation_metrics["image_only_steps"] == (2 if enabled else 0)
    backend.close()


@pytest.mark.parametrize("change_after,expected", [("CLICK", ["CLICK"]), ("HOTKEY", ["CLICK", "HOTKEY"])])
def test_both_original_fresh_focus_checks_still_refuse_changed_window(change_after, expected):
    env = Environment()
    backend = OSWorldBackend(env, pause=0, image_only_fill_intermediates=True)
    action = offered(backend)
    env.change_after = change_after
    receipt = backend.execute(action, "observation", "decision")
    assert not receipt.success and not backend.verify(action, receipt).passed
    assert [x["action_type"] for x in env.actions] == expected
    assert env.require_a11y_tree is True and backend._post_snapshot is None
    assert backend._action_observation is None
    backend.close()


@pytest.mark.parametrize("read_number,expected", [(1, []), (2, ["CLICK"]), (3, ["CLICK", "HOTKEY"])])
def test_deadline_after_each_full_read_still_blocks_next_input(monkeypatch, read_number, expected):
    now = [100.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    env = Environment()
    backend = OSWorldBackend(env, pause=0, image_only_fill_intermediates=True)
    action = offered(backend)
    env.reads.clear()

    def advance():
        if env.reads.count(True) == read_number:
            now[0] += 2

    env.on_read = advance
    receipt = backend.execute_with_deadline(action, "observation", "decision", deadline=101.0)
    assert not receipt.success and receipt.output["failure_code"] == "input_deadline_exceeded"
    assert [x["action_type"] for x in env.actions] == expected
    assert receipt.output["environment_steps"] == len(expected)
    assert env.require_a11y_tree is True and backend._post_snapshot is None
    backend.close()


@pytest.mark.parametrize("kind", ["CLICK", "HOTKEY"])
def test_error_after_submission_restores_accessibility_and_never_retries(kind):
    env = Environment()
    backend = OSWorldBackend(env, pause=0, image_only_fill_intermediates=True)
    action = offered(backend)
    env.fail_after = kind
    receipt = backend.execute(action, "observation", "decision")
    assert not receipt.success
    assert [x["action_type"] for x in env.actions].count(kind) == 1
    assert all(x["action_type"] != "TYPING" for x in env.actions)
    assert env.require_a11y_tree is True
    assert backend._action_observation is None and backend._post_snapshot is None
    backend.close()


@pytest.mark.parametrize("operation", ["fill", "visual_fill"])
@pytest.mark.parametrize("explicit_conditions", [False, True])
def test_final_typing_cannot_relabel_a_partial_response_as_full_effect(operation, explicit_conditions):
    env = Environment()
    backend = OSWorldBackend(env, pause=0, image_only_fill_intermediates=True)
    if explicit_conditions:
        action = offered(backend, operation)
    else:
        snapshot = backend.observe()
        target = AgentTarget(name="Field", role="textbox", app="editor")
        step = AgentStep("fill", operation, value="new", target=target if operation == "fill" else None,
                         box=(125, 166, 375, 216) if operation == "visual_fill" else None, label="Fill Field")
        action = backend.compile(replace(step, postconditions=()), snapshot, snapshot)[0]
    normal_step = env.step

    def partial_typing(action, pause):
        observation, reward, done, info = normal_step(action, pause)
        if action["action_type"] == "TYPING":
            observation["accessibility_tree"] = None
            info = {IMAGE_ONLY_INFO_KEY: IMAGE_ONLY_VERSION}
        return observation, reward, done, info

    env.step = partial_typing
    receipt = backend.execute(action, "observation", "decision")
    assert not receipt.success and not backend.verify(action, receipt).passed
    assert len(env.actions) == 3 and env.require_a11y_tree is True
    assert backend._action_observation is None and backend._post_snapshot is None
    backend.close()


@pytest.mark.parametrize("action", [
    {"action_type": "TYPING", "parameters": {"text": "new"}},
    {"action_type": "CLICK", "parameters": {"x": 1, "y": 2, "button": "right", "num_clicks": 1}},
    {"action_type": "CLICK", "parameters": {"x": 1, "y": 2, "button": "left", "num_clicks": 2}},
    {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "s"]}},
])
def test_image_only_helper_never_submits_other_actions(action):
    env = Environment()
    with pytest.raises(ValueError):
        native_image_only_step(env, action, pause=0)
    assert env.actions == [] and env.require_a11y_tree is True


@pytest.mark.parametrize("value", [None, False, 1, "true"])
def test_feature_requires_actual_enabled_accessibility(value):
    env = Environment()
    env.require_a11y_tree = value
    with pytest.raises(ValueError):
        OSWorldBackend(env, image_only_fill_intermediates=True)
    assert env.actions == []
