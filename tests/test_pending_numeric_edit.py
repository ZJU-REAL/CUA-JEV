"""Same-control numeric editor text is diagnostic, never committed success."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import (
    condition_evidence,
    navigation_control_evidence,
    pending_numeric_buffer_match,
    value_matches,
)
from cua_jev.osworld_backend import OSWorldBackend, _accessibility_observation
from cua_jev.trace import JsonlTrace

STATE = "{https://accessibility.ubuntu.example.org/ns/state}"
VALUE = "{https://accessibility.ubuntu.example.org/ns/value}"
COMPONENT = "{https://accessibility.ubuntu.example.org/ns/component}"
FOREIGN = "{https://unrelated.invalid/ns/state}"
FIXTURE = json.loads((Path(__file__).parent / "fixtures/osworld_pending_numeric_edit.json").read_text())
TARGET = AgentTarget(name="Minimal number of undo levels:", app="gimp", window="Preferences")


def tree():
    root = ET.fromstring(FIXTURE["xml"])
    return root, next(root.iter("spin-button"))


def snapshot(root):
    controls, metadata = [], {}
    title, text, semantic = _accessibility_observation(
        ET.tostring(root, encoding="unicode"), screen_size=(1920, 1080),
        controls_out=controls, metadata=metadata,
    )
    return AgentSnapshot("osworld://desktop", title, text, tuple(controls),
                         semantic_text=semantic, a11y_complete=metadata["complete"])


def field(observation):
    return next(c for c in observation.controls if c.role == "spinbutton")


def test_recorded_same_node_keeps_committed_value_and_buffer_separate():
    root, node = tree()
    assert FIXTURE["source_sha256"] == "2d15afc4eab060972181aec36b130b2e0a8b6c4e389c367207257ba50a8db02c"
    observed = snapshot(root)
    control = field(observed)
    assert observed.a11y_complete
    assert control.label == TARGET.name and control.app == "gimp" and control.window == "Preferences"
    assert node.text == control.edit_buffer == "100"
    assert node.get(VALUE + "value") == control.value == "5.0"
    assert control.to_dict()["edit_buffer"] == "100"
    assert not value_matches(control, "100")
    assert condition_evidence(AgentCondition("value_equals", TARGET, "100"), observed) == {
        "outcome": "false", "reason": "state_mismatch",
    }
    assert pending_numeric_buffer_match(control, "100") == "literal"


def test_buffer_is_fingerprint_evidence_but_not_an_independent_navigation_effect():
    root, node = tree()
    before = snapshot(root)
    node.text = "101"
    after = snapshot(root)
    assert field(before).value == field(after).value
    assert before.fingerprint() != after.fingerprint()
    assert navigation_control_evidence(field(before)) == navigation_control_evidence(field(after))


@pytest.mark.parametrize("source", ["valuetext", "child", "sibling", "absent"])
def test_display_from_another_location_is_not_the_same_control_edit_buffer(source):
    root, node = tree()
    node.text = None
    if source == "valuetext":
        node.set("{https://accessibility.ubuntu.example.org/ns/attributes}valuetext", "100")
    elif source == "child":
        ET.SubElement(node, "label", {"name": "100"}).text = "100"
    elif source == "sibling":
        parent = next(p for p in root.iter() if node in list(p))
        ET.SubElement(parent, "label", {"name": "100"}).text = "100"
    observed = snapshot(root)
    assert field(observed).edit_buffer is None and field(observed).value == "5.0"


@pytest.mark.parametrize("state", ["editable", "focused", "enabled", "sensitive", "visible", "showing"])
@pytest.mark.parametrize("trusted", ["false", None])
def test_foreign_state_cannot_supply_missing_or_false_numeric_buffer_evidence(state, trusted):
    root, node = tree()
    node.attrib.pop(STATE + state)
    if trusted is not None:
        node.set(STATE + state, trusted)
    node.set(FOREIGN + state, "true")
    assert field(snapshot(root)).edit_buffer is None


@pytest.mark.parametrize("mutation", ["value", "current", "geometry", "no_geometry", "plain_state"])
def test_buffer_requires_same_trusted_value_and_geometry(mutation):
    root, node = tree()
    if mutation in {"value", "current"}:
        node.set("{https://unrelated.invalid/ns/value}" + mutation, "5.0")
    elif mutation == "geometry":
        node.attrib.pop(COMPONENT + "screencoord")
        node.set("{https://unrelated.invalid/ns/component}screencoord", "(579, 547)")
    elif mutation == "no_geometry":
        node.attrib.pop(COMPONENT + "size")
    else:
        node.set("focused", node.attrib.pop(STATE + "focused"))
    assert field(snapshot(root)).edit_buffer is None


@pytest.mark.parametrize("role", ["entry", "text", "slider", "canvas"])
def test_only_actual_editable_spinbutton_exposes_a_numeric_buffer(role):
    root, node = tree()
    node.tag = role
    assert all(control.edit_buffer is None for control in snapshot(root).controls)


@pytest.mark.parametrize("length,complete", [(128, True), (129, False)])
@pytest.mark.parametrize("character", ["1", " "])
def test_raw_buffer_bound_never_truncates_then_claims_complete(length, complete, character):
    root, node = tree()
    node.text = character * length
    observed = snapshot(root)
    assert observed.a11y_complete is complete
    assert field(observed).edit_buffer == (node.text if complete else None)


@pytest.mark.parametrize("buffer,expected,match", [
    ("100", "100", "literal"), ("100.0", "100", "numeric"),
    ("1e2", "+100.00", "numeric"), ("100.0000000000000000001", "100", None),
    ("NaN", "NaN", None), ("Infinity", "Infinity", None), ("100px", "100px", None),
    ("100,0", "100,0", None), ("１００", "１００", None), ("100 ", "100 ", None),
    ("1e1001", "1e1001", None), ("1" * 129, "1" * 129, None),
])
def test_pending_diagnostic_uses_only_bounded_finite_ascii_numbers(buffer, expected, match):
    root, _ = tree()
    control = replace(field(snapshot(root)), edit_buffer=buffer)
    assert pending_numeric_buffer_match(control, expected) == match


@pytest.mark.parametrize("committed", ["100", "100.0", "1e2", "NaN", "5px", "1" * 129])
def test_committed_equal_or_non_numeric_cannot_be_a_pending_numeric_edit(committed):
    root, _ = tree()
    assert pending_numeric_buffer_match(replace(field(snapshot(root)), value=committed), "100") is None


class NumericEnvironment:
    def __init__(self):
        self.root, self.node = tree()
        self.node.text = "5"
        self.actions = []
        self.fail_typing = False
        self.on_typing = None
        self.lose_focus_on_pending = False
        self.after_typing_reads = None
        buffer = io.BytesIO()
        Image.new("RGB", (1920, 1080), "white").save(buffer, format="PNG")
        self.frame = buffer.getvalue()

    def _get_obs(self):
        if self.after_typing_reads is not None:
            self.after_typing_reads += 1
            if self.lose_focus_on_pending and self.after_typing_reads >= 2:
                self.node.set(STATE + "focused", "false")
        return {"screenshot": self.frame,
                "accessibility_tree": ET.tostring(self.root, encoding="unicode")}

    def step(self, action, pause):
        self.actions.append(action)
        if action["action_type"] == "TYPING":
            if self.fail_typing:
                raise RuntimeError("typing failed")
            self.node.text = action["parameters"]["text"]
            self.after_typing_reads = 0
            if self.on_typing is not None:
                self.on_typing()
        elif action["action_type"] == "PRESS":
            assert action["parameters"]["key"] == "enter"
            self.node.set(VALUE + "value", "100.0")
        return self._get_obs(), 0, False, {}


def filled_environment(*, recovery="observe", setup=None):
    env = NumericEnvironment()
    if setup is not None:
        setup(env)
    backend = OSWorldBackend(env, pause=0)
    before = backend.observe()
    step = AgentStep("fill", "fill", target=TARGET, value="100", recovery=recovery,
                     postconditions=(AgentCondition("value_equals", TARGET, "100"),))
    candidate = backend.compile(step, before, before)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    return env, backend, candidate, receipt


def test_real_fill_diagnoses_buffer_without_verification_input_replay_or_extra_budget():
    env, backend, candidate, receipt = filled_environment()
    assert receipt.success and receipt.output["environment_steps"] == 3
    result = backend.verify(candidate, receipt)
    assert not result.passed and not result.details["postconditions_met"]
    diagnostic = result.details["pending_numeric_edit"]
    assert diagnostic == receipt.output["pending_numeric_edit"]
    assert diagnostic == {
        "reason": "edit_buffer_matches_but_committed_value_differs",
        "target": {"name": TARGET.name, "role": "spinbutton", "app": "gimp", "window": "Preferences"},
        "control_ref": field(backend._snapshot(env._get_obs())).ref,
        "committed_value": "5.0", "edit_buffer": "100", "buffer_match": "literal",
        "focused": True, "task_success": False,
    }
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert backend.take_unverified_recovery_context(candidate, receipt) is None
    pending = backend.verify_pending(candidate, receipt)
    assert not pending.passed and pending.details["pending_numeric_edit"] == diagnostic
    env.node.set(VALUE + "value", "100.0")
    settled = backend.verify_pending(candidate, receipt)
    assert settled.passed and "pending_numeric_edit" not in settled.details
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    backend.close()


@pytest.mark.parametrize("mutation", ["unfocused", "duplicate", "incomplete", "new_value", "textbox"])
def test_latest_pending_observation_must_still_support_the_diagnostic(mutation):
    env, backend, candidate, receipt = filled_environment()
    if mutation == "unfocused":
        env.node.set(STATE + "focused", "false")
    elif mutation == "duplicate":
        parent = next(p for p in env.root.iter() if env.node in list(p))
        duplicate = copy.deepcopy(env.node)
        duplicate.set("name", TARGET.name)
        env.node.set("name", TARGET.name)
        parent.append(duplicate)
    elif mutation == "incomplete":
        env.node.text = "100" + " " * 126
    elif mutation == "new_value":
        env.node.text = "101"
    else:
        env.node.tag = "entry"
    result = backend.verify_pending(candidate, receipt)
    assert not result.passed and "pending_numeric_edit" not in result.details
    assert len(env.actions) == 3


@pytest.mark.parametrize("change", ["copy_receipt", "mutate_payload", "new_input", "new_execute", "close"])
def test_diagnostic_is_bound_to_original_receipt_payload_candidate_and_input_generation(change):
    env, backend, candidate, receipt = filled_environment()
    if change == "copy_receipt":
        receipt = copy.deepcopy(receipt)
    elif change == "mutate_payload":
        receipt.output["pending_numeric_edit"]["committed_value"] = "999"
    elif change == "new_input":
        backend._step("PRESS", {"key": "enter"})
    elif change == "new_execute":
        backend.execute(candidate, "new-observation", "new-decision")
    else:
        backend.close()
    assert "pending_numeric_edit" not in backend.verify(candidate, receipt).details
    assert "pending_numeric_edit" not in backend.verify_pending(candidate, receipt).details


def test_changed_candidate_cannot_recover_diagnostic():
    env, backend, candidate, receipt = filled_environment()
    forged = replace(candidate, description="different candidate")
    assert "pending_numeric_edit" not in backend.verify(forged, receipt).details
    assert "pending_numeric_edit" not in backend.verify_pending(forged, receipt).details
    assert len(env.actions) == 3


@pytest.mark.parametrize("change", ["copy_receipt", "new_input"])
def test_exhausted_pending_budget_cannot_resurrect_unbound_diagnostic(change):
    env, backend, candidate, receipt = filled_environment()
    for _ in range(2):
        assert "pending_numeric_edit" in backend.verify_pending(candidate, receipt).details
    if change == "copy_receipt":
        receipt = copy.deepcopy(receipt)
    else:
        backend._step("PRESS", {"key": "enter"})
    result = backend.verify_pending(candidate, receipt)
    assert not result.passed and "pending_numeric_edit" not in result.details


@pytest.mark.parametrize("mutation", ["unfocused", "incomplete", "committed_equal"])
def test_final_typing_response_itself_must_support_pending_diagnostic(mutation):
    def setup(env):
        def mutate():
            if mutation == "unfocused":
                env.node.set(STATE + "focused", "false")
            elif mutation == "incomplete":
                env.node.text += " " * 126
            else:
                env.node.set(VALUE + "value", "100.0")
        env.on_typing = mutate
    env, backend, candidate, receipt = filled_environment(setup=setup)
    assert receipt.success and len(env.actions) == 3
    assert "pending_numeric_edit" not in receipt.output
    assert "pending_numeric_edit" not in backend.verify(candidate, receipt).details


def test_stale_preflight_sends_no_input_and_exposes_no_pending_diagnostic():
    env = NumericEnvironment()
    backend = OSWorldBackend(env, pause=0)
    before = backend.observe()
    step = AgentStep("fill", "fill", target=TARGET, value="100",
                     postconditions=(AgentCondition("value_equals", TARGET, "100"),))
    candidate = backend.compile(step, before, before)[0]
    env.node.text = "100"
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and receipt.output["environment_steps"] == 0
    assert "pending_numeric_edit" not in receipt.output
    assert "pending_numeric_edit" not in backend.verify(candidate, receipt).details
    assert env.actions == []


def test_failed_partial_fill_has_no_pending_numeric_diagnostic():
    env, backend, candidate, receipt = filled_environment(setup=lambda env: setattr(env, "fail_typing", True))
    assert not receipt.success and receipt.output["environment_steps"] == 3
    assert "pending_numeric_edit" not in receipt.output
    assert "pending_numeric_edit" not in backend.verify_pending(candidate, receipt).details
    assert len(env.actions) == 3


def test_only_explicit_guarded_commit_step_can_send_enter_and_verify_committed_value():
    env, backend, candidate, receipt = filled_environment(recovery="replan")
    assert not backend.verify(candidate, receipt).passed
    assert len(env.actions) == 3
    before = backend.observe()
    commit = AgentStep("commit", "key_press", value="enter",
                       preconditions=(AgentCondition("focused", TARGET),),
                       postconditions=(AgentCondition("value_equals", TARGET, "100"),))
    offered = backend.compile(commit, before, before)
    assert len(offered) == 1
    committed = backend.execute(offered[0], "new-observation", "new-decision")
    assert committed.success and backend.verify(offered[0], committed).passed
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert "pending_numeric_edit" not in committed.output


@pytest.mark.parametrize("strategy", ["commit", "refill", "focus_lost"])
def test_real_runner_forwards_latest_diagnostic_without_committing_or_resetting_failures(
    strategy, monkeypatch,
):
    monkeypatch.setattr("time.sleep", lambda _: None)
    if strategy == "focus_lost":
        monkeypatch.setattr("time.perf_counter", lambda: 1000.0)
    env = NumericEnvironment()
    env.lose_focus_on_pending = strategy == "focus_lost"
    backend, trace = OSWorldBackend(env, pause=0), JsonlTrace()
    target = {"name": TARGET.name, "role": "spinbutton", "app": "gimp", "window": "Preferences"}
    effect = {"kind": "value_equals", "target": target, "value": "100"}

    class Planning:
        def __init__(self):
            self.calls = []

        def plan(self, goal, observation, history, *, max_actions):
            self.calls.append({"max_actions": max_actions, "history": copy.deepcopy(history)})
            if strategy == "focus_lost" and len(self.calls) == 2:
                raw = {"summary": "Stop to inspect the lost focus", "done": True, "steps": []}
            elif strategy == "commit" and len(self.calls) == 2:
                assert max_actions == 1
                raw = {"summary": "Commit the already entered numeric edit", "done": False,
                       "steps": [{"id": "commit", "operation": "key_press", "value": "enter",
                                  "preconditions": [{"kind": "focused", "target": target}],
                                  "postconditions": [effect], "recovery": "observe"}],
                       "completion_conditions": [effect]}
            else:
                raw = {"summary": "Set the numeric field", "done": False,
                       "steps": [{"id": "fill", "operation": "fill", "target": target,
                                  "value": "100", "postconditions": [effect], "recovery": "observe"}]}
            plan = parse_plan(raw, observation, profile="desktop", max_actions=max_actions)
            _validate_desktop_execution_plan(plan, observation, max_actions)
            return plan

    planner = Planning()
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, max_failures=3, max_steps=6,
                           max_plans=2 if strategy == "focus_lost" else 5),
    ).run("Set minimum undo steps to 100")
    records = [e["payload"] for e in trace.events if e["kind"] == "agent_step"]
    histories = planner.calls[1]["history"]
    feedback = next(item for item in reversed(histories) if item.get("pending_numeric_edit")
                    or item.get("operation") == "fill")
    assert not records[0]["verified"] and result.context_recoveries == 0
    assert result.post_snapshot_reuses == 0
    if strategy == "focus_lost":
        # This hook loses focus only on a later read. Eager replanning uses
        # the real original typing frame; advisory done sends no input and
        # performs no such read. Do not claim that lost focus was observed.
        assert feedback["pending_numeric_edit_source"] == "original_unverified_typing_response"
        assert feedback["pending_numeric_edit"]["edit_buffer"] == "100"
        assert env.after_typing_reads == 1 and result.recovery_observations == 0
        assert result.status == "unverified" and not result.success and result.plan_completion_claims == 0
        assert not any(action["action_type"] == "PRESS" for action in env.actions)
        assert any("pending_numeric_edit" in record[1].output for record in backend._verified.values())
        assert len(planner.calls) == 2 and len(env.actions) == 3
    else:
        assert feedback["pending_numeric_edit"]["edit_buffer"] == "100"
        assert feedback["pending_numeric_edit"]["committed_value"] == "5.0"
        assert "instead of repeating the fill" in feedback["feedback"]
        if strategy == "commit":
            assert [record["verified"] for record in records] == [False, True]
            assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
            assert len(planner.calls) == 2 and result.plan_completion_claims == 1
            assert result.status == "unverified" and env.node.get(VALUE + "value") == "100.0"
        else:
            assert result.status == "stuck" and result.reason == "consecutive action failures"
            assert [record["verified"] for record in records] == [False, False, False]
            assert len(planner.calls) == 3 and len(env.actions) == 9
            assert all(action["action_type"] != "PRESS" for action in env.actions)
            assert env.node.get(VALUE + "value") == "5.0"
