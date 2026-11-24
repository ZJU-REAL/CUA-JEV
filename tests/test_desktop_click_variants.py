"""Explicit mouse semantics survive planning, binding and the typed worker boundary."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_browser import BrowserAgentBackend
from cua_jev.agent_planner import MultimodalPlanner, parse_plan
from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend


def png(color="white"):
    output = io.BytesIO()
    Image.new("RGB", (800, 600), color).save(output, format="PNG")
    return output.getvalue()


class Environment:
    def __init__(self):
        self.frame = png()
        self.calls = []
        self.closed = False
        self.controller = SimpleNamespace(execute_python_command=lambda _: {"status": "success"})

    def _get_obs(self):
        return {
            "screenshot": self.frame,
            "accessibility_tree": (
                '<desktop xmlns:c="urn:component"><frame name="Example" active="true">'
                '<button name="Item" c:screencoord="(80, 60)" c:size="(160, 120)"/>'
                '<entry name="Name" editable="true" c:screencoord="(80, 220)" c:size="(160, 60)"/>'
                '</frame></desktop>'
            ),
        }

    def step(self, action, pause=0):
        self.calls.append((copy.deepcopy(action), pause))
        self.controller.execute_python_command("trusted upstream input")
        self.frame = png("gray")
        return self._get_obs(), 0, False, {}

    def close(self):
        self.closed = True


@pytest.fixture
def desktop():
    env = Environment()
    backend = OSWorldBackend(env, pause=0)
    try:
        yield env, backend, backend.observe()
    finally:
        backend.close()


@pytest.fixture(scope="module")
def worker():
    scripts = Path(__file__).parents[1] / "scripts"
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(scripts))
        spec = importlib.util.spec_from_file_location(
            "click_worker_test", scripts / "osworld_remote_worker.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = old_path


def raw_plan(action, *, summary="Use the observed target"):
    return {"summary": summary, "steps": [action], "done": False}


def raw_click(snapshot, operation="visual_click", **kwargs):
    action = {"id": "input", "operation": operation}
    if operation == "click":
        action["ref"] = next(control.ref for control in snapshot.controls if control.label == "Item")
    else:
        action.update(label="Item", box=[100, 100, 300, 300], box_space="normalized_1000")
    return {**action, **kwargs}


@pytest.mark.parametrize("operation", ["click", "visual_click"])
@pytest.mark.parametrize("button, count", [("left", 1), ("right", 1), ("left", 2)])
def test_click_variants_reach_typed_input_and_receipt_without_inference(desktop, operation, button, count):
    env, backend, snapshot = desktop
    raw = raw_plan(raw_click(snapshot, operation, button=button, click_count=count))
    step = parse_plan(raw, snapshot, profile="desktop").steps[0]
    candidate = backend.compile(step, snapshot, snapshot)[0]
    assert candidate.arguments["button"] == button
    assert candidate.arguments["click_count"] == count
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success
    assert backend.verify(candidate, receipt).passed
    assert env.calls == [({"action_type": "CLICK", "parameters": {
        "x": 160, "y": 120, "button": button, "num_clicks": count,
    }}, 0)]
    assert (receipt.output["button"], receipt.output["click_count"]) == (button, count)


def test_summary_does_not_change_legacy_single_left_click(desktop):
    env, backend, snapshot = desktop
    step = parse_plan(raw_plan(raw_click(snapshot), summary="Right-click or double-click the item"),
                      snapshot, profile="desktop").steps[0]
    assert (step.button, step.click_count) == ("left", 1)
    receipt = backend.execute(backend.compile(step, snapshot, snapshot)[0], "obs", "decision")
    assert receipt.success
    assert env.calls[0][0]["parameters"]["button"] == "left"
    assert env.calls[0][0]["parameters"]["num_clicks"] == 1


@pytest.mark.parametrize("button, count", [
    ("middle", 1), ("RIGHT", 1), (None, 1), (["left"], 1), (False, 1),
    ("left", True), ("left", 1.0), ("left", "2"), ("left", 0), ("left", 3), ("right", 2),
])
def test_invalid_click_variants_fail_parser_and_backend_before_input(desktop, button, count):
    env, backend, snapshot = desktop
    with pytest.raises(ValueError, match="click"):
        parse_plan(raw_plan(raw_click(snapshot, button=button, click_count=count)),
                   snapshot, profile="desktop")
    step = AgentStep("bad", "visual_click", box=(100, 100, 300, 300), label="Item",
                     button=button, click_count=count)
    assert backend.compile(step, snapshot, snapshot) == ()
    with pytest.raises(ValueError, match="click"):
        step.to_dict()
    assert env.calls == []


@pytest.mark.parametrize("operation", ["fill", "visual_fill", "check", "hotkey", "wait"])
@pytest.mark.parametrize("field, value", [("button", "left"), ("click_count", 1)])
def test_nonclick_explicit_mouse_fields_are_illegal_even_when_default(desktop, operation, field, value):
    env, backend, snapshot = desktop
    with pytest.raises(ValueError, match="only click and visual_click"):
        parse_plan(raw_plan({"id": "bad", "operation": operation, field: value}),
                   snapshot, profile="desktop")
    assert backend.compile(AgentStep("bad", operation, button="right"), snapshot, snapshot) == ()
    assert env.calls == []


@pytest.mark.parametrize("button, count", [("right", 1), ("left", 2)])
def test_click_variants_keep_immediate_preflight_and_candidate_integrity(desktop, button, count):
    env, backend, snapshot = desktop
    step = parse_plan(raw_plan(raw_click(snapshot, button=button, click_count=count)),
                      snapshot, profile="desktop").steps[0]
    candidate = backend.compile(step, snapshot, snapshot)[0]
    altered = replace(candidate, arguments={**candidate.arguments, "button": "left", "click_count": 1})
    assert not backend.execute(altered, "obs", "decision").success
    assert env.calls == []
    candidate = backend.compile(step, snapshot, snapshot)[0]
    env.frame = png("black")
    assert backend.compile(step, snapshot, backend.observe()) == ()
    assert not backend.execute(candidate, "obs", "decision").success
    assert env.calls == []


@pytest.mark.parametrize("button, count", [("left", 1), ("right", 1), ("left", 2)])
def test_worker_session_preserves_validated_mouse_semantics(worker, tmp_path, button, count):
    env = Environment()
    action = {"action_type": "CLICK", "parameters": {
        "x": 160, "y": 120, "button": button, "num_clicks": count,
    }}
    original = copy.deepcopy(action)
    session = worker.WorkerSession(env, env._get_obs(), tmp_path)
    try:
        response = session.handle({"id": 1, "method": "step", "action": action, "pause": 0.25})
        assert env.calls == [(original, 0.25)]
        assert action == original
        assert response["observation"]["screenshot"]
        assert response["reward"] == 0 and response["done"] is False
    finally:
        session.close()
    assert env.closed and session.steps == 1


@pytest.mark.parametrize("button, count", [
    ("middle", 1), (["left"], 1), (None, 1), ("left", True), ("left", 1.0),
    ("left", 0), ("left", 3), ("right", 2),
])
def test_worker_rejects_invalid_mouse_semantics_before_environment_input(worker, tmp_path, button, count):
    env = Environment()
    session = worker.WorkerSession(env, env._get_obs(), tmp_path)
    try:
        with pytest.raises(worker.ProtocolError, match="invalid click"):
            session.handle({"id": 1, "method": "step", "action": {
                "action_type": "CLICK", "parameters": {
                    "x": 160, "y": 120, "button": button, "num_clicks": count,
                },
            }})
        assert env.calls == [] and session.steps == 0
    finally:
        session.close()


@pytest.mark.parametrize("operation", ["click", "visual_click"])
@pytest.mark.parametrize("button, count", [("right", 1), ("left", 2)])
def test_browser_rejects_unsupported_mouse_variants_at_both_boundaries(desktop, operation, button, count):
    _, _, snapshot = desktop
    snapshot = replace(snapshot, url="https://example.test")
    action = raw_click(snapshot, operation, button=button, click_count=count)
    action.pop("box_space", None)
    with pytest.raises(ValueError, match="browser click supports only"):
        parse_plan(raw_plan(action), snapshot)
    backend = object.__new__(BrowserAgentBackend)
    backend._allowed = lambda _url: True
    step = AgentStep("input", operation, button=button, click_count=count)
    assert backend.compile(step, snapshot, snapshot) == ()


@pytest.mark.parametrize("kind", ["fill", "navigation", "right", "double", "wait"])
def test_step_serialization_roundtrips_desktop_without_illegal_nonclick_fields(desktop, kind):
    _, _, snapshot = desktop
    target = AgentTarget(name="Name")
    if kind == "fill":
        step = AgentStep("input", "fill", target=target, value="example",
                         postconditions=(AgentCondition("value_equals", target, "example"),))
    elif kind == "navigation":
        step = AgentStep("input", "click", target=AgentTarget(window="Example"), navigation=True,
                         label="Reach the target", postconditions=(AgentCondition("exists", target),))
    elif kind == "wait":
        step = AgentStep("input", "wait", value="250")
    else:
        step = AgentStep("input", "visual_click", label="Item", box=(100, 100, 300, 300),
                         button="right" if kind == "right" else "left",
                         click_count=2 if kind == "double" else 1)
    serialized = step.to_dict()
    if step.operation not in {"click", "visual_click"}:
        assert "button" not in serialized and "click_count" not in serialized
    parsed = parse_plan(raw_plan(json.loads(json.dumps(serialized))), snapshot, profile="desktop")
    assert parsed.steps == (step,)


@pytest.mark.parametrize("horizon", [1, 6])
@pytest.mark.parametrize("button,click_count", [("left", 1), ("right", 1), ("left", 2)])
def test_both_planning_horizons_receive_explicit_mouse_schema(desktop, horizon, button, click_count):
    _, _, snapshot = desktop
    response = raw_plan(raw_click(snapshot, button=button, click_count=click_count))
    # Narrative text cannot override explicit gesture fields in either format.
    response["summary"] = "Right-click the target"

    def transport(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if horizon == 1:
            assert 'button="left" or "right"' in prompt
            assert "click_count=1 or 2" in prompt
            assert "No other operation" in prompt
            assert "summary never changes these fields" in prompt
        else:
            assert "click/visual_click may declare button left/right and click_count 1/2" in prompt
            assert "right permits only 1" in prompt
            assert "The default is left/1" in prompt
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)}}]})

    planner = MultimodalPlanner("https://example.test/v1", "test", profile="desktop",
                                client=httpx.Client(transport=httpx.MockTransport(transport)))
    try:
        plan = planner.plan("Open the observed context menu", snapshot, [], max_actions=horizon)
        assert (plan.steps[0].button, plan.steps[0].click_count) == (button, click_count)
    finally:
        planner.close()
