"""Explicit attachment pixels map to the native desktop; defaults remain compatible."""

import base64
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _desktop_snapshot_payload, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image


@pytest.fixture
def snapshot():
    output = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(output, "PNG")
    return AgentSnapshot("osworld://desktop", "Editor", "", (), image=screenshot_image(output.getvalue()))


def visual(space="image_pixels"):
    return {"operation": "visual_click", "label": "Visible menu item", "box_space": space,
            "box": [95, 425, 165, 439]}


def call(snapshot, *, contract="legacy_v23", preference=None, limit=1, history=None):
    requests = []
    raw = {"summary": "Click current item", "steps": [{"id": "one", **visual()}], "done": False}
    if limit > 1 and not history and contract != "legacy_v23":
        raw = {"intent": "act_current", "label": "Click current item", "action": visual()}

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        settings = {} if preference is None else {"visual_box_space": preference}
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop",
                                    desktop_contract=contract, client=client, **settings)
        plan = planner.plan("Click the visible item", snapshot, history or [], max_actions=limit)
    return plan, requests[0]


@pytest.mark.parametrize("contract", [
    "legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
])
@pytest.mark.parametrize("limit,history", [(1, []), (6, []), (6, [{"recovery_required": True}])])
def test_pixel_preference_is_consistent_in_all_prompts_and_actual_jpeg_geometry(
    snapshot, contract, limit, history,
):
    _plan, body = call(snapshot, contract=contract, preference="image_pixels", limit=limit, history=history)
    instructions = body["messages"][0]["content"]
    assert 'PREFER box_space="normalized_1000"' not in instructions
    assert 'box_space="image_pixels"' in instructions
    assert "exactly 1280 pixels wide and 720 pixels high" in instructions
    state = json.loads(body["messages"][1]["content"][0]["text"])
    assert state["screenshot_geometry"] == {
        "width_px": 1280, "height_px": 720, "preferred_box_space": "image_pixels",
        "supported_box_spaces": ["normalized_1000", "image_pixels"], "default_box_space": "normalized_1000",
    }
    attachment = body["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]
    assert Image.open(io.BytesIO(base64.b64decode(attachment))).size == (1280, 720)


@pytest.mark.parametrize("contract", [
    "legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
])
@pytest.mark.parametrize("limit,history", [(1, []), (6, []), (6, [{"recovery_required": True}])])
def test_explicit_default_and_omitted_preference_have_identical_requests(snapshot, contract, limit, history):
    first = call(snapshot, contract=contract, limit=limit, history=history)
    second = call(snapshot, contract=contract, preference="normalized_1000", limit=limit, history=history)
    assert first == second


def test_box_conversion_uses_encoded_image_and_execution_uses_native_desktop(snapshot):
    raw = {"summary": "Current visible target", "steps": [{"id": "one", **visual()}], "done": False}
    step = parse_plan(raw, snapshot, profile="desktop", max_actions=1).steps[0]
    assert step.box == (74, 590, 129, 610)

    class Desktop:
        actions = []

        def _get_obs(self):
            return snapshot

        def step(self, action, pause=0):
            self.actions.append(action)
            return snapshot, 0, False, {}

    class Backend(OSWorldBackend):
        def _snapshot(self, observation):
            return observation

    env = Desktop()
    backend = Backend(env, pause=0)
    candidate = backend.compile(step, snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "obs", "decision")
    assert receipt.success and env.actions == [{"action_type": "CLICK", "parameters": {
        "x": 194, "y": 648, "button": "left", "num_clicks": 1,
    }}]
    assert not backend.verify(candidate, receipt).passed  # A click alone proves no task effect.
    raw["steps"][0]["box_space"] = "normalized_1000"
    declared_normalized = parse_plan(raw, snapshot, profile="desktop", max_actions=1).steps[0]
    assert declared_normalized.box == (95, 425, 165, 439)  # Never infer/relabel a declared space.


def test_explicit_control_box_names_preserve_bounds_and_capabilities_without_mutating_snapshot(snapshot):
    observed = replace(snapshot, controls=(AgentControl(
        "file", "Visible object", "label", role="label", app="Desktop", box=(938, 852, 993, 868),
    ),))
    before = observed.fingerprint()
    old = _desktop_snapshot_payload(observed)
    explicit = _desktop_snapshot_payload(observed, explicit_box_spaces=True)
    assert explicit["controls"][0]["box_1000"] == (938, 852, 993, 868)
    assert "box" not in explicit["controls"][0] and "box" not in explicit["control_defaults"]
    assert explicit["controls"][0]["operations"] == []  # Geometry grants no native action.
    explicit["control_defaults"]["box"] = explicit["control_defaults"].pop("box_1000")
    for c in explicit["controls"]:
        if "box_1000" in c:
            c["box"] = c.pop("box_1000")
    assert explicit == old and observed.fingerprint() == before


@pytest.mark.parametrize("limit,history", [(1, []), (6, []), (6, [{"recovery_required": True}])])
def test_v33_labels_native_boxes_separately_from_attachment_pixels_in_both_modes(snapshot, limit, history):
    observed = replace(snapshot, controls=(AgentControl(
        "file", "Visible object", "label", role="label", app="Desktop", box=(938, 852, 993, 868),
    ),))
    _plan, request = call(observed, contract="property_intent_v33", preference="image_pixels",
                          limit=limit, history=history)
    state = json.loads(request["messages"][1]["content"][0]["text"])
    control = state["snapshot"]["controls"][0]
    assert state["screenshot_geometry"]["control_box_space"] == "normalized_1000"
    assert state["screenshot_geometry"]["height_px"] == 720
    assert control["box_1000"][3] == 868 and "box" not in control
    instructions = request["messages"][0]["content"]
    assert ("NOT JPEG pixels" in instructions
            or "box_1000 measurements remain normalized" in instructions)

    # Directly reusing the explicitly normalized observed box is legal. Labelling
    # those same numbers as JPEG pixels is rejected, not silently corrected.
    raw = {"summary": "Click visible object", "done": False, "steps": [{
        "id": "one", "operation": "visual_click", "label": "Visible object",
        "box_space": "normalized_1000", "box": control["box_1000"],
    }]}
    step = parse_plan(raw, observed, profile="desktop", max_actions=1).steps[0]
    assert step.box == observed.controls[0].box
    raw["steps"][0]["box_space"] = "image_pixels"
    with pytest.raises(ValueError, match="fit this screenshot"):
        parse_plan(raw, observed, profile="desktop", max_actions=1)


@pytest.mark.parametrize("preference,profile", [([], "desktop"), (True, "desktop"), ("unknown", "desktop"),
                                               ("image_pixels", "browser")])
def test_invalid_or_browser_pixel_preference_is_rejected_before_client_construction(preference, profile):
    with pytest.raises(ValueError, match="visual box space"):
        MultimodalPlanner("https://offline.invalid", "mock", profile=profile, visual_box_space=preference)
