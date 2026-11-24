"""Observed proxy-parent geometry helps planning without adding executable inputs."""

import base64
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _desktop_snapshot_payload
from cua_jev.agent_types import AgentStep, AgentTarget
from cua_jev.desktop_input_context import input_geometry_payload
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image

STATE = "{https://accessibility.ubuntu.example.org/ns/state}"
COMPONENT = "{https://accessibility.ubuntu.example.org/ns/component}"
ATTRIBUTE = "{https://accessibility.ubuntu.example.org/ns/attributes}"
FOREIGN = "{https://untrusted.example/ns/lookalike}"
VISIBLE = {STATE + key: "true" for key in ("enabled", "sensitive", "showing", "visible")}
PROXY_NAME = "Current editor accessibility proxy"


@pytest.fixture(scope="module")
def png():
    output = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(output, "PNG")
    return output.getvalue()


def form():
    root = ET.Element("desktop-frame")
    app = ET.SubElement(root, "application", {"name": "Editor", **VISIBLE})
    frame = ET.SubElement(app, "frame", {
        "name": "Settings", **VISIBLE, STATE + "active": "true",
        COMPONENT + "screencoord": "(70, 64)", COMPONENT + "size": "(1850, 1016)",
    })
    parent = ET.SubElement(frame, "section", {
        "name": "", **VISIBLE, ATTRIBUTE + "class": "observed editor wrapper",
        COMPONENT + "screencoord": "(578, 141)", COMPONENT + "size": "(997, 20)",
    })
    proxy = ET.SubElement(parent, "entry", {
        "name": PROXY_NAME, **VISIBLE, STATE + "editable": "true",
        STATE + "focused": "true", STATE + "focusable": "true",
        COMPONENT + "screencoord": "(723, 141)", COMPONENT + "size": "(1, 1)",
    })
    ET.SubElement(frame, "push-button", {
        "name": "Existing button", **VISIBLE,
        COMPONENT + "screencoord": "(100, 300)", COMPONENT + "size": "(100, 30)",
    })
    return root, frame, parent, proxy


def observed(root, png):
    # _snapshot only decodes the supplied public artifacts. This sentinel cannot
    # observe an environment or submit input even if the fixture is misused.
    backend = OSWorldBackend(object(), pause=0)
    snapshot = backend._snapshot({
        "screenshot": png, "accessibility_tree": ET.tostring(root, encoding="unicode"),
    })
    return backend, snapshot


def proxy_control(snapshot):
    return next(control for control in snapshot.controls if control.label == PROXY_NAME)


def test_actual_immediate_parent_is_projected_inward_from_raw_desktop_pixels(png):
    root, _frame, _parent, _proxy = form()
    backend, snapshot = observed(root, png)
    context, = snapshot.input_geometry
    payload = input_geometry_payload(snapshot, (1280, 720))
    assert context.input_rect_px == (723, 141, 724, 142)
    assert context.ancestor_rect_px == (578, 141, 1575, 161)
    assert payload["rect_image_pixels"] == [386, 94, 1050, 107]
    assert payload["ancestor"]["distance"] == 1
    assert payload["attachment_size_px"] == [1280, 720]
    assert payload["attachment_sha256"] == hashlib.sha256(snapshot.image.jpeg).hexdigest()
    assert payload["grants_input_capability"] is False
    assert backend.environment_steps == 0


def test_hint_does_not_change_controls_fingerprints_operations_or_candidates(png):
    root, _frame, _parent, _proxy = form()
    backend, snapshot = observed(root, png)
    without_hint = replace(snapshot, input_geometry=())
    proxy = proxy_control(snapshot)
    assert proxy.box is None
    assert snapshot == without_hint
    assert snapshot.to_dict() == without_hint.to_dict()
    assert snapshot.scope_key() == without_hint.scope_key()
    assert snapshot.fingerprint() == without_hint.fingerprint()
    assert snapshot.controls == without_hint.controls
    wire = _desktop_snapshot_payload(snapshot, explicit_box_spaces=True)
    assert next(c for c in wire["controls"] if c["ref"] == proxy.ref)["operations"] == []
    target = AgentTarget(name=PROXY_NAME, role="textbox", app="Editor", window="Settings")
    fill = AgentStep("edit", "fill", target=target, value="query")
    assert backend.compile(fill, snapshot, snapshot) == ()
    assert backend.compile(fill, without_hint, without_hint) == ()
    click = AgentStep("click", "click", target=AgentTarget(
        name="Existing button", role="button", app="Editor", window="Settings",
    ))
    candidates = backend.compile(click, snapshot, snapshot)
    assert len(candidates) == 1
    # Fresh issuance IDs differ on every compile; capabilities, arguments and
    # bindings must otherwise remain exactly equal when the hint is removed.
    other = backend.compile(click, without_hint, without_hint)
    assert [replace(c, id="issuance") for c in candidates] == [replace(c, id="issuance") for c in other]
    assert input_geometry_payload(snapshot, (1280, 720)) is not None
    assert snapshot.fingerprint() == without_hint.fingerprint()


@pytest.mark.parametrize("change", [
    "hidden-input", "disabled-input", "hidden-parent", "disabled-parent",
    "foreign-focus", "foreign-parent-component", "conflicting-parent-component", "negative-lookalike",
])
def test_untrusted_hidden_or_disabled_evidence_supplies_no_parent_hint(png, change):
    root, _frame, parent, proxy = form()
    if change == "hidden-input":
        proxy.set(STATE + "showing", "false")
    elif change == "disabled-input":
        proxy.set(STATE + "enabled", "false")
    elif change == "hidden-parent":
        parent.set(STATE + "visible", "false")
    elif change == "disabled-parent":
        parent.set(STATE + "sensitive", "false")
    elif change == "foreign-focus":
        proxy.set(FOREIGN + "focused", proxy.attrib.pop(STATE + "focused"))
    elif change == "foreign-parent-component":
        parent.set(FOREIGN + "screencoord", parent.attrib.pop(COMPONENT + "screencoord"))
    elif change == "conflicting-parent-component":
        parent.set(FOREIGN + "size", "(1500, 40)")
    else:
        proxy.set(FOREIGN + "editable", "false")
    _backend, snapshot = observed(root, png)
    assert snapshot.input_geometry == ()
    assert input_geometry_payload(snapshot, (1280, 720)) is None


@pytest.mark.parametrize("change", ["missing-geometry", "window-boundary", "out-of-bounds"])
def test_missing_boundary_or_outside_parent_geometry_is_not_presented(png, change):
    root, _frame, parent, _proxy = form()
    if change == "missing-geometry":
        parent.attrib.pop(COMPONENT + "size")
    elif change == "window-boundary":
        parent.tag = "window"
        parent.set("name", "Unrelated window boundary")
    else:
        parent.set(COMPONENT + "size", "(2000, 20)")
    _backend, snapshot = observed(root, png)
    assert snapshot.input_geometry == ()


def test_unusable_direct_parent_does_not_fall_back_to_a_valid_grandparent(png):
    root, _frame, parent, proxy = form()
    parent.remove(proxy)
    immediate = ET.SubElement(parent, "section", VISIBLE)
    immediate.append(proxy)
    _backend, snapshot = observed(root, png)
    assert snapshot.a11y_complete
    assert proxy_control(snapshot).focused
    assert snapshot.input_geometry == ()


def test_two_current_focused_inputs_leave_ownership_unresolved(png):
    root, _frame, parent, proxy = form()
    duplicate = ET.SubElement(parent, "entry", dict(proxy.attrib))
    duplicate.set("name", "Another focused proxy")
    duplicate.set(COMPONENT + "screencoord", "(800, 141)")
    _backend, snapshot = observed(root, png)
    assert len([c for c in snapshot.controls if c.editable and c.focused]) == 2
    assert snapshot.input_geometry == ()


def test_incomplete_observation_cannot_contribute_even_an_earlier_valid_hint(png):
    root, frame, _parent, _proxy = form()
    ET.SubElement(frame, "label", {"name": "x" * 2200, **VISIBLE})
    _backend, snapshot = observed(root, png)
    assert not snapshot.a11y_complete
    assert snapshot.input_geometry == ()


def test_a_different_attached_jpeg_invalidates_the_old_context(png):
    root, _frame, _parent, _proxy = form()
    _backend, snapshot = observed(root, png)
    output = io.BytesIO()
    Image.new("RGB", (1920, 1080), "black").save(output, "PNG")
    newer_image = screenshot_image(output.getvalue())
    assert newer_image.region == snapshot.image.region
    assert input_geometry_payload(replace(snapshot, image=newer_image), (1280, 720)) is None


def test_duplicate_observation_reference_cannot_rebind_context_to_an_arbitrary_control(png):
    root, _frame, _parent, _proxy = form()
    _backend, snapshot = observed(root, png)
    proxy = proxy_control(snapshot)
    ambiguous = replace(snapshot, controls=(*snapshot.controls, replace(proxy, label="Other input")))
    assert input_geometry_payload(ambiguous, (1280, 720)) is None


@pytest.mark.parametrize("owner_change", [{"app": "Other application"}, {"window": "Other window"}])
def test_old_geometry_is_not_attached_to_changed_controls_even_when_jpeg_is_unchanged(png, owner_change):
    root, _frame, _parent, _proxy = form()
    _backend, snapshot = observed(root, png)
    proxy = proxy_control(snapshot)
    changed = replace(snapshot, controls=tuple(
        replace(control, **owner_change) if control is proxy else control for control in snapshot.controls
    ))
    assert changed.image is snapshot.image
    assert changed.fingerprint() != snapshot.fingerprint()
    assert input_geometry_payload(changed, (1280, 720)) is None


def test_parent_rectangle_must_contain_the_actual_input_proxy(png):
    root, _frame, parent, _proxy = form()
    parent.set(COMPONENT + "screencoord", "(900, 141)")
    parent.set(COMPONENT + "size", "(400, 20)")
    _backend, snapshot = observed(root, png)
    assert snapshot.input_geometry == ()


def test_a_background_window_does_not_supply_current_input_context(png):
    root, frame, _parent, _proxy = form()
    frame.attrib.pop(STATE + "active")
    _backend, snapshot = observed(root, png)
    assert proxy_control(snapshot).focused
    assert not proxy_control(snapshot).active
    assert snapshot.input_geometry == ()


def planner_request(snapshot, contract, limit):
    requests = []
    action = {"operation": "visual_fill", "label": "Visible input area",
              "box_space": "image_pixels", "box": [386, 94, 1050, 107], "value": "query"}
    response = ({"intent": "act_current", "label": "Search in the visible input", "action": action}
                if limit > 1 else
                {"summary": "Search in the visible input", "done": False,
                 "steps": [{"id": "search", **action}]})

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner = MultimodalPlanner("https://offline.invalid", "fixture", profile="desktop",
                                    desktop_contract=contract, visual_box_space="image_pixels", client=client)
        plan = planner.plan("Search for the requested setting", snapshot, [], max_actions=limit)
    body, = requests
    state = json.loads(body["messages"][1]["content"][0]["text"])
    return plan, body, state


@pytest.mark.parametrize("limit", [1, 6])
def test_v33_both_planning_modes_receive_source_bound_readonly_geometry(png, limit):
    root, _frame, _parent, _proxy = form()
    _backend, snapshot = observed(root, png)
    fingerprint = snapshot.fingerprint()
    plan, body, state = planner_request(snapshot, "property_intent_v33", limit)
    geometry = state["observed_input_ancestor_geometry"]
    assert geometry == input_geometry_payload(snapshot, (1280, 720))
    attachment = base64.b64decode(body["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1])
    assert attachment == snapshot.image.jpeg
    assert hashlib.sha256(attachment).hexdigest() == geometry["attachment_sha256"]
    with Image.open(io.BytesIO(attachment)) as image:
        assert list(image.size) == geometry["attachment_size_px"]
    proxy = next(c for c in state["snapshot"]["controls"] if c["ref"] == proxy_control(snapshot).ref)
    assert proxy["operations"] == []
    assert proxy.get("box_1000", state["snapshot"]["control_defaults"]["box_1000"]) is None
    assert "Its rectangle belongs to that parent" in body["messages"][0]["content"]
    assert plan.steps[0].operation == "visual_fill"
    assert plan.steps[0].box == (302, 131, 820, 149)
    assert snapshot.fingerprint() == fingerprint


def test_older_contract_does_not_receive_the_new_planner_context(png):
    root, _frame, _parent, _proxy = form()
    _backend, snapshot = observed(root, png)
    plan, body, state = planner_request(snapshot, "legacy_v23", 1)
    assert "observed_input_ancestor_geometry" not in state
    assert "input proxy's immediate structural parent" not in body["messages"][0]["content"]
    assert plan.steps[0].operation == "visual_fill"
