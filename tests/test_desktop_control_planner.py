"""Optional control packing must save actual request bytes and preserve real grounding."""

import base64
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _desktop_snapshot_payload
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.desktop_control_encoding import (
    CONTROL_ENCODING_INSTRUCTIONS,
    compact_snapshot_controls,
    decode_snapshot_controls,
)
from cua_jev.desktop_snapshot_encoding import compact_snapshot_ancestors
from cua_jev.osworld_backend import screenshot_image


def observed(count=300):
    return AgentSnapshot("osworld://desktop", "Preferences", "Public current text", tuple(
        AgentControl(f"d{index}", f"Observed label {index}", "text", role="label", app="Editor",
                     window="Preferences", active=True,
                     ancestors=("frame: Preferences", "panel: Current section", "section: Detail"))
        for index in range(count)
    ))


def request(snapshot, *, limit=6, pooling=True, compact=None, planner_class=MultimodalPlanner, reply=None):
    requests = []
    if reply is None:
        reply = ({"intent": "complete", "label": "Inspect current state"} if limit > 1 else
                 {"summary": "Inspect current state", "steps": [], "done": True})

    def respond(incoming):
        requests.append(incoming)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        settings = {} if compact is None else {"compact_desktop_controls": compact}
        planner = planner_class("https://offline.invalid", "mock", profile="desktop",
                                desktop_contract="property_intent_v33", pool_desktop_ancestors=pooling,
                                enable_thinking=True, client=client, **settings)
        plan = planner.plan("Inspect this current interface", snapshot, [], max_actions=limit)
    return plan, requests[0]


@pytest.mark.parametrize("limit", [1, 6])
@pytest.mark.parametrize("pooling", [False, True])
def test_both_modes_decode_exactly_and_save_over4096_in_the_actual_request(limit, pooling):
    snapshot = observed()
    original = snapshot.to_dict()
    fingerprint, scope = snapshot.fingerprint(), snapshot.scope_key()
    _plain, uncompressed = request(snapshot, limit=limit, pooling=pooling, compact=False)
    _packed, packed = request(snapshot, limit=limit, pooling=pooling, compact=True)
    assert len(uncompressed.content) - len(packed.content) > 4096
    body = json.loads(packed.content)
    state = json.loads(body["messages"][1]["content"][0]["text"])
    encoded = state["snapshot"]
    assert "control_encoding" in encoded
    expected = _desktop_snapshot_payload(snapshot, explicit_box_spaces=True)
    if pooling:
        expected = compact_snapshot_ancestors(expected)
    assert decode_snapshot_controls(encoded) == json.loads(json.dumps(expected))
    assert CONTROL_ENCODING_INSTRUCTIONS in body["messages"][0]["content"]
    assert "missing boolean flags mean false" not in body["messages"][0]["content"]
    assert snapshot.to_dict() == original and snapshot.fingerprint() == fingerprint
    assert snapshot.scope_key() == scope


@pytest.mark.parametrize("limit", [1, 6])
@pytest.mark.parametrize("pooling", [False, True])
def test_small_state_does_not_pay_for_tables_or_instruction_overhead(limit, pooling):
    snapshot = observed(10)
    _plain, ordinary = request(snapshot, limit=limit, pooling=pooling, compact=False)
    _packed, fallback = request(snapshot, limit=limit, pooling=pooling, compact=True)
    assert ordinary.content == fallback.content
    assert "control_encoding" not in json.loads(
        json.loads(fallback.content)["messages"][1]["content"][0]["text"],
    )["snapshot"]


@pytest.mark.parametrize("limit", [1, 6])
def test_default_and_explicit_off_generate_the_same_request(limit):
    snapshot = observed()
    _plan, omitted = request(snapshot, limit=limit)
    _plan, off = request(snapshot, limit=limit, compact=False)
    assert omitted.content == off.content


def test_packed_profiles_never_make_a_readonly_ref_fillable():
    snapshot = observed()
    assert "control_encoding" in compact_snapshot_controls(_desktop_snapshot_payload(snapshot))
    reply = {"summary": "Try an unavailable field", "done": False,
             "steps": [{"id": "s1", "operation": "fill", "ref": "d0", "value": "literal text"}]}
    with pytest.raises(ValueError, match="invalid multimodal planning response"):
        request(snapshot, limit=1, compact=True, reply=reply)


def test_packed_request_still_binds_a_real_current_executable_ref():
    source = observed()
    button = AgentControl("apply", "Apply", "button", role="button", app="Editor", window="Preferences",
                          box=(100, 100, 150, 150), active=True)
    snapshot = replace(source, controls=(*source.controls, button))
    reply = {"summary": "Choose current Apply", "done": False,
             "steps": [{"id": "s1", "operation": "click", "ref": "apply"}]}
    plan, packed = request(snapshot, limit=1, compact=True, reply=reply)
    assert plan.steps[0].ref == "apply" and snapshot.controls[-1] is button
    state = json.loads(json.loads(packed.content)["messages"][1]["content"][0]["text"])
    assert "control_encoding" in state["snapshot"]


def test_actual_jpeg_attachment_and_geometry_remain_identical_after_compaction():
    data = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(data, "PNG")
    image = screenshot_image(data.getvalue())
    snapshot = replace(observed(), image=image)
    _plan, plain = request(snapshot, compact=False)
    _plan, packed = request(snapshot, compact=True)
    before = json.loads(plain.content)["messages"][1]["content"]
    after = json.loads(packed.content)["messages"][1]["content"]
    assert before[1] == after[1]
    assert base64.b64decode(after[1]["image_url"]["url"].split(",", 1)[1]) == image.jpeg
    assert json.loads(before[0]["text"])["screenshot_geometry"] == json.loads(after[0]["text"])[
        "screenshot_geometry"
    ]
    assert len(plain.content) - len(packed.content) > 4096
    assert snapshot.image is image


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_compaction_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="compact_desktop_controls"):
        MultimodalPlanner("https://offline.invalid", "mock", profile="desktop",
                          compact_desktop_controls=value)


def test_compaction_cannot_accidentally_change_the_browser_contract():
    with pytest.raises(ValueError, match="desktop profile"):
        MultimodalPlanner("https://offline.invalid", "mock", compact_desktop_controls=True)
