"""Large public desktops retain evidence without relaxing completeness or input caps."""

from __future__ import annotations

import hashlib
import io
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import (
    MultimodalPlanner,
    _desktop_snapshot_payload,
    _grounded_navigation_scopes,
)
from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, executable_control
from cua_jev.osworld_backend import OSWorldBackend

FIXTURE = json.loads((Path(__file__).parent / "fixtures/osworld_control_capacity.json").read_text(
    encoding="utf-8",
))
STATE = "{https://accessibility.ubuntu.example.org/ns/state}"
COMPONENT = "{https://accessibility.ubuntu.example.org/ns/component}"
VISIBLE = {STATE + key: "true" for key in ("enabled", "sensitive", "showing", "visible")}


@lru_cache(maxsize=1)
def png():
    output = io.BytesIO()
    Image.new("RGB", tuple(FIXTURE["screen_size"]), "white").save(output, format="PNG")
    return output.getvalue()


def snapshot(root):
    observation = {"screenshot": png(), "accessibility_tree": ET.tostring(root, encoding="unicode")}
    backend = OSWorldBackend(SimpleNamespace(_get_obs=lambda: observation), pause=0)
    try:
        return backend.observe()
    finally:
        backend.close()


def recorded_option_with_control_count(count):
    """Keep the real option last, beyond the former cap; padding has no input capability."""
    root = ET.fromstring(FIXTURE["xml"])
    app = root.find("application")
    assert app is not None
    for index in reversed(range(count - FIXTURE["base_control_count"])):
        app.insert(0, ET.Element("label", {"name": f"n{index}", **VISIBLE}))
    return root


def generic_desktop():
    root = ET.Element("desktop-frame")
    app = ET.SubElement(root, "application", {"name": "Fixture"})
    frame = ET.SubElement(app, "frame", {"name": "Settings", STATE + "active": "true", **VISIBLE})
    return root, frame


def field(parent, value, y):
    return ET.SubElement(parent, "entry", {
        "name": "Value", "value": value, STATE + "editable": "true", **VISIBLE,
        COMPONENT + "screencoord": f"(100, {y})", COMPONENT + "size": "(200, 30)",
    })


def duplicate_fields(count):
    root, frame = generic_desktop()
    field(frame, "wanted", 100)
    for index in range(count - 3):  # The frame and two real fields are retained controls too.
        ET.SubElement(frame, "label", {"name": f"n{index}", **VISIBLE})
    field(frame, "old", 150)
    return root


def test_fixture_is_a_small_recorded_option_excerpt_with_immutable_provenance():
    assert [item["observed_full_control_count"] for item in FIXTURE["sources"]] == [261, 270]
    assert FIXTURE["sources"][0]["sha256"] == (
        "933c2f043546e77e88be75cd3aec8cd823dfd1b6460fe7796a6ef1821013cafe"
    )
    assert hashlib.sha256(FIXTURE["xml"].encode()).hexdigest() == FIXTURE["xml_sha256"]
    observed = snapshot(ET.fromstring(FIXTURE["xml"]))
    assert observed.a11y_complete
    assert len(observed.controls) == FIXTURE["base_control_count"] == 4
    option = observed.controls[-1]
    assert option.label == "Light" and option.role == "cell"
    assert option.selectable and not option.selected and not option.editable
    assert executable_control(option, "click") and not executable_control(option, "fill")


@pytest.mark.parametrize("count", [261, 270, 512])
def test_recorded_option_survives_large_complete_desktop_and_public_payload(count):
    observed = snapshot(recorded_option_with_control_count(count))
    assert observed.a11y_complete
    assert len(observed.controls) == count
    option = observed.controls[-1]
    assert option.label == "Light" and executable_control(option, "click")
    payload = _desktop_snapshot_payload(observed)
    assert payload["a11y_complete"] is True
    assert len(payload["controls"]) == count
    assert payload["controls"][-1]["ref"] == option.ref
    assert payload["controls"][-1]["label"] == "Light"
    assert "selected" in payload["controls"][-1]["condition_kinds"]


@pytest.mark.parametrize("count", [513, 640])
def test_controls_past_512_keep_absence_and_navigation_unknown(count):
    observed = snapshot(recorded_option_with_control_count(count))
    assert len(observed.controls) == 512
    assert not observed.a11y_complete
    assert not any(control.label == "Light" for control in observed.controls)
    assert condition_evidence(AgentCondition("absent", AgentTarget(name="Light", app="gimp")), observed) == {
        "outcome": "unknown", "reason": "incomplete_observation",
    }
    assert _grounded_navigation_scopes(observed) == []
    payload = _desktop_snapshot_payload(observed)
    assert payload["a11y_complete"] is False and len(payload["controls"]) == 512


@pytest.mark.parametrize(("count", "reason"), [(270, "ambiguous_target"), (513, "incomplete_observation")])
def test_capacity_never_proves_uniqueness_by_hiding_a_second_field(count, reason):
    observed = snapshot(duplicate_fields(count))
    fields = [control for control in observed.controls if control.label == "Value"]
    assert len(fields) == (2 if count == 270 else 1)
    assert fields[0].value == "wanted"  # A desired value must not resolve two matching fields.
    ready = AgentCondition("ready_to_fill", AgentTarget(name="Value", app="Fixture", window="Settings"),
                           "wanted")
    assert condition_evidence(ready, observed) == {"outcome": "unknown", "reason": reason}


@pytest.mark.parametrize("choices", [255, 256])
def test_large_complete_tree_keeps_independent_255_navigation_action_limit(choices):
    root, frame = generic_desktop()
    for index in range(choices):
        ET.SubElement(frame, "push-button", {
            "name": f"Choice {index}", **VISIBLE,
            COMPONENT + "screencoord": f"({20 + index % 18 * 100}, {40 + index // 18 * 40})",
            COMPONENT + "size": "(80, 25)",
        })
    observed = snapshot(root)
    assert observed.a11y_complete and len(observed.controls) == choices + 1
    scopes = _grounded_navigation_scopes(observed)
    expected = [{"target": {"app": "Fixture"}, "candidate_count": 255}] if choices == 255 else []
    assert scopes == expected
    step = AgentStep("reach", "click", label="Reach the next settings region", navigation=True,
                     target=AgentTarget(app="Fixture"),
                     postconditions=(AgentCondition("exists", AgentTarget(name="Next region")),))
    backend = OSWorldBackend(SimpleNamespace(), pause=0)
    try:
        offered = backend.compile(step, observed, observed)
        assert len(offered) == (255 if choices == 255 else 0)
    finally:
        backend.close()


@pytest.mark.parametrize("limit", ["name", "body", "nodes", "text", "raw_xml"])
def test_raising_control_capacity_does_not_relax_other_observation_bounds(limit):
    root, frame = generic_desktop()
    if limit == "name":
        ET.SubElement(frame, "label", {"name": "x" * 2001, **VISIBLE})
    elif limit == "body":
        ET.SubElement(frame, "label", {"name": "Long", **VISIBLE}).text = "x" * 2001
    elif limit == "nodes":
        for _ in range(6001):
            ET.SubElement(frame, "panel", VISIBLE)
        ET.SubElement(frame, "label", {"name": "Beyond node budget", **VISIBLE})
    elif limit == "text":
        for index in range(130):
            ET.SubElement(frame, "label", {"name": f"{index}:" + "x" * 800, **VISIBLE})
    else:
        frame.text = " " * 2_000_001
    observed = snapshot(root)
    assert not observed.a11y_complete
    assert len(observed.controls) < 512  # Failure is independent of the retained-control capacity.
    assert len(observed.text) <= 96000
    if limit == "nodes":
        assert not any(control.label == "Beyond node budget" for control in observed.controls)
    assert condition_evidence(AgentCondition("absent", AgentTarget(name="Unseen")), observed) == {
        "outcome": "unknown", "reason": "incomplete_observation",
    }


def test_mock_planner_request_includes_control_512_and_true_completeness_without_second_truncation():
    observed = snapshot(recorded_option_with_control_count(512))
    calls = []

    def answer(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"][0]["text"])["snapshot"]
        calls.append(payload)
        assert payload["a11y_complete"] is True
        assert len(payload["controls"]) == 512
        assert payload["controls"][-1]["label"] == "Light"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "summary": "Choose the currently observed option", "done": False,
            "steps": [{"id": "s1", "operation": "click", "ref": payload["controls"][-1]["ref"]}],
        })}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner("https://offline.invalid/v1", "mock", api_key="offline-test",
                                    profile="desktop", client=client)
        try:
            plan = planner.plan("Select the visible item", observed, [], max_actions=1)
        finally:
            planner.close()
    assert len(calls) == 1
    assert plan.steps[0].ref == observed.controls[-1].ref
