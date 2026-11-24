"""Use the existing native family without dropping any declared effect guard."""

import hashlib
import json
import re
import socket
from dataclasses import replace

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_pending_numeric_edit import NumericEnvironment
from test_property_preparation import initial as initial
from test_semantic_tools_planner import packet
from test_semantic_tools_v56_guidance import OPTIONS, _body

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentControl
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_semantic_tools_v58 import encode_semantic_declaration as encode58
from cua_jev.desktop_semantic_tools_v64 import encode_semantic_declaration as encode64
from cua_jev.osworld_backend import OSWorldBackend

# Actual passed88 frozen-source Mock request, recorded before this candidate:
# runs/v88-fixed8-runtime-draft/local-dry-02.json, SHA 63e5ccba...79d5595c.
ORIGINAL_H6_BODY = "acd6d88ce82f854fae03ebfecc5e4dac5e6ecfe30d1c777ea8dc7cbc11694729"
ORIGINAL_H1_BODY = "c910011b1fbc7bc9821fff5dc06ae426cf6ff9e53f7dd9b62e62bec53b31e990"
ORIGINAL_H1_SCHEMAS = {
    "declare_native_current_action": "16cac31050e1b83bd5a00a178480623c4cf63751db713d883cd13ba41402ea78",
    "declare_semantic_current_action": "3d102546ca9aa1c1a1aa95f87984764eee429c8fe9be5dfbd0e81856c7b180e8",
    "declare_visual_current_action": "9164ae6810efab7e6fb8f20c35c497ba9664bb859097ae16bbca30a59c717fae",
    "declare_key_page_action": "8d24990127ba5d6413688ffd9b8f3438780ad53f2b54e798ca13a9251dd087c0",
    "declare_completion": "a191c8271d4cfa3fece728c5f0570c3efb73c73e9880f64e3d7a399bdc434088",
}
GOAL = "Set the minimum number of undo steps to 100."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("Only local MockTransport is permitted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def model(client, output_format="semantic_tools_v64"):
    return MultimodalPlanner(
        "http://10.130.138.46:8010/v1", "dashscope/qwen3.7-plus", api_key=None,
        client=client, allow_insecure_http=True, profile="desktop",
        desktop_contract="property_intent_v52", desktop_output_format=output_format,
        desktop_planning_observation="full_multimodal_v1", visual_box_space="image_pixels",
        pool_desktop_ancestors=True, enable_thinking=True, thinking_budget=None, request_timeout_s=90,
    )


def capture(snapshot, reply, *, horizon=1, output_format="semantic_tools_v64", formal=False):
    requests = []

    def answer(request):
        body = json.loads(request.content)
        requests.append(request)
        return httpx.Response(200, json=reply(body) if callable(reply) else reply)

    with httpx.Client(transport=httpx.MockTransport(answer), trust_env=False) as client:
        planner = model(client, output_format)
        plan = planner.plan(GOAL, snapshot, [], max_actions=horizon, whole_goal_recheck=formal,
                            whole_goal_inband_epoch=None if formal else 0)
    assert planner.calls == len(requests) == 1
    return plan, requests[0]


def test_normal_H6_request_body_exactly_matches_actual_passed88_before_this_change():
    environment = NumericEnvironment()
    backend = OSWorldBackend(environment, pause=0.5)
    try:
        snapshot = backend.observe()
        reply = packet({
            "label": "Requested numeric result", "scope": {"app": "gimp"},
            "edit_operation": "set", "edit_value": "100", "numeric_finalize_key": "enter",
            "confirmation_label": "Confirm requested result", "completion": "goal",
        }, name="declare_property_stage")
        plan, request = capture(snapshot, reply, horizon=6)
        assert hashlib.sha256(request.content).hexdigest() == ORIGINAL_H6_BODY
        assert plan.reusable and plan.steps[0].property_edit.numeric_finalize_key == "enter"
        assert not environment.actions and backend.environment_steps == 0
    finally:
        backend.close()


def test_H1_offers_all_five_exact_original_parameter_schemas_and_original_budget():
    environment = NumericEnvironment()
    backend = OSWorldBackend(environment, pause=0.5)
    try:
        snapshot = backend.observe()
        field = next(control for control in snapshot.controls if control.role == "spinbutton")
        plan, request = capture(snapshot, packet({
            "summary": "Set current field", "id": "current", "done": False,
            "operation": "fill", "ref": field.ref, "value": "100",
        }, name="declare_native_current_action"))
        body = json.loads(request.content)
        assert {tool["function"]["name"]: _digest(tool["function"]["parameters"])
                for tool in body["tools"]} == ORIGINAL_H1_SCHEMAS
        assert body["enable_thinking"] is True and "thinking_budget" not in body
        assert request.extensions["timeout"]["read"] == 90
        assert any(block["type"] == "image_url" for block in body["messages"][1]["content"])
        assert plan.steps[0].ref == field.ref and not plan.steps[0].postconditions
        assert not environment.actions and backend.environment_steps == 0
    finally:
        backend.close()


class Environment:
    def __init__(self, snapshot):
        self.snapshot, self.actions = snapshot, []

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        self.snapshot = replace(self.snapshot, controls=(*self.snapshot.controls, AgentControl(
            "information", "Actual Information", "button", role="button", app="Editor", window="Main",
            active=True, box=(400, 300, 600, 400),
        )))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def test_actual_native_example_maps_and_executes_one_current_input_without_future_predicates(initial):
    environment = Environment(initial)
    backend = Backend(environment, pause=0)
    observed_example = []

    def answer(body):
        match = re.search(r"declare_native_current_action arguments (\{[^{}]+\})\. ",
                          body["messages"][0]["content"])
        assert match is not None
        example = json.loads(match[1])
        function = next(tool["function"] for tool in body["tools"]
                        if tool["function"]["name"] == "declare_native_current_action")
        Draft202012Validator(function["parameters"]).validate(example)
        assert example["ref"] == "CURRENT_REF" and not {"target", "postconditions"} & example.keys()
        observed_example.append(dict(example))
        example["ref"] = "help"  # Bind syntax-only placeholder to an actual observed ref.
        return packet(example, name="declare_native_current_action")

    try:
        before = backend.observe()
        plan, _ = capture(before, answer)
        assert len(observed_example) == 1 and not plan.steps[0].postconditions
        candidate, = backend.compile(plan.steps[0], before, before)
        receipt = backend.execute(candidate, "observation", "decision")
        assert receipt.success and backend.verify(candidate, receipt).passed
        assert receipt.output["environment_steps"] == len(environment.actions) == 1
    finally:
        backend.close()


def test_explicit_unknown_semantic_effect_still_fails_after_the_original_positive_input(initial):
    environment = Environment(initial)
    backend = Backend(environment, pause=0)
    condition = {"kind": "exists", "target": {
        "name": "Actual Information", "role": "menuitem", "app": "Editor", "window": "Main",
    }}
    try:
        before = backend.observe()
        plan, _ = capture(before, packet({
            "summary": "Explicitly guarded menu action", "id": "current", "done": False,
            "operation": "click", "target": {"name": "Help", "role": "button", "app": "Editor"},
            "postconditions": [condition],
        }, name="declare_semantic_current_action"))
        candidate, = backend.compile(plan.steps[0], before, before)
        receipt = backend.execute(candidate, "observation", "decision")
        assert receipt.success and receipt.output["environment_steps"] == len(environment.actions) == 1
        assert not backend.verify(candidate, receipt).passed
        assert receipt.output["postcondition_evidence"][0]["outcome"] == "unknown"
        assert receipt.output["postcondition_evidence"][0]["target"]["role"] == "menuitem"
        assert plan.steps[0].postconditions[0].target.role == "menuitem"
    finally:
        backend.close()


def test_wrong_owner_semantic_action_still_rejects_before_any_input(initial):
    environment = Environment(initial)
    backend = Backend(environment, pause=0)
    try:
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            capture(backend.observe(), packet({
                "summary": "Wrong owner", "id": "current", "done": False, "operation": "click",
                "target": {"name": "Help", "role": "button", "app": "Other"},
                "postconditions": [{"kind": "exists", "target": {"name": "Future"}}],
            }, name="declare_semantic_current_action"))
        assert not environment.actions and backend.environment_steps == 0
    finally:
        backend.close()


@pytest.mark.parametrize("output_format", ["semantic_tools_v55", "semantic_tools_v58", "semantic_json_v66"])
def test_legacy_H1_callers_keep_original_guidance_and_recorded58_body(output_format):
    environment = NumericEnvironment()
    backend = OSWorldBackend(environment, pause=0.5)
    try:
        snapshot = backend.observe()
        field = next(control for control in snapshot.controls if control.role == "spinbutton")
        _, request = capture(snapshot, packet({
            "summary": "Current field", "id": "current", "done": False,
            "operation": "fill", "ref": field.ref, "value": "100",
        }, name="declare_native_current_action"), output_format=output_format)
        system = json.loads(request.content)["messages"][0]["content"]
        assert "Prefer this family when the intended current action" not in system
        assert "Activate the observed current control" not in system
        assert '"target":{"name":"New","role":"push-button","app":"Editor"}' in system
        if output_format == "semantic_tools_v58":
            assert hashlib.sha256(request.content).hexdigest() == ORIGINAL_H1_BODY
        assert not environment.actions and backend.environment_steps == 0
    finally:
        backend.close()


def test_v64_default_codec_H1_and_actual_formal_review_keep_original58_body(initial):
    body = _body(initial, 1)
    original, _ = encode58(body, max_actions=1, **OPTIONS)
    default, _ = encode64(body, max_actions=1, **OPTIONS)
    assert default == original
    reply = packet({"summary": "Current control", "id": "only", "done": False,
                    "operation": "click", "ref": "help"}, name="declare_native_current_action")
    _, actual = capture(initial, reply, formal=True)
    _, ancestor = capture(initial, reply, output_format="semantic_tools_v58", formal=True)
    assert actual.content == ancestor.content
    system = json.loads(actual.content)["messages"][0]["content"]
    assert "Activate the observed current control" not in system
