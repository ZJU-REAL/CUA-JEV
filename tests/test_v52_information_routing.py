"""Information-only recovery routing does not authorize input or claim completion."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from test_native_first_runner_recovery import Backend, immediate_visual, uploaded
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import (
    AgentConfig,
    AgentRunner,
    DirectPlanPolicy,
    _v52_request_provenance,
    _v52_visual_information_anchor,
)
from cua_jev.agent_types import AgentPlan, AgentStep, AgentTarget
from cua_jev.osworld_backend import screenshot_image
from cua_jev.trace import JsonlTrace


class ImageOnlyEnvironment:
    def __init__(self, snapshot, *, change_next_native=False):
        self.snapshot, self.actions, self.observes = snapshot, [], 0
        self.change_next_native = change_next_native
        self.pending_native_change = False

    def _get_obs(self):
        self.observes += 1
        if self.pending_native_change:
            self.snapshot = replace(self.snapshot, title="A different native state")
            self.pending_native_change = False
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        pixels = io.BytesIO()
        Image.new("RGB", (800, 600), "red" if len(self.actions) % 2 else "blue").save(pixels, "PNG")
        self.snapshot = replace(self.snapshot, image=screenshot_image(pixels.getvalue()))
        # This changes only the NEXT ordinary observation, not the actual
        # returned input frame or its receipt fingerprint.
        self.pending_native_change = self.change_next_native
        return self.snapshot, 0, False, {}


def run_planner(initial, replies, *, contract="property_intent_v52", native=True,
                backend_type=Backend, planner_type=MultimodalPlanner,
                change_next_native=False, on_request=None, **config):
    env = ImageOnlyEnvironment(initial, change_next_native=change_next_native)
    trace, requests = JsonlTrace(), []

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        if on_request:
            on_request(len(requests), env)
        raw = replies[min(len(requests) - 1, len(replies) - 1)]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = planner_type(
            "https://offline.invalid", "fixture", api_key="offline", client=client, profile="desktop",
            desktop_contract=contract, desktop_output_format="json_schema_v50",
            desktop_planning_observation="native_first_v51" if native else "full_multimodal_v1",
        )
        result = AgentRunner(
            backend_type(env, pause=0), planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(**{
                "allow_actions": True, "max_steps": 2, "max_plans": 4, "max_failures": 3,
                "plan_horizon": 6, "plan_ttl_s": 300, "timeout_s": 300, **config,
            }),
        ).run("Complete the current general desktop task")
        actual_calls = planner.calls
    return env, requests, result, trace.events, actual_calls


def payloads(events, kind):
    return [event["payload"] for event in events if event["kind"] == kind]


def test_verified_image_only_progress_retains_image_need_with_fresh_native_match(initial):
    env, requests, result, events, calls = run_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect the screenshot"}, immediate_visual()],
    )
    assert [uploaded(request) for request in requests] == [False, True, True]
    assert calls == result.planner_calls == 3 and len(env.actions) == result.steps == 2
    assert result.success is False and result.plan_reuses == 0
    provenance = payloads(events, "agent_planning_observation")
    assert [item["effective_max_actions"] for item in provenance] == [6, 1, 1]
    assert [item["logical_max_actions"] for item in provenance] == [6, 1, 6]
    assert all(item["native_execution_fingerprint"] == initial.fingerprint() for item in provenance)
    assert provenance[-1]["routing_reason"] == "verified_visual_progress_same_native"
    routing = payloads(events, "agent_visual_information_routing")
    assert [item["status"] for item in routing] == ["armed", "retained", "armed"]
    receipts = payloads(events, "receipt")
    assert routing[0]["source_receipt_id"] == receipts[0]["receipt_id"]
    assert all(item["information_only"] and not item["task_success"] for item in routing)
    assert env.observes >= 6  # No snapshot/image is cached by the information anchor.


def test_anchor_invalidates_when_next_fresh_native_state_changes(initial):
    env, requests, result, events, calls = run_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect"}, immediate_visual(),
                  {"intent": "inspect_visual", "label": "Inspect again"}, immediate_visual()],
        change_next_native=True,
    )
    assert [uploaded(request) for request in requests] == [False, True, False, True]
    assert result.planner_calls == calls == 4 and len(env.actions) == 2
    routing = payloads(events, "agent_visual_information_routing")
    assert routing[0]["status"] == "armed" and routing[1]["status"] == "invalidated"
    assert routing[1]["native_execution_fingerprint"] != routing[1]["fingerprint"]


def test_v51_preserves_original_inspect_recovery_round_trips_and_trace_shape(initial):
    env, requests, result, events, calls = run_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect"}, immediate_visual(),
                  {"intent": "inspect_visual", "label": "Inspect again"}, immediate_visual()],
        contract="property_intent_v51",
    )
    assert [uploaded(request) for request in requests] == [False, True, False, True]
    assert result.planner_calls == calls == 4 and len(env.actions) == 2
    assert not payloads(events, "agent_visual_information_routing")
    assert all("native_execution_fingerprint" not in item
               for item in payloads(events, "agent_planning_observation"))


def test_full_image_baseline_cannot_activate_native_information_routing(initial):
    raw = {"intent": "act_current", "label": "Help", "action": {"operation": "click", "ref": "help"}}
    env, requests, result, events, calls = run_planner(initial, [raw], native=False, max_steps=1)
    assert len(env.actions) == calls == result.planner_calls == 1 and uploaded(requests[0])
    assert not payloads(events, "agent_visual_information_routing")
    provenance = payloads(events, "agent_planning_observation")
    assert provenance[0]["native_execution_fingerprint"] == initial.fingerprint()
    assert provenance[0]["routing_reason"] == "full_multimodal_root"


def test_vlm_h1_baseline_keeps_full_image_and_no_information_anchor(initial):
    env, requests, result, events, calls = run_planner(
        initial, [immediate_visual()], mode="vlm", max_steps=1,
    )
    assert result.planner_calls == calls == len(requests) == len(env.actions) == 1
    assert uploaded(requests[0]) and not payloads(events, "agent_visual_information_routing")
    provenance = payloads(events, "agent_planning_observation")
    assert provenance[0]["effective_max_actions"] == 1
    assert provenance[0]["routing_reason"] == "vlm_baseline"


class NoNativeCandidates(Backend):
    def compile(self, step, planned, current):
        return super().compile(step, planned, current) if step.operation.startswith("visual_") else ()


@pytest.mark.parametrize("contract,expected", [
    ("property_intent_v52", [False, True]), ("property_intent_v51", [False, False]),
])
def test_atomic_zero_candidates_changes_only_v52_next_information_route(initial, contract, expected):
    raw = {"intent": "act_current", "label": "Help", "action": {"operation": "click", "ref": "help"}}
    env, requests, result, events, calls = run_planner(
        initial, [raw, immediate_visual()], contract=contract, backend_type=NoNativeCandidates,
        max_steps=1, max_plans=2,
    )
    assert [uploaded(request) for request in requests] == expected
    assert result.planner_calls == calls == 2
    assert len(env.actions) == (1 if contract.endswith("52") else 0)
    assert len(payloads(events, "agent_information_recovery")) == (contract.endswith("52"))
    assert result.replan_reasons["target_unavailable"] == 1


def test_reusable_unavailable_stage_keeps_original_two_waits_and_failure_limit(initial, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)

    class Planner:
        profile = "desktop"
        desktop_contract = "property_intent_v52"
        desktop_planning_observation = "native_first_v51"

        def plan(self, _goal, snapshot, _history, *, max_actions):
            self.last_request_provenance = {
                "image_uploaded": False, "planning_observation": "native_first_v51",
                "max_actions": max_actions, "a11y_complete": snapshot.a11y_complete,
            }
            return AgentPlan("Unavailable semantic stage", (
                AgentStep("unavailable", "click", target=AgentTarget(name="Missing"), recovery="observe"),
            ), reusable=True)

    env, trace = ImageOnlyEnvironment(initial), JsonlTrace()
    result = AgentRunner(
        NoNativeCandidates(env), Planner(), policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, max_failures=1),
    ).run("Complete the declared stage")
    recovery = payloads(trace.events, "agent_recovery")
    assert [item["reason"] for item in recovery] == [
        "awaiting_conditions", "awaiting_conditions", "no_executable_target",
    ]
    assert result.recovery_observations == 2 and result.status == "stuck"
    assert result.planner_calls == 1 and env.actions == []
    assert not payloads(trace.events, "agent_information_recovery")


def test_information_anchor_does_not_extend_max_plans(initial):
    env, requests, result, events, calls = run_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect"}, immediate_visual()], max_plans=2,
    )
    assert result.status == "max_plans" and calls == result.planner_calls == len(requests) == 2
    assert len(env.actions) == 1
    assert payloads(events, "agent_visual_information_routing")[0]["status"] == "armed"


def test_v52_timeout_request_still_records_actual_input_native_fingerprint(initial):
    def timeout(_index, _env):
        raise httpx.ReadTimeout("Offline fixture timeout")

    env, requests, result, events, calls = run_planner(
        initial, [immediate_visual()], on_request=timeout, max_plans=1,
    )
    assert result.planner_calls == calls == len(requests) == 1 and env.actions == []
    provenance = payloads(events, "agent_planning_observation")
    assert len(provenance) == 1 and provenance[0]["native_execution_fingerprint"] == initial.fingerprint()
    assert provenance[0]["image_uploaded"] is False and provenance[0]["routing_reason"] == "native_root"


def test_original_planner_history_downgrade_is_recorded_as_actual_h1(initial):
    class HistoryRecoveryPlanner(MultimodalPlanner):
        def plan(self, goal, snapshot, history, *, max_actions=6):
            return super().plan(goal, snapshot, [*history, {"recovery_required": True}],
                                max_actions=max_actions)

    env, requests, result, events, calls = run_planner(
        initial, [immediate_visual()], planner_type=HistoryRecoveryPlanner, max_steps=1,
    )
    assert result.planner_calls == calls == len(requests) == len(env.actions) == 1
    provenance = payloads(events, "agent_planning_observation")
    assert len(provenance) == 1 and provenance[0]["image_uploaded"] is True
    assert provenance[0]["logical_max_actions"] == provenance[0]["requested_max_actions"] == 6
    assert provenance[0]["effective_max_actions"] == 1
    assert provenance[0]["routing_reason"] == "planner_history_single_action_recovery"
    assert provenance[0]["native_execution_fingerprint"] == initial.fingerprint()


def test_information_anchor_does_not_extend_episode_clock(initial, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])

    def elapsed(index, _env):
        if index == 3:
            now[0] += 301

    env, requests, result, _events, calls = run_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect"}, immediate_visual()], on_request=elapsed,
    )
    assert result.status == "timeout" and calls == result.planner_calls == len(requests) == 3
    assert len(env.actions) == 1 and not result.success


def anchor_inputs(initial):
    receipt = SimpleNamespace(success=True, receipt_id="actual-receipt", output={
        "environment_steps": 1, "effects": ["image"],
        "before_fingerprint": initial.fingerprint(), "after_fingerprint": initial.fingerprint(),
    })
    context = {"planning_observation": "native_first_v51", "image_uploaded": True,
               "a11y_complete": True, "max_actions": 1}
    return receipt, context


@pytest.mark.parametrize("change", [
    "failed", "unverified", "zero_input", "bool_input", "missing_input", "other_effect",
    "native_changed", "original_mismatch", "invalid_hash", "nonvisual", "partial", "no_image", "network",
    "not_recovery", "not_h1", "not_uploaded", "partial_request", "no_request", "no_receipt_id",
])
def test_anchor_requires_real_verified_full_h1_same_native_visual_input(initial, change):
    receipt, context = anchor_inputs(initial)
    snapshot, step = initial, AgentStep("visual", "visual_click")
    verified, recovery, network = True, True, False
    if change == "failed":
        receipt.success = False
    elif change == "unverified":
        verified = False
    elif change in {"zero_input", "bool_input", "missing_input"}:
        receipt.output["environment_steps"] = {
            "zero_input": 0, "bool_input": True, "missing_input": None,
        }[change]
    elif change == "other_effect":
        receipt.output["effects"] = ["image", "postconditions"]
    elif change == "native_changed":
        receipt.output["after_fingerprint"] = "a" * 64
    elif change == "original_mismatch":
        snapshot = replace(initial, title="Different original decision state")
    elif change == "invalid_hash":
        receipt.output.update(before_fingerprint="not-a-fingerprint", after_fingerprint="not-a-fingerprint")
    elif change == "nonvisual":
        step = replace(step, operation="click")
    elif change == "partial":
        snapshot = replace(initial, a11y_complete=False)
    elif change == "no_image":
        snapshot = replace(initial, image=None)
    elif change == "network":
        network = True
    elif change == "not_recovery":
        recovery = False
    elif change == "not_h1":
        context["max_actions"] = 6
    elif change == "not_uploaded":
        context["image_uploaded"] = False
    elif change == "partial_request":
        context["a11y_complete"] = False
    elif change == "no_request":
        context = None
    elif change == "no_receipt_id":
        receipt.receipt_id = ""
    assert _v52_visual_information_anchor(
        snapshot, step, receipt, verified=verified, request_information=context,
        recovery=recovery, network_recovery=network,
    ) is None


@pytest.mark.parametrize("operation", ["visual_click", "visual_fill"])
def test_anchor_is_only_receipt_identity_and_exact_native_hash(initial, operation):
    receipt, context = anchor_inputs(initial)
    anchor = _v52_visual_information_anchor(
        initial, AgentStep("visual", operation), receipt, verified=True,
        request_information=context, recovery=True, network_recovery=False,
    )
    assert anchor == {"fingerprint": initial.fingerprint(), "source_receipt_id": receipt.receipt_id}


def test_request_provenance_is_typed_copied_and_matches_actual_limit():
    planner = SimpleNamespace(last_request_provenance={
        "planning_observation": "native_first_v51", "image_uploaded": False,
        "a11y_complete": True, "max_actions": 6,
    })
    copied = _v52_request_provenance(planner, 6)
    planner.last_request_provenance["image_uploaded"] = True
    assert copied["image_uploaded"] is False
    assert _v52_request_provenance(planner, 1) is None
    planner.last_request_provenance["max_actions"] = 1
    assert _v52_request_provenance(planner, 6)["max_actions"] == 1
    planner.last_request_provenance["max_actions"] = True
    assert _v52_request_provenance(planner, 1) is None
    planner.last_request_provenance = None
    assert _v52_request_provenance(planner, 6) is None
