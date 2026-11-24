"""Every native-to-image transition is an explicit, counted, zero-input model request."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner, VisualInspectionRequested
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace


def immediate_visual():
    return {"summary": "Click the current screenshot target", "done": False,
            "steps": [{"id": "visual", "operation": "visual_click", "label": "Visible control",
                       "box": [100, 100, 300, 200]}]}


class Environment:
    def __init__(self, snapshot):
        self.snapshot, self.actions, self.observes = snapshot, [], 0

    def _get_obs(self):
        self.observes += 1
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        self.snapshot = replace(self.snapshot, title="Changed", text="The requested control was clicked")
        return self._get_obs(), 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def run_real_planner(snapshot, replies, *, native=True, max_plans=4, on_request=None, **config):
    env, trace, requests = Environment(snapshot), JsonlTrace(), []

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        if on_request:
            on_request(len(requests), body, env)
        raw = replies[min(len(requests) - 1, len(replies) - 1)]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner(
            "https://offline.invalid", "fixture", api_key="offline", client=client, profile="desktop",
            desktop_contract="property_intent_v51", desktop_output_format="json_schema_v50",
            desktop_planning_observation="native_first_v51" if native else "full_multimodal_v1",
        )
        result = AgentRunner(
            Backend(env, pause=0), planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(**{
                "allow_actions": True, "max_steps": 1, "max_plans": max_plans,
                "max_failures": 3, "plan_horizon": 6, "plan_ttl_s": 300, "timeout_s": 300, **config,
            }),
        ).run("Complete a general screenshot task")
        actual_calls = planner.calls
    return env, requests, result, trace.events, actual_calls


def uploaded(body):
    return any(item["type"] == "image_url" for item in body["messages"][1]["content"])


def test_inspect_visual_is_zero_input_then_real_counted_full_image_recovery(initial):
    input_counts = []
    env, requests, result, events, actual_calls = run_real_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect the screenshot"}, immediate_visual()],
        on_request=lambda _index, _body, env: input_counts.append(len(env.actions)),
    )
    assert [uploaded(body) for body in requests] == [False, True]
    assert result.planner_calls == actual_calls == len(requests) == 2
    assert len(env.actions) == result.steps == 1
    assert input_counts == [0, 0]
    inspect = [e["payload"] for e in events if e["kind"] == "agent_visual_context_requested"]
    assert len(inspect) == 1 and inspect[0]["input_sent"] is False
    assert inspect[0]["planner_calls"] == 1 and inspect[0]["next_max_actions"] == 1
    plans = [e["payload"] for e in events if e["kind"] == "agent_plan"]
    assert len(plans) == 1 and plans[0]["max_actions"] == 1 and plans[0]["recovery"] is True
    provenance = [e["payload"] for e in events if e["kind"] == "agent_planning_observation"]
    assert [p["image_uploaded"] for p in provenance] == [False, True]
    assert [p["effective_max_actions"] for p in provenance] == [6, 1]
    assert result.plan_reuses == 0 and result.post_snapshot_reuses == 0


@pytest.mark.parametrize("hidden_prepare", [False, True])
def test_no_image_visual_candidate_is_rejected_then_original_image_path_can_execute(initial, hidden_prepare):
    visual = {"operation": "visual_click", "label": "Never seen target", "box": [100, 100, 300, 200]}
    raw = ({"intent": "pursue_property", "label": "Set requested value", "scope": {"app": "Editor"},
            "edit": {"operation": "set", "value": "42"}, "numeric_finalize": {"key": "none"},
            "prepare": visual}
           if hidden_prepare else {
               "intent": "act_current", "label": "Current visual target", "action": visual})
    env, requests, result, events, actual_calls = run_real_planner(initial, [raw, immediate_visual()])
    assert [uploaded(body) for body in requests] == [False, True]
    assert len(env.actions) == 1 and result.planner_calls == actual_calls == 2
    failures = [e["payload"] for e in events if e["kind"] == "agent_recovery"]
    assert any(f.get("reason") == "invalid_plan" for f in failures)
    assert len([e for e in events if e["kind"] == "receipt"]) == 1


def test_incomplete_ax_directly_selects_one_full_image_call_without_phantom_root_call(initial):
    env, requests, result, events, actual_calls = run_real_planner(
        replace(initial, a11y_complete=False), [immediate_visual()],
    )
    assert result.planner_calls == actual_calls == len(requests) == 1 and uploaded(requests[0])
    plan = next(e["payload"] for e in events if e["kind"] == "agent_plan")
    assert plan["max_actions"] == 1
    provenance = next(e["payload"] for e in events if e["kind"] == "agent_planning_observation")
    assert provenance["logical_max_actions"] == 6 and provenance["effective_max_actions"] == 1
    assert provenance["image_uploaded"] is True and provenance["a11y_complete"] is False
    assert len(env.actions) == 1


def test_full_image_baseline_still_sends_image_in_root_request(initial):
    raw = {"intent": "act_current", "label": "Click current Help",
           "action": {"operation": "click", "ref": "help"}}
    env, requests, result, _events, actual_calls = run_real_planner(initial, [raw], native=False)
    assert len(requests) == actual_calls == result.planner_calls == 1 and uploaded(requests[0])
    assert len(env.actions) == 1


def test_inspect_visual_does_not_extend_original_max_plans_or_create_an_action(initial):
    env, requests, result, events, actual_calls = run_real_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect the screenshot"}], max_plans=1,
    )
    assert result.status == "max_plans" and result.planner_calls == actual_calls == len(requests) == 1
    assert env.actions == [] and result.steps == result.plan_reuses == 0
    assert not any(e["kind"] == "agent_plan" for e in events)


def test_inspect_visual_does_not_extend_original_episode_timeout(initial, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])

    def elapsed(_index, _body, _env):
        now[0] += 301

    env, requests, result, _events, actual_calls = run_real_planner(
        initial, [{"intent": "inspect_visual", "label": "Inspect the screenshot"}], on_request=elapsed,
    )
    assert result.status == "timeout" and result.planner_calls == actual_calls == len(requests) == 1
    assert env.actions == []


def test_typed_visual_request_preserves_existing_failure_streak(initial):
    class Planner:
        profile = "desktop"

        def __init__(self):
            self.calls = 0

        def plan(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise VisualInspectionRequested()
            raise ValueError("Invalid bounded output")

    env, planner = Environment(initial), Planner()
    result = AgentRunner(
        Backend(env), planner, policy=DirectPlanPolicy(),
        config=AgentConfig(allow_actions=True, max_failures=2, max_plans=4),
    ).run("Inspect the requested UI")
    assert result.status == "planner_error" and result.planner_calls == planner.calls == 3
    assert env.actions == []


def test_bad_pure_effective_limit_cannot_increment_model_calls(initial):
    class Planner:
        def effective_max_actions(self, _snapshot, _limit):
            return 0

        def plan(self, *_args, **_kwargs):
            raise AssertionError("No real request is permitted after bad pure hook")

    env = Environment(initial)
    result = AgentRunner(Backend(env), Planner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Inspect UI")
    assert result.status == "error" and result.planner_calls == 0 and env.actions == []
