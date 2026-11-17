"""Actual MockTransport/OSWorld memory input checks for conditional recovery wording."""
import copy
import json

import httpx
import pytest
from test_local_read_timeout_recovery_v68 import RecordingPlanner
from test_osworld_backend import png
from test_semantic_tools_v57 import packet
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_runner_v60 import events

from cua_jev import agent_runner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace

GOAL = "Apply the requested literal to the visible input"
TARGET = {"name": "Unavailable input", "role": "textbox"}


def missing():
    return {"summary": "Apply requested text", "id": "one", "done": False, "operation": "fill",
            "target": TARGET, "value": "requested-literal",
            "postconditions": [{"kind": "value_equals", "target": TARGET, "value": "requested-literal"}]}


def visual(**changes):
    return {"summary": "Edit the visible input", "id": "one", "done": False,
            "operation": "visual_fill", "action_label": "Visible input", "box_space": "image_pixels",
            "box": [100, 100, 260, 160], "value": "requested-literal", **changes}


class Environment:
    def __init__(self):
        self.frame = png()
        self.calls = []
        self.xml = ('<desktop><frame name="Panel" active="true">'
                    '<label name="Caption"/></frame></desktop>')

    def _get_obs(self):
        return {"screenshot": self.frame, "accessibility_tree": self.xml}

    def step(self, action, pause=0):
        self.calls.append(action)
        if action["action_type"] == "TYPING":
            self.frame = png("gray")
        return self._get_obs(), 0, False, {}


def run(answer, *, mode="hybrid", env=None, **settings):
    env = env or Environment()
    backend = OSWorldBackend(env, pause=0)
    requests, trace = [], JsonlTrace()

    def respond(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return answer(request, len(requests), env)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = make_planner(client, fmt="semantic_json_v66", cls=RecordingPlanner)
        config = {"allow_actions": True, "mode": mode, "plan_horizon": 1,
                  "max_plans": 2, "max_failures": 3, "max_steps": 1, "timeout_s": 420, **settings}
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
                             config=AgentConfig(**config)).run(GOAL)
    return result, env, backend, planner, trace, requests


@pytest.mark.parametrize("mode", ["vlm", "hybrid"])
def test_actual_zero_match_feedback_names_existing_visual_tool_in_both_modes(mode):
    def answer(_request, call, _env):
        return httpx.Response(200, json=packet(missing() if call == 1 else visual(),
            "declare_semantic_current_action" if call == 1 else "declare_visual_current_action"))

    result, env, backend, planner, trace, requests = run(answer, mode=mode)
    assert result.planner_calls == len(requests) == 2
    assert [item["options"]["max_actions"] for item in planner.invocations] == [1, 1]
    assert [action["action_type"] for action in env.calls] == ["CLICK", "HOTKEY", "TYPING"]
    assert result.steps == 1 and backend.environment_steps == 3 and not result.success
    feedback = events(trace, "agent_recovery")[0]
    assert "matched current controls=0" in feedback["validation"]
    assert agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE in feedback["feedback"]
    assert "Unavailable input" not in feedback["feedback"]
    body = json.loads(requests[1].content)
    history = json.loads(body["messages"][1]["content"][0]["text"])["recent_actions"]
    assert history[0] == planner.invocations[1]["history"][0]
    assert agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE in history[0]["feedback"]
    assert "declare_visual_current_action" in [tool["function"]["name"] for tool in body["tools"]]
    assert body["enable_thinking"] is True and "thinking_budget" not in body
    assert not events(trace, "agent_high_level_context_requested")


def test_without_explicit_visual_declaration_it_still_stops_at_original_failures():
    def answer(_request, _call, _env):
        return httpx.Response(200, json=packet(missing(), "declare_semantic_current_action"))

    result, env, _backend, planner, trace, requests = run(answer, max_plans=20)
    assert result.status == "planner_error" and result.planner_calls == len(requests) == 3
    assert not env.calls and len(events(trace, "agent_recovery")) == 3
    assert all(agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE in row["feedback"]
               for row in events(trace, "agent_recovery"))
    assert planner.calls == 3


@pytest.mark.parametrize("failure", ["out_of_bounds", "image_changed"])
def test_explicit_visual_fallback_still_obeys_original_image_and_box_guards(failure):
    def answer(_request, call, env):
        if call == 1:
            return httpx.Response(200, json=packet(missing(), "declare_semantic_current_action"))
        if failure == "image_changed":
            env.frame = png("black")
        raw = visual(box=[-10, 0, 100, 100]) if failure == "out_of_bounds" else visual()
        return httpx.Response(200, json=packet(raw, "declare_visual_current_action"))

    result, env, backend, _planner, trace, requests = run(answer)
    assert len(requests) == result.planner_calls == 2 and not env.calls and backend.environment_steps == 0
    assert agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE in events(trace, "agent_recovery")[0]["feedback"]
    if failure == "out_of_bounds":
        assert "visual box" in events(trace, "agent_recovery")[1]["validation"]
        second_feedback = events(trace, "agent_recovery")[1]["feedback"]
        assert agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE not in second_feedback


def test_feedback_does_not_renew_episode_deadline(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])

    def answer(_request, _call, _env):
        now[0] = 431.0
        return httpx.Response(200, json=packet(missing(), "declare_semantic_current_action"))

    result, env, _backend, planner, _trace, requests = run(answer)
    assert result.status == "timeout" and len(requests) == result.planner_calls == 1
    assert planner.deadlines == [430.0, None] and not env.calls


@pytest.mark.parametrize("kind", ["transport", "invalid_box", "matched_readonly"])
def test_other_actual_failures_keep_existing_feedback(kind):
    def answer(request, _call, env):
        if kind == "transport":
            raise httpx.ReadTimeout("private detail", request=request)
        if kind == "invalid_box":
            raw, name = visual(box=[-10, 0, 100, 100]), "declare_visual_current_action"
        else:
            raw, name = missing(), "declare_semantic_current_action"
            raw["target"] = {"name": "Unavailable input"}
            raw["postconditions"] = [{"kind": "value_equals", "target": raw["target"],
                                      "value": "requested-literal"}]
        return httpx.Response(200, json=packet(raw, name))

    env = Environment()
    if kind == "matched_readonly":
        env.xml = ('<desktop><frame name="Panel" active="true">'
                   '<label name="Unavailable input"/></frame></desktop>')
    _result, env, _backend, _planner, trace, _requests = run(answer, env=env)
    assert not env.calls
    assert all(agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE not in row["feedback"]
               for row in events(trace, "agent_recovery"))


def test_formal_review_missing_target_does_not_get_new_feedback():
    def answer(_request, call, _env):
        raw = {"label": "Observed panel", "conditions": [{"kind": "text_contains", "value": "Panel"}]}
        return httpx.Response(200, json=packet(raw if call == 1 else missing(),
            "declare_completion" if call == 1 else "declare_semantic_current_action"))

    result, env, _backend, planner, trace, requests = run(answer, whole_goal_recheck=True)
    assert len(requests) == result.planner_calls == 2 and not env.calls
    assert planner.invocations[1]["options"]["whole_goal_recheck"] is True
    assert all(agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE not in row["feedback"]
               for row in events(trace, "agent_recovery"))


def test_exact_local_classifier_excludes_repair_no_image_nonoffered_and_nonlocal_errors():
    def answer(_request, _call, _env):
        return httpx.Response(200, json=packet(missing(), "declare_semantic_current_action"))

    _result, _env, _backend, planner, _trace, _requests = run(answer, max_plans=1)
    error = ValueError("invalid multimodal planning response (ValueError)")
    assert agent_runner._native_missing_visual_feedback(planner, error)
    for options in ({"whole_goal_request": True}, {"repair_context": {"original_offer": True}}):
        assert agent_runner._native_missing_visual_feedback(planner, error, **options) is None
    original = copy.deepcopy(planner.last_request_provenance)
    for field, value in (("image_uploaded", False), ("max_actions", 6), ("parameter_schema_sha256", {})):
        planner.last_request_provenance = {**original, field: value}
        assert agent_runner._native_missing_visual_feedback(planner, error) is None
    planner.last_request_provenance = original
    assert agent_runner._native_missing_visual_feedback(planner, RuntimeError(str(error))) is None
    planner.last_validation_error += " untrusted text"
    assert agent_runner._native_missing_visual_feedback(planner, error) is None


def test_actual_successful_H1_complete_body_remains_original58():
    env = Environment()
    backend = OSWorldBackend(env, pause=0)
    observed = backend.observe()
    captured = []
    def answer(request):
        captured.append(request.content)
        return httpx.Response(200, json=packet(visual(), "declare_visual_current_action"))
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        plans = [make_planner(client, fmt=fmt).plan(GOAL, observed, [], max_actions=1)
                 for fmt in ("semantic_tools_v58", "semantic_json_v66")]
    backend.close()
    assert captured[0] == captured[1] and plans[0] == plans[1]
    assert agent_runner._NATIVE_MISSING_VISUAL_GUIDANCE.encode() not in captured[1]
