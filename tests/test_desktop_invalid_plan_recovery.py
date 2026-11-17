"""Invalid desktop plans recover through one verified action, with unchanged budgets."""

from __future__ import annotations

import io
import json

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace


class Desktop:
    """Typed local GUI fixture: opening exposes two independently checked options."""

    def __init__(self, *, ignore_open=False, opened=False, alpha=False):
        self.opened, self.alpha, self.beta = opened, alpha, False
        self.ignore_open = ignore_open
        self.inputs = []

    def _get_obs(self):
        options = ""
        if self.opened:
            options = (
                f'<check-box name="Alpha" enabled="true" checked="{str(self.alpha).lower()}" '
                'screencoord="(100, 200)" size="(140, 40)"/>'
                f'<check-box name="Beta" enabled="true" checked="{str(self.beta).lower()}" '
                'screencoord="(400, 200)" size="(140, 40)"/>'
            )
        xml = (
            '<desktop><application name="Fixture"><frame name="Settings" active="true">'
            '<button name="Open" enabled="true" screencoord="(100, 40)" size="(140, 40)"/>'
            + options + '</frame></application></desktop>'
        )
        image = io.BytesIO()
        Image.new("RGB", (800, 600), "gray" if self.opened else "white").save(image, "PNG")
        return {"screenshot": image.getvalue(), "accessibility_tree": xml}

    def step(self, action, pause=0):
        assert action["action_type"] == "CLICK"
        x, y = action["parameters"]["x"], action["parameters"]["y"]
        self.inputs.append((x, y))
        if y < 100:
            if not self.ignore_open:
                self.opened = True
        elif x < 300:
            self.alpha = not self.alpha
        else:
            self.beta = not self.beta
        return self._get_obs(), 0, False, {}


def target(name):
    return {"name": name, "app": "Fixture"}


def condition(kind, name):
    return {"kind": kind, "target": target(name)}


def open_step(*, navigation=False):
    return {
        "id": "open", "operation": "click", "label": "Reach the options",
        "navigation": navigation,
        "target": {"app": "Fixture"} if navigation else target("Open"),
        "postconditions": [condition("exists", "Alpha")], "recovery": "observe",
    }


def check_step(name, **extra):
    return {"id": name.lower(), "operation": "check", "target": target(name),
            "postconditions": [condition("checked", name)], **extra}


def plan(*steps, done=False):
    return {"summary": "Enable the two options", "steps": list(steps), "done": done}


def invalid_plan(kind):
    if kind == "navigation_contract":
        return plan(open_step(), check_step("Alpha", depends_on=["open"]))
    if kind == "mixed_visual":
        return plan({"id": "visual", "operation": "visual_click", "label": "Open",
                     "box": [125, 65, 300, 140]}, check_step("Alpha", depends_on=["visual"]))
    assert kind == "model_json"
    return "private-model-output: not JSON"


class RecordedPlanner(MultimodalPlanner):
    def __init__(self, replies, *, profile="desktop"):
        self.replies = replies
        self.limits = []
        self.requests = []
        super().__init__(
            "https://model.example/v1", "test-model", profile=profile,
            client=httpx.Client(transport=httpx.MockTransport(self._response)),
        )

    def _response(self, request):
        self.requests.append(json.loads(request.content))
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        if reply == "timeout":
            raise httpx.ReadTimeout("private-transport-detail", request=request)
        if reply == "http_error":
            return httpx.Response(503, text="private-provider-body")
        if reply == "invalid_envelope":
            return httpx.Response(200, text="private-provider-body: not JSON")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": reply if isinstance(reply, str) else json.dumps(reply)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    def plan(self, goal, snapshot, history, *, max_actions=6):
        self.limits.append(max_actions)
        return super().plan(goal, snapshot, history, max_actions=max_actions)

    def history(self, index):
        return json.loads(self.requests[index]["messages"][1]["content"][0]["text"])["recent_actions"]


@pytest.fixture(autouse=True)
def no_poll_sleep(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _seconds: None)


def run(desktop, planner, *, accept=True, **config):
    backend, trace = OSWorldBackend(desktop, pause=0), JsonlTrace()
    try:
        result = AgentRunner(
            backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, **config),
        ).run("Enable Alpha and Beta", success=(lambda _: desktop.alpha and desktop.beta) if accept else None)
    finally:
        backend.close()
        planner.close()
    return result, trace


@pytest.mark.parametrize("failure", ["navigation_contract", "mixed_visual", "model_json"])
def test_invalid_desktop_plan_recovers_one_verified_action_then_restores_horizon(failure):
    desktop = Desktop()
    planner = RecordedPlanner([
        invalid_plan(failure), plan(open_step()), plan(check_step("Alpha"), check_step("Beta")),
    ])
    result, trace = run(desktop, planner)
    assert result.success and result.planner_calls == planner.calls == 3
    assert planner.limits == [6, 1, 6]
    assert result.steps == len(desktop.inputs) == 3 and result.plan_reuses == 1
    assert result.replan_reasons["invalid_plan"] == 1
    assert result.replan_reasons["recovery_completed"] == 1
    assert result.model_usage == {"prompt_tokens": 30, "completion_tokens": 15}
    assert planner.history(1)[-1]["recovery_required"] is True
    assert planner.history(2)[-1]["verified"] is True
    assert planner.history(2)[-1]["recovery_required"] is False
    plans = [event["payload"] for event in trace.events if event["kind"] == "agent_plan"]
    assert [(item["max_actions"], item["recovery"]) for item in plans] == [(1, True), (6, False)]
    assert "private-model-output" not in str(trace.events)


@pytest.mark.parametrize("failure", ["http_error", "timeout", "invalid_envelope"])
def test_transport_and_outer_protocol_failures_do_not_downgrade_desktop_planning(failure):
    planner = RecordedPlanner([failure, plan(
        open_step(navigation=True), check_step("Alpha", depends_on=["open"]),
        check_step("Beta", depends_on=["open"]),
    )])
    result, trace = run(Desktop(), planner)
    assert result.success and planner.limits == [6, 6]
    assert result.steps == 3 and result.planner_calls == 2
    assert "recovery_required" not in planner.history(1)[-1]
    assert "private-" not in str(trace.events)


def test_transport_failure_during_recovery_does_not_clear_recovery_or_reset_failure_budget():
    planner = RecordedPlanner([
        invalid_plan("navigation_contract"), "http_error", plan(open_step()),
        plan(check_step("Alpha"), check_step("Beta")),
    ])
    result, _trace = run(Desktop(), planner)
    assert result.success and planner.limits == [6, 1, 1, 6]
    assert result.planner_calls == 4 and result.steps == 3


def test_persistent_invalid_plans_keep_original_failure_and_total_call_limits():
    planner = RecordedPlanner([invalid_plan("model_json")])
    desktop = Desktop()
    result, _trace = run(desktop, planner)
    assert result.status == "planner_error" and result.reason == "consecutive planning failures"
    assert planner.limits == [6, 1, 1]
    assert result.planner_calls == planner.calls == 3 and result.steps == 0 and desktop.inputs == []


@pytest.mark.parametrize("options,status,limits", [
    ({"max_plans": 2}, "max_plans", [6, 1]),
    ({"max_failures": 1}, "planner_error", [6]),
])
def test_recovery_does_not_extend_planning_budgets(options, status, limits):
    desktop, planner = Desktop(), RecordedPlanner([invalid_plan("model_json")])
    result, _trace = run(desktop, planner, **options)
    assert result.status == status and planner.limits == limits
    assert result.planner_calls == len(limits) and result.steps == 0 and desktop.inputs == []


def test_recovery_input_consumes_normal_action_budget():
    desktop = Desktop()
    planner = RecordedPlanner([invalid_plan("model_json"), plan(open_step())])
    result, _trace = run(desktop, planner, max_steps=1)
    assert result.status == "max_steps" and planner.limits == [6, 1]
    assert result.steps == len(desktop.inputs) == 1


def test_sent_but_unconfirmed_input_stays_single_action_and_exhausts_action_budget():
    desktop = Desktop(ignore_open=True)
    planner = RecordedPlanner([invalid_plan("mixed_visual"), plan(open_step())])
    result, trace = run(desktop, planner)
    assert result.status == "stuck" and result.reason == "consecutive action failures"
    assert planner.limits == [6, 1, 1, 1]
    assert result.planner_calls == 4 and result.steps == len(desktop.inputs) == 3
    assert result.recovery_observations == 6
    assert all(event["payload"]["verified"] is False
               for event in trace.events if event["kind"] == "agent_step")
    for index in (2, 3):
        feedback = planner.history(index)[-1]
        assert feedback["reason"] == "postcondition_unconfirmed"
        assert feedback["recovery_required"] is True and feedback["verified"] is False


def test_already_satisfied_step_is_not_a_verified_recovery_input():
    desktop = Desktop(opened=True, alpha=True)
    planner = RecordedPlanner([
        invalid_plan("model_json"), plan(check_step("Alpha")), plan(check_step("Beta")), plan(done=True),
    ])
    result, _trace = run(desktop, planner, accept=False)
    assert result.status == "unverified" and planner.limits == [6, 1, 1, 6]
    assert result.condition_skips == 1 and result.steps == len(desktop.inputs) == 1
    assert planner.history(2)[-1]["recovery_required"] is True
    assert planner.history(3)[-1]["verified"] is True


def test_verified_visual_recovery_then_a_new_contract_error_enters_single_action_again():
    desktop = Desktop()
    visual = {"id": "visual", "operation": "visual_click", "label": "Open",
              "button": "right", "box": [125, 65, 300, 140]}
    planner = RecordedPlanner([
        invalid_plan("mixed_visual"), plan(visual), invalid_plan("navigation_contract"),
        plan(check_step("Beta")), plan(check_step("Alpha")),
    ])
    result, trace = run(desktop, planner, max_failures=2)
    assert result.success and planner.limits == [6, 1, 6, 1, 6]
    assert result.planner_calls == planner.calls == 5 and result.steps == len(desktop.inputs) == 3
    assert result.replan_reasons["invalid_plan"] == 2
    assert result.model_usage == {"prompt_tokens": 50, "completion_tokens": 25}
    plans = [event["payload"] for event in trace.events if event["kind"] == "agent_plan"]
    assert [(item["max_actions"], item["recovery"], item["reusable"]) for item in plans] == [
        (1, True, False), (1, True, True), (6, False, True),
    ]
    for index in (1, 3):
        assert planner.history(index)[-1]["recovery_required"] is True
    for index in (2, 4):
        assert planner.history(index)[-1]["verified"] is True
        assert planner.history(index)[-1]["recovery_required"] is False
    assert all(event["payload"]["verified"]
               for event in trace.events if event["kind"] == "agent_step")


@pytest.mark.parametrize("profile,mode,limits", [
    ("browser", "hybrid", [6, 6, 6]),
    ("browser", "vlm", [1, 1, 1]),
    ("desktop", "vlm", [1, 1, 1]),
])
def test_browser_and_vlm_validation_failures_keep_the_original_planning_path(profile, mode, limits):
    desktop = Desktop()
    planner = RecordedPlanner([invalid_plan("model_json")], profile=profile)
    result, _trace = run(desktop, planner, mode=mode)
    assert result.status == "planner_error" and planner.limits == limits
    assert result.steps == 0 and desktop.inputs == []
    assert all("recovery_required" not in item for item in planner.history(-1))


def test_desktop_url_without_an_explicit_planner_profile_does_not_enable_recovery():
    class UnprofiledPlanner:
        def __init__(self):
            self.limits = []

        def plan(self, _goal, _snapshot, _history, *, max_actions):
            self.limits.append(max_actions)
            raise ValueError("private invalid plan")

        def close(self):
            pass

    planner = UnprofiledPlanner()
    result, trace = run(Desktop(), planner)
    assert result.status == "planner_error" and planner.limits == [6, 6, 6]
    assert "private invalid plan" not in str(trace.events)
