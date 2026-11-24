"""Whole-goal review must revisit omitted work without weakening execution checks."""

import json
from dataclasses import replace

import httpx
import pytest
from test_agent_runner import Backend
from test_desktop_output_schema_contract import observed as _observed_fixture
from test_semantic_tools_planner import model, packet

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Risk
from cua_jev.policy import JevFailure, JevRequestError
from cua_jev.trace import JsonlTrace

GOAL = "Fill Name with Ada AND City with Hangzhou; both values must be present."


def condition(name, value):
    return AgentCondition("value_equals", AgentTarget(name=name), value)


def stage(name, value, *, completion=True):
    check = condition(name, value)
    step = AgentStep(name, "fill", "e0" if name == "Name" else "e1", value,
                     label=name, postconditions=(check,))
    return AgentPlan("Complete " + name, (step,), reusable=True,
                     completion_conditions=(check,) if completion else ())


def finished():
    return AgentPlan("Entire original goal observed", (), done=True, reusable=True,
                     completion_conditions=(condition("Name", "Ada"),
                                            condition("City", "Hangzhou")))


class InputBackend(Backend):
    def __init__(self):
        super().__init__()
        self.image = _observed_fixture.__wrapped__().image
        self.observations = 0

    def observe(self):
        self.observations += 1
        return replace(super().observe(), image=self.image)

    def execute(self, *args):
        receipt = super().execute(*args)
        return replace(receipt, output={**receipt.output, "environment_steps": 1})


class SequencePlanner:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def plan(self, goal, snapshot, history, *, max_actions=6, whole_goal_recheck=False):
        self.calls.append({"goal": goal, "image": snapshot.image, "limit": max_actions,
                           "review": whole_goal_recheck})
        answer = next(self.answers)
        return answer() if callable(answer) else answer


def run(answers, *, enabled=True, backend=None, planner=None, policy=None, **config):
    backend = backend or InputBackend()
    planner = planner or SequencePlanner(answers)
    trace = JsonlTrace()
    result = AgentRunner(backend, planner, policy=policy or DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(allow_actions=True, whole_goal_recheck=enabled,
                                            **config)).run(GOAL)
    return result, backend, planner, trace


def test_guarded_capacity_completion_still_reviews_goal_without_rearming_recovery():
    class CapacityOnce:
        calls = 0

        def choose(self, observation, candidates):
            self.calls += 1
            if self.calls == 1:
                failure = JevFailure(observation.observation_id, "a" * 64, "http_status", 1,
                                     False, 400, 1, ((1, "http_status", 400, 1),),
                                     service_code="max_tokens_exceeded")
                raise JevRequestError("bounded fixture refusal", failure)
            return DirectPlanPolicy().choose(observation, candidates)

    backend = InputBackend()
    backend.values = {"Name": "Ada"}
    partial_done = AgentPlan("Only Name observed", (), done=True, reusable=True,
                            completion_conditions=(condition("Name", "Ada"),))
    planner = SequencePlanner([
        stage("Name", "Other"), partial_done, stage("City", "Hangzhou"), finished(),
    ])
    planner.profile = "desktop"
    result, backend, planner, trace = run(
        [], planner=planner, backend=backend, policy=CapacityOnce(), singleton_fast_path=False,
        jev_capacity_recovery=True, jev_network_recovery=True,
    )
    assert backend.executed == ["City"] and backend.values["Name"] == "Ada"
    assert [c["limit"] for c in planner.calls] == [6, 1, 1, 1]
    assert result.capacity_recovery_attempts == 1 and result.network_recovery_attempts == 0
    assert result.capacity_recovery_actions == result.capacity_recovery_verified == 0
    assert result.whole_goal_recheck_calls == 2 and result.whole_goal_recheck_done == 1
    failures = [e["payload"] for e in trace.events if e["kind"] == "policy_failure"]
    assert len(failures) == 1 and failures[0]["retryable"] is False


def test_review_continues_omitted_requirement_then_checks_complete_empty_once():
    result, backend, planner, trace = run([stage("Name", "Ada"), stage("City", "Hangzhou"),
                                           finished()])
    assert backend.values == {"Name": "Ada", "City": "Hangzhou"}
    assert backend.executed == ["Name", "City"]
    assert [c["review"] for c in planner.calls] == [False, True, True]
    assert all(c["goal"] == GOAL and c["image"] is not None for c in planner.calls)
    assert result.planner_calls == 3 and result.steps == 2
    assert result.whole_goal_recheck_calls == 2
    assert result.whole_goal_recheck_continues == result.whole_goal_recheck_done == 1
    assert result.status == "unverified" and not result.success
    assert backend.observations > result.steps
    stages = [e["payload"]["stage"] for e in trace.events
              if e["kind"] == "agent_whole_goal_recheck"]
    assert stages.count("empty_done_fresh_checked") == 1


def test_default_off_keeps_legacy_planner_api_and_omitted_work_behavior():
    class Legacy:
        def plan(self, goal, snapshot, history, *, max_actions=6):
            return stage("Name", "Ada")

    result, backend, _, trace = run([], enabled=False, planner=Legacy())
    assert backend.values == {"Name": "Ada"}
    assert result.planner_calls == 1 and result.status == "unverified"
    assert not any(k.startswith("whole_goal_recheck") for k in result.to_dict())
    assert not any(e["kind"] == "agent_whole_goal_recheck" for e in trace.events)


def test_partial_stage_exhaustion_does_not_review_each_action():
    result, backend, planner, _ = run([stage("Name", "Ada", completion=False),
                                     stage("City", "Hangzhou"), finished()])
    assert [c["review"] for c in planner.calls] == [False, False, True]
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
    assert backend.executed == ["Name", "City"]
    assert "whole_goal_recheck_continues" not in result.to_dict()


def test_no_extra_planning_budget_is_granted_for_review():
    result, backend, planner, _ = run([stage("Name", "Ada")], max_plans=1)
    assert result.status == "max_plans" and len(planner.calls) == 1
    assert result.whole_goal_recheck_calls == 0 and backend.executed == ["Name"]


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_config_requires_explicit_boolean(bad):
    with pytest.raises(ValueError, match="whole_goal_recheck must be a boolean"):
        AgentConfig(whole_goal_recheck=bad)


def test_unconditional_initial_done_cannot_bypass_review_or_loop():
    backend = InputBackend()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    result, _, planner, _ = run([AgentPlan("Done without proof", (), done=True), finished()],
                                 backend=backend)
    assert not backend.executed
    assert [c["review"] for c in planner.calls] == [False, True]
    assert result.whole_goal_recheck_done == 1 and result.status == "unverified"


def test_review_itself_cannot_stop_with_unconditional_empty_done():
    result, backend, planner, _ = run([stage("Name", "Ada"),
                                      AgentPlan("Done without proof", (), done=True)], max_failures=1)
    assert result.status == "planner_error"
    assert result.whole_goal_recheck_calls == 1 and result.whole_goal_recheck_done == 0
    assert len(planner.calls) == 2 and backend.executed == ["Name"]


def test_empty_review_unknown_condition_keeps_unknown_and_fails_closed():
    unknown = AgentPlan("Not observable", (), done=True, reusable=True,
                        completion_conditions=(AgentCondition("exists", AgentTarget(name="Absent")),))
    result, backend, planner, trace = run([stage("Name", "Ada"), unknown], max_failures=1)
    assert result.status == "stuck" and result.whole_goal_recheck_done == 0
    assert len(planner.calls) == 2 and backend.executed == ["Name"]
    recovery = [e["payload"] for e in trace.events if e["kind"] == "agent_recovery"]
    assert any(any(c["outcome"] == "unknown" for c in e.get("completion_conditions", []))
               for e in recovery)


def test_deadline_after_review_is_not_extended_for_empty_completion(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: now[0])

    def late():
        now[0] = 3.0
        return finished()

    result, backend, planner, _ = run([stage("Name", "Ada"), late], timeout_s=2)
    assert result.status == "timeout" and result.duration_ms >= 3000
    assert len(planner.calls) == 2 and backend.executed == ["Name"]
    assert result.whole_goal_recheck_done == 0


def test_readonly_zero_receipts_do_not_rearm_whole_goal_review():
    class ReadOnly(InputBackend):
        def compile(self, step, planned, current):
            return (ActionCandidate(step.id, Channel.CONTROL, "agent.read", "Read current state",
                                    risk=Risk.READ_ONLY, intent=step.id, verifier="agent.effect"),)

        def execute(self, candidate, observation_id, decision_id):
            return ActionReceipt(observation_id, decision_id, candidate.id, candidate.channel,
                                 candidate.capability, True, 0, 0.001,
                                 output={"environment_steps": 0})

    noop = AgentPlan("Local state read", (AgentStep("read", "wait", value="250"),), reusable=True,
                     completion_conditions=(AgentCondition("text_contains", value="Form"),))
    result, backend, planner, _ = run([noop, noop], backend=ReadOnly())
    assert result.status == "stuck" and "new actual input" in result.reason
    assert len(planner.calls) == 2 and result.whole_goal_recheck_calls == 1
    assert result.whole_goal_recheck_done == 0 and not backend.executed


def test_deadline_reached_during_final_conditions_cannot_report_review_done(monkeypatch):
    from cua_jev import agent_runner

    now, reviewed = [0.0], [False]
    original = agent_runner.conditions_hold
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])

    def slow_conditions(conditions, snapshot):
        held = original(conditions, snapshot)
        if reviewed[0] and held:
            now[0] = 3.0
        return held

    def reviewed_done():
        reviewed[0] = True
        return finished()

    monkeypatch.setattr(agent_runner, "conditions_hold", slow_conditions)
    backend = InputBackend()
    backend.values = {"City": "Hangzhou"}
    result, backend, _, _ = run([stage("Name", "Ada"), reviewed_done],
                                backend=backend, timeout_s=2)
    assert result.status == "timeout" and result.whole_goal_recheck_done == 0
    assert backend.executed == ["Name"]


@pytest.mark.parametrize("limit", [1, 6])
def test_actual_planner_review_preserves_full_goal_image_strong_wire_and_one_http(limit):
    observed = _observed_fixture.__wrapped__()
    requests = []
    conditions = [{"kind": "window_contains", "value": "Main"}]
    raw = ({"summary": "Entire goal visible", "done": True, "completion_conditions": conditions}
           if limit == 1 else {"label": "Entire goal visible", "conditions": conditions})
    name = "declare_current_completion" if limit == 1 else "declare_completion"

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v56")
        parsed = planner.plan(GOAL, observed, [{"recovery_required": True}],
                              max_actions=limit, whole_goal_recheck=True)
    assert parsed.done and parsed.completion_conditions
    assert len(requests) == planner.calls == 1
    body = json.loads(requests[0].content)
    assert "WHOLE USER GOAL RECHECK" in body["messages"][0]["content"]
    assert "EVERY requirement" in body["messages"][0]["content"]
    content = body["messages"][1]["content"]
    state = json.loads(next(b["text"] for b in content if b["type"] == "text"))
    assert state["goal"] == GOAL
    assert any(b["type"] == "image_url" for b in content)
    assert len(body["tools"]) == (5 if limit == 1 else 8)
    assert planner.last_request_provenance["image_uploaded"] is True
    assert planner.last_request_provenance["max_actions"] == limit
    assert planner.last_request_provenance["whole_goal_recheck"] is True
    assert requests[0].extensions["timeout"]["read"] == 90


def test_actual_planner_rejects_review_without_image_before_http():
    observed = replace(_observed_fixture.__wrapped__(), image=None)
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: requests.append(r))) as client:
        planner = model(client, desktop_output_format="semantic_tools_v56")
        with pytest.raises(ValueError, match="full desktop image"):
            planner.plan(GOAL, observed, [], whole_goal_recheck=True)
    assert not requests
