"""Formal review uses the existing H1 request and original runner checks (offline)."""

import copy
import json

import httpx
import pytest
from test_goal_recheck_frame_UNEXECUTED import ProofBackend, RecordingPlanner
from test_semantic_tools_v57 import packet
from test_semantic_tools_v57 import planner as make_planner
from test_structured_declaration_planner_runner_v66 import ProductionMemoryFields
from test_whole_goal_recheck import GOAL, SequencePlanner, finished, run, stage

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.trace import JsonlTrace


def target(name, **changes):
    return {"name": name, "role": "textbox", "app": "Editor", "window": "Main", **changes}


def conditions(*names):
    return [{"kind": "value_equals", "target": target(name),
             "value": "Ada" if name == "Name" else "Hangzhou"} for name in names]


def completion(*names):
    return packet({"label": "Requested values observed", "conditions": conditions(*names)},
                  "declare_completion", content=None)


class CapturingPlanner(MultimodalPlanner):
    def plan(self, goal, snapshot, history, *, max_actions=6, whole_goal_recheck=False, **options):
        self.invocations.append({"goal": goal, "snapshot": snapshot, "history": copy.deepcopy(history),
                                 "max_actions": max_actions, "review": whole_goal_recheck})
        return super().plan(goal, snapshot, history, max_actions=max_actions,
                            whole_goal_recheck=whole_goal_recheck, **options)


def actual_episode(backend, responses, *, max_plans=4, max_failures=1):
    responses, requests = iter(responses), []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=next(responses))

    trace = JsonlTrace()
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt="semantic_tools_v64", cls=CapturingPlanner)
        planner.invocations = []
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(mode="hybrid", allow_actions=True, whole_goal_recheck=True,
                plan_horizon=6, max_plans=max_plans, max_failures=max_failures,
                max_steps=3)).run(GOAL)
    return result, planner, requests, trace


def test_completed_goal_still_requires_one_full_goal_fresh_strong_H1_review():
    backend = ProductionMemoryFields()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    result, planner, requests, _ = actual_episode(backend, [completion("Name", "City")] * 2)
    assert result.status == "unverified" and not result.success
    assert result.planner_calls == planner.calls == len(requests) == 2
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
    assert backend.environment_steps == result.steps == 0 and not backend.executed
    assert [(call["max_actions"], call["review"]) for call in planner.invocations] == [
        (6, False), (1, True)]
    assert all(call["goal"] == GOAL for call in planner.invocations)
    review = planner.invocations[1]
    assert review["snapshot"] is not planner.invocations[0]["snapshot"]
    assert backend.frames[-1] is not review["snapshot"]  # Original final fresh stopping check.
    body = json.loads(requests[1].content)
    assert len(body["tools"]) == 5 and body["enable_thinking"] is True
    assert requests[1].extensions["timeout"]["read"] == 90
    assert any(block["type"] == "image_url" for block in body["messages"][1]["content"])
    reference = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: (
        reference.append(request) or httpx.Response(200, json=completion("Name", "City"))
    ))) as client:
        make_planner(client, fmt="semantic_tools_v58").plan(
            GOAL, review["snapshot"], review["history"], max_actions=1, whole_goal_recheck=True)
    assert len(reference) == 1 and reference[0].content == requests[1].content


def test_incomplete_review_executes_one_original_action_then_resumes_normal_H6():
    backend = ProductionMemoryFields()
    backend.values = {"Name": "Ada"}
    current = {"summary": "Fill the observed remaining field", "id": "city", "done": False,
               "operation": "fill", "target": target("City"), "value": "Hangzhou",
               "postconditions": conditions("City")}
    result, planner, requests, trace = actual_episode(backend, [completion("Name"),
        packet(current, "declare_semantic_current_action", content=None),
        completion("Name", "City"), completion("Name", "City")])
    assert [(call["max_actions"], call["review"]) for call in planner.invocations] == [
        (6, False), (1, True), (6, False), (1, True)]
    assert result.planner_calls == planner.calls == len(requests) == 4
    assert [len(json.loads(request.content)["tools"]) for request in requests] == [6, 5, 6, 5]
    assert backend.executed == ["City"] and backend.values == {"Name": "Ada", "City": "Hangzhou"}
    assert backend.environment_steps == result.steps == 1
    assert result.whole_goal_recheck_calls == 2
    assert result.whole_goal_recheck_continues == result.whole_goal_recheck_done == 1
    assert result.status == "unverified" and not result.success
    assert not any(event["kind"] == "agent_recovery" for event in trace.events)


def test_invalidated_reused_completion_frame_stays_H1_after_second_observe():
    backend = ProofBackend(drift="late_observe")
    backend.values = {"City": "Hangzhou"}
    planner = RecordingPlanner([stage("Name", "Ada"), finished()])
    result, _, _, _ = run([], backend=backend, planner=planner,
        reuse_verified_post_snapshot=True, reuse_whole_goal_planning_snapshot=True)
    assert [call["limit"] for call in planner.calls] == [6, 1]
    assert [call["review"] for call in planner.calls] == [False, True]
    assert planner.frames[1] is not backend.offered[0]
    assert result.whole_goal_planning_snapshot_reuses == 0
    assert result.planner_calls == 2 and result.steps == 1
    assert result.whole_goal_recheck_done == 1 and backend.executed == ["Name"]


@pytest.mark.parametrize("budget", ["plans", "deadline"])
def test_formal_H1_grants_no_additional_plans_or_episode_time(monkeypatch, budget):
    now = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: now[0])

    def late():
        now[0] = 3.0
        return finished()

    planner = SequencePlanner([stage("Name", "Ada"), late])
    settings = {"max_plans": 1} if budget == "plans" else {"timeout_s": 2}
    result, backend, _, _ = run([], planner=planner, **settings)
    assert backend.executed == ["Name"] and result.whole_goal_recheck_done == 0
    assert result.planner_calls == len(planner.calls) == (1 if budget == "plans" else 2)
    assert result.status == ("max_plans" if budget == "plans" else "timeout")
    assert [call["limit"] for call in planner.calls] == ([6] if budget == "plans" else [6, 1])


def test_formal_H1_keeps_wrong_owner_target_rejection_and_sends_zero_input():
    backend = ProductionMemoryFields()
    backend.values = {"Name": "Ada"}
    bad = {"summary": "Rejected owner", "id": "city", "done": False,
           "operation": "fill", "target": target("City", app="Other"), "value": "Hangzhou",
           "postconditions": conditions("City")}
    result, planner, requests, _ = actual_episode(backend, [completion("Name"),
        packet(bad, "declare_semantic_current_action", content=None)])
    assert result.status == "planner_error" and result.whole_goal_recheck_done == 0
    assert result.planner_calls == planner.calls == len(requests) == 2
    assert [call["max_actions"] for call in planner.invocations] == [6, 1]
    assert backend.environment_steps == result.steps == 0 and not backend.executed
    assert backend.values == {"Name": "Ada"}


def test_V_and_nonreview_H6_routing_remain_original():
    for mode, limits in [("hybrid", [6, 6, 1]), ("vlm", [1, 1, 1])]:
        planner = SequencePlanner([stage("Name", "Ada", completion=False),
                                   stage("City", "Hangzhou"), finished()])
        result, backend, _, _ = run([], planner=planner, mode=mode)
        assert [call["limit"] for call in planner.calls] == limits
        assert [call["review"] for call in planner.calls] == [False, False, True]
        assert result.planner_calls == 3 and result.steps == 2
        assert result.whole_goal_recheck_done == 1 and backend.executed == ["Name", "City"]
