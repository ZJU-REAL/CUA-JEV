import time
from dataclasses import replace

import pytest

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentPlan, AgentSnapshot
from cua_jev.benchmark_miniwob import PlanOrderPolicy
from cua_jev.models import ActionCandidate, Channel, Risk
from cua_jev.osworld_mcp import (
    JointOSWorldBackend,
    JointStep,
    ToolCatalog,
    parse_joint_plan,
    tool_request_completed,
)
from cua_jev.trace import JsonlTrace
from cua_jev.vision import WindowImage

TOOLS = [{"name": "test.set", "inputSchema": {
    "type": "object", "properties": {"value": {"type": "integer"}},
    "required": ["value"], "additionalProperties": False,
}}]
FRAME = AgentSnapshot("desktop://test", "test", "unchanged", (),
                      image=WindowImage(b"\xff\xd8test", bytes(4096), (0, 0, 1000, 1000)))


def raw_plan(count=1):
    return {"summary": "test-only plan", "done": False, "steps": [
        {"id": f"s{i}", "depends_on": [f"s{i-1}"] if i else [], "options": [
            {"kind": "mcp", "tool": "test.set", "arguments": {"value": i}, "label": "set test value"}
        ]} for i in range(count)
    ]}


class Session:
    call_timeout_s = 30

    def __init__(self, result=None, fail=False):
        self.calls, self.fail, self.closed = [], fail, False
        self.result = result if result is not None else {
            "isError": False, "structuredContent": {"success": True}, "content": []}

    def list_tools(self):
        return TOOLS

    def call_tool(self, name, arguments, **kwargs):
        self.calls.append((name, arguments))
        assert 0 < kwargs["timeout_s"] <= self.call_timeout_s
        if self.fail:
            raise TimeoutError("test unknown outcome")
        return self.result

    def close(self):
        self.closed = True


class Gui:
    def __init__(self):
        self.observations, self.closed, self.compiles = 0, False, []

    def observe(self):
        self.observations += 1
        return FRAME

    def compile(self, step, planned, current):
        self.compiles.append(step)
        return (ActionCandidate("gui-test", Channel.GUI, "test.gui", "test GUI option",
                                risk=Risk.LOCAL_WRITE, intent=step.id),)

    def close(self):
        self.closed = True


def test_validates_whole_long_plan_before_executing_prefix():
    raw = raw_plan(13)
    plan = parse_joint_plan(raw, FRAME, ToolCatalog(TOOLS), 6)
    assert len(plan.steps) == 6
    assert plan.steps[-1].depends_on == ("s4",)
    assert plan.steps[0].to_dict()["options"][0]["arguments"] == {"value": 0}
    raw["steps"][-1]["options"][0]["arguments"]["value"] = "bad suffix"
    with pytest.raises(ValueError, match="schema"):
        parse_joint_plan(raw, FRAME, ToolCatalog(TOOLS), 6)


@pytest.mark.parametrize("change", ["forward_dependency", "duplicate", "unknown_tool", "extra_field"])
def test_rejects_unbound_or_invalid_plans(change):
    raw = raw_plan(2)
    if change == "forward_dependency":
        raw["steps"][0]["depends_on"] = ["s1"]
    elif change == "duplicate":
        raw["steps"][1]["id"] = "s0"
    elif change == "unknown_tool":
        raw["steps"][0]["options"][0]["tool"] = "not.in.catalog"
    else:
        raw["steps"][0]["options"][0]["code"] = "unoffered action"
    with pytest.raises(ValueError):
        parse_joint_plan(raw, FRAME, ToolCatalog(TOOLS), 6)


def test_gui_and_mcp_are_offered_for_same_intent_without_equivalence_claim():
    raw = raw_plan()
    raw["steps"][0]["options"].append({"kind": "gui", "action": {
        "operation": "visual_click", "box": [10, 10, 100, 100], "label": "test button"}})
    plan = parse_joint_plan(raw, FRAME, ToolCatalog(TOOLS), 1)
    gui, session = Gui(), Session()
    backend = JointOSWorldBackend(gui, session)
    choices = backend.compile(plan.steps[0], FRAME, FRAME)
    assert {c.channel for c in choices} == {Channel.GUI, Channel.MCP}
    assert {c.intent for c in choices} == {"s0"}
    assert gui.compiles[0].operation == "visual_click"


@pytest.mark.parametrize("result", [
    {"isError": True},
    {"isError": False, "structuredContent": {"success": False}},
    {"isError": False, "content": [{"type": "text", "text": '{"error":"failed"}'}]},
    {"isError": False, "structuredContent": {"status": "failed"}},
])
def test_tool_errors_are_not_completed_requests(result):
    assert not tool_request_completed(result)


def test_receipt_bound_one_shot_post_capture_and_no_implicit_goal_success():
    gui, session = Gui(), Session()
    backend = JointOSWorldBackend(gui, session)
    step = parse_joint_plan(raw_plan(), FRAME, backend.catalog, 1).steps[0]
    candidate, = backend.compile(step, FRAME, FRAME)
    receipt = backend.execute(candidate, "o", "d")
    assert len(session.calls) == 1 and gui.observations == 1
    verification = backend.verify(candidate, receipt)
    assert verification.passed
    assert verification.details["semantic_goal_verified"] is False
    assert receipt.output["task_success"] is False
    assert backend.take_verified_post_snapshot(candidate, receipt)[0] is FRAME
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert not backend.verify(candidate, replace(receipt, success=False)).passed
    with pytest.raises(ValueError, match="consumed"):
        backend.execute(candidate, "o2", "d2")
    backend.close()
    assert gui.closed and session.closed


def test_unknown_outcome_is_not_retried_and_no_capture_is_reused():
    gui, session = Gui(), Session(fail=True)
    backend = JointOSWorldBackend(gui, session)
    step = parse_joint_plan(raw_plan(), FRAME, backend.catalog, 1).steps[0]
    candidate, = backend.compile(step, FRAME, FRAME)
    receipt = backend.execute(candidate, "o", "d")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert len(session.calls) == 1 and gui.observations == 0
    assert backend.take_verified_post_snapshot(candidate, receipt) is None


def test_stale_or_expired_candidate_never_reaches_tool():
    backend = JointOSWorldBackend(Gui(), Session())
    step = parse_joint_plan(raw_plan(), FRAME, backend.catalog, 1).steps[0]
    candidate, = backend.compile(step, FRAME, FRAME)
    backend.observe()
    with pytest.raises(ValueError):
        backend.execute(candidate, "o", "d")
    candidate, = backend.compile(step, FRAME, FRAME)
    with pytest.raises(TimeoutError):
        backend.execute_with_deadline(candidate, "o", "d", deadline=time.perf_counter() - 1)
    assert not backend.session.calls


def test_existing_runner_dispatches_mcp_and_preserves_final_evaluator_boundary():
    backend = JointOSWorldBackend(Gui(), Session())

    class Planner:
        def __init__(self):
            self.calls = 0

        def plan(self, goal, snapshot, history, *, max_actions):
            self.calls += 1
            if self.calls == 1:
                return parse_joint_plan(raw_plan(2), snapshot, backend.catalog, max_actions)
            return AgentPlan("advisory completion", (), done=True)

    planner, trace = Planner(), JsonlTrace()
    result = AgentRunner(backend, planner, policy=PlanOrderPolicy(), trace=trace,
                         config=AgentConfig(mode="hybrid", allow_actions=True, max_steps=4,
                                            max_plans=4, adaptive_dispatch=True,
                                            reuse_verified_post_snapshot=True)).run("test goal")
    assert len(backend.session.calls) == 2
    assert result.success is False  # No external evaluator was supplied.
    assert result.planner_calls == 2
    assert result.jev_calls == 0
    assert result.plan_reuses == 1
    assert all(not e["payload"].get("task_success", False) for e in trace.events
               if e["kind"] == "verification")
    assert all(isinstance(s, JointStep) for s in parse_joint_plan(
        raw_plan(2), FRAME, backend.catalog, 2).steps)
