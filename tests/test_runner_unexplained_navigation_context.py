"""Runner asks for information only after exact new56 verified navigation evidence."""

from dataclasses import replace

import pytest
from test_osworld_navigation_context_signal import Environment, _observation

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_navigation_policy import enable_full_legal_navigation_gestures
from cua_jev.models import Verification
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import RulePolicy
from cua_jev.trace import JsonlTrace


class ReturningEnvironment(Environment):
    def _get_obs(self):
        self.reads += 1
        return _observation(len(self.inputs) == 1)

    def step(self, action, pause=0):
        self.inputs.append(action)
        return _observation(len(self.inputs) == 1), 0, False, {}


class SpyBackend(OSWorldBackend):
    def __init__(self, env, *, change=None):
        super().__init__(env, pause=0)
        self.context_queries = 0
        self.change = change

    def take_unexplained_navigation_context(self, candidate, receipt):
        self.context_queries += 1
        return super().take_unexplained_navigation_context(candidate, receipt)

    def execute(self, *args):
        receipt = super().execute(*args)
        if self.change == "zero":
            return replace(receipt, output={**receipt.output, "environment_steps": 0})
        return receipt

    def verify(self, *args):
        if self.change == "unverified":
            return Verification(False, "agent.effect", {"navigation": True})
        return super().verify(*args)

    def verify_pending(self, *args):
        if self.change == "unverified":
            return Verification(False, "agent.effect", {"navigation": True})
        return super().verify_pending(*args)


class Planner:
    profile = "desktop"
    desktop_contract = "property_intent_v52"
    desktop_output_format = "semantic_tools_v56"

    def __init__(self, role="frame"):
        self.role = role
        self.requests = []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.requests.append({"limit": max_actions, "snapshot": snapshot, "history": history})
        if len(self.requests) == 1:
            plan = AgentPlan("Reach Options", (AgentStep(
                "reach", "click", target=AgentTarget(app="Demo"), navigation=True,
                native_navigation_gestures=True,
                label="Reach Options", recovery="observe",
                postconditions=(AgentCondition("exists", AgentTarget(name="Options", role=self.role)),),
            ),), reusable=True)
            return (enable_full_legal_navigation_gestures(plan)
                    if self.desktop_output_format == "semantic_tools_v56" else plan)
        return AgentPlan("Close current Options", (AgentStep(
            "close", "click", target=AgentTarget(name="Accept", role="button", app="Demo"),
            postconditions=(AgentCondition("absent", AgentTarget(name="Options", role="dialog")),),
        ),), reusable=True)


def exercise(*, role="frame", change=None, steps=1, old=False, allow=True, max_plans=40):
    env = ReturningEnvironment()
    backend = SpyBackend(env, change=change)
    planner = Planner(role)
    if old:
        planner.desktop_output_format = "semantic_tools_v55"
    trace = JsonlTrace()
    result = AgentRunner(backend, planner, policy=RulePolicy(), trace=trace,
                         config=AgentConfig(allow_actions=allow, max_steps=steps, max_plans=max_plans,
                                            reuse_verified_post_snapshot=False,
                                            reuse_verified_pending_snapshot=False)).run("Use Options")
    return result, env, backend, planner, trace


def test_new56_unexplained_dialog_replans_fresh_h1_after_preserving_verified_input():
    result, env, backend, planner, trace = exercise(steps=2)
    try:
        assert result.unexplained_navigation_context_replans == 1
        assert backend.context_queries == 1 and len(env.inputs) == 2
        assert [r["limit"] for r in planner.requests] == [6, 1]
        assert planner.requests[1]["snapshot"].image is not None
        assert planner.requests[1]["snapshot"].title == "Options"
        assert result.verified_pending_snapshot_reuses == result.post_snapshot_reuses == 0
        assert result.recovery_planning_snapshot_reuses == 0
        steps = [e["payload"] for e in trace.events if e["kind"] == "agent_step"]
        assert len(steps) == 2 and all(e["verified"] for e in steps)
        signals = [e["payload"] for e in trace.events
                   if e["kind"] == "agent_unexplained_navigation_context"]
        assert len(signals) == 1 and signals[0]["input_authority"] is False
        assert signals[0]["arrival_evidence"][0]["outcome"] == "unknown"
        assert signals[0]["reason"] == "unexplained_new_dialog_after_verified_navigation"
        assert result.replan_reasons["unexplained_navigation_context"] == 1
        assert backend._input_evidence_enabled is False
    finally:
        backend.close()


def test_exact_arrival_does_not_add_information_request_or_change_legacy_metric_shape():
    result, env, backend, planner, _trace = exercise(role="dialog")
    try:
        assert len(env.inputs) == len(planner.requests) == 1
        assert backend.context_queries == 1
        assert result.unexplained_navigation_context_replans == 0
        assert "unexplained_navigation_context_replans" not in result.to_dict()
        assert backend._input_evidence_enabled is False
    finally:
        backend.close()


def test_information_signal_does_not_grant_another_planning_call():
    result, env, backend, planner, _trace = exercise(steps=2, max_plans=1)
    try:
        assert result.status == "max_plans" and result.planner_calls == 1
        assert len(env.inputs) == len(planner.requests) == 1
        assert backend.context_queries == 1
        assert result.unexplained_navigation_context_replans == 0
        assert backend._input_evidence_enabled is False
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["old_format", "zero", "unverified", "guard"])
def test_no_context_query_from_oldformat_zero_finalfailure_or_rejected_guard(change):
    result, env, backend, planner, trace = exercise(
        old=change == "old_format", change=change, allow=change != "guard",
    )
    try:
        assert backend.context_queries == 0
        assert result.unexplained_navigation_context_replans == 0
        assert not any(e["kind"] == "agent_unexplained_navigation_context" for e in trace.events)
        assert len(planner.requests) == 1
        assert len(env.inputs) == (0 if change == "guard" else 1)
        assert backend._input_evidence_enabled is False
    finally:
        backend.close()
