'Regression tests for test stage program retirement v60.'

import copy

import pytest
from test_nav_arrival_caption_recovery_v60 import APP, CAPTION, ReadonlyDesktop, menu_xml
from test_program_aware_reuse_auditor_v60 import ProgramEvents
from test_property_preparation import initial as initial
from test_stage_program_contract_v60 import parse_original
from test_stage_program_recovery_v60 import property_run
from test_stage_program_runner_v60 import (
    MemoryFields,
    NumericProgramEnvironment,
    PreferField,
    ProgramPlanner,
    events,
    facts,
    field_member,
    memory_parser,
    run,
    two_fields,
)

from cua_jev import agent_runner
from cua_jev.agent_runner import DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend
from scripts.analyze_stage_program_reuse_v60 import audit_trace


class ContinueRequestedFields(ProgramPlanner):
    """A real new planner invocation may supply the remaining generic work."""

    def __init__(self, backend, raw):
        self.backend = backend
        super().__init__(raw, memory_parser,
                         tuple(facts(name, "requested-" + name) for name in backend.names))

    def plan(self, goal, snapshot, history, *, max_actions=6, whole_goal_recheck=False, **options):
        if self.calls == 0:
            return super().plan(goal, snapshot, history, max_actions=max_actions,
                                whole_goal_recheck=whole_goal_recheck, **options)
        self.calls += 1
        self.requests.append({"snapshot": snapshot, "history": copy.deepcopy(history),
                              "limit": max_actions, "review": whole_goal_recheck})
        remaining = tuple(name for name in self.backend.names
                          if self.backend.values.get(name) != "requested-" + name)
        if remaining:
            raw = field_member("new-independent-plan", remaining[:max_actions])["arguments"]
            return memory_parser({"intent": "edit_fields", **raw}, snapshot, 6)
        return AgentPlan("Fresh whole-goal conditions", (), done=True, reusable=True,
                         completion_conditions=self.done_conditions)


def ordinary_raw(names):
    split = min(5, len(names) - 1)
    return {"label": "Set requested generic fields", "stages": [
        field_member("first-view", names[:split]), field_member("next-view", names[split:]),
    ]}


def normal_retirement(trace):
    retired = events(trace, "agent_program_abandoned")
    assert len(retired) == 1
    row = retired[0]
    assert row["force_semantic_recovery"] is False
    assert row["semantic_recovery_before"] is False
    assert row["semantic_recovery_after"] is False
    assert row["preserve_original_handoff_flags"] is True
    assert row["budget_refunded"] is False
    assert row["old_program_budget_rearmed"] is False
    assert row["program_handoff_pending_before"] is row["program_handoff_pending_after"]
    return row


def test_six_shared_attempts_retire_to_fresh_H6_before_seventh_input_without_refund():
    names = tuple("Requested" + str(index) for index in range(7))
    backend = MemoryFields(names, raw_macro_inputs=3)
    planner = ContinueRequestedFields(backend, ordinary_raw(names))
    input_origins = []
    backend.after_execute = lambda: input_origins.append(planner.calls)
    result, trace = run(backend, planner, max_plans=3)
    assert backend.executed == list(names) and backend.environment_steps == 21
    assert result.steps == 7 and input_origins == [1] * 6 + [2]
    assert planner.requests[1]["limit"] == 6 and planner.requests[1]["review"] is False
    assert [row["planner_calls"] for row in events(trace, "agent_step")] == [1] * 6 + [2]
    retired = normal_retirement(trace)
    assert retired["reason"] == "horizon_reached"
    assert retired["plan_actions"] == 6 and retired["remaining_actions"] == 0
    # The unfinished tail was not compiled again as another stage of origin1.
    assert len(planner.activations) == 1
    assert not any(row["stage_index"] == 1 for row in events(trace, "agent_program_stage_completed"))
    assert len(events(trace, "receipt")) == 7  # No hidden/free/refunded macro.


def test_original_TTL_boundary_retires_to_new_H6_without_renewing_old_program_clock(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = MemoryFields()
    planner = ContinueRequestedFields(backend, ordinary_raw(backend.names))
    input_origins = []

    def first_macro_cost():
        input_origins.append(planner.calls)
        if len(input_origins) == 1:
            now[0] += 10.1

    backend.after_execute = first_macro_cost
    result, trace = run(backend, planner, max_plans=3, plan_ttl_s=10, timeout_s=300)
    assert backend.executed == ["Name", "City"] and input_origins == [1, 2]
    assert result.steps == 2 and planner.requests[1]["limit"] == 6
    retired = normal_retirement(trace)
    assert retired["original_plan_at"] == 10.0 and retired["plan_actions"] == 1
    assert "ttl" in retired["reason"] or "expired" in retired["reason"]
    assert planner.activations == []
    assert all(row["original_plan_at"] == 10.0 for row in events(trace, "agent_program_stage_activated"))


def test_original_PlanExpired_after_slow_selection_sends_no_old_input_then_new_H6(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = MemoryFields()
    planner = ContinueRequestedFields(backend, ordinary_raw(backend.names))
    calls = []
    backend.after_execute = lambda: calls.append(planner.calls)

    class SlowFirstSelection(DirectPlanPolicy):
        count = 0

        def choose(self, observation, candidates):
            self.count += 1
            choice = super().choose(observation, candidates)
            if self.count == 1:
                now[0] += 10.1
            return choice

    result, trace = run(backend, planner, policy=SlowFirstSelection(), max_plans=3,
                        singleton_fast_path=False, plan_ttl_s=10, timeout_s=300)
    assert result.steps == 2 and backend.executed == ["Name", "City"]
    assert calls == [2, 2] and planner.requests[1]["limit"] == 6
    assert any(row["reason"] == "plan_expired_before_action" for row in events(trace, "agent_recovery"))
    retired = normal_retirement(trace)
    assert retired["plan_actions"] == 0 and retired["attempt_receipts"] == []


def numeric_program():
    target = AgentTarget(name="Requested numeric limit", role="spinbutton",
                         app="NumberTool", window="Preferences")
    condition = AgentCondition("value_equals", target, "100")
    raw_target = {"name": target.name, "role": target.role, "app": target.app, "window": target.window}
    raw = {"label": "Commit and inspect the requested numeric property", "stages": [
        {"id": "numeric", "tool": "declare_property_stage", "arguments": {
            "label": "Set requested numeric limit", "scope": {"app": "NumberTool", "window": "Preferences"},
            "edit_operation": "set", "edit_value": "100", "numeric_finalize_key": "enter",
            "confirmation_label": "none", "completion": "stage"}},
        {"id": "inspect", "tool": "declare_navigation_stage", "arguments": {
            "label": "Inspect the committed setting", "scope": {"app": "NumberTool"},
            "arrival": [{"kind": "value_equals", "target": raw_target, "value": "100"}],
            "completion": "stage"}},
    ]}
    return ProgramPlanner(raw, parse_original, (condition,))


def test_declared_numeric_key_expiry_preserves_original_true_H1_flag_and_false_fill_cost(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    environment, planner = NumericProgramEnvironment(), numeric_program()

    class SlowDeclaredKey(PreferField):
        selected_keys = 0

        def choose(self, observation, candidates):
            choice = super().choose(observation, candidates)
            selected = next(candidate for candidate in candidates if candidate.id == choice.candidate_id)
            # PublicDecisionPolicy deliberately strips all candidate arguments.
            # This fixture's only key offer is the already declared numeric key.
            if selected.capability == "agent.osworld.key_press":
                assert selected.arguments == {}
                self.selected_keys += 1
                now[0] += 10.1
            return choice

    backend = OSWorldBackend(environment, pause=0)
    policy = SlowDeclaredKey()
    result, trace = run(backend, planner, policy=policy, singleton_fast_path=False,
                        reuse_verified_post_snapshot=True, plan_ttl_s=10, timeout_s=300)
    assert policy.selected_keys == 1
    assert planner.requests[1]["limit"] == 1 and planner.activations == []
    assert [action["action_type"] for action in environment.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert result.steps == 1 and backend.environment_steps == 3
    assert events(trace, "agent_step")[0]["verified"] is False
    assert any(row["status"] == "pending" for row in events(trace, "agent_declared_numeric_finalize"))
    assert not any(row["status"] == "committed" for row in events(trace, "agent_declared_numeric_finalize"))
    retired = events(trace, "agent_program_abandoned")[0]
    assert retired["semantic_recovery_before"] is retired["semantic_recovery_after"] is True
    assert retired["budget_refunded"] is False and retired["plan_actions"] == 1
    assert retired["attempt_receipts"][0]["final_verified"] is False
    assert not events(trace, "agent_program_stage_completed")


@pytest.mark.parametrize("failure", ["receipt", "verification"])
def test_true_failure_still_uses_original_H1_and_keeps_actual_attempt(failure):
    backend = MemoryFields(failure=failure)
    planner = two_fields(backend)
    result, trace = run(backend, planner)
    assert planner.requests[1]["limit"] == 1 and planner.activations == []
    assert backend.executed == ["Name"] and result.steps == 1
    retired = events(trace, "agent_program_abandoned")[0]
    assert retired["semantic_recovery_after"] is True and retired["budget_refunded"] is False
    assert retired["semantic_recovery_before"] is True or retired["force_semantic_recovery"] is True
    assert len(retired["attempt_receipts"]) == 1 and retired["attempt_receipts"][0]["final_verified"] is False


def test_explicit_replan_policy_still_H1_and_does_not_refund_failed_native_fill(initial, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    result, trace, planner, environment = property_run(initial, "replan")
    assert planner.requests[1]["limit"] == 1 and planner.activations == []
    assert result.steps == 1 and result.recovery_observations == 0
    assert [action["action_type"] for action in environment.actions] == ["CLICK", "HOTKEY", "TYPING"]
    retired = events(trace, "agent_program_abandoned")[0]
    assert retired["semantic_recovery_after"] is True
    assert retired["semantic_recovery_before"] is True or retired["force_semantic_recovery"] is True
    assert retired["budget_refunded"] is False


def test_actual_caption_diagnostic_remains_zero_input_H1_not_normal_budget_retirement():
    desktop, backend = ReadonlyDesktop(menu_xml()), None
    backend = OSWorldBackend(desktop, pause=0)
    missing = AgentCondition("exists", AgentTarget(name=CAPTION, role="menuitem", app=APP))
    raw = {"label": "Reach the requested generic menu and future field", "stages": [
        {"id": "reach", "tool": "declare_navigation_stage", "arguments": {
            "label": "Reach requested menu", "scope": {"app": APP}, "completion": "stage",
            "arrival": [{"kind": "exists", "target": {"name": CAPTION, "role": "menuitem", "app": APP}}]}},
        {"id": "later", "tool": "declare_field_edits_stage", "arguments": {
            "label": "Fill future field", "completion": "stage", "edits": [
                {"operation": "fill", "target": {"name": "Future field", "role": "textbox", "app": APP},
                 "value": "requested"}]}},
    ]}
    planner = ProgramPlanner(raw, parse_original, (missing,))

    class NeverPolicy:
        def choose(self, *_args):
            raise AssertionError("Actual caption diagnostic must occur before any Jev selection")

    result, trace = run(backend, planner, policy=NeverPolicy(), singleton_fast_path=False)
    assert desktop.inputs == result.steps == 0 and planner.requests[1]["limit"] == 1
    assert events(trace, "agent_observed_navigation_arrival_diagnostic")
    retired = events(trace, "agent_program_abandoned")[0]
    assert retired["force_semantic_recovery"] is True and retired["semantic_recovery_after"] is True
    assert retired["budget_refunded"] is False and retired["plan_actions"] == 0


def test_auditor_allows_normal_retirement_then_new_H6_but_never_crosses_model_origin():
    fixture = ProgramEvents()
    fixture.runtime(1)
    fixture.complete()
    fixture.advance()
    fixture.runtime(1)
    fixture.add("agent_program_abandoned", {**fixture.common(), "reason": "plan_expired",
        "whole_program_retired": True, "force_semantic_recovery": False,
        "semantic_recovery_before": False, "semantic_recovery_after": False,
        "preserve_original_handoff_flags": True, "budget_refunded": False})
    fixture.call = 2
    fixture.add("agent_model_output_transport", {
        "call": 2, "max_actions": 6, "declaration": {"program_declaration": False}})
    fixture.add("agent_planning_observation", {
        "call": 2, "requested_max_actions": 6, "effective_max_actions": 6})
    fixture.add("agent_plan", {"call": 2, "reusable": True})
    fixture.runtime(1)
    fixture.add("agent_episode", {"steps": 3, "status": "unverified", "duration_ms": 1000.0})
    report = audit_trace(fixture.events)
    assert report["program_metric_status"] == "complete"
    assert report["positive_input_program_continuations"] == 1
    assert report["cross_stage_positive_input_program_continuations"] == 1
    assert len(report["program_epochs"][0]["qualified"]) == 2
    assert report["program_epochs"][0]["status"] == "abandoned"
    assert report["acceptance_satisfied"] is False
