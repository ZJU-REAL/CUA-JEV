'Regression tests for test stage program recovery v60.'

import json
from dataclasses import replace

import pytest
from test_deferred_property_confirmation import current_dialog
from test_property_pending_field_effect import TransientFieldEnvironment
from test_property_preparation import Backend
from test_property_preparation import initial as initial
from test_stage_program_contract_v60 import parse_original
from test_stage_program_entry_admission_v60 import GateFields, gated_raw
from test_stage_program_runner_v60 import (
    MemoryFields,
    NumericProgramEnvironment,
    PreferField,
    ProgramPlanner,
    events,
    facts,
    field_member,
    memory_parser,
    numeric_planner,
    run,
)
from test_stale_plan_recovery import Backend as StaleBackend
from test_stale_plan_recovery import Environment as StaleEnvironment

from cua_jev.agent_runner import DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentTarget
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend


def delayed_property(policy=None):
    first = {"id": "set", "tool": "declare_property_stage", "arguments": {
        "label": "Set requested field", "scope": {"app": "Editor", "window": "Preferences"},
        "edit_operation": "set", "edit_value": "50", "numeric_finalize_key": "none",
        "confirmation_label": "Confirm requested property", "completion": "stage"}}
    if policy is not None:
        first["recovery"] = policy
    return {"label": "Set and inspect requested property", "stages": [first,
        {"id": "inspect", "tool": "declare_navigation_stage", "arguments": {
            "label": "Observe the returned main view", "scope": {"app": "Editor"},
            "arrival": [{"kind": "exists", "target": {"name": "Main", "role": "frame", "app": "Editor"}}],
            "completion": "stage"}},
    ]}


class SelectFieldThenOK(DirectPlanPolicy):
    def choose(self, observation, candidates):
        choice = next((candidate for candidate in candidates if candidate.capability == "agent.osworld.fill"
                       and "Wrap column" in candidate.description), None)
        choice = choice or next((candidate for candidate in candidates
                                 if candidate.description.startswith("click OK ")), candidates[0])
        return Decision(observation.observation_id, choice.id,
                        {candidate.id: float(candidate is choice) for candidate in candidates},
                        1.0, "offline-recovery-policy", 0)


def property_run(initial, policy, *, transient_reads=1, **settings):
    environment = TransientFieldEnvironment(
        current_dialog(initial), transient="missing", reads=transient_reads,
    )
    backend = Backend(environment, pause=0)
    done = AgentCondition("exists", AgentTarget(name="Main", role="frame", app="Editor"))
    planner = ProgramPlanner(delayed_property(policy), parse_original, (done,))
    result, trace = run(backend, planner, policy=SelectFieldThenOK(), **settings)
    return result, trace, planner, environment


@pytest.mark.parametrize("policy", [None, "observe"])
def test_default_inherits_original_bounded_property_poll_then_local_program_continues(
    initial, monkeypatch, policy,
):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    result, trace, planner, environment = property_run(initial, policy)
    assert environment.closed and result.steps == 2 and result.planner_calls == 2
    pending = events(trace, "agent_pending_effect")
    assert 1 <= len(pending) <= 2 and pending[-1]["passed"] is True
    assert result.recovery_observations == len(pending)
    assert not events(trace, "agent_program_abandoned")
    gates = [item for item in events(trace, "agent_program_recovery_policy")
             if item["phase"] == "post_input_pending"]
    assert any(item["stage_recovery"] == "observe" and item["original_observe_eligible"] is True
               and item["optional_observe_permitted"] is True
               and item["selected_original_branch"] == "bounded_observe" for item in gates)
    assert all(item["event_observation_calls"] == 0 for item in gates)
    assert [request["review"] for request in planner.requests] == [False, True]


def test_explicit_replan_vetoes_optional_poll_but_retains_actual_failed_fill_cost(initial, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    result, trace, planner, environment = property_run(initial, "replan")
    assert not environment.closed and result.steps == 1 and result.recovery_observations == 0
    assert [action["action_type"] for action in environment.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert planner.requests[1]["limit"] == 1 and planner.activations == []
    assert not events(trace, "agent_pending_effect")
    assert events(trace, "agent_program_abandoned")
    gate = next(item for item in events(trace, "agent_program_recovery_policy")
                if item["phase"] == "post_input_pending" and item["original_observe_eligible"] is True)
    assert gate["stage_recovery"] == "replan" and gate["optional_observe_permitted"] is False
    assert gate["selected_original_branch"] == "original_nonobserve_path"
    receipt = events(trace, "receipt")[0]
    assert receipt["output"]["environment_steps"] > 0
    assert events(trace, "agent_step")[0]["verified"] is False


def test_observe_declaration_cannot_upgrade_original_replan_step_or_grant_a_poll_ticket():
    backend = MemoryFields(("Name", "City", "Gate"))
    raw = {"label": "Original current guards still apply", "stages": [
        {**field_member("first", ("Name",)), "recovery": "observe"}, field_member("second", ("City",)),
    ]}

    def original_replan_step(raw, snapshot, horizon):
        plan = memory_parser(raw, snapshot, horizon)
        return replace(plan, steps=tuple(replace(step, recovery="replan", preconditions=(
            facts("Gate", "ready"),)) for step in plan.steps))

    planner = ProgramPlanner(raw, original_replan_step,
        (facts("Name", "requested-Name"), facts("City", "requested-City")))
    result, trace = run(backend, planner)
    assert backend.executed == [] and result.steps == result.recovery_observations == 0
    assert planner.requests[1]["limit"] == 1 and planner.activations == []
    gates = [item for item in events(trace, "agent_program_recovery_policy")
             if item["phase"] == "awaiting_conditions"]
    assert gates and all(item["original_observe_eligible"] is False
                         and item["optional_observe_permitted"] is False for item in gates)


def test_observe_preserves_original_two_read_cap_and_never_unlocks_next_stage_on_timeout(
    initial, monkeypatch,
):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    result, trace, planner, environment = property_run(initial, "observe", transient_reads=100)
    assert 0 < result.recovery_observations <= 2
    assert len(events(trace, "agent_pending_effect")) == result.recovery_observations
    assert result.steps == 1 and not environment.closed and planner.activations == []
    assert planner.requests[1]["limit"] == 1
    assert not events(trace, "agent_program_stage_completed")


def test_observe_does_not_create_a_new_poll_ticket_for_false_entry_admission():
    backend = GateFields()
    raw = gated_raw()
    raw["stages"][0]["recovery"] = "observe"
    planner = ProgramPlanner(raw, memory_parser,
        (facts("Name", "requested-Name"), facts("City", "requested-City")),
        after_first_response=lambda: backend.values.__setitem__("Gate", "blocked"))
    result, trace = run(backend, planner)
    assert backend.executed == [] and backend.compilations == 0
    assert result.recovery_observations == 0 and planner.requests[1]["limit"] == 1
    assert not events(trace, "agent_pending_effect")
    assert not events(trace, "agent_program_stage_completed")


def test_explicit_replan_does_not_cancel_original_declared_numeric_key_contingency(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    original = numeric_planner()
    raw = json.loads(original.program.declaration_json)
    raw["stages"][0]["recovery"] = "replan"
    planner = ProgramPlanner(raw, parse_original, original.done_conditions)
    environment = NumericProgramEnvironment()
    result, trace = run(OSWorldBackend(environment, pause=0), planner, policy=PreferField())
    assert [action["action_type"] for action in environment.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert result.steps == 2 and result.planner_calls == 2 and len(planner.activations) == 1
    assert [item["verified"] for item in events(trace, "agent_step")] == [False, True]
    assert not events(trace, "agent_program_abandoned")


def test_explicit_replan_does_not_cancel_original_once_stale_preflight_refresh():
    environment, backend = StaleEnvironment(), None
    backend = StaleBackend(environment, stale_at=(1,))
    raw = {"label": "Enable requested options", "stages": [
        {"id": name.lower(), "tool": "declare_field_edits_stage", "recovery": "replan", "arguments": {
            "label": "Enable " + name, "completion": "stage", "edits": [
                {"operation": "check", "target": {"name": name, "role": "checkbox", "app": "Settings"}}]}}
        for name in ("Alpha", "Beta")
    ]}
    done = tuple(AgentCondition("checked", AgentTarget(name=name, app="Settings"))
                 for name in ("Alpha", "Beta"))
    planner = ProgramPlanner(raw, parse_original, done)
    result, trace = run(backend, planner, singleton_fast_path=False)
    assert result.stale_recovery_attempts == 1 and result.steps == 3 and result.planner_calls == 2
    assert len(environment.actions) == 2 and len(planner.activations) == 1
    assert not events(trace, "agent_program_abandoned")
