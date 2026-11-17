'Regression tests for test stage program runner v60.'

import copy
from dataclasses import replace

import pytest
from test_declared_numeric_finalization import ConfirmEnvironment
from test_goal_recheck_frame_UNEXECUTED import ProofBackend
from test_navigation_continuation_runner_draft import MenuEnvironment, ObservedBackend, ScriptedPolicy
from test_pending_numeric_edit import STATE, VALUE
from test_readonly_property_observation import image as image
from test_readonly_property_observation import setup as readonly_setup
from test_stage_program_contract_v60 import parse_original
from test_stale_plan_recovery import Backend as StaleBackend
from test_stale_plan_recovery import Environment as StaleEnvironment

from cua_jev import agent_runner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_semantics import conditions_hold, matching_controls
from cua_jev.desktop_stage_program import activate_program_stage, make_program_plan, parse_stage_program
from cua_jev.models import ActionReceipt, Decision, Risk, Verification
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace


def facts(name, value, app="Editor", window="Main"):
    target = AgentTarget(name=name, role="textbox", app=app, window=window)
    return AgentCondition("value_equals", target, value)


def field_member(identifier, names):
    return {"id": identifier, "tool": "declare_field_edits_stage", "arguments": {
        "label": "Fill the requested generic fields", "completion": "stage",
        "edits": [{"operation": "fill", "target": {
            "name": name, "role": "textbox", "app": "Editor", "window": "Main"},
            "value": "requested-" + name} for name in names],
    }}


def memory_parser(raw, snapshot, horizon):
    """Known memory backend parser, not a substitute for the real native tests."""
    assert raw["intent"] == "edit_fields" and horizon == 6
    steps = []
    for index, edit in enumerate(raw["edits"]):
        target = AgentTarget(**edit["target"])
        matches = [control for control in snapshot.controls if control.label == target.name
                   and control.role == target.role and control.app == target.app
                   and control.window == target.window]
        if len(matches) != 1:
            raise ValueError("memory current target missing or ambiguous")
        ready = AgentCondition("ready_to_" + edit["operation"], target, edit.get("value", ""))
        steps.append(AgentStep(f"field-{index}", edit["operation"], target=target,
            value=edit.get("value", ""), label=target.name,
            depends_on=(steps[-1].id,) if steps else (),
            preconditions=(ready,), postconditions=(facts(target.name, edit["value"]), ready),
            recovery="observe"))
    return AgentPlan(raw["label"], tuple(steps), reusable=True)


class MemoryFields(ProofBackend):
    def __init__(self, names=("Name", "City"), *, reveal_after_first=False, raw_macro_inputs=1,
                 failure=None, after_execute=None):
        super().__init__()
        self.names, self.reveal_after_first = names, reveal_after_first
        self.raw_macro_inputs, self.failure, self.after_execute = raw_macro_inputs, failure, after_execute
        self.executed_candidate_ids = []
        self._field_bindings = {}

    def observe(self):
        # Reuse the exact source-bound proof bookkeeping of the existing fixture.
        original = super().observe()
        names = (self.names[:1] if self.reveal_after_first and not self.values.get(self.names[0])
                 else self.names)
        controls = tuple(AgentControl(f"field-{index}", name, "input", role="textbox",
            app="Editor", window="Main", editable=True, active=True,
            value=self.values.get(name, ""), box=(50, 50 + index * 80, 400, 100 + index * 80))
            for index, name in enumerate(names))
        snapshot = replace(original, url="osworld://desktop", title="Main", controls=controls)
        self._last_observed_snapshot = snapshot
        self.frames[-1] = snapshot
        return snapshot

    def compile(self, step, planned, current):
        if step.operation != "fill":
            return super().compile(step, planned, current)
        matches = matching_controls(current, step.target)
        assert len(matches) == 1 and matches[0].editable
        control = matches[0]
        candidates = tuple(replace(candidate, arguments={**candidate.arguments,
            "ref": control.ref, "bound_control": control.to_dict()})
            for candidate in super().compile(step, planned, current))
        for candidate in candidates:
            self._field_bindings[candidate.id] = candidate.to_dict(), control
        return candidates

    def execute(self, candidate, observation_id, decision_id):
        original, control = self._field_bindings.pop(candidate.id)
        assert candidate.to_dict() == original
        assert candidate.arguments["ref"] == control.ref and candidate.description == control.label
        self.fail = self.failure == "receipt" and not self.executed
        receipt = super().execute(candidate, observation_id, decision_id)
        self.executed_candidate_ids.append(candidate.id)
        self.executed[-1] = control.label
        self.environment_steps += self.raw_macro_inputs - 1
        receipt = replace(receipt, output={**receipt.output, "environment_steps": self.raw_macro_inputs})
        if self.after_execute:
            self.after_execute()
        return receipt

    def verify(self, candidate, receipt):
        if self.failure == "verification":
            return Verification(False, "memory unconfirmed effect")
        return super().verify(candidate, receipt)


class ProgramPlanner:
    profile = "desktop"
    desktop_contract = "property_intent_v52"
    desktop_output_format = "semantic_tools_v60"

    def __init__(self, raw, parser, done_conditions, *, after_first_response=None, on_activation=None):
        self.program = parse_stage_program(raw, max_actions=6)
        self.parser, self.done_conditions = parser, done_conditions
        self.calls, self.requests, self.activations = 0, [], []
        self.after_first_response, self.on_activation = after_first_response, on_activation

    def plan(self, goal, snapshot, history, *, max_actions=6, whole_goal_recheck=False, **_options):
        self.calls += 1
        self.requests.append({"snapshot": snapshot, "history": copy.deepcopy(history),
                              "limit": max_actions, "review": whole_goal_recheck})
        if self.calls == 1:
            wrapped = make_program_plan(self.program, snapshot, parse_stage=self.parser)
            if self.after_first_response:
                self.after_first_response()
            return wrapped
        return AgentPlan("Fresh whole-goal observation", (), done=True, reusable=True,
                         completion_conditions=self.done_conditions)

    def activate_program_stage(self, program, index, snapshot, *, remaining_actions):
        self.activations.append({"index": index, "snapshot": snapshot, "remaining": remaining_actions})
        if self.on_activation:
            self.on_activation()
        return activate_program_stage(program, index, snapshot, remaining_actions=remaining_actions,
                                      parse_stage=self.parser)


def run(backend, planner, *, policy=None, **settings):
    trace = JsonlTrace()
    config = {"allow_actions": True, "whole_goal_recheck": True, "plan_horizon": 6,
              "max_plans": 2, "max_failures": 3, "max_steps": 20, **settings}
    result = AgentRunner(backend, planner, policy=policy or DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(**config)).run("Complete all declared generic desktop work")
    return result, trace


def events(trace, kind):
    return [event["payload"] for event in trace.events if event["kind"] == kind]


def two_fields(backend, **options):
    raw = {"label": "Reach both requested field states", "stages": [
        field_member("first", ("Name",)), field_member("later", ("City",)),
    ]}
    return ProgramPlanner(raw, memory_parser, (facts("Name", "requested-Name"),
                                               facts("City", "requested-City")), **options)


def test_future_UI_activates_from_actual_new_frame_without_intermediate_model_request():
    backend = MemoryFields(reveal_after_first=True)
    planner = two_fields(backend)
    result, trace = run(backend, planner)
    assert backend.executed == ["Name", "City"]
    assert result.planner_calls == planner.calls == 2  # Initial program + original whole-goal review.
    assert [request["review"] for request in planner.requests] == [False, True]
    assert len(planner.activations) == 1 and planner.activations[0]["remaining"] == 5
    request_frame = planner.requests[0]["snapshot"]
    active_frame = planner.activations[0]["snapshot"]
    assert active_frame is not request_frame
    assert "City" not in [control.label for control in request_frame.controls]
    assert "City" in [control.label for control in active_frame.controls]
    completed = events(trace, "agent_program_stage_completed")
    assert [item["stage_index"] for item in completed] == [0, 1]
    assert [item["plan_actions"] for item in completed] == [1, 2]
    assert len({item["original_plan_at"] for item in completed}) == 1
    assert events(trace, "agent_program_completed")[0]["whole_goal_review_still_required"] is True
    assert not result.success  # No independent evaluator is invented by local stage completion.
    origin = events(trace, "agent_program_declared")[0]["origin_planner_call"]
    assert not any(item["call"] == origin for item in events(trace, "agent_plan"))


def test_delayed_initial_response_uses_original_fresh_path_before_readonly_advance(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = MemoryFields(reveal_after_first=True)

    def model_delay_and_changed_UI():
        now[0] += 1.1
        backend.values["Name"] = "requested-Name"

    planner = two_fields(backend, after_first_response=model_delay_and_changed_UI)
    result, trace = run(backend, planner)
    assert backend.executed == ["City"] and result.planner_calls == 2
    assert planner.activations[0]["snapshot"] is not planner.requests[0]["snapshot"]
    first = events(trace, "agent_program_stage_completed")[0]
    assert first["reason"] == "read_only_predicates" and first["receipt_ids"] == []
    assert first["plan_actions"] == 0 and first["positive_input"] is False
    assert first["source"] == "backend.observe"


def test_fresh_predicate_only_stages_send_no_GUI_and_get_no_receipt_or_positive_credit():
    backend = MemoryFields()
    backend.values = {name: "requested-" + name for name in backend.names}
    planner = two_fields(backend)
    result, trace = run(backend, planner)
    assert backend.executed == [] and result.steps == 0 and result.planner_calls == 2
    completed = events(trace, "agent_program_stage_completed")
    assert len(completed) == 2
    assert all(item["reason"] == "read_only_predicates" and item["receipt_ids"] == []
               and item["positive_input"] is False and item["plan_actions"] == 0 for item in completed)


@pytest.mark.parametrize("proof_drift", ["age", "fingerprint", "image"])
def test_predicate_truth_without_actual_current_full_proof_cannot_advance(proof_drift):
    class UntrustedProof(MemoryFields):
        def current_full_observation_proof(self, snapshot, *, expected=None):
            proof = super().current_full_observation_proof(snapshot, expected=expected)
            if proof is None:
                return None
            if proof_drift == "age":
                return (proof[0] - 1.1, *proof[1:])
            index = 3 if proof_drift == "fingerprint" else 4
            return tuple("0" * 64 if i == index else value for i, value in enumerate(proof))

    backend = UntrustedProof()
    backend.values = {name: "requested-" + name for name in backend.names}
    planner = two_fields(backend)
    _result, trace = run(backend, planner)
    assert backend.executed == [] and planner.activations == []
    assert not events(trace, "agent_program_stage_completed")
    assert events(trace, "agent_program_abandoned")
    assert planner.requests[1]["limit"] == 1


@pytest.mark.parametrize("failure", ["receipt", "verification"])
def test_failed_or_unknown_actual_effect_abandons_all_future_stages_and_requests_H1(failure):
    backend = MemoryFields(failure=failure)
    planner = two_fields(backend)
    _result, trace = run(backend, planner)
    assert backend.executed == ["Name"] and planner.activations == []
    assert planner.requests[1]["limit"] == 1
    assert not events(trace, "agent_program_stage_completed")
    assert events(trace, "agent_program_abandoned")[0]["whole_program_retired"] is True


def test_seventh_logical_attempt_is_blocked_across_stage_even_when_macros_have_three_raw_inputs():
    names = tuple(f"Field{index}" for index in range(7))
    backend = MemoryFields(names, raw_macro_inputs=3)
    raw = {"label": "Requested field values", "stages": [field_member("batch-one", names[:5]),
                                                         field_member("batch-two", names[5:])]}
    planner = ProgramPlanner(raw, memory_parser, tuple(facts(name, "requested-" + name) for name in names))
    result, trace = run(backend, planner)
    assert backend.executed == list(names[:6]) and backend.environment_steps == 18
    assert result.steps == 6
    assert planner.activations[0]["remaining"] == 1
    assert not any(item["stage_index"] == 1 for item in events(trace, "agent_program_stage_completed"))
    assert events(trace, "agent_program_abandoned")


def test_stage_activation_does_not_renew_original_TTL_before_next_input(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: now[0])
    backend = MemoryFields(after_execute=lambda: now.__setitem__(0, now[0] + 9.9))
    planner = two_fields(backend, on_activation=lambda: now.__setitem__(0, now[0] + 0.2))
    _result, trace = run(backend, planner, plan_ttl_s=10, timeout_s=300)
    assert backend.executed == ["Name"]
    activated = events(trace, "agent_program_stage_activated")
    assert all(item["original_plan_at"] == 10.0 for item in activated)
    assert events(trace, "agent_program_abandoned")


class NumericProgramEnvironment(ConfirmEnvironment):
    """Reuse original native numeric protocol with generic application labels."""
    def __init__(self, *, bad=None):
        super().__init__("enter")
        next(self.root.iter("application")).set("name", "NumberTool")
        self.node.set("name", "Requested numeric limit")
        self.bad = bad
        if bad == "wrong_buffer":
            self.on_typing = lambda: setattr(self.node, "text", "200")
        if bad == "unknown_focus":
            self.on_typing = lambda: self.node.set(STATE + "focused", "false")

    def step(self, action, pause):
        if self.bad == "key_uncommitted" and action["action_type"] == "PRESS":
            self.actions.append(action)
            return self._get_obs(), 0, False, {}
        return super().step(action, pause)


def numeric_planner():
    target = {"name": "Requested numeric limit", "role": "spinbutton",
              "app": "NumberTool", "window": "Preferences"}
    committed = AgentCondition("value_equals", AgentTarget(**target), "100")
    raw = {"label": "Commit and inspect the requested numeric property", "stages": [
        {"id": "numeric", "tool": "declare_property_stage", "arguments": {
            "label": "Set the requested numeric limit",
            "scope": {"app": "NumberTool", "window": "Preferences"},
            "edit_operation": "set", "edit_value": "100", "numeric_finalize_key": "enter",
            "confirmation_label": "none", "completion": "stage"}},
        {"id": "inspect", "tool": "declare_navigation_stage", "entry_conditions": [
            {"kind": "value_equals", "target": target, "value": "100"}], "arguments": {
            "label": "Observe the committed numeric property", "scope": {"app": "NumberTool"},
            "arrival": [{"kind": "value_equals", "target": target, "value": "100"}], "completion": "stage"}},
    ]}
    return ProgramPlanner(raw, parse_original, (committed,))


class PreferField(DirectPlanPolicy):
    def choose(self, observation, candidates):
        selected = next((candidate for candidate in candidates
                         if candidate.capability == "agent.osworld.fill"), candidates[0])
        return Decision(observation.observation_id, selected.id,
                        {candidate.id: float(candidate is selected) for candidate in candidates},
                        1.0, "offline-policy", 0)


def test_pending_numeric_fill_keeps_current_program_until_declared_key_really_commits(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    environment, planner = NumericProgramEnvironment(), numeric_planner()
    backend = OSWorldBackend(environment, pause=0)
    result, trace = run(backend, planner, policy=PreferField(), reuse_verified_post_snapshot=True)
    assert [action["action_type"] for action in environment.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert environment.node.get(VALUE + "value") == "100.0"
    assert result.steps == 2 and result.planner_calls == 2
    runtime = events(trace, "agent_step")
    assert [item["verified"] for item in runtime] == [False, True]
    completed = events(trace, "agent_program_stage_completed")
    assert [item["stage_index"] for item in completed] == [0, 1]
    assert completed[0]["plan_actions"] == 2
    assert len(planner.activations) == 1
    assert planner.activations[0]["snapshot"].controls
    assert not events(trace, "agent_program_abandoned")


@pytest.mark.parametrize("bad", ["wrong_buffer", "unknown_focus", "key_uncommitted"])
def test_unknown_numeric_buffer_or_failed_declared_key_never_unlocks_next_stage(monkeypatch, bad):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    environment, planner = NumericProgramEnvironment(bad=bad), numeric_planner()
    _result, trace = run(OSWorldBackend(environment, pause=0), planner, policy=PreferField())
    assert planner.activations == [] and planner.requests[1]["limit"] == 1
    assert not events(trace, "agent_program_stage_completed")
    assert events(trace, "agent_program_abandoned")
    if bad != "key_uncommitted":
        assert all(action["action_type"] != "PRESS" for action in environment.actions)


def test_actual_readonly_receipts_keep_original_attempt_charge_without_GUI_or_reserved_budget(image):
    _observed, environment, backend, _candidate = readonly_setup(image, kind="numeric", value="100")
    raw = {"label": "Recheck requested numeric properties", "stages": [
        {"id": f"property-{index}", "tool": "declare_property_stage", "arguments": {
            "label": f"Requested property {index}", "scope": {"app": "Editor"},
            "edit_operation": "set", "edit_value": "100", "numeric_finalize_key": "enter",
            "confirmation_label": "none", "completion": "stage"}} for index in range(4)
    ]}
    condition = AgentCondition("value_equals", AgentTarget(name="Requested limit", app="Editor"), "100")
    planner = ProgramPlanner(raw, parse_original, (condition,))
    assert sum(stage.declared_slots for stage in planner.program.stages) > 6

    class ReadonlyChoice(DirectPlanPolicy):
        def choose(self, observation, candidates):
            selected = next(candidate for candidate in candidates if candidate.risk == Risk.READ_ONLY)
            return Decision(observation.observation_id, selected.id,
                            {candidate.id: float(candidate is selected) for candidate in candidates},
                            1.0, "offline-readonly", 0)

    result, trace = run(backend, planner, policy=ReadonlyChoice())
    assert environment.actions == [] and backend.environment_steps == 0
    assert result.steps == 4 and result.planner_calls == 2
    completed = events(trace, "agent_program_stage_completed")
    assert len(completed) == 4 and [item["plan_actions"] for item in completed] == [1, 2, 3, 4]
    assert all(item["reason"] == "verified_receipt" and item["receipt_ids"]
               and item["read_only"] is True and item["positive_input"] is False for item in completed)


class UnknownExitDialog(MemoryFields):
    """New editable control does not prove the preceding named-label arrival."""
    def __init__(self):
        self.dialog_open = False
        super().__init__()

    def observe(self):
        original = super().observe()
        control = (AgentControl("unnamed-editor", "", "input", role="textbox", app="Shell",
                                editable=True, active=True, box=(20, 20, 300, 80)) if self.dialog_open else
                   AgentControl("open", "Open requested view", "button", role="button", app="Shell",
                                active=True, box=(20, 20, 300, 80)))
        snapshot = replace(original, title="Dialog" if self.dialog_open else "Main", controls=(control,))
        self._last_observed_snapshot = snapshot
        self.frames[-1] = snapshot
        return snapshot

    def execute(self, candidate, observation_id, decision_id):
        self.executed.append(candidate.id)
        self.environment_steps += 1
        self.dialog_open = True
        return ActionReceipt(observation_id, decision_id, candidate.id, candidate.channel,
            candidate.capability, True, 0, 0.001,
            output={"environment_steps": 1, "postconditions_met": False})

    def verify(self, candidate, receipt):
        return Verification(False, "current named-label arrival remains unknown",
                            {"postconditions_met": False})


def test_next_textbox_entry_true_cannot_override_current_named_label_arrival_unknown():
    arrival = AgentCondition("exists", AgentTarget(name="Action label", role="text", app="Shell"))
    entry = AgentCondition("exists", AgentTarget(role="textbox", app="Shell"))
    raw = {"label": "Open and then edit the requested view", "stages": [
        {"id": "open", "tool": "declare_navigation_stage", "arguments": {
            "label": "Open requested view", "scope": {"app": "Shell"},
            "arrival": [{"kind": "exists", "target": {
                "name": "Action label", "role": "text", "app": "Shell", "window": ""}}],
            "completion": "stage"}},
        {"id": "edit", "tool": "declare_field_edits_stage", "entry_conditions": [
            {"kind": "exists", "target": {"role": "textbox", "app": "Shell"}}], "arguments": {
            "label": "Edit the now available field", "completion": "stage",
            "edits": [{"operation": "fill", "target": {"role": "textbox", "app": "Shell"},
                       "value": "requested literal"}]}},
    ]}

    def first_parser(raw, snapshot, horizon):
        assert raw["intent"] == "navigate"  # The second parser must never be called.
        step = AgentStep("open", "click", ref="open", label="Open requested view", navigation=True,
                         target=AgentTarget(app="Shell"), postconditions=(arrival,))
        return AgentPlan("Open requested view", (step,), reusable=True)

    backend = UnknownExitDialog()
    planner = ProgramPlanner(raw, first_parser, (arrival,))
    _result, trace = run(backend, planner)
    current = backend.observe()
    assert conditions_hold((entry,), current) and not conditions_hold((arrival,), current)
    assert len(backend.executed) == 1 and planner.activations == []
    assert planner.requests[1]["limit"] == 1
    assert not events(trace, "agent_program_stage_completed")
    assert events(trace, "agent_program_abandoned")


def test_stage_boundary_does_not_rearm_known_cycle_concession_or_refund_retained_attempt(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)
    environment = MenuEnvironment()
    backend = ObservedBackend(environment)
    arrival_target = {"name": "Arrival", "app": "Demo", "window": "Main"}
    final_target = {"name": "Final arrival", "app": "Demo", "window": "Main"}
    raw = {"label": "Reach two declared destinations", "stages": [
        {"id": "first", "tool": "declare_navigation_stage", "arguments": {
            "label": "Reach first destination", "scope": {"app": "Demo", "window": "Main"},
            "arrival": [{"kind": "exists", "target": arrival_target}], "completion": "stage"}},
        {"id": "second", "tool": "declare_navigation_stage", "entry_conditions": [
            {"kind": "exists", "target": arrival_target}], "arguments": {
            "label": "Reach second destination", "scope": {"app": "Demo", "window": "Main"},
            "arrival": [{"kind": "exists", "target": final_target}], "completion": "stage"}},
    ]}
    planner = ProgramPlanner(raw, parse_original, (AgentCondition("exists", AgentTarget(**final_target)),))
    policy = ScriptedPolicy(("Parent", "Parent", "Destination", "Parent", "Parent", "Destination"))
    _result, trace = run(backend, planner, policy=policy, navigation_continuation=True,
                         reuse_verified_post_snapshot=False, reuse_verified_pending_snapshot=False,
                         singleton_fast_path=False)
    assert len(planner.activations) == 1
    assert planner.activations[0]["remaining"] == 3  # true,false-retained,true all charged.
    transitions = events(trace, "agent_program_stage_activated")
    assert transitions[1]["plan_actions"] == 3
    assert len(environment.inputs) <= 5
    assert sum(action["parameters"]["y"] >= 100 for action in environment.inputs) == 1
    assert all(observation.state["navigation_memory"]["known_repeat_used"] is True
               for observation, _candidates in policy.requests[3:])
    assert not any(item["stage_index"] == 1 for item in events(trace, "agent_program_stage_completed"))
    assert events(trace, "agent_program_abandoned")


@pytest.mark.parametrize("stale_at", [(1,), (1, 3)])
def test_original_bounded_stale_refresh_stays_same_stage_and_is_not_rearmed(stale_at):
    environment = StaleEnvironment()
    backend = StaleBackend(environment, stale_at=stale_at)
    raw = {"label": "Enable the requested independent options", "stages": [
        {"id": name.lower(), "tool": "declare_field_edits_stage", "arguments": {
            "label": "Enable requested option " + name, "completion": "stage",
            "edits": [{"operation": "check", "target": {
                "name": name, "role": "checkbox", "app": "Settings"}}]}}
        for name in ("Alpha", "Beta")
    ]}
    conditions = tuple(AgentCondition("checked", AgentTarget(name=name, app="Settings"))
                       for name in ("Alpha", "Beta"))
    planner = ProgramPlanner(raw, parse_original, conditions)
    result, trace = run(backend, planner, singleton_fast_path=False)
    assert result.stale_recovery_attempts == 1
    assert len(planner.activations) == 1
    first, retry = backend.executed_candidates[:2]
    assert first.id != retry.id
    assert first.arguments["bound_control"]["box"] != retry.arguments["bound_control"]["box"]
    steps = events(trace, "agent_step")
    assert steps[0]["verified"] is False
    if len(stale_at) == 1:
        assert len(environment.actions) == 2 and result.steps == 3
        assert result.planner_calls == 2 and planner.requests[-1]["review"] is True
        assert not events(trace, "agent_program_abandoned")
        assert len(events(trace, "agent_program_stage_completed")) == 2
    else:
        assert len(environment.actions) == 1
        assert planner.requests[1]["limit"] == 1
        assert events(trace, "agent_program_abandoned")
        assert not any(item["stage_index"] == 1 for item in events(trace, "agent_program_stage_completed"))
