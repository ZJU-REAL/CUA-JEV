"""Receipt-bound completion, boolean binding and fair immediate planning; offline only."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from test_current_ref_root_admission import snapshot as snapshot
from test_native_toggle_actions import STATE, Desktop, switch
from test_property_preparation import Backend, Choice, Environment, dialog_snapshot, raw
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner, _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.desktop_property import property_field_steps
from cua_jev.desktop_property_protocol import (
    immediate_protocol_instructions,
    property_protocol_instructions,
)
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend


def compile(data, observed):
    plan = parse_plan(data, observed, max_actions=6, profile="desktop",
                      allow_property_preparation=True, allow_property_goal_conditions=True,
                      allow_property_value_binding=True, allow_observed_property_goal=True)
    _validate_desktop_execution_plan(plan, observed, 6)
    return plan


def goal_raw():
    data = raw()
    del data["goal_conditions"]
    return {**data, "completion": "goal"}


def run(env, data, events, *, missing_receipt=False):
    class Planner:
        def plan(self, goal, observed, history, **kwargs):
            return compile(data, observed)

    class CurrentBackend(Backend):
        def bound_property_step(self, candidate, receipt):
            return None if missing_receipt else super().bound_property_step(candidate, receipt)

    return AgentRunner(CurrentBackend(env, pause=0), Planner(), policy=Choice(env),
                       progress=events.append, config=AgentConfig(
                           allow_actions=True, max_plans=1, max_failures=1,
                       )).run("Change the requested column property")


def test_one_plan_prepares_and_edits_then_checks_actual_field_without_invented_future_names(initial):
    data, events = goal_raw(), []
    before = copy.deepcopy(data)
    env = Environment(initial)
    result = run(env, data, events)
    assert data == before and result.planner_calls == 1 and result.plan_reuses == 1
    assert result.plan_completion_claims == 1 and result.status == "unverified" and not result.success
    assert result.candidate_counts[1] > 1
    bound, = [e for e in events if e["event"] == "agent_property_goal_bound"]
    assert bound["conditions"][0]["target"]["name"] == "Wrap column"
    assert bound["receipt_id"] and bound["explicit_conditions_retained"] and not bound["task_success"]
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column", "50",
    ]


def test_native_current_preparation_is_validated_as_one_action_before_the_reusable_template(initial):
    data = goal_raw()
    data["prepare"] = {"operation": "click", "ref": "help"}
    plan = compile(data, initial)
    env = Environment(initial)
    backend = Backend(env, pause=0)
    candidate, = backend.compile(plan.steps[0], initial, initial)
    assert candidate.arguments["bound_control"]["ref"] == "help"
    changed = replace(initial, controls=tuple(c for c in initial.controls if c.ref != "help"))
    env.snapshot = changed
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and not env.actions


def test_numeric_literal_stage_does_not_publish_boolean_preference_to_the_policy(initial):
    env, metadata = Environment(initial), []
    data = goal_raw()
    data["edit"]["operation"] = "set"

    class Planner:
        def plan(self, _goal, observed, _history, **kwargs):
            return compile(data, observed)

    class Policy(Choice):
        def choose(self, observation, candidates):
            metadata.extend(s["property_goal"] for s in observation.state["ready_steps"]
                            if "property_goal" in s)
            return super().choose(observation, candidates)

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Policy(env),
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Set requested column")
    assert result.plan_completion_claims == 1 and result.planner_calls == 1 and metadata
    assert all("bind_boolean_states" not in s for s in metadata)


def test_boolean_literal_stage_still_publishes_typed_state_preference_and_executes_one_input():
    env, metadata = Desktop(), []
    env.after = lambda root: switch(root).set(STATE + "checked", "true")
    backend = OSWorldBackend(env, pause=0)

    class Planner:
        def plan(self, _goal, observed, _history, **kwargs):
            return compile({"intent": "pursue_property", "label": "Enable requested notification setting",
                            "scope": {"app": "gnome-control-center"},
                            "edit": {"operation": "set", "value": "on"}, "completion": "goal"}, observed)

    class Policy:
        def choose(self, observation, candidates):
            metadata.extend(s["property_goal"] for s in observation.state["ready_steps"]
                            if "property_goal" in s)
            chosen = next(c for c in candidates if c.capability == "agent.osworld.check")
            return Decision(observation.observation_id, chosen.id,
                            {c.id: float(c is chosen) for c in candidates}, 1, "offline-choice", 0)

    result = AgentRunner(backend, Planner(), policy=Policy(), config=AgentConfig(
        allow_actions=True, singleton_fast_path=False, max_plans=1,
    )).run("Enable the requested notification setting")
    assert result.plan_completion_claims == 1 and result.planner_calls == 1 and len(env.calls) == 1
    assert metadata and all(s.get("bind_boolean_states") is True for s in metadata)


@pytest.mark.parametrize("case", [
    "missing_receipt", "field_regressed", "explicit_unknown", "dialog", "modal",
])
def test_no_whole_goal_stop_without_receipt_fresh_effect_and_all_explicit_conditions(initial, case):
    class CurrentEnvironment(Environment):
        def _get_obs(self):
            if case == "field_regressed" and any(a["action_type"] == "TYPING" for a in self.actions):
                # Change only after the actual final field was filled/verified.
                if any(c.ref == "wrap" and c.value == "50" for c in self.snapshot.controls):
                    self.snapshot = replace(self.snapshot, controls=tuple(
                        replace(c, value="81") if c.ref == "wrap" else c for c in self.snapshot.controls
                    ))
            return self.snapshot

    data, events = goal_raw(), []
    if case == "explicit_unknown":
        data["goal_conditions"] = [{"kind": "exists", "target": {"name": "Unobserved fact"}}]
    env = CurrentEnvironment(initial)
    if case in {"dialog", "modal"}:
        env.snapshot = dialog_snapshot(initial, role="dialog" if case == "dialog" else "frame", value="80")
        if case == "modal":
            env.snapshot = replace(env.snapshot, controls=tuple(
                replace(c, modal_context=True) if c.ref == "wrap" else c for c in env.snapshot.controls
            ))
        env.opened, env.focus = True, ""
        data.pop("prepare")
    result = run(env, data, events, missing_receipt=case == "missing_receipt")
    assert result.plan_completion_claims == 0 and not result.success
    assert len([a for a in env.actions if a["action_type"] == "TYPING"]) <= 2
    if case in {"dialog", "modal"}:
        assert any(e["event"] == "agent_property_confirmation_required" for e in events)


@pytest.mark.parametrize("value,operation", [("on", "check"), ("off", "uncheck"), ("ON", "check")])
def test_literal_boolean_keeps_text_and_generates_typed_check_candidate(snapshot, value, operation):
    data = {"intent": "pursue_property", "label": "Set requested backup state",
            "scope": {"app": "Editor"}, "edit": {"operation": "set", "value": value},
            "completion": "goal"}
    plan = compile(data, snapshot)
    choices = property_field_steps(plan.steps[0], snapshot, snapshot.controls)
    toggle, = [s for c, s in choices if c.ref == "toggle"]
    text, = [s for c, s in choices if c.ref == "field"]
    assert toggle.operation == operation and toggle.value == ""
    assert text.operation == "fill" and text.value == value
    assert toggle.postconditions[0].kind == ("checked" if operation == "check" else "unchecked")


def test_boolean_binding_does_not_admit_unknown_state_or_radio_uncheck(snapshot):
    data = {"intent": "pursue_property", "label": "Set option off", "scope": {"app": "Editor"},
            "edit": {"operation": "set", "value": "off"}}
    for changed in (replace(snapshot.controls[2], role="radio", input_type="radio", checked=True),
                    replace(snapshot.controls[2], role="menuitem", input_type="", tag="button")):
        current = replace(snapshot, controls=(changed, snapshot.controls[0]))
        plan = compile(data, current)
        assert not property_field_steps(plan.steps[0], current, (changed,))


def test_explicit_wrong_predicate_is_not_repaired_by_observed_completion(snapshot):
    data = {"intent": "pursue_property", "label": "Enable backup", "scope": {"app": "Editor"},
            "edit": {"operation": "set", "value": "on"}, "completion": "goal",
            "goal_conditions": [{"kind": "checked", "target": {"role": "menuitem", "name": "Backup"}}]}
    with pytest.raises(ValueError, match="checkable role"):
        compile(data, snapshot)


@pytest.mark.parametrize("change", ["old", "condition_budget", "private", "invalid_completion"])
def test_completion_extension_stays_opt_in_and_bounded(snapshot, change):
    data = {"intent": "pursue_property", "label": "Rename item", "scope": {"app": "Editor"},
            "edit": {"operation": "set", "value": "draft"}, "completion": "goal"}
    if change == "old":
        with pytest.raises(ValueError):
            parse_plan(data, snapshot, profile="desktop", allow_property_value_binding=True)
        return
    if change == "condition_budget":
        data["goal_conditions"] = [{"kind": "exists", "target": {"name": "Backup"}}] * 7
    elif change == "private":
        data["edit"]["bind_boolean_states"] = True
    else:
        data["completion"] = True
    with pytest.raises(ValueError):
        compile(data, snapshot)


def test_v33_baseline_and_hybrid_recovery_have_identical_immediate_requests(snapshot):
    requests = []

    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "summary": "Use current menu row", "done": False,
            "steps": [{"id": "s1", "operation": "click", "ref": "button"}],
        })}}]})

    history = [{"recovery_required": True, "feedback": "Input effect unconfirmed"}]
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        planner = MultimodalPlanner(base_url="https://example.test/v1", model="fixture",
                                    client=client, profile="desktop", desktop_contract="property_intent_v33")
        first = planner.plan("Create requested document", snapshot, [], max_actions=1)
        second = planner.plan("Create requested document", snapshot, history, max_actions=1)
    assert requests[0]["messages"][0] == requests[1]["messages"][0] and first == second
    instruction = requests[0]["messages"][0]["content"]
    assert instruction.startswith("IMMEDIATE ACTION CONTRACT")
    assert "PENDING NUMERIC EDITS" in instruction and "key_press" in instruction
    assert "preconditions" in instruction and "postconditions" in instruction
    assert not first.reusable and first.steps[0].ref == "button"


@pytest.mark.parametrize("visual_box_space", ["normalized_1000", "image_pixels"])
@pytest.mark.parametrize("limit", [6, 12])
def test_v33_prompts_are_compact_and_do_not_request_guessed_future_labels(visual_box_space, limit):
    prompt = property_protocol_instructions(limit, visual_box_space=visual_box_space)
    assert len(prompt) < 7000
    assert len(immediate_protocol_instructions()) < 3500
    assert "completion=goal" in prompt
