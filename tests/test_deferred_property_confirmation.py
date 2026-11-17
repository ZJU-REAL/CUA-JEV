"""One declared property and confirmation, with real backend guards; entirely offline."""

from dataclasses import replace

import pytest
from test_observed_property_goal import compile, goal_raw
from test_property_preparation import Backend, DialogEnvironment, dialog_snapshot
from test_property_preparation import initial as initial

from cua_jev.agent_planner import parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.desktop_property_confirmation import property_confirmation_steps
from cua_jev.errors import PolicyError
from cua_jev.models import Decision


def current_dialog(initial, *, value="80", cancel=True, owner_role="dialog"):
    current = dialog_snapshot(initial, value=value, role=owner_role)
    controls = tuple(replace(c, ancestors=(f"{owner_role}: Preferences",)) if c.ref != "dialog" else c
                     for c in current.controls)
    if owner_role == "frame":
        controls = tuple(replace(c, modal_context=True, modal=c.ref == "dialog", modal_owner_ref="dialog")
                         for c in controls)
    if cancel:
        controls += (replace(controls[-1], ref="cancel", label="Cancel", box=(600, 800, 700, 900)),)
    return replace(current, controls=controls)


def requested(*, prepare=False):
    raw = goal_raw()
    if not prepare:
        raw.pop("prepare")
    return {**raw, "confirm": {"label": "Confirm the requested property change"}}


def selected_field():
    target = AgentTarget(name="Wrap column", role="textbox", app="Editor", window="Preferences")
    return AgentStep("property", "fill", target=target, value="50", postconditions=(
        AgentCondition("value_equals", target, "50"), AgentCondition("ready_to_fill", target, "50"),
    ))


def run(env, *, data=None, missing_receipt=False, max_steps=10, confidence=1.0):
    events, decisions = [], []

    class Planner:
        def plan(self, _goal, observed, _history, **_kwargs):
            return compile(data if data is not None else requested(), observed)

    class CurrentBackend(Backend):
        def bound_property_step(self, candidate, receipt):
            return None if missing_receipt else super().bound_property_step(candidate, receipt)

    class Policy:
        def choose(self, observation, candidates):
            field = next((c for c in candidates if c.capability == "agent.osworld.fill"
                          and "Wrap column" in c.description
                          or c.description.startswith("select option Light ")), None)
            selected = field or next((c for c in candidates
                                      if c.description.startswith("click OK ")),
                                     candidates[0])
            explicit = observation.state.get("explicit_property_confirmation") is True
            decisions.append({"explicit_confirmation": explicit, "candidates": len(candidates)})
            if explicit and confidence == 0:
                raise PolicyError("offline confirmation selection failed")
            return Decision(observation.observation_id, selected.id,
                            {c.id: float(c is selected) for c in candidates},
                            confidence if explicit else 1.0, "offline-policy", 0)

    config = AgentConfig(allow_actions=True, max_plans=1, max_failures=1, max_steps=max_steps)
    result = AgentRunner(CurrentBackend(env, pause=0), Planner(), policy=Policy(), progress=events.append,
                         config=config).run("Set and confirm the requested property")
    return result, events, decisions


@pytest.mark.parametrize("max_steps", [2, 10])
@pytest.mark.parametrize("owner_role", ["dialog", "frame"])
def test_one_plan_fills_then_selects_real_buttons_and_freshly_stops(initial, max_steps, owner_role):
    env = DialogEnvironment(current_dialog(initial, owner_role=owner_role))
    result, events, decisions = run(env, max_steps=max_steps)
    assert env.closed and result.planner_calls == 1 and result.plan_reuses == 1
    assert not result.success
    if max_steps == 2:
        assert result.status == "max_steps" and result.plan_completion_claims == 0
    else:
        assert result.status == "unverified" and result.plan_completion_claims == 1
    assert result.property_edits == 1 and result.semantic_steps == 2 and result.steps == 2
    assert decisions[-1] == {"explicit_confirmation": True, "candidates": 2}
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == ["50"]
    assert sum(e["event"] == "agent_property_commit" for e in events) == 1
    bound, = [e for e in events if e["event"] == "agent_property_goal_bound"]
    assert bound["step"] == "confirm" and bound["conditions"][0]["kind"] == "absent"
    assert bound["receipt_id"] and bound["explicit_conditions_retained"] and not bound["task_success"]
    assert not any(e["event"] == "agent_property_confirmation_required" for e in events)


def test_an_already_correct_field_needs_no_retyping_before_its_explicit_confirmation(initial):
    env = DialogEnvironment(current_dialog(initial, value="50"))
    result, events, _decisions = run(env)
    assert env.closed and result.planner_calls == 1 and result.plan_completion_claims == 1
    assert not any(a["action_type"] == "TYPING" for a in env.actions)
    assert any(e["event"] == "agent_property_commit_bound" for e in events)


def test_selected_literal_option_can_be_confirmed_in_the_same_plan_without_text_input(initial):
    current = current_dialog(initial)
    option = replace(current.controls[1], label="Light", role="cell", tag="button", input_type="",
                     editable=False, value="", selectable=True, selected=False)
    current = replace(current, controls=(current.controls[0], option, *current.controls[2:]))

    class Environment(DialogEnvironment):
        def step(self, action, pause):
            if action["action_type"] == "CLICK" and action["parameters"]["y"] < 480:
                self.actions.append(action)
                self.snapshot = replace(self.snapshot, controls=tuple(
                    replace(c, selected=True) if c.label == "Light" else c for c in self.snapshot.controls
                ))
                return self.snapshot, 0, False, {}
            return super().step(action, pause)

    data = {**requested(), "label": "Choose the requested Light option",
            "edit": {"operation": "set", "value": "Light"}}
    env = Environment(current)
    result, events, _ = run(env, data=data)
    assert env.closed and result.planner_calls == 1 and result.plan_reuses == 1
    assert result.plan_completion_claims == 1 and not result.success
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK"]
    assert any(e["event"] == "agent_property_commit" for e in events)


def test_dialog_closure_must_still_hold_in_a_fresh_observation_before_stopping(initial):
    class Environment(DialogEnvironment):
        def _get_obs(self):
            if self.closed:
                self.snapshot = current_dialog(initial, value="50")
            return self.snapshot

    env = Environment(current_dialog(initial))
    result, events, _ = run(env)
    assert env.closed and any(e["event"] == "agent_property_commit" for e in events)
    assert result.plan_completion_claims == 0 and not result.success


@pytest.mark.parametrize("issue", ["missing_receipt", "field_regressed", "unknown_goal", "false_goal"])
def test_no_unverified_field_commit_or_dropped_explicit_goal(initial, issue):
    class Environment(DialogEnvironment):
        def _get_obs(self):
            if issue == "field_regressed" and any(a["action_type"] == "TYPING" for a in self.actions):
                self.snapshot = replace(self.snapshot, controls=tuple(
                    replace(c, value="81") if c.ref == "wrap" else c for c in self.snapshot.controls
                ))
            return self.snapshot

    env, data = Environment(current_dialog(initial)), requested()
    if issue in {"unknown_goal", "false_goal"}:
        data["goal_conditions"] = [{"kind": "exists", "target": {"name": "Unobserved fact"}}]
        if issue == "false_goal":
            data["goal_conditions"] = [{"kind": "value_equals", "target": {"name": "Wrap column"},
                                        "value": "81"}]
    result, events, _ = run(env, data=data, missing_receipt=issue == "missing_receipt")
    assert result.plan_completion_claims == 0 and not result.success
    if issue in {"missing_receipt", "field_regressed"}:
        assert not env.closed and not any(e["event"] == "agent_property_commit" for e in events)
    else:
        assert env.closed  # A legal confirmation does not repair the false/unknown whole-goal claim.
        bound, = [e for e in events if e["event"] == "agent_property_goal_bound"]
        assert bound["explicit_conditions_retained"]


@pytest.mark.parametrize("single_button", [False, True])
def test_even_a_single_confirmation_requires_a_policy_decision_before_button_input(initial, single_button):
    env = DialogEnvironment(current_dialog(initial, cancel=not single_button))
    result, events, decisions = run(env, confidence=0)
    assert not env.closed and not result.success
    assert decisions[-1] == {"explicit_confirmation": True, "candidates": 1 if single_button else 2}
    assert not any(e["event"] == "agent_property_commit" for e in events)


def test_original_action_budget_still_stops_before_confirmation(initial):
    env = DialogEnvironment(current_dialog(initial))
    result, events, _ = run(env, max_steps=1)
    assert result.steps == 1 and not env.closed
    assert not any(e["event"] == "agent_property_commit" for e in events)


def test_compiler_template_cannot_be_executed_directly_and_preparation_remains_current(initial):
    current = current_dialog(initial, value="50")
    plan = compile(requested(), current)
    assert plan.steps[-1].operation == "confirm_property" and plan.steps[-1].target is None
    backend = Backend(DialogEnvironment(current), pause=0)
    assert backend.compile(plan.steps[-1], current, current) == ()
    data = requested(prepare=True)
    prepared = compile(data, initial)
    assert [s.id for s in prepared.steps] == ["prepare", "property", "confirm"]
    with pytest.raises(ValueError, match="logical step budget"):
        parse_plan(data, initial, max_actions=2, profile="desktop", allow_property_preparation=True,
                   allow_observed_property_goal=True, allow_property_goal_conditions=True)


@pytest.mark.parametrize("issue", [
    "incomplete", "field_missing", "duplicate_field", "inactive_dialog", "disabled_dialog",
    "foreign_dialog", "field_ancestry", "no_dialog_role", "static_buttons",
    "anonymous_dialog", "duplicate_dialog",
    "modal_field_without_modal_owner",
    "window_prefix", "app_prefix",
])
def test_only_complete_unique_actual_field_and_owning_dialog_can_authorize_candidates(initial, issue):
    current = current_dialog(initial, value="50")
    template = compile(requested(), current).steps[-1]
    controls = list(current.controls)
    if issue == "incomplete":
        current = replace(current, a11y_complete=False)
    elif issue == "field_missing":
        controls = [c for c in controls if c.ref != "wrap"]
    elif issue == "duplicate_field":
        controls.append(replace(controls[1], ref="duplicate", value="81"))
    elif issue == "inactive_dialog":
        controls[0] = replace(controls[0], active=False)
    elif issue == "disabled_dialog":
        controls[0] = replace(controls[0], disabled=True)
    elif issue == "foreign_dialog":
        controls[0] = replace(controls[0], app="Foreign")
    elif issue == "field_ancestry":
        controls[1] = replace(controls[1], ancestors=())
    elif issue == "no_dialog_role":
        controls[0] = replace(controls[0], role="frame")
    elif issue == "anonymous_dialog":
        controls[0] = replace(controls[0], label="")
    elif issue == "duplicate_dialog":
        controls.append(replace(controls[0], ref="other_dialog"))
    elif issue == "modal_field_without_modal_owner":
        controls = [replace(c, role="frame", modal_context=False) if c.ref == "dialog"
                    else replace(c, modal_context=True, ancestors=("frame: Preferences",)) for c in controls]
    elif issue == "window_prefix":
        controls = [replace(c, window="Preferences Other") for c in controls]
    elif issue == "app_prefix":
        controls = [replace(c, app="Editor Other") for c in controls]
    elif issue == "static_buttons":
        controls = [replace(c, role="label", tag="text") if c.role == "button" else c for c in controls]
    current = replace(current, controls=tuple(controls))
    assert property_confirmation_steps(template, selected_field(), current) == ()


def test_foreign_and_outside_buttons_are_not_added_and_normal_preflight_blocks_later_regression(initial):
    current = current_dialog(initial, value="50")
    controls = current.controls + (
        replace(current.controls[-1], ref="outside", label="Outside", ancestors=()),
        replace(current.controls[-1], ref="foreign", label="Foreign", app="Foreign"),
    )
    current = replace(current, controls=controls)
    template = compile(requested(), current).steps[-1]
    actual = property_confirmation_steps(template, selected_field(), current)
    assert {s.target.name for s in actual} == {"OK", "Cancel"}
    env = DialogEnvironment(current)
    backend = Backend(env, pause=0)
    step = next(s for s in actual if s.target.name == "OK")
    candidate, = backend.compile(step, current, current)
    env.snapshot = replace(current, controls=tuple(
        replace(c, value="81") if c.ref == "wrap" else c for c in current.controls
    ))
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and receipt.output["environment_steps"] == 0 and not env.actions


@pytest.mark.parametrize("count", [255, 256])
def test_confirmation_candidate_budget_never_silently_truncates_actual_buttons(initial, count):
    current = current_dialog(initial, value="50")
    template = compile(requested(), current).steps[-1]
    buttons = tuple(replace(current.controls[2], ref=f"b{i}", label=f"Action {i}",
                            box=((i % 16) * 60, 50 + (i // 16) * 50,
                                 (i % 16) * 60 + 40, 80 + (i // 16) * 50)) for i in range(count))
    current = replace(current, controls=(*current.controls[:2], *buttons))
    choices = property_confirmation_steps(template, selected_field(), current)
    assert len(choices) == (255 if count == 255 else 0)


@pytest.mark.parametrize("confirmation", [True, {}, {"label": ""}, {"label": "x", "target": {"name": "OK"}}])
def test_confirmation_is_an_explicit_bounded_label_not_an_inferred_button(initial, confirmation):
    with pytest.raises(ValueError):
        compile({**requested(), "confirm": confirmation}, current_dialog(initial))


def test_legacy_protocol_and_model_authored_internal_marker_cannot_enable_confirmation(initial):
    with pytest.raises(ValueError):
        parse_plan(requested(), current_dialog(initial), max_actions=6, profile="desktop",
                   allow_property_value_binding=True)
    with pytest.raises(ValueError):
        parse_plan({"summary": "private", "done": False, "steps": [{
            "id": "confirm", "operation": "click", "ref": "help", "property_confirmation_of": "property",
        }]}, initial, profile="desktop")
