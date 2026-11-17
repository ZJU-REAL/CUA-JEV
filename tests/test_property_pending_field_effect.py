"""Actual selected fields use the existing receipt-bound observation budget."""

from dataclasses import replace

import pytest
from test_deferred_property_confirmation import current_dialog, requested
from test_observed_property_goal import compile
from test_property_preparation import Backend, DialogEnvironment
from test_property_preparation import initial as initial

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.models import Decision


class TransientFieldEnvironment(DialogEnvironment):
    def __init__(self, snapshot, *, transient="missing", reads=0, option=False, toggle_target=None):
        super().__init__(snapshot)
        self.transient, self.remaining = transient, reads
        self.effect_sent = False
        self.pending_reads = 0
        self.option = option
        self.toggle_target = toggle_target

    def temporary(self):
        field = next(c for c in self.snapshot.controls if c.ref == "wrap")
        if self.transient == "missing":
            controls = tuple(c for c in self.snapshot.controls if c.ref != "wrap")
        elif self.transient == "duplicate":
            controls = (*self.snapshot.controls, replace(field, ref="other-field"))
        else:
            controls = tuple(
                c if c.ref != "wrap" else
                replace(c, value="80", selected=False) if self.transient == "wrong_value" else
                replace(c, app="Editor Twin") if self.transient == "other_app" else
                replace(c, window="Preferences Twin") if self.transient == "other_window" else c
                for c in self.snapshot.controls
            )
        return replace(self.snapshot, controls=controls)

    def _get_obs(self):
        if self.effect_sent and not self.closed:
            self.pending_reads += 1
            if self.remaining > 0:
                self.remaining -= 1
                return self.temporary()
        return self.snapshot

    def step(self, action, pause):
        if ((self.option or self.toggle_target is not None) and action["action_type"] == "CLICK"
                and action["parameters"]["y"] < 480):
            self.actions.append(action)
            self.snapshot = replace(self.snapshot, controls=tuple(
                (replace(c, selected=True) if self.option else replace(c, checked=self.toggle_target))
                if c.ref == "wrap" else c for c in self.snapshot.controls
            ))
            self.effect_sent = True
            return self.temporary(), 0, False, {}
        result = super().step(action, pause)
        if action["action_type"] == "TYPING":
            self.effect_sent = True
            return self.temporary(), 0, False, {}
        return result


def run(env, *, data=None, backend_type=Backend, timeout_s=120, max_failures=1,
        plan_ttl_s=300, max_steps=10):
    events = []

    class Planner:
        def plan(self, _goal, current, _history, **_kwargs):
            return compile(data or requested(), current)

    class Policy:
        def choose(self, observation, candidates):
            chosen = next((c for c in candidates
                           if c.capability in {"agent.osworld.fill", "agent.osworld.check",
                                               "agent.osworld.uncheck"} and "Wrap column" in c.description
                           or c.description.startswith("select option Light ")), None)
            chosen = chosen or next(c for c in candidates if c.description.startswith("click OK "))
            return Decision(observation.observation_id, chosen.id,
                            {c.id: float(c is chosen) for c in candidates}, 1, "offline-policy", 0)

    result = AgentRunner(
        backend_type(env, pause=0), Planner(), policy=Policy(), progress=events.append,
        config=AgentConfig(allow_actions=True, max_plans=1, max_failures=max_failures,
                           max_steps=max_steps, timeout_s=timeout_s, plan_ttl_s=plan_ttl_s),
    ).run("Set and confirm the requested property")
    return result, events


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _duration: None)


@pytest.mark.parametrize("reads", [0, 1])
@pytest.mark.parametrize("transient", ["missing", "wrong_value"])
def test_selected_field_reappears_and_confirms_in_the_same_plan(initial, transient, reads):
    env = TransientFieldEnvironment(current_dialog(initial), transient=transient, reads=reads)
    result, events = run(env)
    pending = [e for e in events if e["event"] == "agent_pending_effect"]
    assert len(pending) == reads + 1 and pending[-1]["passed"] is True
    assert all(e["selected_property_effect"] and e["effect_operation"] == "fill" for e in pending)
    assert result.recovery_observations == reads + 1
    assert result.planner_calls == 1 and result.property_edits == 1 and result.steps == 2
    assert env.closed and result.plan_completion_claims == 1 and not result.success
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == ["50"]
    assert any(e["event"] == "agent_property_commit" for e in events)


def test_selected_option_recovery_uses_actual_selected_effect(initial):
    current = current_dialog(initial)
    option = replace(current.controls[1], label="Light", role="cell", tag="button", input_type="",
                     editable=False, value="", selectable=True, selected=False)
    current = replace(current, controls=(current.controls[0], option, *current.controls[2:]))
    env = TransientFieldEnvironment(current, reads=1, option=True)
    result, events = run(env, data={**requested(), "label": "Select the requested Light option",
                                  "edit": {"operation": "set", "value": "Light"}})
    assert result.planner_calls == 1 and result.recovery_observations == 2 and env.closed
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK"]
    assert any(e["event"] == "agent_property_commit" for e in events)


@pytest.mark.parametrize("desired", [True, False])
def test_boolean_field_recovery_uses_the_actual_checked_state(initial, desired):
    current = current_dialog(initial)
    checkbox = replace(current.controls[1], role="checkbox", input_type="checkbox", editable=False,
                       value="", checked=not desired)
    current = replace(current, controls=(current.controls[0], checkbox, *current.controls[2:]))
    env = TransientFieldEnvironment(current, reads=1, toggle_target=desired)
    result, events = run(env, data={**requested(), "label": "Set the requested toggle",
                                  "edit": {"operation": "set", "value": "on" if desired else "off"}})
    assert result.planner_calls == 1 and result.recovery_observations == 2 and env.closed
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK"]
    assert any(e["event"] == "agent_property_commit" for e in events)


@pytest.mark.parametrize("kind", ["false", "unknown"])
def test_late_field_effect_does_not_drop_any_declared_whole_goal_condition(initial, kind):
    data = {**requested(), "goal_conditions": [{"kind": "value_equals", "value": "81",
             "target": {"name": "Wrap column" if kind == "false" else "Unknown field"}}]}
    env = TransientFieldEnvironment(current_dialog(initial))
    result, events = run(env, data=data)
    assert env.closed and any(e["event"] == "agent_property_commit" for e in events)
    assert result.plan_completion_claims == 0 and not result.success


def test_failed_typed_receipt_cannot_enter_property_effect_wait_or_unlock_input(initial):
    class FailedInput(TransientFieldEnvironment):
        def step(self, action, pause):
            result = super().step(action, pause)
            if action["action_type"] == "TYPING":
                raise RuntimeError("offline uncertain input response")
            return result

    env = FailedInput(current_dialog(initial))
    result, events = run(env)
    assert result.recovery_observations == 0 and not env.closed
    assert not any(e["event"] in {"agent_pending_effect", "agent_property_commit"} for e in events)


@pytest.mark.parametrize("transient", ["missing", "wrong_value", "duplicate", "other_app", "other_window"])
def test_unconfirmed_effect_never_unlocks_confirmation_or_more_input(initial, transient):
    env = TransientFieldEnvironment(current_dialog(initial), transient=transient, reads=100)
    result, events = run(env)
    pending = [e for e in events if e["event"] == "agent_pending_effect"]
    assert len(pending) == 2 and not any(e["passed"] for e in pending)
    assert result.planner_calls == 1 and not env.closed and not result.plan_completion_claims
    assert not any(e["event"] in {"agent_property_commit", "agent_property_commit_bound"} for e in events)
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == ["50"]


def test_unavailable_bound_receipt_does_not_infer_a_field_effect_from_the_template(initial):
    class MissingBinding(Backend):
        def bound_property_step(self, _candidate, _receipt):
            return None

    env = TransientFieldEnvironment(current_dialog(initial))
    result, events = run(env, backend_type=MissingBinding)
    assert result.recovery_observations == 0 and not env.closed
    assert not any(e["event"] == "agent_property_commit" for e in events)


def test_field_observation_recovery_does_not_extend_the_original_action_budget(initial):
    env = TransientFieldEnvironment(current_dialog(initial))
    result, events = run(env, max_steps=1)
    assert result.steps == 1 and result.recovery_observations == 1 and result.status == "max_steps"
    assert not env.closed and not result.plan_completion_claims
    assert not any(e["event"] == "agent_property_commit" for e in events)


@pytest.mark.parametrize("deadline", ["episode", "plan"])
def test_late_positive_verification_cannot_authorize_input_past_deadline(initial, monkeypatch, deadline):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    class LateBackend(Backend):
        def verify_pending(self, candidate, receipt):
            verified = super().verify_pending(candidate, receipt)
            assert verified.passed
            clock[0] = 5.0
            return verified

    env = TransientFieldEnvironment(current_dialog(initial))
    result, events = run(env, backend_type=LateBackend, timeout_s=1 if deadline == "episode" else 120,
                         plan_ttl_s=1 if deadline == "plan" else 300, max_failures=3)
    pending, = [e for e in events if e["event"] == "agent_pending_effect"]
    assert pending["deadline_exceeded"] and not pending["passed"]
    assert not env.closed and not result.plan_completion_claims
    if deadline == "episode":
        assert result.status == "timeout"
    assert not any(e["event"] == "agent_property_commit" for e in events)
