"""Opt-in satisfied literal proof; exercise original compiler, bindings and runner."""

from dataclasses import replace

import pytest
from test_deferred_property_confirmation import current_dialog, requested
from test_desktop_property_stage import Backend, control, snapshot
from test_desktop_property_stage import image as image
from test_observed_property_goal import compile
from test_property_preparation import DialogEnvironment
from test_property_preparation import initial as initial
from test_readonly_property_observation import CountingDesktop, data, plan

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.models import Decision, Risk


def observed_option(image, *, selected=True, extra=()):
    option = replace(control("light", "Light"), role="cell", selectable=True, selected=selected)
    return snapshot(image, (option, control("help", "Help", y=300), *extra))


def compile_choices(observed, *, enabled=True, raw=None):
    env = CountingDesktop(observed)
    backend = Backend(env, pause=0, satisfied_property_readonly_fast_path=enabled)
    root = plan(observed, raw or data(value="Light"))
    return backend.compile(root.steps[0], observed, observed), env, backend


def test_opt_in_reuses_original_readonly_candidate_and_default_pool_unchanged(image):
    observed = observed_option(image)
    original, _, default_backend = compile_choices(observed, enabled=False)
    choices, env, backend = compile_choices(observed)
    assert len(original) > 1 and len(choices) == 1
    candidate, = choices
    old = next(c for c in original if c.arguments.get("property_phase") == "confirm")
    assert replace(candidate, id=old.id) == old
    assert backend._issued[candidate.id].step == default_backend._issued[old.id].step
    assert candidate.risk == Risk.READ_ONLY and candidate.arguments["property_phase"] == "confirm"
    assert env.reads == backend.environment_steps == 0
    receipt = backend.execute(candidate, "observed", "singleton")
    assert receipt.success and backend.verify(candidate, receipt).passed
    assert receipt.output["environment_steps"] == 0 and env.actions == []
    assert receipt.output["observation_source"] == "fresh_preflight_read_only"
    assert env.reads == 1


@pytest.mark.parametrize("state", ["unselected", "incomplete", "unknown", "duplicate", "owner"])
def test_unsatisfied_unknown_ambiguous_or_wrong_owner_never_gets_the_shortcut(image, state):
    observed = observed_option(image, selected=state != "unselected")
    option = observed.controls[0]
    if state == "incomplete":
        observed = replace(observed, a11y_complete=False)
    elif state == "unknown":
        observed = replace(observed, controls=(replace(option, selectable=False),
                                                *observed.controls[1:]))
    elif state == "duplicate":
        observed = replace(observed, controls=(*observed.controls, replace(option, ref="twin")))
    elif state == "owner":
        observed = replace(observed, controls=(replace(option, app="Other"), *observed.controls[1:]))
    if state == "incomplete":
        with pytest.raises(ValueError, match="requires a complete observation"):
            plan(observed, data(value="Light"))
        complete = replace(observed, a11y_complete=True)
        root = plan(complete, data(value="Light"))
        env = CountingDesktop(observed)
        backend = Backend(env, pause=0, satisfied_property_readonly_fast_path=True)
        assert backend.compile(root.steps[0], complete, observed) == ()
        assert env.reads == backend.environment_steps == 0 and env.actions == []
        return
    choices, env, backend = compile_choices(observed)
    assert not (len(choices) == 1 and choices[0].arguments.get("property_phase") == "confirm")
    assert env.reads == backend.environment_steps == 0 and env.actions == []


def test_another_legal_field_or_missing_exact_literal_keeps_policy_choice(image):
    editable = control("field", "Another property", field=True, value="old", y=450)
    observed = observed_option(image, extra=(editable,))
    choices, _, _ = compile_choices(observed)
    assert len(choices) > 1
    assert any(c.arguments.get("property_phase") == "edit" for c in choices)
    changed = replace(observed, controls=tuple(
        replace(c, label="LIGHT") if c.ref == "light" else c for c in observed.controls))
    choices, _, _ = compile_choices(changed)
    assert len(choices) > 1  # Strict literal identity; no inferred normalization shortcut.


@pytest.mark.parametrize("drift", ["unselected", "owner", "duplicate", "incomplete"])
def test_original_fresh_preflight_rejects_drift_without_any_input(image, drift):
    observed = observed_option(image)
    choices, env, backend = compile_choices(observed)
    candidate, = choices
    option, *others = observed.controls
    if drift == "unselected":
        env.snapshot = replace(observed, controls=(replace(option, selected=False), *others))
    elif drift == "owner":
        env.snapshot = replace(observed, controls=(replace(option, window="Other"), *others))
    elif drift == "duplicate":
        env.snapshot = replace(observed, controls=(*observed.controls, replace(option, ref="twin")))
    else:
        env.snapshot = replace(observed, a11y_complete=False)
    receipt = backend.execute(candidate, "observed", "singleton")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert receipt.output["environment_steps"] == backend.environment_steps == 0
    assert env.actions == [] and env.reads == 1


@pytest.mark.parametrize("unknown_guard", [False, True])
def test_real_runner_skips_only_readonly_policy_but_keeps_explicit_commit_and_goal(initial, unknown_guard):
    current = current_dialog(initial)
    option = replace(current.controls[1], label="Light", role="cell", tag="button", input_type="",
                     editable=False, value="", selectable=True, selected=True)
    current = replace(current, controls=(current.controls[0], option, *current.controls[2:]))
    raw = {**requested(), "label": "Select Light and confirm",
           "edit": {"operation": "set", "value": "Light"}}
    if unknown_guard:
        raw["goal_conditions"] = [{"kind": "exists", "target": {"name": "Missing required fact"}}]
    env, events, policy_calls, executed = DialogEnvironment(current), [], [], []

    class Planner:
        def plan(self, _goal, observed, _history, **_kwargs):
            return compile(raw, observed)

    class Policy:
        def choose(self, observation, candidates):
            assert observation.state.get("explicit_property_confirmation") is True
            policy_calls.append(len(candidates))
            chosen = next(c for c in candidates if c.description.startswith("click OK "))
            return Decision(observation.observation_id, chosen.id,
                            {c.id: float(c is chosen) for c in candidates}, 1, "offline-commit", 0)

    class RecordingBackend(Backend):
        def execute(self, candidate, *args, **kwargs):
            receipt = super().execute(candidate, *args, **kwargs)
            executed.append((candidate, receipt))
            return receipt

    backend = RecordingBackend(env, pause=0, satisfied_property_readonly_fast_path=True)
    result = AgentRunner(backend, Planner(), policy=Policy(), progress=events.append,
                         config=AgentConfig(allow_actions=True, singleton_fast_path=True,
                                            max_plans=1, max_failures=1)).run("Select Light and confirm")
    assert policy_calls == [2] and env.closed
    assert [a["action_type"] for a in env.actions] == ["CLICK"]
    assert result.property_confirmations == 1 and result.property_edits == 0
    assert result.navigation_steps == 0 and not result.success
    assert result.plan_completion_claims == (0 if unknown_guard else 1)
    receipts = [e for e in events if e["event"] == "agent_step"]
    assert receipts[0]["policy_source"] == "deterministic-singleton"
    # This original event labels the logical property template, not the bound action.
    assert receipts[0]["navigation"] is True
    readonly, readonly_receipt = executed[0]
    assert readonly.risk == Risk.READ_ONLY and readonly.arguments["property_phase"] == "confirm"
    assert readonly_receipt.success and readonly_receipt.output["property_confirmation"] is True
    assert readonly_receipt.output["environment_steps"] == 0
    assert readonly_receipt.output["observation_source"] == "fresh_preflight_read_only"
    assert len([r for _c, r in executed if r.output["environment_steps"] > 0]) == 1
    assert any(e["event"] == "agent_property_commit" for e in events)
    assert any(e["event"] == "agent_property_goal_bound" and e["explicit_conditions_retained"]
               for e in events)


@pytest.mark.parametrize("bad", [1, "true", None])
def test_opt_in_requires_an_actual_boolean(image, bad):
    env = CountingDesktop(observed_option(image))
    with pytest.raises(ValueError, match="read-only fast path must be boolean"):
        Backend(env, pause=0, satisfied_property_readonly_fast_path=bad)
    assert env.reads == 0 and env.actions == []
