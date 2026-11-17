"""A missing declared app is observed afresh, never filled in or clicked again."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from test_deferred_property_confirmation import current_dialog, requested
from test_observed_property_goal import compile
from test_property_preparation import Backend, DialogEnvironment
from test_property_preparation import initial as initial

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentControl, AgentStep, AgentTarget
from cua_jev.models import Decision


def entry(initial):
    button = AgentControl(
        "prefs",
        "Preferences",
        "button",
        role="button",
        app="Editor",
        window="Main",
        active=True,
        box=(100, 100, 300, 200),
    )
    return replace(initial, controls=(initial.controls[0], button))


class MissingScope(DialogEnvironment):
    def __init__(self, initial, *, missing_reads=0, variant="restored"):
        self.initial = entry(initial)
        super().__init__(self.initial)
        self.prefs_open = False
        self.scope_pending = False
        self.missing_reads = missing_reads
        self.variant = variant
        self.reads = 0
        self.pending_reads = 0

    def unavailable(self):
        return replace(
            self.snapshot, controls=tuple(replace(control, app="") for control in self.snapshot.controls)
        )

    def _get_obs(self):
        self.reads += 1
        if self.scope_pending:
            self.pending_reads += 1
            if self.missing_reads:
                self.missing_reads -= 1
                return self.unavailable()
            if self.variant == "missing":
                return self.unavailable()
            if self.variant == "wrong_app":
                return replace(
                    self.snapshot,
                    controls=tuple(replace(control, app="Editor Twin") for control in self.snapshot.controls),
                )
            if self.variant == "ambiguous":
                return replace(
                    self.snapshot,
                    controls=(
                        *self.snapshot.controls,
                        *(
                            replace(control, ref="twin-" + control.ref, app="Editor Twin")
                            for control in self.snapshot.controls
                        ),
                    ),
                )
            if self.variant == "readonly":
                return replace(self.snapshot, controls=(self.snapshot.controls[0],))
            if self.variant == "partial_owner":
                return replace(
                    self.snapshot,
                    controls=tuple(
                        replace(control, app="") if control.ref == "wrap" else control
                        for control in self.snapshot.controls
                    ),
                )
            if self.variant == "incomplete":
                return replace(self.snapshot, a11y_complete=False)
            if self.variant == "unchanged":
                return self.initial
            if self.variant == "error":
                raise TimeoutError("offline observation failed")
            self.scope_pending = False
        return self.snapshot

    def step(self, action, pause):
        if not self.prefs_open:
            assert action["action_type"] == "CLICK"
            self.actions.append(action)
            self.prefs_open = self.scope_pending = True
            self.opened = True
            self.snapshot = current_dialog(self.initial)
            return self.unavailable(), 0, False, {}
        return super().step(action, pause)


def selected_navigation(initial, **kwargs):
    env = MissingScope(initial, **kwargs)
    backend = Backend(env, pause=0)
    plan = compile(requested(), env.initial)
    step = plan.steps[0]
    candidate = next(
        candidate
        for candidate in backend.compile(step, env.initial, env.initial)
        if candidate.description.startswith("click Preferences ")
    )
    receipt = backend.execute(candidate, "observation", "decision")
    return env, backend, plan, candidate, receipt


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _duration: None)


@pytest.mark.parametrize("missing_reads", [0, 1])
def test_actual_scope_recovery_is_route_progress_only_and_preserves_original_receipt(initial, missing_reads):
    env, backend, _plan, candidate, receipt = selected_navigation(initial, missing_reads=missing_reads)
    original = copy.deepcopy(receipt.to_dict())
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert backend.property_navigation_effect_pending(candidate, receipt)
    for index in range(missing_reads + 1):
        checked = backend.verify_pending(candidate, receipt)
        assert checked.passed is (index == missing_reads)
        assert checked.details["postconditions_met"] is False
        assert checked.details["property_done"] is False and checked.details["task_success"] is False
    assert checked.details["accessibility_progress"] is True
    assert checked.details["observation_source"] == "verified_pending_effect_observation"
    assert checked.details["pending_observation_fingerprint"] == env.snapshot.fingerprint()
    assert receipt.output["after_fingerprint"] != checked.details["pending_observation_fingerprint"]
    assert receipt.to_dict() == original and len(env.actions) == 1
    assert env.pending_reads == missing_reads + 1
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    reused = backend.take_verified_pending_effect_snapshot(candidate, receipt)
    assert reused[0] is env.snapshot
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None


@pytest.mark.parametrize(
    "variant",
    ["missing", "wrong_app", "ambiguous", "readonly", "partial_owner", "incomplete", "unchanged", "error"],
)
def test_missing_or_ambiguous_scope_never_satisfies_empty_postconditions(initial, variant):
    env, backend, _plan, candidate, receipt = selected_navigation(initial, variant=variant)
    for _ in range(3):
        checked = backend.verify_pending(candidate, receipt)
        assert not checked.passed and checked.details["postconditions_met"] is False
        assert checked.details["property_done"] is False
    assert env.pending_reads == 2 and len(env.actions) == 1
    assert not backend.property_navigation_effect_pending(candidate, receipt)
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("tamper", ["copy", "receipt", "candidate"])
def test_copied_or_mutated_receipts_or_candidates_cannot_poll_or_consume(initial, tamper):
    env, backend, _plan, candidate, receipt = selected_navigation(initial)
    if tamper == "copy":
        receipt = replace(receipt)
    elif tamper == "receipt":
        receipt.output["after_fingerprint"] = "mutated"
    else:
        candidate = replace(candidate, arguments={**candidate.arguments, "value": "mutated"})
    assert not backend.property_navigation_effect_pending(candidate, receipt)
    assert not backend.verify_pending(candidate, receipt).passed
    assert env.pending_reads == 0 and len(env.actions) == 1
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("invalidation", ["observe", "execute", "age", "receipt_copy", "receipt_mutation"])
def test_pending_cache_has_its_own_one_use_receipt_and_freshness_budget(initial, monkeypatch, invalidation):
    clock = [10.0]
    monkeypatch.setattr("cua_jev.osworld_backend.time.perf_counter", lambda: clock[0])
    env, backend, _plan, candidate, receipt = selected_navigation(initial)
    assert backend.verify_pending(candidate, receipt).passed
    if invalidation == "observe":
        backend.observe()
    elif invalidation == "execute":
        refused = backend.execute(candidate, "new-observation", "new-decision")
        assert not refused.success and refused.output["environment_steps"] == 0
    elif invalidation == "age":
        clock[0] += 1.001
    elif invalidation == "receipt_copy":
        receipt = replace(receipt)
    else:
        receipt.output["after_fingerprint"] = "mutated after wait"
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None
    assert len(env.actions) == 1


def test_an_unrelated_observation_invalidates_the_ticket_without_adding_new_wait_budget(initial):
    env, backend, _plan, candidate, receipt = selected_navigation(initial, variant="missing")
    backend.observe()
    assert not backend.property_navigation_effect_pending(candidate, receipt)
    reads = env.reads
    assert not backend.verify_pending(candidate, receipt).passed
    assert env.reads == reads and len(env.actions) == 1


def test_immediate_and_readonly_effects_never_register_the_navigation_wait(initial):
    env = MissingScope(initial)
    backend = Backend(env, pause=0)
    step = AgentStep("one", "click", target=AgentTarget(name="Preferences", app="Editor"), recovery="observe")
    (candidate,) = backend.compile(step, env.initial, env.initial)
    receipt = backend.execute(candidate, "observation", "decision")
    assert not backend.property_navigation_effect_pending(candidate, receipt)
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None
    env.scope_pending = False
    env.snapshot = current_dialog(env.initial, value="50")
    plan = compile(requested(), env.snapshot)
    candidates = backend.compile(plan.steps[0], env.snapshot, env.snapshot)
    candidate = next(
        candidate for candidate in candidates if candidate.arguments.get("property_phase") == "confirm"
    )
    receipt = backend.execute(candidate, "read-only-observation", "read-only-decision")
    assert receipt.success and receipt.output["environment_steps"] == 0
    assert not backend.property_navigation_effect_pending(candidate, receipt)
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None


def test_explicit_navigation_conditions_keep_the_existing_verifier_path(initial):
    env = MissingScope(initial, variant="unchanged")
    backend = Backend(env, pause=0)
    goal = AgentCondition("exists", AgentTarget(name="Unobserved marker", app="Editor"))
    step = AgentStep(
        "explicit",
        "click",
        label="Reach the explicit marker",
        target=AgentTarget(app="Editor"),
        navigation=True,
        native_navigation_gestures=True,
        postconditions=(goal,),
        recovery="observe",
    )
    (candidate,) = backend.compile(step, env.initial, env.initial)
    receipt = backend.execute(candidate, "explicit-observation", "explicit-decision")
    assert not backend.property_navigation_effect_pending(candidate, receipt)
    assert not backend.verify_pending(candidate, receipt).passed
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None


def test_readonly_zero_input_action_cannot_refresh_the_prior_navigation_ticket(initial):
    env, backend, _plan, old_candidate, old_receipt = selected_navigation(initial, variant="missing")
    assert not backend.verify_pending(old_candidate, old_receipt).passed
    env.scope_pending = False
    env.snapshot = current_dialog(env.initial, value="50")
    plan = compile(requested(), env.snapshot)
    candidate = next(
        candidate
        for candidate in backend.compile(plan.steps[0], env.snapshot, env.snapshot)
        if candidate.arguments.get("property_phase") == "confirm"
    )
    receipt = backend.execute(candidate, "readonly-observation", "readonly-decision")
    assert receipt.success and receipt.output["environment_steps"] == 0
    reads = env.reads
    assert not backend.property_navigation_effect_pending(old_candidate, old_receipt)
    assert not backend.verify_pending(old_candidate, old_receipt).passed
    assert env.reads == reads and len(env.actions) == 1


def run(initial, *, variant="restored", missing_reads=0, backend_type=Backend, data=None, **config):
    env = MissingScope(initial, variant=variant, missing_reads=missing_reads)
    backend = backend_type(env, pause=0)
    plan = compile(data or requested(), env.initial)
    events = []

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    class Policy:
        def choose(self, observation, candidates):
            selected = next((c for c in candidates if c.description.startswith("click Preferences ")), None)
            selected = selected or next(
                (c for c in candidates if c.description.startswith("fill Wrap column ")), None
            )
            selected = selected or next(c for c in candidates if c.description.startswith("click OK "))
            return Decision(
                observation.observation_id,
                selected.id,
                {c.id: float(c is selected) for c in candidates},
                1.0,
                "offline-policy",
                0,
            )

    result = AgentRunner(
        backend,
        Planner(),
        policy=Policy(),
        progress=events.append,
        config=AgentConfig(
            **{"allow_actions": True, "max_plans": 1, "max_failures": 1, "max_steps": 10, **config}
        ),
    ).run("Set and confirm the requested property")
    return env, backend, plan, result, events


def test_runner_reuses_one_plan_waits_for_owner_and_keeps_confirmation_and_whole_goal(initial):
    env, _backend, plan, result, events = run(initial, missing_reads=1)
    assert result.planner_calls == 1 and result.plan_reuses == 2 and env.closed
    assert result.property_edits == 1
    assert any(event["event"] == "agent_property_commit" for event in events)
    assert result.navigation_steps == 1 and result.recovery_observations == 2
    assert result.steps == 3 and result.plan_completion_claims == 1 and not result.success
    assert [action["parameters"]["text"] for action in env.actions if action["action_type"] == "TYPING"] == [
        "50"
    ]
    reused = [
        event
        for event in events
        if event["event"] == "agent_observation_reuse"
        and event["source"] == "verified_pending_effect_observation"
    ]
    assert len(reused) == 1
    pending = [event for event in events if event["event"] == "agent_pending_effect"]
    assert len(pending) == 2 and all(event["property_navigation_wait"] for event in pending)
    assert pending[-1]["passed"] and not pending[0]["passed"]
    assert pending[-1]["receipt_id"] == reused[0]["receipt_id"]
    assert pending[-1]["observation_source"] == "verified_pending_effect_observation"
    assert len(pending[-1]["pending_observation_fingerprint"]) == 64
    assert plan.steps[-1].property_confirmation_of and plan.complete_property_goal


def test_pending_reuse_saves_one_expensive_read_without_removing_action_preflight(initial):
    class NoPendingCache(Backend):
        def take_verified_pending_effect_snapshot(self, candidate, receipt):
            return None

    normal_env, normal_backend, _plan, normal, _events = run(initial)
    uncached_env, uncached_backend, _plan, uncached, _events = run(initial, backend_type=NoPendingCache)
    assert normal_env.actions == uncached_env.actions
    assert normal.planner_calls == uncached.planner_calls == 1
    assert uncached_env.reads == normal_env.reads + 1
    assert uncached_backend.fresh_observe_calls == normal_backend.fresh_observe_calls + 1
    assert normal.runner_observe_calls + 1 == uncached.runner_observe_calls


@pytest.mark.parametrize("condition", ["false", "unknown"])
def test_pending_route_never_drops_declared_whole_goal_conditions(initial, condition):
    data = {
        **requested(),
        "goal_conditions": [
            {
                "kind": "value_equals",
                "value": "99",
                "target": {
                    "name": "Wrap column" if condition == "false" else "Unknown value",
                    "app": "Editor",
                },
            }
        ],
    }
    env, _backend, _plan, result, events = run(initial, data=data)
    assert env.closed and result.property_edits == 1 and result.planner_calls == 1
    assert not result.plan_completion_claims and not result.success
    assert any(event["event"] == "agent_property_commit" for event in events)


def test_pending_reads_do_not_expand_the_original_action_budget(initial):
    env, _backend, _plan, result, events = run(initial, missing_reads=1, max_steps=2)
    assert result.steps == 2 and result.recovery_observations == 2 and result.planner_calls == 1
    assert result.status == "max_steps" and not env.closed and not result.plan_completion_claims
    assert not any(event["event"] == "agent_property_commit" for event in events)


@pytest.mark.parametrize("variant", ["missing", "wrong_app", "ambiguous", "readonly", "unchanged"])
def test_runner_never_unlocks_a_field_or_confirmation_without_restored_scope(initial, variant):
    env, _backend, _plan, result, events = run(initial, variant=variant)
    assert result.recovery_observations == 2 and result.planner_calls == 1
    assert result.property_edits == result.property_confirmations == 0 and len(env.actions) == 1
    assert not result.plan_completion_claims and not env.closed
    assert not any(event["event"] == "agent_observation_reuse" for event in events)


@pytest.mark.parametrize("deadline", ["episode", "plan"])
def test_late_positive_pending_observation_cannot_extend_original_time_budget(initial, monkeypatch, deadline):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])

    class Late(Backend):
        def verify_pending(self, candidate, receipt):
            result = super().verify_pending(candidate, receipt)
            assert result.passed
            clock[0] = 0.5  # Still within the independent one-second decision-cache age.
            return result

    env, _backend, _plan, result, events = run(
        initial,
        backend_type=Late,
        timeout_s=0.25 if deadline == "episode" else 120,
        plan_ttl_s=0.25 if deadline == "plan" else 300,
    )
    assert len(env.actions) == 1 and not env.closed and not result.plan_completion_claims
    (pending,) = [event for event in events if event["event"] == "agent_pending_effect"]
    assert pending["deadline_exceeded"] and not pending["passed"]
    assert not any(event["event"] == "agent_observation_reuse" for event in events)
