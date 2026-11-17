"""Regression tests for navigation continuation runner."""

import copy
import io
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_navigation_policy import enable_full_legal_navigation_gestures
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace


class MenuEnvironment:
    def __init__(self):
        self.open = True
        self.arrived = False
        self.inputs = []
        self.reads = 0
        self.owner = "Main"

    def _get_obs(self):
        self.reads += 1
        image = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(image, "PNG")
        child = ('<menu name="Parent popup"><menu-item name="Destination" enabled="true" '
                 'screencoord="(100, 120)" size="(180, 40)"/></menu>') if self.open else ""
        goal = '<label name="Arrival"/>' if self.arrived else ""
        tree = (f'<desktop><application name="Demo"><frame name="{self.owner}" active="true">'
                '<menu-item name="Parent" enabled="true" screencoord="(100, 40)" size="(180, 40)"/>'
                + child + goal + '</frame></application></desktop>')
        return {"screenshot": image.getvalue(), "accessibility_tree": tree}

    def step(self, action, pause=0):
        assert action["action_type"] == "CLICK"
        self.inputs.append(copy.deepcopy(action))
        if action["parameters"]["y"] < 100:
            self.open = not self.open
        else:
            assert self.open
            self.arrived = True
        return self._get_obs(), 0, False, {}


class Planner:
    profile = "desktop"
    desktop_contract = "property_intent_v52"
    desktop_output_format = "semantic_tools_v57"  # Future routing is outside this draft.

    def __init__(self):
        self.requests = []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.requests.append((max_actions, snapshot, copy.deepcopy(history)))
        effect = AgentCondition("exists", AgentTarget(name="Arrival", app="Demo"))
        step = AgentStep("reach", "click", target=AgentTarget(app="Demo"), navigation=True,
                         native_navigation_gestures=True, label="Reach Destination",
                         postconditions=(effect,), recovery="observe")
        if max_actions == 1:
            # A second cycle can leave the menu closed. Ground the recovery in
            # the actual current controls; do not pretend Destination is present.
            target_name, immediate_effect = "Destination", effect
            if not any(control.label == "Destination" for control in snapshot.controls):
                target_name = "Parent"
                immediate_effect = AgentCondition("exists", AgentTarget(name="Destination", app="Demo"))
            step = AgentStep("current", "click", target=AgentTarget(name=target_name, app="Demo"),
                             postconditions=(immediate_effect,), recovery="observe")
        return enable_full_legal_navigation_gestures(AgentPlan(
            "Reach Destination", (step,), reusable=True, completion_conditions=(effect,),
        ))


class ScriptedPolicy:
    def __init__(self, labels):
        self.labels = list(labels)
        self.requests = []

    def choose(self, observation, candidates):
        self.requests.append((observation, tuple(candidates)))
        requested = self.labels[min(len(self.requests) - 1, len(self.labels) - 1)]
        label, button, count = requested if isinstance(requested, tuple) else (requested, "left", 1)
        def gesture(candidate):
            if "(right button," in candidate.description:
                return "right", 1
            return "left", 2 if "2 click(s)" in candidate.description else 1
        selected = next(c for c in candidates if c.description.startswith(f"click {label} [")
                        and gesture(c) == (button, count))
        return Decision(observation.observation_id, selected.id,
                        {c.id: float(c.id == selected.id) for c in candidates},
                        1.0, "scripted-test-policy", 0.0)


class ObservedBackend(OSWorldBackend):
    def __init__(self, environment, *, fresh_change=None):
        super().__init__(environment, pause=0)
        self.pools = []
        self.sources_cleared = []
        self.fresh_change = fresh_change

    def compile(self, *args):
        pool = super().compile(*args)
        self.pools.append(tuple(c.to_dict() for c in pool))
        return pool

    def clear_navigation_memory_sources(self):
        self.sources_cleared.append(self.environment_steps)
        super().clear_navigation_memory_sources()

    def observe(self):
        snapshot = super().observe()
        if len(self.env.inputs) == 2 and self.fresh_change:
            if self.fresh_change == "incomplete":
                return replace(snapshot, a11y_complete=False)
            if self.fresh_change == "focus":
                controls = tuple(replace(c, focused=not c.focused) for c in snapshot.controls)
                return replace(snapshot, controls=controls)
        return snapshot


@pytest.fixture(autouse=True)
def no_poll_sleep(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _seconds: None)


def exercise(labels=("Parent", "Parent", "Destination"), *, enabled=True, change=None, **settings):
    environment = MenuEnvironment()
    backend = ObservedBackend(environment, fresh_change=change)
    planner, policy, trace = Planner(), ScriptedPolicy(labels), JsonlTrace()
    result = AgentRunner(
        backend, planner, policy=policy, trace=trace,
        config=AgentConfig(allow_actions=True, navigation_continuation=enabled,
                           reuse_verified_post_snapshot=False,
                           reuse_verified_pending_snapshot=False, **settings),
    ).run("Reach the requested Destination")
    return result, environment, backend, planner, policy, trace


def test_real_cycle_stays_finalfalse_but_one_fresh_full_pool_choice_can_finish_same_plan():
    result, environment, backend, planner, policy, trace = exercise()
    try:
        assert environment.arrived and len(environment.inputs) == result.steps == 3
        assert [r[0] for r in planner.requests] == [6] and result.planner_calls == 1
        steps = [e["payload"] for e in trace.events if e["kind"] == "agent_step"]
        assert [e["verified"] for e in steps] == [True, False, True]
        continuation = next(
            e["payload"] for e in trace.events if e["kind"] == "agent_navigation_continuation"
        )
        assert continuation["action_failures"] == 1 and continuation["plan_actions"] == 2
        assert continuation["final_verified"] is False and continuation["input_authority"] is False
        assert len(backend.sources_cleared) == 1
        final_observation, public_pool = policy.requests[2]
        assert len(public_pool) == len(backend.pools[2])
        assert [c.id for c in public_pool] == [c["id"] for c in backend.pools[2]]
        assert all(c.arguments == {} for c in public_pool)
        memory = final_observation.state["navigation_memory"]
        assert memory["known_repeat_used"] is True and memory["whole_goal_proven"] is False
        bound = [e for e in memory["edges"] if e["binding"] == "current_unique"]
        assert bound and all(e["candidate_id"] in {c.id for c in public_pool} for e in bound)
        assert result.plan_reuses == 1  # The false cycle itself was not counted.
        assert backend._input_evidence_enabled is False
    finally:
        backend.close()


def test_reselection_of_observed_loop_edge_is_traced_but_sends_no_third_input():
    result, environment, backend, planner, policy, trace = exercise(
        labels=("Parent", "Parent", "Parent", "Destination"), max_steps=3,
    )
    try:
        declined = [e["payload"] for e in trace.events if e["kind"] == "decision_admission"
                    and not e["payload"]["approved"]]
        assert len(declined) == 1 and declined[0]["input_sent"] is False
        decisions = [e["payload"] for e in trace.events if e["kind"] == "decision"]
        receipts = [e["payload"] for e in trace.events if e["kind"] == "receipt"]
        assert len(decisions) == len(receipts) + 1
        assert [r[0] for r in planner.requests] == [6, 1]
        assert len(environment.inputs) == 3 and result.steps == 3
        assert not any(e["kind"] in {"policy_failure", "agent_capacity_recovery", "agent_network_recovery"}
                       for e in trace.events)
    finally:
        backend.close()


@pytest.mark.parametrize("settings,inputs,limits", [
    ({"plan_horizon": 2}, 3, [2, 1]),
    ({"max_failures": 1}, 2, [6]),
    ({"max_steps": 2}, 2, [6]),
])
def test_cycle_never_grants_an_extra_horizon_failure_or_input_budget(settings, inputs, limits):
    result, environment, backend, _planner, policy, trace = exercise(**settings)
    try:
        assert len(environment.inputs) == result.steps == inputs
        assert [r[0] for r in _planner.requests] == limits
        assert len(policy.requests) == 2
        assert not any(e["kind"] == "agent_navigation_continuation"
                       and e["payload"]["stage"] == "fresh_continuation_requested" for e in trace.events)
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["incomplete", "focus"])
def test_fresh_state_must_match_exact_native_evidence_not_only_scope(change):
    result, environment, backend, planner, policy, trace = exercise(change=change, max_steps=3, max_plans=2)
    try:
        assert [r[0] for r in planner.requests] == [6, 1]
        assert len(policy.requests) == 2  # No continued Jev choice after evidence was lost.
        assert 2 <= len(environment.inputs) == result.steps <= 3
        assert any(e["kind"] == "agent_navigation_continuation"
                   and e["payload"].get("reason") == "navigation_continuation_evidence_changed"
                   for e in trace.events)
        assert backend._input_evidence_enabled is False
    finally:
        backend.close()


def test_disabled_policy_keeps_original_cycle_to_h1_without_memory_or_admission():
    result, environment, backend, planner, policy, trace = exercise(enabled=False, max_steps=3)
    try:
        assert [r[0] for r in planner.requests] == [6, 1]
        assert all("navigation_memory" not in observation.state for observation, _ in policy.requests)
        assert not any(e["kind"] in {"decision_admission", "agent_navigation_transition",
                                     "agent_navigation_continuation"} for e in trace.events)
        assert len(environment.inputs) == result.steps == 3
    finally:
        backend.close()


def test_retained_cycle_failure_is_not_reset_before_a_no_input_admission_failure():
    result, environment, backend, planner, _policy, trace = exercise(
        labels=("Parent", "Parent", "Parent"), max_failures=2,
    )
    try:
        assert result.status == "stuck" and len(environment.inputs) == result.steps == 2
        assert [r[0] for r in planner.requests] == [6]
        declined = next(e["payload"] for e in trace.events if e["kind"] == "agent_navigation_continuation"
                        and e["payload"]["stage"] == "selection_declined")
        assert declined["action_failures"] == 2
    finally:
        backend.close()


def test_another_actual_cycle_cannot_receive_a_second_concession():
    result, environment, backend, planner, _policy, trace = exercise(
        labels=("Parent", "Parent", ("Parent", "right", 1), "Destination"), max_steps=4,
    )
    try:
        assert [r[0] for r in planner.requests] == [6, 1]
        assert len(environment.inputs) == result.steps == 4
        assert environment.arrived is False  # The fourth input only reopens the menu within the cap.
        steps = [e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step"]
        assert steps[:3] == [True, False, False]
        assert sum(e["kind"] == "agent_navigation_continuation"
                   and e["payload"]["stage"] == "fresh_continuation_requested" for e in trace.events) == 1
    finally:
        backend.close()


def test_original_episode_deadline_still_stops_before_any_extra_input(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])
    original = ObservedBackend.observe

    def slow_fresh(backend):
        if len(backend.env.inputs) >= 2:
            clock[0] = 21.0
        return original(backend)

    monkeypatch.setattr(ObservedBackend, "observe", slow_fresh)
    result, environment, backend, planner, _policy, _trace = exercise(timeout_s=20)
    try:
        assert result.status == "timeout" and len(environment.inputs) == 2
        assert [r[0] for r in planner.requests] == [6]
    finally:
        backend.close()


@pytest.mark.parametrize("value", [1, "true", None])
def test_continuation_switch_requires_exact_bool(value):
    with pytest.raises(ValueError, match="navigation_continuation"):
        AgentConfig(navigation_continuation=value)
