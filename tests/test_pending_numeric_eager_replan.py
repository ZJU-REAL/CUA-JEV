"""Original uncommitted numeric text prompts a model decision, never an automatic key."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from test_pending_numeric_edit import (
    STATE,
    TARGET,
    VALUE,
    NumericEnvironment,
    field,
    filled_environment,
)

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)


def test_original_typing_frame_is_unverified_and_planning_only_once():
    env, backend, candidate, receipt = filled_environment()
    original = copy.deepcopy(receipt.to_dict())
    assert backend.pending_numeric_replan(candidate, receipt)
    assert not backend.verify(candidate, receipt).passed
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert backend.take_verified_pending_effect_snapshot(candidate, receipt) is None
    observed, _received = backend.take_pending_numeric_planning_snapshot(candidate, receipt)
    assert observed.a11y_complete and field(observed).focused
    assert field(observed).value == "5.0" and field(observed).edit_buffer == "100"
    assert observed.fingerprint() == receipt.output["after_fingerprint"]
    assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is None
    assert not backend.pending_numeric_replan(candidate, receipt)
    assert not backend.verify(candidate, receipt).passed and receipt.to_dict() == original
    assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING"]


@pytest.mark.parametrize("change", ["copy", "payload", "candidate", "observe", "execute", "input", "close"])
def test_planning_ticket_never_survives_changed_receipt_or_observation_input_generation(change):
    env, backend, candidate, receipt = filled_environment()
    if change == "copy":
        receipt = replace(receipt)
    elif change == "payload":
        receipt.output["pending_numeric_edit"]["committed_value"] = "mutated"
    elif change == "candidate":
        candidate = replace(candidate, description="mutated")
    elif change == "observe":
        backend.observe()
    elif change == "execute":
        backend.execute(candidate, "new-observation", "new-decision")
    elif change == "input":
        backend._step("PRESS", {"key": "enter"})
    else:
        backend.close()
    assert not backend.pending_numeric_replan(candidate, receipt)
    assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is None
    assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is None


def test_old_typing_frame_can_suggest_replanning_but_cannot_supply_an_expired_snapshot(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    env, backend, candidate, receipt = filled_environment()
    clock[0] += 1.001
    assert backend.pending_numeric_replan(candidate, receipt)
    assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is None
    assert not backend.pending_numeric_replan(candidate, receipt)
    assert len(env.actions) == 3 and not backend.verify(candidate, receipt).passed


@pytest.mark.parametrize(
    "mutation", ["unfocused", "incomplete", "already_committed", "different_buffer", "textbox"]
)
def test_original_typing_return_itself_must_contain_the_existing_strict_numeric_fact(mutation):
    def setup(env):
        def mutate():
            if mutation == "unfocused":
                env.node.set(STATE + "focused", "false")
            elif mutation == "incomplete":
                env.node.text += " " * 126
            elif mutation == "already_committed":
                env.node.set(VALUE + "value", "100.0")
            elif mutation == "different_buffer":
                env.node.text = "101"
            else:
                env.node.tag = "entry"

        env.on_typing = mutate

    _env, backend, candidate, receipt = filled_environment(setup=setup)
    assert not backend.pending_numeric_replan(candidate, receipt)
    assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is None


class Planner:
    def __init__(self, *, strategy="commit", property_plan=False, on_recovery=None):
        self.calls = []
        self.strategy = strategy
        self.property_plan = property_plan
        self.on_recovery = on_recovery

    def plan(self, goal, observation, history, *, max_actions):
        self.calls.append(
            {"max_actions": max_actions, "snapshot": observation, "history": copy.deepcopy(history)}
        )
        target = {"name": TARGET.name, "role": "spinbutton", "app": "gimp", "window": "Preferences"}
        effect = {"kind": "value_equals", "target": target, "value": "100"}
        if len(self.calls) == 1 and self.property_plan:
            raw = {
                "intent": "pursue_property",
                "label": "Set the requested numeric property",
                "scope": {"app": "gimp"},
                "edit": {"operation": "fill", "value": "100"},
                "confirm": {"label": "Confirm the requested setting"},
                "completion": "goal",
            }
        elif len(self.calls) == 1 or self.strategy == "refill":
            raw = {
                "summary": "Fill the requested number",
                "done": False,
                "steps": [
                    {
                        "id": "fill",
                        "operation": "fill",
                        "target": target,
                        "value": "100",
                        "postconditions": [effect],
                        "recovery": "observe",
                    }
                ],
            }
        else:
            assert max_actions == 1
            if self.on_recovery:
                self.on_recovery()
            step = (
                {
                    "id": "commit",
                    "operation": "key_press",
                    "value": "enter",
                    "preconditions": [{"kind": "focused", "target": target}],
                }
                if self.strategy == "commit"
                else {
                    "id": "wait",
                    "operation": "wait",
                    "value": "250",
                    "preconditions": [{"kind": "focused", "target": target}],
                }
            )
            raw = {
                "summary": "Explicit one-action numeric recovery",
                "done": False,
                "steps": [
                    {
                        **step,
                        "postconditions": [effect],
                        "recovery": "observe",
                    }
                ],
                "completion_conditions": [effect],
            }
        plan = parse_plan(
            raw,
            observation,
            profile="desktop",
            max_actions=max_actions,
            allow_property_preparation=max_actions > 1,
            allow_observed_property_goal=max_actions > 1,
        )
        _validate_desktop_execution_plan(plan, observation, max_actions)
        return plan


def run(*, env=None, planner=None, backend_type=OSWorldBackend, **config):
    env, planner = env or NumericEnvironment(), planner or Planner()
    backend, trace = backend_type(env, pause=0), JsonlTrace()

    class Policy(DirectPlanPolicy):
        def choose(self, observation, candidates):
            selected = next(
                (candidate for candidate in candidates if candidate.capability == "agent.osworld.fill"),
                candidates[0],
            )
            return Decision(
                observation.observation_id,
                selected.id,
                {candidate.id: float(candidate is selected) for candidate in candidates},
                1.0,
                "offline-policy",
                0,
            )

    result = AgentRunner(
        backend,
        planner,
        policy=Policy(),
        trace=trace,
        config=AgentConfig(
            **{"allow_actions": True, "max_failures": 2, "max_steps": 6, "max_plans": 2, **config}
        ),
    ).run("Set the requested numeric value and confirm it")
    return env, backend, planner, result, trace.events


def test_numeric_recovery_asks_the_model_once_without_two_effect_polls_or_automatic_commit():
    env, _backend, planner, result, events = run()
    assert result.planner_calls == 2 and result.recovery_observations == 0
    assert [call["max_actions"] for call in planner.calls] == [6, 1]
    assert result.steps == 2 and result.plan_completion_claims == 1 and not result.success
    steps = [event["payload"] for event in events if event["kind"] == "agent_step"]
    assert [step["verified"] for step in steps] == [False, True]
    assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert env.node.get(VALUE + "value") == "100.0"
    assert result.post_snapshot_reuses == 0
    reused = [event["payload"] for event in events if event["kind"] == "agent_planning_observation_reuse"]
    assert len(reused) == 1 and reused[0]["max_actions"] == 1 and reused[0]["verified"] is False
    assert reused[0]["source"] == "unverified_numeric_typing_response_for_planning"
    assert field(planner.calls[1]["snapshot"]).value == "5.0"
    assert field(planner.calls[1]["snapshot"]).edit_buffer == "100"
    assert not any(event["kind"] == "agent_pending_effect" for event in events)


def test_failed_property_plan_never_automatically_resumes_its_old_confirm_step():
    env, _backend, planner, result, events = run(planner=Planner(property_plan=True))
    assert result.planner_calls == 2 and planner.calls[1]["max_actions"] == 1
    assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert not any(event["kind"] == "agent_property_commit" for event in events)


@pytest.mark.parametrize("strategy", ["wait", "refill"])
def test_model_selected_wait_or_bad_refill_does_not_reset_the_failure_budget(strategy):
    env, _backend, planner, result, events = run(planner=Planner(strategy=strategy), max_plans=5)
    assert result.planner_calls == 2 and result.status == "stuck"
    assert result.recovery_observations == (2 if strategy == "wait" else 0)
    assert all(
        event["payload"]["step"] == "wait" for event in events if event["kind"] == "agent_pending_effect"
    )
    assert all(not event["payload"]["verified"] for event in events if event["kind"] == "agent_step")
    assert not any(action["action_type"] == "PRESS" for action in env.actions)
    assert len(env.actions) == (6 if strategy == "refill" else 3)


def test_existing_failure_limit_does_not_gain_an_extra_numeric_repair_opportunity():
    env, _backend, planner, result, _events = run(max_failures=1)
    assert len(planner.calls) == 1 and result.status == "stuck"
    assert len(env.actions) == 3 and result.recovery_observations == 0


def test_post_model_fresh_focus_check_still_rejects_enter_from_the_planning_frame():
    env = NumericEnvironment()
    env.lose_focus_on_pending = True
    env, _backend, planner, result, _events = run(env=env)
    assert len(planner.calls) == 2 and field(planner.calls[1]["snapshot"]).focused
    assert env.node.get(STATE + "focused") == "false"
    assert not any(action["action_type"] == "PRESS" for action in env.actions)
    assert env.node.get(VALUE + "value") == "5.0" and not result.plan_completion_claims


def test_asynchronous_commit_is_seen_by_fresh_post_model_observation_and_needs_no_key():
    class AutoCommit(NumericEnvironment):
        def _get_obs(self):
            if self.after_typing_reads is not None and self.after_typing_reads >= 1:
                self.node.set(VALUE + "value", "100.0")
            return super()._get_obs()

    env, _backend, planner, result, _events = run(env=AutoCommit())
    assert field(planner.calls[1]["snapshot"]).value == "5.0"
    assert env.node.get(VALUE + "value") == "100.0"
    assert not any(action["action_type"] == "PRESS" for action in env.actions)
    assert result.condition_skips == 1 and result.plan_completion_claims == 1


def test_numeric_planning_cache_is_optional_without_changing_actions_or_effect_poll_budget():
    cached_env, _backend, cached, result, cached_events = run()
    fresh_env, _backend, fresh, fresh_result, fresh_events = run(reuse_verified_post_snapshot=False)
    assert cached_env.actions == fresh_env.actions
    assert len(cached.calls) == len(fresh.calls) == 2
    assert result.recovery_observations == fresh_result.recovery_observations == 0
    assert any(event["kind"] == "agent_planning_observation_reuse" for event in cached_events)
    assert not any(event["kind"] == "agent_planning_observation_reuse" for event in fresh_events)


def test_model_time_cannot_extend_episode_budget_or_send_enter(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    planner = Planner(on_recovery=lambda: clock.__setitem__(0, 2.0))
    env, _backend, planner, result, _events = run(planner=planner, timeout_s=1)
    assert len(planner.calls) == 2 and result.status == "timeout"
    assert len(env.actions) == 3 and not result.plan_completion_claims


@pytest.mark.parametrize("declaration", ["advisory_done", "stale_explicit_goal", "async_actual_goal"])
def test_model_completion_after_numeric_planning_frame_still_uses_fresh_goal_evidence(declaration):
    env = NumericEnvironment()

    class CompletionPlanner(Planner):
        def plan(self, goal, observation, history, *, max_actions):
            if not self.calls:
                return super().plan(goal, observation, history, max_actions=max_actions)
            assert max_actions == 1
            self.calls.append(
                {"max_actions": max_actions, "snapshot": observation, "history": copy.deepcopy(history)}
            )
            raw = {"summary": "Explicit model stopping declaration", "done": True, "steps": []}
            if declaration == "stale_explicit_goal":
                assert field(observation).focused
                env.node.set(STATE + "focused", "false")
                raw["completion_conditions"] = [
                    {
                        "kind": "focused",
                        "target": {
                            "name": TARGET.name,
                            "app": "gimp",
                            "window": "Preferences",
                        },
                    }
                ]
            elif declaration == "async_actual_goal":
                assert field(observation).value == "5.0"
                env.node.set(VALUE + "value", "100.0")
                raw["completion_conditions"] = [
                    {
                        "kind": "value_equals",
                        "value": "100",
                        "target": {
                            "name": TARGET.name,
                            "app": "gimp",
                            "window": "Preferences",
                        },
                    }
                ]
            result = parse_plan(raw, observation, profile="desktop", max_actions=max_actions)
            _validate_desktop_execution_plan(result, observation, max_actions)
            return result

    env, backend, planner, result, events = run(env=env, planner=CompletionPlanner())
    assert result.planner_calls == 2 and result.post_snapshot_reuses == 0 and not result.success
    assert len(env.actions) == 3 and len(planner.calls) == 2
    assert result.plan_completion_claims == (1 if declaration == "async_actual_goal" else 0)
    assert all(not event["payload"]["verified"] for event in events if event["kind"] == "agent_step")
    assert any(event["kind"] == "agent_planning_observation_reuse" for event in events)
    if declaration == "advisory_done":
        assert result.status == "unverified" and "model reported completion" in result.reason
    assert not any(value[2].passed for value in backend._verified.values())


def test_available_unverified_frame_is_not_adopted_for_an_ordinary_multi_action_request():
    _env, backend, candidate, receipt = filled_environment()
    assert backend.pending_numeric_replan(candidate, receipt)
    taken = []
    original = backend.take_pending_numeric_planning_snapshot

    def take(*args):
        taken.append(args)
        return original(*args)

    backend.take_pending_numeric_planning_snapshot = take

    class OrdinaryPlanner:
        limits = []

        def plan(self, goal, snapshot, history, *, max_actions):
            self.limits.append(max_actions)
            return parse_plan(
                {"summary": "Only an advisory declaration", "done": True, "steps": []},
                snapshot,
                profile="desktop",
                max_actions=max_actions,
            )

    planner = OrdinaryPlanner()
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True, max_plans=1)
    ).run("Inspect")
    assert planner.limits == [6] and taken == [] and not result.success


def test_vlm_baseline_uses_only_a_single_action_recovery_planning_frame():
    env, _backend, planner, result, events = run(mode="vlm")
    assert [call["max_actions"] for call in planner.calls] == [1, 1]
    assert result.recovery_observations == 0
    assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert all(
        event["payload"]["max_actions"] == 1
        for event in events
        if event["kind"] == "agent_planning_observation_reuse"
    )
