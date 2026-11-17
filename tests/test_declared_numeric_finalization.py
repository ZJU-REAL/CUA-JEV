"""An initial model-declared key uses original numeric evidence, never an implicit key."""

from __future__ import annotations

import copy
from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest
from test_pending_numeric_edit import COMPONENT, STATE, VALUE, NumericEnvironment

from cua_jev.agent_planner import parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _duration: None)


def declared_plan(snapshot, *, key="enter", confirm=True, horizon=6):
    raw = {
        "intent": "pursue_property", "label": "Set the requested numeric setting",
        "scope": {"app": "gimp"}, "edit": {"operation": "set", "value": "100"},
        **({"numeric_finalize": {"key": key}} if key else {}),
        **({"confirm": {"label": "Confirm the requested setting"}, "completion": "goal"}
           if confirm else {}),
    }
    return parse_plan(
        raw, snapshot, profile="desktop", max_actions=horizon, allow_property_preparation=True,
        allow_property_value_binding=True, allow_observed_property_goal=True,
        allow_property_numeric_finalization=True,
    )


def pending(*, key="enter", setup=None):
    env = NumericEnvironment()
    if setup:
        setup(env)
    backend = OSWorldBackend(env, pause=0)
    before = backend.observe()
    stage = declared_plan(before, key=key).steps[0]
    candidate = next(c for c in backend.compile(stage, before, before)
                     if c.arguments["operation"] == "fill")
    receipt = backend.execute(candidate, "obs", "decision")
    return env, backend, stage, candidate, receipt, before


def compile_key(backend, stage, candidate, receipt, planned, *, cached=True):
    frame = backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) if cached else None
    current = frame[0] if frame else backend.observe()
    return backend.compile_declared_numeric_finalize(candidate, receipt, stage, planned, current)


def test_declared_key_context_once_and_new_receipt_binds_field_without_rewriting_fill():
    env, backend, stage, candidate, receipt, planned = pending()
    original = copy.deepcopy(receipt.to_dict())
    assert backend.declared_numeric_finalize_pending(candidate, receipt, stage)
    assert not backend.verify(candidate, receipt).passed
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    frame = backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage)
    assert frame and backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is None
    actual, choices = backend.compile_declared_numeric_finalize(candidate, receipt, stage, planned, frame[0])
    assert actual.operation == "key_press" and actual.value == "enter" and len(choices) == 1
    assert choices[0].arguments["source_receipt_id"] == receipt.receipt_id
    committed = backend.execute(choices[0], "keyobs", "keydecision")
    assert committed.success and backend.verify(choices[0], committed).passed
    assert committed.receipt_id != receipt.receipt_id
    assert backend.bound_property_step(choices[0], committed).operation == "fill"
    assert not backend.verify(candidate, receipt).passed and receipt.to_dict() == original
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert backend.compile_declared_numeric_finalize(candidate, receipt, stage, planned, frame[0]) is None


@pytest.mark.parametrize("mutation", ["receipt_copy", "receipt_data", "candidate", "stage", "input", "close"])
def test_original_source_identity_and_deep_payload_must_survive_until_key(mutation):
    env, backend, stage, candidate, receipt, planned = pending()
    if mutation == "receipt_copy":
        receipt = replace(receipt)
    elif mutation == "receipt_data":
        receipt.output["pending_numeric_edit"]["edit_buffer"] = "200"
    elif mutation == "candidate":
        candidate = replace(candidate, description="forged")
    elif mutation == "stage":
        stage = replace(stage, property_edit=replace(stage.property_edit, numeric_finalize_key="tab"))
    elif mutation == "input":
        backend._step("PRESS", {"key": "enter"})
    else:
        backend.close()
    assert backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is None
    assert not backend.declared_numeric_finalize_pending(candidate, receipt, stage)
    assert all(a["action_type"] != "PRESS" for a in env.actions) or mutation == "input"


def test_no_declaration_uses_existing_model_recovery_cache_only():
    env, backend, stage, candidate, receipt, _planned = pending(key="")
    assert backend.pending_numeric_replan(candidate, receipt)
    assert not backend.declared_numeric_finalize_pending(candidate, receipt, stage)
    assert backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is None
    assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is not None
    assert len(env.actions) == 3


def test_key_cache_expiry_requires_fresh_observation_before_compilation(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    env, backend, stage, candidate, receipt, planned = pending()
    now[0] += 1.001
    assert backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is None
    current = backend.observe()
    actual, choices = backend.compile_declared_numeric_finalize(candidate, receipt, stage, planned, current)
    assert actual.value == "enter" and len(choices) == 1 and len(env.actions) == 3


@pytest.mark.parametrize("mutation", ["buffer", "focus", "owner", "geometry", "duplicate", "partial"])
def test_existing_fresh_preflight_refuses_changed_original_field_or_buffer_without_press(mutation):
    env, backend, stage, candidate, receipt, planned = pending()
    _actual, choices = compile_key(backend, stage, candidate, receipt, planned)
    if mutation == "buffer":
        env.node.text = "200"
    elif mutation == "focus":
        env.node.set(STATE + "focused", "false")
    elif mutation == "owner":
        next(env.root.iter("application")).set("name", "other")
    elif mutation == "geometry":
        env.node.set(COMPONENT + "screencoord", "(580, 547)")
    elif mutation == "duplicate":
        parent = next(p for p in env.root.iter() if env.node in list(p))
        parent.append(copy.deepcopy(env.node))
    else:
        env.node.text = "100" + " " * 129
    key_receipt = backend.execute(choices[0], "keyobs", "keydecision")
    assert not key_receipt.success and key_receipt.output["environment_steps"] == 0
    assert backend.bound_property_step(choices[0], key_receipt) is None
    assert not backend.verify(candidate, receipt).passed and len(env.actions) == 3


class ConfirmEnvironment(NumericEnvironment):
    def __init__(self, key="enter"):
        super().__init__()
        self.key, self.saved = key, False
        dialog = next(self.root.iter("dialog"))
        ET.SubElement(dialog, "push-button", {
            "name": "OK", STATE + "enabled": "true", STATE + "sensitive": "true",
            STATE + "showing": "true", STATE + "visible": "true",
            COMPONENT + "screencoord": "(700, 820)", COMPONENT + "size": "(80, 32)",
        })

    def step(self, action, pause):
        if action["action_type"] == "PRESS":
            self.actions.append(action)
            assert action["parameters"]["key"] == self.key
            self.node.set(VALUE + "value", "100.0")
            if self.key == "tab":
                self.node.set(STATE + "focused", "false")
            return self._get_obs(), 0, False, {}
        if action["action_type"] == "CLICK" and self.node.get(VALUE + "value") == "100.0":
            self.actions.append(action)
            app = next(self.root.iter("application"))
            app.remove(next(self.root.iter("dialog")))
            ET.SubElement(app, "frame", {
                "name": "Main", STATE + "active": "true", STATE + "enabled": "true",
                STATE + "sensitive": "true", STATE + "showing": "true", STATE + "visible": "true",
                COMPONENT + "screencoord": "(0, 0)", COMPONENT + "size": "(1920, 1080)",
            })
            self.saved = True
            return self._get_obs(), 0, False, {}
        return super().step(action, pause)


class Planner:
    desktop_contract = "property_intent_v49"
    profile = "desktop"

    def __init__(self, *, key="enter"):
        self.calls, self.key = [], key

    def plan(self, goal, observation, history, *, max_actions):
        self.calls.append((observation, copy.deepcopy(history), max_actions))
        if len(self.calls) == 1:
            return declared_plan(observation, key=self.key, horizon=max_actions)
        return parse_plan({"summary": "done", "done": True, "steps": []}, observation)


def run(*, env=None, backend_type=OSWorldBackend, key="enter", progress=None, **config):
    env, planner = env or ConfirmEnvironment(key), Planner(key=key)
    backend, trace = backend_type(env, pause=0), JsonlTrace()

    class Policy(DirectPlanPolicy):
        def choose(self, observation, candidates):
            choice = next((c for c in candidates if c.capability == "agent.osworld.fill"), candidates[0])
            return Decision(observation.observation_id, choice.id,
                            {c.id: float(c is choice) for c in candidates}, 1, "offline", 0)

    result = AgentRunner(
        backend, planner, policy=Policy(), trace=trace, progress=progress,
        config=AgentConfig(**{
            "allow_actions": True, "max_steps": 3, "max_plans": 2, "plan_horizon": 4,
            "plan_ttl_s": 300, "max_failures": 3, **config,
        }),
    ).run("Set the numeric value and confirm")
    return env, backend, planner, result, trace.events


@pytest.mark.parametrize("key", ["enter", "tab"])
def test_initial_explicit_policy_keeps_original_plan_and_unlocks_only_receipt_verified_confirm(key):
    env, _backend, planner, result, events = run(key=key)
    assert env.saved and result.planner_calls == 1 and len(planner.calls) == 1
    assert result.steps == 3 and result.recovery_observations == 0
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS", "CLICK"]
    receipts = [e["payload"] for e in events if e["kind"] == "receipt"]
    verifications = [e["payload"] for e in events if e["kind"] == "verification"]
    assert [v["passed"] for v in verifications] == [False, True, True]
    assert receipts[0]["output"]["postconditions_met"] is False
    bound = next(e["payload"] for e in events if e["kind"] == "agent_property_commit_bound")
    assert bound["source_receipt_id"] == receipts[1]["receipt_id"] != receipts[0]["receipt_id"]
    context = [e["payload"] for e in events if e["kind"] == "agent_declared_numeric_finalize_context"]
    assert len(context) == 1 and context[0]["verified"] is False
    assert not any(e["kind"] == "agent_planning_observation_reuse" for e in events)


def test_failure_budget_one_never_grants_declared_key_or_old_confirmation():
    env, _backend, planner, result, events = run(max_failures=1)
    assert result.status == "stuck" and result.steps == 1 and len(planner.calls) == 1
    assert len(env.actions) == 3 and not env.saved
    assert not any(e["kind"] == "agent_declared_numeric_finalize_context" for e in events)


def test_original_ttl_expires_during_key_fresh_preflight_without_key_or_confirm(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])

    class SlowKeyBackend(OSWorldBackend):
        def observe(self):
            if any(b.numeric_finalize is not None for b in self._issued.values()):
                now[0] += 301
            return super().observe()

    env, _backend, _planner, result, _events = run(backend_type=SlowKeyBackend)
    assert result.status == "timeout" and result.steps == 2
    assert len(env.actions) == 3 and not env.saved


@pytest.mark.parametrize("mutation", ["receipt", "buffer_after_cache_expiry"])
def test_unavailable_key_context_discards_original_plan_and_never_refills(mutation, monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    env = ConfirmEnvironment()

    class ChangedSourceBackend(OSWorldBackend):
        def compile_declared_numeric_finalize(self, candidate, receipt, stage, planned, current):
            if mutation == "receipt":
                receipt.output["pending_numeric_edit"]["edit_buffer"] = "forged"
            return super().compile_declared_numeric_finalize(candidate, receipt, stage, planned, current)

    def progress(event):
        if (event["event"] == "agent_declared_numeric_finalize"
                and event.get("status") == "pending" and mutation == "buffer_after_cache_expiry"):
            now[0] += 1.001
            env.node.text = "200"

    env, _backend, planner, result, events = run(
        env=env, backend_type=ChangedSourceBackend, progress=progress, max_steps=5,
    )
    assert len(env.actions) == 3 and not env.saved
    assert [call[2] for call in planner.calls] == [4, 1]
    assert result.status == "unverified"
    assert any(e["kind"] == "agent_recovery" and e["payload"].get("reason")
               == "declared_numeric_finalize_unavailable" for e in events)
    assert not any(e["kind"] == "agent_property_commit_bound" for e in events)


@pytest.mark.parametrize("failure", ["no_commit", "transport"])
def test_failed_new_key_does_not_bind_old_field_or_retry_key_or_confirm(failure):
    env, backend, stage, candidate, receipt, planned = pending()
    _actual, choices = compile_key(backend, stage, candidate, receipt, planned)
    original_step = env.step

    def step(action, pause):
        if action["action_type"] == "PRESS":
            env.actions.append(action)
            if failure == "transport":
                raise RuntimeError("synthetic lost key response")
            return env._get_obs(), 0, False, {}
        return original_step(action, pause)

    env.step = step
    key_receipt = backend.execute(choices[0], "keyobs", "keydecision")
    assert not backend.verify(choices[0], key_receipt).passed
    assert backend.bound_property_step(choices[0], key_receipt) is None
    assert not backend.verify(candidate, receipt).passed
    assert backend.compile_declared_numeric_finalize(
        candidate, receipt, stage, planned, backend.observe(),
    ) is None
    assert len(env.actions) == 4


def test_runner_failed_declared_key_goes_to_fresh_model_recovery_without_old_confirm_or_refill():
    class NoCommitEnvironment(ConfirmEnvironment):
        def step(self, action, pause):
            if action["action_type"] == "PRESS":
                self.actions.append(action)
                return self._get_obs(), 0, False, {}
            return super().step(action, pause)

    env, _backend, planner, result, events = run(env=NoCommitEnvironment(), max_steps=5)
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING", "PRESS"]
    assert [c[2] for c in planner.calls] == [4, 1] and result.status == "unverified"
    assert result.recovery_observations == 0 and not env.saved
    assert not any(e["kind"] == "agent_property_commit_bound" for e in events)


def test_original_plan_ttl_alone_can_block_key_while_episode_has_time(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])

    class SlowKeyBackend(OSWorldBackend):
        def observe(self):
            if any(b.numeric_finalize is not None for b in self._issued.values()):
                now[0] += 11
            return super().observe()

    env, _backend, planner, result, events = run(
        backend_type=SlowKeyBackend, max_steps=5, plan_ttl_s=10, timeout_s=300,
    )
    assert len(env.actions) == 3 and not env.saved and result.status == "unverified"
    assert [c[2] for c in planner.calls] == [4, 1]
    key_receipt = next(e["payload"] for e in events if e["kind"] == "receipt"
                       and e["payload"]["capability"] == "agent.osworld.key_press")
    assert key_receipt["output"]["failure_code"] == "input_deadline_exceeded"
    assert key_receipt["output"]["environment_steps"] == 0


def test_runner_no_initial_key_declaration_preserves_single_action_model_recovery():
    env, _backend, planner, result, events = run(key="", max_steps=5)
    assert len(env.actions) == 3 and not env.saved
    assert [c[2] for c in planner.calls] == [4, 1] and result.status == "unverified"
    assert not any(e["kind"].startswith("agent_declared_numeric_finalize") for e in events)
    assert any(e["kind"] == "agent_pending_numeric_replan" for e in events)


def test_keyonly_context_cannot_be_substituted_with_forged_or_expired_snapshot(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("time.perf_counter", lambda: now[0])
    _env, backend, stage, candidate, receipt, planned = pending()
    frame = backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage)
    assert backend.compile_declared_numeric_finalize(
        candidate, receipt, stage, planned, replace(frame[0]),
    ) is None
    _env, backend, stage, candidate, receipt, planned = pending()
    frame = backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage)
    now[0] += 1.001
    assert backend.compile_declared_numeric_finalize(candidate, receipt, stage, planned, frame[0]) is None


@pytest.mark.parametrize("route", ["key", "model"])
def test_declared_key_context_and_legacy_planning_cache_are_mutually_exclusive(route):
    _env, backend, stage, candidate, receipt, _planned = pending()
    if route == "key":
        assert backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is not None
        assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is None
    else:
        assert backend.take_pending_numeric_planning_snapshot(candidate, receipt) is not None
        assert backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is None


@pytest.mark.parametrize("mutation", ["partial", "failed", "no_focus", "changed_buffer"])
def test_original_typing_must_be_complete_successful_focused_and_exact_pending(mutation):
    def setup(env):
        if mutation == "failed":
            env.fail_typing = True
        elif mutation == "partial":
            env.on_typing = lambda: setattr(env.node, "text", "100" + " " * 129)
        elif mutation == "no_focus":
            env.on_typing = lambda: env.node.set(STATE + "focused", "false")
        else:
            env.on_typing = lambda: setattr(env.node, "text", "200")

    env, backend, stage, candidate, receipt, _planned = pending(setup=setup)
    assert not backend.declared_numeric_finalize_pending(candidate, receipt, stage)
    assert backend.take_declared_numeric_finalize_snapshot(candidate, receipt, stage) is None
    assert not backend.verify(candidate, receipt).passed
    assert len(env.actions) == 3
