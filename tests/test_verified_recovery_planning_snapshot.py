"""Verified recovery information reuse, with real backend receipt and freshness checks."""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

import cua_jev.agent_runner as _RUNNER
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image
from cua_jev.trace import JsonlTrace


class Clock:
    now = 1000.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr("time.time", value.read)
    monkeypatch.setattr("time.perf_counter", value.read)
    monkeypatch.setattr("time.sleep", value.advance)
    return value


@pytest.fixture
def snapshot():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, format="PNG")
    return AgentSnapshot("osworld://desktop", "Settings", "Settings", tuple(
        AgentControl(ref, label, "input", role="checkbox", input_type="checkbox", app="Settings",
                     box=box)
        for ref, label, box in (("a", "Alpha", (100, 100, 200, 200)),
                                ("b", "Beta", (400, 100, 500, 200)))
    ), image=screenshot_image(buffer.getvalue()))


class Environment:
    def __init__(self, snapshot):
        self.snapshot, self.inputs, self.reads = snapshot, [], 0

    def _get_obs(self):
        self.reads += 1
        return self.snapshot

    def step(self, action, pause):
        self.inputs.append(action)
        assert action["action_type"] == "CLICK"
        ref = "a" if action["parameters"]["x"] < 250 else "b"
        self.snapshot = replace(self.snapshot, controls=tuple(
            replace(control, checked=not control.checked) if control.ref == ref else control
            for control in self.snapshot.controls
        ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def __init__(self, env):
        super().__init__(env, pause=0)
        self.take_calls = 0

    def _snapshot(self, observation):
        return observation

    def take_verified_post_snapshot(self, candidate, receipt):
        self.take_calls += 1
        return super().take_verified_post_snapshot(candidate, receipt)


def checked(name):
    return AgentCondition("checked", AgentTarget(name=name, app="Settings"))


def check_step(name):
    return AgentStep(name.lower(), "check", target=checked(name).target,
                     postconditions=(checked(name),))


def source(backend):
    current = backend.env.snapshot
    candidate = backend.compile(check_step("Alpha"), current, current)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and backend.verify(candidate, receipt).passed
    return candidate, receipt, receipt.output["after_fingerprint"]


class Planner:
    profile = "desktop"
    desktop_contract = "property_intent_v52"

    def __init__(self, env, *, final="action", provenance=True,
                 planning_observation="full_multimodal_v1"):
        self.env, self.final, self.provenance = env, final, provenance
        self.planning_observation = planning_observation
        self.snapshots, self.limits = [], []
        self.last_request_provenance = None

    def plan(self, goal, snapshot, history, *, max_actions):
        self.snapshots.append(snapshot)
        self.limits.append(max_actions)
        self.last_request_provenance = {
            "planning_observation": self.planning_observation, "image_uploaded": True,
            "a11y_complete": True, "max_actions": max_actions if self.provenance else 6,
        }
        if len(self.snapshots) == 1:
            raise ValueError("fixture rejected a broad plan before any input")
        if len(self.snapshots) == 2:
            assert max_actions == 1
            return AgentPlan("Recover one option", (check_step("Alpha"),), reusable=True)
        if self.final == "advisory" or len(self.snapshots) > 3:
            return AgentPlan("Model advisory completion", (), done=True)
        if self.final == "conditions":
            # The cached planning input is true, but the live state changes while
            # this model thinks. Completion must still read the new false state.
            self.env.snapshot = replace(self.env.snapshot, controls=tuple(
                replace(control, checked=False) if control.ref == "a" else control
                for control in self.env.snapshot.controls
            ))
            return AgentPlan("Check the whole goal", (), done=True,
                             completion_conditions=(checked("Alpha"),))
        return AgentPlan("Enable Beta", (check_step("Beta"),), reusable=True,
                         completion_conditions=(checked("Beta"),))


def run(snapshot, *, reuse=True, final="action", progress=None, provenance=True, max_plans=4,
        planning_observation="full_multimodal_v1"):
    env, trace = Environment(snapshot), JsonlTrace()
    backend = Backend(env)
    planner = Planner(env, final=final, provenance=provenance, planning_observation=planning_observation)
    result = _RUNNER.AgentRunner(
        backend, planner, policy=_RUNNER.DirectPlanPolicy(), trace=trace,
        progress=(lambda event: progress(event, backend)) if progress else None,
        config=_RUNNER.AgentConfig(
            allow_actions=True, max_steps=6, max_plans=max_plans, timeout_s=300, plan_ttl_s=300,
            reuse_verified_post_snapshot=reuse,
        ),
    ).run("Enable the two requested options")
    return env, backend, planner, result, trace.events


def test_verified_h1_recovery_saves_one_planning_read_but_keeps_both_input_preflights(clock, snapshot):
    enabled = run(snapshot)
    disabled = run(snapshot, reuse=False)
    env, backend, planner, result, events = enabled
    assert planner.limits == [6, 1, 6]
    assert len(env.inputs) == len(disabled[0].inputs) == 2
    assert result.planner_calls == disabled[3].planner_calls == 3
    assert result.runner_observe_calls == disabled[3].runner_observe_calls - 1
    assert result.fresh_observe_calls == disabled[3].fresh_observe_calls - 1
    assert backend.fresh_observe_calls - result.runner_observe_calls == 2
    assert result.post_snapshot_reuses == result.plan_reuses == 0
    assert result.recovery_planning_snapshot_reuses == 1
    reused = [e["payload"] for e in events if e["kind"] == "agent_recovery_planning_observation_reuse"]
    assert len(reused) == 1 and reused[0]["planning_only"] is True
    assert reused[0]["task_success"] is False
    assert planner.snapshots[2].controls[0].checked is True
    # A whole-goal stop still performs the final independent fresh observation.
    assert events[-2]["kind"] == "agent_plan_completion"


@pytest.mark.parametrize("invalidate", ["expiry", "observe"])
def test_runner_falls_back_when_recovery_return_is_old_or_invalidated(clock, snapshot, invalidate):
    def progress(event, backend):
        if event.get("event") == "agent_step" and event.get("planner_calls") == 2:
            if invalidate == "expiry":
                clock.advance(1.01)
            else:
                backend.observe()
    _, _, _, result, events = run(snapshot, progress=progress)
    assert result.recovery_planning_snapshot_reuses == 0
    assert not any(e["kind"] == "agent_recovery_planning_observation_reuse" for e in events)
    assert result.planner_calls == 3


@pytest.mark.parametrize("final", ["advisory", "conditions"])
def test_reused_planning_frame_never_proves_model_completion(clock, snapshot, final):
    env, _, planner, result, events = run(snapshot, final=final)
    assert result.recovery_planning_snapshot_reuses == 1
    assert len(env.inputs) == 1
    assert result.status == "unverified" and result.success is False
    assert result.plan_completion_claims == 0
    if final == "conditions":
        assert planner.snapshots[2].controls[0].checked is True
        assert planner.snapshots[3].controls[0].checked is False
        assert any(e["kind"] == "agent_recovery"
                   and e["payload"].get("reason") == "completion_unconfirmed" for e in events)


def test_untrusted_non_h1_provenance_cannot_offer_planning_ticket(clock, snapshot):
    _, _, _, result, _ = run(snapshot, provenance=False)
    assert result.recovery_planning_snapshot_reuses == 0


@pytest.mark.parametrize("observation", ["native_first_v51", "unknown", None])
def test_full_image_response_does_not_opt_another_observation_contract_into_cache(
    clock, snapshot, observation,
):
    enabled = run(snapshot, planning_observation=observation)
    disabled = run(snapshot, reuse=False, planning_observation=observation)
    assert enabled[3].recovery_planning_snapshot_reuses == 0
    assert enabled[3].runner_observe_calls == disabled[3].runner_observe_calls
    assert enabled[3].fresh_observe_calls == disabled[3].fresh_observe_calls
    assert len(enabled[0].inputs) == len(disabled[0].inputs) == 2


def test_planning_budget_stops_without_consuming_ticket_or_extra_call(clock, snapshot):
    env, backend, planner, result, _ = run(snapshot, max_plans=2)
    assert len(env.inputs) == 1 and planner.limits == [6, 1]
    assert result.planner_calls == 2 and result.status == "max_plans"
    assert result.recovery_planning_snapshot_reuses == backend.take_calls == 0


def test_real_backend_ticket_is_single_use(clock, snapshot):
    backend = Backend(Environment(snapshot))
    ticket = source(backend)
    frame = _RUNNER._take_verified_recovery_planning_snapshot(backend, ticket)
    assert frame[0] is backend.env.snapshot
    assert _RUNNER._take_verified_recovery_planning_snapshot(backend, ticket) is None
    assert backend.fresh_observe_calls == 1


@pytest.mark.parametrize("invalid", [
    "expired", "future", "observe", "new_input", "input_counter", "copied_receipt",
    "altered_receipt", "candidate", "failed", "zero_input", "partial", "no_image", "fingerprint",
])
def test_backend_receipt_counters_fullness_and_fingerprint_are_not_relaxed(clock, snapshot, invalid):
    backend = Backend(Environment(snapshot))
    candidate, receipt, fingerprint = source(backend)
    if invalid in {"expired", "future"}:
        clock.advance(1.01 if invalid == "expired" else -0.01)
    elif invalid == "observe":
        backend.observe()
    elif invalid == "new_input":
        backend._step("CLICK", {"x": 360, "y": 90})
    elif invalid == "input_counter":
        backend.environment_steps += 1
    elif invalid == "copied_receipt":
        receipt = replace(receipt)
    elif invalid == "altered_receipt":
        receipt.output["after_fingerprint"] = "f" * 64
    elif invalid == "candidate":
        candidate = replace(candidate, description="different issued candidate")
    elif invalid == "failed":
        receipt = replace(receipt, success=False)
    elif invalid == "zero_input":
        receipt.output["environment_steps"] = 0
    elif invalid in {"partial", "no_image", "fingerprint"}:
        saved = backend._post_snapshot
        frame = saved[3]
        if invalid == "partial":
            frame = replace(frame, a11y_complete=False)
        elif invalid == "no_image":
            frame = replace(frame, image=None)
        else:
            frame = replace(frame, title="another observation")
        backend._post_snapshot = (*saved[:3], frame, *saved[4:])
    assert _RUNNER._take_verified_recovery_planning_snapshot(
        backend, (candidate, receipt, fingerprint),
    ) is None
