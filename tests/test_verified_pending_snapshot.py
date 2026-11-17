"""Verified pending-frame reuse retains actual input and observation provenance."""

import io
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from PIL import Image

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import RulePolicy
from cua_jev.trace import JsonlTrace


def _png(color):
    output = io.BytesIO()
    Image.new("RGB", (800, 600), color).save(output, format="PNG")
    return output.getvalue()


def _xml(*, saved=False, finished=False):
    return (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:c="https://accessibility.ubuntu.example.org/ns/component">'
        '<application name="Demo"><frame name="Editor" s:showing="true" s:active="true">'
        '<panel name="General" s:showing="true">'
        '<push-button name="Save" s:showing="true" s:visible="true" s:enabled="true" '
        's:sensitive="true" c:screencoord="(80,120)" c:size="(120,40)"/>'
        '<push-button name="Finish" s:showing="true" s:visible="true" s:enabled="true" '
        's:sensitive="true" c:screencoord="(240,120)" c:size="(120,40)"/>'
        + ('<label name="Saved" s:showing="true"/>' if saved else '')
        + ('<label name="Finished" s:showing="true"/>' if finished else '')
        + '</panel></frame></application></desktop-frame>'
    )


class Environment:
    def __init__(self):
        self.inputs = []
        self.reads = 0
        self.expose_saved = True
        self.partial = False
        self.controller = SimpleNamespace(execute_python_command=lambda command: {"status": "success"})

    def _observation(self, *, step_return=False):
        saved = bool(self.inputs) and self.expose_saved and not (step_return and len(self.inputs) == 1)
        return {"screenshot": _png("gray" if self.inputs else "white"),
                "accessibility_tree": None if self.partial else _xml(
                    saved=saved, finished=len(self.inputs) > 1,
                )}

    def _get_obs(self):
        self.reads += 1
        return self._observation()

    def step(self, action, pause=0):
        self.inputs.append(action)
        return self._observation(step_return=True), 0, False, {}


def _step(name="Save", effect="Saved", *, depends=()):
    return AgentStep(name.lower(), "click", target=AgentTarget(name=name), depends_on=depends,
                     recovery="observe", postconditions=(AgentCondition("exists", AgentTarget(name=effect)),))


@pytest.fixture
def pending():
    env = Environment()
    backend = OSWorldBackend(env, pause=0)
    backend.set_input_observation_evidence_enabled(True)
    observed = backend.observe()
    step = _step()
    candidate = backend.compile(step, observed, observed)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and receipt.output["environment_steps"] == 1
    assert not backend.verify(candidate, receipt).passed
    yield backend, env, observed, candidate, receipt
    backend.close()


def _complete(pending):
    backend, env, observed, candidate, receipt = pending
    assert backend.verify_pending(candidate, receipt).passed
    return backend, env, observed, candidate, receipt


def test_final_pending_frame_has_actual_before_and_separate_fingerprint(pending):
    backend, env, requested, candidate, receipt = _complete(pending)
    evidence = backend.peek_input_observation_evidence(candidate, receipt)
    assert evidence.actual_before is not requested  # A real preflight was performed.
    assert evidence.from_pending and evidence.owner_app == "Demo"
    assert evidence.step_data == evidence.binding.step.to_dict()
    original_data = receipt.to_dict()
    old_fp = receipt.output["after_fingerprint"]
    reads, inputs = env.reads, len(env.inputs)
    frame, received_at, fingerprint = backend.take_verified_pending_snapshot(candidate, receipt)
    assert frame is evidence.actual_final_after and received_at == evidence.received_at
    assert fingerprint == frame.fingerprint() and fingerprint != old_fp
    assert receipt.to_dict() == original_data
    assert (env.reads, len(env.inputs)) == (reads, inputs)
    assert backend.take_verified_pending_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("change", ["observe", "input", "copy_receipt", "candidate", "fingerprint", "age"])
def test_stale_forged_or_superseded_frames_are_not_reused(pending, monkeypatch, change):
    backend, env, _requested, candidate, receipt = _complete(pending)
    if change == "observe":
        backend.observe()
    elif change == "input":
        now = backend._input_evidence.actual_final_after
        next_candidate = backend.compile(_step("Finish", "Finished"), now, now)[0]
        backend.execute(next_candidate, "next observation", "next decision")
    elif change == "copy_receipt":
        receipt = replace(receipt)
    elif change == "candidate":
        candidate = replace(candidate, description="different")
    elif change == "fingerprint":
        backend._input_evidence = replace(backend._input_evidence, after_fingerprint="0" * 64)
    else:
        clock = backend._input_evidence.received_at + 1.001
        monkeypatch.setattr("cua_jev.osworld_backend.time.perf_counter", lambda: clock)
    assert backend.take_verified_pending_snapshot(candidate, receipt) is None
    assert backend._input_evidence is None


@pytest.mark.parametrize("change", ["false", "partial", "zero", "bool_count"])
def test_unverified_partial_and_nonpositive_input_do_not_authorize_cache(pending, change):
    backend, env, _requested, candidate, receipt = pending
    if change in {"false", "partial"}:
        env.expose_saved = False
        env.partial = change == "partial"
        assert not backend.verify_pending(candidate, receipt).passed
    else:
        source = backend._input_evidence
        count = False if change == "bool_count" else 0
        forged = replace(receipt, output={**receipt.output, "environment_steps": count})
        backend._remember_input_observation(source.binding, candidate, forged, source.actual_before,
                                            source.actual_final_after, source.received_at,
                                            source.verification,
                                            from_pending=True)
    assert backend.take_verified_pending_snapshot(candidate, receipt) is None


def test_reuse_never_replaces_actual_input_guard(pending):
    backend, env, _requested, candidate, receipt = _complete(pending)
    frame, _received, _fingerprint = backend.take_verified_pending_snapshot(candidate, receipt)
    next_candidate = backend.compile(_step("Finish", "Finished"), frame, frame)[0]
    env.expose_saved = False  # External UI change after cached decision frame.
    reads = env.reads
    refusal = backend.execute(next_candidate, "next observation", "next decision")
    assert not refusal.success and refusal.output["failure_code"] == "stale_observation"
    assert refusal.output["environment_steps"] == 0
    assert env.reads > reads and len(env.inputs) == 1


class Planner:
    profile = "desktop"

    def __init__(self, *, initial_failure=False):
        self.calls = 0
        self.initial_failure = initial_failure

    def plan(self, goal, snapshot, history, *, max_actions):
        self.calls += 1
        if self.initial_failure and self.calls == 1:
            raise ValueError("synthetic invalid plan")
        if self.initial_failure:
            saved = any(c.label == "Saved" for c in snapshot.controls)
            step = _step("Finish", "Finished") if saved else _step()
            return AgentPlan("one recovery action", (step,), reusable=True)
        return AgentPlan("two ordinary actions", (_step(), _step("Finish", "Finished", depends=("save",))),
                         reusable=True)


@pytest.mark.parametrize("new,old,expected", [(False, True, 0), (True, False, 0), (True, True, 1)])
def test_runner_joint_optin_separate_metric_and_old_shape(new, old, expected):
    env = Environment()
    backend = OSWorldBackend(env, pause=0)
    trace = JsonlTrace()
    config = AgentConfig(mode="hybrid", allow_actions=True, max_steps=2, plan_horizon=2,
                         reuse_verified_post_snapshot=old, reuse_verified_pending_snapshot=new)
    runner = AgentRunner(backend, Planner(), policy=RulePolicy(), config=config, trace=trace)
    result = runner.run("Save then finish")
    assert len(env.inputs) == 2
    assert result.verified_pending_snapshot_reuses == expected
    pending_events = [e for e in trace.events if e["kind"] == "agent_verified_pending_observation_reuse"
                      and e["payload"]["source"] == "verified_ordinary_pending_effect_observation"]
    assert len(pending_events) == expected
    if expected:
        assert pending_events[0]["payload"]["sdk_receipt_fingerprint_unchanged"] is True
        assert len(pending_events[0]["payload"]["native_execution_fingerprint"]) == 64
        assert result.post_snapshot_reuses == 0 and result.recovery_planning_snapshot_reuses == 0
    else:
        assert "verified_pending_snapshot_reuses" not in result.to_dict()
    assert backend._input_evidence_enabled is False
    assert asdict(config)["reuse_verified_pending_snapshot"] is new
    backend.close()


def test_existing_invalid_plan_recovery_does_not_offer_pending_cache():
    env = Environment()
    backend = OSWorldBackend(env, pause=0)
    trace = JsonlTrace()
    result = AgentRunner(
        backend, Planner(initial_failure=True), policy=RulePolicy(), trace=trace,
        config=AgentConfig(mode="hybrid", allow_actions=True, max_steps=2, plan_horizon=2,
                           reuse_verified_pending_snapshot=True),
    ).run("Save then finish")
    assert len(env.inputs) == 2
    assert result.verified_pending_snapshot_reuses == 0
    assert not any(e["kind"] == "agent_verified_pending_observation_reuse" and
                   e["payload"]["source"] == "verified_ordinary_pending_effect_observation"
                   for e in trace.events)
    backend.close()
