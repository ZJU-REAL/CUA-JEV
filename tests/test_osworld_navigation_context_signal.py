"""Information routing uses actual guarded input and its final observed context."""

import io
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_types import AgentCondition, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend


def _observation(opened):
    image = io.BytesIO()
    Image.new("RGB", (800, 600), "gray" if opened else "white").save(image, format="PNG")
    role, name, button = ("dialog", "Options", "Accept") if opened else ("frame", "Main", "Open")
    xml = (
        '<desktop-frame xmlns:s="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:c="https://accessibility.ubuntu.example.org/ns/component">'
        f'<application name="Demo"><{role} name="{name}" s:showing="true" s:active="true">'
        f'<push-button name="{button}" s:showing="true" s:visible="true" s:enabled="true" '
        's:sensitive="true" c:screencoord="(80,120)" c:size="(120,40)"/>'
        f'</{role}></application></desktop-frame>'
    )
    return {"screenshot": image.getvalue(), "accessibility_tree": xml}


class Environment:
    def __init__(self, deferred=False):
        self.inputs = []
        self.reads = 0
        self.deferred = deferred

    def _get_obs(self):
        self.reads += 1
        return _observation(bool(self.inputs))

    def step(self, action, pause=0):
        self.inputs.append(action)
        return _observation(not self.deferred), 0, False, {}


def _execute(*, deferred=False, role="frame", enabled=True):
    env = Environment(deferred)
    backend = OSWorldBackend(env, pause=0)
    backend.set_input_observation_evidence_enabled(enabled)
    observed = backend.observe()
    step = AgentStep(
        "reach", "click", target=AgentTarget(app="Demo"), navigation=True,
        label="Reach the Options view", recovery="observe",
        postconditions=(AgentCondition("exists", AgentTarget(name="Options", role=role)),),
    )
    candidate = backend.compile(step, observed, observed)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and receipt.output["environment_steps"] == 1
    verification = backend.verify(candidate, receipt)
    if deferred:
        assert not verification.passed
        verification = backend.verify_pending(candidate, receipt)
    assert verification.passed
    return backend, env, candidate, receipt


@pytest.mark.parametrize("deferred", [False, True])
def test_unexplained_dialog_uses_real_final_frame_once_without_input_or_receipt_mutation(deferred):
    backend, env, candidate, receipt = _execute(deferred=deferred)
    try:
        original = receipt.to_dict()
        counts = (env.reads, len(env.inputs))
        evidence = backend.peek_input_observation_evidence(candidate, receipt)
        assert next(c for c in evidence.actual_final_after.controls if c.role == "dialog").disabled
        signal = backend.take_unexplained_navigation_context(candidate, receipt)
        assert signal["reason"] == "unexplained_new_dialog_after_verified_navigation"
        assert signal["evidence_source"] == ("verified_pending_full" if deferred else "sdk_after_input")
        assert signal["evidence_after_fingerprint"] == evidence.actual_final_after.fingerprint()
        assert signal["sdk_receipt_after_fingerprint"] == original["output"]["after_fingerprint"]
        assert signal["arrival_evidence"][0]["outcome"] != "true"
        assert signal["task_success"] is False and signal["input_authority"] is False
        assert receipt.to_dict() == original and (env.reads, len(env.inputs)) == counts
        assert backend.take_unexplained_navigation_context(candidate, receipt) is None
    finally:
        backend.close()


@pytest.mark.parametrize("change", ["disabled", "exact_arrival", "observe", "receipt", "candidate", "age"])
def test_unbound_superseded_expired_or_already_explained_state_never_requests_replanning(change, monkeypatch):
    backend, env, candidate, receipt = _execute(
        role="dialog" if change == "exact_arrival" else "frame", enabled=change != "disabled",
    )
    try:
        if change == "observe":
            backend.observe()
        elif change == "receipt":
            receipt = replace(receipt)
        elif change == "candidate":
            candidate = replace(candidate, description="modified")
        elif change == "age":
            now = backend._input_evidence.received_at + 1.001
            monkeypatch.setattr("cua_jev.osworld_backend.time.perf_counter", lambda: now)
        counts = (env.reads, len(env.inputs))
        assert backend.take_unexplained_navigation_context(candidate, receipt) is None
        assert (env.reads, len(env.inputs)) == counts
    finally:
        backend.close()
