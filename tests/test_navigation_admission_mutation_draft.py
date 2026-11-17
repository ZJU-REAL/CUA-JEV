"""Regression tests for navigation admission mutation."""

import json

import pytest

from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Decision, Observation, Verification
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import (
    AgentRuntime,
    DecisionAdmissionContractError,
    DecisionAdmissionRejected,
    DecisionAdmissionRejection,
    PolicySelectionFailure,
)
from cua_jev.trace import JsonlTrace
from cua_jev.verify import VerifierRegistry


class Policy:
    def choose(self, observation, candidates):
        self.decision = Decision(observation.observation_id, candidates[0].id,
                                 {c.id: float(i == 0) for i, c in enumerate(candidates)},
                                 1.0, "fixture", 0.0)
        return self.decision


def fixture(hook):
    observation = Observation("Navigate", "Reach target", {
        "controls": [{"expanded": True}], "data": {"items": [1, 2]},
    })
    candidates = [ActionCandidate(
        f"candidate-{i}", Channel.GUI, "fixture.click", f"current control {i}",
        arguments={"enabled": True, "nested": {"label": "public"}},
        expected={"field": {"ready": True}}, verifier="fixture.effect",
    ) for i in range(2)]
    inputs, policy, trace = [], Policy(), JsonlTrace()
    executors, verifiers = ExecutorRegistry(), VerifierRegistry()

    def execute(candidate, observation_id, decision_id):
        inputs.append(candidate)
        return ActionReceipt(observation_id, decision_id, candidate.id, candidate.channel,
                             candidate.capability, True, 0, 0, {"environment_steps": 1})

    executors.register(Channel.GUI, execute)
    verifiers.register("fixture.effect", lambda *_: Verification(True, "fixture.effect"))
    shared = {"observation": observation, "candidates": candidates, "policy": policy}
    callback = (
        None if hook is None else lambda obs, decision, candidate: hook(shared, obs, decision, candidate)
    )
    runtime = AgentRuntime(policy=policy, guard=ActionGuard(), executors=executors,
                           verifiers=verifiers, trace=trace, decision_admission=callback)
    return runtime, shared, inputs, trace


@pytest.mark.parametrize("field", ["observation", "arguments", "expected", "probabilities", "nonfinite"])
@pytest.mark.parametrize("result_kind", ["none", "typed_reject"])
def test_nested_clone_mutation_is_blocked_even_when_callback_returns_a_valid_type(field, result_kind):
    def mutate(shared, obs, decision, candidate):
        assert obs is not shared["observation"] and candidate is not shared["candidates"][0]
        assert decision is not shared["policy"].decision
        if field == "observation":
            obs.state["controls"][0]["expanded"] = 1  # True == 1 must not mask this change.
        elif field == "arguments":
            candidate.arguments["nested"]["label"] = "mutated-private-string"
        elif field == "expected":
            candidate.expected["field"]["ready"] = False
        elif field == "probabilities":
            decision.probabilities[candidate.id] = True
        else:
            candidate.arguments["nonfinite"] = float("nan")
        return (None if result_kind == "none" else
                DecisionAdmissionRejection("reach", "a" * 64, "b" * 64))

    runtime, shared, inputs, trace = fixture(mutate)
    with pytest.raises(DecisionAdmissionContractError) as caught:
        runtime.step(shared["observation"], shared["candidates"])
    assert caught.value.reason == "hook_value_mutation" and inputs == []
    assert shared["observation"].state["controls"][0]["expanded"] is True
    assert shared["candidates"][0].arguments["nested"]["label"] == "public"
    assert shared["candidates"][0].expected["field"]["ready"] is True
    assert shared["policy"].decision.probabilities["candidate-0"] == 1.0
    assert not any(e["kind"] in {"receipt", "verification", "policy_failure"} for e in trace.events)
    assert "mutated-private-string" not in json.dumps(trace.events)


@pytest.mark.parametrize("field", ["observation", "chosen", "unchosen", "probabilities", "order", "ids"])
def test_external_reference_mutation_of_shared_guard_data_or_offers_is_also_blocked(field):
    def mutate(shared, _obs, _decision, _candidate):
        if field == "observation":
            shared["observation"].state["data"]["items"].append(3)
        elif field == "chosen":
            shared["candidates"][0].arguments["enabled"] = 1
        elif field == "unchosen":
            shared["candidates"][1].expected["field"]["ready"] = False
        elif field == "probabilities":
            shared["policy"].decision.probabilities["candidate-0"] = True
        elif field == "order":
            shared["candidates"].reverse()
        else:
            object.__setattr__(shared["candidates"][1], "id", "mutated-private-id")
        return None

    runtime, shared, inputs, trace = fixture(mutate)
    with pytest.raises(DecisionAdmissionContractError) as caught:
        runtime.step(shared["observation"], shared["candidates"])
    assert caught.value.reason == "hook_value_mutation" and inputs == []
    assert not any(e["kind"] in {"receipt", "verification", "policy_failure"} for e in trace.events)
    assert "mutated-private-id" not in json.dumps(trace.events)


@pytest.mark.parametrize("kind", ["ordinary", "fake_policy_failure"])
def test_callback_exception_is_sanitized_and_cannot_enter_selection_failure_recovery(kind):
    def bad_callback(*_args):
        if kind == "fake_policy_failure":
            raise PolicySelectionFailure(None)
        raise RuntimeError("private callback exception detail")

    runtime, shared, inputs, trace = fixture(bad_callback)
    with pytest.raises(DecisionAdmissionContractError) as caught:
        runtime.step(shared["observation"], shared["candidates"])
    assert caught.value.reason == "callback_failed" and inputs == []
    assert not isinstance(caught.value, (PolicySelectionFailure, DecisionAdmissionRejected))
    assert "private callback exception detail" not in json.dumps(trace.events)
    assert not any(e["kind"] == "policy_failure" for e in trace.events)


def test_nonmutating_typed_rejection_still_declines_without_receipt():
    expected = DecisionAdmissionRejection("reach", "a" * 64, "b" * 64)
    runtime, shared, inputs, trace = fixture(lambda *_: expected)
    with pytest.raises(DecisionAdmissionRejected) as caught:
        runtime.step(shared["observation"], shared["candidates"])
    assert caught.value.rejection == expected and inputs == []
    assert not any(e["kind"] == "receipt" for e in trace.events)


def test_wrong_or_forged_result_cannot_reach_executor():
    def forged(*_args):
        record = DecisionAdmissionRejection("reach", "a" * 64, "b" * 64)
        object.__setattr__(record, "edge_key", {"private": "not a hash"})
        return record

    runtime, shared, inputs, trace = fixture(forged)
    with pytest.raises(DecisionAdmissionContractError) as caught:
        runtime.step(shared["observation"], shared["candidates"])
    assert caught.value.reason == "invalid_hook_result" and inputs == []
    assert "not a hash" not in json.dumps(trace.events)


def test_none_hook_does_not_call_value_binding_or_copy_path(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("hook-only projection must not run")

    monkeypatch.setattr("cua_jev.runtime._admission_json", forbidden)
    monkeypatch.setattr("cua_jev.runtime._admission_hook_values", forbidden)
    runtime, shared, inputs, trace = fixture(None)
    result = runtime.step(shared["observation"], shared["candidates"])
    assert result.verification.passed and inputs == [shared["candidates"][0]]
    assert not any(e["kind"] == "decision_admission" for e in trace.events)
