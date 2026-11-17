"""Regression tests for navigation admission and evidence."""

import copy
import json
from dataclasses import replace

import pytest
from test_navigation_continuation_runner_draft import MenuEnvironment, Planner, ScriptedPolicy

from cua_jev.desktop_navigation_memory import NavigationMemory
from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Observation, Verification
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime, DecisionAdmissionRejected, DecisionAdmissionRejection
from cua_jev.trace import JsonlTrace
from cua_jev.verify import VerifierRegistry


def test_typed_hook_preserves_choice_and_guard_but_rejects_before_any_executor_or_receipt():
    observation = Observation("Navigate", "Reach destination", {})
    candidates = [ActionCandidate("current", Channel.GUI, "fixture.click", "click Parent [current]")]
    inputs, seen = [], []
    registry, trace = ExecutorRegistry(), JsonlTrace()
    registry.register(Channel.GUI, lambda *args: inputs.append(args))
    rejection = DecisionAdmissionRejection("reach", "a" * 64, "b" * 64)

    def admission(obs, decision, candidate):
        seen.append((obs, decision, candidate))
        return rejection

    runtime = AgentRuntime(policy=ScriptedPolicy(["Parent"]), guard=ActionGuard(),
                           executors=registry, trace=trace, decision_admission=admission)
    before = copy.deepcopy([c.to_dict() for c in candidates])
    with pytest.raises(DecisionAdmissionRejected) as caught:
        runtime.step(observation, candidates)
    assert caught.value.rejection == rejection and inputs == []
    assert [c.to_dict() for c in candidates] == before
    assert seen[0][2] is not candidates[0] and seen[0][2].to_dict() == candidates[0].to_dict()
    assert next(e["payload"] for e in trace.events if e["kind"] == "guard")["approved"] is True
    assert not any(e["kind"] in {"receipt", "verification", "policy_failure"} for e in trace.events)


def test_default_none_hook_has_no_new_event_or_authority():
    observation = Observation("Navigate", "Reach destination", {})
    candidate = ActionCandidate("current", Channel.GUI, "fixture.click", "click Parent [current]",
                                verifier="fixture.effect")
    inputs, registry, verifiers, trace = [], ExecutorRegistry(), VerifierRegistry(), JsonlTrace()

    def execute(action, observation_id, decision_id):
        inputs.append(action)
        return ActionReceipt(observation_id, decision_id, action.id, action.channel,
                             action.capability, True, 0, 0, {"environment_steps": 1})

    registry.register(Channel.GUI, execute)
    verifiers.register("fixture.effect", lambda *_: Verification(True, "fixture.effect"))
    result = AgentRuntime(policy=ScriptedPolicy(["Parent"]), guard=ActionGuard(),
                          executors=registry, verifiers=verifiers, trace=trace).step(observation, [candidate])
    assert result.verification.passed and inputs == [candidate]
    assert not any(e["kind"] == "decision_admission" for e in trace.events)


def loop_evidence():
    environment = MenuEnvironment()
    backend = OSWorldBackend(environment, pause=0)
    backend.set_input_observation_evidence_enabled(True)
    before = backend.observe()
    stage = Planner().plan("Navigate", before, [], max_actions=6).steps[0]
    memory = NavigationMemory(6)
    memory.reset(1)
    for index in range(2):
        current = backend.observe()
        offered = backend.compile(stage, before, current)
        chosen = next(c for c in offered if c.arguments["bound_control"]["label"] == "Parent"
                      and (c.arguments["button"], c.arguments["click_count"]) == ("left", 1))
        receipt = backend.execute(chosen, f"observation-{index}", f"decision-{index}")
        backend.verify(chosen, receipt)
        evidence = backend.navigation_transition_evidence(chosen, receipt)
        assert evidence is not None and receipt.output["environment_steps"] == 1
        # The facade is a non-consuming peek of this exact bounded actual record.
        assert backend.navigation_transition_evidence(chosen, receipt) == evidence
        witness = memory.record(evidence, final_verified=index == 0, cycle=index == 1)
    assert witness is not None
    memory.request(witness)
    backend.clear_navigation_memory_sources()
    return environment, backend, stage, before, memory, chosen, receipt


def test_unique_rebinding_uses_new_ids_preserves_full_pool_and_small_wire():
    environment, backend, stage, planned, memory, old_chosen, receipt = loop_evidence()
    try:
        current = backend.observe()
        offered = backend.compile(stage, planned, current)
        original = copy.deepcopy([c.to_dict() for c in offered])
        state = backend.navigation_memory_state(stage, current)
        assert state is not None and memory.bind_offers(stage.id, state, offered) is None
        memory.bind_observation("fresh")
        wire = memory.public()
        ids = {c.id for c in offered}
        mapped = [e["candidate_id"] for e in wire["edges"] if e["binding"] == "current_unique"]
        assert mapped and set(mapped) <= ids and old_chosen.id not in ids
        exhausted = next(c for c in offered if memory.decline("fresh", c.id) is not None)
        alternative = next(c for c in offered if c.arguments["bound_control"]["label"] == "Destination"
                           and (c.arguments["button"], c.arguments["click_count"]) == ("left", 1))
        assert memory.decline("fresh", alternative.id) is None and exhausted.id != alternative.id
        assert [c.to_dict() for c in offered] == original
        text = json.dumps(wire, ensure_ascii=False)
        assert receipt.receipt_id not in text and state.native not in text and state.scope not in text
        assert all(key not in text for key in ['"arguments"', '"box"', '"value"'])
        assert len(wire["edges"]) <= 6 and all(len(e["arrival"]) <= 8 for e in wire["edges"])
    finally:
        backend.close()


def test_backend_peek_rejects_changed_receipt_candidate_and_counter_without_new_io():
    environment = MenuEnvironment()
    backend = OSWorldBackend(environment, pause=0)
    backend.set_input_observation_evidence_enabled(True)
    try:
        before = backend.observe()
        stage = Planner().plan("Navigate", before, [], max_actions=6).steps[0]
        candidate = next(c for c in backend.compile(stage, before, before)
                         if c.arguments["bound_control"]["label"] == "Parent"
                         and (c.arguments["button"], c.arguments["click_count"]) == ("left", 1))
        receipt = backend.execute(candidate, "o", "d")
        backend.verify(candidate, receipt)
        reads, inputs = environment.reads, len(environment.inputs)
        assert backend.navigation_transition_evidence(candidate, receipt) is not None
        changed = replace(receipt, output={**receipt.output, "environment_steps": 0})
        assert backend.navigation_transition_evidence(candidate, changed) is None
        changed_candidate = replace(candidate, arguments={**candidate.arguments, "button": "right"})
        assert backend.navigation_transition_evidence(changed_candidate, receipt) is None
        assert (environment.reads, len(environment.inputs)) == (reads, inputs)
        backend.observe()
        assert backend.navigation_transition_evidence(candidate, receipt) is None
    finally:
        backend.close()


def test_scope_equality_does_not_make_changed_native_focus_a_known_state():
    _environment, backend, stage, planned, memory, _candidate, _receipt = loop_evidence()
    try:
        current = backend.observe()
        state = backend.navigation_memory_state(stage, current)
        changed = replace(
            current, controls=tuple(replace(c, focused=not c.focused) for c in current.controls)
        )
        other = backend.navigation_memory_state(stage, changed)
        assert state.scope == other.scope and state.native != other.native
        offered = backend.compile(stage, planned, changed)
        assert memory.bind_offers(stage.id, other, offered) == "navigation_continuation_evidence_changed"
        assert not memory.local_attempt_active
    finally:
        backend.close()


def test_same_identity_aliases_cannot_bind_a_history_edge_or_authorize_ref_reuse():
    _environment, backend, stage, planned, memory, _candidate, _receipt = loop_evidence()
    try:
        current = backend.observe()
        original = next(c for c in current.controls if c.label == "Parent")
        duplicate = replace(original, ref="different-current-ref", box=(500, 40, 680, 80))
        ambiguous = replace(current, controls=(*current.controls, duplicate))
        state = backend.navigation_memory_state(stage, ambiguous)
        pool = backend.compile(stage, planned, ambiguous)
        assert memory.bind_offers(stage.id, state, pool) == "navigation_continuation_evidence_changed"
        assert not memory.local_attempt_active
    finally:
        backend.close()


def test_lost_owner_evidence_does_not_become_a_known_state_even_with_the_old_hash():
    _environment, backend, stage, planned, memory, _candidate, _receipt = loop_evidence()
    try:
        current = backend.observe()
        state = backend.navigation_memory_state(stage, current)
        changed_owner = replace(state, owner=("Demo", "Different observed owner", ""))
        pool = backend.compile(stage, planned, current)
        assert memory.bind_offers(stage.id, changed_owner, pool) == "navigation_continuation_evidence_changed"
        assert not memory.local_attempt_active
    finally:
        backend.close()


def test_empty_current_pool_cannot_spend_a_second_observation_as_a_legal_edge():
    _environment, backend, stage, _planned, memory, _candidate, _receipt = loop_evidence()
    try:
        current = backend.observe()
        state = backend.navigation_memory_state(stage, current)
        assert memory.bind_offers(stage.id, state, ()) == "navigation_continuation_evidence_changed"
        assert not memory.local_attempt_active
    finally:
        backend.close()


def test_same_identity_read_only_menu_alias_cannot_authorize_transition_or_concession():
    """The original ambiguous fixture is retained as a real-backend negative."""
    from test_navigation_continuation_runner_draft import ObservedBackend

    from cua_jev.agent_runner import AgentConfig, AgentRunner

    class AliasedMenuEnvironment(MenuEnvironment):
        def _get_obs(self):
            observed = super()._get_obs()
            observed["accessibility_tree"] = observed["accessibility_tree"].replace(
                '<menu name="Parent popup">', '<menu name="Parent">',
            )
            return observed

    environment = AliasedMenuEnvironment()
    backend = OSWorldBackend(environment, pause=0)
    backend.set_input_observation_evidence_enabled(True)
    try:
        before = backend.observe()
        stage = Planner().plan("Navigate", before, [], max_actions=6).steps[0]
        offered = backend.compile(stage, before, before)
        parent = [c for c in before.controls if c.label == "Parent"]
        assert len(parent) == 2 and sum(c.box is not None for c in parent) == 1
        current = [c for c in offered if c.arguments["bound_control"]["label"] == "Parent"
                   and (c.arguments["button"], c.arguments["click_count"]) == ("left", 1)]
        assert len(current) == 1  # Do not confuse a legal-pool singleton with identity proof.
        chosen = current[0]
        receipt = backend.execute(chosen, "original-aliased-observation", "original-aliased-decision")
        verification = backend.verify(chosen, receipt)
        assert receipt.output["environment_steps"] == 1 and verification.passed is True
        assert backend.peek_input_observation_evidence(chosen, receipt) is not None
        assert backend.navigation_transition_evidence(chosen, receipt) is None
    finally:
        backend.close()

    environment = AliasedMenuEnvironment()
    backend, planner, trace = ObservedBackend(environment), Planner(), JsonlTrace()
    try:
        result = AgentRunner(
            backend, planner, policy=ScriptedPolicy(["Parent", "Parent", "Destination"]), trace=trace,
            config=AgentConfig(allow_actions=True, navigation_continuation=True,
                               reuse_verified_post_snapshot=False,
                               reuse_verified_pending_snapshot=False),
        ).run("Reach the requested Destination")
        assert [request[0] for request in planner.requests] == [6, 1]
        assert environment.arrived and len(environment.inputs) == result.steps == 3
        plans = [e for e in trace.events if e["kind"] == "agent_plan"]
        assert [(e["payload"]["max_actions"], e["payload"]["recovery"]) for e in plans] == [
            (6, False), (1, True),
        ]
        # A genuine original H1 recovery remains outside the new continuation authority.
        assert not any(e["kind"] == "agent_navigation_transition"
                       for e in trace.events if e["sequence"] > plans[-1]["sequence"])
        assert not any(e["kind"] == "agent_navigation_continuation"
                       and e["payload"]["stage"] == "fresh_continuation_requested"
                       for e in trace.events)
    finally:
        backend.close()
