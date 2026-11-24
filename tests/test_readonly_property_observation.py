"""One real preflight verifies a locked read-only field; no input is delegated."""

import copy
from dataclasses import replace

import pytest
from test_desktop_property_stage import Backend, Desktop, control, snapshot
from test_desktop_property_stage import image as image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import NativeDesktopEvidence
from cua_jev.models import Decision, Risk


class CountingDesktop(Desktop):
    def __init__(self, observed):
        super().__init__(observed)
        self.reads = 0
        self.on_read = None

    def _get_obs(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self)
        return self.snapshot


def data(*, value="100", conditions=None):
    result = {"intent": "pursue_property", "label": "Set the requested limit",
              "scope": {"app": "Editor"}, "edit": {"operation": "set", "value": value},
              "completion": "goal"}
    if conditions is not None:
        result["goal_conditions"] = conditions
    return result


def plan(observed, raw=None):
    parsed = parse_plan(raw or data(), observed, profile="desktop", max_actions=6,
                        allow_property_goal_conditions=True, allow_property_preparation=True,
                        allow_property_value_binding=True, allow_observed_property_goal=True)
    _validate_desktop_execution_plan(parsed, observed, 6)
    return parsed


def setup(image, *, kind="fill", value="100"):
    field = control("field", "Requested limit", field=True, value=value)
    request = data()
    if kind == "numeric":
        field = replace(field, role="spinbutton", input_type="number", edit_buffer=value)
    elif kind in {"check", "uncheck"}:
        field = replace(field, role="checkbox", input_type="checkbox", editable=False,
                        value="", checked=kind == "check")
        request = data(value="on" if kind == "check" else "off")
    elif kind == "option":
        field = replace(field, label="Light", role="cell", tag="button", input_type="",
                        editable=False, selectable=True, selected=True, value="")
        request = data(value="Light")
    observed = snapshot(image, (field,))
    env = CountingDesktop(observed)
    backend = Backend(env, pause=0)
    stage = plan(observed, request).steps[0]
    operation = "click" if kind == "option" else "fill" if kind == "numeric" else kind
    candidate = next(c for c in backend.compile(stage, observed, observed)
                     if c.arguments.get("property_phase") == "confirm"
                     and c.arguments["operation"] == operation)
    return observed, env, backend, candidate


@pytest.mark.parametrize("kind", ["fill", "numeric", "check", "uncheck", "option"])
def test_one_actual_new_observation_supplies_every_readonly_effect_and_no_cache(image, kind):
    observed, env, backend, candidate = setup(image, kind=kind)
    assert candidate.risk == Risk.READ_ONLY and backend._issued[candidate.id].confirmation is True
    assert env.reads == backend.fresh_observe_calls == backend.environment_steps == 0
    receipt = backend.execute(candidate, "observation", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and verification.passed
    assert receipt.output["property_confirmation"] and verification.details["property_confirmation"]
    assert receipt.output["observation_source"] == "fresh_preflight_read_only"
    assert verification.details["observation_source"] == "fresh_preflight_read_only"
    assert receipt.output["before_fingerprint"] == receipt.output["after_fingerprint"]
    assert receipt.output["after_fingerprint"] == observed.fingerprint()
    assert all(fact["satisfied"] for fact in verification.details["postcondition_evidence"])
    assert receipt.output["environment_steps"] == backend.environment_steps == 0 and env.actions == []
    assert env.reads == backend.fresh_observe_calls == 1
    assert backend._last_observed_snapshot is observed
    assert backend._action_observation is None and backend._post_snapshot is None
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert backend.verify_pending(candidate, receipt).passed
    assert backend.bound_property_step(candidate, receipt) is not None
    assert env.reads == backend.fresh_observe_calls == 1  # verify/reporting perform no I/O.
    backend.close()
    assert not backend._readonly_property_receipts


@pytest.mark.parametrize("change", ["value", "missing", "duplicate", "app", "window", "ancestor",
                                    "disabled", "incomplete", "numeric_buffer", "native_unknown"])
def test_drifted_field_owner_buffer_or_incomplete_observation_fails_without_any_input(image, change):
    kind = "numeric" if change == "numeric_buffer" else "check" if change == "native_unknown" else "fill"
    observed, env, backend, candidate = setup(image, kind=kind)
    field = observed.controls[0]
    controls = observed.controls
    if change == "missing":
        controls = ()
    elif change == "duplicate":
        controls = (*controls, replace(field, ref="twin"))
    elif change == "native_unknown":
        controls = (replace(field, native=NativeDesktopEvidence(complete=False)),)
    elif change != "incomplete":
        changes = {
            "value": {"value": "80"}, "app": {"app": "Editor Twin"},
            "window": {"window": "Other Main"}, "ancestor": {"ancestors": ("panel: Other",)},
            "disabled": {"disabled": True}, "numeric_buffer": {"edit_buffer": "101"},
        }[change]
        controls = (replace(field, **changes),)
    env.snapshot = replace(observed, controls=controls, a11y_complete=change != "incomplete")
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert env.actions == [] and receipt.output["environment_steps"] == backend.environment_steps == 0
    assert env.reads == backend.fresh_observe_calls == 1
    actual = backend.bound_property_step(candidate, receipt)
    assert actual is not None and actual.target.name == "Requested limit"
    assert actual.postconditions and not actual.navigation
    assert not backend.verify_pending(candidate, receipt).passed
    assert env.reads == 1 and backend._post_snapshot is None


@pytest.mark.parametrize("drift", ["property_effect", "postconditions", "unique_target", "checked_effect"])
def test_hidden_drift_cannot_turn_the_locked_readonly_candidate_into_input(image, monkeypatch, drift):
    kind = "check" if drift == "checked_effect" else "fill"
    _observed, env, backend, candidate = setup(image, kind=kind)
    if drift == "property_effect":
        monkeypatch.setattr("cua_jev.osworld_backend.property_effect_holds", lambda *_args: False)
    elif drift == "postconditions":
        monkeypatch.setattr("cua_jev.osworld_backend._postcondition_evidence",
                            lambda *_args: [{"satisfied": False, "outcome": "unknown"}])
    elif drift == "unique_target":
        monkeypatch.setattr(backend, "_target_after", lambda *_args, **_kwargs: None)
    else:
        monkeypatch.setattr("cua_jev.osworld_backend.checked_matches", lambda *_args: False)
    calls = []
    monkeypatch.setattr(backend, "_perform", lambda *_args, **_kwargs: calls.append("input_attempt"))
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert calls == env.actions == [] and receipt.output["environment_steps"] == 0
    assert env.reads == backend.fresh_observe_calls == 1


@pytest.mark.parametrize("change", ["id", "risk", "phase", "operation", "target", "description"])
def test_changed_public_candidate_cannot_use_or_downgrade_the_issued_readonly_binding(image, change):
    _observed, env, backend, candidate = setup(image)
    if change in {"id", "risk", "description"}:
        candidate = replace(candidate, **{"id": {"id": "forged"}, "risk": {"risk": Risk.LOCAL_WRITE},
                                          "description": {"description": "changed"}}[change])
    else:
        arguments = copy.deepcopy(candidate.arguments)
        arguments.update({"phase": {"property_phase": "edit"}, "operation": {"operation": "click"},
                          "target": {"target": {"name": "Another field", "app": "Editor"}}}[change])
        candidate = replace(candidate, arguments=arguments)
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert env.reads == backend.fresh_observe_calls == backend.environment_steps == 0 and env.actions == []


@pytest.mark.parametrize("change", ["unlocked", "missing_stage", "navigation", "target_owner", "effects"])
def test_altered_private_binding_does_not_authorize_a_readonly_field_or_any_input(image, change):
    _observed, env, backend, candidate = setup(image)
    binding = backend._issued[candidate.id]
    if change == "unlocked":
        binding = replace(binding, confirmation=False)
    elif change == "missing_stage":
        binding = replace(binding, property_step=None)
    elif change == "navigation":
        binding = replace(binding, step=replace(binding.step, navigation=True))
    elif change == "target_owner":
        binding = replace(binding, target=replace(binding.target, window="Other Main"))
    else:
        binding = replace(binding, step=replace(binding.step, postconditions=()))
    backend._issued[candidate.id] = binding
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert env.reads == backend.fresh_observe_calls == backend.environment_steps == 0 and env.actions == []


@pytest.mark.parametrize("change", ["copy", "id", "output", "candidate", "backend", "close"])
def test_readonly_receipt_identity_payload_and_owning_backend_remain_bound(image, change):
    observed, env, backend, candidate = setup(image)
    receipt = backend.execute(candidate, "observation", "decision")
    assert backend.verify(candidate, receipt).passed
    if change == "copy":
        receipt = copy.deepcopy(receipt)
    elif change == "id":
        receipt = replace(receipt, receipt_id="forged")
    elif change == "output":
        receipt.output["environment_steps"] = 9
    elif change == "candidate":
        candidate = replace(candidate, description="changed")
    elif change == "backend":
        backend = Backend(CountingDesktop(observed), pause=0)
    else:
        backend.close()
    assert not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert backend.bound_property_step(candidate, receipt) is None
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert env.reads == 1 and env.actions == []


@pytest.mark.parametrize("change", ["copy", "payload", "candidate", "success"])
def test_failed_readonly_receipt_reports_only_its_original_field_without_effect_authority(image, change):
    observed, env, backend, candidate = setup(image)
    env.snapshot = replace(observed, controls=(replace(observed.controls[0], value="80"),))
    receipt = backend.execute(candidate, "observation", "decision")
    actual = backend.bound_property_step(candidate, receipt)
    assert actual is not None and actual.operation == "fill" and not actual.navigation
    assert actual.target.name == "Requested limit" and actual.target.app == "Editor"
    assert {condition.kind for condition in actual.postconditions} == {"value_equals", "ready_to_fill"}
    assert not receipt.success and not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    if change == "copy":
        receipt = copy.deepcopy(receipt)
    elif change == "payload":
        receipt.output["environment_steps"] = 1
    elif change == "candidate":
        candidate = replace(candidate, description="different field")
    else:
        receipt = replace(receipt, success=True)
    assert backend.bound_property_step(candidate, receipt) is None
    assert not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert env.actions == [] and env.reads == backend.fresh_observe_calls == 1
    assert backend.environment_steps == 0


def test_real_property_fill_keeps_its_three_inputs_focus_reads_and_actual_response_cache(image):
    observed = snapshot(image, (control("field", "Requested limit", field=True, value="80"),))
    env = CountingDesktop(observed)
    backend = Backend(env, pause=0)
    candidate = next(c for c in backend.compile(plan(observed).steps[0], observed, observed)
                     if c.arguments["operation"] == "fill")
    assert candidate.risk == Risk.LOCAL_WRITE and not backend._issued[candidate.id].confirmation
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and backend.verify(candidate, receipt).passed
    assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert env.reads == backend.fresh_observe_calls == 3
    assert receipt.output["environment_steps"] == backend.environment_steps == 3
    assert "observation_source" not in receipt.output
    cached = backend.take_verified_post_snapshot(candidate, receipt)
    assert cached is not None and cached[0] is env.snapshot


class TypedPolicy:
    def choose(self, observation, candidates):
        chosen = next(c for c in candidates if c.capability == "agent.osworld.fill")
        return Decision(observation.observation_id, chosen.id,
                        {c.id: float(c is chosen) for c in candidates}, 1, "offline-readonly-selection", 0)


def run(observed, *, raw=None, on_read=None):
    env, events = CountingDesktop(observed), []
    env.on_read = on_read
    root = plan(observed, raw)

    class Planner:
        def plan(self, *_args, **_kwargs):
            return root

    backend = Backend(env, pause=0)
    result = AgentRunner(backend, Planner(), policy=TypedPolicy(), progress=events.append,
                         config=AgentConfig(allow_actions=True, max_plans=1, max_failures=1,
                                            max_steps=5)).run("Set the requested limit")
    return result, backend, env, events


def test_final_whole_goal_still_uses_a_later_real_observation(image):
    observed = snapshot(image, (control("field", "Requested limit", field=True, value="100"),))
    reads = []

    def change_after_confirmation(env):
        reads.append(env.reads)
        if env.reads == 4:
            env.snapshot = replace(env.snapshot, controls=(replace(env.snapshot.controls[0], value="80"),))

    result, backend, env, events = run(observed, on_read=change_after_confirmation)
    assert result.steps == result.property_confirmations == 1 and result.property_edits == 0
    assert any(event["event"] == "agent_property_goal_bound" for event in events)
    assert not result.plan_completion_claims and not result.success
    assert reads == [1, 2, 3, 4] and backend.fresh_observe_calls == 4
    assert env.actions == [] and backend.environment_steps == 0


def test_failed_readonly_recovery_retains_the_concrete_field_and_effect_conditions(image):
    observed = snapshot(image, (control("field", "Requested limit", field=True, value="100"),))

    def change_before_confirmation(env):
        if env.reads == 3:
            env.snapshot = replace(env.snapshot, controls=(replace(env.snapshot.controls[0], value="80"),))

    result, backend, env, events = run(observed, on_read=change_before_confirmation)
    recovery = next(event for event in events if event["event"] == "agent_recovery")
    assert recovery["operation"] == "fill"
    assert recovery["target"]["name"] == "Requested limit"
    assert {condition["kind"] for condition in recovery["postconditions"]} == {
        "value_equals", "ready_to_fill",
    }
    assert not recovery["verified"] and result.property_confirmations == result.property_edits == 0
    assert not result.success and not result.plan_completion_claims
    assert env.actions == [] and backend.environment_steps == 0 and env.reads == 3


@pytest.mark.parametrize("state", ["false", "unknown"])
def test_readonly_field_does_not_drop_explicit_false_or_unknown_whole_goal_condition(image, state):
    field = control("field", "Requested limit", field=True, value="100")
    controls = ((field, control("guard", "Required guard", field=True, value="bad"))
                if state == "false" else (field,))
    observed = snapshot(image, controls)
    raw = data(conditions=[{"kind": "value_equals", "value": "good",
                            "target": {"name": "Required guard", "app": "Editor", "role": "textbox"}}])
    result, backend, env, events = run(observed, raw=raw)
    assert result.steps == result.property_confirmations == 1
    assert not result.plan_completion_claims and not result.success
    bound = next(event for event in events if event["event"] == "agent_property_goal_bound")
    assert bound["explicit_conditions_retained"]
    assert env.actions == [] and backend.environment_steps == 0 and env.reads >= 4


def test_readonly_field_and_new_whole_goal_read_can_complete_advisory_stage_without_input(image):
    observed = snapshot(image, (control("field", "Requested limit", field=True, value="100"),))
    result, backend, env, events = run(observed)
    assert result.plan_completion_claims == 1 and result.property_confirmations == 1
    assert result.status == "unverified" and not result.success
    assert env.reads == backend.fresh_observe_calls == 4 and env.actions == []
    assert any(event["event"] == "agent_plan_completion" for event in events)
