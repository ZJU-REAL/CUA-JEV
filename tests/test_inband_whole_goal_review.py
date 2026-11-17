"""Regression tests for inband whole goal review."""

import json
import time
from dataclasses import replace

import httpx
import pytest
from test_desktop_output_schema_contract import observed as observed
from test_semantic_tools_planner import model, packet
from test_verified_recovery_planning_snapshot import (
    Backend as ReceiptBackend,
)
from test_verified_recovery_planning_snapshot import (
    Environment,
    check_step,
    checked,
    source,
)
from test_verified_recovery_planning_snapshot import (
    clock as clock,
)
from test_verified_recovery_planning_snapshot import (
    snapshot as snapshot,
)
from test_whole_goal_recheck import GOAL, InputBackend, finished, run, stage

from cua_jev.agent_runner import (
    AgentConfig,
    AgentRunner,
    DirectPlanPolicy,
    _take_verified_recovery_planning_snapshot,
)
from cua_jev.agent_types import AgentPlan
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace

_run_with_original_defaults = run


def run(*args, **kwargs):
    kwargs.setdefault("inband_whole_goal_recheck", True)
    return _run_with_original_defaults(*args, **kwargs)


class EpochBackend(InputBackend):
    def __init__(self):
        super().__init__()
        self.environment_steps = 0
        self.fresh_observe_calls = 0
        self._closed = False

    def observe(self):
        result = replace(super().observe(), a11y_complete=True)
        self.fresh_observe_calls += 1
        self._last_observed_snapshot = result
        self._last_observed_received_at = time.perf_counter()
        return result

    current_full_observation_proof = OSWorldBackend.current_full_observation_proof

    def execute(self, *args):
        self.environment_steps += 1
        return super().execute(*args)


class InbandPlanner:
    profile = "desktop"
    supports_inband_whole_goal_recheck = True

    def __init__(self, answers, *, marker=True, after_return=None):
        self.answers = iter(answers)
        self.calls = []
        self.last_request_provenance = None
        self.marker = marker
        self.after_return = after_return

    def plan(self, goal, snapshot, history, *, max_actions=6,
             whole_goal_recheck=False, whole_goal_inband_epoch=None):
        self.last_request_provenance = None
        self.calls.append({"goal": goal, "limit": max_actions, "review": whole_goal_recheck,
                           "inband_epoch": whole_goal_inband_epoch})
        if self.marker and whole_goal_inband_epoch is not None:
            self.last_request_provenance = {
                "image_uploaded": snapshot.image is not None, "a11y_complete": snapshot.a11y_complete,
                "whole_goal_inband_review": {
                    "version": "all_original_goal_review_v1", "input_epoch": whole_goal_inband_epoch,
                },
            }
        answer = next(self.answers)
        if self.after_return:
            self.after_return(self)
        return answer


def already_complete():
    backend = EpochBackend()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    return backend


@pytest.mark.parametrize("mode,limit", [("vlm", 1), ("hybrid", 6)])
def test_normal_empty_review_keeps_fresh_check_and_saves_only_second_dedicated_call(mode, limit):
    backend = already_complete()
    planner = InbandPlanner([finished()])
    result, backend, planner, trace = run([], backend=backend, planner=planner, mode=mode)
    assert len(planner.calls) == result.planner_calls == 1
    assert planner.calls[0] == {"goal": GOAL, "limit": limit, "review": False, "inband_epoch": 0}
    assert backend.observations >= 2 and not backend.executed
    assert result.status == "unverified" and result.success is False
    assert result.whole_goal_inband_review_calls == result.whole_goal_inband_review_done == 1
    assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 0
    assert any(e["kind"] == "agent_whole_goal_inband_review" and
               e["payload"]["stage"] == "empty_done_fresh_checked" for e in trace.events)


def test_action_bearing_local_completion_still_dedicated_reviews_omitted_requirement():
    backend = EpochBackend()
    planner = InbandPlanner([stage("Name", "Ada"), stage("City", "Hangzhou"), finished()])
    result, backend, planner, _ = run([], backend=backend, planner=planner)
    assert backend.executed == ["Name", "City"]
    assert [call["review"] for call in planner.calls] == [False, True, True]
    assert result.whole_goal_recheck_calls == 2 and result.whole_goal_inband_review_done == 0


@pytest.mark.parametrize("failure", ["missing_marker", "bool_epoch", "new_input", "new_observation",
                                      "no_image_marker", "unconditional", "old_planner"])
def test_unknown_or_changed_context_does_not_bypass_dedicated_review(failure):
    backend = already_complete()

    def after(planner):
        if len(planner.calls) != 1:
            return
        if failure == "bool_epoch":
            planner.last_request_provenance["whole_goal_inband_review"]["input_epoch"] = False
        elif failure == "new_input":
            backend.environment_steps += 1  # Simulated input during model latency, no GUI.
        elif failure == "new_observation":
            backend.observe()
        elif failure == "no_image_marker":
            planner.last_request_provenance["image_uploaded"] = False

    first = (AgentPlan("Unconditional advisory only", (), done=True)
             if failure == "unconditional" else finished())
    planner = InbandPlanner([first, finished()], marker=failure != "missing_marker", after_return=after)
    if failure == "old_planner":
        planner.supports_inband_whole_goal_recheck = False
    result, backend, planner, _ = run([], backend=backend, planner=planner)
    assert len(planner.calls) == 2 and planner.calls[1]["review"] is True
    assert result.whole_goal_inband_review_done == 0 and result.whole_goal_recheck_done == 1
    assert not backend.executed


def test_model_empty_done_cannot_use_pre_model_conditions_when_final_fresh_state_changes():
    backend = already_complete()
    original_observe = backend.observe

    def changed_final():
        if backend.observations >= 1:
            backend.values["City"] = "not complete now"
        return original_observe()

    backend.observe = changed_final
    result, _, _, _ = run([], backend=backend, planner=InbandPlanner([finished()]), max_plans=1)
    assert result.whole_goal_inband_review_done == result.whole_goal_recheck_done == 0
    assert result.status != "unverified" and not backend.executed


def test_disabled_feature_sends_no_new_keyword_and_omits_new_result_metrics():
    planner = InbandPlanner([finished()])
    result, _, planner, trace = run([], backend=already_complete(), planner=planner, enabled=False)
    assert planner.calls[0]["inband_epoch"] is None and planner.calls[0]["review"] is False
    assert not any(key.startswith("whole_goal_inband") for key in result.to_dict())
    assert not any(e["kind"] == "agent_whole_goal_inband_review" for e in trace.events)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_sdk_receipt_planning_frame_can_review_without_being_called_fresh(clock, snapshot, enabled):
    env, trace = Environment(snapshot), JsonlTrace()

    class RecordingReceiptBackend(ReceiptBackend):
        def __init__(self, environment):
            super().__init__(environment)
            self.proof_records = []

        def current_full_observation_proof(self, current, *, expected=None, source=None):
            counters = self.environment_steps, self.fresh_observe_calls
            was_latest_observe = current is self._last_observed_snapshot
            proof = super().current_full_observation_proof(current, expected=expected, source=source)
            self.proof_records.append((current, source, expected, counters, was_latest_observe, proof))
            return proof

    backend = RecordingReceiptBackend(env)

    class RecoveryPlanner:
        profile = "desktop"
        desktop_contract = "property_intent_v52"
        desktop_output_format = "semantic_tools_v56"
        supports_inband_whole_goal_recheck = True
        last_request_provenance = None

        def __init__(self):
            self.calls = []
            self.frames = []
            self.dispatch_counts = []

        def plan(self, goal, current, history, *, max_actions=6,
                 whole_goal_recheck=False, whole_goal_inband_epoch=None):
            self.calls.append((max_actions, whole_goal_recheck, whole_goal_inband_epoch))
            self.frames.append(current)
            self.dispatch_counts.append((backend.environment_steps, backend.fresh_observe_calls))
            self.last_request_provenance = {"planning_observation": "full_multimodal_v1",
                "max_actions": max_actions, "image_uploaded": True, "a11y_complete": True}
            if whole_goal_inband_epoch is not None:
                self.last_request_provenance["whole_goal_inband_review"] = {
                    "version": "all_original_goal_review_v1", "input_epoch": whole_goal_inband_epoch}
            if enabled:
                assert not whole_goal_recheck, "eligible normal completion must not create a fourth call"
            elif len(self.calls) == 4:
                assert whole_goal_recheck and whole_goal_inband_epoch is None
                return AgentPlan("Original dedicated whole-goal review", (), done=True,
                                 completion_conditions=(checked("Alpha"),))
            if len(self.calls) == 1:
                raise ValueError("fixture broad-plan validation failed without input")
            if len(self.calls) == 2:
                return AgentPlan("Recover one current option", (check_step("Alpha"),), reusable=True)
            assert len(self.calls) == 3
            return AgentPlan("Entire Alpha goal observed", (), done=True,
                             completion_conditions=(checked("Alpha"),))

    planner = RecoveryPlanner()
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, whole_goal_recheck=True,
            inband_whole_goal_recheck=enabled,
            reuse_verified_post_snapshot=True, reuse_verified_pending_snapshot=True,
            max_steps=6, max_plans=4, timeout_s=300, plan_ttl_s=300)).run("Enable Alpha")
    if not enabled:
        assert result.planner_calls == 4 and result.recovery_planning_snapshot_reuses == 1
        assert result.whole_goal_recheck_calls == result.whole_goal_recheck_done == 1
        assert result.whole_goal_inband_review_calls == result.whole_goal_inband_review_done == 0
        assert planner.calls == [(6, False, None), (1, False, None),
                                 (6, False, None), (1, True, None)]
        assert backend.proof_records == [] and backend.take_calls == 1 and len(env.inputs) == 1
        assert backend.fresh_observe_calls == planner.dispatch_counts[-1][1] + 1
        assert result.status == "unverified" and not result.success
        assert not any(e["kind"] == "agent_whole_goal_inband_review" for e in trace.events)
        return
    assert result.planner_calls == 3 and result.recovery_planning_snapshot_reuses == 1
    assert result.whole_goal_recheck_calls == 0 and result.whole_goal_inband_review_done == 1
    requests = [e["payload"] for e in trace.events if e["kind"] == "agent_whole_goal_inband_review"
                and e["payload"]["stage"] == "requested" and e["payload"]["call"] == 3]
    assert len(requests) == 1
    assert requests[0]["observation_source"] == "verified_h1_recovery_step_response_for_planning"
    assert backend.take_calls == 1 and len(env.inputs) == 1
    assert planner.calls == [(6, False, 0), (1, False, 0), (6, False, 1)]
    sdk_records = [record for record in backend.proof_records if record[1] is not None]
    assert len(sdk_records) == 3
    initial = sdk_records[0]
    current, receipt_source, expected, counters, was_latest_observe, proof = initial
    assert current is planner.frames[2] and expected is None and not was_latest_observe
    assert type(receipt_source) is tuple and len(receipt_source) == 2
    candidate, receipt = receipt_source
    assert proof is not None
    assert receipt.success and receipt.output["after_fingerprint"] == proof[3]
    assert candidate is sdk_records[1][1][0] is sdk_records[2][1][0]
    assert proof is not None and proof[1:3] == counters == planner.dispatch_counts[2]
    assert all(record[0] is current and record[1][1] is receipt and record[2] == proof
               and record[3] == counters and record[4] is False and record[5] == proof
               for record in sdk_records[1:])
    assert requests[0]["native_execution_fingerprint"] == proof[3]
    assert requests[0]["image_sha256"] == proof[4]
    # Planning SDK-after is never renamed fresh. Exactly one subsequent real read
    # checks the empty-done conditions; no new input or fourth planner call occurs.
    assert (backend.environment_steps, backend.fresh_observe_calls) == (counters[0], counters[1] + 1)
    assert result.status == "unverified" and not result.success


@pytest.mark.parametrize("invalidate", ["read", "input", "age", "receipt"])
def test_sdk_proof_uses_original_record_and_falls_back_on_actual_invalidation(clock, snapshot, invalidate):
    backend = ReceiptBackend(Environment(snapshot))
    backend.set_input_observation_evidence_enabled(True)
    backend.observe()
    candidate, receipt, fingerprint = source(backend)
    frame = _take_verified_recovery_planning_snapshot(backend, (candidate, receipt, fingerprint))
    assert frame is not None
    proof = backend.current_full_observation_proof(frame[0], source=(candidate, receipt))
    assert proof is not None and proof[3] == fingerprint
    if invalidate == "read":
        backend.observe()
    elif invalidate == "input":
        backend.environment_steps += 1
    elif invalidate == "age":
        clock.advance(1.001)
    else:
        receipt.output["after_fingerprint"] = "0" * 64
    assert backend.current_full_observation_proof(
        frame[0], source=(candidate, receipt), expected=proof,
    ) is None


@pytest.mark.parametrize("requested,recovery,effective", [(1, False, 1), (6, False, 6), (6, True, 1)])
def test_production_planner_one_post_all_original_goal_review_preserves_normal_routing(
    observed, requested, recovery, effective,
):
    requests = []
    conditions = [{"kind": "window_contains", "value": "Main"}]
    arguments = ({"summary": "Entire goal observed", "done": True, "completion_conditions": conditions}
                 if effective == 1 else {"label": "Entire goal observed", "conditions": conditions})
    name = "declare_current_completion" if effective == 1 else "declare_completion"

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=packet(arguments, name=name))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v56")
        plan = planner.plan(GOAL, observed, [{"recovery_required": recovery}], max_actions=requested,
                            whole_goal_inband_epoch=3)
    assert len(requests) == planner.calls == 1 and plan.done and plan.completion_conditions
    assert planner.last_request_provenance["max_actions"] == effective
    assert planner.last_request_provenance["whole_goal_inband_review"]["input_epoch"] == 3
    assert "whole_goal_recheck" not in planner.last_request_provenance
    assert len(requests[0]["tools"]) == (5 if effective == 1 else 8)
    assert "EVERY requirement in the original" in requests[0]["messages"][0]["content"]
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert state["goal"] == GOAL


@pytest.mark.parametrize("bad_epoch", [False, -1, "0"])
def test_invalid_epoch_never_sends_http(observed, bad_epoch):
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request))) as client:
        planner = model(client, desktop_output_format="semantic_tools_v56")
        with pytest.raises(ValueError, match="normal desktop input epoch"):
            planner.plan(GOAL, observed, [], whole_goal_inband_epoch=bad_epoch)
    assert not calls
