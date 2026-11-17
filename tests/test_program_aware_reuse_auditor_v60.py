"Regression tests for test program aware reuse auditor v60."

import copy
import hashlib
import json

import pytest

from scripts.analyze_stage_program_reuse_v60 import (
    SELECTION_KIND,
    analyze,
    audit_trace,
    canonical,
    canonical_sha,
    compiled_matches,
    legacy_audit_trace,
    load_ref,
    normalized_conditions,
    normalized_target,
)


def target(name="Requested field"):
    return {"name": name, "role": "textbox", "app": "GenericEditor", "window": "Main"}


def condition(name="Requested field"):
    return {"kind": "value_equals", "target": normalized_target(target(name)), "value": "requested"}


def facts(conditions):
    return [
        {"kind": item["kind"], "target": item.get("target"), "outcome": "true", "satisfied": True}
        for item in conditions
    ]


class ProgramEvents:
    """Finite exact event contract fixture, not a GUI/backend simulator."""

    def __init__(self, *, numeric=False):
        self.events, self.steps, self.logical, self.inputs, self.reads = [], 0, 0, 0, 2
        self.stage, self.attempts, self.call = 0, [], 1
        self.numeric = numeric
        first = {
            "id": "first",
            "tool": "declare_field_edits_stage",
            "arguments": {
                "label": "Apply first requested field",
                "completion": "stage",
                "edits": [{"operation": "fill", "target": target(), "value": "requested"}],
            },
        }
        if numeric:
            first = {
                "id": "first",
                "tool": "declare_property_stage",
                "arguments": {
                    "label": "Apply requested numeric property",
                    "scope": {"app": "GenericEditor"},
                    "completion": "stage",
                    "edit_operation": "fill",
                    "edit_value": "100",
                    "numeric_finalize_key": "enter",
                    "confirmation_label": "none",
                },
            }
        later = {
            "id": "later",
            "tool": "declare_field_edits_stage",
            "arguments": {
                "label": "Apply future field",
                "completion": "stage",
                "edits": [{"operation": "fill", "target": target("Future field"), "value": "requested"}],
            },
        }
        self.source = {"label": "Perform generic declared work", "stages": [first, later]}
        self.sha = canonical_sha(self.source)
        self.program_id = "1:" + self.sha
        self.add(
            "agent_model_output_transport",
            {
                "call": 1,
                "response_transport": "semantic_tools_v60",
                "max_actions": 6,
                "http_wall_ms": 1234.5,
                "declaration": {
                    "tool_name": "declare_stage_program",
                    "program_declaration": True,
                    "program_declaration_sha256": self.sha,
                    "mapped_object_sha256": canonical_sha({"intent": "stage_program", **self.source}),
                    # Intentionally original whitespace bytes, unlike canonical source SHA.
                    "argument_sha256": hashlib.sha256(json.dumps(self.source, indent=2).encode()).hexdigest(),
                },
            },
        )
        self.add(
            "agent_planning_observation",
            {
                "call": 1,
                "effective_max_actions": 6,
                "requested_max_actions": 6,
                "a11y_complete": True,
                "image_uploaded": True,
                "native_execution_fingerprint": "a" * 64,
            },
        )
        declaration = {
            "label": self.source["label"],
            "max_actions": 6,
            "declaration_sha256": self.sha,
            "source_declaration": copy.deepcopy(self.source),
            "source_declaration_status": "original_canonical_json",
            "stages": [
                {
                    **copy.deepcopy(member),
                    "entry_conditions": [],
                    "recovery": "observe",
                    "declared_slots": 1,
                    "compiled": False,
                }
                for member in self.source["stages"]
            ],
        }
        self.add("agent_program_declared", {**self.common(), "declaration": declaration})
        self.add(
            "agent_program_stage_activated",
            {
                **self.common(),
                "source": "actual_planner_request",
                "snapshot_fingerprint": "a" * 64,
                "compiled_steps": self.compiled(),
                "real_planner_invocation": True,
                "freshness_claim": False,
                "admitted": False,
            },
        )
        self.add(
            "agent_program_stage_admitted",
            {
                **self.common(),
                **self.frame(),
                "admitted": True,
                "entry_conditions": [],
                "one_shot_entry_admission": True,
                "initial_plan_not_recompiled": True,
                "planned_request_binding_unchanged": True,
                "original_action_preconditions_and_preflight_still_required": True,
            },
        )

    def add(self, kind, payload):
        event = {"sequence": len(self.events), "run_id": "public-synthetic", "kind": kind, "payload": payload}
        self.events.append(event)
        return payload

    def common(self):
        return {
            "program_id": self.program_id,
            "origin_planner_call": self.call,
            "stage_index": self.stage,
            "stage_id": self.source["stages"][self.stage]["id"],
            "original_plan_at": 100.25,
            "plan_actions": self.logical,
            "remaining_actions": max(0, 6 - self.logical),
            "logical_attempt_budget": 6,
            "input_authority": False,
            "task_success": False,
        }

    def frame(self, source="backend.observe"):
        return {
            "source": source,
            "snapshot_fingerprint": str(self.stage + 1) * 64,
            "image_sha256": "f" * 64,
            "environment_steps": self.inputs,
            "fresh_observe_calls": self.reads,
        }

    def compiled(self):
        if self.numeric and self.stage == 0:
            return [
                {
                    "id": "property",
                    "operation": "click",
                    "navigation": True,
                    "depends_on": [],
                    "postconditions": [],
                    "target": normalized_target({"app": "GenericEditor"}),
                    "property_edit": {"operation": "fill", "value": "100", "numeric_finalize_key": "enter"},
                }
            ]
        name = "Requested field" if self.stage == 0 else "Future field"
        ready = {"kind": "ready_to_fill", "target": normalized_target(target(name)), "value": "requested"}
        return [
            {
                "id": "field",
                "operation": "fill",
                "value": "requested",
                "target": normalized_target(target(name)),
                "depends_on": [],
                "preconditions": [ready],
                "postconditions": [condition(name), ready],
                "recovery": "observe",
            }
        ]

    def runtime(
        self,
        n=1,
        *,
        verified=True,
        success=True,
        pending=False,
        commit=None,
        stale=False,
        polling=None,
        policy=None,
    ):
        self.steps += 1
        rid, cid, did = f"receipt-{self.steps}", f"candidate-{self.steps}", f"decision-{self.steps}"
        intent = "property" if self.numeric and self.stage == 0 else "field"
        candidate = {
            "id": cid,
            "channel": "gui",
            "capability": "agent.osworld.key_press" if commit else "agent.osworld.fill",
            "intent": intent,
            "arguments": {"operation": "key_press" if commit else "fill"},
        }
        self.add("candidates", {"items": [candidate]})
        self.add(
            "decision",
            {"candidate_id": cid, "decision_id": did, "observation_id": "observation-" + str(self.steps)},
        )
        self.add("guard", {"candidate_id": cid, "approved": True})
        output = {"environment_steps": n, "postcondition_evidence": facts([condition()])}
        if stale:
            output["failure_code"] = "stale_observation"
        self.add(
            "receipt",
            {
                "receipt_id": rid,
                "candidate_id": cid,
                "decision_id": did,
                "observation_id": "observation-" + str(self.steps),
                "channel": "gui",
                "capability": candidate["capability"],
                "success": success,
                "output": output,
            },
        )
        self.add("verification", {"passed": verified if polling is None else False, "details": {}})
        if policy is not None:
            self.add(
                "agent_program_recovery_policy",
                {
                    **self.common(),
                    "phase": "post_input_pending",
                    "stage_recovery": policy,
                    "original_observe_eligible": True,
                    "optional_observe_permitted": policy == "observe",
                    "selected_original_branch": "bounded_observe"
                    if policy == "observe"
                    else "original_nonobserve_path",
                    "event_observation_calls": 0,
                    "native_observation_calls_at_decision": self.reads,
                    "source_receipt_id": rid,
                    "original_effect_recovery": "observe",
                },
            )
        if polling is not None:
            for index in range(polling):
                self.reads += 1
                self.add(
                    "agent_pending_effect",
                    {
                        "step": intent,
                        "attempt": index + 1,
                        "passed": verified,
                        "deadline_exceeded": not verified,
                    },
                )
        if pending:
            self.add(
                "agent_declared_numeric_finalize",
                {"status": "pending", "source_receipt_id": rid, "key": "enter", "verified": False},
            )
        if commit:
            self.add(
                "agent_declared_numeric_finalize",
                {
                    "status": "committed",
                    "source_receipt_id": commit,
                    "receipt_id": rid,
                    "key": "enter",
                    "verified": verified,
                },
            )
        self.add(
            "agent_step",
            {"step": self.steps, "candidate": cid, "planner_calls": self.call, "verified": verified},
        )
        self.inputs += n
        self.logical += int(verified or pending)
        self.attempts.append(
            {
                "receipt_id": rid,
                "environment_steps": n,
                "final_verified": verified,
                "declared_numeric_contingency": pending,
                "bounded_stale_refresh": stale,
            }
        )
        return rid

    def complete(self, source="backend.observe"):
        self.reads += 1
        name = "Requested field" if self.stage == 0 else "Future field"
        conditions = (
            [condition(name)] if self.numeric and self.stage == 0 else self.compiled()[0]["postconditions"]
        )
        self.add(
            "agent_program_stage_completed",
            {
                **self.common(),
                **self.frame(source),
                "reason": "verified_receipt"
                if any(item["final_verified"] for item in self.attempts)
                else "read_only_predicates",
                "read_only": all(item["environment_steps"] == 0 for item in self.attempts),
                "receipt_ids": [item["receipt_id"] for item in self.attempts if item["final_verified"]],
                "attempt_receipts": copy.deepcopy(self.attempts),
                "terminal_conditions": facts(conditions),
                "positive_input": any(item["environment_steps"] > 0 for item in self.attempts),
                "pure_predicate_logical_attempts": 0,
                "runtime_receipts_keep_original_logical_attempt_charge": True,
                "positive_verified_input": any(
                    item["environment_steps"] > 0 and item["final_verified"] for item in self.attempts
                ),
            },
        )

    def advance(self):
        prior = self.events[-1]["payload"]
        self.stage += 1
        self.attempts = []
        full = {
            key: prior[key]
            for key in (
                "source",
                "snapshot_fingerprint",
                "image_sha256",
                "environment_steps",
                "fresh_observe_calls",
            )
        }
        self.add(
            "agent_program_stage_activated",
            {
                **self.common(),
                **full,
                "compiled_steps": self.compiled(),
                "http_calls": 0,
                "admitted": True,
                "one_shot_entry_admission": True,
                "entry_conditions": [],
            },
        )

    def finish(self):
        self.add(
            "agent_program_completed",
            {**self.common(), "all_declared_stages_verified": True, "whole_goal_review_still_required": True},
        )
        self.add(
            "agent_episode",
            {"steps": self.steps, "status": "timeout", "duration_ms": 420123.4, "plan_reuses": 90},
        )
        return self.events


def valid_program(*, first_n=3, second_n=1, numeric=False):
    fixture = ProgramEvents(numeric=numeric)
    if numeric:
        source = fixture.runtime(3, verified=False, pending=True)
        fixture.runtime(1, commit=source)
    elif first_n is not None:
        fixture.runtime(first_n)
    fixture.complete()
    fixture.advance()
    if second_n is not None:
        fixture.runtime(second_n)
    fixture.complete()
    return fixture.finish()


def payload(events, kind, occurrence=0):
    return [event["payload"] for event in events if event["kind"] == kind][occurrence]


def resequence(events):
    for index, event in enumerate(events):
        event["sequence"] = index
    return events


def test_two_real_stage_inputs_share_one_origin_while_legacy_original_stays_unknown():
    events = valid_program()
    report = audit_trace(events)
    assert report["program_metric_status"] == "complete"
    assert report["positive_input_program_continuations"] == 1
    assert report["cross_stage_positive_input_program_continuations"] == 1
    assert report["legacy_fixed_original"] == legacy_audit_trace(events)
    assert report["legacy_fixed_original"]["positive_input_plan_continuations"] is None
    assert report["program_epochs"][0]["logical_actions"] == 2
    assert report["program_epochs"][0]["raw_environment_inputs_complete"] == 4
    assert report["agent_episode_original"]["duration_ms"] == 420123.4
    assert report["acceptance_satisfied"] is False


@pytest.mark.parametrize("first_n,second_n", [(None, 1), (0, 1), (None, None), (0, 0)])
def test_predicate_and_runtime_readonly_are_not_positive_anchors(first_n, second_n):
    report = audit_trace(valid_program(first_n=first_n, second_n=second_n))
    assert report["program_metric_status"] == "complete"
    assert report["positive_input_program_continuations"] == 0
    assert report["program_epochs"][0]["logical_actions"] == int(first_n is not None) + int(
        second_n is not None
    )


def test_pending_false_numeric_fill_is_charged_only_verified_key_anchors():
    report = audit_trace(valid_program(numeric=True))
    epoch = report["program_epochs"][0]
    assert report["program_metric_status"] == "complete"
    assert report["positive_input_program_continuations"] == 1
    assert epoch["logical_actions"] == 3 and epoch["raw_environment_inputs_complete"] == 5
    assert [row["receipt_id"] for row in epoch["qualified"]] == ["receipt-2", "receipt-3"]


@pytest.mark.parametrize(
    "change",
    [
        "no_admission",
        "duplicate_admission",
        "changed_clock",
        "reset_attempts",
        "wrong_remaining",
        "future_http",
        "unknown_exit",
        "wrong_exit_scope",
        "changed_fp",
        "changed_image",
        "old_frame_source",
        "future_compiled",
        "source_mutation",
        "mapped_sha",
        "fake_old_plan",
        "duplicate_receipt",
    ],
)
def test_unproven_program_continuity_remains_unknown(change):
    events = valid_program()
    if change == "no_admission":
        events = [event for event in events if event["kind"] != "agent_program_stage_admitted"]
    elif change == "duplicate_admission":
        row = copy.deepcopy(
            next(event for event in events if event["kind"] == "agent_program_stage_admitted")
        )
        events.insert(row["sequence"] + 1, row)
    elif change == "changed_clock":
        payload(events, "agent_program_stage_activated", 1)["original_plan_at"] += 1
    elif change == "reset_attempts":
        payload(events, "agent_program_stage_activated", 1)["plan_actions"] = 0
    elif change == "wrong_remaining":
        payload(events, "agent_program_stage_activated", 1)["remaining_actions"] = 6
    elif change == "future_http":
        payload(events, "agent_program_stage_activated", 1)["http_calls"] = 1
    elif change in {"unknown_exit", "wrong_exit_scope"}:
        fact = payload(events, "agent_program_stage_completed")["terminal_conditions"][0]
        if change == "unknown_exit":
            fact.update(outcome="unknown", satisfied=False)
        else:
            fact["target"]["app"] = "DifferentOwner"
    elif change in {"changed_fp", "changed_image"}:
        key = "snapshot_fingerprint" if change == "changed_fp" else "image_sha256"
        payload(events, "agent_program_stage_activated", 1)[key] = "b" * 64
    elif change == "old_frame_source":
        payload(events, "agent_program_stage_completed")["source"] = "actual_planner_request"
    elif change == "future_compiled":
        payload(events, "agent_program_stage_activated")["compiled_steps"].append(
            {"id": "future", "operation": "fill"}
        )
    elif change == "source_mutation":
        payload(events, "agent_program_declared")["declaration"]["source_declaration"]["label"] = "changed"
    elif change == "mapped_sha":
        payload(events, "agent_model_output_transport")["declaration"]["mapped_object_sha256"] = "not-a-sha"
    elif change == "fake_old_plan":
        events.insert(
            3, {"kind": "agent_plan", "run_id": "public-synthetic", "payload": {"call": 1, "reusable": True}}
        )
    else:
        payload(events, "receipt", 1)["receipt_id"] = "receipt-1"
    report = audit_trace(resequence(events))
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None


@pytest.mark.parametrize("value", [None, True, "3", float("nan"), float("inf")])
def test_missing_or_nonnumeric_runtime_count_cannot_be_zero_or_positive(value):
    events = valid_program()
    payload(events, "receipt")["output"]["environment_steps"] = value
    report = audit_trace(events)
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None


@pytest.mark.parametrize("change", ["missing_commit", "different_source", "false_key", "field_retry"])
def test_numeric_contingency_requires_exact_new_final_verified_key(change):
    events = valid_program(numeric=True)
    if change == "missing_commit":
        events = [
            event
            for event in events
            if not (
                event["kind"] == "agent_declared_numeric_finalize"
                and event["payload"].get("status") == "committed"
            )
        ]
    elif change == "different_source":
        payload(events, "agent_declared_numeric_finalize", 1)["source_receipt_id"] = "other"
    elif change == "false_key":
        payload(events, "agent_step", 1)["verified"] = False
    else:
        for kind in ("candidates", "receipt"):
            row = payload(events, kind, 1)
            (row["items"][0] if kind == "candidates" else row)["capability"] = "agent.osworld.fill"
    report = audit_trace(resequence(events))
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None


def stale_readonly_program():
    fixture = ProgramEvents()
    rid = fixture.runtime(0, verified=False, success=False, stale=True)
    fixture.add("agent_stale_recovery", {"stage": "requested", "receipt_id": rid, "per_plan_limit": 1})
    fixture.add(
        "agent_observation_reuse",
        {"receipt_id": rid, "source": "stale_rejection_preflight", "response_age_ms": 25.0},
    )
    fixture.complete("stale_rejection_preflight")
    fixture.advance()
    fixture.runtime(1)
    fixture.complete()
    return fixture.finish()


def test_real_bounded_stale_preflight_may_complete_readonly_without_positive_credit():
    report = audit_trace(stale_readonly_program())
    assert report["program_metric_status"] == "complete"
    assert report["positive_input_program_continuations"] == 0
    assert report["program_epochs"][0]["logical_actions"] == 1


@pytest.mark.parametrize("change", ["missing_reuse", "wrong_receipt", "old_reuse", "fake_stale_source"])
def test_stale_source_label_without_actual_refusal_reuse_chain_is_unknown(change):
    events = stale_readonly_program() if change != "fake_stale_source" else valid_program()
    if change == "missing_reuse":
        events = [event for event in events if event["kind"] != "agent_observation_reuse"]
    elif change == "wrong_receipt":
        payload(events, "agent_observation_reuse")["receipt_id"] = "other"
    elif change == "old_reuse":
        payload(events, "agent_observation_reuse")["response_age_ms"] = 1001
    else:
        payload(events, "agent_program_stage_completed")["source"] = "stale_rejection_preflight"
    assert audit_trace(resequence(events))["program_metric_status"] == "unknown"


@pytest.mark.parametrize(
    "policy,polls,expected", [("observe", 2, "complete"), ("observe", 3, "unknown"), ("replan", 1, "unknown")]
)
def test_recovery_route_is_bound_to_actual_pending_observations(policy, polls, expected):
    fixture = ProgramEvents()
    fixture.runtime(1, polling=polls, policy=policy)
    fixture.complete()
    fixture.advance()
    fixture.runtime(1)
    fixture.complete()
    assert audit_trace(fixture.finish())["program_metric_status"] == expected


def rejected_program_then_original_h1(*, orphan=False):
    events = [{"run_id": "public-synthetic", "kind": "agent_recovery", "payload": {"reason": "invalid_plan"}}]
    events.extend(ProgramEvents().events[:2])  # Actual except -> finally transport/planning order.
    fixture = ProgramEvents()
    fixture.events, fixture.call = events, 2
    fixture.add("agent_planning_observation", {"call": 2})
    if not orphan:
        fixture.add("agent_plan", {"call": 2, "reusable": False})
    fixture.runtime(1)
    fixture.add("agent_episode", {"steps": 1, "status": "stuck", "duration_ms": 9999.0})
    return resequence(fixture.events)


def test_rejected_program_cost_kept_without_invented_accepted_source_then_legal_h1():
    events = rejected_program_then_original_h1()
    report = audit_trace(events)
    assert report["program_metric_status"] == "complete"
    assert report["positive_input_program_continuations"] == 0
    assert report["program_epochs"] == []
    assert report["program_requests"][0]["rejected_without_input"] is True
    assert report["program_requests"][0]["transport"]["http_wall_ms"] == 1234.5
    assert report["legacy_fixed_original"] == legacy_audit_trace(events)


def test_receipt_without_accepted_program_or_bound_legacy_epoch_is_unknown():
    assert audit_trace(rejected_program_then_original_h1(orphan=True))["program_metric_status"] == "unknown"


def test_orphan_program_event_is_missing_epoch_unknown_not_not_applicable():
    events = [
        {"sequence": 0, "run_id": "orphan", "kind": "agent_program_completed", "payload": {}},
        {
            "sequence": 1,
            "run_id": "orphan",
            "kind": "agent_episode",
            "payload": {"steps": 0, "status": "stuck"},
        },
    ]
    report = audit_trace(events)
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None
    assert len(report["orphan_program_events"]) == 1


def test_multiple_actual_rejections_match_their_own_following_finally_transport():
    rows = rejected_program_then_original_h1()[:3]
    second = copy.deepcopy(rows)
    second[1]["payload"]["call"] = second[2]["payload"]["call"] = 2
    rows.extend(second)
    rows.append(
        {
            "run_id": "public-synthetic",
            "kind": "agent_episode",
            "payload": {"steps": 0, "status": "planner_error"},
        }
    )
    report = audit_trace(resequence(rows))
    assert report["program_metric_status"] == "complete"
    assert [row["actual_rejection_sequence"] for row in report["program_requests"]] == [0, 3]
    assert report["positive_input_program_continuations"] == 0


def test_ordered_transport_hash_is_preserved_without_claiming_canonical_byte_equivalence():
    events = valid_program()
    transport = payload(events, "agent_model_output_transport")["declaration"]
    transport["mapped_object_sha256"] = "0" * 64  # Opaque original ordered SHA, independently unavailable.
    report = audit_trace(events)
    assert report["program_metric_status"] == "complete"
    epoch = report["program_epochs"][0]
    assert epoch["observed_ordered_mapped_object_sha256_not_independently_reconstructed"] == "0" * 64
    assert epoch["independently_recomputed_canonical_mapped_object_sha256"] != "0" * 64


def test_unaccepted_unrejected_later_request_is_unknown_even_after_valid_epoch():
    events = valid_program()
    transport = copy.deepcopy(events[0])
    transport["payload"]["call"] = 2
    events.insert(-1, transport)
    report = audit_trace(resequence(events))
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None


def test_seventh_actual_logical_attempt_is_unknown_without_budget_reset():
    fixture = ProgramEvents()
    for _index in range(6):
        fixture.runtime(1)
    fixture.complete()
    fixture.advance()
    fixture.runtime(1)
    fixture.complete()
    report = audit_trace(fixture.finish())
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None


@pytest.mark.parametrize(
    "operation,value,compiled_value", [("press", "Enter", "Enter"), ("key_press", "ENTER", "enter")]
)
def test_original_commit_operations_keys_and_target_whitespace_are_not_reinterpreted(
    operation, value, compiled_value
):
    edit = {"operation": "fill", "target": target("  Requested field  "), "value": " requested "}
    effect = {"kind": "value_equals", "target": edit["target"], "value": edit["value"]}
    commit = {
        "operation": operation,
        "value": value,
        "preconditions": [{"kind": "enabled", "target": target()}],
        "postconditions": [
            {"kind": "absent", "target": {"name": "Dialog", "role": "dialog", "app": "GenericEditor"}}
        ],
    }
    member = {"tool": "declare_field_edits_stage", "arguments": {"edits": [edit], "commit": commit}}
    steps = [
        {
            "id": "edit1",
            **edit,
            "target": normalized_target(edit["target"]),
            "depends_on": [],
            "preconditions": normalized_conditions(
                [{"kind": "ready_to_fill", "target": edit["target"], "value": edit["value"]}]
            ),
            "postconditions": normalized_conditions(
                [effect, {"kind": "ready_to_fill", "target": edit["target"], "value": edit["value"]}]
            ),
        },
        {
            "id": "commit",
            "operation": operation,
            "value": compiled_value,
            "depends_on": ["edit1"],
            "preconditions": normalized_conditions([effect, *commit["preconditions"]]),
            "postconditions": normalized_conditions(commit["postconditions"]),
        },
    ]
    assert compiled_matches(member, steps) == {"edit1", "commit"}
    bad = copy.deepcopy(steps)
    bad[-1]["value"] = "different"
    with pytest.raises(ValueError):
        compiled_matches(member, bad)
    bad = copy.deepcopy(steps)
    bad[-1]["preconditions"] = normalized_conditions([effect])
    with pytest.raises(ValueError):
        compiled_matches(member, bad)
    assert steps[0]["value"] == " requested "  # Literal field whitespace is not stripped.


def test_commit_only_native_caption_readiness_and_derived_dialog_guard_keep_original_path():
    button = {"name": "OK", "role": "button", "app": "GenericEditor", "window": "Preferences"}
    dialog = {"name": "Preferences", "role": "dialog", "app": "GenericEditor", "window": "Preferences"}
    before = {
        "kind": "value_equals",
        "target": {**target("Value"), "window": "Preferences"},
        "value": "requested",
    }
    actual_guard = {**before, "target": normalized_target({**target("Value:"), "window": "Preferences"})}
    ready = {"kind": "ready_to_fill", "target": actual_guard["target"], "value": "requested"}
    exists = {"kind": "exists", "target": normalized_target(dialog), "value": ""}
    member = {
        "tool": "declare_field_edits_stage",
        "arguments": {
            "edits": [],
            "commit": {"operation": "click", "target": button, "preconditions": [before]},
        },
    }
    actual = {
        "id": "commit",
        "operation": "click",
        "target": normalized_target(button),
        "depends_on": [],
        "preconditions": [actual_guard, ready, exists],
        "derived_dialog_commit_effect": True,
        "postconditions": normalized_conditions([{"kind": "absent", "target": dialog}]),
    }
    assert compiled_matches(member, [actual]) == {"commit"}
    bad = copy.deepcopy(actual)
    bad["preconditions"][0]["target"]["app"] = "DifferentOwner"
    with pytest.raises(ValueError):
        compiled_matches(member, [bad])


def test_public_legacy_only_result_is_exact_original_and_program_not_applicable():
    fixture = ProgramEvents()
    fixture.events = []
    fixture.add("agent_plan", {"call": 1, "reusable": True})
    fixture.runtime(3)
    fixture.runtime(1)
    fixture.add("agent_episode", {"steps": 2, "status": "timeout", "duration_ms": 420123.4})
    original = legacy_audit_trace(fixture.events)
    report = audit_trace(fixture.events)
    assert original["positive_input_plan_continuations"] == 1
    assert report["legacy_fixed_original"] == original
    assert report["positive_input_plan_continuations"] == 1
    assert report["program_metric_status"] == "not_applicable"


def write_ref(root, name, value, *, lines=False):
    path = root / "runs" / name
    raw = ("\n".join(json.dumps(row) for row in value) if lines else json.dumps(value)).encode()
    path.write_bytes(raw)
    return {"path": "runs/" + name, "sha256": hashlib.sha256(raw).hexdigest()}


def test_terminal_consumer_retains_failure_timeout_and_missing_trace_unknown(tmp_path):
    (tmp_path / "runs").mkdir()
    source = write_ref(tmp_path, "source.json", {"files": {"src/public.py": "x"}})
    terminal = write_ref(
        tmp_path,
        "terminal.json",
        {"terminal": True, "active_child_may_be_running": False, "source_manifest_sha256": source["sha256"]},
    )
    summary = write_ref(
        tmp_path,
        "summary.json",
        {
            "runs": [
                {
                    "task": "a",
                    "mode": "hybrid",
                    "official_score": 0,
                    "evaluation_status": "ok",
                    "agent_ms": 420123.4,
                },
                {
                    "task": "b",
                    "mode": "hybrid",
                    "official_score": None,
                    "evaluation_status": "not_run",
                    "agent_ms": None,
                },
            ]
        },
    )
    official = write_ref(
        tmp_path, "official.json", {"closed": True, "official_score": 0, "evaluation_status": "ok"}
    )
    trace = write_ref(tmp_path, "trace.jsonl", valid_program(), lines=True)
    manifest = {
        "kind": SELECTION_KIND,
        "terminal": True,
        "terminal_evidence": terminal,
        "episodes": [
            {
                "id": "a-H",
                "task": "a",
                "mode": "hybrid",
                "summary": summary,
                "source_manifest": source,
                "official": official,
                "trace": trace,
            },
            {
                "id": "b-H",
                "task": "b",
                "mode": "hybrid",
                "summary": summary,
                "source_manifest": source,
                "official": None,
                "trace": None,
            },
        ],
    }
    report = analyze(tmp_path, manifest)
    assert len(report["episodes"]) == 2
    assert report["episodes"][0]["original_summary_row"]["agent_ms"] == 420123.4
    assert report["episodes"][0]["original_summary_row"]["official_score"] == 0
    assert report["episodes"][1]["metric_status"] == "unknown"
    assert report["episodes"][1]["positive_input_program_continuations"] is None
    assert report["episodes"][1]["original_summary_row"]["official_score"] is None
    assert report["acceptance_satisfied"] is False
    changed = {**trace, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        load_ref(tmp_path, changed, lines=True)
    assert canonical({"a": 1}) == '{"a":1}'
