"Offline audit of original plan and stage-program execution evidence."

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

FIXED_ANALYZER_SHA256 = "7bd28f3ad68be2a1354f671ce366d2cb96a62bdfcfe10810d98f634ba0067abe"
PROGRAM_INTENT = "stage_program"
PROGRAM_TOOL = "declare_stage_program"
PROGRAM_FORMAT = "semantic_tools_v60"
FULL_SOURCES = {
    "backend.observe",
    "verified_step_response",
    "verified_pending_effect_observation",
    "verified_ordinary_pending_effect_observation",
    "stale_rejection_preflight",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def legacy_audit_trace(events):
    """Join exact runtime blocks, then use the runner's final step verdict."""
    require(isinstance(events, list) and len(events) <= 20_000, "too many trace events")
    epochs, blocks, issues = [], [], []
    active = candidates = decision = guard = pending = None
    seen_receipts, seen_decisions = set(), set()
    last_step, run_id = 0, None
    episode = None

    def issue(reason):
        issues.append(reason)
        if active is not None:
            active["uncertain"] = True
            active["anchor"] = False

    for index, event in enumerate(events):
        require(
            isinstance(event, dict) and type(event.get("sequence")) is int and event["sequence"] == index,
            "noncontiguous trace sequence",
        )
        require(isinstance(event.get("run_id"), str) and event["run_id"], "missing run identity")
        run_id = run_id or event["run_id"]
        require(event["run_id"] == run_id, "mixed run identities")
        kind, p = event.get("kind"), event.get("payload")
        require(isinstance(p, dict), "invalid trace payload")
        if kind == "agent_plan":
            if pending is not None:
                issue("new plan before prior receipt acquired a final agent_step")
                pending = None
            if type(p.get("call")) is not int or p["call"] <= 0 or type(p.get("reusable")) is not bool:
                issue("invalid plan epoch identity")
                active = None
                continue
            active = {
                "plan_sequence": index,
                "call": p["call"],
                "reusable": p["reusable"],
                "qualified_steps": [],
                "positive_continuations_observed": 0,
                "uncertain": False,
                "anchor": False,
            }
            epochs.append(active)
            candidates = decision = guard = None
        elif kind == "agent_planning_observation" and active is not None and p.get("call") != active["call"]:
            active = None  # A new actual model request cannot be hidden inside one epoch.
        elif kind == "agent_recovery" and p.get("reason") == "invalid_plan":
            active = None
        elif kind == "candidates":
            items = p.get("items", [])
            candidates = (
                {c.get("id"): c for c in items if isinstance(c, dict)} if isinstance(items, list) else {}
            )
            if len(candidates) != len(items) or None in candidates:
                issue("ambiguous candidate list")
            decision = guard = None
        elif kind == "decision":
            decision, guard = p, None
            if p.get("decision_id") in seen_decisions:
                issue("duplicate decision identity")
            seen_decisions.add(p.get("decision_id"))
        elif kind == "guard":
            guard = p
        elif kind == "receipt":
            if pending is not None:
                issue("another receipt before final agent_step")
            candidate = (candidates or {}).get(p.get("candidate_id"))
            valid = bool(
                candidate is not None
                and decision is not None
                and guard is not None
                and isinstance(decision.get("decision_id"), str)
                and decision["decision_id"]
                and isinstance(decision.get("observation_id"), str)
                and decision["observation_id"]
                and p.get("decision_id") == decision.get("decision_id")
                and p.get("observation_id") == decision.get("observation_id")
                and p.get("candidate_id") == decision.get("candidate_id")
                and p.get("candidate_id") == guard.get("candidate_id")
                and guard.get("approved") is True
                and p.get("capability") == candidate.get("capability")
                and p.get("channel") == candidate.get("channel")
                and isinstance(p.get("receipt_id"), str)
                and p["receipt_id"]
                and p["receipt_id"] not in seen_receipts
            )
            seen_receipts.add(p.get("receipt_id"))
            pending = {
                "receipt_sequence": index,
                "receipt": p,
                "candidate": candidate,
                "binding_valid": valid,
                "verification": None,
                "pending_effects": [],
                "epoch": active,
            }
        elif kind == "verification" and pending is not None:
            if pending["verification"] is not None:
                pending["binding_valid"] = False
            pending["verification"] = p
        elif kind == "agent_pending_effect" and pending is not None:
            pending["pending_effects"].append(p)
        elif kind == "agent_step":
            if type(p.get("step")) is not int or p["step"] != last_step + 1:
                issue("noncontiguous agent_step count")
            last_step = p.get("step") if type(p.get("step")) is int else last_step
            if pending is None:
                issue("agent_step lacks receipt")
                continue
            receipt, candidate, epoch = pending["receipt"], pending["candidate"], pending["epoch"]
            verification = pending["verification"]
            n = receipt.get("output", {}).get("environment_steps")
            joined = bool(
                pending["binding_valid"]
                and epoch is not None
                and epoch is active
                and candidate is not None
                and candidate.get("intent")
                and p.get("candidate") == receipt.get("candidate_id")
                and p.get("planner_calls") == epoch["call"]
                and type(n) is int
                and n >= 0
                and type(receipt.get("success")) is bool
                and type(p.get("verified")) is bool
                and isinstance(verification, dict)
                and type(verification.get("passed")) is bool
            )
            pending_passed = False
            for effect in pending["pending_effects"]:
                if effect.get("step") != candidate.get("intent") if candidate else True:
                    joined = False
                if "receipt_id" in effect and effect["receipt_id"] != receipt.get("receipt_id"):
                    joined = False
                if effect.get("passed") is True and effect.get("deadline_exceeded") is False:
                    pending_passed = True
            supported = verification is not None and (verification.get("passed") is True or pending_passed)
            classification = "unknown"
            if joined:
                if p["verified"] and (not supported or receipt["success"] is not True):
                    joined = False
                elif n == 0:
                    classification = "zero_input_verified" if p["verified"] else "zero_input_unverified"
                elif receipt["success"] and p["verified"] and supported:
                    classification = (
                        "positive_verified_after_pending" if pending_passed else "positive_verified"
                    )
                else:
                    classification = "positive_unverified_or_failed"
            block = {
                "agent_step_sequence": index,
                "agent_step": p.get("step"),
                "receipt_sequence": pending["receipt_sequence"],
                "receipt_id": receipt.get("receipt_id"),
                "candidate_id": receipt.get("candidate_id"),
                "decision_id": receipt.get("decision_id"),
                "plan_sequence": epoch["plan_sequence"] if epoch else None,
                "plan_call": epoch["call"] if epoch else None,
                "environment_steps": n,
                "receipt_success": receipt.get("success"),
                "original_verification_passed": verification.get("passed") if verification else None,
                "pending_effect_passed": pending_passed,
                "final_agent_step_verified": p.get("verified"),
                "classification": classification,
            }
            blocks.append(block)
            if not joined:
                issue("incomplete or contradictory runtime receipt/verification/agent_step binding")
            elif classification.startswith("positive_verified") and epoch["reusable"]:
                if epoch["anchor"]:
                    epoch["positive_continuations_observed"] += 1
                epoch["anchor"] = True
                epoch["qualified_steps"].append(p["step"])
            pending = None
        elif kind == "agent_episode":
            if episode is not None:
                issue("duplicate terminal agent_episode")
            episode = p
    if pending is not None:
        issue("orphan receipt at end of trace")
    if (
        episode is None
        or type(episode.get("steps")) is not int
        or episode["steps"] != last_step
        or not isinstance(episode.get("status"), str)
    ):
        issue("missing or inconsistent terminal agent_episode")
    observed = sum(e["positive_continuations_observed"] for e in epochs)
    return {
        "run_id": run_id,
        "metric_status": "complete" if not issues else "partial_or_unknown",
        "positive_input_plan_continuations": observed if not issues else None,
        "observed_positive_continuations_lower_bound": observed,
        "original_plan_reuses": episode.get("plan_reuses") if episode else None,
        "runner_status_original": episode.get("status") if episode else None,
        "agent_duration_ms_original": episode.get("duration_ms") if episode else None,
        "agent_steps": blocks,
        "epochs": [{k: v for k, v in e.items() if k != "anchor"} for e in epochs],
        "issues": issues,
    }


# PROGRAM-AWARE LEDGER BELOW: original events only, never injected agent_plan.


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def hash64(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def normalized_target(value):
    if value is None:
        return None
    require(
        type(value) is dict and set(value) <= {"name", "role", "app", "window", "ancestor", "match"},
        "invalid declared semantic target",
    )
    result = {
        key: value.get(key, "exact" if key == "match" else "")
        for key in ("name", "role", "app", "window", "ancestor", "match")
    }
    require(all(type(item) is str for item in result.values()), "target field type")
    result = {key: item if key == "match" else item.strip() for key, item in result.items()}
    return result


def normalized_conditions(values):
    require(type(values) is list and len(values) <= 8, "condition list bound")
    result = []
    for value in values:
        require(
            type(value) is dict
            and set(value) <= {"kind", "target", "value"}
            and type(value.get("kind")) is str
            and value["kind"],
            "condition shape",
        )
        require(type(value.get("value", "")) is str, "condition value type")
        result.append(
            {
                "kind": value["kind"],
                "target": normalized_target(value.get("target")),
                "value": value.get("value", ""),
            }
        )
    return result


def facts_true(facts, expected=None, *, allow_empty=False):
    if type(facts) is not list or not (facts or allow_empty):
        return False
    if any(
        type(fact) is not dict or fact.get("outcome") != "true" or fact.get("satisfied") is not True
        for fact in facts
    ):
        return False
    if expected is not None:
        if len(facts) != len(expected):
            return False
        return all(
            fact.get("kind") == condition["kind"]
            and normalized_target(fact.get("target")) == condition["target"]
            for fact, condition in zip(facts, expected, strict=True)
        )
    return True


def runtime_join(events):
    """Original candidate/decision/guard/receipt/final-verdict join, no plan fiction."""
    candidates = decision = guard = pending = None
    seen_receipts, seen_decisions = set(), set()
    blocks, issues = [], []
    last_step = 0
    for index, event in enumerate(events):
        kind, payload = event["kind"], event["payload"]
        if kind == "candidates":
            items = payload.get("items", [])
            candidates = (
                {item.get("id"): item for item in items if type(item) is dict} if type(items) is list else {}
            )
            if type(items) is not list or len(candidates) != len(items) or None in candidates:
                issues.append({"sequence": index, "reason": "ambiguous candidate list"})
            decision = guard = None
        elif kind == "decision":
            decision, guard = payload, None
            if payload.get("decision_id") in seen_decisions:
                issues.append({"sequence": index, "reason": "duplicate decision identity"})
            seen_decisions.add(payload.get("decision_id"))
        elif kind == "guard":
            guard = payload
        elif kind == "receipt":
            if pending is not None:
                issues.append({"sequence": index, "reason": "receipt before prior final agent_step"})
            candidate = (candidates or {}).get(payload.get("candidate_id"))
            valid = bool(
                candidate is not None
                and decision is not None
                and guard is not None
                and type(decision.get("decision_id")) is str
                and decision["decision_id"]
                and type(decision.get("observation_id")) is str
                and decision["observation_id"]
                and payload.get("decision_id") == decision.get("decision_id")
                and payload.get("observation_id") == decision.get("observation_id")
                and payload.get("candidate_id") == decision.get("candidate_id") == guard.get("candidate_id")
                and guard.get("approved") is True
                and payload.get("capability") == candidate.get("capability")
                and payload.get("channel") == candidate.get("channel")
                and type(payload.get("receipt_id")) is str
                and payload["receipt_id"]
                and payload["receipt_id"] not in seen_receipts
            )
            seen_receipts.add(payload.get("receipt_id"))
            pending = {
                "receipt_sequence": index,
                "receipt": payload,
                "candidate": candidate,
                "joined": valid,
                "verification": None,
                "pending_effects": [],
            }
        elif kind == "verification" and pending is not None:
            if pending["verification"] is not None:
                pending["joined"] = False
            pending["verification"] = payload
        elif kind == "agent_pending_effect" and pending is not None:
            pending["pending_effects"].append({"sequence": index, **payload})
        elif kind == "agent_step":
            if not integer(payload.get("step"), 1) or payload["step"] != last_step + 1:
                issues.append({"sequence": index, "reason": "noncontiguous final agent_step"})
            last_step = payload.get("step") if integer(payload.get("step"), 1) else last_step
            if pending is None:
                issues.append({"sequence": index, "reason": "final agent_step without receipt"})
                continue
            receipt, candidate, verification = (
                pending["receipt"],
                pending["candidate"],
                pending["verification"],
            )
            inputs = receipt.get("output", {}).get("environment_steps")
            joined = bool(
                pending["joined"]
                and candidate is not None
                and type(candidate.get("intent")) is str
                and candidate["intent"]
                and payload.get("candidate") == receipt.get("candidate_id")
                and integer(payload.get("planner_calls"), 1)
                and integer(inputs)
                and type(receipt.get("success")) is bool
                and type(payload.get("verified")) is bool
                and type(verification) is dict
                and type(verification.get("passed")) is bool
            )
            passed_pending = False
            for effect in pending["pending_effects"]:
                if candidate is None or effect.get("step") != candidate.get("intent"):
                    joined = False
                if "receipt_id" in effect and effect["receipt_id"] != receipt.get("receipt_id"):
                    joined = False
                passed_pending |= effect.get("passed") is True and effect.get("deadline_exceeded") is False
            supported = verification is not None and (verification.get("passed") is True or passed_pending)
            if payload.get("verified") is True and (not supported or receipt.get("success") is not True):
                joined = False
            classification = "unknown"
            if joined:
                classification = (
                    ("zero_input_verified" if payload["verified"] else "zero_input_unverified")
                    if inputs == 0
                    else ("positive_verified_after_pending" if passed_pending else "positive_verified")
                    if payload["verified"]
                    else "positive_unverified_or_failed"
                )
            block = {
                **pending,
                "final_sequence": index,
                "final_step": payload,
                "planner_call": payload.get("planner_calls"),
                "environment_steps": inputs,
                "joined": joined,
                "classification": classification,
            }
            blocks.append(block)
            if not joined:
                issues.append({"sequence": index, "reason": "contradictory runtime binding or final verdict"})
            pending = None
    if pending is not None:
        issues.append({"sequence": pending["receipt_sequence"], "reason": "orphan receipt"})
    episode = [event["payload"] for event in events if event["kind"] == "agent_episode"]
    if len(episode) != 1 or not integer(episode[0].get("steps")) or episode[0]["steps"] != last_step:
        issues.append({"sequence": len(events) - 1, "reason": "missing/inconsistent terminal agent_episode"})
    return blocks, issues


def compiled_matches(stage, steps):
    """Check declaration membership without importing or rerunning UI compilers."""
    require(
        type(steps) is list and steps and all(type(step) is dict for step in steps), "compiled current steps"
    )
    ids = [step.get("id") for step in steps]
    require(
        all(type(identifier) is str and identifier for identifier in ids) and len(set(ids)) == len(ids),
        "compiled step identity",
    )
    prior = set()
    for step in steps:
        dependencies = step.get("depends_on", [])
        require(
            type(dependencies) is list
            and all(type(item) is str for item in dependencies)
            and set(dependencies) <= prior,
            "compiled dependency points outside current stage",
        )
        prior.add(step["id"])
    arguments, tool = stage["arguments"], stage["tool"]

    def action_fields(declared, actual):
        operation = declared["operation"]  # press and key_press are distinct original operations.
        require(actual.get("operation") == operation, "declared action operation changed")
        value = declared.get("value", "")
        if operation in {"key_press", "hotkey"} and type(value) is str and value.isascii():
            value = value.lower()
        require(actual.get("value", "") == value, "declared action literal/key changed")
        for key, default in (("ref", ""), ("button", "left"), ("click_count", 1)):
            if key in declared:
                require(actual.get(key, default) == declared[key], "declared action binding/click changed")
        if "box" in declared and declared.get("box_space", "normalized_1000") == "normalized_1000":
            require(actual.get("box") == declared["box"], "declared normalized visual preparation changed")
        if "target" in declared and "ref" not in declared["target"]:
            require(
                normalized_target(actual.get("target")) == normalized_target(declared["target"]),
                "declared action semantic target changed",
            )
        # Initial current-ref target/guard resolution uses the original request
        # snapshot. The trace does not reproduce that entire snapshot; keep its
        # original ref source and actual guarded runtime join, never fabricate
        # a newly resolved target or treat it as a future-stage ref permission.

    if tool == "declare_navigation_stage":
        require(
            len(steps) == 1
            and steps[0].get("navigation") is True
            and steps[0].get("operation") == "click"
            and not steps[0].get("ref")
            and steps[0].get("box") is None
            and not steps[0].get("value")
            and normalized_target(steps[0].get("target")) == normalized_target(arguments["scope"]),
            "navigation may not precompile future stages",
        )
        require(
            steps[0].get("postconditions") == normalized_conditions(arguments["arrival"]),
            "compiled navigation arrival differs from original declaration",
        )
    elif tool == "declare_field_edits_stage":
        edits = arguments["edits"]
        require(
            len(steps) == len(edits) + int("commit" in arguments),
            "field steps include undeclared future work",
        )
        for edit, step in zip(edits, steps, strict=False):
            require(
                step.get("operation") == edit["operation"]
                and step.get("value", "") == edit.get("value", "")
                and normalized_target(step.get("target")) == normalized_target(edit.get("target")),
                "compiled field edit differs from source declaration",
            )
            ready = {
                "kind": "ready_to_" + edit["operation"],
                "target": edit["target"],
                "value": edit.get("value", ""),
            }
            require(
                step.get("preconditions", [])
                == normalized_conditions([*edit.get("preconditions", []), ready]),
                "authored field guard changed",
            )
            effect = {
                "kind": {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}[
                    edit["operation"]
                ],
                "target": edit["target"],
                **({"value": edit["value"]} if edit["operation"] == "fill" else {}),
            }
            require(
                step.get("postconditions") == normalized_conditions([effect, ready]),
                "compiled field effect changed",
            )
        if "commit" in arguments:
            commit, actual = arguments["commit"], steps[-1]
            action_fields(commit, actual)
            authored = commit.get("preconditions", [])
            if not any(type(item.get("target")) is dict and "ref" in item["target"] for item in authored):
                guards = normalized_conditions(authored)
                if edits:
                    effects = [
                        {
                            "kind": {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}[
                                edit["operation"]
                            ],
                            "target": edit["target"],
                            **({"value": edit["value"]} if edit["operation"] == "fill" else {}),
                        }
                        for edit in edits
                    ]
                    require(
                        actual.get("preconditions", []) == normalized_conditions(effects) + guards,
                        "authored/generated commit guard changed",
                    )
                else:
                    actual_guards = actual.get("preconditions", [])[: len(guards)]
                    require(len(actual_guards) == len(guards), "commit-only authored guards missing")
                    persistent = {
                        "value_equals": "fill",
                        "checked": "check",
                        "unchecked": "uncheck",
                        "selected": "click",
                    }
                    readiness = []
                    for declared_guard, guard in zip(guards, actual_guards, strict=True):
                        require(
                            guard.get("kind") == declared_guard["kind"]
                            and guard.get("value", "") == declared_guard["value"],
                            "commit-only authored guard effect changed",
                        )
                        expected_target, actual_target = (
                            declared_guard["target"],
                            normalized_target(guard.get("target")),
                        )
                        if expected_target != actual_target:
                            require(
                                expected_target is not None
                                and actual_target is not None
                                and declared_guard["kind"] in persistent
                                and expected_target["match"] == "exact"
                                and all(
                                    expected_target[key] == actual_target[key]
                                    for key in ("role", "app", "window", "ancestor", "match")
                                )
                                and " ".join(
                                    expected_target["name"]
                                    .strip()
                                    .removesuffix(":")
                                    .strip()
                                    .casefold()
                                    .split()
                                )
                                == " ".join(
                                    actual_target["name"].strip().removesuffix(":").strip().casefold().split()
                                ),
                                "commit-only rebound exceeds original native caption rule",
                            )
                        if guard["kind"] in persistent and guard["kind"] != "selected":
                            ready = {
                                "kind": "ready_to_" + persistent[guard["kind"]],
                                "target": actual_target,
                                "value": guard.get("value", ""),
                            }
                            if ready not in readiness:
                                readiness.append(ready)
                    if actual.get("derived_dialog_commit_effect") is True:
                        post = actual.get("postconditions", [])
                        require(
                            len(post) == 1 and post[0].get("kind") == "absent",
                            "derived commit-only dialog effect missing",
                        )
                        readiness.append({"kind": "exists", "target": post[0].get("target"), "value": ""})
                    require(
                        actual.get("preconditions", [])[len(guards) :] == readiness,
                        "commit-only original readiness/derived guards changed",
                    )
            if "recovery" in commit:
                require(actual.get("recovery") == commit["recovery"], "authored commit recovery changed")
            if "postconditions" in commit:
                require(
                    actual.get("postconditions") == normalized_conditions(commit["postconditions"]),
                    "field commit effect changed",
                )
    elif tool == "declare_property_stage":
        fields = [step for step in steps if type(step.get("property_edit")) is dict]
        require(len(fields) == 1, "property must have only its actual current template")
        field = fields[0]
        edit = field["property_edit"]
        require(
            field.get("id") == "property"
            and field.get("operation") == "click"
            and field.get("navigation") is True
            and not field.get("ref")
            and field.get("box") is None,
            "property must retain its original inert navigation template",
        )
        require(
            edit.get("operation") == arguments["edit_operation"]
            and edit.get("value", "") == arguments.get("edit_value", ""),
            "property literal changed",
        )
        expected_key = (
            ""
            if arguments["numeric_finalize_key"] == "none"
            else arguments["numeric_finalize_key"].casefold()
        )
        require(edit.get("numeric_finalize_key", "") == expected_key, "numeric strategy changed")
        require(
            normalized_target(field.get("target")) == normalized_target(arguments["scope"]),
            "compiled property scope changed",
        )
        prepare, confirm = "prepare" in arguments, arguments["confirmation_label"] != "none"
        require(len(steps) == 1 + int(prepare) + int(confirm), "property compiled undeclared stages")
        if prepare:
            action_fields(arguments["prepare"], steps[0])
        if confirm:
            confirmation = steps[-1]
            require(
                confirmation.get("id") == "confirm"
                and confirmation.get("operation") == "confirm_property"
                and confirmation.get("property_confirmation_of") == field["id"]
                and confirmation.get("depends_on") == [field["id"]]
                and not confirmation.get("ref")
                and not confirmation.get("value")
                and confirmation.get("box") is None
                and confirmation.get("target") is None
                and not confirmation.get("preconditions")
                and not confirmation.get("postconditions"),
                "unguarded property confirmation",
            )
    else:
        raise ValueError("unsupported declared stage")
    return set(ids)


def _audit_trace_impl(events):
    """Audit original program events; retain original fixed result separately."""
    legacy = legacy_audit_trace(events)  # Exact original logic, no injected/filtered events.
    blocks, join_issues = runtime_join(events)
    by_final = {block["final_sequence"]: block for block in blocks}
    by_receipt = {block["receipt"].get("receipt_id"): block for block in blocks}
    transports, observations, ordinary_plans = {}, {}, {}
    for index, event in enumerate(events):
        payload = event["payload"]
        if event["kind"] == "agent_model_output_transport":
            transports.setdefault(payload.get("call"), []).append((index, payload))
        elif event["kind"] == "agent_planning_observation":
            observations.setdefault(payload.get("call"), []).append((index, payload))
        elif event["kind"] == "agent_plan":
            ordinary_plans.setdefault(payload.get("call"), []).append((index, payload))
    requests = [
        {
            "call": call,
            "sequence": sequence,
            "transport": payload,
            "accepted": False,
            "rejected_without_input": False,
        }
        for call, rows in transports.items()
        for sequence, payload in rows
        if type(payload.get("declaration")) is dict
        and payload["declaration"].get("program_declaration") is True
    ]
    epochs, active, closed_ids, orphan_program_events = [], None, set(), []

    def issue(epoch, reason, index):
        epoch["issues"].append({"sequence": index, "reason": reason})

    def common(epoch, payload, index):
        if not (
            payload.get("program_id") == epoch["program_id"]
            and integer(payload.get("origin_planner_call"), 1)
            and payload.get("origin_planner_call") == epoch["origin_planner_call"]
            and finite(payload.get("original_plan_at"))
            and payload.get("original_plan_at") == epoch["original_plan_at"]
            and integer(payload.get("plan_actions"))
            and payload["plan_actions"] == epoch["logical_actions"]
            and integer(payload.get("remaining_actions"))
            and payload.get("remaining_actions") == max(0, epoch["horizon"] - epoch["logical_actions"])
            and payload.get("logical_attempt_budget") == epoch["horizon"]
            and payload.get("input_authority") is False
            and payload.get("task_success") is False
        ):
            issue(epoch, "program origin/clock/actual logical budget changed", index)

    def full_frame(epoch, payload, index):
        frame = {
            key: payload.get(key)
            for key in (
                "snapshot_fingerprint",
                "image_sha256",
                "environment_steps",
                "fresh_observe_calls",
                "source",
            )
        }
        if not (
            hash64(frame["snapshot_fingerprint"])
            and hash64(frame["image_sha256"])
            and integer(frame["environment_steps"])
            and integer(frame["fresh_observe_calls"], 1)
            and frame["source"] in FULL_SOURCES
        ):
            issue(epoch, "source-bound current full frame unavailable", index)
        if frame["source"] == "stale_rejection_preflight":
            # This is a real zero-input refusal's newly read full preflight,
            # never a verified input or a positive anchor. The reuse event has
            # no frame hashes in the actual runner; bind its exact source
            # receipt and bounded age, then the boundary's proof fields and
            # identical subsequent activation frame, without inventing fields.
            links = [item for item in epoch["stale_reuse_links"] if item["sequence"] < index]
            link = links[-1] if links else None
            source = by_receipt.get(link["receipt_id"]) if link else None
            if not (
                link is not None
                and source is not None
                and source["joined"]
                and source["final_step"].get("verified") is False
                and source["environment_steps"] == 0
                and source["receipt"].get("success") is False
                and source["receipt"].get("output", {}).get("failure_code") == "stale_observation"
                and source["final_sequence"] < link["recovery_sequence"] < link["sequence"] < index
                and not any(source["final_sequence"] < other["receipt_sequence"] < index for other in blocks)
                and finite(link["response_age_ms"])
                and 0 <= link["response_age_ms"] <= 1000
            ):
                issue(epoch, "stale source lacks actual bounded refusal/recovery/reuse chain", index)
            elif link.get("frame") is not None and link["frame"] != frame:
                issue(epoch, "one stale full-frame reuse changed fingerprint/image/counters", index)
            elif (
                link.get("frame") is None
                and epoch["last_frame"] is not None
                and frame["fresh_observe_calls"] <= epoch["last_frame"]["fresh_observe_calls"]
            ):
                issue(epoch, "stale preflight source lacks its actual new full read", index)
            elif link is not None:
                link["frame"] = frame
        if epoch["input_baseline"] is None and integer(frame["environment_steps"]):
            epoch["input_baseline"] = frame["environment_steps"]
        if (
            epoch["input_baseline"] is not None
            and frame["environment_steps"] != epoch["input_baseline"] + epoch["raw_inputs"]
        ):
            issue(epoch, "unaccounted environment input counter", index)
        if (
            epoch["last_frame"] is not None
            and integer(frame["fresh_observe_calls"])
            and frame["fresh_observe_calls"] < epoch["last_frame"]["fresh_observe_calls"]
        ):
            issue(epoch, "full-frame read counter reset", index)
        epoch["last_frame"] = frame
        return frame

    def end(epoch, status, index):
        epoch["status"], epoch["end_sequence"] = status, index
        closed_ids.add(epoch["program_id"])

    def stage_fields(epoch, payload, index):
        common(epoch, payload, index)
        stage = epoch["stages"][epoch["stage_index"]]
        if (
            not integer(payload.get("stage_index"))
            or payload.get("stage_index") != epoch["stage_index"]
            or payload.get("stage_id") != stage["id"]
        ):
            issue(epoch, "stage order or identity changed", index)

    for index, event in enumerate(events):
        kind, payload = event["kind"], event["payload"]
        if kind in {"agent_model_output_transport", "agent_planning_observation"}:
            if (
                active is not None
                and integer(payload.get("call"), 1)
                and payload["call"] != active["origin_planner_call"]
            ):
                end(active, "ended_by_new_actual_model_call", index)
                active = None
        if kind == "agent_program_declared":
            if active is not None:
                issue(active, "new declaration without retiring prior program", index)
                end(active, "overwritten", index)
            declaration = payload.get("declaration", {})
            epoch = {
                "program_id": payload.get("program_id"),
                "origin_planner_call": payload.get("origin_planner_call"),
                "declaration_sequence": index,
                "original_plan_at": payload.get("original_plan_at"),
                "horizon": payload.get("logical_attempt_budget"),
                "stages": [],
                "stage_index": 0,
                "logical_actions": 0,
                "raw_inputs": 0,
                "raw_input_count_complete": True,
                "input_baseline": None,
                "last_frame": None,
                "stage": None,
                "qualified": [],
                "issues": [],
                "status": "active",
                "numeric_ids": set(),
                "numeric_waiting": set(),
                "numeric_committed_keys": {},
                "unexplained": set(),
                "known_repeat_used": False,
                "stale_anchors": [],
                "recovery_gates": [],
                "stale_recovery_sources": {},
                "stale_reuse_links": [],
                "event_evidence": [{"sequence": index, "kind": kind, "payload_original": payload}],
            }
            epochs.append(epoch)
            active = epoch
            try:
                call = epoch["origin_planner_call"]
                require(
                    epoch["program_id"] not in closed_ids
                    and sum(item["origin_planner_call"] == call for item in epochs) == 1,
                    "accepted program identity/origin reused",
                )
                require(
                    integer(call, 1)
                    and integer(epoch["horizon"], 2)
                    and epoch["horizon"] <= 12
                    and finite(epoch["original_plan_at"]),
                    "program identity/clock/horizon",
                )
                source = declaration.get("source_declaration")
                require(
                    declaration.get("source_declaration_status") == "original_canonical_json"
                    and type(source) is dict
                    and set(source) == {"label", "stages"},
                    "accepted original source declaration missing",
                )
                stages = source["stages"]
                require(
                    type(stages) is list and 2 <= len(stages) <= epoch["horizon"],
                    "original ordered stage bound",
                )
                sha = canonical_sha(source)
                require(
                    declaration.get("declaration_sha256") == sha
                    and declaration.get("max_actions") == epoch["horizon"]
                    and epoch["program_id"] == f"{call}:{sha}",
                    "accepted declaration canonical SHA/identity",
                )
                normalized = declaration.get("stages")
                require(
                    type(normalized) is list and len(normalized) == len(stages),
                    "normalized declaration count",
                )
                require(len({stage["id"] for stage in stages}) == len(stages), "duplicate stage IDs")
                for source_stage, stage in zip(stages, normalized, strict=True):
                    require(
                        type(source_stage.get("id")) is str
                        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", source_stage["id"])
                        and type(source_stage.get("arguments")) is dict
                        and source_stage["tool"]
                        in {"declare_navigation_stage", "declare_property_stage", "declare_field_edits_stage"}
                        and source_stage["arguments"].get("completion") == "stage",
                        "inert semantic members only",
                    )
                    require(
                        stage.get("id") == source_stage["id"]
                        and stage.get("tool") == source_stage["tool"]
                        and stage.get("arguments") == source_stage["arguments"]
                        and stage.get("compiled") is False
                        and stage.get("entry_conditions")
                        == normalized_conditions(source_stage.get("entry_conditions", []))
                        and stage.get("recovery") == source_stage.get("recovery", "observe"),
                        "normalized source member mismatch or fake future compilation",
                    )
                epoch["stages"] = stages
                transport_rows, planning_rows = transports.get(call, []), observations.get(call, [])
                require(
                    len(transport_rows) == len(planning_rows) == 1
                    and transport_rows[0][0] < planning_rows[0][0] < index,
                    "one actual transport/planning origin required",
                )
                transport, planning = transport_rows[0][1], planning_rows[0][1]
                mapped = transport.get("declaration", {})
                require(
                    transport.get("response_transport") == PROGRAM_FORMAT
                    and transport.get("max_actions") == epoch["horizon"]
                    and mapped.get("tool_name") == PROGRAM_TOOL
                    and mapped.get("program_declaration") is True
                    and mapped.get("program_declaration_sha256") == sha
                    and hash64(mapped.get("mapped_object_sha256"))
                    and hash64(mapped.get("argument_sha256")),
                    "real program transport/source/mapping binding",
                )
                require(
                    planning.get("effective_max_actions")
                    == planning.get("requested_max_actions")
                    == epoch["horizon"]
                    and planning.get("a11y_complete") is True
                    and type(planning.get("image_uploaded")) is bool
                    and hash64(planning.get("native_execution_fingerprint")),
                    "actual original planning configuration",
                )
                require(not ordinary_plans.get(call), "program must not manufacture an old agent_plan epoch")
                epoch["request_fingerprint"] = planning["native_execution_fingerprint"]
                epoch["source_declaration"] = source
                epoch["observed_original_argument_sha256_not_independently_reconstructed"] = mapped[
                    "argument_sha256"
                ]
                epoch["observed_ordered_mapped_object_sha256_not_independently_reconstructed"] = mapped[
                    "mapped_object_sha256"
                ]
                epoch["independently_recomputed_canonical_mapped_object_sha256"] = canonical_sha(
                    {"intent": PROGRAM_INTENT, **source}
                )
                for request in requests:
                    if request["call"] == call:
                        request["accepted"] = True
                common(epoch, payload, index)
            except (ValueError, KeyError, TypeError) as error:
                issue(epoch, "declaration/origin invalid: " + str(error), index)
        elif kind.startswith("agent_program_") and kind != "agent_program_declared":
            if active is None:
                orphan_program_events.append({"sequence": index, "kind": kind, "payload_original": payload})
                continue
            epoch = active
            epoch["event_evidence"].append({"sequence": index, "kind": kind, "payload_original": payload})
            if not epoch["stages"]:
                continue
            if kind == "agent_program_stage_activated":
                target_index = payload.get("stage_index")
                common(epoch, payload, index)
                try:
                    if epoch["stage"] is None:
                        require(
                            target_index == 0
                            and payload.get("source") == "actual_planner_request"
                            and payload.get("real_planner_invocation") is True
                            and payload.get("freshness_claim") is False
                            and payload.get("snapshot_fingerprint") == epoch.get("request_fingerprint")
                            and payload.get("admitted") is False,
                            "stage0 request is not fresh admission",
                        )
                        admitted, frame = False, None
                    else:
                        previous = epoch["stage"]
                        require(
                            previous["completed"]
                            and target_index == epoch["stage_index"] + 1
                            and not epoch["unexplained"],
                            "next entry cannot override unverified current exit",
                        )
                        require(
                            payload.get("http_calls") == 0
                            and type(payload.get("http_calls")) is int
                            and payload.get("admitted") is True
                            and payload.get("one_shot_entry_admission") is True,
                            "next activation is not a zero-HTTP admitted stage",
                        )
                        frame = full_frame(epoch, payload, index)
                        require(
                            frame == previous["completion_frame"],
                            "activation must use exact completed-stage full frame",
                        )
                        require(
                            facts_true(
                                payload.get("entry_conditions"),
                                normalized_conditions(
                                    epoch["stages"][target_index].get("entry_conditions", [])
                                ),
                                allow_empty=True,
                            ),
                            "future entry not actually true",
                        )
                        admitted = True
                    require(
                        integer(target_index)
                        and target_index < len(epoch["stages"])
                        and payload.get("stage_id") == epoch["stages"][target_index]["id"],
                        "activation stage identity",
                    )
                    ids = compiled_matches(epoch["stages"][target_index], payload.get("compiled_steps"))
                    epoch["stage_index"] = target_index
                    epoch["stage"] = {
                        "index": target_index,
                        "activation_sequence": index,
                        "admitted": admitted,
                        "compiled_ids": ids,
                        "compiled_steps": payload["compiled_steps"],
                        "blocks": [],
                        "completed": False,
                        "completion_frame": None,
                        "entry_admission_frame": frame,
                    }
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    issue(epoch, "activation invalid: " + str(error), index)
            elif kind == "agent_program_stage_admitted":
                stage_fields(epoch, payload, index)
                stage = epoch["stage"]
                if stage is None or stage["admitted"] or epoch["stage_index"] != 0:
                    issue(epoch, "duplicate or misplaced one-shot entry admission", index)
                else:
                    frame = full_frame(epoch, payload, index)
                    valid = (
                        payload.get("admitted") is True
                        and payload.get("one_shot_entry_admission") is True
                        and payload.get("initial_plan_not_recompiled") is True
                        and payload.get("planned_request_binding_unchanged") is True
                        and payload.get("original_action_preconditions_and_preflight_still_required") is True
                        and facts_true(
                            payload.get("entry_conditions"),
                            normalized_conditions(epoch["stages"][0].get("entry_conditions", [])),
                            allow_empty=True,
                        )
                    )
                    if not valid:
                        issue(epoch, "stage0 current full entry admission missing", index)
                    stage["admitted"], stage["entry_admission_frame"] = valid, frame
            elif kind == "agent_program_recovery_policy":
                stage_fields(epoch, payload, index)
                declared = epoch["stages"][epoch["stage_index"]].get("recovery", "observe")
                original = payload.get("original_observe_eligible")
                permitted = original is True and declared == "observe"
                if not (
                    payload.get("stage_recovery") == declared
                    and type(original) is bool
                    and payload.get("optional_observe_permitted") is permitted
                    and type(payload.get("event_observation_calls")) is int
                    and payload["event_observation_calls"] == 0
                    and payload.get("selected_original_branch")
                    == ("bounded_observe" if permitted else "original_nonobserve_path")
                ):
                    issue(epoch, "recovery declaration cannot upgrade or fabricate original route", index)
                gate = {
                    "sequence": index,
                    "stage_index": epoch["stage_index"],
                    "payload_original": payload,
                    "actual_pending_effect_count": None,
                    "awaiting_observe_budget_charges": 0,
                }
                if payload.get("phase") == "post_input_pending":
                    block = by_receipt.get(payload.get("source_receipt_id"))
                    if block is None or not block["receipt_sequence"] < index < block["final_sequence"]:
                        issue(epoch, "recovery route lacks exact source receipt", index)
                    else:
                        actual_step = (
                            next(
                                (
                                    step
                                    for step in epoch["stage"]["compiled_steps"]
                                    if step["id"] == block["candidate"].get("intent")
                                ),
                                None,
                            )
                            if epoch["stage"]
                            else None
                        )
                        if (
                            payload.get("original_effect_recovery") not in {"observe", "replan"}
                            or (original is True and payload.get("original_effect_recovery") != "observe")
                            or (
                                actual_step is not None
                                and "property_edit" not in actual_step
                                and not actual_step.get("property_confirmation_of")
                                and payload.get("original_effect_recovery")
                                != actual_step.get("recovery", "replan")
                            )
                        ):
                            issue(
                                epoch,
                                "stage recovery cannot upgrade the actual original effect policy",
                                index,
                            )
                        actual = len(block["pending_effects"])
                        gate["actual_pending_effect_count"] = actual
                        if actual and not permitted:
                            issue(epoch, "nonobserve policy actually polled", index)
                        launch = block["verification"].get("details", {}).get("application_wait", {})
                        cap = (
                            20
                            if type(launch) is dict
                            and launch.get("eligible") is True
                            and not any(
                                e.get("property_navigation_wait") is True for e in block["pending_effects"]
                            )
                            else 2
                        )
                        if actual > cap:
                            issue(epoch, "original pending observation cap exceeded", index)
                elif payload.get("phase") != "awaiting_conditions":
                    issue(epoch, "unknown original recovery gate", index)
                epoch["recovery_gates"].append(gate)
            elif kind == "agent_program_stage_completed":
                stage_fields(epoch, payload, index)
                stage = epoch["stage"]
                if (
                    stage is None
                    or not stage["admitted"]
                    or stage["completed"]
                    or epoch["unexplained"]
                    or epoch["numeric_waiting"]
                ):
                    issue(epoch, "unadmitted/unknown stage cannot complete", index)
                    continue
                frame = full_frame(epoch, payload, index)
                if not facts_true(payload.get("terminal_conditions")):
                    issue(epoch, "nonempty true terminal facts required", index)
                # For ordinary compiled effects, bind the exact terminal
                # predicates. For a dynamic property/confirmation template,
                # bind the actual selected receipt's recorded effect evidence.
                # No future stage entry may substitute for the current exit.
                depended_on = {
                    dependency
                    for step in stage["compiled_steps"]
                    for dependency in step.get("depends_on", [])
                }
                expected_terminal = []
                for step in stage["compiled_steps"]:
                    if step["id"] in depended_on:
                        continue
                    conditions = step.get("postconditions", [])
                    if not conditions:
                        actuals = [
                            block
                            for block in stage["blocks"]
                            if block["candidate"].get("intent") == step["id"]
                            and block["final_step"].get("verified") is True
                        ]
                        conditions = (
                            actuals[-1]["receipt"].get("output", {}).get("postcondition_evidence", [])
                            if actuals
                            else []
                        )
                    if not conditions:
                        issue(
                            epoch,
                            "terminal effect lacks compiled or selected-receipt predicate identity",
                            index,
                        )
                    expected_terminal.extend(
                        {"kind": condition.get("kind"), "target": normalized_target(condition.get("target"))}
                        for condition in conditions
                    )
                if not facts_true(payload.get("terminal_conditions"), expected_terminal):
                    issue(epoch, "current exit facts differ from actual terminal effect", index)
                attempts = payload.get("attempt_receipts")
                expected = [
                    {
                        "receipt_id": block["receipt"]["receipt_id"],
                        "environment_steps": block["environment_steps"],
                        "final_verified": block["final_step"]["verified"],
                    }
                    for block in stage["blocks"]
                ]
                if (
                    type(attempts) is not list
                    or len(attempts) != len(expected)
                    or any(
                        any(
                            attempt.get(key) != value or type(attempt.get(key)) is not type(value)
                            for key, value in row.items()
                        )
                        for attempt, row in zip(attempts or [], expected, strict=False)
                    )
                ):
                    issue(epoch, "actual failed/readonly/positive attempts not retained", index)
                verified_ids = [
                    block["receipt"]["receipt_id"]
                    for block in stage["blocks"]
                    if block["final_step"]["verified"] is True
                ]
                if payload.get("receipt_ids") != verified_ids:
                    issue(
                        epoch,
                        "stage verified receipts must not include false fill or fabricated receipt",
                        index,
                    )
                if (
                    payload.get("reason") != ("verified_receipt" if verified_ids else "read_only_predicates")
                    or payload.get("read_only")
                    is not all(block["environment_steps"] == 0 for block in stage["blocks"])
                    or payload.get("positive_input")
                    is not any(integer(block["environment_steps"], 1) for block in stage["blocks"])
                    or not integer(payload.get("pure_predicate_logical_attempts"))
                    or payload["pure_predicate_logical_attempts"] != 0
                    or payload.get("runtime_receipts_keep_original_logical_attempt_charge") is not True
                ):
                    issue(
                        epoch,
                        "readonly/failed/positive charge classification contradicts actual attempts",
                        index,
                    )
                if payload.get("positive_verified_input") is not any(
                    block["environment_steps"] > 0 and block["final_step"]["verified"] is True
                    for block in stage["blocks"]
                ):
                    issue(epoch, "positive input verification credit mismatch", index)
                stage["completed"], stage["completion_frame"] = True, frame
            elif kind == "agent_program_completed":
                stage_fields(epoch, payload, index)
                if (
                    epoch["stage"] is None
                    or not epoch["stage"]["completed"]
                    or epoch["stage_index"] != len(epoch["stages"]) - 1
                    or payload.get("all_declared_stages_verified") is not True
                    or payload.get("whole_goal_review_still_required") is not True
                ):
                    issue(epoch, "program completion bypassed ordered stages or whole-goal review", index)
                end(epoch, "completed_local_stages_not_task_success", index)
                active = None
            elif kind == "agent_program_abandoned":
                stage_fields(epoch, payload, index)
                if payload.get("whole_program_retired") is not True:
                    issue(epoch, "unexplained program tail survived abandonment", index)
                end(epoch, "abandoned", index)
                active = None
        if active is not None and kind == "agent_declared_numeric_finalize":
            source_id = payload.get("source_receipt_id")
            source = by_receipt.get(source_id)
            if payload.get("status") == "pending":
                arguments = active["stages"][active["stage_index"]]["arguments"] if active["stages"] else {}
                if not (
                    source is not None
                    and source["planner_call"] == active["origin_planner_call"]
                    and source["receipt_sequence"] < index < source["final_sequence"]
                    and source["final_step"].get("verified") is False
                    and type(arguments.get("numeric_finalize_key")) is str
                    and arguments["numeric_finalize_key"] != "none"
                    and type(payload.get("key")) is str
                    and payload["key"].casefold() == arguments["numeric_finalize_key"].casefold()
                ):
                    issue(active, "pending numeric attempt lacks original declared key/source receipt", index)
                else:
                    active["numeric_ids"].add(source_id)
                    active["numeric_waiting"].add(source_id)
            elif payload.get("status") == "committed":
                key = by_receipt.get(payload.get("receipt_id"))
                if not (
                    source_id in active["numeric_waiting"]
                    and source is not None
                    and key is not None
                    and source_id != payload.get("receipt_id")
                    and source["final_sequence"] < key["receipt_sequence"] < index < key["final_sequence"]
                    and source["final_step"].get("verified") is False
                    and key["final_step"].get("verified") is True
                    and key["joined"]
                    and key["planner_call"] == active["origin_planner_call"]
                    and key["receipt"].get("capability") == "agent.osworld.key_press"
                    and payload.get("verified") is True
                ):
                    issue(
                        active,
                        "numeric commit must bind a new verified key without rewriting the false fill",
                        index,
                    )
                else:
                    active["numeric_committed_keys"][payload["receipt_id"]] = source_id
        if index in by_final and active is not None:
            epoch, block = active, by_final[index]
            stage = epoch["stage"]
            admitted_binding = bool(
                stage is not None
                and stage["admitted"]
                and not stage["completed"]
                and block["joined"]
                and block["planner_call"] == epoch["origin_planner_call"]
                and block["candidate"].get("intent") in stage["compiled_ids"]
            )
            if not admitted_binding:
                issue(epoch, "receipt not bound to the actually admitted current compiled stage", index)
            if epoch["unexplained"]:
                issue(epoch, "same program continued unexplained failed/unknown input", index)
            rid = block["receipt"].get("receipt_id")
            numeric_authorized = not epoch["numeric_waiting"] or rid in epoch["numeric_committed_keys"]
            if epoch["numeric_waiting"] and rid not in epoch["numeric_waiting"]:
                if block["receipt"].get("capability") != "agent.osworld.key_press":
                    issue(epoch, "numeric contingency retried field or skipped guarded key", index)
                if block["final_step"].get("verified") is True and rid not in epoch["numeric_committed_keys"]:
                    issue(epoch, "numeric key lacks original committed source binding", index)
            if rid in epoch["numeric_committed_keys"]:
                epoch["numeric_waiting"].discard(epoch["numeric_committed_keys"][rid])
            if stage is not None:
                stage["blocks"].append(block)
            if integer(block["environment_steps"]):
                epoch["raw_inputs"] += block["environment_steps"]
            else:
                epoch["raw_input_count_complete"] = False
            if block["final_step"].get("verified") is True:
                epoch["logical_actions"] += 1  # Includes original runtime readonly receipt attempts.
                if (
                    admitted_binding
                    and numeric_authorized
                    and not epoch["unexplained"]
                    and block["environment_steps"] > 0
                ):
                    epoch["qualified"].append(
                        {
                            "stage_index": epoch["stage_index"],
                            "agent_step": block["final_step"]["step"],
                            "receipt_id": block["receipt"]["receipt_id"],
                            "sequence": index,
                        }
                    )
            elif block["receipt"].get("receipt_id") in epoch["numeric_ids"]:
                epoch["logical_actions"] += 1  # Pending numeric fill remains false, charged once.
            else:
                epoch["unexplained"].add(block["receipt"].get("receipt_id"))
            if epoch["logical_actions"] > epoch["horizon"]:
                issue(epoch, "original shared logical horizon exceeded", index)
        if active is not None and kind == "agent_stale_recovery" and payload.get("stage") == "requested":
            rid = payload.get("receipt_id")
            block = by_receipt.get(rid)
            if block is None or not (
                block["final_sequence"] < index
                and block["environment_steps"] == 0
                and block["planner_call"] == active["origin_planner_call"]
                and active["stage"] is not None
                and block in active["stage"]["blocks"]
                and block["receipt"].get("output", {}).get("failure_code") == "stale_observation"
            ):
                issue(active, "stale refresh has no exact zero-input refusal", index)
            else:
                active["unexplained"].discard(rid)
                anchor = len(active["qualified"]) if payload.get("per_progress_limit") == 1 else -1
                if anchor in active["stale_anchors"]:
                    issue(active, "original stale refresh ticket rearmed without permitted progress", index)
                active["stale_anchors"].append(anchor)
                active["stale_recovery_sources"][rid] = index
        if (
            active is not None
            and kind == "agent_observation_reuse"
            and payload.get("source") == "stale_rejection_preflight"
        ):
            rid = payload.get("receipt_id")
            active["stale_reuse_links"].append(
                {
                    "sequence": index,
                    "receipt_id": rid,
                    "recovery_sequence": active["stale_recovery_sources"].get(rid, -1),
                    "response_age_ms": payload.get("response_age_ms"),
                    "frame": None,
                }
            )
        if (
            active is not None
            and kind == "agent_navigation_continuation"
            and payload.get("stage") == "fresh_continuation_requested"
        ):
            rid, block = payload.get("receipt_id"), by_receipt.get(payload.get("receipt_id"))
            if (
                block is None
                or active["known_repeat_used"]
                or block["final_step"].get("verified") is not False
                or not integer(block["environment_steps"], 1)
                or not (
                    block["planner_call"] == active["origin_planner_call"]
                    and active["stage"] is not None
                    and block in active["stage"]["blocks"]
                    and block["receipt"].get("success") is True
                )
            ):
                issue(active, "known-repeat continuation is not one retained false attempt", index)
            else:
                active["known_repeat_used"] = True
                active["unexplained"].discard(rid)
                active["logical_actions"] += 1
                if (
                    payload.get("plan_actions") != active["logical_actions"]
                    or payload.get("original_plan_at") != active["original_plan_at"]
                ):
                    issue(active, "retained cycle charge/clock differs from original", index)
        if active is not None and kind == "agent_recovery" and payload.get("reason") == "awaiting_conditions":
            gates = [
                gate
                for gate in active["recovery_gates"]
                if gate["payload_original"].get("phase") == "awaiting_conditions"
            ]
            if not gates or gates[-1]["payload_original"].get("optional_observe_permitted") is not True:
                issue(active, "actual awaiting observation lacks original permitted route", index)
            else:
                gates[-1]["awaiting_observe_budget_charges"] += 1
                before = gates[-1]["payload_original"].get("recovery_observations_before")
                if not integer(before) or before >= 2 or payload.get("attempt") != before + 1:
                    issue(active, "original awaiting observation cap/counter changed", index)
        if kind == "agent_episode" and active is not None:
            end(active, "ended_by_terminal_episode", index)
            active = None
    if active is not None:
        issue(active, "program trace did not reach terminal episode", len(events) - 1)
    for epoch in epochs:
        for problem in join_issues:
            if epoch["declaration_sequence"] <= problem["sequence"] <= epoch.get("end_sequence", len(events)):
                issue(epoch, problem["reason"], problem["sequence"])
    # Requests rejected by original first activation/validator are not accepted
    # epochs. They need no invented source_declaration and retain all HTTP cost.
    # Actual runner emits invalid_plan in except, then current-call transport
    # and planning events from finally. Match this real order one-to-one within
    # the immediately preceding model-output interval, not to arbitrary later
    # failures or an invented accepted epoch.
    used_rejections = set()
    transport_sequences = sorted(sequence for rows in transports.values() for sequence, _ in rows)
    for request in requests:
        if request["accepted"]:
            continue
        previous = max(
            (sequence for sequence in transport_sequences if sequence < request["sequence"]), default=-1
        )
        rejections = [
            index
            for index in range(previous + 1, request["sequence"])
            if events[index]["kind"] == "agent_recovery"
            and events[index]["payload"].get("reason") == "invalid_plan"
            and index not in used_rejections
        ]
        if len(rejections) == 1:
            rejected = rejections[0]
            request["rejected_without_input"] = not any(
                block["planner_call"] == request["call"] for block in blocks
            ) and not any(
                events[index]["kind"] in {"agent_plan", "agent_program_declared", "receipt"}
                for index in range(rejected + 1, request["sequence"])
            )
            request["actual_rejection_sequence"] = rejected
            used_rejections.add(rejected)
    legacy_bound_ids = {
        block["receipt_id"] for block in legacy["agent_steps"] if block["classification"] != "unknown"
    }
    orphan_blocks = [
        block
        for block in blocks
        if not any(
            epoch["declaration_sequence"]
            <= block["receipt_sequence"]
            <= epoch.get("end_sequence", len(events))
            and block["planner_call"] == epoch["origin_planner_call"]
            for epoch in epochs
        )
        and block["receipt"].get("receipt_id") not in legacy_bound_ids
    ]
    program_relevant = bool(epochs or requests or orphan_program_events)
    complete = (
        program_relevant
        and not orphan_blocks
        and not orphan_program_events
        and all(not epoch["issues"] for epoch in epochs)
    )
    complete = complete and all(
        request["accepted"] or request["rejected_without_input"] for request in requests
    )
    counts = sum(max(0, len(epoch["qualified"]) - 1) for epoch in epochs)
    cross = sum(
        sum(
            left["stage_index"] != right["stage_index"]
            for left, right in zip(epoch["qualified"], epoch["qualified"][1:], strict=False)
        )
        for epoch in epochs
    )
    exported = []
    for epoch in epochs:
        row = {
            key: value
            for key, value in epoch.items()
            if key
            not in {"stage", "numeric_ids", "numeric_waiting", "unexplained", "stale_anchors", "raw_inputs"}
        }
        row["raw_environment_inputs_complete"] = (
            epoch["raw_inputs"] if epoch["raw_input_count_complete"] else None
        )
        row["raw_environment_inputs_observed_partial_sum"] = epoch["raw_inputs"]
        exported.append(row)
    return {
        "kind": "program_aware_positive_input_reuse_trace_audit_v1",
        "run_id": legacy["run_id"],
        "legacy_fixed_original": legacy,
        "legacy_fixed_applies_to_programs": False,
        "program_metric_status": "complete"
        if complete
        else "unknown"
        if program_relevant
        else "not_applicable",
        "positive_input_program_continuations": counts if complete else None,
        "cross_stage_positive_input_program_continuations": cross if complete else None,
        "observed_program_continuations_lower_bound": counts,
        "program_requests": requests,
        "program_epochs": exported,
        "orphan_program_events": orphan_program_events,
        "unbound_runtime_receipts": [block["receipt"].get("receipt_id") for block in orphan_blocks],
        "runtime_join_issues": join_issues,
        "agent_episode_original": next(
            (event["payload"] for event in events if event["kind"] == "agent_episode"), None
        ),
        "mechanism_is_not_official_task_success": True,
        "acceptance_satisfied": False,
        "limitations": [
            "Original argument and insertion-order mapped hashes remain observed metadata; "
            "canonical source does not preserve original argument whitespace or nested insertion order.",
            "Source-bound reported true facts and origin continuity are audited, "
            "not a new UI observation or independent replay.",
            "Constant original_plan_at proves origin continuity; wall timestamps are not "
            "a conversion of the original perf-counter TTL.",
        ],
    }


def audit_trace(events):
    """Fail closed for malformed program evidence; retain legacy behavior/result."""
    try:
        result = _audit_trace_impl(events)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        relevant = type(events) is list and any(
            type(event) is dict
            and (
                str(event.get("kind", "")).startswith("agent_program_")
                or (
                    event.get("kind") == "agent_model_output_transport"
                    and type(event.get("payload")) is dict
                    and type(event["payload"].get("declaration")) is dict
                    and event["payload"]["declaration"].get("program_declaration") is True
                )
            )
            for event in events
        )
        if not relevant:
            raise  # Pure legacy malformed input keeps the original function behavior.
        try:
            legacy = legacy_audit_trace(events)
        except (ValueError, KeyError, TypeError, IndexError):
            legacy = {"metric_status": "unknown", "positive_input_plan_continuations": None}
        result = {
            "kind": "program_aware_positive_input_reuse_trace_audit_v1",
            "legacy_fixed_original": legacy,
            "legacy_fixed_applies_to_programs": False,
            "program_metric_status": "unknown",
            "positive_input_program_continuations": None,
            "cross_stage_positive_input_program_continuations": None,
            "observed_program_continuations_lower_bound": None,
            "program_epochs": [],
            "error_type": type(error).__name__,
            "malformed_evidence_not_coerced_to_zero": True,
            "agent_episode_original": next(
                (
                    event.get("payload")
                    for event in events
                    if type(event) is dict and event.get("kind") == "agent_episode"
                ),
                None,
            ),
            "acceptance_satisfied": False,
        }
    legacy = result["legacy_fixed_original"]
    result["metric_status"] = (
        legacy["metric_status"]
        if result["program_metric_status"] == "not_applicable"
        else result["program_metric_status"]
    )
    result["positive_input_plan_continuations"] = legacy.get("positive_input_plan_continuations")
    result["legacy_fixed_metric_status_original"] = legacy["metric_status"]
    result["method"] = (
        "Original source/ordered fresh admission plus actual joined final-verified positive inputs, "
        "under one real VLM origin."
    )
    return result


SELECTION_KIND = "root_selected_positive_input_reuse_evidence_v1"


def load_ref(root, reference, *, lines=False, evidence=None):
    """Hash/read bounded original files; no discovery, execution or trace edits."""
    require(
        type(reference) is dict
        and set(reference) == {"path", "sha256"}
        and type(reference["path"]) is str
        and hash64(reference["sha256"]),
        "invalid artifact ref",
    )
    runs = (root / "runs").resolve()
    path = Path(reference["path"])
    path = path if path.is_absolute() else root / path
    require(
        not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(runs),
        "artifact must be a real local runs file",
    )
    require(path.stat().st_size <= (64_000_000 if lines else 4_000_000), "artifact exceeds read bound")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "artifact changed")

    def nonfinite(value):
        raise ValueError("nonfinite JSON literal: " + value)

    data = raw.decode("utf-8-sig")
    parsed = (
        [json.loads(line, parse_constant=nonfinite) for line in data.splitlines() if line.strip()]
        if lines
        else json.loads(data, parse_constant=nonfinite)
    )
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"], "artifact changed during read"
    )
    if evidence is not None:
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": reference["sha256"],
                "bytes": len(raw),
                "format": "jsonl_original" if lines else "json_original",
            }
        )
    return parsed


def analyze(root, manifest):
    """Report terminal selected artifacts, retaining every selected failure/unknown."""
    require(
        type(manifest) is dict
        and manifest.get("kind") == SELECTION_KIND
        and manifest.get("terminal") is True,
        "root-selected terminal evidence is required",
    )
    reads = []
    terminal = load_ref(root, manifest["terminal_evidence"], evidence=reads)
    require(
        terminal.get("terminal") is True and terminal.get("active_child_may_be_running") is not True,
        "selected batch is not terminal",
    )
    episodes = manifest.get("episodes")
    require(type(episodes) is list and 1 <= len(episodes) <= 32, "bounded selected episodes required")
    identities = [(row.get("id"), row.get("task"), row.get("mode")) for row in episodes]
    require(len(set(identities)) == len(identities), "duplicate selected identity")
    rows = []
    for selected in episodes:
        row = {
            "id": selected.get("id"),
            "task": selected.get("task"),
            "mode": selected.get("mode"),
            "input_refs_original": selected,
            "metric_status": "unknown",
            "program_metric_status": "unknown",
            "positive_input_program_continuations": None,
            "positive_input_plan_continuations": None,
            "official_outcome_original": None,
        }
        try:
            summary = load_ref(root, selected["summary"], evidence=reads)
            source = load_ref(root, selected["source_manifest"], evidence=reads)
            require(type(source.get("files")) is dict, "source inventory missing")
            if terminal.get("source_manifest_sha256"):
                require(
                    terminal["source_manifest_sha256"] == selected["source_manifest"]["sha256"],
                    "selected source differs from terminal evidence",
                )
            matching = [
                item
                for item in summary.get("runs", [])
                if item.get("task") == selected["task"] and item.get("mode") == selected["mode"]
            ]
            require(len(matching) == 1, "summary identity ambiguous")
            original = matching[0]
            row["original_summary_row"] = original  # All costs/statuses, no success/timeout filtering.
            official = (
                load_ref(root, selected["official"], evidence=reads) if selected.get("official") else None
            )
            row["official_outcome_original"] = official
            if selected.get("copy_manifest"):
                load_ref(root, selected["copy_manifest"], evidence=reads)
            require(selected.get("trace") is not None, "original trace missing; no zero inferred")
            audit = audit_trace(load_ref(root, selected["trace"], lines=True, evidence=reads))
            row.update(audit)
            valid_official = (
                official is not None
                and official.get("closed") is True
                and type(official.get("official_score")) is type(original.get("official_score"))
                and official.get("official_score") == original.get("official_score")
                and official.get("evaluation_status") == original.get("evaluation_status")
            )
            if not valid_official:
                row["metric_status"] = "unknown"
                row["program_metric_status"] = "unknown"
                row["positive_input_program_continuations"] = None
                row["cross_stage_positive_input_program_continuations"] = None
                # Preserve the raw original fixed result under legacy_fixed_original.
                row["positive_input_plan_continuations"] = None
                row.setdefault("artifact_issues", []).append(
                    "original closed official/summary agreement unavailable"
                )
        except (ValueError, KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as error:
            row.update(
                metric_status="unknown",
                program_metric_status="unknown",
                positive_input_program_continuations=None,
                cross_stage_positive_input_program_continuations=None,
                positive_input_plan_continuations=None,
                error_type=type(error).__name__,
            )
        rows.append(row)
    return {
        "kind": "program_aware_positive_input_reuse_offline_audit_v1",
        "episodes": rows,
        "complete_episodes": sum(row["metric_status"] == "complete" for row in rows),
        "unknown_or_partial_episodes": sum(row["metric_status"] != "complete" for row in rows),
        "source_artifacts_read": reads,
        "fixed_legacy_analyzer_source_sha256": FIXED_ANALYZER_SHA256,
        "original_fixed_analyzer_unchanged_and_separately_required": True,
        "mechanism_is_not_official_task_success": True,
        "acceptance_satisfied": False,
        "method": "Original event ledger only; new program proof and original fixed legacy "
        "result remain separate. "
        "All selected episodes and original costs/statuses are retained, including unknown/missing trace. "
        "This consumer neither starts processes nor declares full benchmark acceptance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = load_ref(root, {"path": str(args.manifest), "sha256": args.manifest_sha256})
    result = analyze(root, manifest)
    result["selection_manifest_sha256"] = args.manifest_sha256
    result["analyzer_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output if args.output.is_absolute() else root / args.output
    require(
        output.parent.is_dir()
        and output.parent.resolve().is_relative_to((root / "runs").resolve())
        and not output.exists(),
        "choose a new output within runs",
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
