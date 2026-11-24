"""Slow multimodal planning, fast Jev choices, and independent termination."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

from .agent_planner import VisualInspectionRequested, _consume_local_read_timeout
from .agent_types import AgentBackend, AgentCondition, AgentPlan, AgentPlanner, AgentSnapshot, AgentStep
from .desktop_phase_handoff_v62 import HighLevelContextRequest
from .desktop_semantics import condition_evidence, condition_holds, conditions_hold, matching_controls
from .desktop_stage_program import StageProgramPlan
from .dispatch import DispatchEscalation, DispatchPolicy
from .errors import PolicyError
from .guard import ActionGuard
from .models import ActionCandidate, Channel, Decision, Observation
from .open_browser import PublicDecisionPolicy
from .policy import DecisionPolicy, JevFailure, JevPolicy
from .registry import ExecutorRegistry
from .runtime import (
    AgentRuntime,
    DecisionAdmissionRejected,
    DecisionAdmissionRejection,
    PolicySelectionFailure,
)
from .trace import JsonlTrace
from .verify import VerifierRegistry

STALE_RECOVERY_VERSION = "recompile_once_v2"
STALE_RECOVERY_PROGRESS_VERSION = "recompile_per_verified_input_v1"


_NATIVE_MISSING_VISUAL_GUIDANCE = (
    "If the CURRENT attached screenshot clearly shows a usable visual target, the already offered "
    "declare_visual_current_action can declare ONE visual_click or visual_fill. Use its original "
    "direct summary,id,done=false,operation,action_label,box_space,box fields; visual_fill also "
    "requires the requested nonempty ASCII literal value. No action wrapper, ref, semantic target "
    "or predicates in this visual family. Ground the label and box in the actual current image; "
    "do not invent native field, focus or owner evidence. All original image bounds, fresh-input, "
    "keyboard/focus, budget and completion rules still apply. This reminder grants no input or success."
)


def _native_missing_visual_feedback(planner, error, *, whole_goal_request=False, repair_context=None):
    """Describe an existing offered H1 option after the local zero-match refusal.

    This only changes diagnostic text; no declaration, target, action or guard is
    supplied. The failure string is the original validator's complete static form.
    """
    provenance = getattr(planner, "last_request_provenance", None)
    validation = getattr(planner, "last_validation_error", None)
    if (not isinstance(error, ValueError)
            or str(error) != "invalid multimodal planning response (ValueError)"
            or whole_goal_request or repair_context is not None
            or getattr(planner, "profile", None) != "desktop"
            or type(provenance) is not dict or provenance.get("max_actions") != 1
            or provenance.get("image_uploaded") is not True
            or provenance.get("whole_goal_recheck") is True
            or type(provenance.get("parameter_schema_sha256")) is not dict
            or "declare_visual_current_action" not in provenance["parameter_schema_sha256"]
            or not isinstance(validation, str)):
        return None
    pattern = (
        r"step 1: immediate target has no current executable control; "
        r"Requested operation=(?:click|fill|check|uncheck); matched current controls=0; "
        r"observed operation capabilities=none\. Capabilities are diagnostic only; "
        r"explicitly choose a uniquely grounded legal action\. "
        r"Use a currently executable target or one screenshot-grounded visual fallback"
    )
    return _NATIVE_MISSING_VISUAL_GUIDANCE if re.fullmatch(pattern, validation) else None

def _text_focus_guard_feedback(receipt) -> dict | None:
    """Only fixed backend facts change feedback, never actions or authority."""
    if (receipt.success is not False or receipt.capability != "agent.osworld.fill"
            or type(receipt.output) is not dict
            or receipt.output.get("failure_code") != "text_focus_guard_rejected"):
        return None
    item = receipt.output.get("text_focus_guard")
    facts = {"fresh_observe_returned", "a11y_complete", "image_present", "target_matched",
             "observed_disabled", "observed_editable", "observed_focused_true", "typing_not_sent"}
    if (type(item) is not dict or set(item) != facts | {"version", "stage"}
            or item["version"] != "text_focus_guard_diagnostic_v65"
            or type(item["stage"]) is not str
            or item["stage"] not in {"after_focus_click", "after_select_all"}
            or item["fresh_observe_returned"] is not True or item["typing_not_sent"] is not True
            or type(item["target_matched"]) is not bool
            or any(value is not None and type(value) is not bool
                   for key, value in item.items() if key in facts)):
        return None
    if not item["target_matched"] and any(item[key] is not None for key in (
            "observed_disabled", "observed_editable", "observed_focused_true")):
        return None
    if (item["target_matched"] and item["observed_disabled"] is False
            and item["observed_editable"] is True and item["observed_focused_true"] is True):
        return None
    inputs = receipt.output.get("environment_steps")
    if type(inputs) is not int or inputs != (1 if item["stage"] == "after_focus_click" else 2):
        return None
    prefix = ("The focus-establishing click was sent" if inputs == 1
              else "The focus-establishing click and select-all key were sent")
    return {"text_focus_guard": dict(item), "feedback": (
        prefix + ", but text typing was not sent. Safe editable text focus was not proven "
        "by the current observation. Inspect the fresh current UI and choose a newly grounded "
        "legal action; do not blindly repeat the same fill. This diagnostic grants no input "
        "authority or completion."
    )}


def _v52_request_provenance(planner, requested_limit: int) -> dict | None:
    """Read only the planner's actual request configuration, never model output."""
    provenance = getattr(planner, "last_request_provenance", None)
    if (type(provenance) is not dict
            or provenance.get("planning_observation") not in {
                "native_first_v51", "full_multimodal_v1"}
            or type(provenance.get("image_uploaded")) is not bool
            or type(provenance.get("a11y_complete")) is not bool
            or type(provenance.get("max_actions")) is not int
            or provenance["max_actions"] not in {1, requested_limit}):
        return None
    return {key: provenance[key] for key in (
        "planning_observation", "image_uploaded", "a11y_complete", "max_actions",
    )}


def _v52_visual_information_anchor(
    snapshot: AgentSnapshot, step: AgentStep, receipt, *, verified: bool,
    request_information: dict | None, recovery: bool, network_recovery: bool,
) -> dict | None:
    """An image-only verified input can retain information need, never permission."""
    output = receipt.output
    before, after = output.get("before_fingerprint"), output.get("after_fingerprint")
    if (not request_information or not recovery or network_recovery
            or request_information.get("planning_observation") != "native_first_v51"
            or request_information.get("image_uploaded") is not True
            or type(request_information.get("max_actions")) is not int
            or request_information["max_actions"] != 1
            or request_information.get("a11y_complete") is not True
            or step.operation not in {"visual_click", "visual_fill"}
            or receipt.success is not True or verified is not True
            or type(output.get("environment_steps")) is not int
            or output["environment_steps"] <= 0 or output.get("effects") != ["image"]
            or snapshot.a11y_complete is not True or snapshot.image is None
            or type(receipt.receipt_id) is not str or not receipt.receipt_id
            or type(before) is not str or re.fullmatch(r"[0-9a-f]{64}", before) is None
            or after != before or snapshot.fingerprint() != before):
        return None
    return {"fingerprint": after, "source_receipt_id": receipt.receipt_id}


def _take_verified_recovery_planning_snapshot(backend, source):
    """Consume a full verified H1 response for the immediately following plan only.

    The backend owns receipt identity, input counters and the one-shot age check.
    This extra gate binds the full planning frame to the original receipt's native
    fingerprint; it never grants input or supplies goal/completion evidence.
    """
    candidate, receipt, fingerprint = source
    output = getattr(receipt, "output", None)
    take = getattr(backend, "take_verified_post_snapshot", None)
    if (
        getattr(receipt, "success", None) is not True or not callable(take)
        or type(output) is not dict or type(output.get("environment_steps")) is not int
        or output["environment_steps"] <= 0
        or type(fingerprint) is not str or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        or output.get("after_fingerprint") != fingerprint
    ):
        return None
    cached = take(candidate, receipt)
    if not isinstance(cached, tuple) or len(cached) != 2:
        return None
    snapshot, received_at = cached
    if (
        not isinstance(snapshot, AgentSnapshot) or snapshot.a11y_complete is not True
        or snapshot.image is None or snapshot.fingerprint() != fingerprint
        or type(received_at) not in {int, float} or not math.isfinite(received_at)
        or not 0 <= time.perf_counter() - received_at <= 1.0
    ):
        return None
    return snapshot, received_at


def _whole_goal_full_observation_proof(backend, snapshot, *, source=None, expected=None):
    """Pure source-bound planning proof; never input or final-completion authority."""
    query = getattr(backend, "current_full_observation_proof", None)
    if (not callable(query) or not isinstance(snapshot, AgentSnapshot)
            or snapshot.a11y_complete is not True or snapshot.image is None
            or type(snapshot.image.jpeg) is not bytes or not snapshot.image.jpeg):
        return None
    before = (getattr(backend, "environment_steps", None),
              getattr(backend, "fresh_observe_calls", None))
    if (type(before[0]) is not int or before[0] < 0
            or type(before[1]) is not int or before[1] <= 0):
        return None
    try:
        options = {"expected": expected}
        if source is not None:
            options["source"] = source
        proof = query(snapshot, **options)
        after = (getattr(backend, "environment_steps", None),
                 getattr(backend, "fresh_observe_calls", None))
        if (before != after or type(proof) is not tuple or len(proof) != 5
                or type(proof[0]) not in (int, float) or not math.isfinite(proof[0])
                or type(proof[1]) is not int or proof[1] < 0
                or type(proof[2]) is not int or proof[2] <= 0
                or (proof[1], proof[2]) != before
                or not 0 <= time.perf_counter() - proof[0] <= 1.0
                or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in proof[3:])
                or proof[3] != snapshot.fingerprint()
                or proof[4] != hashlib.sha256(snapshot.image.jpeg).hexdigest()
                or (expected is not None and (type(expected) is not tuple or proof != expected))):
            return None
        return proof
    except Exception:
        return None


@dataclass(frozen=True)
class _UnexplainedPlanningFrame:
    """One runner-local planning input, never decision/effect/goal authority."""

    snapshot: AgentSnapshot
    proof: tuple
    evidence: object
    candidate: ActionCandidate
    candidate_data: dict
    receipt: object
    receipt_data: dict
    verification_data: dict
    owner_app: str
    input_epoch: int
    context_fingerprint: str = ""


def _capture_unexplained_pending_planning_frame(backend, candidate, receipt, input_epoch):
    """Peek before the original context detector consumes its actual evidence."""
    peek = getattr(backend, "peek_input_observation_evidence", None)
    if not callable(peek) or type(input_epoch) is not int or input_epoch <= 0:
        return None
    try:
        evidence = peek(candidate, receipt)
        if (evidence is None or evidence.from_pending is not True
                or evidence.receipt is not receipt or receipt.success is not True
                or type(evidence.input_steps) is not int or evidence.input_steps <= 0
                or evidence.verification.passed is not True
                or evidence.verification.details.get("navigation") is not True
                or evidence.verification.details.get("accessibility_progress") is not True
                or type(evidence.owner_app) is not str or not evidence.owner_app.strip()
                or evidence.binding.target is None
                or evidence.binding.target.app != evidence.owner_app):
            return None
        snapshot = evidence.actual_final_after
        proof = _whole_goal_full_observation_proof(backend, snapshot)
        if (proof is None or proof[0] != evidence.received_at
                or proof[1:3] != (evidence.environment_steps, evidence.observation_count)
                or proof[3] != evidence.after_fingerprint or proof[4] != evidence.image_sha256
                or candidate.to_dict() != evidence.candidate_data
                or receipt.to_dict() != evidence.receipt_data):
            return None
        return _UnexplainedPlanningFrame(
            snapshot, proof, evidence, candidate, candidate.to_dict(), receipt, receipt.to_dict(),
            evidence.verification.to_dict(), evidence.owner_app, input_epoch,
        )
    except Exception:
        return None


def _bind_unexplained_planning_signal(frame, signal):
    """The unchanged detector must establish its original unknown-arrival signal."""
    if not isinstance(frame, _UnexplainedPlanningFrame) or type(signal) is not dict:
        return None
    fingerprint = signal.get("context_fingerprint")
    arrival = signal.get("arrival_evidence")
    if (signal.get("reason") != "unexplained_new_dialog_after_verified_navigation"
            or signal.get("input_authority") is not False or signal.get("task_success") is not False
            or signal.get("evidence_source") != "verified_pending_full"
            or signal.get("receipt_id") != frame.receipt.receipt_id
            or signal.get("candidate_id") != frame.candidate.id
            or signal.get("positive_environment_steps") != frame.receipt.output.get("environment_steps")
            or signal.get("evidence_after_fingerprint") != frame.proof[3]
            or type(fingerprint) is not str or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
            or type(arrival) is not list or not arrival
            or all(type(fact) is dict and fact.get("outcome") == "true" for fact in arrival)):
        return None
    return replace(frame, context_fingerprint=fingerprint)


def _unexplained_planning_frame_is_current(backend, frame, input_epoch):
    """Read/epoch/deep-data drift or the original one-second age declines reuse."""
    if (not isinstance(frame, _UnexplainedPlanningFrame)
            or frame.input_epoch != input_epoch or type(input_epoch) is not int
            or not frame.context_fingerprint or not frame.owner_app.strip()):
        return False
    try:
        verification = backend.verify(frame.candidate, frame.receipt)
        if (frame.receipt is not frame.evidence.receipt
                or frame.snapshot is not frame.evidence.actual_final_after
                or frame.proof[1:3] != (frame.evidence.environment_steps, frame.evidence.observation_count)
                or frame.owner_app != frame.evidence.owner_app
                or frame.receipt.success is not True
                or frame.candidate.to_dict() != frame.candidate_data
                or frame.receipt.to_dict() != frame.receipt_data
                or verification is not frame.evidence.verification
                or verification.to_dict() != frame.verification_data):
            return False
        return _whole_goal_full_observation_proof(
            backend, frame.snapshot, expected=frame.proof,
        ) is not None
    except Exception:
        return False


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "hybrid"
    max_steps: int = 30
    max_plans: int = 40
    plan_horizon: int = 6
    plan_ttl_s: float = 60
    timeout_s: float = 300
    max_failures: int = 3
    allow_actions: bool = False
    singleton_fast_path: bool = True
    adaptive_dispatch: bool = False
    reuse_verified_post_snapshot: bool = True
    defer_immediate_plan_refresh: bool = False
    reuse_verified_pending_snapshot: bool = False
    jev_network_recovery: bool = False
    stale_recovery_on_progress: bool = False
    # VLM baseline: a zero-input stale preflight rejection (page changed while the
    # model was thinking) is re-grounded by a fresh VLM step instead of being
    # charged as an action failure; bounded per verified progress.
    vlm_stale_replans: int = 0
    jev_capacity_recovery: bool = False
    whole_goal_recheck: bool = False
    jev_capacity_recovery_policy: str = "vlm_once_v1"
    reuse_whole_goal_planning_snapshot: bool = False
    inband_whole_goal_recheck: bool = False
    navigation_continuation: bool = False
    reuse_unexplained_context_planning_observation: bool = False

    def __post_init__(self):
        if type(self.adaptive_dispatch) is not bool:
            raise ValueError("adaptive_dispatch must be a boolean")
        if self.adaptive_dispatch and self.mode != "hybrid":
            raise ValueError("adaptive_dispatch requires hybrid mode")
        if type(self.defer_immediate_plan_refresh) is not bool:
            raise ValueError("defer_immediate_plan_refresh must be a boolean")
        if type(self.singleton_fast_path) is not bool:
            raise ValueError("singleton_fast_path must be a boolean")
        if type(self.reuse_verified_post_snapshot) is not bool:
            raise ValueError("reuse_verified_post_snapshot must be a boolean")
        if type(self.reuse_verified_pending_snapshot) is not bool:
            raise ValueError("reuse_verified_pending_snapshot must be a boolean")
        if type(self.jev_network_recovery) is not bool:
            raise ValueError("jev_network_recovery must be a boolean")
        if type(self.jev_capacity_recovery) is not bool:
            raise ValueError("jev_capacity_recovery must be a boolean")
        if (type(self.jev_capacity_recovery_policy) is not str
                or self.jev_capacity_recovery_policy not in {"vlm_once_v1", "vlm_phase_v2", "vlm_phase_v3"}):
            raise ValueError("jev_capacity_recovery_policy must be vlm_once_v1, vlm_phase_v2 or vlm_phase_v3")
        for field in ("reuse_whole_goal_planning_snapshot", "inband_whole_goal_recheck",
                      "navigation_continuation", "reuse_unexplained_context_planning_observation"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be a boolean")
        if type(self.whole_goal_recheck) is not bool:
            raise ValueError("whole_goal_recheck must be a boolean")
        if type(self.stale_recovery_on_progress) is not bool:
            raise ValueError("stale_recovery_on_progress must be a boolean")
        if self.mode not in {"hybrid", "vlm"}:
            raise ValueError("mode must be hybrid or vlm")
        if (
            not all(
                type(n) is int and n > 0
                for n in (self.max_steps, self.max_plans, self.plan_horizon, self.max_failures)
            )
            or self.plan_horizon > 12
        ):
            raise ValueError("invalid agent step/plan budgets")
        if not 0 < self.plan_ttl_s <= 300 or not 0 < self.timeout_s <= 3600:
            raise ValueError("invalid agent time budgets")


@dataclass(frozen=True)
class AgentResult:
    status: str
    reason: str
    steps: int
    duration_ms: float
    planner_calls: int
    planner_wall_ms: float
    jev_calls: int
    jev_wall_ms: float
    backend_wall_ms: float
    action_wall_ms: float
    effect_check_wall_ms: float
    observation_wall_ms: float
    acceptance_wall_ms: float
    plan_reuses: int
    replan_reasons: dict[str, int]
    routes: dict[str, int]
    model_usage: dict
    candidate_counts: tuple[int, ...] = ()
    multi_candidate_steps: int = 0
    semantic_steps: int = 0
    condition_skips: int = 0
    recovery_observations: int = 0
    navigation_steps: int = 0
    navigation_subgoals: int = 0
    plan_completion_claims: int = 0
    singleton_decisions: int = 0
    post_snapshot_reuses: int = 0
    runner_observe_calls: int = 0
    fresh_observe_calls: int | None = None
    distinct_target_counts: tuple[int | None, ...] = ()
    context_recoveries: int = 0
    jev_request_attempts: int | None = None
    jev_network_failures: int = 0
    network_recovery_attempts: int = 0
    network_recovery_actions: int = 0
    network_recovery_verified: int = 0
    property_edits: int = 0
    property_confirmations: int = 0
    property_subgoals: int = 0
    stale_recovery_attempts: int = 0
    stale_snapshot_reuses: int = 0
    recovery_planning_snapshot_reuses: int = 0
    capacity_recovery_enabled: bool = False
    jev_capacity_failures: int = 0
    capacity_recovery_attempts: int = 0
    capacity_recovery_actions: int = 0
    capacity_recovery_verified: int = 0
    verified_pending_snapshot_reuses: int = 0
    whole_goal_recheck_calls: int = 0
    whole_goal_recheck_continues: int = 0
    whole_goal_recheck_done: int = 0
    whole_goal_planning_snapshot_reuses: int = 0
    unexplained_navigation_context_replans: int = 0
    whole_goal_inband_review_calls: int = 0
    whole_goal_inband_review_done: int = 0
    post_plan_observations_deferred: int = 0
    equivalent_rule_decisions: int = 0
    dispatch_refreshes: int = 0
    dispatch_s2_replans: int = 0

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict:
        result = {**asdict(self), "success": self.success}
        if not self.capacity_recovery_enabled:
            for name in ("capacity_recovery_enabled", "jev_capacity_failures",
                         "capacity_recovery_attempts", "capacity_recovery_actions",
                         "capacity_recovery_verified"):
                result.pop(name)
        if not self.verified_pending_snapshot_reuses:
            result.pop("verified_pending_snapshot_reuses")
        for name in ("whole_goal_recheck_calls", "whole_goal_recheck_continues",
                     "whole_goal_recheck_done", "whole_goal_planning_snapshot_reuses",
                     "unexplained_navigation_context_replans",
                     "whole_goal_inband_review_calls", "whole_goal_inband_review_done"):
            if not result[name]:
                result.pop(name)
        return result


class DirectPlanPolicy:
    """One fresh VLM action per step, same backend and deterministic route order."""

    def choose(self, observation: Observation, candidates: Sequence[ActionCandidate]) -> Decision:
        selected = candidates[0]
        return Decision(
            observation.observation_id,
            selected.id,
            {c.id: float(c.id == selected.id) for c in candidates},
            1.0,
            "vlm-direct",
            0.0,
        )


class _NetworkRecoveryPolicy:
    def choose(self, observation, candidates):
        if len(candidates) != 1:
            raise PolicyError("network recovery requires one current candidate")
        selected = candidates[0]
        return Decision(observation.observation_id, selected.id, {selected.id: 1.0},
                        1.0, "vlm-network-recovery", 0.0)



def _is_jev_capacity_failure(failure) -> bool:
    """A typed current-call capacity refusal, never a network retry permission."""
    return (
        isinstance(failure, JevFailure)
        and failure.code == "http_status"
        and type(failure.http_status) is int and failure.http_status == 400
        and failure.service_code == "max_tokens_exceeded"
        and failure.retryable is False and failure.network_retryable is False
    )


class _CapacityRecoveryPolicy:
    def choose(self, observation, candidates):
        if len(candidates) != 1:
            raise PolicyError("capacity recovery requires one current candidate")
        selected = candidates[0]
        return Decision(observation.observation_id, selected.id, {selected.id: 1.0},
                        1.0, "vlm-capacity-recovery", 0.0)

def _distinct_bound_target_count(
    candidates: Sequence[ActionCandidate], snapshot: AgentSnapshot,
) -> int | None:
    """Count current bound UI targets, independent of routes and mouse gestures.

    A keyboard/visual action, an unsupported backend, or incomplete binding
    metadata makes this decision's target count unknown, never zero. Names and
    semantic identities alone cannot distinguish two separate current controls.
    """
    if not candidates:
        return None
    controls = {control.ref: control.to_dict() for control in snapshot.controls}
    if len(controls) != len(snapshot.controls):
        return None
    refs = set()
    for candidate in candidates:
        bound = candidate.arguments.get("bound_control")
        if not isinstance(bound, dict):
            return None
        ref = bound.get("ref")
        if not isinstance(ref, str) or not ref or controls.get(ref) != bound:
            return None
        refs.add(ref)
    return len(refs)


class _MeteredPolicy:
    def __init__(self, inner, *, direct_singletons=False):
        self.inner = inner
        self.calls = 0
        self.wall_ms = 0.0
        self.direct_singletons = direct_singletons
        self.singleton_decisions = 0
        self._last_singleton = False
        self.request_attempts = 0 if isinstance(inner, JevPolicy) else None

    @property
    def last_exchange(self):
        return {} if self._last_singleton else getattr(self.inner, "last_exchange", {})

    def choose(self, observation, candidates):
        self._last_singleton = (self.direct_singletons and len(candidates) == 1
                                and observation.state.get("explicit_property_confirmation") is not True)
        if self._last_singleton:
            # Selection is deterministic when the legal set contains one item.
            # Execution still passes through the same guard and fresh preflight.
            self.singleton_decisions += 1
            candidate = candidates[0]
            return Decision(observation.observation_id, candidate.id, {candidate.id: 1.0},
                            1.0, "deterministic-singleton", 0.0)
        self.calls += 1
        started = time.perf_counter()
        try:
            return self.inner.choose(observation, candidates)
        finally:
            self.wall_ms += (time.perf_counter() - started) * 1000
            exchange = getattr(self.inner, "last_exchange", {})
            if self.request_attempts is not None and isinstance(exchange, dict):
                request = exchange.get("request", {})
                state = request.get("state", {}) if isinstance(request, dict) else {}
                attempts = exchange.get("attempts")
                if (isinstance(state, dict) and state.get("observation_id") == observation.observation_id
                        and type(attempts) is int and attempts > 0):
                    self.request_attempts += attempts


def _control_values(snapshot: AgentSnapshot) -> dict[tuple, list[tuple[str, bool]]]:
    values: dict[tuple, list[tuple[str, bool]]] = {}
    for control in snapshot.controls:
        values.setdefault(control.identity(), []).append((control.value, control.checked))
    return {identity: sorted(states) for identity, states in values.items()}


def _direct_target_satisfied(step: AgentStep, snapshot: AgentSnapshot) -> bool:
    """Skipping typed edits must prove their effect on the intended target."""
    kind = {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}.get(step.operation)
    if kind is None:
        return True
    return step.target is not None and condition_holds(
        AgentCondition(kind, step.target, step.value if step.operation == "fill" else ""), snapshot,
    )


def _condition_facts(conditions: tuple[AgentCondition, ...], snapshot: AgentSnapshot) -> list[dict]:
    """Expose predicate outcomes without sending their private comparison values."""
    return [
        {"kind": item.kind, "target": asdict(item.target) if item.target else None,
         "satisfied": condition_holds(item, snapshot), **condition_evidence(item, snapshot)}
        for item in conditions
    ]


def _unconfirmed_application_wait(details: dict) -> bool:
    """Shell progress cannot release a launcher still awaiting its application."""
    launch = details.get("application_wait")
    if not isinstance(launch, dict) or not launch or details.get("postconditions_met") is True:
        return False
    # A receipt-authorized active-app handoff retains ordinary navigation
    # progress semantics; it does not turn an unmatched goal predicate true.
    handoff = (
        details.get("navigation") is True and details.get("accessibility_progress") is True
        and isinstance(details.get("navigation_app"), str) and bool(details["navigation_app"].strip())
    )
    return launch.get("eligible") is True or not handoff


def _postcondition_feedback(step: AgentStep, details: dict) -> list[dict]:
    """Use only matching, typed post-action evidence; never expose comparison values."""
    facts = [
        {"kind": item.kind, "target": asdict(item.target) if item.target else None}
        for item in step.postconditions
    ]
    evidence = details.get("postcondition_evidence")
    if not isinstance(evidence, list) or len(evidence) != len(facts):
        return facts
    for fact, observed in zip(facts, evidence, strict=True):
        if (
            isinstance(observed, dict)
            and observed.get("kind") == fact["kind"] and observed.get("target") == fact["target"]
            and observed.get("outcome") in {"true", "false", "unknown"}
            and observed.get("reason") in {
                "condition_met", "state_mismatch", "target_missing", "ambiguous_target",
                "unsupported_state", "incomplete_observation", "invalid_condition",
            }
        ):
            fact.update(outcome=observed["outcome"], reason=observed["reason"],
                        satisfied=observed["outcome"] == "true")
    return facts


def _unavailable_step_facts(step: AgentStep, snapshot: AgentSnapshot) -> dict:
    if step.property_confirmation_of:
        return {
            "step_id": step.id, "operation": "click", "candidate_count": 0,
            "reason": "confirmation_context_unbound", "source_step": step.property_confirmation_of,
            "feedback": (
                "No confirmation input is authorized: the verified field or current owning-form "
                "evidence is unavailable. Inspect the live field and form; explicitly choose a current "
                "guarded action without inferring a button or key."
            ),
        }
    conditions = _condition_facts(step.preconditions, snapshot)
    matches = (
        matching_controls(snapshot, step.target, allow_scope=step.navigation) if step.target else ()
    )
    return {
        "step_id": step.id,
        "operation": step.operation,
        "target": asdict(step.target) if step.target else None,
        "reason": "preconditions_unmet" if not all(c["satisfied"] for c in conditions)
        else "no_executable_target",
        "preconditions": conditions,
        "match_count": len(matches),
        "candidate_count": 0,
        "matches": [
            {"ref": c.ref, "tag": c.tag, "role": c.role, "enabled": not c.disabled,
             "has_box": c.box is not None, "editable": c.editable}
            for c in matches[:8]
        ],
    }


class _EpisodeExpired(RuntimeError):
    pass


class _PlanExpired(RuntimeError):
    pass


class AgentRunner:
    def __init__(
        self,
        backend: AgentBackend,
        planner: AgentPlanner,
        *,
        policy: DecisionPolicy | None = None,
        config: AgentConfig | None = None,
        trace: JsonlTrace | None = None,
        progress: Callable[[dict], None] | None = None,
    ):
        self.backend, self.planner = backend, planner
        self.config = config or AgentConfig()
        if self.config.mode == "hybrid" and policy is None:
            raise ValueError("hybrid mode requires an explicit decision policy")
        self.policy = policy if self.config.mode == "hybrid" else DirectPlanPolicy()
        self.trace = trace or JsonlTrace()
        self.progress = progress

    def run(self, goal: str, *, success: Callable[[AgentSnapshot], bool] | None = None,
            success_uses_live_state: bool = False) -> AgentResult:
        # Only callers whose evaluator ignores the supplied frame may opt in.
        if type(success_uses_live_state) is not bool or (success_uses_live_state and success is None):
            raise ValueError("success_uses_live_state requires an independent live evaluator")
        if not goal.strip():
            raise ValueError("goal is required")
        cfg = self.config
        stage_program_enabled = (
            cfg.mode == "hybrid" and cfg.whole_goal_recheck
            and getattr(self.planner, "profile", None) == "desktop"
            and getattr(self.planner, "desktop_contract", None) == "property_intent_v52"
            and getattr(self.planner, "desktop_output_format", None) in {
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}
            and callable(getattr(self.planner, "activate_program_stage", None))
        )
        navigation_arrival_diagnostics_enabled = (
            cfg.mode == "hybrid" and getattr(self.planner, "profile", None) == "desktop"
            and getattr(self.planner, "desktop_contract", None) == "property_intent_v52"
            and getattr(self.planner, "desktop_output_format", None) in {
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}
        )
        navigation_enabled = cfg.navigation_continuation and cfg.mode == "hybrid"
        navigation_memory = None
        if navigation_enabled:
            if (getattr(self.planner, "profile", None) != "desktop"
                    or getattr(self.planner, "desktop_output_format", None) not in {
                        "semantic_tools_v57", "semantic_tools_v58",
                        "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}
                    or getattr(self.planner, "desktop_contract", None) != "property_intent_v52"
                    or not all(callable(getattr(self.backend, name, None)) for name in (
                        "navigation_memory_state", "navigation_transition_evidence",
                        "clear_navigation_memory_sources", "set_input_observation_evidence_enabled"))):
                raise ValueError("navigation continuation requires explicit57 and bound desktop evidence")
            from .desktop_navigation_memory import NavigationMemory, NavigationTransition

            navigation_memory = NavigationMemory(cfg.plan_horizon)
        start = time.perf_counter()

        def admit_navigation(observation, decision, candidate):
            rejection = navigation_memory.decline(observation.observation_id, candidate.id)
            if rejection is None:
                return None
            return DecisionAdmissionRejection(rejection.stage, rejection.state_fingerprint,
                                              rejection.edge_key)

        def execute_before_deadline(candidate, observation_id, decision_id):
            # A slow Jev response must not start an action after the episode budget.
            if time.perf_counter() - start >= cfg.timeout_s:
                raise _EpisodeExpired()
            if plan is not None and plan.reusable and time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                raise _PlanExpired()
            execute_with_deadline = getattr(self.backend, "execute_with_deadline", None)
            if callable(execute_with_deadline):
                deadline = start + cfg.timeout_s
                if plan is not None and plan.reusable:
                    deadline = min(deadline, plan_at + cfg.plan_ttl_s)
                return execute_with_deadline(
                    candidate, observation_id, decision_id, deadline=deadline,
                )
            return self.backend.execute(candidate, observation_id, decision_id)

        executors = ExecutorRegistry()
        for channel in (Channel.SCRIPT, Channel.GUI, Channel.API, Channel.MCP, Channel.CONTROL):
            executors.register(channel, execute_before_deadline)
        verifiers = VerifierRegistry()
        verifiers.register("agent.effect", self.backend.verify)
        metered = _MeteredPolicy(
            self.policy, direct_singletons=(cfg.mode == "hybrid" and cfg.singleton_fast_path
                                           and not cfg.adaptive_dispatch),
        )
        dispatch = (DispatchPolicy(PublicDecisionPolicy(metered),
                                   selection_kind=getattr(self.policy, "selection_kind", "jev"))
                    if cfg.adaptive_dispatch else None)
        dispatch_refresh_attempted = False
        dispatch_refreshes = dispatch_s2_replans = 0
        dispatch_failure_streak = 0
        runtime = AgentRuntime(
            policy=dispatch or PublicDecisionPolicy(metered),
            guard=ActionGuard(allow_writes=cfg.allow_actions, allow_destructive=cfg.allow_actions),
            executors=executors,
            verifiers=verifiers,
            trace=self.trace,
            decision_admission=admit_navigation if navigation_memory is not None else None,
        )
        recovery_runtime = AgentRuntime(
            policy=_NetworkRecoveryPolicy(), guard=runtime.guard,
            executors=executors, verifiers=verifiers, trace=self.trace,
        )
        capacity_runtime = (
            AgentRuntime(
                policy=_CapacityRecoveryPolicy(), guard=runtime.guard,
                executors=executors, verifiers=verifiers, trace=self.trace,
            ) if cfg.jev_capacity_recovery else None
        )
        initial_usage = dict(getattr(self.planner, "usage_totals", {}))
        counts: Counter = Counter()
        initial_observe_calls = getattr(self.backend, "fresh_observe_calls", None)

        inband_last_fresh = None
        last_runner_fresh_snapshot = None
        last_runner_fresh_proof = None

        def inband_frame_proof(current, origin, source=None):
            if not (cfg.whole_goal_recheck and cfg.inband_whole_goal_recheck):
                return None
            proof = _whole_goal_full_observation_proof(self.backend, current, source=source)
            return (current, proof, origin, source) if proof is not None else None

        def fresh_observe(*, completion_proof=False):
            nonlocal inband_last_fresh, last_runner_fresh_snapshot, last_runner_fresh_proof
            inband_last_fresh = None
            last_runner_fresh_snapshot = last_runner_fresh_proof = None
            counts["runner_observe_calls"] += 1
            observed = self.backend.observe()
            # Capture once on real return, before any progress/trace callback.
            last_runner_fresh_snapshot = observed
            if (cfg.whole_goal_recheck and (cfg.inband_whole_goal_recheck
                    or (completion_proof and cfg.reuse_whole_goal_planning_snapshot
                        and cfg.reuse_verified_post_snapshot))):
                proof = _whole_goal_full_observation_proof(self.backend, observed)
                if proof is not None:
                    if cfg.inband_whole_goal_recheck:
                        inband_last_fresh = (observed, proof, "backend.observe", None)
                    if (completion_proof and cfg.reuse_whole_goal_planning_snapshot
                            and cfg.reuse_verified_post_snapshot):
                        last_runner_fresh_proof = proof
            return observed

        reasons: Counter = Counter()
        routes: Counter = Counter()
        history: list[dict] = []
        plan: AgentPlan | None = None
        active_program = None
        program_stage_index = 0
        program_stage_admitted = False
        program_origin_call = None
        program_id = None
        program_receipts: list[tuple[str, int]] = []
        program_attempt_receipts: list[dict] = []
        program_verified_effects: dict[str, AgentStep] = {}
        program_handoff_pending = False
        high_level_context_fresh_pending = False
        planner_timeout_fresh_pending = False
        program_frame_origin = "unavailable"
        program_frame_source = None
        planned: AgentSnapshot | None = None
        completed: set[str] = set()
        property_effects: dict[str, tuple[AgentStep, str]] = {}
        completed_conditions: dict[tuple, tuple[str, bool]] = {}
        expected_values: dict[tuple, list[tuple[str, bool]]] = {}
        plan_at = 0.0
        plan_actions = 0
        # Planning errors and unsuccessful actions have independent streaks.
        # Action failures include unavailable targets and unconfirmed completion:
        # accepting another syntactically valid plan is not execution progress.
        planning_failures = 0
        action_failures = 0
        pending_reason = "initial"
        recovery_observations = 0
        semantic_recovery = False
        invalid_plan_recovery = False
        recovery_plan = False
        pending_prepare_repair = None
        active_prepare_repair = None
        prepare_repair_used = False
        navigation_seen: dict[str, set[str]] = {}
        candidate_counts: list[int] = []
        distinct_target_counts: list[int | None] = []
        post_snapshot_source = None
        recovery_planning_snapshot_source = None
        unexplained_planning_pending = unexplained_planning_source = None
        pending_numeric_planning_source = None
        pending_numeric_finalize_source = None
        stale_snapshot_source = None
        context_recovery_pending = None
        context_recovery_active = False
        context_recovery_planned = False
        network_recovery_active = False
        network_recovery_planned = False
        capacity_recovery_active = False
        capacity_recovery_planned = False
        capacity_recovery_fresh_required = False
        selection_recovery_ticket_used = False
        capacity_phase = cfg.jev_capacity_recovery_policy in {"vlm_phase_v2", "vlm_phase_v3"}
        capacity_phase_v3 = cfg.jev_capacity_recovery_policy == "vlm_phase_v3"
        stale_recovery_used = False
        vlm_stale_replans = 0
        whole_goal_input_epoch = 0
        unexplained_planning_input_epoch = 0
        whole_goal_attempted_epoch = None
        whole_goal_recheck_pending = False
        whole_goal_pending_frame = None
        whole_goal_reviewed_done_plan = None
        whole_goal_reviewed_done_epoch = None
        whole_goal_inband_done = None
        unexplained_context_recovery_active = False
        unexplained_context_enabled = (
            cfg.mode == "hybrid" and getattr(self.planner, "profile", None) == "desktop"
            and getattr(self.planner, "desktop_output_format", None) in {
                "semantic_tools_v56", "semantic_tools_v57", "semantic_tools_v58",
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}
            and getattr(self.planner, "desktop_contract", None) == "property_intent_v52"
            and callable(getattr(self.backend, "take_unexplained_navigation_context", None))
        )
        unexplained_planning_reuse_enabled = (
            cfg.reuse_unexplained_context_planning_observation and cfg.reuse_verified_post_snapshot
            and unexplained_context_enabled
            and getattr(self.planner, "desktop_output_format", None) == "semantic_tools_v64"
            and getattr(self.planner, "desktop_planning_observation", None) == "full_multimodal_v1"
        )
        v52_request_trace = (
            getattr(self.planner, "profile", None) == "desktop"
            and getattr(self.planner, "desktop_contract", None) == "property_intent_v52"
        )
        v52_information_routing = (
            v52_request_trace and cfg.mode == "hybrid"
            and getattr(self.planner, "desktop_planning_observation", None) == "native_first_v51"
        )
        visual_information_anchor = None
        plan_request_information = None
        status, reason = "max_steps", "action budget exhausted"

        def emit(kind: str, payload: dict):
            self.trace.append(kind, payload)
            if self.progress:
                self.progress({"event": kind, **payload})

        if dispatch is not None:
            dispatch.emit = emit

        def clear_navigation_sources():
            # Preserve accepted plan/planned baseline, plan_at, completed effects,
            # failure counts and every used ticket. Only discard cached authority.
            nonlocal post_snapshot_source, recovery_planning_snapshot_source, stale_snapshot_source
            nonlocal pending_numeric_planning_source, pending_numeric_finalize_source
            nonlocal pending_prepare_repair, active_prepare_repair, context_recovery_pending
            nonlocal visual_information_anchor
            nonlocal whole_goal_pending_frame, whole_goal_inband_done, inband_last_fresh
            nonlocal unexplained_planning_pending, unexplained_planning_source
            nonlocal last_runner_fresh_snapshot, last_runner_fresh_proof
            nonlocal whole_goal_source, whole_goal_planning_frame, whole_goal_reused_proof
            nonlocal inband_current_frame, inband_offer, inband_marked, inband_epoch_dispatched
            post_snapshot_source = recovery_planning_snapshot_source = stale_snapshot_source = None
            unexplained_planning_pending = unexplained_planning_source = None
            pending_numeric_planning_source = pending_numeric_finalize_source = None
            pending_prepare_repair = active_prepare_repair = context_recovery_pending = None
            visual_information_anchor = None
            whole_goal_pending_frame = whole_goal_inband_done = None
            inband_last_fresh = None
            last_runner_fresh_snapshot = last_runner_fresh_proof = None
            whole_goal_source = whole_goal_planning_frame = whole_goal_reused_proof = None
            inband_current_frame = inband_offer = None
            inband_marked = False
            inband_epoch_dispatched = None
            clear_backend_navigation = getattr(self.backend, "clear_navigation_memory_sources", None)
            if callable(clear_backend_navigation):
                clear_backend_navigation()
            clear_repair = getattr(self.planner, "set_prepare_repair_context", None)
            if callable(clear_repair):
                clear_repair(None)


        def retry_capacity_plan(why, *, accepted_plan=False, goal_review=False):
            # A new request is not same-plan reuse. No ticket, failure streak,
            # plan clock, action/plan budget or episode deadline is renewed here.
            nonlocal plan, planned, pending_reason, semantic_recovery
            nonlocal capacity_recovery_planned, capacity_recovery_fresh_required
            nonlocal invalid_plan_recovery, recovery_plan, whole_goal_recheck_pending
            nonlocal post_snapshot_source, recovery_planning_snapshot_source, stale_snapshot_source
            nonlocal pending_numeric_planning_source, pending_numeric_finalize_source
            nonlocal pending_prepare_repair, active_prepare_repair, context_recovery_pending
            nonlocal visual_information_anchor, plan_request_information
            nonlocal source, recovery_planning_source, stale_source, numeric_source, finalize_source, cached
            nonlocal numeric_planning_frame, numeric_finalize_frame, recovery_planning_frame
            nonlocal whole_goal_pending_frame, whole_goal_inband_done, inband_last_fresh
            nonlocal last_runner_fresh_snapshot, last_runner_fresh_proof
            nonlocal whole_goal_planning_frame, whole_goal_reused_proof
            nonlocal inband_current_frame, inband_offer, inband_marked, inband_epoch_dispatched
            if not (capacity_phase_v3 and capacity_recovery_active
                    and planning_failures < cfg.max_failures and action_failures < cfg.max_failures):
                return False
            if accepted_plan and time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                return False
            capacity_recovery_planned = False
            capacity_recovery_fresh_required = True
            semantic_recovery = True
            invalid_plan_recovery = recovery_plan = False
            plan = planned = None
            completed.clear()
            property_effects.clear()
            completed_conditions.clear()
            expected_values.clear()
            navigation_seen.clear()
            post_snapshot_source = recovery_planning_snapshot_source = stale_snapshot_source = None
            pending_numeric_planning_source = pending_numeric_finalize_source = None
            pending_prepare_repair = active_prepare_repair = context_recovery_pending = None
            visual_information_anchor = plan_request_information = None
            source = recovery_planning_source = stale_source = None
            numeric_source = finalize_source = cached = None
            numeric_planning_frame = numeric_finalize_frame = recovery_planning_frame = None
            whole_goal_pending_frame = whole_goal_inband_done = inband_last_fresh = None
            last_runner_fresh_snapshot = last_runner_fresh_proof = None
            whole_goal_planning_frame = whole_goal_reused_proof = None
            inband_current_frame = inband_offer = inband_epoch_dispatched = None
            inband_marked = False
            if goal_review:
                whole_goal_recheck_pending = True
            clear_evidence = getattr(self.backend, "clear_input_observation_evidence", None)
            if callable(clear_evidence):
                clear_evidence()
            clear_repair = getattr(self.planner, "set_prepare_repair_context", None)
            if callable(clear_repair):
                clear_repair(None)
            pending_reason = "capacity_recovery_invalid_plan"
            feedback = {"reason": pending_reason, "cause": why, "recovery_required": True,
                        "input_sent": False, "verified": False, "task_success": False,
                        "feedback": "The recovery plan sent no input and is unusable. Inspect a new "
                                    "fresh observation and return one uniquely grounded immediate action "
                                    "or a guarded completion. Original failure and episode budgets remain."}
            history.append(feedback)
            emit("agent_capacity_recovery", {
                **feedback, "stage": "invalid_plan_retry_requested", "policy": "vlm_phase_v3",
                "planning_failures": planning_failures, "action_failures": action_failures,
                "shared_ticket_used": selection_recovery_ticket_used,
                "accepted_plan_retired": accepted_plan, "previous_plan_at_unchanged": plan_at,
            })
            return True

        def whole_goal_stop_gate(*, completion_frame=None):
            nonlocal plan, planned, pending_reason, whole_goal_recheck_pending
            nonlocal post_snapshot_source, recovery_planning_snapshot_source, stale_snapshot_source
            nonlocal pending_numeric_planning_source, pending_numeric_finalize_source
            nonlocal pending_prepare_repair, active_prepare_repair, visual_information_anchor
            nonlocal plan_request_information, context_recovery_pending
            nonlocal semantic_recovery, invalid_plan_recovery, recovery_plan
            nonlocal network_recovery_active, capacity_recovery_active, context_recovery_active
            nonlocal capacity_recovery_planned, capacity_recovery_fresh_required
            nonlocal unexplained_context_recovery_active, whole_goal_pending_frame
            if not cfg.whole_goal_recheck:
                return "stop"
            if time.perf_counter() - start >= cfg.timeout_s:
                raise _EpisodeExpired()
            if (cfg.inband_whole_goal_recheck and whole_goal_inband_done is not None
                    and plan is whole_goal_inband_done[0]
                    and whole_goal_input_epoch == whole_goal_inband_done[1]
                    and type(getattr(self.backend, "environment_steps", None)) is int
                    and self.backend.environment_steps == whole_goal_inband_done[2]
                    and type(getattr(self.backend, "fresh_observe_calls", None)) is int
                    and self.backend.fresh_observe_calls == whole_goal_inband_done[4] + 1
                    and plan.done and not plan.steps and plan.completion_conditions):
                counts["whole_goal_inband_review_done"] += 1
                emit("agent_whole_goal_inband_review", {
                    "stage": "empty_done_fresh_checked", "input_epoch": whole_goal_input_epoch,
                    "call": whole_goal_inband_done[3], "task_success": False,
                })
                return "stop"
            if (plan is whole_goal_reviewed_done_plan
                    and whole_goal_reviewed_done_epoch == whole_goal_input_epoch):
                counts["whole_goal_recheck_done"] += 1
                emit("agent_whole_goal_recheck", {
                    "stage": "empty_done_fresh_checked", "input_epoch": whole_goal_input_epoch,
                    "task_success": False,
                })
                return "stop"
            if whole_goal_attempted_epoch == whole_goal_input_epoch:
                return "blocked"
            whole_goal_recheck_pending = True
            whole_goal_pending_frame = None
            if (cfg.reuse_whole_goal_planning_snapshot and cfg.reuse_verified_post_snapshot
                    and completion_frame is not None
                    and completion_frame is last_runner_fresh_snapshot
                    and last_runner_fresh_proof is not None):
                proof = _whole_goal_full_observation_proof(
                    self.backend, completion_frame, expected=last_runner_fresh_proof,
                )
                if proof is not None:
                    whole_goal_pending_frame = (completion_frame, proof, whole_goal_input_epoch)
            emit("agent_whole_goal_recheck", {
                "stage": "required", "input_epoch": whole_goal_input_epoch,
                "conditions": _condition_facts(plan.completion_conditions, snapshot),
                "task_success": False,
            })
            plan = planned = None
            completed.clear()
            property_effects.clear()
            completed_conditions.clear()
            expected_values.clear()
            navigation_seen.clear()
            post_snapshot_source = recovery_planning_snapshot_source = stale_snapshot_source = None
            pending_numeric_planning_source = pending_numeric_finalize_source = None
            pending_prepare_repair = active_prepare_repair = None
            visual_information_anchor = plan_request_information = context_recovery_pending = None
            capacity_review = capacity_phase_v3 and capacity_recovery_active
            # End the completed claim's phase, never rearm its spent recovery ticket or counters.
            semantic_recovery = invalid_plan_recovery = recovery_plan = False
            network_recovery_active = capacity_recovery_active = context_recovery_active = False
            if capacity_review:
                # Review is still H1 until verified positive input. Checked empty
                # done remains legal and never fabricates a recovery input/credit.
                capacity_recovery_active = semantic_recovery = True
                capacity_recovery_planned = False
                capacity_recovery_fresh_required = True
                whole_goal_pending_frame = None
            unexplained_context_recovery_active = False
            clear_evidence = getattr(self.backend, "clear_input_observation_evidence", None)
            if callable(clear_evidence):
                clear_evidence()
            clear_repair = getattr(self.planner, "set_prepare_repair_context", None)
            if callable(clear_repair):
                clear_repair(None)
            pending_reason = "whole_goal_recheck"
            history.append({"reason": pending_reason, "recovery_required": False,
                            "feedback": "Reassess every requirement of the original complete user "
                                        "goal; a completed local stage does not establish it."})
            return "review"

        def program_trace_fields():
            return {
                "program_id": program_id, "origin_planner_call": program_origin_call,
                "stage_index": program_stage_index,
                "stage_id": active_program.stages[program_stage_index].id,
                "original_plan_at": plan_at, "plan_actions": plan_actions,
                "remaining_actions": max(0, active_program.max_actions - plan_actions),
                "logical_attempt_budget": active_program.max_actions,
                "input_authority": False, "task_success": False,
                "legacy_strict_analyzer_semantically_sufficient": False,
            }

        def abandon_program(why, *, force_recovery=True):
            # Retire declarations, never refund a receipt/attempt or rearm a ticket.
            # The existing failure branch still owns its original accounting.
            nonlocal active_program, program_handoff_pending, semantic_recovery
            if active_program is None:
                return
            emit("agent_program_abandoned", {
                **program_trace_fields(), "reason": why,
                "receipt_ids": [item[0] for item in program_receipts],
                "attempt_receipts": list(program_attempt_receipts),
                "whole_program_retired": True,
                "force_semantic_recovery": force_recovery,
                "semantic_recovery_before": semantic_recovery,
                "semantic_recovery_after": bool(semantic_recovery or force_recovery),
                "program_handoff_pending_before": program_handoff_pending,
                "program_handoff_pending_after": bool(program_handoff_pending or force_recovery),
                "preserve_original_handoff_flags": not force_recovery,
                "budget_refunded": False, "old_program_budget_rearmed": False,
            })
            active_program = None
            if force_recovery:
                program_handoff_pending = True
                semantic_recovery = True

        def retire_program_budget(why):
            # Normal retirement owns no new recovery permission. The original
            # plan=None path still promotes an expiring numeric key continuation
            # to H1 when required; clean TTL/horizon retirement may request H6.
            nonlocal plan, pending_reason
            plan = None
            pending_reason = why
            abandon_program(why, force_recovery=False)
            return "retired"

        def admit_initial_program_stage(current):
            """One current-frame admission, without recompiling request-bound steps.

            Entry conditions do not become continuous input authority. The
            original current action/identity/preflight guards remain unchanged.
            """
            nonlocal program_stage_admitted
            if program_stage_admitted:
                return True
            if (active_program is None or program_stage_index != 0
                    or program_frame_origin not in {
                        "backend.observe", "verified_step_response",
                        "verified_pending_effect_observation",
                        "verified_ordinary_pending_effect_observation",
                        "stale_rejection_preflight"}):
                abandon_program("initial_stage_current_admission_frame_unavailable")
                return False
            proof = _whole_goal_full_observation_proof(
                self.backend, current, source=program_frame_source,
            )
            if proof is None:
                abandon_program("initial_stage_current_admission_proof_unavailable")
                return False
            entry_facts = _condition_facts(active_program.stages[0].entry_conditions, current)
            if (any(item.get("outcome") != "true" for item in entry_facts)
                    or _whole_goal_full_observation_proof(
                        self.backend, current, source=program_frame_source, expected=proof,
                    ) is None):
                emit("agent_program_stage_admission_rejected", {
                    **program_trace_fields(), "entry_conditions": entry_facts,
                    "source": program_frame_origin, "snapshot_fingerprint": proof[3],
                    "image_sha256": proof[4], "environment_steps": proof[1],
                    "fresh_observe_calls": proof[2], "input_sent": False,
                })
                abandon_program("initial_stage_current_entry_not_true")
                return False
            emit("agent_program_stage_admitted", {
                **program_trace_fields(), "admitted": True,
                "entry_conditions": entry_facts, "source": program_frame_origin,
                "snapshot_fingerprint": proof[3], "image_sha256": proof[4],
                "environment_steps": proof[1], "fresh_observe_calls": proof[2],
                "one_shot_entry_admission": True,
                "original_action_preconditions_and_preflight_still_required": True,
                "initial_plan_not_recompiled": True,
                "planned_request_binding_unchanged": True,
            })
            if _whole_goal_full_observation_proof(
                    self.backend, current, source=program_frame_source, expected=proof,
            ) is None:
                abandon_program("initial_stage_admission_frame_changed_after_trace")
                return False
            program_stage_admitted = True
            return True

        def program_boundary(current):
            """Local declarations have no execution/observation/retry capability.

            The only boundary frame is a current source-bound full frame already
            obtained by the original loop. An SDK-after is not called a fresh
            backend.observe. Future entry conditions never prove current exit.
            """
            nonlocal active_program, program_stage_index, plan, planned, pending_reason
            nonlocal program_stage_admitted
            nonlocal completed, property_effects, completed_conditions, expected_values
            nonlocal navigation_seen, program_receipts, program_verified_effects
            nonlocal program_attempt_receipts
            nonlocal post_snapshot_source, recovery_planning_snapshot_source, stale_snapshot_source
            nonlocal pending_numeric_planning_source, pending_numeric_finalize_source
            nonlocal pending_prepare_repair, active_prepare_repair
            if active_program is None or plan is None:
                return "idle"
            if time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                return retire_program_budget("plan_expired")
            if not program_stage_admitted and not admit_initial_program_stage(current):
                return "abandoned"
            while active_program is not None:
                # Do not activate from genuine recovery, numeric partial frames,
                # stale rejection or a context which the old runner invalidated.
                if (semantic_recovery or recovery_plan or context_recovery_active
                        or network_recovery_active or capacity_recovery_active
                        or pending_prepare_repair is not None or active_prepare_repair is not None
                        or program_handoff_pending):
                    abandon_program("original_recovery_required")
                    return "abandoned"
                if time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                    return retire_program_budget("plan_expired")
                # Stage-local skips use exactly the original predicate/target
                # requirements, in dependency order. Commit only a complete set.
                trial = set(completed)
                skips = []
                for item in plan.steps:
                    if (item.id not in trial and set(item.depends_on) <= trial
                            and item.postconditions
                            and conditions_hold(item.postconditions, current)
                            and _direct_target_satisfied(item, current)):
                        trial.add(item.id)
                        skips.append(item)
                if not plan.steps or len(trial) != len(plan.steps):
                    return "idle"
                if program_frame_origin not in {
                        "backend.observe", "verified_step_response",
                        "verified_pending_effect_observation",
                        "verified_ordinary_pending_effect_observation",
                        "stale_rejection_preflight"}:
                    abandon_program("current_full_stage_boundary_source_unavailable")
                    return "abandoned"
                proof = _whole_goal_full_observation_proof(
                    self.backend, current, source=program_frame_source,
                )
                if proof is None:
                    abandon_program("current_full_stage_boundary_proof_unavailable")
                    return "abandoned"
                # Only terminal effects must survive the expected transition.
                # Readiness for an earlier edit may disappear when an explicitly
                # guarded commit closes its dialog. Use the actual verified
                # property/commit effect, not an empty property template.
                depended_on = {dep for item in plan.steps for dep in item.depends_on}
                terminal = [item for item in plan.steps if item.id not in depended_on]
                terminal_conditions = []
                for item in terminal:
                    actual = program_verified_effects.get(item.id, item)
                    if item.id in property_effects:
                        actual = property_effects[item.id][0]
                    if (not actual.postconditions
                            or not _direct_target_satisfied(actual, current)):
                        abandon_program("current_stage_terminal_effect_unavailable")
                        return "abandoned"
                    terminal_conditions.extend(actual.postconditions)
                terminal_conditions.extend(plan.completion_conditions)
                facts = _condition_facts(tuple(terminal_conditions), current)
                if (not facts or any(item.get("outcome") != "true" for item in facts)
                        or _whole_goal_full_observation_proof(
                            self.backend, current, source=program_frame_source, expected=proof,
                        ) is None):
                    abandon_program("current_stage_terminal_not_true")
                    return "abandoned"
                for item in skips:
                    completed.add(item.id)
                    counts["condition_skips"] += 1
                    counts["navigation_subgoals"] += item.navigation
                    emit("agent_condition_satisfied", {"step": item.id})
                readonly_predicates = not program_receipts
                emit("agent_program_stage_completed", {
                    **program_trace_fields(),
                    "reason": "read_only_predicates" if readonly_predicates else "verified_receipt",
                    "read_only": all(item["environment_steps"] == 0
                                     for item in program_attempt_receipts),
                    "receipt_ids": [rid for rid, _ in program_receipts],
                    "attempt_receipts": list(program_attempt_receipts),
                    "positive_input": any(type(item["environment_steps"]) is int
                                          and item["environment_steps"] > 0
                                          for item in program_attempt_receipts),
                    "positive_verified_input": any(n > 0 for _, n in program_receipts),
                    "pure_predicate_logical_attempts": 0,
                    "runtime_receipts_keep_original_logical_attempt_charge": True,
                    "terminal_conditions": facts,
                    "snapshot_fingerprint": proof[3], "image_sha256": proof[4],
                    "source": program_frame_origin,
                    "environment_steps": proof[1], "fresh_observe_calls": proof[2],
                })
                # Trace/progress callbacks cannot make the old proof current.
                if _whole_goal_full_observation_proof(
                        self.backend, current, source=program_frame_source, expected=proof,
                ) is None:
                    abandon_program("stage_boundary_changed_after_completion_trace")
                    return "abandoned"
                if program_stage_index + 1 == len(active_program.stages):
                    emit("agent_program_completed", {
                        **program_trace_fields(), "all_declared_stages_verified": True,
                        "whole_goal_review_still_required": True,
                    })
                    active_program = None
                    # Use the original whole-goal gate and its original fresh
                    # final empty-done check. Program completion is not success.
                    return whole_goal_stop_gate(completion_frame=current)
                index = program_stage_index + 1
                if time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                    return retire_program_budget("plan_expired")
                # Entry is a once-per-stage admission on this exact full
                # frame, not a permanent substitute for action preconditions.
                entry_facts = _condition_facts(
                    active_program.stages[index].entry_conditions, current,
                )
                if any(item.get("outcome") != "true" for item in entry_facts):
                    abandon_program("next_stage_current_entry_not_true")
                    return "abandoned"
                calls_before = counts["planner_calls"]
                planner_calls_before = getattr(self.planner, "calls", None)
                try:
                    activated = self.planner.activate_program_stage(
                        active_program, index, current,
                        remaining_actions=max(0, active_program.max_actions - plan_actions),
                    )
                except (ValueError, RuntimeError):
                    abandon_program("next_stage_original_validation_rejected")
                    return "abandoned"
                if (not isinstance(activated, AgentPlan) or not activated.reusable
                        or not activated.steps or activated.done or activated.prepare_repair_token
                        or counts["planner_calls"] != calls_before
                        or getattr(self.planner, "calls", None) != planner_calls_before
                        or _whole_goal_full_observation_proof(
                            self.backend, current, source=program_frame_source, expected=proof,
                        ) is None):
                    abandon_program("next_stage_activation_contract_changed")
                    return "abandoned"
                plan = activated
                planned = current
                program_stage_index = index
                program_stage_admitted = True
                # Only stage-local compiler IDs/baselines are renewed. Global
                # plan_actions, plan_at, call/provenance, failures and every used
                # recovery ticket remain exactly the original objects/values.
                completed = set()
                property_effects = {}
                completed_conditions = {}
                expected_values = _control_values(current)
                navigation_seen = {}
                program_receipts = []
                program_attempt_receipts = []
                program_verified_effects = {}
                post_snapshot_source = recovery_planning_snapshot_source = None
                stale_snapshot_source = None
                pending_numeric_planning_source = pending_numeric_finalize_source = None
                pending_prepare_repair = active_prepare_repair = None
                if navigation_memory is not None:
                    repeat_ticket_used = navigation_memory.used
                    previous_accepted_plan_edge = navigation_memory.previous
                    navigation_memory.reset(program_origin_call)
                    # New stage compiler IDs do not renew the accepted
                    # program's one known-repeat concession or relabel a
                    # same-program edge as a different accepted plan.
                    navigation_memory.used = repeat_ticket_used
                    navigation_memory.previous = previous_accepted_plan_edge
                emit("agent_program_stage_activated", {
                    **program_trace_fields(), "source": program_frame_origin,
                    "admitted": True, "entry_conditions": entry_facts,
                    "one_shot_entry_admission": True,
                    "original_action_preconditions_and_preflight_still_required": True,
                    "snapshot_fingerprint": proof[3], "image_sha256": proof[4],
                    "environment_steps": proof[1], "fresh_observe_calls": proof[2],
                    "compiled_steps": [step.to_dict() for step in plan.steps],
                    "http_calls": 0, "future_stages_still_uncompiled": True,
                })
                # Continue only to prove an already satisfied read-only stage.
                # Otherwise the original compiler/guard/executor below runs it.
            return "idle"

        def completed_conditions_hold(snapshot: AgentSnapshot) -> bool:
            values = _control_values(snapshot)
            return all(
                values.get(identity) == [expected] for identity, expected in completed_conditions.items()
            )

        set_planner_deadline = None
        restore_input_evidence = None
        try:
            configure_evidence = getattr(self.backend, "set_input_observation_evidence_enabled", None)
            if (((cfg.reuse_verified_pending_snapshot and cfg.reuse_verified_post_snapshot)
                 or unexplained_context_enabled or navigation_memory is not None
                 or stage_program_enabled
                 or (cfg.whole_goal_recheck and cfg.inband_whole_goal_recheck
                     and cfg.reuse_verified_post_snapshot)) and callable(configure_evidence)):
                restore_input_evidence = configure_evidence(True)
            set_planner_deadline = getattr(self.planner, "set_episode_deadline", None)
            if callable(set_planner_deadline):
                set_planner_deadline(start + cfg.timeout_s)
            while counts["steps"] < cfg.max_steps:
                if high_level_context_fresh_pending or planner_timeout_fresh_pending:
                    # Discard only cached authority after the preceding request's
                    # finally traced its actual wire/usage. Do not reset clocks,
                    # failure streaks, attempt counts or used recovery tickets.
                    clear_navigation_sources()
                    high_level_context_fresh_pending = False
                    planner_timeout_fresh_pending = False
                if active_program is not None and (
                        plan is None or semantic_recovery or recovery_plan
                        or network_recovery_active or capacity_recovery_active
                        or context_recovery_active):
                    abandon_program(
                        pending_reason or "original_plan_invalidated", force_recovery=False,
                    )
                if program_handoff_pending:
                    # A failed program cannot survive through a numeric/stale
                    # local shortcut; original charges and failure history stay.
                    clear_navigation_sources()
                    plan = None
                    semantic_recovery = True
                    pending_reason = pending_reason or "stage_program_recovery_required"
                    program_handoff_pending = False
                if time.perf_counter() - start >= cfg.timeout_s:
                    status, reason = "timeout", "episode time budget exhausted"
                    break
                began = time.perf_counter()
                whole_goal_source, whole_goal_pending_frame = whole_goal_pending_frame, None
                source, post_snapshot_source = post_snapshot_source, None
                unexplained_planning_source, unexplained_planning_pending = unexplained_planning_pending, None
                recovery_planning_source, recovery_planning_snapshot_source = (
                    recovery_planning_snapshot_source, None
                )
                stale_source, stale_snapshot_source = stale_snapshot_source, None
                numeric_source, pending_numeric_planning_source = pending_numeric_planning_source, None
                finalize_source, pending_numeric_finalize_source = pending_numeric_finalize_source, None
                navigation_fresh_observed = False
                if navigation_memory is not None and navigation_memory.pending is not None:
                    # A cycle's initial backend true verdict must not resurrect SDK frames.
                    source = recovery_planning_source = stale_source = None
                    numeric_source = finalize_source = None
                program_frame_origin = "unavailable"
                program_frame_source = None
                completion_check = (
                    plan is not None and plan.completion_conditions
                    and ((plan.reusable and plan.steps and len(completed) == len(plan.steps))
                         or (plan.done and not plan.steps))
                )
                if (active_program is not None
                        and program_stage_index + 1 < len(active_program.stages)):
                    # Intermediate completion is not the final whole-goal gate.
                    # Existing full verified response reuse remains source-bound.
                    completion_check = False
                take_snapshot = getattr(self.backend, "take_verified_post_snapshot", None)
                take_pending_snapshot = getattr(self.backend, "take_verified_pending_effect_snapshot", None)
                cached = None
                cache_source = "verified_step_response"
                if (
                    cfg.reuse_verified_post_snapshot and source is not None
                    and (success is None or success_uses_live_state) and not completion_check
                ):
                    if len(source) == 3 and source[2] == "verified_ordinary_pending_effect_observation":
                        take_ordinary_pending = getattr(self.backend, "take_verified_pending_snapshot", None)
                        if (cfg.reuse_verified_pending_snapshot and not semantic_recovery
                                and not (network_recovery_active or capacity_recovery_active)
                                and not context_recovery_active
                                and pending_prepare_repair is None and active_prepare_repair is None
                                and stale_source is None and numeric_source is None
                                and finalize_source is None
                                and callable(take_ordinary_pending)):
                            pending_frame = take_ordinary_pending(*source[:2])
                            if pending_frame is not None:
                                pending_snapshot, pending_received_at, pending_fingerprint = pending_frame
                                if (pending_snapshot.a11y_complete is True
                                        and pending_snapshot.image is not None
                                        and pending_snapshot.fingerprint() == pending_fingerprint):
                                    cached = pending_snapshot, pending_received_at
                                    cache_source = source[2]
                    elif callable(take_pending_snapshot):
                        cached = take_pending_snapshot(*source)
                        if cached is not None:
                            cache_source = "verified_pending_effect_observation"
                    if cached is None and len(source) == 2 and callable(take_snapshot):
                        cached = take_snapshot(*source)
                take_stale_snapshot = getattr(self.backend, "take_stale_preflight_snapshot", None)
                if (
                    cached is None and cfg.reuse_verified_post_snapshot and stale_source is not None
                    and success is None and not completion_check and callable(take_stale_snapshot)
                ):
                    cached = take_stale_snapshot(*stale_source)
                    if cached is not None:
                        source = stale_source
                        cache_source = "stale_rejection_preflight"
                numeric_planning_frame = None
                numeric_finalize_frame = None
                take_finalize_frame = getattr(self.backend, "take_declared_numeric_finalize_snapshot", None)
                if (
                    finalize_source is not None and plan is not None and plan.reusable
                    and any(step is finalize_source[2] for step in plan.steps)
                    and finalize_source[2].id not in completed and success is None
                    and not completion_check and not recovery_plan and not semantic_recovery
                    and plan_actions < cfg.plan_horizon
                    and time.perf_counter() - plan_at < cfg.plan_ttl_s
                    and callable(take_finalize_frame)
                ):
                    numeric_finalize_frame = take_finalize_frame(*finalize_source)
                take_numeric_frame = getattr(self.backend, "take_pending_numeric_planning_snapshot", None)
                if (
                    cached is None and source is None and stale_source is None and numeric_source is not None
                    and finalize_source is None
                    and cfg.reuse_verified_post_snapshot and success is None and plan is None
                    and (cfg.mode == "vlm" or semantic_recovery)
                    and not (network_recovery_active or capacity_recovery_active)
                    and not context_recovery_active
                    and pending_prepare_repair is None and action_failures < cfg.max_failures
                    and counts["planner_calls"] < cfg.max_plans and callable(take_numeric_frame)
                ):
                    numeric_planning_frame = take_numeric_frame(*numeric_source)
                inband_current_frame = None
                recovery_planning_frame = None
                unexplained_recovery_planning_only = False
                if (
                    cfg.reuse_verified_post_snapshot and recovery_planning_source is not None
                    and source is None and stale_source is None and numeric_source is None
                    and finalize_source is None and cached is None and plan is None
                    and success is None and not completion_check and context_recovery_pending is None
                    and not (network_recovery_active or capacity_recovery_active)
                    and not context_recovery_active
                    and pending_prepare_repair is None and active_prepare_repair is None
                    and action_failures < cfg.max_failures and counts["planner_calls"] < cfg.max_plans
                ):
                    recovery_planning_frame = _take_verified_recovery_planning_snapshot(
                        self.backend, recovery_planning_source[:3],
                    )
                    unexplained_recovery_planning_only = (
                        recovery_planning_frame is not None and len(recovery_planning_source) == 4
                        and recovery_planning_source[3] == "unexplained_context_planning_only"
                    )
                whole_goal_planning_frame = None
                whole_goal_reused_proof = None
                if (cfg.whole_goal_recheck and cfg.reuse_whole_goal_planning_snapshot
                        and cfg.reuse_verified_post_snapshot and whole_goal_recheck_pending
                        and plan is None and success is None
                        and pending_reason == "whole_goal_recheck" and whole_goal_source is not None
                        and counts["planner_calls"] < cfg.max_plans
                        and whole_goal_source[2] == whole_goal_input_epoch
                        and source is None and recovery_planning_source is None
                        and stale_source is None and numeric_source is None and finalize_source is None
                        and cached is None and recovery_planning_frame is None
                        and numeric_planning_frame is None and numeric_finalize_frame is None
                        and not (network_recovery_active or capacity_recovery_active
                                 or context_recovery_active or semantic_recovery)
                        and context_recovery_pending is None and pending_prepare_repair is None
                        and active_prepare_repair is None):
                    saved_frame, proof, _ = whole_goal_source
                    if _whole_goal_full_observation_proof(
                            self.backend, saved_frame, expected=proof) is not None:
                        whole_goal_planning_frame = saved_frame, proof
                unexplained_planning_frame = None
                if (unexplained_planning_reuse_enabled and unexplained_planning_source is not None
                        and plan is None and pending_reason == "unexplained_navigation_context"
                        and semantic_recovery and unexplained_context_recovery_active
                        and success is None and not completion_check and not whole_goal_recheck_pending
                        and active_program is None and not program_handoff_pending
                        and source is None and recovery_planning_source is None and stale_source is None
                        and numeric_source is None and finalize_source is None and cached is None
                        and recovery_planning_frame is None and whole_goal_planning_frame is None
                        and numeric_planning_frame is None and numeric_finalize_frame is None
                        and not (network_recovery_active or capacity_recovery_active
                                 or context_recovery_active)
                        and context_recovery_pending is None and pending_prepare_repair is None
                        and active_prepare_repair is None and action_failures < cfg.max_failures
                        and counts["planner_calls"] < cfg.max_plans
                        and _unexplained_planning_frame_is_current(
                            self.backend, unexplained_planning_source, unexplained_planning_input_epoch)):
                    unexplained_planning_frame = unexplained_planning_source
                key_only_observation = numeric_finalize_frame is not None
                if unexplained_planning_frame is not None:
                    snapshot = unexplained_planning_frame.snapshot
                    # Preserve the complete-goal review instruction, never completion credit.
                    inband_current_frame = inband_frame_proof(
                        snapshot, "verified_pending_unexplained_context_planning_frame",
                    )
                    inband_last_fresh = None
                    last_runner_fresh_snapshot = last_runner_fresh_proof = None
                elif recovery_planning_frame is not None:
                    snapshot, received_at = recovery_planning_frame
                    inband_current_frame = inband_frame_proof(
                        snapshot, "verified_h1_recovery_step_response_for_planning",
                        recovery_planning_source[:2],
                    )
                    counts["recovery_planning_snapshot_reuses"] += 1
                    emit("agent_recovery_planning_observation_reuse", {
                        "receipt_id": recovery_planning_source[1].receipt_id,
                        "response_age_ms": max(0, (time.perf_counter() - received_at) * 1000),
                        "source": "verified_h1_recovery_step_response_for_planning",
                        "native_execution_fingerprint": snapshot.fingerprint(),
                        "planning_only": True, "verified": True, "task_success": False,
                    })
                elif numeric_finalize_frame is not None:
                    snapshot, received_at = numeric_finalize_frame
                    emit("agent_declared_numeric_finalize_context", {
                        "receipt_id": finalize_source[1].receipt_id,
                        "response_age_ms": max(0, (time.perf_counter() - received_at) * 1000),
                        "source": "declared_numeric_finalize_key_only_response",
                        "key": finalize_source[2].property_edit.numeric_finalize_key,
                        "verified": False, "task_success": False,
                    })
                elif numeric_planning_frame is not None:
                    snapshot, received_at = numeric_planning_frame
                    emit("agent_planning_observation_reuse", {
                        "receipt_id": numeric_source[1].receipt_id,
                        "response_age_ms": max(0, (time.perf_counter() - received_at) * 1000),
                        "source": "unverified_numeric_typing_response_for_planning",
                        "max_actions": 1, "verified": False, "task_success": False,
                    })
                elif whole_goal_planning_frame is not None:
                    snapshot = whole_goal_planning_frame[0]
                elif cached is None:
                    snapshot = fresh_observe(completion_proof=bool(
                        completion_check and plan is not whole_goal_reviewed_done_plan
                    ))
                    navigation_fresh_observed = True
                    if capacity_recovery_fresh_required:
                        capacity_recovery_fresh_required = False
                        emit("agent_capacity_recovery", {
                            "stage": "fresh_observed", "source": "backend.observe",
                            "task_success": False,
                        })
                else:
                    snapshot, received_at = cached
                    if cache_source in {"verified_step_response", "verified_pending_effect_observation",
                                        "verified_ordinary_pending_effect_observation"}:
                        inband_current_frame = inband_frame_proof(
                            snapshot, cache_source,
                            source[:2] if cache_source == "verified_step_response" else None,
                        )
                    pending_cache = cache_source == "verified_ordinary_pending_effect_observation"
                    counts["verified_pending_snapshot_reuses" if pending_cache else
                           "stale_snapshot_reuses" if cache_source == "stale_rejection_preflight"
                           else "post_snapshot_reuses"] += 1
                    reuse_kind = (
                        "agent_verified_pending_observation_reuse"
                        if pending_cache else "agent_observation_reuse"
                    )
                    emit(reuse_kind, {
                        "receipt_id": source[1].receipt_id,
                        "response_age_ms": max(0, (time.perf_counter() - received_at) * 1000),
                        "source": cache_source,
                        **({"native_execution_fingerprint": snapshot.fingerprint(),
                            "observation_source": "ordinary_pending_effect_full_observe",
                            "sdk_receipt_fingerprint_unchanged": True} if pending_cache else {}),
                    })
                if active_program is not None:
                    if snapshot is last_runner_fresh_snapshot:
                        program_frame_origin = "backend.observe"
                        program_frame_source = None
                    elif cached is not None and cache_source in {
                            "verified_step_response", "verified_pending_effect_observation",
                            "verified_ordinary_pending_effect_observation",
                            "stale_rejection_preflight"}:
                        program_frame_origin = cache_source
                        program_frame_source = (
                            source[:2] if cache_source == "verified_step_response" else None
                        )
                counts["observation_ms"] += (time.perf_counter() - began) * 1000
                if ((network_recovery_active or capacity_recovery_active
                     or whole_goal_recheck_pending
                     or (whole_goal_reviewed_done_plan is not None and plan is whole_goal_reviewed_done_plan))
                        and time.perf_counter() - start >= cfg.timeout_s):
                    raise _EpisodeExpired()
                if success is not None:
                    began = time.perf_counter()
                    accepted = bool(success(snapshot))
                    counts["acceptance_ms"] += (time.perf_counter() - began) * 1000
                    if accepted:
                        status, reason = "success", "independent acceptance checks passed"
                        break
                if context_recovery_pending is not None:
                    context = context_recovery_pending
                    context_recovery_pending = None
                    confirm_context = getattr(self.backend, "confirm_unverified_recovery_context", None)
                    if not callable(confirm_context) or not confirm_context(context, snapshot):
                        status, reason = "stuck", "new recovery context no longer observed"
                        break
                if active_program is not None:
                    boundary = program_boundary(snapshot)
                    if boundary == "review":
                        continue
                    if boundary == "blocked":
                        status, reason = "stuck", "whole-goal recheck requires new actual input"
                        break
                    if boundary == "abandoned":
                        continue
                if (
                    plan is not None and plan.defer_dialog_field_completion is True
                    and plan.completion_conditions and plan.steps
                    and len(completed) == len(plan.steps)
                    and conditions_hold(plan.completion_conditions, snapshot)
                ):
                    from .desktop_property import pending_dialog_field_completion

                    dialog = pending_dialog_field_completion(plan, snapshot)
                    if dialog is not None:
                        feedback = {
                            "reason": "property_confirmation_required",
                            "feedback": (
                                "The declared field conditions hold, but the field remains inside "
                                "an observed dialog or modal form. This is not persistence evidence. "
                                "Inspect the current UI and explicitly plan any required guarded "
                                "confirmation; "
                                "do not repeat the property edit or infer an automatic Enter/Save/OK."
                            ),
                            "pending_dialog": dialog,
                            "completion_conditions": _condition_facts(plan.completion_conditions, snapshot),
                        }
                        history.append(feedback)
                        emit("agent_property_confirmation_required", feedback)
                        plan = None
                        pending_reason = "property_confirmation_required"
                if (
                    plan is not None and plan.completion_conditions
                    and ((plan.reusable and plan.steps and len(completed) == len(plan.steps))
                         or (plan.done and not plan.steps))
                    and conditions_hold(plan.completion_conditions, snapshot)
                ):
                    # This is the planner's explicit whole-goal stopping claim,
                    # checked only after every step and against a fresh observation.
                    # It never establishes independent task success.
                    counts["plan_completion_claims"] += 1
                    emit("agent_plan_completion", {
                        "conditions": _condition_facts(plan.completion_conditions, snapshot),
                        "completed_steps": sorted(completed),
                        "task_success": False,
                        "independent_check_rejected": success is not None,
                    })
                    if success is None:
                        gate = whole_goal_stop_gate(completion_frame=snapshot)
                        if gate == "review":
                            continue
                        if gate == "blocked":
                            status, reason = "stuck", "whole-goal recheck requires new actual input"
                            break
                        status, reason = (
                            "unverified", "plan completion conditions met; no independent evaluator"
                        )
                        break
                    action_failures += 1
                    plan = None
                    pending_reason = "completion_rejected"
                    history.append({"feedback": "Independent goal checks still fail; continue the task."})
                    if action_failures >= cfg.max_failures:
                        status, reason = "task_failure", "plan completion contradicted acceptance checks"
                        break
                if plan is not None and plan.done and not plan.steps and plan.completion_conditions:
                    # Empty completion declarations use the same fresh observation
                    # as independent acceptance, never the pre-request screenshot.
                    facts = _condition_facts(plan.completion_conditions, snapshot)
                    action_failures += 1
                    pending_reason = "completion_unconfirmed"
                    feedback = {
                        "reason": pending_reason,
                        "feedback": (
                            "No input was sent. The declared goal completion is not confirmed in the "
                            "fresh observation. Inspect current state and continue the task. Unknown "
                            "is not evidence that the condition is false or that another input is needed."
                        ),
                        "completion_conditions": facts,
                    }
                    history.append(feedback)
                    emit("agent_recovery", feedback)
                    plan = None
                    if action_failures >= cfg.max_failures:
                        status, reason = "stuck", "goal completion conditions remain unconfirmed"
                        break
                if plan is not None:
                    if cfg.mode == "vlm" and plan_actions:
                        pending_reason = "vlm_every_step"
                    elif plan.reusable:
                        # Semantic plans survive expected UI transitions. Each
                        # action and its conditions are rebound to this observation.
                        if time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                            pending_reason = "plan_expired"
                        elif plan_actions >= cfg.plan_horizon:
                            pending_reason = "horizon_reached"
                        elif len(completed) == len(plan.steps):
                            pending_reason = "plan_exhausted"
                    elif snapshot.scope_key() != planned.scope_key():
                        pending_reason = "page_or_controls_changed"
                    elif not completed_conditions_hold(snapshot):
                        pending_reason = "completed_state_changed"
                        history.append(
                            {"feedback": "A completed field changed; inspect and repair before submitting."}
                        )
                    elif _control_values(snapshot) != expected_values:
                        pending_reason = "field_state_changed"
                        history.append(
                            {"feedback": "An unplanned field change invalidated the plan; reassess."}
                        )
                    elif time.perf_counter() - plan_at >= cfg.plan_ttl_s:
                        pending_reason = "plan_expired"
                    elif plan_actions >= cfg.plan_horizon:
                        pending_reason = "horizon_reached"
                    elif len(completed) == len(plan.steps):
                        pending_reason = "plan_exhausted"
                    if pending_reason:
                        if active_program is not None:
                            abandon_program(pending_reason, force_recovery=False)
                        plan = None
                if plan is None:
                    if active_program is not None:
                        abandon_program(
                            pending_reason or "original_replan_required", force_recovery=False,
                        )
                    if program_handoff_pending:
                        clear_navigation_sources()
                        semantic_recovery = True
                        program_handoff_pending = False
                    if key_only_observation:
                        # The independent numeric response never becomes a
                        # planner/goal frame when the original plan expires.
                        began = time.perf_counter()
                        snapshot = fresh_observe()
                        counts["observation_ms"] += (time.perf_counter() - began) * 1000
                        key_only_observation = False
                    if finalize_source is not None:
                        finalize_source = None
                        semantic_recovery = True
                    if time.perf_counter() - start >= cfg.timeout_s:
                        status, reason = "timeout", "episode time budget exhausted before planning"
                        break
                    if network_recovery_active and network_recovery_planned:
                        status, reason = "stuck", "network recovery plan invalidated or produced no input"
                        break
                    if capacity_recovery_active and capacity_recovery_fresh_required:
                        status, reason = "stuck", "capacity recovery lacks an actual fresh observation"
                        break
                    if capacity_recovery_active and capacity_recovery_planned:
                        status, reason = "stuck", "capacity recovery plan invalidated or produced no input"
                        break
                    if context_recovery_active and context_recovery_planned:
                        status, reason = "stuck", "final context recovery plan invalidated"
                        break
                    if counts["planner_calls"] >= cfg.max_plans:
                        status, reason = "max_plans", "planning budget exhausted"
                        break
                    # Formal review uses the existing strong H1 shape, including a fresh-frame retry.
                    logical_limit = (1 if cfg.mode == "vlm" or semantic_recovery
                                     or whole_goal_recheck_pending else cfg.plan_horizon)
                    prepare_browser = getattr(self.backend, "prepare_planning_snapshot", None)
                    if callable(prepare_browser):
                        began = time.perf_counter()
                        snapshot = prepare_browser(snapshot, reason=pending_reason or "initial")
                        counts["observation_ms"] += (time.perf_counter() - began) * 1000
                    requested_limit = logical_limit
                    request_native_fingerprint = snapshot.fingerprint() if v52_request_trace else None
                    request_routing_reason = (
                        "vlm_baseline" if cfg.mode == "vlm" else
                        "existing_single_action_recovery" if semantic_recovery else
                        "native_root" if v52_information_routing else "full_multimodal_root"
                    )
                    if v52_information_routing and visual_information_anchor is not None:
                        anchor_matches = (
                            not (network_recovery_active or capacity_recovery_active)
                            and snapshot.a11y_complete is True
                            and snapshot.image is not None
                            and request_native_fingerprint == visual_information_anchor["fingerprint"]
                        )
                        emit("agent_visual_information_routing", {
                            "status": "retained" if anchor_matches else "invalidated",
                            "reason": "same_complete_native_fingerprint" if anchor_matches else
                                      "current_observation_or_recovery_changed",
                            **visual_information_anchor,
                            "native_execution_fingerprint": request_native_fingerprint,
                            "information_only": True, "task_success": False,
                        })
                        if anchor_matches:
                            requested_limit = 1
                            request_routing_reason = "verified_visual_progress_same_native"
                        else:
                            visual_information_anchor = None
                    effective_limit = getattr(self.planner, "effective_max_actions", None)
                    if callable(effective_limit):
                        routed_limit = requested_limit
                        requested_limit = effective_limit(snapshot, routed_limit)
                        if type(requested_limit) is not int or requested_limit not in {1, routed_limit}:
                            raise ValueError("planner observation hook returned an invalid action limit")
                        if requested_limit != routed_limit:
                            request_routing_reason = "incomplete_native_snapshot"
                    reasons[pending_reason or "no_ready_action"] += 1
                    counts["planner_calls"] += 1
                    if pending_reason == "unexplained_navigation_context":
                        counts["unexplained_navigation_context_replans"] += 1
                    whole_goal_request = whole_goal_recheck_pending
                    if whole_goal_request:
                        whole_goal_recheck_pending = False
                        whole_goal_attempted_epoch = whole_goal_input_epoch
                        counts["whole_goal_recheck_calls"] += 1
                        request_routing_reason = "whole_goal_recheck"
                        emit("agent_whole_goal_recheck", {
                            "stage": "requested", "call": counts["planner_calls"],
                            "input_epoch": whole_goal_input_epoch, "task_success": False,
                        })
                    if context_recovery_active:
                        context_recovery_planned = True
                    if network_recovery_active:
                        network_recovery_planned = True
                    if capacity_recovery_active:
                        capacity_recovery_planned = True
                    began = time.perf_counter()
                    repair_frame = pending_prepare_repair
                    repair_context = (repair_frame.context(snapshot, now=time.perf_counter())
                                      if repair_frame is not None else None)
                    set_repair_context = getattr(self.planner, "set_prepare_repair_context", None)
                    if not callable(set_repair_context) or requested_limit != 1:
                        repair_context = None
                    if repair_frame is not None and repair_context is None:
                        emit("agent_prepare_repair", {"status": "offer_expired_or_context_changed",
                                                       "input_sent": False})
                    pending_prepare_repair = None  # One explicit offer, consumed by this request.
                    repaired_frame = None
                    request_information = None
                    inband_offer = None
                    inband_marked = False
                    planner_invoked = False
                    inband_epoch_dispatched = None
                    inband_provenance_before = getattr(self.planner, "last_request_provenance", None)
                    frame_proof = (inband_last_fresh if inband_last_fresh is not None
                                   and snapshot is inband_last_fresh[0] else inband_current_frame)
                    if (cfg.whole_goal_recheck and cfg.inband_whole_goal_recheck
                            and not whole_goal_request
                            and getattr(self.planner, "profile", None) == "desktop"
                            and getattr(self.planner, "supports_inband_whole_goal_recheck", False) is True
                            and frame_proof is not None and snapshot is frame_proof[0]):
                        proof = _whole_goal_full_observation_proof(
                            self.backend, snapshot, source=frame_proof[3], expected=frame_proof[1],
                        )
                        if proof is not None:
                            inband_offer = whole_goal_input_epoch, proof, frame_proof[2], frame_proof[3]
                    try:
                        if callable(set_repair_context):
                            set_repair_context(repair_context)
                        if repair_context is not None and callable(set_planner_deadline):
                            set_planner_deadline(min(start + cfg.timeout_s, repair_frame.deadline))
                        planning_options = {"max_actions": requested_limit}
                        if whole_goal_request:
                            planning_options["whole_goal_recheck"] = True
                        elif inband_offer is not None:
                            proof = _whole_goal_full_observation_proof(
                                self.backend, snapshot, source=inband_offer[3], expected=inband_offer[1],
                            )
                            if proof is not None and whole_goal_input_epoch == inband_offer[0]:
                                planning_options["whole_goal_inband_epoch"] = inband_offer[0]
                                inband_epoch_dispatched = inband_offer[0]
                            else:
                                inband_offer = None
                        if whole_goal_request and whole_goal_planning_frame is not None:
                            # Callback/bookkeeping may have consumed the age budget or read/input.
                            proof = whole_goal_planning_frame[1]
                            if _whole_goal_full_observation_proof(
                                    self.backend, snapshot, expected=proof) is None:
                                observed_at = time.perf_counter()
                                snapshot = fresh_observe()
                                counts["observation_ms"] += (time.perf_counter() - observed_at) * 1000
                                requested_limit = logical_limit
                                if callable(effective_limit):
                                    requested_limit = effective_limit(snapshot, logical_limit)
                                    if (type(requested_limit) is not int
                                            or requested_limit not in {1, logical_limit}):
                                        raise ValueError(
                                            "planner observation hook returned an invalid action limit"
                                        )
                                planning_options["max_actions"] = requested_limit
                                request_native_fingerprint = (
                                    snapshot.fingerprint() if v52_request_trace else None
                                )
                                whole_goal_planning_frame = None
                            else:
                                whole_goal_reused_proof = proof
                            if time.perf_counter() - start >= cfg.timeout_s:
                                status, reason = "timeout", "episode time budget exhausted before planning"
                                break
                        if whole_goal_reused_proof is not None:
                            counts["whole_goal_planning_snapshot_reuses"] += 1
                        unexplained_dispatch_age_ms = None
                        if unexplained_planning_frame is not None:
                            if not _unexplained_planning_frame_is_current(
                                    self.backend, unexplained_planning_frame,
                                    unexplained_planning_input_epoch):
                                observed_at = time.perf_counter()
                                snapshot = fresh_observe()
                                counts["observation_ms"] += (time.perf_counter() - observed_at) * 1000
                                requested_limit = logical_limit
                                if callable(effective_limit):
                                    requested_limit = effective_limit(snapshot, logical_limit)
                                    if requested_limit != 1:
                                        raise ValueError("unexplained context recovery requires original H1")
                                planning_options["max_actions"] = requested_limit
                                request_native_fingerprint = (
                                    snapshot.fingerprint() if v52_request_trace else None
                                )
                                unexplained_planning_frame = None
                                inband_offer = None
                                inband_epoch_dispatched = None
                                planning_options.pop("whole_goal_inband_epoch", None)
                                if (cfg.whole_goal_recheck and cfg.inband_whole_goal_recheck
                                        and getattr(self.planner,
                                                    "supports_inband_whole_goal_recheck", False) is True):
                                    proof = _whole_goal_full_observation_proof(self.backend, snapshot)
                                    if proof is not None:
                                        inband_offer = whole_goal_input_epoch, proof, "backend.observe", None
                                        planning_options["whole_goal_inband_epoch"] = whole_goal_input_epoch
                                        inband_epoch_dispatched = whole_goal_input_epoch
                                if time.perf_counter() - start >= cfg.timeout_s:
                                    raise _EpisodeExpired()
                            else:
                                unexplained_dispatch_age_ms = max(
                                    0, (time.perf_counter() - unexplained_planning_frame.proof[0]) * 1000,
                                )
                        planner_invoked = True
                        program_offer = None
                        try:
                            plan = self.planner.plan(goal, snapshot, history[-8:], **planning_options)
                        finally:
                            if unexplained_planning_frame is not None:
                                emit("agent_unexplained_context_planning_observation_reuse", {
                                    "call": counts["planner_calls"],
                                    "source": "verified_pending_unexplained_context_full_frame",
                                    "receipt_id": unexplained_planning_frame.receipt.receipt_id,
                                    "input_epoch": unexplained_planning_input_epoch,
                                    "native_execution_fingerprint": unexplained_planning_frame.proof[3],
                                    "image_sha256": unexplained_planning_frame.proof[4],
                                    "age_at_dispatch_ms": unexplained_dispatch_age_ms,
                                    "whole_goal_review_requested":
                                        planning_options.get("whole_goal_inband_epoch") is not None,
                                    "planning_only": True, "input_authority": False, "task_success": False,
                                    "freshness_after_model_not_asserted": True,
                                })
                        if isinstance(plan, HighLevelContextRequest):
                            if (cfg.mode != "hybrid" or requested_limit <= 1
                                    or getattr(self.planner, "profile", None) != "desktop"
                                    or getattr(self.planner, "desktop_contract", None)
                                    != "property_intent_v52"
                                    or getattr(self.planner, "desktop_output_format", None)
                                    not in {"semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}
                                    or whole_goal_request or repair_context is not None
                                    or semantic_recovery or recovery_plan
                                    or network_recovery_active or capacity_recovery_active
                                    or context_recovery_active or unexplained_context_recovery_active):
                                raise ValueError("high-level context request requires ordinary62 H6")
                            context_request = plan
                            plan = None  # Typed context is never an executable plan.
                            planning_failures += 1  # Original zero-input planning boundary.
                            pending_reason = "high_level_context_requested"
                            feedback = {
                                "reason": pending_reason, "context_reason": context_request.reason,
                                "input_sent": False, "task_success": False, "recovery_required": True,
                                "feedback": (
                                    "The high-level phase requested current context; no input was sent. "
                                    "Use the next fresh full observation for one original grounded "
                                    "current action. Resume high-level planning only after its effect "
                                    "is verified, within the unchanged episode and recovery budgets."
                                ),
                            }
                            history.append(feedback)
                            emit("agent_high_level_context_requested", {
                                **feedback, "declaration_sha256": context_request.declaration_sha256,
                                "planner_calls": counts["planner_calls"], "next_max_actions": 1,
                                "planning_failures": planning_failures,
                                "action_failures": action_failures, "logical_attempts": 0,
                                "positive_input_credit": False, "original_plan_at": plan_at,
                                "original_plan_actions": plan_actions,
                                "selection_recovery_ticket_used": selection_recovery_ticket_used,
                            })
                            if planning_failures >= cfg.max_failures:
                                status = "planner_error"
                                reason = "high-level context handoff budget exhausted"
                                break
                            semantic_recovery = True
                            high_level_context_fresh_pending = True
                            continue
                        if isinstance(plan, StageProgramPlan):
                            if (not stage_program_enabled or requested_limit <= 1
                                    or semantic_recovery or recovery_plan
                                    or repair_context is not None
                                    or network_recovery_active or capacity_recovery_active
                                    or context_recovery_active
                                    or unexplained_context_recovery_active
                                    or plan.program.max_actions != requested_limit):
                                raise ValueError("stage program requires ordinary H6 context")
                            program_offer = plan.program
                            plan = plan.initial_plan
                            if (not plan.reusable or not plan.steps or plan.done
                                    or plan.prepare_repair_token):
                                raise ValueError("stage program requires validated initial stage")
                        if whole_goal_reused_proof is not None:
                            emit("agent_whole_goal_planning_observation_reuse", {
                                "call": counts["planner_calls"],
                                "source": "successful_completion_backend_observe",
                                "native_execution_fingerprint": whole_goal_reused_proof[3],
                                "image_sha256": whole_goal_reused_proof[4], "planning_only": True,
                                "task_success": False,
                            })
                        if whole_goal_request and not plan.steps and not (
                            plan.done and plan.completion_conditions
                        ):
                            raise ValueError("whole-goal recheck empty completion requires conditions")
                        if requested_limit == 1 and any(step.navigation for step in plan.steps):
                            raise ValueError("single-action planning cannot delegate navigation")
                        if context_recovery_active and len(plan.steps) > 1:
                            raise ValueError("new-context recovery permits one immediate action")
                        if network_recovery_active and (
                            len(plan.steps) > 1
                            or any(step.operation == "wait" for step in plan.steps)
                            or (not plan.steps and not (plan.done and plan.completion_conditions))
                        ):
                            raise ValueError("network recovery permits one action or guarded completion")
                        if capacity_recovery_active and (
                            len(plan.steps) > 1
                            or any(step.operation == "wait" for step in plan.steps)
                            or (not plan.steps and not (plan.done and plan.completion_conditions))
                            or plan.prepare_repair_token
                        ):
                            raise ValueError("capacity recovery permits one action or guarded completion")
                        if plan.prepare_repair_token:
                            if repair_context is None or repair_frame is None:
                                raise ValueError("preparation repair has no current runner offer")
                            from .desktop_plan_repair import repair_property_prepare

                            plan = repair_property_prepare(
                                repair_frame, plan, snapshot, now=time.perf_counter(),
                            )
                            repaired_frame = repair_frame
                    except VisualInspectionRequested:
                        # A declared need to inspect pixels is zero input, not
                        # a completed plan or a fresh TTL/failure allowance.
                        plan = None
                        semantic_recovery = True
                        pending_reason = "visual_context_requested"
                        feedback = {
                            "reason": pending_reason, "input_sent": False,
                            "task_success": False, "recovery_required": True,
                            "feedback": "The native planning request needs the current screenshot. "
                                        "Inspect one fresh full-image observation and return one "
                                        "current grounded action under the original episode budgets.",
                        }
                        history.append(feedback)
                        emit("agent_visual_context_requested", {
                            **feedback, "planner_calls": counts["planner_calls"],
                            "next_max_actions": 1,
                        })
                        continue
                    except (ValueError, RuntimeError) as exc:
                        plan = None  # A rejected response must never survive into the next loop.
                        planning_failures += 1
                        pending_reason = "invalid_plan"
                        # Invalid desktop plans have not sent input. Recover with
                        # one fresh action, without retrying the same broad plan
                        # or changing either failure streak. Transport failures
                        # do not establish that the planning contract is wrong.
                        validation_recovery = (
                            isinstance(exc, ValueError) and cfg.mode == "hybrid"
                            and (getattr(self.planner, "profile", None) == "desktop"
                                 or (cfg.adaptive_dispatch
                                     and getattr(self.planner, "profile", None) == "browser"))
                        )
                        transport_failure = _consume_local_read_timeout(self.planner, exc)
                        # recovery_plan describes the previously accepted plan;
                        # after verified H1 it may outlive that recovery phase.
                        # Bind this HTTP phase to its local capsule/actual limits.
                        timeout_recovery = (
                            type(transport_failure) is dict
                            and transport_failure["kind"] == "local_post_read_timeout"
                            and cfg.mode == "hybrid"
                            and cfg.plan_horizon == logical_limit == requested_limit == 6
                            and getattr(self.planner, "profile", None) == "desktop"
                            and getattr(self.planner, "desktop_contract", None) == "property_intent_v52"
                            and getattr(self.planner, "desktop_output_format", None) == "semantic_json_v66"
                            and transport_failure["max_actions"] == 6
                            and transport_failure["response_transport"] == "semantic_json_v66"
                            and transport_failure["phase"] == "normal_high_level"
                            and transport_failure["normal_json_schema"] is True
                            and transport_failure["normal_high_level_phase"] is True
                            and transport_failure["image_uploaded"] is True
                            and transport_failure["whole_goal_recheck"] is not True
                            and not (
                                whole_goal_request
                                or repair_context is not None
                                or semantic_recovery
                                or network_recovery_active
                                or capacity_recovery_active
                                or context_recovery_active
                                or unexplained_context_recovery_active
                                or active_program is not None
                                or program_handoff_pending
                            )
                            and planning_failures < cfg.max_failures
                            and action_failures < cfg.max_failures
                            and counts["planner_calls"] < cfg.max_plans
                            and time.perf_counter() - start < cfg.timeout_s
                        )
                        if validation_recovery or timeout_recovery:
                            semantic_recovery = invalid_plan_recovery = True
                        if timeout_recovery:
                            pending_reason = "normal_H6_local_read_timeout"
                            planner_timeout_fresh_pending = True
                            emit(
                                "agent_planner_transport_recovery",
                                {
                                    "kind": "local_post_read_timeout",
                                    "source": "local_HTTP_client_exception",
                                    "call": counts["planner_calls"],
                                    "actual_planner_call": transport_failure["actual_planner_call"],
                                    "next_max_actions": 1,
                                    "planning_failures": planning_failures,
                                    "action_failures": action_failures,
                                    "input_sent": False,
                                    "positive_input_credit": False,
                                    "task_success": False,
                                    "original_plan_at": plan_at,
                                    "original_plan_actions": plan_actions,
                                    "original_episode_deadline": start + cfg.timeout_s,
                                    "selection_recovery_ticket_used": selection_recovery_ticket_used,
                                    "provider_completion_and_cost_unknown": True,
                                },
                            )
                        response_structure = getattr(self.planner, "last_response_structure", None)
                        feedback = {
                            "feedback": (
                                'The high-level request locally timed out while reading '
                                'its response; no UI input was sent. Inspect one fresh f'
                                'ull observation for an original grounded current action'
                                ' under the same remaining budgets. Provider completion/'
                                'cost is unknown.'
                                if timeout_recovery else
                                "Planning failed validation; no input was sent. Return one current "
                                "grounded recovery action; resume broader planning only after its "
                                "effect is verified."
                                if validation_recovery else
                                "Planning failed validation; return a valid grounded plan."
                            ),
                            "error_type": type(exc).__name__,
                            "validation": getattr(self.planner, "last_validation_error", ""),
                            **({"response_structure": response_structure}
                               if type(response_structure) is dict else {}),
                            **({"model_usage": dict(self.planner.last_usage)}
                               if type(getattr(self.planner, "last_usage", None)) is dict else {}),
                        }
                        visual_hint = _native_missing_visual_feedback(
                            self.planner, exc, whole_goal_request=whole_goal_request,
                            repair_context=repair_context,
                        )
                        if visual_hint is not None:
                            feedback["feedback"] += " " + visual_hint
                        if validation_recovery or timeout_recovery:
                            feedback["recovery_required"] = True
                        history.append(feedback)
                        # Keep additional diagnostics out of feedback/recent_actions.
                        rejection_trace_fields = {}
                        if getattr(self.planner, "desktop_output_format", None) in {
                            "semantic_tools_v57", "semantic_tools_v58",
                            "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64",
                                "semantic_json_v66"}:
                            rejection = getattr(self.planner, "last_declaration_rejection", None)
                            if type(rejection) is dict:
                                rejection_trace_fields = {"declaration_rejection": rejection}
                        else:
                            rejection = getattr(self.planner, "last_declaration_rejection", None)
                            if (type(rejection) is dict and rejection.get("version")
                                    == "bounded_prepared_property_scope_rejection_v1"):
                                rejection_trace_fields = {"declaration_rejection": rejection}
                        emit("agent_recovery", {
                            "reason": pending_reason, **feedback, **rejection_trace_fields,
                        })
                        if network_recovery_active:
                            status, reason = "planner_error", "network recovery planning failed"
                            break
                        if capacity_recovery_active:
                            if capacity_phase_v3 and isinstance(exc, ValueError):
                                if planning_failures >= cfg.max_failures:
                                    status, reason = "planner_error", "consecutive planning failures"
                                    break
                                if retry_capacity_plan("planning_validation", goal_review=whole_goal_request):
                                    continue
                            status, reason = "planner_error", "capacity recovery planning failed"
                            break
                        if context_recovery_active:
                            status, reason = "planner_error", "final context recovery planning failed"
                            break
                        if planning_failures >= cfg.max_failures:
                            status, reason = "planner_error", "consecutive planning failures"
                            break
                        continue
                    finally:
                        if callable(set_repair_context):
                            set_repair_context(None)
                        if repair_context is not None and callable(set_planner_deadline):
                            set_planner_deadline(start + cfg.timeout_s)
                        counts["planner_ms"] += (time.perf_counter() - began) * 1000
                        provenance = getattr(self.planner, "last_request_provenance", None)
                        inband_marked = bool(
                            planner_invoked and inband_offer is not None
                            and inband_epoch_dispatched == inband_offer[0]
                            and type(provenance) is dict
                            and provenance is not inband_provenance_before
                            and type(provenance.get("whole_goal_inband_review")) is dict
                            and type(provenance["whole_goal_inband_review"].get("input_epoch")) is int
                            and provenance.get("whole_goal_inband_review") == {
                                "version": "all_original_goal_review_v1", "input_epoch": inband_offer[0],
                            }
                            and provenance.get("image_uploaded") is True
                            and provenance.get("a11y_complete") is True
                        )
                        if inband_marked:
                            counts["whole_goal_inband_review_calls"] += 1
                            emit("agent_whole_goal_inband_review", {
                                "stage": "requested", "call": counts["planner_calls"],
                                "input_epoch": inband_offer[0], "task_success": False,
                                "source": "planner_request_configuration", "extra_HTTP_call": False,
                                "observation_source": inband_offer[2],
                                "native_execution_fingerprint": inband_offer[1][3],
                                "image_sha256": inband_offer[1][4],
                            })
                        if (getattr(self.planner, "desktop_output_format", None) in {
                                    "tool_declaration_v52", "semantic_tools_v53",
                                    "semantic_tools_v54", "semantic_tools_v55",
                                    "semantic_tools_v56", "semantic_tools_v57",
                                    "semantic_tools_v58",
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}
                                and isinstance(provenance, dict)
                                and provenance.get("response_transport") in {
                                    "tool_declaration_v52", "semantic_tools_v53",
                                    "semantic_tools_v54", "semantic_tools_v55",
                                    "semantic_tools_v56", "semantic_tools_v57",
                                    "semantic_tools_v58",
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}):
                            transport_event = {
                                "call": counts["planner_calls"],
                                "source": "planner_wire_transport",
                                "response_transport": provenance["response_transport"],
                                "image_uploaded": provenance["image_uploaded"],
                                "max_actions": provenance["max_actions"],
                                "declaration": getattr(self.planner, "last_response_transport", None),
                            }
                            if getattr(self.planner, "desktop_output_format", None) in {
                                    "semantic_tools_v53", "semantic_tools_v54", "semantic_tools_v55",
                                    "semantic_tools_v56", "semantic_tools_v57",
                                    "semantic_tools_v58",
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}:
                                transport_event["requested_format"] = self.planner.desktop_output_format
                                http_wall = provenance.get("http_wall_ms")
                                if (type(http_wall) in {int, float}
                                        and math.isfinite(http_wall) and http_wall >= 0):
                                    transport_event["http_wall_ms"] = http_wall
                            if getattr(self.planner, "desktop_output_format", None) in {
                                    "semantic_tools_v54", "semantic_tools_v55",
                                    "semantic_tools_v56", "semantic_tools_v57",
                                    "semantic_tools_v58",
                "semantic_tools_v60", "semantic_tools_v62", "semantic_tools_v64", "semantic_json_v66"}:
                                for field in (
                                    "ancestor_response_transport", "ancestor_declaration_schema_sha256",
                                    "declaration_schema_sha256", "declaration_tools_sha256",
                                    "parameter_schema_sha256",
                                    "ancestor_parameter_schema_sha256", "generation_projection_format",
                                ):
                                    transport_event[field] = provenance.get(field)
                            emit("agent_model_output_transport", transport_event)
                        if v52_request_trace:
                            request_information = _v52_request_provenance(self.planner, requested_limit)
                            if request_information is not None:
                                if request_information["max_actions"] != requested_limit:
                                    request_routing_reason = "planner_history_single_action_recovery"
                                request_information.update(
                                    native_execution_fingerprint=request_native_fingerprint,
                                    routing_reason=request_routing_reason,
                                )
                                emit("agent_planning_observation", {
                                    "call": counts["planner_calls"],
                                    "logical_max_actions": logical_limit,
                                    "requested_max_actions": requested_limit,
                                    "effective_max_actions": request_information["max_actions"],
                                    "source": "planner_request_configuration",
                                    **request_information,
                                })
                        elif (isinstance(provenance, dict)
                                and provenance.get("planning_observation") == "native_first_v51"
                                and type(provenance.get("image_uploaded")) is bool
                                and type(provenance.get("a11y_complete")) is bool
                                and type(provenance.get("max_actions")) is int):
                            emit("agent_planning_observation", {
                                "call": counts["planner_calls"], "logical_max_actions": logical_limit,
                                "effective_max_actions": provenance["max_actions"],
                                "image_uploaded": provenance["image_uploaded"],
                                "a11y_complete": provenance["a11y_complete"],
                                "source": "planner_request_configuration",
                                "planning_observation": "native_first_v51",
                            })
                    if (inband_marked and not whole_goal_request and unexplained_planning_frame is None
                            and not unexplained_recovery_planning_only
                            and plan.done and not plan.steps and plan.completion_conditions
                            and whole_goal_input_epoch == inband_offer[0]
                            and type(getattr(self.backend, "environment_steps", None)) is int
                            and self.backend.environment_steps == inband_offer[1][1]
                            and type(getattr(self.backend, "fresh_observe_calls", None)) is int
                            and self.backend.fresh_observe_calls == inband_offer[1][2]):
                        whole_goal_inband_done = (
                            plan, inband_offer[0], inband_offer[1][1], counts["planner_calls"],
                            inband_offer[1][2],
                        )
                    if whole_goal_request:
                        if plan.steps:
                            counts["whole_goal_recheck_continues"] += 1
                        else:
                            whole_goal_reviewed_done_plan = plan
                            whole_goal_reviewed_done_epoch = whole_goal_input_epoch
                        emit("agent_whole_goal_recheck", {
                            "stage": "continued" if plan.steps else "empty_done_pending_fresh",
                            "call": counts["planner_calls"], "input_epoch": whole_goal_input_epoch,
                            "task_success": False,
                        })
                    # Reset only after both planner and runner validation pass.
                    if not (network_recovery_active or capacity_recovery_active):
                        planning_failures = 0
                    # Bind information routing to this accepted request. A later
                    # request cannot change the provenance of an existing plan.
                    plan_request_information = request_information
                    planned = snapshot
                    plan_at = repaired_frame.plan_at if repaired_frame is not None else time.perf_counter()
                    active_prepare_repair = repaired_frame
                    recovery_plan = semantic_recovery and repaired_frame is None
                    if repaired_frame is not None:
                        semantic_recovery = invalid_plan_recovery = False
                        counts["prepare_repairs"] += 1
                        emit("agent_prepare_repair", {
                            "status": "prefix_replaced_before_input", "token": repaired_frame.token,
                            "original_deadline_preserved": True, "original_horizon": repaired_frame.horizon,
                            "input_sent": False, "old_prepare_executed": False,
                            "task_success": False,
                        })
                    elif not recovery_plan:
                        prepare_repair_used = False
                    completed, plan_actions = set(), 0
                    property_effects = {}
                    stale_recovery_used = False
                    navigation_seen = {}
                    if navigation_memory is not None:
                        navigation_memory.reset(counts["planner_calls"])
                    recovery_observations = 0
                    completed_conditions = {}
                    expected_values = _control_values(planned)
                    if program_offer is None:
                        emit(
                            "agent_plan",
                            {
                                "call": counts["planner_calls"],
                                "reason": pending_reason,
                                "summary": plan.summary,
                                "reusable": plan.reusable,
                                "max_actions": (
                                    cfg.plan_horizon
                                    if repaired_frame is not None else requested_limit
                                ),
                                **({"requested_repair_actions": 1}
                                   if repaired_frame is not None else {}),
                                "recovery": recovery_plan,
                                **({"model_usage": dict(self.planner.last_usage)}
                                   if type(getattr(self.planner, "last_usage", None))
                                   is dict else {}),
                                "completion_conditions": _condition_facts(
                                    plan.completion_conditions, snapshot,
                                ),
                                "steps": [step.to_dict() for step in plan.steps],
                                **({"segmentation": dict(self.planner.last_plan_segmentation)}
                                   if isinstance(getattr(self.planner, "last_plan_segmentation", None), dict)
                                   else {}),
                                **({"defer_dialog_field_completion": True}
                                   if plan.defer_dialog_field_completion else {}),
                                **({"complete_property_goal": True}
                                   if plan.complete_property_goal else {}),
                            },
                        )
                    if program_offer is not None:
                        active_program = program_offer
                        program_stage_index = 0
                        program_stage_admitted = False
                        program_origin_call = counts["planner_calls"]
                        program_id = f"{program_origin_call}:{active_program.declaration_sha256}"
                        program_receipts = []
                        program_attempt_receipts = []
                        program_verified_effects = {}
                        emit("agent_program_declared", {
                            **program_trace_fields(), "declaration": active_program.to_dict(),
                            "future_stages_are_declarations_not_compiled_steps": True,
                        })
                        emit("agent_program_stage_activated", {
                            **program_trace_fields(), "source": "actual_planner_request",
                            "snapshot_fingerprint": snapshot.fingerprint(),
                            "compiled_steps": [step.to_dict() for step in plan.steps],
                            "real_planner_invocation": True, "freshness_claim": False,
                            "admitted": False, "current_full_entry_admission_pending": True,
                            "future_stages_still_uncompiled": True,
                        })
                    pending_reason = ""
                    if not plan.steps:
                        if plan.done and plan.completion_conditions:
                            # Reobserve after the model call, then check independent
                            # acceptance and every declared goal predicate above.
                            continue
                        # Model completion is never treated as an independent verdict.
                        if success is None:
                            gate = whole_goal_stop_gate()
                            if gate == "review":
                                continue
                            if gate == "blocked":
                                status, reason = "stuck", "whole-goal recheck requires new actual input"
                                break
                            status, reason = (
                                "unverified",
                                "model reported completion; no independent evaluator",
                            )
                            break
                        plan = None
                        pending_reason = "completion_rejected"
                        action_failures += 1
                        history.append({"feedback": "Independent goal checks still fail; continue the task."})
                        if action_failures >= cfg.max_failures:
                            status, reason = "task_failure", "model completion contradicted acceptance checks"
                            break
                        continue
                    began = time.perf_counter()
                    if ((network_recovery_active or capacity_recovery_active)
                            and began - start >= cfg.timeout_s):
                        raise _EpisodeExpired()
                    defer_refresh = (
                        cfg.defer_immediate_plan_refresh and not plan.reusable
                        and not plan.done and len(plan.steps) == 1 and active_program is None
                        and getattr(self.backend, "fresh_preflight_guaranteed", False) is True
                    )
                    if defer_refresh:
                        # Compile only against the original planning frame. The
                        # backend must still acquire and validate a NEW preflight
                        # before input; this frame cannot certify completion.
                        counts["post_plan_observations_deferred"] += 1
                        emit("agent_observation_deferred", {"purpose": "immediate_plan_refresh",
                                                            "fresh_execution_preflight": True})
                    else:
                        snapshot = fresh_observe()
                        counts["observation_ms"] += (time.perf_counter() - began) * 1000
                    if not defer_refresh and not plan.reusable and (
                        snapshot.scope_key() != planned.scope_key()
                        or _control_values(snapshot) != _control_values(planned)
                    ):
                        plan = None
                        pending_reason = "changed_during_planning"
                        if context_recovery_active:
                            status, reason = "stuck", "new recovery context changed during planning"
                            break
                        continue
                if active_program is not None:
                    if snapshot is last_runner_fresh_snapshot:
                        program_frame_origin = "backend.observe"
                        program_frame_source = None
                    boundary = program_boundary(snapshot)
                    if boundary == "review":
                        continue
                    if boundary == "blocked":
                        status, reason = "stuck", "whole-goal recheck requires new actual input"
                        break
                    if boundary in {"abandoned", "retired"}:
                        continue
                if active_prepare_repair is not None and "prepare" not in completed:
                    if active_prepare_repair.context(snapshot, now=time.perf_counter()) is None:
                        emit("agent_prepare_repair", {
                            "status": "cancelled_before_compilation", "input_sent": False,
                            "reason": "original_deadline_or_observed_owner_changed",
                        })
                        active_prepare_repair = None
                        plan = None
                        semantic_recovery = True
                        action_failures += 1
                        pending_reason = "prepare_repair_context_changed"
                        if action_failures >= cfg.max_failures:
                            status, reason = "stuck", "preparation repair context remained unavailable"
                            break
                        continue
                if plan.reusable and finalize_source is None:
                    # Idempotent subgoals can already hold (for example a checked
                    # option). Do not toggle them again just to consume a step.
                    for step in plan.steps:
                        if (
                            step.id not in completed
                            and set(step.depends_on) <= completed
                            and step.postconditions
                            and conditions_hold(step.postconditions, snapshot)
                            and _direct_target_satisfied(step, snapshot)
                        ):
                            completed.add(step.id)
                            counts["condition_skips"] += 1
                            counts["navigation_subgoals"] += step.navigation
                            emit("agent_condition_satisfied", {"step": step.id})
                    if len(completed) == len(plan.steps):
                        if (
                            plan.completion_conditions
                            and conditions_hold(plan.completion_conditions, snapshot)
                        ):
                            # Recheck on the next fresh loop observation, including
                            # independent acceptance if supplied by the caller.
                            continue
                        if context_recovery_active:
                            status, reason = "stuck", "new context recovery produced no input"
                            break
                        if plan_actions == 0:
                            feedback = {
                                "reason": "stage_already_satisfied",
                                "input_sent": False,
                                "task_success": False,
                                "recovery_required": True,
                                "evidence_scope": "current_declared_state_and_direct_target_checks",
                                "skipped_action_execution_proven": False,
                                "skipped_non_idempotent_operations": sorted({
                                    step.operation for step in plan.steps if step.operation in {
                                        "type_text", "hotkey", "key_press", "press", "scroll", "back",
                                    }
                                }),
                                "feedback": (
                                    "No input was sent. Only the declared state predicates and direct "
                                    "target checks currently hold. This does not show that a skipped "
                                    "input ran, that the step's described purpose was achieved, or that "
                                    "the whole user goal is complete. Inspect the remaining work. If "
                                    "final confirmation is still required, explicitly plan its observed "
                                    "control with persistent state guards and an unmet observable effect."
                                ) + (
                                    " A currently true focus predicate does not show where the cursor "
                                    "is or whether text was changed or saved, a page scrolled, or "
                                    "navigation occurred. Choose a different currently grounded strategy "
                                    "or an observable effect that is not already true. Do not repeat the "
                                    "skipped action with the same already-true predicates."
                                    if any(step.operation in {
                                        "type_text", "hotkey", "key_press", "press", "scroll", "back",
                                    } for step in plan.steps) else ""
                                ),
                                "satisfied_steps": [
                                    {"step_id": step.id, "operation": step.operation,
                                     "navigation": step.navigation,
                                     "target": asdict(step.target) if step.target else None}
                                    for step in plan.steps
                                ],
                            }
                            history.append(feedback)
                            emit("agent_stage_satisfied", feedback)
                            # A fully skipped stage needs a different immediate
                            # next action, not another broad reach/edit plan.
                            # Existing budgets and failure counts remain intact.
                            semantic_recovery = True
                        if recovery_plan and not invalid_plan_recovery and plan_actions > 0:
                            semantic_recovery = False
                            history.append({
                                "feedback": "Recovery conditions hold; continue planning from current state.",
                                "recovery_required": False,
                            })
                        plan = None
                        pending_reason = "plan_exhausted"
                        continue
                if active_program is not None and plan_actions >= active_program.max_actions:
                    # A remaining-zero stage may have completed via the pure
                    # predicate path above. A seventh runtime macro is forbidden,
                    # including runtime read-only attempts, just as before.
                    retire_program_budget("horizon_reached")
                    continue
                ready = [
                    step
                    for step in plan.steps
                    if step.id not in completed and set(step.depends_on) <= completed
                ]
                offered: list[ActionCandidate] = []
                compiled_effects: dict[str, AgentStep] = {}
                if finalize_source is not None:
                    ready = [step for step in ready if step is finalize_source[2]]
                for step in ready:
                    if finalize_source is not None:
                        compile_finalize = getattr(self.backend, "compile_declared_numeric_finalize", None)
                        bound = (compile_finalize(*finalize_source, planned, snapshot)
                                 if callable(compile_finalize) else None)
                        if bound is not None:
                            actual, choices = bound
                            offered.extend(choices)
                            compiled_effects.update((c.id, actual) for c in choices)
                        continue
                    if plan.reusable and not conditions_hold(step.preconditions, snapshot):
                        continue
                    if step.property_confirmation_of:
                        from .desktop_property_confirmation import property_confirmation_steps

                        source = property_effects.get(step.property_confirmation_of)
                        if source is not None:
                            for actual in property_confirmation_steps(step, source[0], snapshot):
                                choices = self.backend.compile(actual, planned, snapshot)
                                offered.extend(choices)
                                compiled_effects.update((c.id, actual) for c in choices)
                            if any(c.intent == step.id for c in offered):
                                emit("agent_property_commit_bound", {
                                    "step": step.id, "source_step": step.property_confirmation_of,
                                    "source_receipt_id": source[1],
                                    "candidate_count": sum(c.intent == step.id for c in offered),
                                    "task_success": False,
                                })
                        continue
                    offered.extend(self.backend.compile(step, planned, snapshot))
                if (navigation_arrival_diagnostics_enabled and requested_limit > 1
                        and plan.reusable and len(ready) == 1 and offered
                        and ready[0].navigation and ready[0].property_edit is None
                        and not recovery_plan and not semantic_recovery
                        and not invalid_plan_recovery and not context_recovery_active
                        and not network_recovery_active and not capacity_recovery_active
                        and not unexplained_context_recovery_active
                        and finalize_source is None and numeric_source is None
                        and pending_prepare_repair is None and active_prepare_repair is None):
                    # A diagnostic is not an alias or an input authorization.
                    # Its source must be the actual current full frame, never an
                    # old uploaded request, numeric partial frame or invented read.
                    diagnostic_origin = None
                    diagnostic_source = None
                    if snapshot is last_runner_fresh_snapshot:
                        diagnostic_origin = "backend.observe"
                    elif (cached is not None and snapshot is cached[0]
                          and cache_source in {
                              "verified_step_response", "verified_pending_effect_observation",
                              "verified_ordinary_pending_effect_observation",
                              "stale_rejection_preflight"}):
                        diagnostic_origin = cache_source
                        if cache_source == "verified_step_response":
                            diagnostic_source = source[:2]
                    diagnostic_proof = (
                        _whole_goal_full_observation_proof(
                            self.backend, snapshot, source=diagnostic_source,
                        ) if diagnostic_origin is not None else None
                    )
                    if diagnostic_proof is not None:
                        from .desktop_target_diagnostics import (
                            observed_navigation_arrival_diagnostics,
                        )

                        diagnostic = observed_navigation_arrival_diagnostics(
                            snapshot, ready[0], _condition_facts(ready[0].postconditions, snapshot),
                            tuple(offered),
                        )
                        if (type(diagnostic) is dict
                                and diagnostic.get("code") == "navigation_arrival_caption_mismatch"
                                and _whole_goal_full_observation_proof(
                                    self.backend, snapshot, source=diagnostic_source,
                                    expected=diagnostic_proof,
                                ) is not None):
                            pending_reason = "observed_navigation_arrival_caption_mismatch"
                            emit("agent_observed_navigation_arrival_diagnostic", {
                                "reason": pending_reason, "diagnostic": diagnostic,
                                "source": diagnostic_origin,
                                "native_execution_fingerprint": diagnostic_proof[3],
                                "image_sha256": diagnostic_proof[4],
                                "environment_steps": diagnostic_proof[1],
                                "fresh_observe_calls": diagnostic_proof[2],
                                "offered_candidate_count": len(offered),
                                "planner_calls": counts["planner_calls"],
                                "plan_actions": plan_actions,
                                "input_sent": False, "input_authority": False,
                                "original_arrival_still_unknown": True, "task_success": False,
                            })
                            abandon_program(pending_reason)
                            # No new failure/attempt is charged for this read-only
                            # handoff. Earlier receipts, failures, time and used
                            # recovery tickets remain original and visible.
                            clear_navigation_sources()
                            if navigation_memory is not None:
                                navigation_memory.pending = None
                                navigation_memory.local_attempt_active = False
                            plan = planned = None
                            semantic_recovery = True
                            program_handoff_pending = False
                            history.append({
                                "reason": pending_reason, "recovery_required": True,
                                "input_sent": False, "task_success": False,
                                "observed_navigation_arrival_diagnostic": diagnostic,
                                "feedback": "The exact declared arrival remains unknown. "
                                            "Current legal controls show a caption discrepancy. "
                                            "Inspect their full actual labels and return one "
                                            "grounded action; do not treat this diagnostic "
                                            "as an alias, a verified effect or task completion.",
                            })
                            continue
                if navigation_memory is not None:
                    navigation_memory.current = None
                    stage = ready[0] if len(ready) == 1 and ready[0].navigation else None
                    state = (self.backend.navigation_memory_state(stage, snapshot)
                             if stage is not None and not recovery_plan and not semantic_recovery
                             and not (network_recovery_active or capacity_recovery_active)
                             and not context_recovery_active else None)
                    memory_reason = None
                    if navigation_memory.pending is not None and (
                            not navigation_fresh_observed or stage is None or state is None):
                        memory_reason = "navigation_continuation_evidence_changed"
                    elif stage is not None and state is not None:
                        memory_reason = navigation_memory.bind_offers(stage.id, state, tuple(offered))
                    if memory_reason is not None:
                        navigation_memory.pending = None
                        navigation_memory.local_attempt_active = False
                        clear_navigation_sources()
                        plan = None
                        semantic_recovery = True
                        pending_reason = memory_reason
                        history.append({"reason": memory_reason, "recovery_required": True,
                                        "feedback": "Local navigation evidence is insufficient. Inspect "
                                                    "a real fresh full image and return one grounded action."
                                        })
                        emit("agent_navigation_continuation", {"stage": "replan_required",
                                                               "reason": memory_reason,
                                                               "input_sent": False, "task_success": False})
                        continue
                if network_recovery_active and len(offered) > 1:
                    action_failures += 1
                    emit("agent_network_recovery", {
                        "stage": "failed", "reason": "ambiguous_current_candidate",
                        "candidate_count": len(offered), "task_success": False,
                    })
                    status, reason = "stuck", "network recovery requires one current candidate"
                    break
                if capacity_recovery_active and len(offered) > 1:
                    action_failures += 1
                    if capacity_phase_v3:
                        planning_failures += 1
                        if retry_capacity_plan("ambiguous_current_candidate", accepted_plan=True,
                                               goal_review=whole_goal_request):
                            continue
                    emit("agent_capacity_recovery", {
                        "stage": "failed", "reason": "ambiguous_current_candidate",
                        "candidate_count": len(offered), "task_success": False,
                    })
                    status, reason = "stuck", "capacity recovery requires one current candidate"
                    break
                desktop_targets = [
                    candidate for candidate in offered
                    if cfg.mode == "vlm" and candidate.capability.startswith("agent.osworld.")
                    and isinstance(candidate.arguments.get("bound_control"), dict)
                ]
                ambiguous_targets = desktop_targets if len({
                    json.dumps(candidate.arguments["bound_control"], sort_keys=True)
                    for candidate in desktop_targets
                }) > 1 else []
                if ambiguous_targets:
                    # A new matching widget can appear while the VLM is thinking.
                    # Direct selection may choose a route, never an unchosen
                    # physical desktop target. Hybrid still chooses among targets.
                    offered = []
                if len(offered) > 255:
                    semantic_recovery = semantic_recovery or plan.reusable
                    history.append({
                        "feedback": "Too many matching controls; narrow targets by name, app or window.",
                        "recovery_required": semantic_recovery,
                    })
                    action_failures += 1
                    plan = None
                    pending_reason = "candidate_budget_exceeded"
                    if action_failures >= cfg.max_failures:
                        status, reason = "stuck", "semantic targets remain too broad"
                        break
                    continue
                if not offered:
                    if cfg.adaptive_dispatch and getattr(self.planner, "profile", None) == "browser":
                        # One fresh read can bind a newly appeared control without
                        # another VLM call. A persistent failure needs bounded H1
                        # repair, not repeated copies of the same multi-step plan.
                        post_snapshot_source = recovery_planning_snapshot_source = None
                        stale_snapshot_source = None
                        if not dispatch_refresh_attempted:
                            dispatch_refresh_attempted = True
                            dispatch_refreshes += 1
                            emit("agent_dispatch", {"route": "observe", "reason": "binding_unavailable",
                                                   "input_sent": False})
                            continue
                        dispatch_failure_streak += 1
                        if dispatch_failure_streak >= cfg.max_failures:
                            status, reason = "stuck", "dispatch binding recovery exhausted"
                            break
                        diagnostics = []
                        for step in ready:
                            original = [c for c in planned.controls if c.ref == step.ref]
                            diagnostics.append({
                                "step_id": step.id, "operation": step.operation,
                                "planning_ref_count": len(original),
                                "current_identity_matches": (
                                    sum(c.identity() == original[0].identity() for c in snapshot.controls)
                                    if len(original) == 1 else None),
                            })
                        history.append({
                            "reason": "binding_unavailable", "input_sent": False,
                            "recovery_required": True, "binding_diagnostics": diagnostics,
                            "feedback": "No executable binding after a fresh observation. A ref does not "
                                        "resolve duplicate control identities. Choose one currently grounded "
                                        "immediate action; a visible unambiguous target may use a visual "
                                        "action. Do not repeat the same unavailable multi-step plan.",
                        })
                        dispatch_s2_replans += 1
                        plan = planned = None
                        semantic_recovery = True
                        pending_reason = "dispatch_binding_unavailable"
                        emit("agent_dispatch", {"route": "vlm", "reason": "binding_unavailable",
                                               "input_sent": False, "requested_horizon": 1})
                        continue
                    if finalize_source is not None:
                        # A failed single-key continuation must never fall
                        # through a later loop into the original fill stage.
                        action_failures += 1
                        semantic_recovery = True
                        plan = None
                        pending_reason = "declared_numeric_finalize_unavailable"
                        feedback = {
                            "reason": pending_reason, "input_sent": False,
                            "verified": False, "task_success": False, "recovery_required": True,
                            "feedback": "The original numeric field, buffer or receipt binding changed. "
                                        "No declared key was sent. Inspect fresh state with one model "
                                        "recovery; do not repeat the original fill implicitly.",
                        }
                        history.append(feedback)
                        emit("agent_recovery", feedback)
                        if action_failures >= cfg.max_failures:
                            status, reason = "stuck", "declared numeric finalization became unavailable"
                            break
                        continue
                    if active_program is not None:
                        original_observe_eligible = bool(
                            not ambiguous_targets and plan.reusable
                            and any(step.recovery == "observe" for step in ready)
                            and recovery_observations < 2
                        )
                        stage_observe_permitted = (
                            active_program.stages[program_stage_index].recovery == "observe"
                        )
                        optional_observe_permitted = (
                            original_observe_eligible and stage_observe_permitted
                        )
                        emit("agent_program_recovery_policy", {
                            **program_trace_fields(), "phase": "awaiting_conditions",
                            "stage_recovery": active_program.stages[program_stage_index].recovery,
                            "original_observe_eligible": original_observe_eligible,
                            "original_ready_step_recoveries": [step.recovery for step in ready],
                            "optional_observe_permitted": optional_observe_permitted,
                            "selected_original_branch": (
                                "bounded_observe" if optional_observe_permitted
                                else "original_nonobserve_path"
                            ),
                            "recovery_observations_before": recovery_observations,
                            "native_observation_calls_at_decision": getattr(
                                self.backend, "fresh_observe_calls", None,
                            ),
                            "event_observation_calls": 0,
                            "original_observation_limits_and_failure_charges_unchanged": True,
                        })
                    if (
                        not ambiguous_targets and plan.reusable
                        and any(step.recovery == "observe" for step in ready)
                        and recovery_observations < 2
                        and (active_program is None
                             or active_program.stages[program_stage_index].recovery == "observe")
                    ):
                        recovery_observations += 1
                        counts["recovery_observations"] += 1
                        emit("agent_recovery", {
                            "reason": "awaiting_conditions", "attempt": recovery_observations
                        })
                        time.sleep(min(0.25, max(0, cfg.timeout_s - (time.perf_counter() - start))))
                        continue
                    action_failures += 1
                    if plan.reusable or ambiguous_targets:
                        diagnostics = [_unavailable_step_facts(step, snapshot) for step in ready]
                        from .desktop_compile_diagnostics import unavailable_visual_facts

                        for step, diagnostic in zip(ready, diagnostics, strict=True):
                            comparison = unavailable_visual_facts(step, planned, snapshot)
                            if comparison is not None:
                                diagnostic["visual_comparison"] = comparison
                        if ambiguous_targets:
                            pending_reason = "ambiguous_target"
                            for diagnostic in diagnostics:
                                diagnostic.update(
                                    reason=pending_reason,
                                    candidate_count=sum(c.intent == diagnostic["step_id"]
                                                        for c in ambiguous_targets),
                                )
                        else:
                            pending_reason = (
                                "preconditions_unmet"
                                if diagnostics and all(
                                    d["reason"] == "preconditions_unmet" for d in diagnostics
                                )
                                else "no_executable_target"
                            )
                        semantic_recovery = True
                        feedback = {
                            "reason": pending_reason,
                            "feedback": (
                                "No input was sent. Multiple current desktop controls match the action. "
                                "Inspect the current screen and choose one uniquely grounded target using "
                                "its name, app, window or ancestor context."
                                if ambiguous_targets else
                                "No input was sent. Inspect current matching facts and choose one grounded "
                                "recovery action; do not repeat an unavailable semantic query."
                            ),
                            "ready_steps": diagnostics,
                            "recovery_required": True,
                        }
                        if (cfg.mode == "hybrid" and not ambiguous_targets
                                and getattr(self.planner, "desktop_contract", None)
                                in {"property_intent_v33", "property_intent_v49", "property_intent_v51",
                                    "property_intent_v52"}
                                and callable(getattr(self.planner, "set_prepare_repair_context", None))
                                and not prepare_repair_used and not recovery_plan
                                and not (network_recovery_active or capacity_recovery_active)
                                and not context_recovery_active
                                and not stale_recovery_used and plan_actions == 0
                                and not completed and not property_effects
                                and len(ready) == 1 and ready[0] is plan.steps[0]
                                and pending_reason == "no_executable_target"
                                and action_failures < cfg.max_failures):
                            from .desktop_plan_repair import suspend_property_prepare

                            pending_prepare_repair = suspend_property_prepare(
                                plan, planned, snapshot, plan_at=plan_at, ttl_s=cfg.plan_ttl_s,
                                horizon=cfg.plan_horizon, now=time.perf_counter(),
                            )
                            if pending_prepare_repair is not None:
                                prepare_repair_used = True
                                feedback["prepare_repair_offered"] = True
                                emit("agent_prepare_repair", {
                                    "status": "offered", "token": pending_prepare_repair.token,
                                    "input_sent": False, "old_prepare_executed": False,
                                    "original_deadline": pending_prepare_repair.deadline,
                                    "task_success": False,
                                })
                        history.append(feedback)
                        emit("agent_recovery", feedback)
                    else:
                        pending_reason = "target_unavailable"
                        if (v52_information_routing and plan_request_information is not None
                                and plan_request_information["planning_observation"] == "native_first_v51"
                                and plan_request_information["image_uploaded"] is False
                                and plan_request_information["a11y_complete"] is True
                                and plan_request_information["max_actions"] > 1):
                            # Preserve the existing waits and failure charge;
                            # only the next request's information route changes.
                            semantic_recovery = True
                            emit("agent_information_recovery", {
                                "reason": "native_atomic_target_unavailable", "input_sent": False,
                                "planner_calls": counts["planner_calls"], "next_max_actions": 1,
                                "task_success": False,
                            })
                        history.append({
                            "feedback": "No current legal target matches; inspect the current observation.",
                            "recovery_required": semantic_recovery,
                        })
                    if capacity_phase_v3 and capacity_recovery_active:
                        planning_failures += 1
                        if retry_capacity_plan("no_current_candidate", accepted_plan=True,
                                               goal_review=whole_goal_request):
                            continue
                    plan = None
                    if action_failures >= cfg.max_failures:
                        status, reason = "stuck", "no freshly grounded action"
                        break
                    if (capacity_phase_v3 and capacity_recovery_active
                            and planning_failures >= cfg.max_failures):
                        status, reason = "planner_error", "consecutive planning failures"
                        break
                    continue
                # The remote planner can take long enough for the live state to change.
                # Backends recheck freshness immediately before execution as well.
                if time.perf_counter() - start >= cfg.timeout_s:
                    status, reason = "timeout", "time budget exhausted before action"
                    break
                candidate_counts.append(len(offered))
                distinct_target_counts.append(_distinct_bound_target_count(offered, snapshot))
                counts["multi_candidate_steps"] += len(offered) > 1
                reusable_plan = plan.reusable
                observation = Observation(
                    goal,
                    plan.summary,
                    {
                        "url": snapshot.url,
                        "title": snapshot.title,
                        "controls": [
                            {
                                "ref": c.ref,
                                "label": c.label,
                                "tag": c.tag,
                                "disabled": c.disabled,
                                "checked": c.checked,
                                "has_value": bool(c.value),
                                "role": c.role,
                                "app": c.app,
                                "window": c.window,
                                "ancestors": list(c.ancestors),
                                "focused": c.focused,
                                "selected": c.selected,
                                "expanded": c.expanded,
                                "editable": c.editable,
                                "active": c.active,
                            }
                            for c in (self.backend.policy_controls(snapshot, offered)
                                      if callable(getattr(self.backend, "policy_controls", None))
                                      else snapshot.controls)
                        ],
                        "completed_steps": sorted(completed),
                        "a11y_complete": snapshot.a11y_complete,
                        "candidate_scope_complete": (
                            self.backend.candidate_scope_complete(offered)
                            if callable(getattr(self.backend, "candidate_scope_complete", None)) else False
                        ),
                        "goal_completion_conditions": (
                            [] if key_only_observation else
                            _condition_facts(plan.completion_conditions, snapshot)
                        ),
                        "recent_actions": history[-6:],
                        **({"navigation_memory": navigation_memory.public()}
                           if navigation_memory is not None
                           and navigation_memory.current is not None else {}),
                        **({"explicit_property_confirmation": True} if compiled_effects else {}),
                        "ready_steps": [
                            {
                                "id": s.id,
                                "operation": s.operation,
                                "label": s.label,
                                "navigation": s.navigation,
                                "target": asdict(s.target) if s.target else None,
                                "conditions": _condition_facts(s.preconditions, snapshot),
                                "completion_conditions": _condition_facts(s.postconditions, snapshot),
                                **({"property_goal": {"operation": s.property_edit.operation,
                                                       "has_value": bool(s.property_edit.value),
                                                       **({"bind_boolean_states": True}
                                                          if s.property_edit.bind_boolean_states
                                                          and s.property_edit.value.casefold()
                                                          in {"on", "off"} else {})}}
                                   if s.property_edit is not None else {}),
                            }
                            for s in ready
                        ] if reusable_plan else [],
                    },
                    source="jev-vlm-agent",
                )
                if navigation_memory is not None:
                    navigation_memory.bind_observation(observation.observation_id)
                try:
                    selected_runtime = (capacity_runtime if capacity_recovery_active else
                                        recovery_runtime if network_recovery_active else runtime)
                    step_result = selected_runtime.step(observation, offered)
                    dispatch_refresh_attempted = False
                except DispatchEscalation as exc:
                    # No input has been sent. Discard cached observation authority.
                    clear_navigation_sources()
                    source = recovery_planning_source = stale_source = None
                    numeric_source = finalize_source = cached = None
                    numeric_planning_frame = numeric_finalize_frame = recovery_planning_frame = None
                    if not dispatch_refresh_attempted:
                        dispatch_refresh_attempted = True
                        dispatch_refreshes += 1
                        emit("agent_dispatch", {"route": "observe", "reason": exc.reason,
                                               "input_sent": False})
                        continue
                    dispatch_failure_streak += 1
                    if dispatch_failure_streak >= cfg.max_failures:
                        status, reason = "stuck", "dispatch evidence recovery exhausted"
                        break
                    dispatch_s2_replans += 1
                    plan = planned = None
                    semantic_recovery = True
                    pending_reason = "dispatch_" + exc.reason
                    history.append({"feedback": "Current structured evidence remains incomplete after "
                                    "a fresh observation. Ground one immediate action using current "
                                    "available evidence; do not assume missing state is false."})
                    emit("agent_dispatch", {"route": "vlm", "reason": exc.reason,
                                           "input_sent": False, "requested_horizon": 1})
                    continue
                except DecisionAdmissionRejected as exc:
                    # The real Jev decision/time is already recorded. No receipt/input exists.
                    action_failures += 1
                    clear_navigation_sources()
                    navigation_memory.pending = None
                    navigation_memory.local_attempt_active = False
                    plan = None
                    semantic_recovery = True
                    invalid_plan_recovery = recovery_plan = False
                    pending_reason = "observed_navigation_edge_exhausted"
                    history.append({"reason": pending_reason, "input_sent": False,
                                    "recovery_required": True, "verified": False,
                                    "feedback": "The selected edge is an already observed part of this "
                                                "unchanged local loop. No new input was sent. Inspect "
                                                "a fresh full image and choose one grounded recovery action."
                                    })
                    emit("agent_navigation_continuation", {
                        "stage": "selection_declined", "reason": exc.rejection.reason,
                        "input_sent": False, "action_failures": action_failures, "task_success": False,
                    })
                    if action_failures >= cfg.max_failures:
                        status, reason = "stuck", "consecutive action failures"
                        break
                    continue
                except PolicySelectionFailure as exc:
                    # AgentRuntime raises this before decision/guard/execution.
                    # Keep its original policy_failure event/hash/timing unchanged.
                    failure = exc.failure
                    counts["jev_network_failures"] += failure.network_retryable
                    capacity_failure = _is_jev_capacity_failure(failure)
                    if cfg.jev_capacity_recovery:
                        counts["jev_capacity_failures"] += capacity_failure
                    network_eligible = (
                        cfg.jev_network_recovery and cfg.mode == "hybrid"
                        and getattr(self.planner, "profile", None) == "desktop"
                        and failure.network_retryable and not counts["network_recovery_attempts"]
                        and not context_recovery_active and not selection_recovery_ticket_used
                    )
                    capacity_eligible = (
                        cfg.jev_capacity_recovery and cfg.mode == "hybrid" and capacity_failure
                        and getattr(self.planner, "profile", None) == "desktop"
                        and not context_recovery_active and not selection_recovery_ticket_used
                        and action_failures < cfg.max_failures and planning_failures < cfg.max_failures
                    )
                    if not (network_eligible or capacity_eligible):
                        raise
                    if time.perf_counter() - start >= cfg.timeout_s:
                        raise _EpisodeExpired() from None
                    if counts["planner_calls"] >= cfg.max_plans:
                        category = "capacity" if capacity_eligible else "network"
                        status = "max_plans"
                        reason = "planning budget exhausted after Jev " + category + " failure"
                        break
                    selection_recovery_ticket_used = True
                    clear_input_evidence = getattr(self.backend, "clear_input_observation_evidence", None)
                    if callable(clear_input_evidence):
                        clear_input_evidence()
                    if capacity_eligible:
                        counts["capacity_recovery_attempts"] += 1
                        capacity_recovery_active = semantic_recovery = True
                        capacity_recovery_planned = False
                        capacity_recovery_fresh_required = True
                        # Invalidate plan/decision/observation/repair authority.
                        # Do not reset failures, counts, used repair/stale tickets,
                        # start, cfg budgets, metered Jev time, or original failure.
                        plan = planned = None
                        invalid_plan_recovery = recovery_plan = False
                        completed.clear()
                        property_effects.clear()
                        completed_conditions.clear()
                        expected_values.clear()
                        navigation_seen.clear()
                        offered.clear()
                        compiled_effects.clear()
                        post_snapshot_source = recovery_planning_snapshot_source = None
                        pending_numeric_planning_source = pending_numeric_finalize_source = None
                        stale_snapshot_source = None
                        source = recovery_planning_source = stale_source = None
                        numeric_source = finalize_source = cached = None
                        numeric_planning_frame = numeric_finalize_frame = recovery_planning_frame = None
                        pending_prepare_repair = active_prepare_repair = None
                        context_recovery_pending = None
                        visual_information_anchor = plan_request_information = None
                        clear_repair = getattr(self.planner, "set_prepare_repair_context", None)
                        if callable(clear_repair):
                            clear_repair(None)
                        pending_reason = "jev_capacity_recovery"
                        feedback = {
                            "reason": pending_reason, "recovery_required": True,
                            "feedback": "Jev refused the selection request before any GUI input because "
                                        "its capacity was exceeded. Inspect a new fresh public observation "
                                        "and return one uniquely grounded immediate action. Do not resend "
                                        "the failed request or claim any input/effect. One shared Jev "
                                        "selection-recovery ticket is allowed in this episode.",
                        }
                        history.append(feedback)
                        emit("agent_capacity_recovery", {
                            "stage": "requested", "policy_failure": failure.to_dict(),
                            **({"policy": cfg.jev_capacity_recovery_policy} if capacity_phase else {}),
                            "episode_recovery_limit": 1, "shared_with_network_recovery": True,
                            "input_sent": False, "task_success": False,
                        })
                        continue
                    counts["network_recovery_attempts"] += 1
                    network_recovery_active = semantic_recovery = True
                    network_recovery_planned = False
                    plan = None
                    post_snapshot_source = None
                    pending_reason = "jev_network_recovery"
                    feedback = {
                        "reason": pending_reason, "recovery_required": True,
                        "feedback": "Jev selection failed before any input. Inspect the fresh public "
                                    "observation and return one uniquely grounded immediate action. "
                                    "No old candidate was selected. Only one network recovery is allowed.",
                    }
                    history.append(feedback)
                    emit("agent_network_recovery", {
                        "stage": "requested", "policy_failure": failure.summary(),
                        "episode_recovery_limit": 1, "task_success": False,
                    })
                    continue
                except _PlanExpired:
                    plan = None
                    if finalize_source is not None:
                        semantic_recovery = True
                    pending_reason = "plan_expired_before_action"
                    emit("agent_recovery", {"reason": pending_reason})
                    continue
                counts["steps"] += 1
                counts["network_recovery_actions"] += network_recovery_active
                input_steps = step_result.receipt.output.get("environment_steps")
                if (unexplained_planning_reuse_enabled and type(input_steps) is int and input_steps > 0):
                    unexplained_planning_input_epoch += 1
                if (cfg.whole_goal_recheck and type(input_steps) is int and input_steps > 0):
                    whole_goal_input_epoch += 1
                counts["capacity_recovery_actions"] += bool(
                    capacity_recovery_active and type(input_steps) is int and input_steps > 0
                )
                counts["backend_ms"] += step_result.receipt.duration_ms
                timings = step_result.receipt.output.get("timings_ms", {})
                counts["action_ms"] += timings.get("action", 0)
                counts["effect_ms"] += timings.get("effect_check", 0)
                chosen = next(c for c in offered if c.id == step_result.decision.candidate_id)
                selected_template = next(s for s in plan.steps if s.id == chosen.intent)
                selected_step = compiled_effects.get(chosen.id, selected_template)
                # A property template is navigation until the policy selects
                # an actual field. Its receipt-bound field carries the real
                # effects, including the existing bounded observation wait.
                # This lookup is reporting evidence, never input permission.
                effect_step = selected_step
                if selected_step.property_edit is not None:
                    bound_step = getattr(self.backend, "bound_property_step", None)
                    if callable(bound_step):
                        actual = bound_step(chosen, step_result.receipt)
                        if isinstance(actual, AgentStep):
                            effect_step = actual
                routes[str(chosen.arguments.get("route", chosen.channel))] += 1
                verified = step_result.receipt.success and step_result.verification.passed
                final_verification_details = step_result.verification.details
                verified = verified and not _unconfirmed_application_wait(final_verification_details)
                if (finalize_source is not None and verified
                        and chosen.arguments.get("declared_numeric_finalize") is True):
                    bound_step = getattr(self.backend, "bound_property_step", None)
                    actual = bound_step(chosen, step_result.receipt) if callable(bound_step) else None
                    if isinstance(actual, AgentStep):
                        # This is the new key's verified field effect. The
                        # original fill receipt remains unverified forever.
                        selected_step, effect_step = selected_template, actual
                        emit("agent_declared_numeric_finalize", {
                            "status": "committed", "source_receipt_id": finalize_source[1].receipt_id,
                            "receipt_id": step_result.receipt.receipt_id,
                            "key": finalize_source[2].property_edit.numeric_finalize_key,
                            "verified": True, "task_success": False,
                        })
                    else:
                        verified = False
                pending_verifier = getattr(self.backend, "verify_pending", None)
                navigation_pending = getattr(self.backend, "property_navigation_effect_pending", None)
                property_navigation_wait = bool(
                    selected_step.property_edit is not None and effect_step.navigation
                    and callable(navigation_pending)
                    and navigation_pending(chosen, step_result.receipt) is True
                )
                numeric_replan = getattr(self.backend, "pending_numeric_replan", None)
                eager_numeric_replan = bool(
                    not verified and reusable_plan and step_result.receipt.success
                    and effect_step.operation == "fill" and callable(numeric_replan)
                    and numeric_replan(chosen, step_result.receipt) is True
                )
                declared_pending = getattr(self.backend, "declared_numeric_finalize_pending", None)
                declared_finalize_requested = bool(
                    eager_numeric_replan and cfg.mode == "hybrid"
                    and selected_template.property_edit is not None
                    and selected_step is selected_template and not recovery_plan and not semantic_recovery
                    and not (network_recovery_active or capacity_recovery_active)
                    and not context_recovery_active
                    and plan_actions + 1 < cfg.plan_horizon and counts["steps"] < cfg.max_steps
                    and action_failures + 1 < cfg.max_failures
                    and time.perf_counter() - start < cfg.timeout_s
                    and time.perf_counter() - plan_at < cfg.plan_ttl_s
                    and callable(declared_pending)
                    and declared_pending(chosen, step_result.receipt, selected_template) is True
                )
                if eager_numeric_replan and not declared_finalize_requested:
                    emit("agent_pending_numeric_replan", {
                        "receipt_id": step_result.receipt.receipt_id,
                        "verified": False, "task_success": False,
                        "reason": "original_numeric_buffer_pending_commit",
                        "next": "request_one_model_recovery_without_effect_polling",
                    })
                application_wait_summary = None
                pending_recovery = False
                if active_program is not None:
                    original_observe_eligible = bool(
                        not verified and reusable_plan and step_result.receipt.success
                        and not eager_numeric_replan
                        and finalize_source is None
                        and (effect_step.postconditions or property_navigation_wait)
                        and effect_step.recovery == "observe"
                        and callable(pending_verifier)
                    )
                    stage_observe_permitted = (
                        active_program.stages[program_stage_index].recovery == "observe"
                    )
                    optional_observe_permitted = (
                        original_observe_eligible and stage_observe_permitted
                    )
                    emit("agent_program_recovery_policy", {
                        **program_trace_fields(), "phase": "post_input_pending",
                        "stage_recovery": active_program.stages[program_stage_index].recovery,
                        "original_effect_recovery": effect_step.recovery,
                        "original_observe_eligible": original_observe_eligible,
                        "optional_observe_permitted": optional_observe_permitted,
                        "selected_original_branch": (
                            "bounded_observe" if optional_observe_permitted
                            else "original_nonobserve_path"
                        ),
                        "source_receipt_id": step_result.receipt.receipt_id,
                        "receipt_final_verified_at_decision": bool(verified),
                        "native_observation_calls_at_decision": getattr(
                            self.backend, "fresh_observe_calls", None,
                        ),
                        "event_observation_calls": 0,
                        "original_observation_limits_and_failure_charges_unchanged": True,
                        "declared_numeric_contingency_independent": bool(
                            declared_finalize_requested,
                        ),
                    })
                if (
                    not verified and reusable_plan and step_result.receipt.success
                    and not eager_numeric_replan
                    and finalize_source is None
                    and (effect_step.postconditions or property_navigation_wait)
                    and effect_step.recovery == "observe"
                    and callable(pending_verifier)
                    and (active_program is None
                         or active_program.stages[program_stage_index].recovery == "observe")
                ):
                    pending_recovery = True
                    launch = final_verification_details.get("application_wait", {})
                    extended_wait = (not property_navigation_wait and isinstance(launch, dict)
                                     and launch.get("eligible") is True)
                    receipt_ended = time.perf_counter() - max(
                        0.0, time.time() - step_result.receipt.ended_at
                    )
                    deadlines = [(start + cfg.timeout_s, "episode_timeout"),
                                 (plan_at + cfg.plan_ttl_s, "plan_expired")]
                    if extended_wait:
                        deadlines.append((receipt_ended + 60, "application_wait_timeout"))
                    deadline, deadline_reason = min(deadlines)
                    attempts = 0
                    wait_reason = "observation_budget_exhausted"
                    # This is an observation-start budget. A synchronous transport
                    # may return late; check again before trusting its result or
                    # allowing the old plan to send any further input.
                    for attempt in range(20 if extended_wait else 2):
                        current_launch = final_verification_details.get("application_wait", {})
                        if current_launch and not current_launch.get("eligible"):
                            wait_reason = current_launch["reason"]
                            break
                        if extended_wait and not current_launch:
                            wait_reason = "application_wait_no_longer_supported"
                            break
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            wait_reason = deadline_reason
                            break
                        time.sleep(min(1.0 if extended_wait else 0.25, remaining))
                        if time.perf_counter() >= deadline:
                            wait_reason = deadline_reason
                            break
                        began = time.perf_counter()
                        pending_verification = pending_verifier(chosen, step_result.receipt)
                        elapsed = (time.perf_counter() - began) * 1000
                        counts["backend_ms"] += elapsed
                        counts["effect_ms"] += elapsed
                        counts["recovery_observations"] += 1
                        attempts += 1
                        final_verification_details = pending_verification.details
                        expired = time.perf_counter() >= deadline
                        pending_accepted = (
                            pending_verification.passed
                            and not _unconfirmed_application_wait(final_verification_details)
                        )
                        pending_provenance = ({
                            "property_navigation_wait": True,
                            "receipt_id": step_result.receipt.receipt_id,
                            "observation_source": final_verification_details.get("observation_source"),
                            "pending_observation_fingerprint": final_verification_details.get(
                                "pending_observation_fingerprint"),
                        } if property_navigation_wait else {})
                        emit("agent_pending_effect", {"step": selected_step.id, "attempt": attempt + 1,
                                                      "passed": pending_accepted and not expired,
                                                      "deadline_exceeded": expired,
                                                      **pending_provenance,
                                                      **({"selected_property_effect": True,
                                                          "effect_operation": effect_step.operation}
                                                         if effect_step is not selected_step else {})})
                        if expired:
                            wait_reason = deadline_reason
                            break
                        if pending_accepted:
                            verified = True
                            wait_reason = "effect_confirmed"
                            break
                        current_launch = final_verification_details.get("application_wait", {})
                        if current_launch and not current_launch.get("eligible"):
                            wait_reason = current_launch["reason"]
                            break
                        if extended_wait and not current_launch:
                            wait_reason = "application_wait_no_longer_supported"
                            break
                    if launch:
                        application_wait_summary = {
                            "reason": wait_reason, "attempts": attempts,
                            "observation_start_budget_s": 60 if extended_wait else 0,
                            "max_observations": 20 if extended_wait else 2,
                            "elapsed_ms": max(0, (time.perf_counter() - receipt_ended) * 1000),
                            "deadline_overrun_ms": max(0, (time.perf_counter() - deadline) * 1000),
                            "deadline_constraint": deadline_reason,
                            "task_success": False,
                        }
                        emit("agent_application_wait", {"step": selected_step.id,
                                                        **application_wait_summary})
                if active_prepare_repair is not None and selected_step.id == "prepare" and verified:
                    input_count = step_result.receipt.output.get("environment_steps")
                    if type(input_count) is not int or input_count <= 0:
                        verified = False
                        emit("agent_prepare_repair", {
                            "status": "prefix_input_unconfirmed", "task_success": False,
                            "receipt_id": step_result.receipt.receipt_id,
                        })
                navigation_complete = bool(final_verification_details.get(
                    "postconditions_met", step_result.receipt.output.get("postconditions_met")
                ))
                navigation_cycle = False
                if selected_step.navigation and verified:
                    before_scope = final_verification_details.get("before_scope")
                    after_scope = final_verification_details.get("after_scope")
                    seen = navigation_seen.setdefault(selected_step.id, set())
                    if isinstance(before_scope, str):
                        seen.add(before_scope)
                    if isinstance(after_scope, str):
                        navigation_cycle = after_scope in seen and not navigation_complete
                        seen.add(after_scope)
                    if navigation_cycle:
                        verified = False
                navigation_witness = None
                navigation_followup_unknown = False
                # Qualify before ordinary failed-effect feedback latches semantic recovery.
                # A genuine recovery plan can never earn local navigation continuation.
                navigation_continuation_eligible = bool(
                    navigation_memory is not None and selected_step.navigation and effect_step.navigation
                    and not recovery_plan and not semantic_recovery
                    and not program_handoff_pending
                    and not (network_recovery_active or capacity_recovery_active)
                    and not context_recovery_active
                )
                if navigation_memory is not None:
                    local_attempt = navigation_memory.local_attempt_active
                    navigation_memory.local_attempt_active = False
                    if navigation_continuation_eligible:
                        transition = self.backend.navigation_transition_evidence(chosen, step_result.receipt)
                        if isinstance(transition, NavigationTransition):
                            known_after = navigation_memory.known.get(
                                (transition.stage, transition.after.native)
                            )
                            navigation_followup_unknown = bool(
                                local_attempt and verified and not navigation_complete
                                and (known_after is None
                                     or not known_after.same_observable_state(transition.after)
                                     or transition.before.owner != transition.after.owner))
                            navigation_witness = navigation_memory.record(
                                transition, final_verified=bool(verified), cycle=navigation_cycle,
                            )
                            emit("agent_navigation_transition", {
                                "stage": transition.stage, "receipt_id": transition.receipt_id,
                                "candidate_id": chosen.id,
                                "before_native_fingerprint": transition.before.native,
                                "after_native_fingerprint": transition.after.native,
                                "backend_effect": transition.backend_verified,
                                "final_verified": bool(verified),
                                "arrival_outcomes": list(transition.arrival), "cycle": navigation_cycle,
                                "input_authority": False, "task_success": False,
                            })
                        elif local_attempt and verified and not navigation_complete:
                            navigation_followup_unknown = True
                stale_recovery = False
                consume_stale = getattr(self.backend, "consume_stale_rejection", None)
                if (
                    not verified and cfg.mode == "hybrid" and reusable_plan
                    and finalize_source is None
                    and not recovery_plan and not semantic_recovery
                    and not (network_recovery_active or capacity_recovery_active)
                    and not context_recovery_active and not stale_recovery_used
                    and counts["steps"] < cfg.max_steps and action_failures + 1 < cfg.max_failures
                    and time.perf_counter() - start < cfg.timeout_s
                    and time.perf_counter() - plan_at < cfg.plan_ttl_s
                    and callable(consume_stale)
                ):
                    stale_recovery = consume_stale(chosen, step_result.receipt) is True
                if active_program is not None:
                    program_attempt_receipts.append({
                        "receipt_id": step_result.receipt.receipt_id,
                        "environment_steps": step_result.receipt.output.get("environment_steps"),
                        "final_verified": bool(verified),
                        "declared_numeric_contingency": bool(declared_finalize_requested),
                        "bounded_stale_refresh": bool(stale_recovery),
                    })
                    # Preserve the original explained same-stage contingencies:
                    # pending numeric key, bounded stale refresh and a proven
                    # known-repeat concession keep their original false receipt
                    # and charges. The original plan=None / H1 / terminal path
                    # retires the program, never advances an unverified stage.
                action_feedback = {
                    "action": chosen.description,
                    "verified": verified,
                    "feedback": "effect checked" if verified else "action failed; reassess live state",
                    "recovery_required": not verified and not stale_recovery
                    and (reusable_plan or semantic_recovery),
                }
                focus_feedback = _text_focus_guard_feedback(step_result.receipt) if not verified else None
                if focus_feedback is not None:
                    action_feedback.update(focus_feedback)
                if stale_recovery:
                    action_feedback.update(
                        reason="zero_input_stale_observation",
                        feedback=(
                            "The backend rejected an old observation before sending input. Keep this "
                            "stage for one fresh observation, candidate compilation and policy choice; "
                            "the refused action is unverified and consumes the original attempt budget."
                        ),
                    )
                if (verified and selected_step.property_edit is not None and navigation_complete
                        and (effect_step.operation in {"fill", "check", "uncheck"}
                             or (selected_step.property_edit.operation == "set"
                                 and not effect_step.navigation and effect_step.operation == "click"
                                 and any(c.kind == "selected" for c in effect_step.postconditions)))):
                    action_feedback["property_stage_result"] = {
                        "status": "selected_field_verified",
                        "operation": effect_step.operation,
                        "target": asdict(effect_step.target) if effect_step.target else None,
                        "whole_goal_proven": False,
                        "next": (
                            "Do not repeat this property stage. Inspect the current UI: if the whole "
                            "goal is complete, declare complete with its evidence; if Save/Apply/Confirm "
                            "remains, use current edit_fields with the already-correct field guarding "
                            "one explicit commit and its observed result conditions."
                        ),
                    }
                    has_confirmation = any(s.property_confirmation_of == selected_step.id for s in plan.steps)
                    if effect_step is not selected_step:
                        property_effects[selected_step.id] = (effect_step, step_result.receipt.receipt_id)
                    if has_confirmation:
                        action_feedback["property_stage_result"]["next"] = (
                            "Continue the explicitly declared confirmation phase in this same plan; "
                            "only current owning-form buttons with fresh field guards may be offered."
                        )
                    emit("agent_property_stage", action_feedback["property_stage_result"])
                    if (plan.complete_property_goal and effect_step is not selected_step
                            and not has_confirmation):
                        # The backend returns this actual field step only for the
                        # selected candidate's verified receipt. Check the bound
                        # effects again on the next fresh observation above.
                        completion = tuple(dict.fromkeys((
                            *plan.completion_conditions, *effect_step.postconditions,
                        )))
                        if len(completion) <= 8:
                            plan = replace(plan, completion_conditions=completion)
                            emit("agent_property_goal_bound", {
                                "step": selected_step.id,
                                "receipt_id": step_result.receipt.receipt_id,
                                "conditions": [asdict(c) for c in effect_step.postconditions],
                                "explicit_conditions_retained": True,
                                "task_success": False,
                            })
                if (verified and selected_template.property_confirmation_of
                        and chosen.id in compiled_effects
                        and type(step_result.receipt.output.get("environment_steps")) is int
                        and step_result.receipt.output["environment_steps"] > 0):
                    emit("agent_property_commit", {
                        "step": selected_template.id,
                        "source_step": selected_template.property_confirmation_of,
                        "target": asdict(selected_step.target),
                        "receipt_id": step_result.receipt.receipt_id,
                        "verified": True, "task_success": False,
                    })
                    if plan.complete_property_goal:
                        completion = tuple(dict.fromkeys((
                            *plan.completion_conditions, *selected_step.postconditions,
                        )))
                        if len(completion) <= 8:
                            plan = replace(plan, completion_conditions=completion)
                            emit("agent_property_goal_bound", {
                                "step": selected_template.id,
                                "receipt_id": step_result.receipt.receipt_id,
                                "conditions": [asdict(c) for c in selected_step.postconditions],
                                "explicit_conditions_retained": True, "task_success": False,
                            })
                if declared_finalize_requested:
                    action_feedback.update(
                        reason="declared_numeric_key_pending", recovery_required=False,
                        pending_numeric_edit=final_verification_details["pending_numeric_edit"],
                        pending_numeric_edit_source="original_unverified_typing_response",
                        feedback=("Original fill remains unverified. Continue only the initially declared "
                                  "single numeric key, with a fresh same-field and buffer check; retain "
                                  "the original plan clock and all attempt/failure budgets."),
                    )
                    emit("agent_declared_numeric_finalize", {
                        "status": "pending", "source_receipt_id": step_result.receipt.receipt_id,
                        "key": selected_template.property_edit.numeric_finalize_key,
                        "verified": False, "task_success": False,
                    })
                elif (not verified and not stale_recovery and (reusable_plan or (
                        cfg.adaptive_dispatch and getattr(self.planner, "profile", None) == "browser"))):
                    semantic_recovery = True
                    action_feedback.update(
                        recovery_required=True,
                        reason="postcondition_unconfirmed" if step_result.receipt.success else "input_failed",
                        step_id=selected_step.id,
                        operation=effect_step.operation,
                        target=asdict(effect_step.target) if effect_step.target else None,
                        postconditions=_postcondition_feedback(effect_step, final_verification_details),
                    )
                    if step_result.receipt.success:
                        prefix = (
                            "Input was sent" if step_result.receipt.output.get("environment_steps")
                            else "The typed action completed"
                        )
                        action_feedback["feedback"] = (
                            f"{prefix} but the required effect was not confirmed. Inspect the actual "
                            "current state; do not blindly repeat the input. A missing accessibility "
                            "predicate does not by itself prove the UI action failed."
                        )
                        if any(item.get("outcome") == "unknown"
                               for item in action_feedback["postconditions"]):
                            action_feedback["feedback"] += (
                                " Some required state cannot be observed by the declared predicates. "
                                "Unknown is not evidence to toggle again; inspect the screenshot or "
                                "repair the target binding before choosing another input."
                            )
                    if navigation_cycle:
                        action_feedback.update(
                            reason="navigation_cycle",
                            feedback="Navigation returned to a previously observed state without completing "
                            "the subgoal. Reassess the current UI and choose one grounded recovery action.",
                        )
                    if application_wait_summary is not None:
                        action_feedback["application_wait"] = application_wait_summary
                        appearance = final_verification_details.get("application_wait", {})
                        if appearance and not appearance.get("eligible"):
                            action_feedback.update(
                                reason=appearance["reason"],
                                application_appearance={key: appearance[key] for key in (
                                    "observed_app", "observed_window", "observed_role"
                                ) if key in appearance},
                                feedback=(
                                    "The launcher input was already sent. The observed application "
                                    "arrival did not confirm the declared predicate. Inspect this "
                                    "fresh application/window evidence and repair the target or "
                                    "condition; do not blindly launch another instance."
                                ),
                            )
                    # Use only the latest verification. A later observation may
                    # have committed the value, lost focus, or removed the field;
                    # the original receipt must not resurrect its old buffer.
                    pending_edit = final_verification_details.get("pending_numeric_edit")
                    if (
                        step_result.receipt.success and effect_step.operation == "fill"
                        and isinstance(pending_edit, dict)
                        and pending_edit.get("reason") == "edit_buffer_matches_but_committed_value_differs"
                        and pending_edit.get("task_success") is False
                    ):
                        action_feedback["pending_numeric_edit"] = pending_edit
                        if eager_numeric_replan:
                            action_feedback["pending_numeric_edit_source"] = (
                                "original_unverified_typing_response"
                            )
                        action_feedback["feedback"] += (
                            " The requested numeric text is present in the focused edit buffer, "
                            "but its committed value still differs. Inspect the fresh field for "
                            "a supported commit action instead of repeating the fill; verify the "
                            "committed value afterwards. This is not a verified effect or task success."
                        )
                        if eager_numeric_replan:
                            action_feedback["feedback"] += (
                                " This describes the original typing response; the value may still "
                                "commit asynchronously. Choose one current recovery action or wait."
                            )
                    emit("agent_recovery", action_feedback)
                history.append(action_feedback)
                emit(
                    "agent_step",
                    {
                        "step": counts["steps"],
                        "candidate": chosen.id,
                        "verified": verified,
                        "planner_calls": counts["planner_calls"],
                        "jev_ms": step_result.decision.latency_ms,
                        "candidate_count": len(offered),
                        "distinct_target_count": distinct_target_counts[-1],
                        "reusable_plan": reusable_plan,
                        "navigation": selected_step.navigation,
                        "subgoal_complete": navigation_complete if selected_step.navigation else verified,
                        "policy_source": step_result.decision.model,
                    },
                )
                if v52_information_routing:
                    next_anchor = _v52_visual_information_anchor(
                        snapshot, selected_step, step_result.receipt, verified=verified,
                        request_information=plan_request_information,
                        recovery=(recovery_plan or bool(
                            plan_request_information and plan_request_information.get("routing_reason")
                            in {"verified_visual_progress_same_native",
                                "planner_history_single_action_recovery"}
                        )),
                        network_recovery=(network_recovery_active or capacity_recovery_active),
                    )
                    if next_anchor is not None or visual_information_anchor is not None:
                        emit("agent_visual_information_routing", {
                            "status": "armed" if next_anchor is not None else "invalidated",
                            "reason": "verified_image_only_input_same_native" if next_anchor else
                                      "step_did_not_verify_same_native_visual_progress",
                            **(next_anchor or visual_information_anchor),
                            "observed_receipt_id": step_result.receipt.receipt_id,
                            "information_only": True, "task_success": False,
                        })
                    visual_information_anchor = next_anchor
                if stale_recovery:
                    stale_recovery_used = True
                    counts["stale_recovery_attempts"] += 1
                    action_failures += 1
                    post_snapshot_source = None
                    stale_snapshot_source = (chosen, step_result.receipt)
                    emit("agent_stale_recovery", {
                        "stage": "requested", "reason": "zero_input_stale_observation",
                        "receipt_id": step_result.receipt.receipt_id,
                        "action_failures": action_failures,
                        **({"per_progress_limit": 1} if cfg.stale_recovery_on_progress
                           else {"per_plan_limit": 1}), "task_success": False,
                    })
                    continue
                if declared_finalize_requested:
                    # The pending fill is an unverified attempted action, not
                    # progress. Its original failure and horizon charge remain.
                    action_failures += 1
                    plan_actions += 1
                    post_snapshot_source = None
                    pending_numeric_finalize_source = (chosen, step_result.receipt, selected_template)
                    pending_reason = ""
                    continue
                if verified:
                    vlm_stale_replans = 0
                    unexplained_recovery_ticket = (
                        unexplained_context_recovery_active and unexplained_planning_reuse_enabled
                    )
                    recovery_planning_eligible = bool(
                        cfg.reuse_verified_post_snapshot and cfg.mode == "hybrid"
                        and recovery_plan and requested_limit == 1
                        and getattr(self.planner, "profile", None) == "desktop"
                        and type(plan_request_information) is dict
                        and plan_request_information.get("planning_observation") == "full_multimodal_v1"
                        and plan_request_information.get("max_actions") == 1
                        and plan_request_information.get("image_uploaded") is True
                        and plan_request_information.get("a11y_complete") is True
                        and not (network_recovery_active or capacity_recovery_active)
                        and not context_recovery_active
                        and (not unexplained_context_recovery_active or unexplained_planning_reuse_enabled)
                        and not pending_recovery and not property_navigation_wait
                        and len(plan.steps) == 1 and not plan.completion_conditions
                        and step_result.receipt.success is True
                        and type(step_result.receipt.output.get("environment_steps")) is int
                        and step_result.receipt.output["environment_steps"] > 0
                        and selected_step.operation != "wait"
                    )
                    if selected_step.id == "prepare":
                        active_prepare_repair = None
                    input_count = step_result.receipt.output.get("environment_steps")
                    if (cfg.stale_recovery_on_progress and stale_recovery_used
                            and type(input_count) is int and input_count > 0
                            and selected_step.operation != "wait"):
                        # A rejected or read-only attempt cannot replenish this
                        # allowance. Preserve the original plan clock and budgets.
                        stale_recovery_used = False
                        emit("agent_stale_recovery", {
                            "stage": "rearmed_after_verified_input", "per_progress_limit": 1,
                            "receipt_id": step_result.receipt.receipt_id, "task_success": False,
                        })
                    if capacity_recovery_active:
                        if callable(clear_input_evidence):
                            clear_input_evidence()
                        input_steps = step_result.receipt.output.get("environment_steps")
                        if type(input_steps) is not int or input_steps <= 0:
                            action_failures += 1
                            status, reason = "stuck", "capacity recovery produced no GUI input"
                            emit("agent_capacity_recovery", {
                                "stage": "failed", "reason": "no_positive_input", "task_success": False,
                            })
                            break
                        counts["capacity_recovery_verified"] += 1
                        capacity_recovery_active = False
                        emit("agent_capacity_recovery", {
                            "stage": "verified", "candidate_id": chosen.id,
                            "receipt_id": step_result.receipt.receipt_id, "task_success": False,
                        })
                    if network_recovery_active:
                        counts["network_recovery_verified"] += 1
                        network_recovery_active = False
                        emit("agent_network_recovery", {
                            "stage": "verified", "candidate_id": chosen.id, "task_success": False,
                        })
                    if context_recovery_active:
                        input_count = step_result.receipt.output.get("environment_steps")
                        if (
                            type(input_count) is not int or input_count <= 0
                            or selected_step.operation == "wait"
                        ):
                            status, reason = "stuck", "new context recovery produced no input"
                            break
                        context_recovery_active = False
                    # Only the runner's final verdict can offer a receipt for
                    # one-shot decision reuse. Recovery and cycle handling have
                    # already run; neither may authorize a cached decision.
                    reusable_pending_observation = bool(
                        property_navigation_wait and final_verification_details.get("observation_source")
                        == "verified_pending_effect_observation"
                    )
                    if ((not pending_recovery or reusable_pending_observation)
                            and not recovery_plan and not semantic_recovery):
                        post_snapshot_source = (chosen, step_result.receipt)
                    if (cfg.reuse_verified_pending_snapshot and cfg.reuse_verified_post_snapshot
                            and pending_recovery and not property_navigation_wait
                            and not recovery_plan and not semantic_recovery and not invalid_plan_recovery
                            and not (network_recovery_active or capacity_recovery_active)
                            and not context_recovery_active
                            and not stale_recovery and finalize_source is None and numeric_source is None
                            and pending_prepare_repair is None and active_prepare_repair is None
                            and selected_step.operation != "wait"
                            and type(input_count) is int and input_count > 0
                            and final_verification_details.get("postconditions_met") is True):
                        post_snapshot_source = (
                            chosen, step_result.receipt, "verified_ordinary_pending_effect_observation",
                        )
                    action_failures = 0
                    dispatch_failure_streak = 0
                    recovery_observations = 0
                    if plan_actions:
                        counts["plan_reuses"] += 1
                    counts["semantic_steps"] += (
                        reusable_plan and not selected_step.operation.startswith("visual_")
                    )
                    if selected_step.navigation:
                        if selected_step.property_edit is not None and not effect_step.navigation:
                            count_key = "property_confirmations" if (
                                final_verification_details.get("property_confirmation") is True
                            ) else "property_edits"
                            counts[count_key] += 1
                        else:
                            counts["navigation_steps"] += 1
                        if navigation_complete:
                            completed.add(chosen.intent)
                            counts["property_subgoals" if selected_step.property_edit is not None
                                   else "navigation_subgoals"] += 1
                    else:
                        completed.add(chosen.intent)
                    plan_actions += 1
                    if active_program is not None:
                        receipt_inputs = step_result.receipt.output.get("environment_steps")
                        if (type(receipt_inputs) is not int or receipt_inputs < 0
                                or type(step_result.receipt.receipt_id) is not str
                                or not step_result.receipt.receipt_id):
                            abandon_program("verified_receipt_accounting_unavailable")
                        else:
                            program_receipts.append(
                                (step_result.receipt.receipt_id, receipt_inputs)
                            )
                            program_verified_effects[selected_template.id] = effect_step
                    if recovery_plan:
                        semantic_recovery = invalid_plan_recovery = False
                        # Keep the final recovery's stopping declaration until the
                        # next fresh observation. Other recovery plans still replan.
                        if not (plan.completion_conditions and len(completed) == len(plan.steps)):
                            plan = None
                        pending_reason = "recovery_completed"
                    if not reusable_plan and selected_step.operation in {
                        "fill", "select", "check", "uncheck"
                    }:
                        targets = [c for c in planned.controls if c.ref == selected_step.ref]
                        condition = step_result.receipt.output.get("postcondition", {})
                        if (
                            len(targets) == 1
                            and condition.get("identity") == list(targets[0].identity())
                            and isinstance(condition.get("value"), str)
                            and type(condition.get("checked")) is bool
                        ):
                            completed_conditions[targets[0].identity()] = (
                                condition["value"],
                                condition["checked"],
                            )
                            expected_values[targets[0].identity()] = [
                                completed_conditions[targets[0].identity()]
                            ]
                        else:
                            plan = None
                            pending_reason = "completed_target_changed"
                    if not reusable_plan and (
                        selected_step.operation.startswith("visual_")
                        or selected_step.operation in {
                            "scroll", "back", "wait", "type_text", "hotkey", "key_press", "press",
                        }
                    ):
                        plan = None
                        pending_reason = "visual_or_page_action"
                    if unexplained_context_recovery_active:
                        unexplained_context_recovery_active = False
                    context_signal = None
                    context_planning_capture = None
                    if (unexplained_context_enabled and reusable_plan and requested_limit > 1
                            and selected_template.navigation
                            and selected_template.native_navigation_gestures
                            and selected_template.operation == "click"
                            and selected_template.property_edit is None
                            and not recovery_plan and not semantic_recovery
                            and type(input_count) is int and input_count > 0
                            and step_result.receipt.success is True):
                        if unexplained_planning_reuse_enabled:
                            context_planning_capture = _capture_unexplained_pending_planning_frame(
                                self.backend, chosen, step_result.receipt, unexplained_planning_input_epoch,
                            )
                        context_signal = self.backend.take_unexplained_navigation_context(
                            chosen, step_result.receipt,
                        )
                    if (type(context_signal) is dict
                            and context_signal.get("input_authority") is False
                            and context_signal.get("task_success") is False):
                        emit("agent_unexplained_navigation_context", {
                            "stage": "replan_required", **context_signal,
                        })
                        plan = planned = None
                        completed.clear()
                        property_effects.clear()
                        completed_conditions.clear()
                        expected_values.clear()
                        navigation_seen.clear()
                        post_snapshot_source = recovery_planning_snapshot_source = None
                        pending_numeric_planning_source = pending_numeric_finalize_source = None
                        stale_snapshot_source = None
                        pending_prepare_repair = active_prepare_repair = None
                        visual_information_anchor = plan_request_information = None
                        context_recovery_pending = None
                        semantic_recovery = unexplained_context_recovery_active = True
                        invalid_plan_recovery = recovery_plan = False
                        pending_reason = "unexplained_navigation_context"
                        unexplained_planning_pending = _bind_unexplained_planning_signal(
                            context_planning_capture, context_signal,
                        )
                        clear_evidence = getattr(self.backend, "clear_input_observation_evidence", None)
                        if callable(clear_evidence):
                            clear_evidence()
                        clear_repair = getattr(self.planner, "set_prepare_repair_context", None)
                        if callable(clear_repair):
                            clear_repair(None)
                        history.append({
                            "reason": pending_reason, "recovery_required": True,
                            "feedback": "A verified input exposed a new active owning context, but "
                                        "the declared arrival is not established. Inspect a real "
                                        "fresh full image and current controls; return one grounded "
                                        "recovery action. Do not repeat navigation, change an unknown "
                                        "predicate to true, or claim the subgoal is complete.",
                            "public_context": context_signal,
                        })
                        continue
                    if navigation_followup_unknown:
                        clear_navigation_sources()
                        plan = None
                        semantic_recovery = True
                        invalid_plan_recovery = recovery_plan = False
                        pending_reason = "navigation_continuation_new_unknown_state"
                        history.append({"reason": pending_reason, "recovery_required": True,
                                        "feedback": "The local continuation exposed a state not "
                                                    "established by its known-state evidence. Preserve "
                                                    "the original arrival facts and inspect a "
                                                    "fresh full image."
                                        })
                        emit("agent_navigation_continuation", {"stage": "replan_required",
                                                               "reason": pending_reason,
                                                               "task_success": False})
                        continue
                    if recovery_planning_eligible and plan is None:
                        # Keep this separate from decision/goal reuse. The next
                        # iteration may consume it once for a new planning call;
                        # the normal model-return read and input preflight stay.
                        recovery_planning_snapshot_source = (
                            chosen, step_result.receipt,
                            step_result.receipt.output.get("after_fingerprint"),
                        )
                        if unexplained_recovery_ticket:
                            recovery_planning_snapshot_source += ("unexplained_context_planning_only",)
                else:
                    first_error = ((step_result.receipt.error or "").strip().splitlines() or [""])[0]
                    if (cfg.mode == "vlm" and vlm_stale_replans < cfg.vlm_stale_replans
                            and "stale" in first_error
                            and not step_result.receipt.output.get("environment_steps")):
                        vlm_stale_replans += 1
                        emit("agent_vlm_stale_replan", {"error": first_error[:160],
                                                        "attempt": vlm_stale_replans, "task_success": False})
                        time.sleep(0.3)
                        post_snapshot_source = None
                        plan = None
                        pending_reason = "effect_failed"
                        continue
                    action_failures += 1
                    if (navigation_continuation_eligible and navigation_cycle
                            and navigation_witness is not None and len(ready) == 1 and not recovery_plan
                            and not (network_recovery_active or capacity_recovery_active)
                            and not context_recovery_active
                            and pending_prepare_repair is None and active_prepare_repair is None
                            and counts["steps"] < cfg.max_steps and counts["planner_calls"] < cfg.max_plans
                            and action_failures < cfg.max_failures and planning_failures < cfg.max_failures
                            and plan_actions + 1 < cfg.plan_horizon
                            and time.perf_counter() - start < cfg.timeout_s
                            and time.perf_counter() - plan_at < cfg.plan_ttl_s):
                        # The input remains finalfalse and has already consumed an action failure.
                        # Retention must charge its actual attempt to the unchanged logical horizon.
                        plan_actions += 1
                        # Only the failed-effect feedback for this eligible ordinary plan
                        # set this latch; keep its finalfalse history and charged failure.
                        semantic_recovery = False
                        navigation_memory.request(navigation_witness)
                        clear_navigation_sources()
                        source = recovery_planning_source = stale_source = None
                        numeric_source = finalize_source = cached = None
                        numeric_planning_frame = numeric_finalize_frame = recovery_planning_frame = None
                        pending_reason = ""
                        emit("agent_navigation_continuation", {
                            "stage": "fresh_continuation_requested",
                            "receipt_id": step_result.receipt.receipt_id,
                            "final_verified": False, "action_failures": action_failures,
                            "plan_actions": plan_actions, "original_plan_at": plan_at,
                            "known_repeat_limit": 1, "input_authority": False, "task_success": False,
                        })
                        continue
                    plan = None
                    pending_reason = "navigation_cycle" if navigation_cycle else "effect_failed"
                    if eager_numeric_replan:
                        pending_numeric_planning_source = (chosen, step_result.receipt)
                        pending_reason = "pending_numeric_commit_requires_model"
                    if network_recovery_active:
                        emit("agent_network_recovery", {
                            "stage": "failed", "reason": "effect_unconfirmed", "task_success": False,
                        })
                        status, reason = "stuck", "network recovery effect unconfirmed"
                        break
                    if capacity_recovery_active:
                        input_count = step_result.receipt.output.get("environment_steps")
                        if (capacity_phase and step_result.receipt.success is True
                                and (not capacity_phase_v3 or (not navigation_cycle
                                     and time.perf_counter() - plan_at < cfg.plan_ttl_s))
                                and type(input_count) is int and input_count > 0
                                and action_failures < cfg.max_failures
                                and planning_failures < cfg.max_failures):
                            # Another H1 plan within this phase is not another Jev
                            # selection-recovery entrance. Keep the shared ticket,
                            # both failure streaks, time origin and all budgets.
                            capacity_recovery_planned = False
                            capacity_recovery_fresh_required = True
                            semantic_recovery = True
                            invalid_plan_recovery = recovery_plan = False
                            plan = planned = None
                            completed.clear()
                            property_effects.clear()
                            completed_conditions.clear()
                            expected_values.clear()
                            navigation_seen.clear()
                            offered.clear()
                            compiled_effects.clear()
                            post_snapshot_source = recovery_planning_snapshot_source = None
                            pending_numeric_planning_source = pending_numeric_finalize_source = None
                            stale_snapshot_source = None
                            source = recovery_planning_source = stale_source = None
                            numeric_source = finalize_source = cached = None
                            numeric_planning_frame = numeric_finalize_frame = recovery_planning_frame = None
                            pending_prepare_repair = active_prepare_repair = None
                            context_recovery_pending = None
                            visual_information_anchor = plan_request_information = None
                            whole_goal_pending_frame = whole_goal_inband_done = None
                            inband_last_fresh = None
                            last_runner_fresh_snapshot = last_runner_fresh_proof = None
                            whole_goal_planning_frame = whole_goal_reused_proof = None
                            if callable(clear_input_evidence):
                                clear_input_evidence()
                            clear_repair = getattr(self.planner, "set_prepare_repair_context", None)
                            if callable(clear_repair):
                                clear_repair(None)
                            emit("agent_capacity_recovery", {
                                "stage": "continuation_requested", "policy": cfg.jev_capacity_recovery_policy,
                                "reason": "effect_unconfirmed_after_positive_input",
                                "receipt_id": step_result.receipt.receipt_id,
                                "positive_environment_steps": input_count,
                                "action_failures": action_failures,
                                "planning_failures": planning_failures,
                                "shared_ticket_used": selection_recovery_ticket_used,
                                "verified": False, "task_success": False,
                            })
                            continue
                        emit("agent_capacity_recovery", {
                            "stage": "failed", "reason": "effect_unconfirmed", "task_success": False,
                        })
                        status, reason = "stuck", "capacity recovery effect unconfirmed"
                        break
                    if action_failures >= cfg.max_failures:
                        # One final repair opportunity for a genuinely new UI
                        # reached by an unconfirmed input. The backend must
                        # bind this evidence to the exact receipt and latest
                        # observation. Neither the failed step nor its effects
                        # become verified, and the failure streak is retained.
                        take_context = getattr(self.backend, "take_unverified_recovery_context", None)
                        input_count = step_result.receipt.output.get("environment_steps")
                        context = None
                        if (
                            not navigation_cycle and not counts["context_recoveries"]
                            and step_result.receipt.success
                            and type(input_count) is int and input_count > 0
                            and counts["steps"] < cfg.max_steps
                            and counts["planner_calls"] < cfg.max_plans
                            and time.perf_counter() - start < cfg.timeout_s
                            and callable(take_context)
                        ):
                            context = take_context(chosen, step_result.receipt)
                        if (
                            isinstance(context, dict)
                            and context.get("reason") == "new_actionable_context_after_unconfirmed_input"
                            and type(context.get("new_control_count")) is int
                            and context["new_control_count"] > 0
                            and isinstance(context.get("context_fingerprint"), str)
                            and len(context["context_fingerprint"]) == 64
                            and all(c in "0123456789abcdef" for c in context["context_fingerprint"])
                        ):
                            counts["context_recoveries"] += 1
                            context_recovery_pending = context
                            context_recovery_active = True
                            semantic_recovery = True
                            post_snapshot_source = None
                            pending_reason = "new_context_recovery"
                            feedback = {
                                "reason": pending_reason, "recovery_required": True,
                                "verified": False, "task_success": False,
                                "action_failures": action_failures,
                                "episode_recovery_limit": 1,
                                "context_fingerprint": context["context_fingerprint"],
                                "new_control_count": context["new_control_count"],
                                "feedback": (
                                    "The input exposed a new actionable interface, but its declared "
                                    "effect remains unconfirmed. One final single-action recovery "
                                    "opportunity remains. Inspect the fresh observation and repair "
                                    "the target or condition; do not repeat the previous input "
                                    "merely because verification was unknown. An unnamed current "
                                    "field can use its observed role and app/window context when "
                                    "they uniquely identify an executable field. Existing or desired "
                                    "field text is not its accessible name."
                                ),
                            }
                            history.append(feedback)
                            emit("agent_context_recovery", feedback)
                            continue
                        status, reason = "stuck", "consecutive action failures"
                        break
            # The last permitted action can itself finish the task.
            if status == "max_steps" and success is not None:
                began = time.perf_counter()
                final = fresh_observe()
                counts["observation_ms"] += (time.perf_counter() - began) * 1000
                began = time.perf_counter()
                if success(final):
                    status, reason = "success", "independent acceptance checks passed"
                counts["acceptance_ms"] += (time.perf_counter() - began) * 1000
        except _EpisodeExpired:
            status, reason = "timeout", "time budget exhausted before action"
        except Exception as exc:
            # Do not print network exception strings, which can carry endpoint credentials.
            status, reason = "error", (
                "PolicyError" if isinstance(exc, PolicySelectionFailure) else type(exc).__name__
            )
            details = {"error_type": reason}
            failure = (exc.failure.summary() if isinstance(exc, PolicySelectionFailure)
                       else {} if counts["network_recovery_attempts"] or counts["capacity_recovery_attempts"]
                       else getattr(self.policy, "last_failure", {}))
            if isinstance(failure, dict) and failure.get("code") in {
                "connection_error", "connect_timeout", "timeout", "http_status",
                "authentication", "invalid_response", "request_error",
            }:
                # Only typed, enumerated diagnostics; never copy exceptions,
                # response bodies, headers or arbitrary provider metadata.
                details["policy_failure"] = {
                    "code": failure["code"],
                    "attempts": failure.get("attempts") if type(failure.get("attempts")) is int else None,
                    "retryable": failure.get("retryable") is True,
                    "http_status": (failure.get("http_status")
                                    if type(failure.get("http_status")) is int else None),
                }
            emit("agent_error", details)
        finally:
            if active_program is not None:
                abandon_program("original_runner_terminal", force_recovery=False)
            if restore_input_evidence is not None:
                configure_evidence(restore_input_evidence)
        final_observe_calls = getattr(self.backend, "fresh_observe_calls", None)
        actual_observe_calls = (
            final_observe_calls - initial_observe_calls
            if type(initial_observe_calls) is int and type(final_observe_calls) is int
            and final_observe_calls >= initial_observe_calls else None
        )
        result = AgentResult(
            status,
            reason,
            counts["steps"],
            (time.perf_counter() - start) * 1000,
            counts["planner_calls"],
            counts["planner_ms"],
            metered.calls if cfg.mode == "hybrid" else 0,
            metered.wall_ms if cfg.mode == "hybrid" else 0,
            counts["backend_ms"],
            counts["action_ms"],
            counts["effect_ms"],
            counts["observation_ms"],
            counts["acceptance_ms"],
            counts["plan_reuses"],
            dict(reasons),
            dict(routes),
            {
                key: value - initial_usage.get(key, 0)
                for key, value in getattr(self.planner, "usage_totals", {}).items()
            },
            tuple(candidate_counts),
            counts["multi_candidate_steps"],
            counts["semantic_steps"],
            counts["condition_skips"],
            counts["recovery_observations"],
            counts["navigation_steps"],
            counts["navigation_subgoals"],
            counts["plan_completion_claims"],
            metered.singleton_decisions + (dispatch.singleton_decisions if dispatch else 0),
            counts["post_snapshot_reuses"],
            counts["runner_observe_calls"],
            actual_observe_calls,
            tuple(distinct_target_counts),
            counts["context_recoveries"],
            metered.request_attempts if cfg.mode == "hybrid" else 0,
            counts["jev_network_failures"],
            counts["network_recovery_attempts"],
            counts["network_recovery_actions"],
            counts["network_recovery_verified"],
            counts["property_edits"],
            counts["property_confirmations"],
            counts["property_subgoals"],
            counts["stale_recovery_attempts"],
            counts["stale_snapshot_reuses"],
            counts["recovery_planning_snapshot_reuses"],
            capacity_recovery_enabled=cfg.jev_capacity_recovery,
            post_plan_observations_deferred=counts["post_plan_observations_deferred"],
            jev_capacity_failures=counts["jev_capacity_failures"],
            capacity_recovery_attempts=counts["capacity_recovery_attempts"],
            capacity_recovery_actions=counts["capacity_recovery_actions"],
            capacity_recovery_verified=counts["capacity_recovery_verified"],
            whole_goal_recheck_calls=counts["whole_goal_recheck_calls"],
            whole_goal_recheck_continues=counts["whole_goal_recheck_continues"],
            whole_goal_recheck_done=counts["whole_goal_recheck_done"],
            whole_goal_planning_snapshot_reuses=counts["whole_goal_planning_snapshot_reuses"],
            whole_goal_inband_review_calls=counts["whole_goal_inband_review_calls"],
            whole_goal_inband_review_done=counts["whole_goal_inband_review_done"],
            unexplained_navigation_context_replans=counts["unexplained_navigation_context_replans"],
            verified_pending_snapshot_reuses=counts["verified_pending_snapshot_reuses"],
            equivalent_rule_decisions=dispatch.equivalent_decisions if dispatch else 0,
            dispatch_refreshes=dispatch_refreshes, dispatch_s2_replans=dispatch_s2_replans,
        )
        emit("agent_episode", result.to_dict())
        if callable(set_planner_deadline):
            set_planner_deadline(None)
        return result
