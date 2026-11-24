"""Bounded observed navigation evidence, never input permission."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from .agent_types import AgentSnapshot, AgentStep
from .desktop_semantics import navigation_scope_controls
from .models import ActionCandidate


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hex(value) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(control) -> str | None:
    """Only fields already present in the public control state; no ref/value/box."""
    if isinstance(control, dict):
        fields = [control.get(k, "") for k in ("label", "role", "tag", "app", "window")]
        ancestry = control.get("ancestors", [])
    else:
        fields = [getattr(control, k, "") for k in ("label", "role", "tag", "app", "window")]
        ancestry = getattr(control, "ancestors", ())
    if (not all(type(v) is str and len(v) <= 2000 for v in fields)
            or not isinstance(ancestry, (list, tuple)) or len(ancestry) > 32
            or not all(type(v) is str and len(v) <= 2000 for v in ancestry)):
        return None
    return _digest([*fields, list(ancestry)])


@dataclass(frozen=True)
class NavigationState:
    native: str
    scope: str
    owner: tuple[str, str, str]
    controls: tuple[tuple[str, str], ...]

    def same_observable_state(self, other: NavigationState) -> bool:
        # This deliberately does not claim identical pixels or hidden application state.
        return (self.native, self.scope, self.owner) == (other.native, other.scope, other.owner)


def navigation_state(step: AgentStep, snapshot: AgentSnapshot, scope: str) -> NavigationState | None:
    if (not step.navigation or step.operation != "click" or step.target is None
            or snapshot.a11y_complete is not True or snapshot.image is None
            or not isinstance(snapshot.image.jpeg, bytes) or not snapshot.image.jpeg or not _hex(scope)):
        return None
    controls = tuple(navigation_scope_controls(snapshot, step.target))
    owners = {(c.app, c.window) for c in controls if c.active and c.app and c.window}
    if len(owners) != 1:
        return None
    app, window = next(iter(owners))
    if any(c.active and (c.app != app or (c.window and c.window != window)) for c in controls):
        return None
    modal_refs = {c.modal_owner_ref for c in controls if c.modal_context}
    if any(c.modal_context and not c.modal_owner_ref for c in controls) or len(modal_refs) > 1:
        return None
    modal_identity = ""
    if modal_refs:
        owners = [c for c in snapshot.controls if c.ref in modal_refs]
        if len(owners) != 1 or (modal_identity := _identity(owners[0])) is None:
            return None
    entries = tuple((c.ref, key) for c in controls if (key := _identity(c)) is not None)
    native = snapshot.fingerprint()
    if not _hex(native) or not entries or len({ref for ref, _ in entries}) != len(entries):
        return None
    return NavigationState(native, scope, (app, window, modal_identity), entries)


@dataclass(frozen=True)
class NavigationAction:
    identity: str
    button: str
    count: int

    @property
    def key(self) -> str:
        return _digest([self.identity, "click", self.button, self.count])


def navigation_action(candidate: ActionCandidate, state: NavigationState) -> NavigationAction | None:
    args = candidate.arguments
    control = args.get("bound_control")
    if (args.get("operation") != "click" or args.get("navigation") is not True
            or args.get("fingerprint") != state.native or args.get("value", "") != ""
            or not isinstance(control, dict)):
        return None
    key = _identity(control)
    button, count = args.get("button", "left"), args.get("click_count", 1)
    if (key is None or button not in {"left", "right"} or type(count) is not int
            or count not in {1, 2} or (button == "right" and count != 1)
            or (control.get("ref"), key) not in state.controls
            or Counter(identity for _, identity in state.controls)[key] != 1):
        return None
    return NavigationAction(key, button, count)


def fact_outcomes(facts) -> tuple[str, ...] | None:
    if not isinstance(facts, (list, tuple)) or len(facts) > 8:
        return None
    outcomes = []
    for fact in facts:
        if (not isinstance(fact, dict) or fact.get("outcome") not in {"true", "false", "unknown"}
                or type(fact.get("satisfied")) is not bool
                or fact["satisfied"] != (fact["outcome"] == "true")):
            return None
        outcomes.append(fact["outcome"])
    return tuple(outcomes)


@dataclass(frozen=True)
class NavigationTransition:
    stage: str
    receipt_id: str
    before: NavigationState
    after: NavigationState
    action: NavigationAction
    input_steps: int
    backend_verified: bool
    arrival: tuple[str, ...]

    def __post_init__(self):
        if (type(self.stage) is not str or not self.stage or len(self.stage) > 64
                or type(self.receipt_id) is not str or not self.receipt_id or len(self.receipt_id) > 128
                or not isinstance(self.before, NavigationState) or not isinstance(self.after, NavigationState)
                or not isinstance(self.action, NavigationAction)
                or type(self.input_steps) is not int or self.input_steps <= 0
                or type(self.backend_verified) is not bool or type(self.arrival) is not tuple
                or len(self.arrival) > 8 or any(v not in {"true", "false", "unknown"} for v in self.arrival)):
            raise ValueError("invalid bounded actual navigation transition")


@dataclass(frozen=True)
class RecordedTransition:
    evidence: NavigationTransition
    final_verified: bool
    cycle: bool


@dataclass(frozen=True)
class CycleWitness:
    stage: str
    expected: NavigationState


@dataclass(frozen=True)
class NavigationDecline:
    stage: str
    state_fingerprint: str
    edge_key: str


class NavigationMemory:
    """Private exact evidence; small advisory wire state; no action selection."""

    def __init__(self, limit: int):
        if type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("navigation memory uses the original horizon")
        self.limit = limit
        self.reset(0)

    def reset(self, epoch: int) -> None:
        previous = getattr(self, "records", [])[-1:]
        previous_epoch = getattr(self, "epoch", 0)
        self.previous = (previous_epoch, previous[0]) if previous else None
        self.epoch = epoch
        self.records: list[RecordedTransition] = []
        self.known: dict[tuple[str, str], NavigationState] = {}
        self.exhausted: set[tuple[str, str, str]] = set()
        self.state_ids: dict[tuple[str, str], str] = {}
        self.used = False
        self.pending: CycleWitness | None = None
        self.local_attempt_active = False
        self.current: tuple[str, NavigationState] | None = None
        self.current_actions: dict[str, NavigationAction] = {}
        self.current_ids: dict[str, str] = {}
        self.observation_id: str | None = None

    def _state_id(self, stage: str, state: NavigationState) -> str:
        key = (stage, state.native)
        if key not in self.state_ids:
            self.state_ids[key] = f"s{len(self.state_ids) + 1}"
        return self.state_ids[key]

    def record(self, item: NavigationTransition, *, final_verified: bool, cycle: bool) -> CycleWitness | None:
        if (type(final_verified) is not bool or type(cycle) is not bool
                or type(item.input_steps) is not int or item.input_steps <= 0
                or type(item.backend_verified) is not bool or len(self.records) >= self.limit
                or any(record.evidence.receipt_id == item.receipt_id for record in self.records)):
            return None
        for state in (item.before, item.after):
            old = self.known.get((item.stage, state.native))
            if old is not None and not old.same_observable_state(state):
                return None
        old_after = self.known.get((item.stage, item.after.native))
        self.known.setdefault((item.stage, item.before.native), item.before)
        self.known.setdefault((item.stage, item.after.native), item.after)
        self._state_id(item.stage, item.before)
        self._state_id(item.stage, item.after)
        self.records.append(RecordedTransition(item, final_verified, cycle))
        if (not cycle or final_verified or item.backend_verified is not True or self.used
                or old_after is None or not old_after.same_observable_state(item.after)
                or item.before.owner != item.after.owner):
            return None
        start = next((i for i, record in enumerate(self.records)
                      if record.evidence.stage == item.stage
                      and record.evidence.before.same_observable_state(item.after)), None)
        if start is None:
            return None
        walk = self.records[start:]
        # Shared owner/scope alone does not prove the observed path is continuous.
        # A real intervening native-state gap remains historical, not exhausted.
        if (not walk
                or not walk[0].evidence.before.same_observable_state(walk[-1].evidence.after)
                or not walk[0].evidence.before.same_observable_state(item.after)
                or any(not left.evidence.after.same_observable_state(right.evidence.before)
                       for left, right in zip(walk, walk[1:], strict=False))):
            return None
        if any(record.evidence.stage != item.stage
               or record.evidence.before.owner != item.after.owner
               or record.evidence.after.owner != item.after.owner
               or (record.evidence.arrival and all(v == "true" for v in record.evidence.arrival))
               for record in walk):
            return None
        for record in walk:
            edge = record.evidence
            self.exhausted.add((edge.stage, edge.before.native, edge.action.key))
        return CycleWitness(item.stage, item.after)

    def request(self, witness: CycleWitness) -> None:
        if (self.used or not self.records or not self.records[-1].cycle
                or self.records[-1].final_verified
                or self.records[-1].evidence.stage != witness.stage
                or not self.records[-1].evidence.after.same_observable_state(witness.expected)):
            raise ValueError("one known-repeat concession per accepted plan")
        self.used = True
        self.pending = witness
        self.local_attempt_active = False
        self.observation_id = None

    def bind_offers(self, stage: str, state: NavigationState,
                    candidates: tuple[ActionCandidate, ...]) -> str | None:
        """Observe the full unchanged pool; return a reason, never another action."""
        self.current = (stage, state)
        self.current_actions = {}
        self.current_ids = {}
        self.observation_id = None
        ambiguous = False
        ambiguous_keys: set[str] = set()
        for candidate in candidates:
            if candidate.intent != stage or candidate.arguments.get("navigation") is not True:
                continue  # Typed property edits/confirmation are outside edge admission.
            action = navigation_action(candidate, state)
            if action is None:
                ambiguous = True
                continue
            if action.key in self.current_ids:
                previous_id = self.current_ids.pop(action.key)
                self.current_actions.pop(previous_id)
                ambiguous_keys.add(action.key)
                ambiguous = True
                continue
            if action.key in ambiguous_keys:
                continue
            self.current_actions[candidate.id] = action
            self.current_ids[action.key] = candidate.id
        if self.pending is None:
            return None
        expected = self.pending
        if (stage != expected.stage or not state.same_observable_state(expected.expected)
                or ambiguous or not self.current_actions):
            return "navigation_continuation_evidence_changed"
        tried = {(r.evidence.before.native, r.evidence.action.key) for r in self.records
                 if r.evidence.stage == stage}
        if not any((state.native, action.key) not in tried for action in self.current_actions.values()):
            return "navigation_continuation_no_untried_edge"
        self.pending = None
        self.local_attempt_active = True
        return None

    def bind_observation(self, observation_id: str) -> None:
        self.observation_id = observation_id

    def decline(self, observation_id: str, candidate_id: str) -> NavigationDecline | None:
        """Pure lookup. Never choose, trim, modify confidence, or perform I/O."""
        if not self.local_attempt_active or observation_id != self.observation_id or self.current is None:
            return None
        stage, state = self.current
        action = self.current_actions.get(candidate_id)
        if action is not None and (stage, state.native, action.key) in self.exhausted:
            return NavigationDecline(stage, state.native, action.key)
        return None

    def public(self) -> dict:
        """At most12 short edges, no native hashes/receipts/deep identities/values/coordinates."""
        items = []
        current_stage, current_state = self.current if self.current is not None else (None, None)
        for index, record in enumerate(self.records, 1):
            edge = record.evidence
            current_id = (self.current_ids.get(edge.action.key)
                          if edge.stage == current_stage and current_state is not None
                          and edge.before.same_observable_state(current_state) else None)
            item = {"edge": f"e{index}", "stage": edge.stage,
                    "from": self._state_id(edge.stage, edge.before),
                    "to": self._state_id(edge.stage, edge.after),
                    "gesture": f"{edge.action.button}/{edge.action.count}",
                    "backend_effect": edge.backend_verified, "final_verdict": record.final_verified,
                    "arrival": list(edge.arrival), "cycle": record.cycle,
                    "binding": "current_unique" if current_id else "historical"}
            if current_id is not None:
                item["candidate_id"] = current_id
                item["observed_loop_edge"] = (
                    edge.stage, edge.before.native, edge.action.key
                ) in self.exhausted
            items.append(item)
        context = {}
        if self.previous is not None:
            epoch, record = self.previous
            context["previous_plan_edge"] = {
                "plan": f"p{epoch}", "stage": record.evidence.stage, "binding": "historical",
                "gesture": f"{record.evidence.action.button}/{record.evidence.action.count}",
                "backend_effect": record.evidence.backend_verified,
                "final_verdict": record.final_verified, "arrival": list(record.evidence.arrival),
            }
        return {"version": "observed_transition_v1", "plan": f"p{self.epoch}",
                "state": None if current_state is None else self._state_id(current_stage, current_state),
                "edges": items, "known_repeat_used": self.used, "whole_goal_proven": False, **context}
