from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from .errors import GuardRejected, PolicyError
from .guard import ActionGuard
from .models import ActionCandidate, ActionReceipt, Decision, Observation, Verification
from .policy import DecisionPolicy, JevRequestError
from .registry import ExecutorRegistry
from .trace import JsonlTrace
from .verify import VerifierRegistry


@dataclass(frozen=True)
class StepResult:
    decision: Decision
    receipt: ActionReceipt
    verification: Verification


class PolicySelectionFailure(PolicyError):
    """A current Jev choose call failed before decision, guard or execution."""

    def __init__(self, failure):
        super().__init__("Jev selection failed before execution")
        self.failure = failure


@dataclass(frozen=True)
class DecisionAdmissionRejection:
    stage_id: str
    state_fingerprint: str
    edge_key: str
    reason: str = "observed_navigation_edge_exhausted"

    def __post_init__(self):
        if (self.reason != "observed_navigation_edge_exhausted"
                or type(self.stage_id) is not str or not self.stage_id or len(self.stage_id) > 64
                or any(type(value) is not str or len(value) != 64
                       or any(c not in "0123456789abcdef" for c in value)
                       for value in (self.state_fingerprint, self.edge_key))):
            raise ValueError("invalid bounded decision-admission rejection")


class DecisionAdmissionRejected(PolicyError):
    """An actual choice was locally declined after guards, before any input."""

    def __init__(self, rejection: DecisionAdmissionRejection):
        super().__init__("Observed navigation edge declined before execution")
        self.rejection = rejection


class DecisionAdmissionContractError(PolicyError):
    """A hook violated value isolation; never a model/network/capacity failure."""

    def __init__(self, reason: str):
        if reason not in {"value_binding_failed", "hook_value_mutation", "callback_failed",
                          "invalid_hook_result"}:
            raise ValueError("unknown decision-admission contract failure")
        super().__init__("Decision admission contract failure")
        self.reason = reason


def _admission_json(value) -> str:
    # Hook-only finite, typed JSON value comparison, never raw contents in logs.
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _admission_hook_values(observation, decision, candidate) -> dict:
    return {"observation": asdict(observation), "decision": asdict(decision),
            "chosen": asdict(candidate)}


DecisionAdmission = Callable[
    [Observation, Decision, ActionCandidate], DecisionAdmissionRejection | None,
]


class AgentRuntime:
    def __init__(
        self,
        *,
        policy: DecisionPolicy,
        guard: ActionGuard,
        executors: ExecutorRegistry,
        verifiers: VerifierRegistry | None = None,
        trace: JsonlTrace | None = None,
        decision_admission: DecisionAdmission | None = None,
    ) -> None:
        if decision_admission is not None and not callable(decision_admission):
            raise ValueError("decision_admission must be callable or None")
        self.decision_admission = decision_admission
        self.policy = policy
        self.guard = guard
        self.executors = executors
        self.verifiers = verifiers or VerifierRegistry()
        self.trace = trace or JsonlTrace()

    def step(self, observation: Observation, candidates: Sequence[ActionCandidate]) -> StepResult:
        self.trace.append("observation", observation.to_dict())
        self.trace.append("candidates", {"items": [candidate.to_dict() for candidate in candidates]})
        try:
            decision = self.policy.choose(observation, candidates)
        except JevRequestError as exc:
            if exc.failure.observation_id != observation.observation_id:
                raise PolicyError("Jev failure observation mismatch") from None
            self.trace.append("policy_failure", exc.failure.to_dict())
            raise PolicySelectionFailure(exc.failure) from None
        self.trace.append("decision", decision.to_dict())
        exchange = getattr(self.policy, "last_exchange", None)
        if exchange:
            self.trace.append("policy_exchange", exchange)
        candidate = next((item for item in candidates if item.id == decision.candidate_id), None)
        self.trace.append(
            "commitment",
            {
                "candidate_id": decision.candidate_id,
                "intent": candidate.intent if candidate else "unknown",
                "channel": str(candidate.channel) if candidate else "unknown",
                "capability": candidate.capability if candidate else "unknown",
            },
        )
        try:
            candidate = self.guard.approve(observation, decision, candidates)
        except GuardRejected as exc:
            self.trace.append(
                "guard",
                {"approved": False, "candidate_id": decision.candidate_id, "reason": str(exc)},
            )
            raise
        self.trace.append(
            "guard",
            {
                "approved": True,
                "candidate_id": candidate.id,
                "checks": ["fresh_observation", "offered_candidate", "risk", "paths", "preconditions"],
            },
        )
        if self.decision_admission is not None:
            # The frozen dataclasses contain mutable dicts. Only an enabled hook
            # pays for isolation and full guard-value/offer-order checks.
            from copy import deepcopy

            guarded_candidate_id = candidate.id

            def contract_failure(reason):
                self.trace.append("decision_admission", {
                    "approved": False, "candidate_id": guarded_candidate_id,
                    "input_sent": False, "reason": reason,
                })
                raise DecisionAdmissionContractError(reason) from None

            try:
                guarded = _admission_hook_values(observation, decision, candidate)
                before_hook = _admission_json(guarded)
                before_all = _admission_json({
                    **guarded, "offered": [asdict(item) for item in candidates],
                })
                hook_observation, hook_decision, hook_candidate = deepcopy(
                    (observation, decision, candidate),
                )
            except Exception:
                contract_failure("value_binding_failed")
            callback_failed = False
            rejection = None
            try:
                rejection = self.decision_admission(hook_observation, hook_decision, hook_candidate)
            except Exception:
                # A callback cannot masquerade as a Jev/capacity/network error.
                callback_failed = True
            try:
                after_all = _admission_json({
                    **_admission_hook_values(observation, decision, candidate),
                    "offered": [asdict(item) for item in candidates],
                })
                after_hook = _admission_json(_admission_hook_values(
                    hook_observation, hook_decision, hook_candidate,
                ))
            except Exception:
                contract_failure("hook_value_mutation")
            if before_all != after_all or before_hook != after_hook:
                contract_failure("hook_value_mutation")
            if callback_failed:
                contract_failure("callback_failed")
            if rejection is not None:
                try:
                    if type(rejection) is not DecisionAdmissionRejection:
                        raise TypeError("hook result type")
                    # Revalidate and detach even a forged/mutated rejection record.
                    rejection = DecisionAdmissionRejection(
                        rejection.stage_id, rejection.state_fingerprint, rejection.edge_key, rejection.reason,
                    )
                except Exception:
                    contract_failure("invalid_hook_result")
            self.trace.append("decision_admission", {
                "approved": rejection is None, "candidate_id": guarded_candidate_id,
                "input_sent": False,
                **(asdict(rejection) if rejection is not None else {}),
            })
            if rejection is not None:
                raise DecisionAdmissionRejected(rejection)
        receipt = self.executors.execute(candidate, observation.observation_id, decision.decision_id)
        self.trace.append("receipt", receipt.to_dict())
        verification = self.verifiers.verify(candidate, receipt)
        self.trace.append("verification", verification.to_dict())
        return StepResult(decision, receipt, verification)
