from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import GuardRejected
from .guard import ActionGuard
from .models import ActionCandidate, ActionReceipt, Decision, Observation, Verification
from .policy import DecisionPolicy
from .registry import ExecutorRegistry
from .trace import JsonlTrace
from .verify import VerifierRegistry


@dataclass(frozen=True)
class StepResult:
    decision: Decision
    receipt: ActionReceipt
    verification: Verification


class StepDeadlineExceeded(TimeoutError):
    """The episode budget expired before another action could be dispatched."""


class AgentRuntime:
    def __init__(
        self,
        *,
        policy: DecisionPolicy,
        guard: ActionGuard,
        executors: ExecutorRegistry,
        verifiers: VerifierRegistry | None = None,
        trace: JsonlTrace | None = None,
    ) -> None:
        self.policy = policy
        self.guard = guard
        self.executors = executors
        self.verifiers = verifiers or VerifierRegistry()
        self.trace = trace or JsonlTrace()

    def step(
        self, observation: Observation, candidates: Sequence[ActionCandidate],
        *, deadline: float | None = None,
    ) -> StepResult:
        """Choose and execute one action; deadline uses the monotonic clock."""
        if deadline is not None and time.monotonic() >= deadline:
            raise StepDeadlineExceeded("episode timeout before policy decision")
        self.trace.append("observation", observation.to_dict())
        self.trace.append("candidates", {"items": [candidate.to_dict() for candidate in candidates]})
        decision = self.policy.choose(observation, candidates)
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
        # A remote decision or guard may outlast the episode budget. Never
        # dispatch its action just because the observation began in time.
        if deadline is not None and time.monotonic() >= deadline:
            raise StepDeadlineExceeded("episode timeout before action execution")
        receipt = self.executors.execute(candidate, observation.observation_id, decision.decision_id)
        self.trace.append("receipt", receipt.to_dict())
        verification = self.verifiers.verify(candidate, receipt)
        self.trace.append("verification", verification.to_dict())
        return StepResult(decision, receipt, verification)
