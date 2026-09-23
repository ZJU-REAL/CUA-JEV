from __future__ import annotations

import inspect
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from .errors import CuaJevError
from .models import ActionCandidate, ActionReceipt, Observation, Verification, state_fingerprint
from .runtime import AgentRuntime, StepResult


class EpisodeStatus(StrEnum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    STUCK = "stuck"
    POLICY_ERROR = "policy_error"
    GUARD_REJECTED = "guard_rejected"
    EXECUTION_ERROR = "execution_error"
    ENVIRONMENT_ERROR = "environment_error"


@dataclass(frozen=True)
class Evaluation:
    success: bool = False
    terminal: bool = False
    reason: str = "continue"
    details: dict = field(default_factory=dict)


class TaskEnvironment(Protocol):
    name: str

    def reset(self) -> None: ...

    def observe(self, history: Sequence[StepResult]) -> Observation: ...

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]: ...

    def evaluate(
        self,
        observation: Observation,
        candidate: ActionCandidate,
        receipt: ActionReceipt,
        verification: Verification,
    ) -> Evaluation: ...


@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int = 30
    timeout_s: float = 300
    unchanged_limit: int = 3
    repeated_action_limit: int = 3


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    status: EpisodeStatus
    reason: str
    steps: tuple[StepResult, ...]
    started_at: float
    ended_at: float
    channel_counts: dict[str, int]

    @property
    def success(self) -> bool:
        return self.status == EpisodeStatus.SUCCESS

    @property
    def duration_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "status": str(self.status),
            "reason": self.reason,
            "steps": len(self.steps),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "channel_counts": self.channel_counts,
        }


class EpisodeRunner:
    """Closed-loop task runner with deterministic termination boundaries."""

    def __init__(
        self,
        runtime_factory: Callable[..., AgentRuntime],
        config: EpisodeConfig | None = None,
    ) -> None:
        self.runtime_factory = runtime_factory
        self.config = config or EpisodeConfig()

    def run(self, environment: TaskEnvironment, *, reset: bool = True) -> EpisodeResult:
        started = time.time()
        history: list[StepResult] = []
        fingerprints: list[str] = []
        actions: list[str] = []
        parameters = inspect.signature(self.runtime_factory).parameters
        runtime = self.runtime_factory(environment) if parameters else self.runtime_factory()
        status = EpisodeStatus.ENVIRONMENT_ERROR
        reason = "episode did not start"
        try:
            if reset:
                environment.reset()
            for _ in range(self.config.max_steps):
                if time.time() - started > self.config.timeout_s:
                    status, reason = EpisodeStatus.TIMEOUT, "episode timeout"
                    break
                observation = environment.observe(history)
                observation = self._with_recent_actions(observation, history)
                fingerprint = state_fingerprint(observation)
                if self._is_stuck(fingerprints, actions):
                    status, reason = EpisodeStatus.STUCK, "state/action repetition limit reached"
                    break
                candidates = tuple(environment.candidates(observation, history))
                if not candidates:
                    status, reason = EpisodeStatus.TASK_FAILURE, "environment offered no legal actions"
                    break
                result = runtime.step(observation, candidates)
                history.append(result)
                fingerprints.append(fingerprint)
                actions.append(result.decision.candidate_id)
                candidate = next(item for item in candidates if item.id == result.decision.candidate_id)
                evaluation = environment.evaluate(observation, candidate, result.receipt, result.verification)
                runtime.trace.append(
                    "evaluation",
                    {
                        "success": evaluation.success,
                        "terminal": evaluation.terminal,
                        "reason": evaluation.reason,
                        "details": evaluation.details,
                    },
                )
                if evaluation.success:
                    status, reason = EpisodeStatus.SUCCESS, evaluation.reason
                    break
                if evaluation.terminal:
                    status, reason = EpisodeStatus.TASK_FAILURE, evaluation.reason
                    break
            else:
                status, reason = EpisodeStatus.MAX_STEPS, "maximum step budget exhausted"
        except CuaJevError as exc:
            name = type(exc).__name__
            status = {
                "PolicyError": EpisodeStatus.POLICY_ERROR,
                "GuardRejected": EpisodeStatus.GUARD_REJECTED,
            }.get(name, EpisodeStatus.EXECUTION_ERROR)
            reason = f"{name}: {exc}"
        except Exception as exc:
            status, reason = EpisodeStatus.ENVIRONMENT_ERROR, f"{type(exc).__name__}: {exc}"
        ended = time.time()
        counts = Counter(str(step.receipt.channel) for step in history)
        result = EpisodeResult(environment.name, status, reason, tuple(history), started, ended, dict(counts))
        runtime.trace.append("episode", result.to_dict())
        for role, adapter in (
            ("planner", getattr(environment, "planner", None)),
            ("vision", getattr(environment, "vision_grounder", None)),
        ):
            usage = getattr(adapter, "usage_totals", None)
            if isinstance(usage, dict) and usage:
                runtime.trace.append("model_usage", {
                    "role": role,
                    "model": str(getattr(adapter, "model", "unknown")),
                    "usage": {key: value for key, value in usage.items()
                              if isinstance(key, str) and type(value) is int and value >= 0},
                })
        completion_hold = _completion_hold_seconds()
        if completion_hold:
            time.sleep(completion_hold)
        close = getattr(runtime.policy, "close", None)
        if close:
            close()
        close_environment = getattr(environment, "close", None)
        if close_environment:
            close_environment()
        return result

    def _is_stuck(self, fingerprints: Sequence[str], actions: Sequence[str]) -> bool:
        state_stuck = (
            len(fingerprints) >= self.config.unchanged_limit
            and len(set(fingerprints[-self.config.unchanged_limit :])) == 1
        )
        action_stuck = (
            len(actions) >= self.config.repeated_action_limit
            and len(set(actions[-self.config.repeated_action_limit :])) == 1
        )
        return state_stuck and action_stuck

    @staticmethod
    def _with_recent_actions(observation: Observation, history: Sequence[StepResult]) -> Observation:
        recent = [
            {
                "candidate_id": step.decision.candidate_id,
                "channel": str(step.receipt.channel),
                "success": step.receipt.success,
                "verified": step.verification.passed,
            }
            for step in history[-8:]
        ]
        return replace(observation, state={**observation.state, "recent_actions": recent})


def _completion_hold_seconds() -> float:
    """Keep visible apps alive briefly so demo recorders can capture terminal state."""
    raw = os.getenv("CUA_JEV_COMPLETION_HOLD_SECONDS", "0")
    try:
        return min(max(float(raw), 0.0), 5.0)
    except ValueError:
        return 0.0
