from __future__ import annotations

import inspect
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from .errors import CuaJevError
from .models import ActionCandidate, ActionReceipt, Observation, Verification, state_fingerprint
from .runtime import AgentRuntime, StepDeadlineExceeded, StepResult


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

    def __post_init__(self) -> None:
        for name in ("max_steps", "unchanged_limit", "repeated_action_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (type(self.timeout_s) not in (int, float)
                or not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout_s must be a positive finite number")


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
        deadline = time.monotonic() + self.config.timeout_s
        history: list[StepResult] = []
        runtime: AgentRuntime | None = None
        status = EpisodeStatus.ENVIRONMENT_ERROR
        reason = "episode did not start"
        failures: list[str] = []
        interrupted = True
        try:
            parameters = inspect.signature(self.runtime_factory).parameters
            runtime = self.runtime_factory(environment) if parameters else self.runtime_factory()
            if reset:
                environment.reset()
            status, reason = self._run_steps(environment, runtime, history, deadline)
            interrupted = False
        except StepDeadlineExceeded as exc:
            status, reason = EpisodeStatus.TIMEOUT, str(exc)
            interrupted = False
        except CuaJevError as exc:
            name = type(exc).__name__
            status = {
                "PolicyError": EpisodeStatus.POLICY_ERROR,
                "GuardRejected": EpisodeStatus.GUARD_REJECTED,
            }.get(name, EpisodeStatus.EXECUTION_ERROR)
            reason = f"{name}: {exc}"
            interrupted = False
        except Exception as exc:
            status, reason = EpisodeStatus.ENVIRONMENT_ERROR, f"{type(exc).__name__}: {exc}"
            interrupted = False
        finally:
            # Telemetry and demo presentation must never bypass resource cleanup.
            # Cancellation still propagates, after releasing the owned resources.
            try:
                if runtime is not None and not interrupted:
                    try:
                        self._record_model_usage(environment, runtime)
                        completion_hold = _completion_hold_seconds()
                        if completion_hold:
                            time.sleep(completion_hold)
                    except Exception as exc:
                        failures.append(f"finalization failed: {type(exc).__name__}: {exc}")
            finally:
                active_exception = sys.exception()
                cleanup_interrupt: BaseException | None = None
                resources = (runtime.policy, environment) if runtime is not None else (environment,)
                for resource in resources:
                    try:
                        close = getattr(resource, "close", None)
                        if close:
                            close()
                    except Exception as exc:
                        failures.append(f"cleanup failed: {type(exc).__name__}: {exc}")
                    except BaseException as exc:
                        # A cancellation during one close must not leak the
                        # remaining resources or suppress an earlier interrupt.
                        if cleanup_interrupt is None:
                            cleanup_interrupt = exc
                if cleanup_interrupt is not None and active_exception is None:
                    raise cleanup_interrupt
        if failures:
            status = EpisodeStatus.ENVIRONMENT_ERROR
            reason = "; ".join((reason, *failures))
        counts = Counter(str(step.receipt.channel) for step in history)
        result = EpisodeResult(
            environment.name, status, reason, tuple(history), started, time.time(), dict(counts),
        )
        if runtime is not None:
            runtime.trace.append("episode", result.to_dict())
        return result

    def _run_steps(
        self, environment: TaskEnvironment, runtime: AgentRuntime,
        history: list[StepResult], deadline: float,
    ) -> tuple[EpisodeStatus, str]:
        fingerprints: list[str] = []
        actions: list[str] = []
        for _ in range(self.config.max_steps):
            if time.monotonic() >= deadline:
                return EpisodeStatus.TIMEOUT, "episode timeout"
            observation = environment.observe(history)
            if time.monotonic() >= deadline:
                return EpisodeStatus.TIMEOUT, "episode timeout after observation"
            observation = self._with_recent_actions(observation, history)
            fingerprint = state_fingerprint(observation)
            if self._is_stuck(fingerprints, actions):
                return EpisodeStatus.STUCK, "state/action repetition limit reached"
            candidates = tuple(environment.candidates(observation, history))
            if time.monotonic() >= deadline:
                return EpisodeStatus.TIMEOUT, "episode timeout after planning"
            if not candidates:
                return EpisodeStatus.TASK_FAILURE, "environment offered no legal actions"
            result = runtime.step(observation, candidates, deadline=deadline)
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
            # Executors have their own I/O limits; an in-flight action cannot be
            # rolled back at the deadline. Keep its receipt, but report timeout.
            if time.monotonic() >= deadline:
                return EpisodeStatus.TIMEOUT, "episode timeout after action verification"
            if evaluation.success:
                return EpisodeStatus.SUCCESS, evaluation.reason
            if evaluation.terminal:
                return EpisodeStatus.TASK_FAILURE, evaluation.reason
        return EpisodeStatus.MAX_STEPS, "maximum step budget exhausted"

    @staticmethod
    def _record_model_usage(environment: TaskEnvironment, runtime: AgentRuntime) -> None:
        adapters = (
            ("planner", getattr(environment, "planner", None)),
            ("vision", getattr(environment, "vision_grounder", None)),
        )
        model_adapters = getattr(environment, "model_adapters", None)
        if model_adapters is not None:
            adapters = model_adapters()
        for role, adapter in adapters:
            usage = getattr(adapter, "usage_totals", None)
            requests = (getattr(adapter, "planning_requests", 0)
                        + getattr(adapter, "vision_requests", 0))
            if isinstance(usage, dict) and (usage or requests):
                runtime.trace.append("model_usage", {
                    "role": role,
                    "model": str(getattr(adapter, "model", "unknown")),
                    "usage": {key: value for key, value in usage.items()
                              if isinstance(key, str) and type(value) is int and value >= 0},
                    "requests": requests,
                })

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
        value = float(raw)
        return min(max(value, 0.0), 5.0) if math.isfinite(value) else 0.0
    except ValueError:
        return 0.0
