"""A real macOS AX/physical-GUI action surface under the existing d: namespace."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .episode import Evaluation
from .executors.common import execute_with_receipt
from .macos_bridge import MacBridge
from .models import ActionCandidate, ActionReceipt, Channel, Observation, Risk, Verification
from .open_desktop import DesktopPlan
from .runtime import StepResult
from .verify import VerifierRegistry


@dataclass(frozen=True)
class MacSnapshot:
    """Semantic projection and private freshness state for one pinned AX window."""

    window_title: str
    native: dict[str, Any] = field(repr=False)

    @property
    def controls(self) -> tuple[Any, ...]:
        from types import SimpleNamespace
        return tuple(SimpleNamespace(**item) for item in self.native["controls"])

    @property
    def visual_targets(self) -> tuple:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": "macOS", "window_title": self.window_title,
            "bundle_id": self.native["bundle_id"], "window_id": self.native["window_id"],
            "bounds": self.native["bounds"], "text": self.native["text"],
            "controls": [{k: v for k, v in item.items() if k != "private_value"}
                         for item in self.native["controls"]],
            "visual_targets": [],
        }

    def for_model(self) -> dict[str, Any]:
        state = self.to_dict()
        for key in ("window_id", "bounds"):
            state.pop(key)
        state["controls"] = [{
            key: value for key, value in item.items()
            if key in {"ref", "name", "control_type", "role", "enabled", "can_invoke", "can_set_text"}
        } for item in state["controls"]]
        return state

    def acceptance_text(self) -> str:
        return "\n".join((self.native["text"], *(
            str(item.get(key, "")) for item in self.native["controls"] for key in ("name", "private_value")
        )))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.native, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_native(cls, native: dict[str, Any]) -> MacSnapshot:
        return cls(native["window_title"], deepcopy(native))


def validate_mac_option(state: MacSnapshot, ref: str, operation: Any, value: Any) -> tuple[str, str, str]:
    control = next((item for item in state.native["controls"] if item["ref"] == ref), None)
    if control is None or not control["enabled"]:
        raise ValueError("desktop ref is unavailable or disabled")
    if operation == "invoke" and control["can_invoke"]:
        operation = "click"
    if operation == "click" and value is None:
        value = ""
    if operation not in {"click", "fill"} or not isinstance(value, str) or len(value) > 1000:
        raise ValueError("unsupported desktop operation or value")
    if operation == "fill" and (not control["can_set_text"] or not value):
        raise ValueError("fill requires a live editable control and nonempty text")
    if operation == "click" and value:
        raise ValueError("click does not accept text")
    return ref, operation, value


class MacAccessibilitySurface:
    namespace = "d"
    supported_capabilities = frozenset({"desktop.click", "desktop.fill"})

    def __init__(
        self, *, window_title_re: str, bundle_id: str | None = None,
        allow_text_input: bool = False, allow_button_actions: bool = False,
        allow_gui: bool = True, bridge: Any | None = None, verify_timeout_s: float = 2,
        record_path: Path | None = None,
    ) -> None:
        if not window_title_re.strip():
            raise ValueError("select one window with a title pattern")
        self.window_title_re = window_title_re
        self.bundle_id = bundle_id
        self.allow_text_input = allow_text_input
        self.allow_button_actions = allow_button_actions
        self.allow_gui = allow_gui
        self.bridge = bridge or MacBridge()
        self.verify_timeout_s = verify_timeout_s
        self.record_path = record_path.resolve() if record_path else None
        self.recording = False
        self._observed: MacSnapshot | None = None
        self._offered: dict[str, tuple[ActionCandidate, MacSnapshot]] = {}
        self._before: dict[str, MacSnapshot] = {}

    def reset(self) -> None:
        if self.recording:
            raise RuntimeError("close the active window recording before resetting the surface")
        self._observed = None
        self._offered.clear()
        self._before.clear()
        if self.record_path:
            if self.record_path.exists() or self.record_path.suffix.lower() != ".mp4":
                raise ValueError("recording needs a new .mp4 path")
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.bridge.call("select", title_pattern=self.window_title_re, bundle_id=self.bundle_id)
        if self.record_path:
            self.bridge.call("record_start", path=str(self.record_path))
            self.recording = True

    def close(self) -> None:
        try:
            if self.recording:
                self.bridge.call("record_stop")
        finally:
            self.recording = False
            self._observed = None
            self._offered.clear()
            self._before.clear()
            self.bridge.close()

    def capture(self) -> MacSnapshot:
        return MacSnapshot.from_native(self.bridge.call("observe"))

    def observe(self, history: Sequence[StepResult]) -> MacSnapshot:
        self._observed = self.capture()
        self._offered.clear()
        return self._observed

    def validate(self, state: MacSnapshot, ref: str, operation: Any, value: Any) -> tuple[str, str, str]:
        return validate_mac_option(state, ref, operation, value)

    def compile(self, state: MacSnapshot, ref: str, operation: str, value: str,
                subgoal: str, index: int) -> tuple[ActionCandidate, ...]:
        ref, operation, value = self.validate(state, ref, operation, value)
        if (operation == "fill" and not self.allow_text_input) or (
            operation == "click" and not self.allow_button_actions
        ):
            return ()
        control = next(item for item in state.native["controls"] if item["ref"] == ref)
        result = []
        for channel, route in ((Channel.API, "ax"), (Channel.GUI, "gui")):
            if route == "gui" and (not self.allow_gui or not control.get("gui_available", False)):
                continue
            if route == "ax" and operation == "click" and not control["can_invoke"]:
                continue
            candidate = ActionCandidate(
                f"option_{index}_{route}", channel, f"desktop.{operation}",
                f"{operation.title()} {control['name'] or control['control_type']} using {route.upper()}",
                {"ref": ref, "operation": operation, "value": value,
                 "window_token": state.native["window_token"], "route": route},
                Risk.LOCAL_WRITE if operation == "fill" else Risk.EXTERNAL_SIDE_EFFECT,
                verifier="macos.effect", intent=f"option_{index}",
            )
            # Frozen dataclasses still contain mutable argument/state dictionaries.
            # Retain private copies so later edits cannot rewrite what was offered.
            self._offered[candidate.id] = (deepcopy(candidate), deepcopy(state))
            result.append(candidate)
        return tuple(result)

    def owns(self, candidate: ActionCandidate) -> bool:
        return candidate.capability in self.supported_capabilities

    def execute(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            offered = self._offered.get(candidate.id)
            if offered is None or offered[0] != candidate:
                raise ValueError("candidate was not compiled from this observation")
            # One observation permits one attempt, even when an action has no
            # visible effect or fails after partially delivering native input.
            self._offered.clear()
            current = self.capture()
            if current.fingerprint() != offered[1].fingerprint():
                raise RuntimeError("macOS window or control changed since observation")
            self._before[decision_id] = current
            return self.bridge.call("act", **candidate.arguments)
        return execute_with_receipt(candidate, observation_id, decision_id, operation)

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("macos.effect", self.verify_effect)

    def verify_effect(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        before = self._before.pop(receipt.decision_id, None)
        if not receipt.success or before is None:
            return Verification(False, "macos.effect", {"error": receipt.error or "missing before state"})
        deadline = time.monotonic() + self.verify_timeout_s
        while True:
            try:
                after = self.capture()
            except (RuntimeError, ValueError) as exc:
                return Verification(False, "macos.effect", {"error": str(exc)})
            if candidate.capability == "desktop.fill":
                control = next((item for item in after.native["controls"]
                                if item["ref"] == candidate.arguments["ref"]), None)
                passed = control is not None and control["private_value"] == candidate.arguments["value"]
            else:
                # Exclude geometry-only changes from action-effect evidence.
                def semantic(state):
                    return (state.window_title, state.native["text"], [
                        {k: v for k, v in item.items() if k not in {"rectangle", "gui_available"}}
                        for item in state.native["controls"]
                    ])
                passed = semantic(before) != semantic(after)
            if passed or time.monotonic() >= deadline:
                return Verification(bool(passed), "macos.effect", {"state_changed": bool(passed)})
            time.sleep(0.05)


class MacDesktopTask:
    """Standalone desktop loop; the same surface can also join open-computer."""

    name = "open-desktop"

    def __init__(self, *, goal: str, planner: Any, window_title_re: str,
                 required_text: str = "", required_title: str = "", **kwargs: Any) -> None:
        if not (required_text or required_title):
            raise ValueError("macOS desktop tasks require a caller-supplied text or title acceptance gate")
        self.goal, self.planner = goal, planner
        self.required_text, self.required_title = required_text, required_title
        self.surface = MacAccessibilitySurface(window_title_re=window_title_re, **kwargs)
        self._snapshot: MacSnapshot | None = None
        self._plan: DesktopPlan | None = None
        self.planner_calls = 0
        self.vision_failures = 0

    def reset(self) -> None:
        self.surface.reset()
        self._plan = None
        self.planner_calls = 0

    def close(self) -> None:
        self.surface.close()

    def observe(self, history: Sequence[StepResult]) -> Observation:
        self._snapshot = self.surface.observe(history)
        return Observation(self.goal, "Advance the selected macOS window task", self._snapshot.to_dict(),
                           source=self.name)

    def _satisfied(self, state: MacSnapshot) -> bool:
        return (not self.required_text or
                self.required_text.casefold() in state.acceptance_text().casefold()) and (
            not self.required_title or self.required_title.casefold() in state.window_title.casefold()
        )

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> tuple[ActionCandidate, ...]:
        assert self._snapshot is not None
        if self._satisfied(self._snapshot):
            return (ActionCandidate("done", Channel.CONTROL, "control.done", "Caller acceptance passed"),)
        self.planner_calls += 1
        self._plan = self.planner.plan(self.goal, self._snapshot, [
            {"candidate_id": step.decision.candidate_id, "verified": step.verification.passed}
            for step in history[-6:]
        ])
        return tuple(candidate for index, option in enumerate(self._plan.options)
                     for candidate in self.surface.compile(self._snapshot, option.ref, option.operation,
                                                           option.value, self._plan.subgoal, index))

    def executor_bindings(self) -> dict[Channel, Any]:
        return {Channel.API: self.surface.execute, Channel.GUI: self.surface.execute}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        self.surface.register_verifiers(registry)

    def evaluate(self, observation, candidate, receipt, verification) -> Evaluation:
        passed = verification.passed and self._satisfied(self.surface.capture())
        return Evaluation(passed, passed, "acceptance_passed" if passed else "continue")
