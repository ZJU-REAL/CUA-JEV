"""Experimental task-agnostic Windows UIA observer and grounded action loop.

This covers accessible controls in one explicitly selected window. It is not
visual grounding, application-level reasoning, or arbitrary Windows automation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .episode import Evaluation
from .errors import CapabilityUnavailable
from .executors.common import execute_with_receipt
from .executors.screen import ScreenController
from .models import ActionCandidate, ActionReceipt, Channel, Observation, Risk, Verification
from .runtime import StepResult
from .verify import VerifierRegistry

MAX_CONTROLS = 80
MAX_OPTIONS = 16


@dataclass(frozen=True)
class DesktopControl:
    ref: str
    index: int
    name: str
    control_type: str
    automation_id: str
    enabled: bool
    rectangle: tuple[int, int, int, int]
    can_invoke: bool
    can_set_text: bool

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class DesktopSnapshot:
    window_title: str
    window_handle: int
    text: str
    controls: tuple[DesktopControl, ...]

    @classmethod
    def capture(cls, window: Any) -> tuple[DesktopSnapshot, dict[str, Any]]:
        controls: list[DesktopControl] = []
        handles: dict[str, Any] = {}
        text_parts: list[str] = []
        for index, wrapper in enumerate(window.descendants()[:400]):
            try:
                info = wrapper.element_info
                control_type = str(info.control_type or "")
                name = str(wrapper.window_text() or getattr(info, "name", "") or "").strip()[:120]
                password = getattr(info, "is_password", False)
                if callable(password):
                    password = password()
                if not wrapper.is_visible() or bool(password):
                    continue
                rectangle = wrapper.rectangle()
                bounds = (
                    int(rectangle.left), int(rectangle.top),
                    int(rectangle.right), int(rectangle.bottom),
                )
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    continue
                if name and control_type in {"Text", "Document", "StatusBar"}:
                    text_parts.append(name)
                can_invoke = callable(getattr(wrapper, "invoke", None))
                can_set_text = callable(getattr(wrapper, "set_edit_text", None))
                if not (name or can_invoke or can_set_text):
                    continue
                ref = f"c{index}"
                control = DesktopControl(
                    ref=ref, index=index, name=name, control_type=control_type,
                    automation_id=str(getattr(info, "automation_id", "") or "")[:120],
                    enabled=bool(wrapper.is_enabled()), rectangle=bounds,
                    can_invoke=can_invoke, can_set_text=can_set_text,
                )
                controls.append(control)
                handles[ref] = wrapper
                if len(controls) >= MAX_CONTROLS:
                    break
            except Exception:
                continue
        snapshot = cls(
            window_title=str(window.window_text())[:160], window_handle=int(window.handle),
            text="\n".join(text_parts)[:1600], controls=tuple(controls),
        )
        return snapshot, handles

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_title": self.window_title, "window_handle": self.window_handle,
            "text": self.text, "controls": [item.to_dict() for item in self.controls],
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DesktopOption:
    ref: str
    operation: str
    value: str = ""


@dataclass(frozen=True)
class DesktopPlan:
    subgoal: str
    options: tuple[DesktopOption, ...]
    success_kind: str
    success_value: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], snapshot: DesktopSnapshot) -> DesktopPlan:
        if not isinstance(data, dict) or not isinstance(data.get("options"), list):
            raise ValueError("desktop planner must return an options list")
        if not 1 <= len(data["options"]) <= MAX_OPTIONS:
            raise ValueError("desktop planner must return 1-16 options")
        known = {item.ref: item for item in snapshot.controls}
        options: list[DesktopOption] = []
        for item in data["options"]:
            if not isinstance(item, dict):
                raise ValueError("desktop planner option must be an object")
            ref, operation, value = item.get("ref"), item.get("operation"), item.get("value", "")
            if not isinstance(ref, str) or ref not in known or operation not in {"click", "fill"}:
                raise ValueError("desktop planner referenced an unavailable control or operation")
            control = known[ref]
            if not control.enabled:
                raise ValueError("desktop planner referenced a disabled control")
            if not isinstance(value, str) or len(value) > 1000:
                raise ValueError("desktop planner value is invalid")
            if operation == "fill" and (not value or not control.can_set_text):
                raise ValueError("fill requires an editable control and nonempty value")
            options.append(DesktopOption(ref, operation, value))
        if len({(item.ref, item.operation, item.value) for item in options}) != len(options):
            raise ValueError("desktop planner returned duplicate options")
        subgoal = data.get("subgoal")
        success = data.get("success")
        if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 240:
            raise ValueError("desktop planner subgoal is invalid")
        if not isinstance(success, dict) or success.get("kind") not in {
            "window_title_contains", "text_contains", "control_exists"
        }:
            raise ValueError("desktop planner success check is invalid")
        value = success.get("value")
        if not isinstance(value, str) or not value or len(value) > 160:
            raise ValueError("desktop planner success value is invalid")
        return cls(subgoal.strip(), tuple(options), str(success["kind"]), value)


class DesktopGoalPlanner(Protocol):
    def plan(
        self, goal: str, snapshot: DesktopSnapshot, recent_actions: Sequence[dict[str, Any]]
    ) -> DesktopPlan: ...


class OpenDesktopTask:
    """One-window UIA task with dynamic model-proposed control refs and two routes."""

    name = "open-desktop"

    def __init__(
        self,
        *,
        goal: str,
        window_title_re: str,
        planner: DesktopGoalPlanner,
        window: Any | None = None,
        screen: ScreenController | None = None,
        allow_text_input: bool = False,
        allow_button_actions: bool = False,
    ) -> None:
        if not goal.strip() or not window_title_re.strip():
            raise ValueError("goal and window title regex are required")
        self.goal = goal.strip()
        self.window_title_re = window_title_re
        self.planner = planner
        self.window = window
        self.screen = screen or ScreenController()
        self.allow_text_input = allow_text_input
        self.allow_button_actions = allow_button_actions
        self._snapshot: DesktopSnapshot | None = None
        self._handles: dict[str, Any] = {}
        self._plan: DesktopPlan | None = None
        self._plan_fingerprint = ""
        self._used: set[int] = set()
        self._before: dict[str, DesktopSnapshot] = {}
        self.planner_calls = 0

    def reset(self) -> None:
        if self.window is None:
            try:
                from pywinauto import Desktop
            except ImportError:
                raise CapabilityUnavailable("install cua-jev[windows] for open desktop tasks") from None
            matches = Desktop(backend="uia").windows(
                title_re=self.window_title_re, visible_only=True
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"window title regex must match exactly one visible window; found {len(matches)}"
                )
            self.window = self.screen.focus_handle(matches[0].handle, maximize=False)
        self._plan = None
        self._used.clear()

    def close(self) -> None:
        self.window = None

    def _capture(self) -> DesktopSnapshot:
        if self.window is None:
            raise RuntimeError("desktop task has not been reset")
        snapshot, handles = DesktopSnapshot.capture(self.window)
        self._handles = handles
        return snapshot

    def observe(self, history: Sequence[StepResult]) -> Observation:
        self._snapshot = self._capture()
        return Observation(
            task=self.goal,
            subgoal=self._plan.subgoal if self._plan else "Discover the next desktop subgoal",
            state=self._snapshot.to_dict(), source=self.name,
        )

    def _satisfied(self, snapshot: DesktopSnapshot) -> bool:
        if self._plan is None:
            return False
        expected = self._plan.success_value.casefold()
        if self._plan.success_kind == "control_exists":
            return any(expected in item.name.casefold() for item in snapshot.controls)
        actual = (
            snapshot.window_title
            if self._plan.success_kind == "window_title_contains" else snapshot.text
        )
        return expected in actual.casefold()

    def _call_planner(self, history: Sequence[StepResult]) -> None:
        assert self._snapshot is not None
        recent = [
            {"candidate_id": step.decision.candidate_id, "verified": step.verification.passed}
            for step in history[-6:]
        ]
        self._plan = self.planner.plan(self.goal, self._snapshot, recent)
        self._plan_fingerprint = self._snapshot.fingerprint()
        self._used.clear()
        self.planner_calls += 1

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]:
        if self._snapshot is None:
            raise RuntimeError("observe before candidate generation")
        if self._plan is None or self._plan_fingerprint != self._snapshot.fingerprint():
            self._call_planner(history)
        if self._satisfied(self._snapshot):
            return (ActionCandidate("done", Channel.CONTROL, "control.done", "Finish after live check."),)
        result = self._build_candidates()
        if not result:
            self._call_planner(history)
            result = self._build_candidates()
        return result

    def _build_candidates(self) -> tuple[ActionCandidate, ...]:
        assert self._snapshot is not None and self._plan is not None
        known = {item.ref: item for item in self._snapshot.controls}
        result: list[ActionCandidate] = []
        for index, option in enumerate(self._plan.options):
            if index in self._used:
                continue
            control = known.get(option.ref)
            if control is None or not control.enabled:
                continue
            if option.operation == "fill" and not self.allow_text_input:
                continue
            if option.operation == "click" and not self.allow_button_actions:
                continue
            arguments = {
                "ref": control.ref, "index": control.index, "name": control.name,
                "control_type": control.control_type, "automation_id": control.automation_id,
                "operation": option.operation, "value": option.value,
            }
            risk = Risk.LOCAL_WRITE if option.operation == "fill" else Risk.EXTERNAL_SIDE_EFFECT
            for channel, suffix in ((Channel.API, "uia"), (Channel.GUI, "gui")):
                if channel == Channel.API and option.operation == "click" and not control.can_invoke:
                    continue
                result.append(
                    ActionCandidate(
                        id=f"option_{index}_{suffix}", channel=channel,
                        capability=f"desktop.{option.operation}",
                        description=f"{option.operation.title()} {control.name or control.control_type} "
                        f"using {suffix.upper()}",
                        arguments=arguments, risk=risk, verifier="desktop.effect",
                        intent=f"option_{index}",
                    )
                )
        return tuple(result)

    def executor_bindings(self) -> dict[Channel, Any]:
        return {Channel.API: self, Channel.GUI: self}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("desktop.effect", self._verify_effect)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            before = self._capture()
            self._before[decision_id] = before
            ref = candidate.arguments["ref"]
            control = next((item for item in before.controls if item.ref == ref), None)
            if control is None or any(
                getattr(control, field) != candidate.arguments[field]
                for field in ("name", "control_type", "automation_id")
            ):
                raise RuntimeError("desktop control changed since observation")
            wrapper = self._handles[ref]
            action = candidate.arguments["operation"]
            if candidate.channel == Channel.API:
                if action == "click":
                    wrapper.invoke()
                else:
                    wrapper.set_edit_text(candidate.arguments["value"])
            elif candidate.channel == Channel.GUI:
                left, top, right, bottom = control.rectangle
                self.screen.click_point((left + right) / 2, (top + bottom) / 2)
                if action == "fill":
                    self.screen.hotkey("ctrl", "a")
                    self.screen.paste_text(candidate.arguments["value"])
            else:
                raise ValueError("unsupported desktop route")
            return {"control": ref, "operation": action}

        return execute_with_receipt(candidate, observation_id, decision_id, operation)

    def _verify_effect(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        if not receipt.success:
            return Verification(False, "desktop.effect", {"error": receipt.error})
        before = self._before.get(receipt.decision_id)
        if before is None:
            return Verification(False, "desktop.effect", {"error": "missing pre-action snapshot"})
        after = self._capture()
        if candidate.arguments["operation"] == "fill":
            wrapper = self._handles.get(candidate.arguments["ref"])
            read_value = getattr(wrapper, "get_value", None)
            passed = callable(read_value) and read_value() == candidate.arguments["value"]
        else:
            passed = after.fingerprint() != before.fingerprint()
        return Verification(bool(passed), "desktop.effect", {"state_changed": bool(passed)})

    def evaluate(
        self,
        observation: Observation,
        candidate: ActionCandidate,
        receipt: ActionReceipt,
        verification: Verification,
    ) -> Evaluation:
        if candidate.capability == "control.done":
            valid = self._satisfied(self._capture())
            return Evaluation(valid, True, "goal_check_passed" if valid else "goal_check_failed")
        if not verification.passed:
            self._plan = None
            return Evaluation(False, False, "action_unverified_replan")
        self._used.add(int(candidate.intent.removeprefix("option_")))
        if self._satisfied(self._capture()):
            return Evaluation(True, True, "goal_check_passed")
        return Evaluation(False, False, "action_verified")
