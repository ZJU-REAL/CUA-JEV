"""Experimental, bounded browser + Windows-window + local-tool action loop.

This composes existing observers/executors instead of encoding an application
workflow. Each plan names live refs, while terminal requirements come from the
caller rather than from a model's self-assessment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .episode import Evaluation
from .executors.cli import RegisteredCliExecutor
from .executors.filesystem import FileSystemExecutor
from .local_tools import ScopedReadOnlyTools, ToolOffer
from .models import ActionCandidate, ActionReceipt, Channel, Observation, Verification
from .open_browser import BrowserPlan, BrowserSnapshot, OpenBrowserTask, PlannedOption
from .open_desktop import DesktopOption, DesktopPlan, DesktopSnapshot, OpenDesktopTask
from .runtime import StepResult
from .verify import VerifierRegistry

MAX_OPTIONS = 16


@dataclass(frozen=True)
class ComputerSnapshot:
    browser: BrowserSnapshot
    desktop: DesktopSnapshot
    tool_offers: tuple[ToolOffer, ...]
    tool_results: tuple[dict[str, Any], ...]
    requirements: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        browser = self.browser.to_dict()
        desktop = self.desktop.to_dict()
        for key in ("elements", "visual_targets"):
            browser[key] = [{**item, "ref": f"b:{item['ref']}"} for item in browser[key]]
        for key in ("controls", "visual_targets"):
            desktop[key] = [{**item, "ref": f"d:{item['ref']}"} for item in desktop[key]]
        return {
            "browser": browser, "desktop": desktop,
            "tool_offers": [
                {**offer.to_dict(), "ref": f"t:{offer.ref}"} for offer in self.tool_offers
            ],
            "tool_results": list(self.tool_results),
            "requirements": self.requirements,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ComputerOption:
    source: str
    ref: str
    operation: str
    value: str = ""


@dataclass(frozen=True)
class ComputerPlan:
    subgoal: str
    options: tuple[ComputerOption, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], snapshot: ComputerSnapshot) -> ComputerPlan:
        if not isinstance(data, dict) or not isinstance(data.get("options"), list):
            raise ValueError("computer planner must return an options list")
        subgoal = data.get("subgoal")
        if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 240:
            raise ValueError("computer planner subgoal is invalid")
        if not 1 <= len(data["options"]) <= MAX_OPTIONS:
            raise ValueError("computer planner must return 1-16 options")
        offers = {item.ref for item in snapshot.tool_offers}
        options: list[ComputerOption] = []
        for item in data["options"]:
            if not isinstance(item, dict) or not isinstance(item.get("ref"), str):
                raise ValueError("computer option must reference one live target")
            namespace, separator, ref = item["ref"].partition(":")
            if not separator or namespace not in {"b", "d", "t"}:
                raise ValueError("computer option must use a b:/d:/t: ref")
            normalized = {**item, "ref": ref}
            if namespace == "b":
                checked = BrowserPlan.from_dict({
                    "subgoal": subgoal, "options": [normalized],
                    "success": {"kind": "url_contains", "value": "__not_a_terminal_check__"},
                }, snapshot.browser).options[0]
                options.append(ComputerOption("browser", checked.ref, checked.operation, checked.value))
            elif namespace == "d":
                control = next(
                    (control for control in snapshot.desktop.controls if control.ref == ref), None
                )
                visual = next(
                    (target for target in snapshot.desktop.visual_targets if target.ref == ref), None
                )
                if control is None and visual is None:
                    raise ValueError("desktop ref is not present in the selected window")
                if control is not None and (
                    control.control_type not in {
                        "Button", "Edit", "CheckBox", "ComboBox", "MenuItem", "ListItem"
                    }
                    or any(token in control.name.casefold() for token in (
                        "close", "minimize", "maximize", "关闭", "最小化", "最大化"
                    ))
                ):
                    raise ValueError("desktop target is system chrome or not an app control")
                if normalized.get("operation") == "invoke" and control is not None and control.can_invoke:
                    normalized["operation"] = "click"
                if normalized.get("value") is None and normalized.get("operation") == "click":
                    normalized["value"] = ""
                checked = DesktopPlan.from_dict({
                    "subgoal": subgoal, "options": [normalized],
                    "success": {"kind": "window_title_contains", "value": "__not_a_terminal_check__"},
                }, snapshot.desktop).options[0]
                options.append(ComputerOption("desktop", checked.ref, checked.operation, checked.value))
            else:
                if ref not in offers or item.get("operation") != "invoke" or item.get("value"):
                    raise ValueError("local tool option must invoke one offered t: ref")
                options.append(ComputerOption("tool", ref, "invoke"))
        if len({(o.source, o.ref, o.operation, o.value) for o in options}) != len(options):
            raise ValueError("computer planner returned duplicate options")
        return cls(subgoal.strip(), tuple(options))


class ComputerGoalPlanner(Protocol):
    def plan(
        self, goal: str, snapshot: ComputerSnapshot, recent_actions: Sequence[dict[str, Any]]
    ) -> ComputerPlan: ...


class OpenComputerTask:
    """One browser origin + one selected Windows window + optional local tools."""

    name = "open-computer"

    def __init__(
        self,
        *,
        goal: str,
        browser: OpenBrowserTask,
        desktop: OpenDesktopTask,
        planner: ComputerGoalPlanner,
        local_tool_root: Path | None = None,
        required_url_contains: str = "",
        required_window_title_contains: str = "",
        required_window_text_contains: str = "",
        required_capabilities: Sequence[str] = (),
    ) -> None:
        if not goal.strip() or browser.goal != goal.strip() or desktop.goal != goal.strip():
            raise ValueError("the composite goal must match both observers")
        if not any((required_url_contains, required_window_title_contains,
                    required_window_text_contains)):
            raise ValueError("at least one caller-supplied observable state gate is required")
        self.goal = goal.strip()
        self.browser = browser
        self.desktop = desktop
        self.planner = planner
        self.local_tools = ScopedReadOnlyTools(local_tool_root) if local_tool_root else None
        self.required_url_contains = required_url_contains
        self.required_window_title_contains = required_window_title_contains
        self.required_window_text_contains = required_window_text_contains
        self.required_capabilities = tuple(required_capabilities)
        supported_actions = {
            "browser.click", "browser.fill", "browser.select", "browser.visual_click",
            "desktop.click", "desktop.fill", "desktop.visual_click",
        }
        for capability in self.required_capabilities:
            if capability not in supported_actions and (
                self.local_tools is None or capability not in {
                    offer.capability for offer in self.local_tools.offers()
                }
            ):
                raise ValueError("required capability is not registered in this task")
        self._snapshot: ComputerSnapshot | None = None
        self._plan: ComputerPlan | None = None
        self._plan_fingerprint = ""
        self._used: set[int] = set()
        self._completed_capabilities: set[str] = set()
        self._tool_results: list[dict[str, Any]] = []
        self._file_executor = FileSystemExecutor()
        self._cli_executor = RegisteredCliExecutor()
        self.planner_calls = 0

    def reset(self) -> None:
        self.browser.reset()
        self.desktop.reset()
        self._plan = None
        self._used.clear()
        self._completed_capabilities.clear()
        self._tool_results.clear()

    def close(self) -> None:
        self.desktop.close()
        self.browser.close()

    def _capture(self) -> ComputerSnapshot:
        return ComputerSnapshot(
            browser=self.browser._capture(), desktop=self.desktop._capture(),
            tool_offers=self.local_tools.offers() if self.local_tools else (),
            tool_results=tuple(self._tool_results[-6:]),
            requirements={
                "url_contains": self.required_url_contains,
                "window_title_contains": self.required_window_title_contains,
                "window_text_contains": self.required_window_text_contains,
                "capabilities": list(self.required_capabilities),
                "completed_capabilities": sorted(self._completed_capabilities),
            },
        )

    def observe(self, history: Sequence[StepResult]) -> Observation:
        self.browser.observe(history)
        self.desktop.observe(history)
        assert self.browser._snapshot is not None and self.desktop._snapshot is not None
        self._snapshot = ComputerSnapshot(
            browser=self.browser._snapshot, desktop=self.desktop._snapshot,
            tool_offers=self.local_tools.offers() if self.local_tools else (),
            tool_results=tuple(self._tool_results[-6:]),
            requirements={
                "url_contains": self.required_url_contains,
                "window_title_contains": self.required_window_title_contains,
                "window_text_contains": self.required_window_text_contains,
                "capabilities": list(self.required_capabilities),
                "completed_capabilities": sorted(self._completed_capabilities),
            },
        )
        return Observation(
            task=self.goal,
            subgoal=self._plan.subgoal if self._plan else "Choose the next grounded cross-app action",
            state=self._snapshot.to_dict(), source=self.name,
        )

    def _satisfied(self, snapshot: ComputerSnapshot) -> bool:
        requirements = (
            not self.required_url_contains or
            self.required_url_contains.casefold() in snapshot.browser.url.casefold(),
            not self.required_window_title_contains or
            self.required_window_title_contains.casefold()
            in snapshot.desktop.window_title.casefold(),
            not self.required_window_text_contains or
            self.required_window_text_contains.casefold() in (
                snapshot.desktop.text + "\n" + "\n".join(
                    item.name for item in snapshot.desktop.controls
                )
            ).casefold(),
            all(item in self._completed_capabilities for item in self.required_capabilities),
        )
        return all(requirements)

    def _call_planner(self, history: Sequence[StepResult]) -> None:
        assert self._snapshot is not None
        recent = [
            {"candidate_id": step.decision.candidate_id, "verified": step.verification.passed}
            for step in history[-6:]
        ]
        for attempt in range(2):
            self.planner_calls += 1
            try:
                self._plan = self.planner.plan(self.goal, self._snapshot, recent)
                break
            except ValueError:
                if attempt:
                    raise
                recent.append({
                    "planner_feedback": "Previous plan failed validation. Reuse only current refs "
                    "and supported operations; no action was executed."
                })
        self._plan_fingerprint = self._snapshot.fingerprint()
        self._used.clear()

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]:
        if self._snapshot is None:
            raise RuntimeError("observe before candidate generation")
        if self._satisfied(self._snapshot):
            return (ActionCandidate(
                "done", Channel.CONTROL, "control.done", "Finish after caller acceptance gates passed.",
            ),)
        if self._plan is None or self._plan_fingerprint != self._snapshot.fingerprint():
            self._call_planner(history)
        result = self._build_candidates()
        if not result:
            self._call_planner(history)
            result = self._build_candidates()
        return result

    def _build_candidates(self) -> tuple[ActionCandidate, ...]:
        assert self._snapshot is not None and self._plan is not None
        tools = {offer.ref: offer for offer in self._snapshot.tool_offers}
        result: list[ActionCandidate] = []
        for index, option in enumerate(self._plan.options):
            if index in self._used:
                continue
            if option.source == "tool":
                tool = tools.get(option.ref)
                if tool is not None and tool.capability not in self._completed_capabilities:
                    result.append(tool.candidate(index))
                continue
            if option.source == "browser":
                self.browser._snapshot = self._snapshot.browser
                self.browser._plan = BrowserPlan(
                    self._plan.subgoal, (PlannedOption(option.ref, option.operation, option.value),),
                    "url_contains", "__not_a_terminal_check__",
                )
                self.browser._used.clear()
                candidates = self.browser._build_candidates()
            else:
                self.desktop._snapshot = self._snapshot.desktop
                self.desktop._plan = DesktopPlan(
                    self._plan.subgoal, (DesktopOption(option.ref, option.operation, option.value),),
                    "window_title_contains", "__not_a_terminal_check__",
                )
                self.desktop._used.clear()
                candidates = self.desktop._build_candidates()
            for candidate in candidates:
                suffix = candidate.id.removeprefix("option_0_")
                result.append(replace(
                    candidate, id=f"option_{index}_{suffix}", intent=f"option_{index}",
                ))
        return tuple(result)

    def executor_bindings(self) -> dict[Channel, Any]:
        return {channel: self for channel in (Channel.API, Channel.CLI, Channel.SCRIPT, Channel.GUI)}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        self.browser.register_verifiers(registry)
        self.desktop.register_verifiers(registry)
        registry.register("tool.result", self.browser._verify_tool_result)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        if candidate.capability.startswith("browser."):
            return self.browser(candidate, observation_id, decision_id)
        if candidate.capability.startswith("desktop."):
            if candidate.channel == Channel.GUI:
                assert self.desktop._snapshot is not None
                self.desktop.screen.focus_handle(self.desktop._snapshot.window_handle, maximize=False)
            return self.desktop(candidate, observation_id, decision_id)
        if candidate.capability.startswith("filesystem.") and self.local_tools is not None:
            return self._file_executor(candidate, observation_id, decision_id)
        if candidate.capability.startswith("cli.") and self.local_tools is not None:
            return self._cli_executor(candidate, observation_id, decision_id)
        raise ValueError("no registered executor for the selected capability")

    def evaluate(
        self,
        observation: Observation,
        candidate: ActionCandidate,
        receipt: ActionReceipt,
        verification: Verification,
    ) -> Evaluation:
        if candidate.capability == "control.done":
            passed = self._satisfied(self._capture())
            return Evaluation(passed, True, "acceptance_passed" if passed else "acceptance_failed")
        if not verification.passed:
            self._plan = None
            return Evaluation(False, False, "action_unverified_replan")
        self._used.add(int(candidate.intent.removeprefix("option_")))
        self._completed_capabilities.add(candidate.capability)
        if candidate.capability.startswith(("filesystem.", "cli.")):
            output = receipt.output
            summary = output.get("text", output.get("stdout", output.get("entries", "")))
            self._tool_results.append({
                "capability": candidate.capability, "result": str(summary)[:2000],
                "verified": True,
            })
        if self._satisfied(self._capture()):
            return Evaluation(True, True, "acceptance_passed")
        return Evaluation(False, False, "action_verified")
