"""Experimental, bounded browser + Windows-window + local-tool action loop.

This composes existing observers/executors instead of encoding an application
workflow. Each plan names live refs, while terminal requirements come from the
caller rather than from a model's self-assessment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .episode import Evaluation
from .local_tools import ScopedReadOnlyTools, ToolOffer
from .mcp_surface import McpCallOffer, McpToolSurface
from .models import ActionCandidate, ActionReceipt, Channel, Observation, Verification
from .open_browser import BrowserSnapshot, OpenBrowserTask
from .open_desktop import DesktopSnapshot, OpenDesktopTask
from .runtime import StepResult
from .surface_providers import (
    ActionSurface,
    BrowserSurface,
    ReadOnlyToolSurface,
    WindowsUiaSurface,
)
from .verify import VerifierRegistry

MAX_OPTIONS = 16


@dataclass(frozen=True)
class ComputerSnapshot:
    browser: BrowserSnapshot
    desktop: DesktopSnapshot
    tool_offers: tuple[ToolOffer, ...]
    tool_results: tuple[dict[str, Any], ...]
    requirements: dict[str, Any]
    mcp_offers: tuple[McpCallOffer, ...] = ()
    providers: dict[str, ActionSurface] = field(default_factory=dict, repr=False, compare=False)

    def state_for(self, namespace: str) -> Any:
        return {
            "b": self.browser, "d": self.desktop, "t": self.tool_offers,
            "m": self.mcp_offers,
        }[namespace]

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
            "mcp_offers": [
                {**offer.to_dict(), "ref": f"m:{offer.ref}"} for offer in self.mcp_offers
            ],
            "tool_results": list(self.tool_results),
            "requirements": self.requirements,
        }

    def for_model(self) -> dict[str, Any]:
        """Keep planning refs and semantics, omit runtime-only handles/geometry."""
        state = self.to_dict()
        browser = state["browser"]
        browser["elements"] = [{
            key: value for key, value in element.items()
            if key in {"ref", "tag", "role", "label", "input_type", "disabled", "href"}
        } for element in browser["elements"]]
        for element in browser["elements"]:
            element["href"] = element.get("href", "")[:240]
        for key in ("visual_image_hash", "visual_viewport"):
            browser.pop(key, None)
        desktop = state["desktop"]
        desktop["controls"] = [{
            key: value for key, value in control.items()
            if key in {"ref", "name", "control_type", "enabled", "can_invoke", "can_set_text"}
        } for control in desktop["controls"]]
        for key in ("window_handle", "visual_region", "visual_image_hash"):
            desktop.pop(key, None)
        return state

    def fingerprint(self) -> str:
        state = {**self.to_dict(), "desktop_private_fingerprint": self.desktop.fingerprint()}
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


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
        options: list[ComputerOption] = []
        for item in data["options"]:
            if not isinstance(item, dict) or not isinstance(item.get("ref"), str):
                raise ValueError("computer option must reference one live target")
            namespace, separator, ref = item["ref"].partition(":")
            provider = snapshot.providers.get(namespace)
            if not separator or provider is None:
                raise ValueError("computer option must use a registered namespaced ref")
            checked = provider.validate(
                snapshot.state_for(namespace), ref, item.get("operation"), item.get("value", "")
            )
            options.append(ComputerOption(namespace, *checked))
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
        desktop: OpenDesktopTask | None,
        desktop_surface: ActionSurface | None = None,
        planner: ComputerGoalPlanner,
        local_tool_root: Path | None = None,
        mcp_surface: McpToolSurface | None = None,
        required_url_contains: str = "",
        required_window_title_contains: str = "",
        required_window_text_contains: str = "",
        required_capabilities: Sequence[str] = (),
    ) -> None:
        if not goal.strip() or browser.goal != goal.strip() or (
            desktop is not None and desktop.goal != goal.strip()
        ):
            raise ValueError("the composite goal must match both observers")
        if desktop_surface is None and desktop is None:
            raise ValueError("a desktop surface provider is required")
        if desktop_surface is not None and desktop_surface.namespace != "d":
            raise ValueError("replacement desktop surface must use the d namespace")
        if not any((required_url_contains, required_window_title_contains,
                    required_window_text_contains)):
            raise ValueError("at least one caller-supplied observable state gate is required")
        self.goal = goal.strip()
        self.browser = browser
        self.desktop = desktop
        self.planner = planner
        self.local_tools = ScopedReadOnlyTools(local_tool_root) if local_tool_root else None
        self.providers: dict[str, ActionSurface] = {
            "b": BrowserSurface(browser),
            "d": desktop_surface or WindowsUiaSurface(desktop),
        }
        if self.local_tools is not None:
            self.providers["t"] = ReadOnlyToolSurface(self.local_tools)
        if mcp_surface is not None:
            self.providers["m"] = mcp_surface
        self.required_url_contains = required_url_contains
        self.required_window_title_contains = required_window_title_contains
        self.required_window_text_contains = required_window_text_contains
        self.required_capabilities = tuple(required_capabilities)
        supported_actions = set().union(
            *(provider.supported_capabilities for provider in self.providers.values())
        )
        for capability in self.required_capabilities:
            if capability not in supported_actions:
                raise ValueError("required capability is not registered in this task")
        self._snapshot: ComputerSnapshot | None = None
        self._plan: ComputerPlan | None = None
        self._plan_fingerprint = ""
        self._used: set[int] = set()
        self._completed_capabilities: set[str] = set()
        self._tool_results: list[dict[str, Any]] = []
        self.planner_calls = 0

    def reset(self) -> None:
        for provider in self.providers.values():
            provider.reset()
        self._plan = None
        self._used.clear()
        self._completed_capabilities.clear()
        self._tool_results.clear()

    def close(self) -> None:
        for provider in reversed(tuple(self.providers.values())):
            provider.close()

    def _capture(self) -> ComputerSnapshot:
        return ComputerSnapshot(
            browser=self.providers["b"].capture(), desktop=self.providers["d"].capture(),
            tool_offers=self.providers["t"].capture() if "t" in self.providers else (),
            mcp_offers=self.providers["m"].capture() if "m" in self.providers else (),
            tool_results=tuple(self._tool_results[-6:]),
            requirements={
                "url_contains": self.required_url_contains,
                "window_title_contains": self.required_window_title_contains,
                "window_text_contains": self.required_window_text_contains,
                "capabilities": list(self.required_capabilities),
                "completed_capabilities": sorted(self._completed_capabilities),
            },
            providers=self.providers,
        )

    def observe(self, history: Sequence[StepResult]) -> Observation:
        self._snapshot = ComputerSnapshot(
            browser=self.providers["b"].observe(history),
            desktop=self.providers["d"].observe(history),
            tool_offers=self.providers["t"].observe(history) if "t" in self.providers else (),
            mcp_offers=self.providers["m"].observe(history) if "m" in self.providers else (),
            tool_results=tuple(self._tool_results[-6:]),
            requirements={
                "url_contains": self.required_url_contains,
                "window_title_contains": self.required_window_title_contains,
                "window_text_contains": self.required_window_text_contains,
                "capabilities": list(self.required_capabilities),
                "completed_capabilities": sorted(self._completed_capabilities),
            },
            providers=self.providers,
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
                ) + "\n" + "\n".join(snapshot.desktop.edit_values)
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
        result: list[ActionCandidate] = []
        for index, option in enumerate(self._plan.options):
            if index in self._used:
                continue
            provider = self.providers[option.source]
            candidates = provider.compile(
                self._snapshot.state_for(option.source), option.ref,
                option.operation, option.value, self._plan.subgoal, index,
            )
            if option.source == "t" and candidates and (
                candidates[0].capability in self._completed_capabilities
            ):
                continue
            result.extend(candidates)
        return tuple(result)

    def executor_bindings(self) -> dict[Channel, Any]:
        return {channel: self for channel in (
            Channel.API, Channel.CLI, Channel.SCRIPT, Channel.GUI, Channel.MCP,
        )}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        for provider in self.providers.values():
            provider.register_verifiers(registry)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        matches = [provider for provider in self.providers.values() if provider.owns(candidate)]
        if len(matches) != 1:
            raise ValueError("selected capability has no unique surface provider")
        return matches[0].execute(candidate, observation_id, decision_id)

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
        index = int(candidate.intent.removeprefix("option_"))
        self._used.add(index)
        self._completed_capabilities.add(candidate.capability)
        if candidate.capability == "mcp.call_tool":
            server = candidate.arguments["server"]
            tool = candidate.arguments["tool"]
            self._completed_capabilities.add(f"mcp.{server}.{tool}")
        if self._plan is not None and self._plan.options[index].source in {"t", "m"}:
            output = receipt.output
            summary = output.get(
                "structured_content", output.get(
                    "text", output.get("stdout", output.get("entries", ""))
                )
            )
            self._tool_results.append({
                "capability": candidate.capability, "result": str(summary)[:2000],
                "verified": True,
            })
        if self._satisfied(self._capture()):
            return Evaluation(True, True, "acceptance_passed")
        return Evaluation(False, False, "action_verified")
