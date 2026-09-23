"""Typed action-surface contract for the experimental cross-app loop.

A new accessibility backend can implement ActionSurface under the ``d`` namespace
without changing the planner's action grammar or the Jev/runtime contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

from .executors.cli import RegisteredCliExecutor
from .executors.filesystem import FileSystemExecutor
from .local_tools import ScopedReadOnlyTools, ToolOffer, verify_readonly_tool_result
from .models import ActionCandidate, ActionReceipt, Channel
from .open_browser import BrowserPlan, BrowserSnapshot, OpenBrowserTask, PlannedOption
from .open_desktop import DesktopOption, DesktopPlan, DesktopSnapshot, OpenDesktopTask
from .runtime import StepResult
from .verify import VerifierRegistry


class ActionSurface(Protocol):
    """Observe live state, validate model refs, compile routes, and execute one."""

    namespace: str
    supported_capabilities: frozenset[str]

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def observe(self, history: Sequence[StepResult]) -> Any: ...

    def capture(self) -> Any: ...

    def validate(
        self, state: Any, ref: str, operation: Any, value: Any
    ) -> tuple[str, str, str]: ...

    def compile(
        self, state: Any, ref: str, operation: str, value: str,
        subgoal: str, index: int,
    ) -> tuple[ActionCandidate, ...]: ...

    def owns(self, candidate: ActionCandidate) -> bool: ...

    def execute(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt: ...

    def register_verifiers(self, registry: VerifierRegistry) -> None: ...


def _renumber(
    candidates: Sequence[ActionCandidate], index: int
) -> tuple[ActionCandidate, ...]:
    return tuple(replace(
        candidate,
        id=f"option_{index}_{candidate.id.removeprefix('option_0_')}",
        intent=f"option_{index}",
    ) for candidate in candidates)


class BrowserSurface:
    namespace = "b"
    supported_capabilities = frozenset({
        "browser.click", "browser.fill", "browser.select", "browser.visual_click",
    })

    def __init__(self, task: OpenBrowserTask) -> None:
        self.task = task

    def reset(self) -> None:
        self.task.reset()

    def close(self) -> None:
        self.task.close()

    def observe(self, history: Sequence[StepResult]) -> BrowserSnapshot:
        return self.task.observe_state(history)

    def capture(self) -> BrowserSnapshot:
        return self.task.capture_state()

    def validate(
        self, state: BrowserSnapshot, ref: str, operation: Any, value: Any
    ) -> tuple[str, str, str]:
        live_refs = {
            item.ref for item in (*state.elements, *state.visual_targets, *state.tool_offers)
        }
        if ref not in live_refs:
            raise ValueError(
                f"browser ref {ref!r} is not live; first live refs are "
                f"{sorted(live_refs)[:12]}"
            )
        checked = BrowserPlan.from_dict({
            "subgoal": "Ground one action", "options": [{
                "ref": ref, "operation": operation, "value": value,
            }],
            "success": {"kind": "url_contains", "value": "__not_a_terminal_check__"},
        }, state).options[0]
        return checked.ref, checked.operation, checked.value

    def compile(
        self, state: BrowserSnapshot, ref: str, operation: str, value: str,
        subgoal: str, index: int,
    ) -> tuple[ActionCandidate, ...]:
        return _renumber(
            self.task.compile_option(state, PlannedOption(ref, operation, value), subgoal), index
        )

    def owns(self, candidate: ActionCandidate) -> bool:
        return candidate.capability in self.supported_capabilities

    def execute(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        return self.task(candidate, observation_id, decision_id)

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        self.task.register_verifiers(registry)


class WindowsUiaSurface:
    namespace = "d"
    supported_capabilities = frozenset({
        "desktop.click", "desktop.fill", "desktop.visual_click",
    })

    def __init__(self, task: OpenDesktopTask) -> None:
        self.task = task

    def reset(self) -> None:
        self.task.reset()

    def close(self) -> None:
        self.task.close()

    def observe(self, history: Sequence[StepResult]) -> DesktopSnapshot:
        return self.task.observe_state(history)

    def capture(self) -> DesktopSnapshot:
        return self.task.capture_state()

    def validate(
        self, state: DesktopSnapshot, ref: str, operation: Any, value: Any
    ) -> tuple[str, str, str]:
        control = next((item for item in state.controls if item.ref == ref), None)
        visual = next((item for item in state.visual_targets if item.ref == ref), None)
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
        if operation == "invoke" and control is not None and control.can_invoke:
            operation = "click"
        if operation == "click" and value is None:
            value = ""
        checked = DesktopPlan.from_dict({
            "subgoal": "Ground one action", "options": [{
                "ref": ref, "operation": operation, "value": value,
            }],
            "success": {"kind": "window_title_contains", "value": "__not_a_terminal_check__"},
        }, state).options[0]
        return checked.ref, checked.operation, checked.value

    def compile(
        self, state: DesktopSnapshot, ref: str, operation: str, value: str,
        subgoal: str, index: int,
    ) -> tuple[ActionCandidate, ...]:
        return _renumber(
            self.task.compile_option(state, DesktopOption(ref, operation, value), subgoal), index
        )

    def owns(self, candidate: ActionCandidate) -> bool:
        return candidate.capability in self.supported_capabilities

    def execute(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        if candidate.channel == Channel.GUI:
            snapshot = self.task.capture_state()
            self.task.screen.focus_handle(snapshot.window_handle, maximize=False)
        return self.task(candidate, observation_id, decision_id)

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        self.task.register_verifiers(registry)


class ReadOnlyToolSurface:
    namespace = "t"

    def __init__(self, tools: ScopedReadOnlyTools) -> None:
        self.tools = tools
        self._file_executor = FileSystemExecutor()
        self._cli_executor = RegisteredCliExecutor()

    @property
    def supported_capabilities(self) -> frozenset[str]:
        return frozenset(item.capability for item in self.tools.offers())

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def observe(self, history: Sequence[StepResult]) -> tuple[ToolOffer, ...]:
        return self.tools.offers()

    def capture(self) -> tuple[ToolOffer, ...]:
        return self.tools.offers()

    def validate(
        self, state: tuple[ToolOffer, ...], ref: str, operation: Any, value: Any
    ) -> tuple[str, str, str]:
        if ref not in {item.ref for item in state} or operation != "invoke" or value:
            raise ValueError("local tool option must invoke one offered t: ref")
        return ref, "invoke", ""

    def compile(
        self, state: tuple[ToolOffer, ...], ref: str, operation: str, value: str,
        subgoal: str, index: int,
    ) -> tuple[ActionCandidate, ...]:
        offer = next((item for item in state if item.ref == ref), None)
        return (offer.candidate(index),) if offer is not None else ()

    def owns(self, candidate: ActionCandidate) -> bool:
        return candidate.capability in self.supported_capabilities

    def execute(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        if candidate.channel == Channel.API:
            return self._file_executor(candidate, observation_id, decision_id)
        if candidate.channel == Channel.CLI:
            return self._cli_executor(candidate, observation_id, decision_id)
        raise ValueError("local tool has no registered execution channel")

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("tool.result", verify_readonly_tool_result)
