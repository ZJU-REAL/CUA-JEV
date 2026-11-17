"""Contracts for screenshot planning and freshly grounded fast actions.

Images are kept in memory. Plan dependencies describe steps, not alternative
routes; executable routes are generated independently by the backend.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

from .desktop_input_context import ObservedInputGeometry
from .models import ActionCandidate, ActionReceipt, Verification
from .vision import WindowImage

if TYPE_CHECKING:
    from .desktop_phase_handoff_v62 import HighLevelContextRequest
    from .desktop_stage_program import StageProgram, StageProgramPlan


@dataclass(frozen=True)
class NativeDesktopEvidence:
    """Observed AT-SPI facts, not inferred mouse effects or an enabled claim.

    None means the source did not report that state. Action names are data;
    only the desktop gesture policy may grant a bounded physical input.
    """

    actions: tuple[str, ...] = ()
    complete: bool = False
    enabled: bool | None = None
    sensitive: bool | None = None
    focusable: bool | None = None
    visible: bool | None = None
    showing: bool | None = None
    parent_enabled: bool | None = None
    parent_sensitive: bool | None = None
    # Aggregated reported evidence: False dominates, then True, then unknown.
    ancestors_enabled: bool | None = None
    ancestors_sensitive: bool | None = None
    leaf: bool = False
    # Only the recognized AT-SPI full positive-state-set adapter may derive
    # False from an absent checked flag. None is unknown, never unchecked.
    state_set_complete: bool = False
    checked_state: bool | None = None


@dataclass(frozen=True)
class AgentControl:
    ref: str
    label: str
    tag: str
    role: str = ""
    input_type: str = ""
    href: str = ""
    value: str = ""
    disabled: bool = False
    checked: bool = False
    options: tuple[str, ...] = ()
    box: tuple[int, int, int, int] | None = None
    app: str = ""
    window: str = ""
    ancestors: tuple[str, ...] = ()
    focused: bool = False
    selected: bool = False
    expanded: bool = False
    editable: bool = False
    active: bool = False
    selectable: bool = False
    expandable: bool = False
    native: NativeDesktopEvidence | None = None
    # Observed numeric editor text; never a substitute for committed value.
    # None means absent or not supported by trustworthy same-control evidence.
    edit_buffer: str | None = None
    # Reported native modal state on this control or its actual ancestry.
    # Read-only completion evidence; never grants an input capability.
    modal_context: bool = False
    # This node's reported modal state and its nearest actual modal container.
    # The reference is observation-local ancestry data, never an input target.
    modal: bool = False
    modal_owner_ref: str = ""

    def identity(self) -> tuple:
        return (self.label, self.tag, self.role, self.input_type, self.href)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentSnapshot:
    url: str
    title: str
    text: str
    controls: tuple[AgentControl, ...]
    image: WindowImage | None = field(default=None, repr=False, compare=False)
    scroll: tuple[int, int] = (0, 0)
    semantic_text: tuple[str, ...] | None = None
    a11y_complete: bool = True
    # Planner-only native ancestry context. It is absent from execution state,
    # equality, scope keys and freshness fingerprints and grants no capability.
    input_geometry: tuple[ObservedInputGeometry, ...] = field(default=(), repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "controls": [item.to_dict() for item in self.controls],
            "scroll": self.scroll,
            "a11y_complete": self.a11y_complete,
        }

    def scope_key(self) -> str:
        # Values and layout can change after successful form actions. Controls
        # must still be rebound unambiguously and execution checks stay strict.
        data = (
            self.url,
            self.title,
            self.semantic_text
            if self.semantic_text is not None
            else tuple(line.strip() for line in self.text.splitlines() if line.strip()),
            sorted((c.identity(), c.disabled, c.options) for c in self.controls),
        )
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class AgentTarget:
    """A semantic query, rebound to currently observed controls before input."""

    name: str = ""
    role: str = ""
    app: str = ""
    window: str = ""
    ancestor: str = ""
    match: str = "exact"


@dataclass(frozen=True)
class AgentCondition:
    kind: str
    target: AgentTarget | None = None
    value: str = ""


@dataclass(frozen=True)
class AgentPropertyEdit:
    """Compiler-owned literal edit template; the policy binds an observed field."""

    operation: str
    value: str = ""
    bind_boolean_states: bool = False
    prefer_verified_selection: bool = False
    # Explicit model strategy, conditional on a later receipt-bound numeric buffer.
    numeric_finalize_key: str = ""


@dataclass(frozen=True)
class AgentStep:
    id: str
    operation: str
    ref: str = ""
    value: str = ""
    depends_on: tuple[str, ...] = ()
    box: tuple[int, int, int, int] | None = None
    label: str = ""
    target: AgentTarget | None = None
    preconditions: tuple[AgentCondition, ...] = ()
    postconditions: tuple[AgentCondition, ...] = ()
    recovery: str = "replan"
    navigation: bool = False
    button: str = "left"
    click_count: int = 1
    # Compiler-owned permission to enumerate observed native mouse gestures.
    # This is deliberately absent from the model's step-field allowlist.
    native_navigation_gestures: bool = False
    property_edit: AgentPropertyEdit | None = None
    # Compiler-only link to one current preparation, never a model dependency.
    property_prepare_dependency: str = ""
    # Compiler provenance for a read-only effect on an explicit current commit.
    derived_dialog_commit_effect: bool = False
    # Compiler-only phase: select an actual guarded confirmation button after
    # this property stage has a receipt-verified field effect. Never an input.
    property_confirmation_of: str = ""

    @property
    def full_legal_navigation_gestures(self) -> bool:
        # New compilers may return a typed navigation view with broader candidate
        # coverage. Old plans retain their exact fields and dataclass encoding.
        return False

    def to_dict(self) -> dict:
        if not valid_click_parameters(self.operation, self.button, self.click_count):
            raise ValueError("unsupported click parameters")
        result = asdict(self)
        if not self.native_navigation_gestures:
            result.pop("native_navigation_gestures")
        if self.full_legal_navigation_gestures:
            result["native_navigation_gestures"] = True
            result["full_legal_navigation_gestures"] = True
        if self.property_edit is None:
            result.pop("property_edit")
        elif not self.property_edit.bind_boolean_states:
            result["property_edit"].pop("bind_boolean_states")
        if self.property_edit is not None and not self.property_edit.prefer_verified_selection:
            result["property_edit"].pop("prefer_verified_selection")
        if self.property_edit is not None and not self.property_edit.numeric_finalize_key:
            result["property_edit"].pop("numeric_finalize_key")
        if not self.property_prepare_dependency:
            result.pop("property_prepare_dependency")
        if not self.derived_dialog_commit_effect:
            result.pop("derived_dialog_commit_effect")
        if not self.property_confirmation_of:
            result.pop("property_confirmation_of")
        result["depends_on"] = list(self.depends_on)
        result["box"] = list(self.box) if self.box is not None else None
        result["preconditions"] = [asdict(condition) for condition in self.preconditions]
        result["postconditions"] = [asdict(condition) for condition in self.postconditions]
        if self.navigation and self.target is not None:
            result["target"] = {
                field: getattr(self.target, field) for field in ("app", "window", "ancestor")
            }
        if self.operation not in {"click", "visual_click"}:
            result.pop("button")
            result.pop("click_count")
        return result


def valid_click_parameters(operation: str, button: str, click_count: int) -> bool:
    """Reject unsupported mouse inputs before compiling or executing a step."""
    if not isinstance(button, str) or button not in {"left", "right"}:
        return False
    if type(click_count) is not int or click_count not in {1, 2}:
        return False
    if button == "right" and click_count != 1:
        return False
    return operation in {"click", "visual_click"} or (button == "left" and click_count == 1)


@dataclass(frozen=True)
class AgentPlan:
    summary: str
    steps: tuple[AgentStep, ...]
    done: bool = False
    reusable: bool = False
    completion_conditions: tuple[AgentCondition, ...] = ()
    # New property protocol: a dialog field effect alone cannot stop the task.
    defer_dialog_field_completion: bool = False
    # Explicit planner claim; bind only to a receipt-verified actual property edit.
    complete_property_goal: bool = False
    # Explicit response to one runner-owned failed preparation. This is never
    # an action, input capability, completion claim or executable receipt.
    prepare_repair_token: str = ""


class AgentBackend(Protocol):
    def observe(self) -> AgentSnapshot: ...

    def compile(
        self, step: AgentStep, planned: AgentSnapshot, current: AgentSnapshot
    ) -> tuple[ActionCandidate, ...]: ...

    def execute(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt: ...

    def verify(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification: ...

    def close(self) -> None: ...


class AgentPlanner(Protocol):
    def plan(
        self, goal: str, snapshot: AgentSnapshot, history: list[dict], *, max_actions: int = 6
    ) -> AgentPlan | StageProgramPlan | HighLevelContextRequest: ...


class StageProgramPlanner(AgentPlanner, Protocol):
    def activate_program_stage(
        self, program: StageProgram, index: int, snapshot: AgentSnapshot, *, remaining_actions: int
    ) -> AgentPlan: ...
