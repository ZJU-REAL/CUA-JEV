"""Bind a fresh preparation to an explicitly reaffirmed, unexecuted property stage.

This helper neither executes input nor verifies receipts. Geometry constrains
the relationship between two model plans; it grants no control capability.
The runner owns zero-input/progress eligibility, receipt checks and budgets.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction

from .agent_types import AgentCondition, AgentPlan, AgentSnapshot, AgentStep
from .desktop_input_context import input_geometry_payload
from .desktop_property import valid_property_stage
from .desktop_property_confirmation import valid_property_confirmation
from .desktop_semantics import matching_controls, normalize_role


def _number(value) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _visual_fill(step: AgentStep) -> bool:
    return bool(
        type(step) is AgentStep and step.operation == "visual_fill"
        and isinstance(step.id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", step.id)
        and isinstance(step.value, str) and 0 < len(step.value) <= 2000
        and step.value.isascii() and "\x00" not in step.value
        and isinstance(step.label, str) and 0 < len(step.label.strip()) <= 240
        and not step.ref and step.target is None and not step.depends_on
        and not step.preconditions and not step.postconditions
        and step.navigation is False and step.native_navigation_gestures is False
        and step.property_edit is None and not step.property_prepare_dependency
        and not step.property_confirmation_of and step.derived_dialog_commit_effect is False
        and step.button == "left" and step.click_count == 1 and step.recovery == "replan"
        and type(step.box) is tuple and len(step.box) == 4
        and all(type(value) is int for value in step.box)
        and 0 <= step.box[0] < step.box[2] <= 1000
        and 0 <= step.box[1] < step.box[3] <= 1000
        and step.box[2] - step.box[0] >= 5 and step.box[3] - step.box[1] >= 5
    )


def _property_root(plan: AgentPlan) -> bool:
    return bool(
        type(plan) is AgentPlan and plan.reusable is True and plan.done is False
        and not getattr(plan, "prepare_repair_token", "")
        and type(plan.steps) is tuple and len(plan.steps) in {2, 3}
        and type(plan.completion_conditions) is tuple and len(plan.completion_conditions) <= 8
        and all(type(condition) is AgentCondition for condition in plan.completion_conditions)
        and type(plan.defer_dialog_field_completion) is bool and type(plan.complete_property_goal) is bool
        and _visual_fill(plan.steps[0]) and plan.steps[0].id == "prepare"
        and type(plan.steps[1]) is AgentStep and plan.steps[1].id == "property"
        and valid_property_stage(plan.steps[1])
        and plan.steps[1].property_prepare_dependency == "prepare"
        and plan.steps[1].depends_on == ("prepare",)
        and not plan.steps[1].property_confirmation_of
        and plan.steps[1].derived_dialog_commit_effect is False
        and (len(plan.steps) == 2 or valid_property_confirmation(plan.steps[2]))
    )


@dataclass(frozen=True)
class _Source:
    owner_key: tuple
    input_key: tuple
    selected_tabs: tuple
    parent_rect: tuple[int, int, int, int]
    desktop_size: tuple[int, int]


def _source(snapshot: AgentSnapshot, scope) -> _Source | None:
    """Require current, source-bound context and one owner inside its app/window."""
    if type(snapshot) is not AgentSnapshot or snapshot.image is None:
        return None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(snapshot.image.jpeg)) as image:
            payload = input_geometry_payload(snapshot, image.size)
        if payload is None:
            return None
        context, = snapshot.input_geometry
        controls = tuple(c for c in snapshot.controls if c.ref == context.input_ref)
        if len(controls) != 1:
            return None
        control, = controls
        if not control.app.strip() or not control.window.strip():
            return None
        if control not in matching_controls(snapshot, scope, allow_scope=True):
            return None
        owners = tuple(c for c in snapshot.controls if normalize_role(c.role) == "frame"
                       and c.active and (c.app, c.window) == (control.app, control.window))
        if len(owners) != 1 or owners[0].disabled:
            return None
        owner, = owners
        tabs = tuple(sorted((c.identity(), c.app, c.window, c.ancestors)
                            for c in snapshot.controls if normalize_role(c.role) == "tab"
                            and c.selected and (c.app, c.window) == (control.app, control.window)))
        if len(tabs) != len(set(tabs)):
            return None
        owner_key = (owner.identity(), owner.app, owner.window, owner.ancestors)
        input_key = (
            control.identity(), control.app, control.window, control.ancestors,
            context.input_rect_px, context.ancestor_role, context.ancestor_class,
            context.ancestor_label, context.ancestor_rect_px, context.desktop_size_px,
        )
        return _Source(owner_key, input_key, tabs, context.ancestor_rect_px, context.desktop_size_px)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None


def _inside(step: AgentStep, source: _Source) -> bool:
    """Match the executor's clamped, floored click pixel to the raw half-open parent."""
    left, top, right, bottom = source.parent_rect
    width, height = source.desktop_size
    x = min(width - 1, int(Fraction((step.box[0] + step.box[2]) * width, 2000)))
    y = min(height - 1, int(Fraction((step.box[1] + step.box[3]) * height, 2000)))
    return left <= x < right and top <= y < bottom


def _required_horizon(plan: AgentPlan) -> int:
    return len(plan.steps) + sum(
        bool(step.property_edit and step.property_edit.numeric_finalize_key) for step in plan.steps
    )


def _seal(frame: PrepareRepair) -> str:
    data = {
        "token": frame.token, "plan": asdict(frame.original_plan), "plan_at": frame.plan_at,
        "deadline": frame.deadline, "horizon": frame.horizon, "owner": frame.owner_key,
        "input": frame._input_key, "tabs": frame._selected_tabs,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PrepareRepair:
    token: str
    original_plan: AgentPlan
    plan_at: float
    deadline: float
    horizon: int
    owner_key: tuple
    _input_key: tuple = field(default=(), repr=False)
    _selected_tabs: tuple = field(default=(), repr=False)
    _binding: str = field(default="", repr=False)

    def _current_source(self, current: AgentSnapshot, now: float) -> _Source | None:
        try:
            if (not isinstance(self.token, str) or not re.fullmatch(r"[0-9a-f]{32}", self.token)
                    or not _property_root(self.original_plan)
                    or type(self.horizon) is not int
                    or not _required_horizon(self.original_plan) <= self.horizon <= 12
                    or not all(_number(value) for value in (self.plan_at, self.deadline, now))
                    or not 0 <= self.plan_at <= now < self.deadline <= self.plan_at + 300
                    or not self._binding or self._binding != _seal(self)):
                return None
            source = _source(current, self.original_plan.steps[1].target)
            if (source is None or source.owner_key != self.owner_key
                    or source.input_key != self._input_key or source.selected_tabs != self._selected_tabs):
                return None
            return source
        except (AttributeError, TypeError, ValueError):
            return None

    def context(self, current: AgentSnapshot, *, now: float) -> dict | None:
        """Offer only a still-valid semantic reaffirmation, never an old visual box."""
        if self._current_source(current, now) is None:
            return None
        plan = self.original_plan
        prop = plan.steps[1]
        edit = {"operation": prop.property_edit.operation}
        if prop.property_edit.operation in {"fill", "set"}:
            edit["value"] = prop.property_edit.value
        return {
            "token": self.token, "operation": "visual_fill", "value": plan.steps[0].value,
            "remaining_time_s": self.deadline - now,
            "retained_stage": {
                "label": prop.label, "scope": asdict(prop.target), "edit": edit,
                **({"numeric_finalize": {"key": prop.property_edit.numeric_finalize_key}}
                   if prop.property_edit.numeric_finalize_key else {}),
                "confirm": {"label": plan.steps[2].label} if len(plan.steps) == 3 else None,
                "whole_goal_conditions": [asdict(condition) for condition in plan.completion_conditions],
                "completion": "goal" if plan.complete_property_goal else "stage",
                "defer_dialog_field_completion": plan.defer_dialog_field_completion,
            },
            "owner_continuity": "observed app/window/ancestry only; no stable native PID identity",
            "geometry_relation": ("Current action's actual click pixel must be inside "
                                  "the raw parent rectangle."),
            "meaning": ("Explicitly reaffirm the retained future stage; "
                        "this does not prove the old prepare ran."),
        }


def suspend_property_prepare(
    plan: AgentPlan, planned: AgentSnapshot, current: AgentSnapshot, *,
    plan_at: float, ttl_s: float, horizon: int, now: float,
) -> PrepareRepair | None:
    """Capture one unexecuted prefix; the runner must separately prove zero progress."""
    try:
        if (not _property_root(plan) or type(horizon) is not int
                or not _required_horizon(plan) <= horizon <= 12
                or not all(_number(value) for value in (plan_at, ttl_s, now))
                or not 0 < ttl_s <= 300 or not 0 <= plan_at <= now < plan_at + ttl_s):
            return None
        before = _source(planned, plan.steps[1].target)
        after = _source(current, plan.steps[1].target)
        if (before is None or after is None or before != after
                or not _inside(plan.steps[0], before)):
            return None
        frame = PrepareRepair(uuid.uuid4().hex, plan, plan_at, plan_at + ttl_s, horizon,
                              before.owner_key, before.input_key, before.selected_tabs)
        return replace(frame, _binding=_seal(frame))
    except (AttributeError, TypeError, ValueError):
        return None


def repair_property_prepare(
    frame: PrepareRepair, immediate: AgentPlan, current: AgentSnapshot, *, now: float,
) -> AgentPlan:
    """Replace before execution; ordinary input guards create the new real receipt."""
    if type(frame) is not PrepareRepair:
        raise ValueError("invalid preparation repair frame")
    source = frame._current_source(current, now)
    if (source is None or type(immediate) is not AgentPlan
            or getattr(immediate, "prepare_repair_token", "") != frame.token
            or immediate.done is not False or immediate.reusable is not False
            or immediate.completion_conditions or immediate.defer_dialog_field_completion is not False
            or immediate.complete_property_goal is not False
            or type(immediate.steps) is not tuple or len(immediate.steps) != 1
            or not _visual_fill(immediate.steps[0])
            or immediate.steps[0].value != frame.original_plan.steps[0].value
            or not _inside(immediate.steps[0], source)):
        raise ValueError("invalid, changed or expired preparation repair")
    return replace(frame.original_plan,
                   steps=(replace(immediate.steps[0], id="prepare"), *frame.original_plan.steps[1:]),
                   prepare_repair_token="")
