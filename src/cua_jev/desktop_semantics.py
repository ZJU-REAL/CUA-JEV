"""Observable desktop target and condition matching shared by runner/backend."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .agent_types import (
    AgentCondition,
    AgentControl,
    AgentSnapshot,
    AgentTarget,
    NativeDesktopEvidence,
    valid_click_parameters,
)

_EXECUTABLE_ROLES = frozenset(
    {
        "button",
        "menuitem",
        "textbox",
        "checkbox",
        "radio",
        "combobox",
        "tab",
        "listitem",
        "treeitem",
        "cell",
        "link",
        "spinbutton",
    }
)


def valid_desktop_box(box: object) -> bool:
    """A clickable normalized rectangle, using the desktop executor's bounds."""
    return (
        isinstance(box, (tuple, list))
        and len(box) == 4
        and all(type(n) is int for n in box)
        and 0 <= box[0] < box[2] <= 1000
        and 0 <= box[1] < box[3] <= 1000
        and box[2] - box[0] >= 5
        and box[3] - box[1] >= 5
    )


def _native_pointer_target(control: AgentControl) -> bool:
    native = control.native
    return (
        isinstance(native, NativeDesktopEvidence)
        and native.complete is True and native.leaf is True
        and control.role == "canvas" and control.tag == "container"
        and bool(control.label.strip()) and bool(control.app.strip())
        and native.enabled is not False and native.sensitive is not False
        and native.focusable is True and native.visible is True and native.showing is True
        and native.parent_enabled is True and native.parent_sensitive is True
        and native.ancestors_enabled is True and native.ancestors_sensitive is True
        and bool(set(native.actions) & {"open", "menu"})
    )


def native_toggle_control(control: AgentControl) -> bool:
    """Identify the separate pointer-only checkable route, including unknowns."""
    return (
        isinstance(control.native, NativeDesktopEvidence)
        and control.role == "checkbox" and control.input_type == "checkbox"
        and control.tag == "container"
    )


def observed_checked_state(control: AgentControl) -> bool | None:
    if control.input_type not in {"checkbox", "radio"}:
        return None
    if native_toggle_control(control):
        return control.native.checked_state if control.native.state_set_complete else None
    return control.checked


def checked_matches(control: AgentControl, expected: bool, snapshot: AgentSnapshot) -> bool:
    if native_toggle_control(control) and not snapshot.a11y_complete:
        return False
    return observed_checked_state(control) is expected


def _native_toggle_target(control: AgentControl) -> bool:
    native = control.native
    return (
        native_toggle_control(control) and native.complete and native.leaf
        and "toggle" in native.actions and observed_checked_state(control) is not None
        and bool(control.label.strip()) and bool(control.app.strip())
        and native.enabled is True and native.sensitive is True
        and native.visible is True and native.showing is True
        and native.parent_enabled is True and native.parent_sensitive is True
        and native.ancestors_enabled is True and native.ancestors_sensitive is True
    )


def executable_control(
    control: AgentControl, operation: str, *, button: str = "left", click_count: int = 1
) -> bool:
    """Check desktop control legality without registering or executing input."""
    if (
        not isinstance(control, AgentControl)
        or not valid_click_parameters(operation, button, click_count)
        or control.disabled
        or control.input_type.lower() in {"password", "file", "hidden"}
        or not valid_desktop_box(control.box)
    ):
        return False
    if native_toggle_control(control):
        # A directly reported toggle permits a pointer state edit without
        # claiming keyboard focus or creating a generic navigation click.
        return operation in {"check", "uncheck"} and _native_toggle_target(control)
    if _native_pointer_target(control):
        # Deliver only the explicitly requested mouse gesture. Advertising
        # "open" does not select a click count or prove that the item opened.
        return operation == "click" and (
            (button == "left" and click_count == 1)
            or (button == "right" and "menu" in control.native.actions)
            or (button == "left" and click_count == 2 and "open" in control.native.actions)
        )
    return (
        operation in {"click", "fill", "check", "uncheck"}
        and control.tag in {"button", "input", "a"}
        and control.role in _EXECUTABLE_ROLES
        and (operation != "fill" or control.editable)
        and (operation not in {"check", "uncheck"} or control.input_type in {"checkbox", "radio"})
        and (operation != "uncheck" or control.input_type != "radio")
    )


def executable_controls(
    snapshot: AgentSnapshot, operation: str, target: AgentTarget, *, allow_scope: bool = False,
    button: str = "left", click_count: int = 1,
) -> tuple[AgentControl, ...]:
    """Return every legal matching desktop control; no limits or side effects.

    Operation arguments, preconditions and candidate budgets are checked by
    callers. Scope-only selectors are reserved for navigation clicks.
    """
    if allow_scope and operation != "click":
        return ()
    matches = matching_controls(snapshot, target, allow_scope=allow_scope)
    if operation in {"check", "uncheck"} and any(native_toggle_control(c) for c in matches):
        # Resolve real state-bearing targets BEFORE filtering by the desired
        # state or executability. A second disabled/unknown checkbox is not
        # evidence that the remaining one is the user's intended target.
        if not snapshot.a11y_complete or sum(c.input_type in {"checkbox", "radio"} for c in matches) != 1:
            return ()
    return tuple(
        control
        for control in matches
        if executable_control(control, operation, button=button, click_count=click_count)
    )


def desktop_click_modes(control: AgentControl) -> tuple[tuple[str, int], ...]:
    """Legal physical gestures, never synonyms for native action names."""
    return tuple(
        (button, count) for button, count in (("left", 1), ("right", 1), ("left", 2))
        if executable_control(control, "click", button=button, click_count=count)
    )


def native_navigation_click_modes(control: AgentControl) -> tuple[tuple[str, int], ...]:
    """Compiler navigation: ordinary left1, additional observed native gestures.

    Mechanical right/double-click legality on an ordinary widget is not evidence
    for a menu/open affordance and must not multiply its navigation candidates.
    """
    if _native_pointer_target(control):
        return desktop_click_modes(control)
    return (("left", 1),) if executable_control(control, "click") else ()


def _normalized(value: str) -> str:
    # AT-SPI desktop labels can insert zero-width spaces as line-break hints.
    # Match their displayed spelling, without changing raw control identities,
    # typed values, joiners, or other meaningful Unicode characters.
    return " ".join(value.replace("\u200b", "").split()).casefold()


def normalize_role(value: str) -> str:
    value = _normalized(value).replace("_", "-").replace(" ", "-")
    return {
        "push-button": "button",
        "pushbutton": "button",
        "toggle-button": "checkbox",
        "toggle": "checkbox",
        "switch": "checkbox",
        "entry": "textbox",
        "text-entry": "textbox",
        "text-field": "textbox",
        "text-box": "textbox",
        "check-box": "checkbox",
        "check-menu-item": "checkbox",
        "radio-button": "radio",
        "radio-menu-item": "radio",
        "menu": "menuitem",
        "menu-item": "menuitem",
        "combo-box": "combobox",
        "page-tab": "tab",
        "list-item": "listitem",
        "tree-item": "treeitem",
        "table-cell": "cell",
        "hyperlink": "link",
        "spin-button": "spinbutton",
    }.get(value, value)


def _empty_display_constraint(target: AgentTarget) -> bool:
    # A supplied qualifier must not silently become an unrestricted selector.
    return any(
        value.strip() and not _normalized(value)
        for value in (target.name, target.role, target.app, target.window, target.ancestor)
    )


def matching_controls(
    snapshot: AgentSnapshot, target: AgentTarget, *, allow_scope: bool = False
) -> tuple[AgentControl, ...]:
    if (not isinstance(target, AgentTarget) or target.match not in {"exact", "contains"}
            or _empty_display_constraint(target)):
        return ()
    if allow_scope:
        if (
            _normalized(target.name)
            or _normalized(target.role)
            or target.match != "exact"
            or not (_normalized(target.app) or _normalized(target.window))
        ):
            return ()
    elif not (_normalized(target.name) or _normalized(target.role)):
        return ()
    name, role = _normalized(target.name), normalize_role(target.role)
    result = []
    for control in snapshot.controls:
        label = _normalized(control.label)
        if name and not (label == name if target.match == "exact" else name in label):
            continue
        observed_role = (
            control.input_type
            if control.input_type in {"checkbox", "radio"}
            else normalize_role(control.role)
        )
        if role and observed_role != role:
            continue
        if target.app and _normalized(target.app) not in _normalized(control.app):
            continue
        if target.window and _normalized(target.window) not in _normalized(control.window):
            continue
        if target.ancestor and not any(
            _normalized(target.ancestor) in _normalized(parent) for parent in control.ancestors
        ):
            continue
        result.append(control)
    return tuple(result)


def navigation_scope_controls(
    snapshot: AgentSnapshot, target: AgentTarget, *, handoff_app: str = ""
) -> tuple[AgentControl, ...]:
    """Resolve the declared scope or one receipt-authorized application's active window.

    A handoff uses the exact observed application identity, never substring
    matching or a union with the desktop launcher scope. Ambiguous active
    windows fail closed; a later dialog may replace the active window. Window
    titles are display data: an unnamed window remains a real scope. The
    desktop parser inherits active state from window roots, so a unique active
    root plus exact app/window identity excludes its inactive siblings.
    """
    if not handoff_app:
        return matching_controls(snapshot, target, allow_scope=True)
    if (not isinstance(target, AgentTarget) or not _normalized(target.app)
            or target.name or target.role or target.window or target.ancestor
            or target.match != "exact" or _empty_display_constraint(target)):
        return ()
    app = _normalized(handoff_app)
    controls = tuple(control for control in snapshot.controls if _normalized(control.app) == app)
    windows = tuple(
        control for control in controls
        if control.active and control.role in {"frame", "window", "dialog"}
    )
    if not snapshot.a11y_complete or len(windows) != 1:
        return ()
    window = windows[0].window
    if not window.strip() and windows[0].label.strip():
        return ()  # Inconsistent with the parser's named-window inheritance.
    return tuple(control for control in controls if control.active and control.window == window)


def navigation_handoff_app(
    before: AgentSnapshot,
    after: AgentSnapshot,
    target: AgentTarget,
    launcher: AgentControl | None,
) -> str:
    """Suggest one new active app after an already-sent desktop launcher input.

    The caller must authorize this only from its successful receipt, at most
    once per navigation step. Ordinary observations must not call it to widen
    the candidate scope. Existing applications and background arrivals cannot
    establish a handoff.
    """
    if (
        not before.a11y_complete or not after.a11y_complete
        or not isinstance(target, AgentTarget) or not target.app.strip()
        or target.window or target.ancestor
        or launcher is None or launcher.window or not launcher.label.strip()
        or launcher.role != "button" or not executable_control(launcher, "click")
        or launcher not in matching_controls(before, target, allow_scope=True)
    ):
        return ""
    previous = {_normalized(control.app) for control in before.controls if control.app.strip()}
    active = {
        _normalized(control.app): control.app for control in after.controls
        if control.active and control.app.strip() and _normalized(control.app) not in previous
    }
    if len(active) != 1:
        return ""
    app = next(iter(active.values()))
    controls = navigation_scope_controls(after, target, handoff_app=app)
    return app if any(executable_control(control, "click") for control in controls) else ""


def _value_bearing(control: AgentControl) -> bool:
    return control.editable or (
        control.tag in {"input", "textarea", "contenteditable", "select"}
        and control.input_type not in {"checkbox", "radio", "button", "submit", "reset", "file", "hidden"}
    )


_DECIMAL_VALUE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]{1,4}))?")


def _bounded_decimal(value: str) -> Decimal | None:
    # Bound both coefficient size and exponent before constructing a Decimal.
    # No stripping, locale syntax, units, Unicode digits or nonfinite tokens.
    if not isinstance(value, str) or len(value) > 128:
        return None
    match = _DECIMAL_VALUE.fullmatch(value)
    if match is None or (match[1] is not None and abs(int(match[1])) > 1000):
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def value_matches(control: AgentControl, expected: str) -> bool:
    """Compare committed values, with exact numeric equivalence for spinbuttons.

    Unsupported numeric syntax retains the original literal-string semantics;
    it never enables a new equivalence. Display text is not value evidence.
    """
    if control.value == expected:
        return True
    if normalize_role(control.role) != "spinbutton":
        return False
    actual, wanted = _bounded_decimal(control.value), _bounded_decimal(expected)
    return actual is not None and wanted is not None and actual == wanted


def pending_numeric_buffer_match(control: AgentControl, expected: str) -> str | None:
    """Describe an uncommitted numeric edit, without changing any predicate.

    Unlike value_matches' legacy literal fallback, diagnostic values must all
    be finite bounded ASCII numbers. This prevents arbitrary display text from
    being reported as evidence of a pending numeric submission.
    """
    if (
        normalize_role(control.role) != "spinbutton" or control.input_type != "number"
        or not control.editable or not control.focused or control.disabled
        or not valid_desktop_box(control.box) or control.edit_buffer is None
    ):
        return None
    committed = _bounded_decimal(control.value)
    buffer = _bounded_decimal(control.edit_buffer)
    wanted = _bounded_decimal(expected)
    if committed is None or buffer is None or wanted is None or buffer != wanted or committed == wanted:
        return None
    return "literal" if control.edit_buffer == expected else "numeric"


def numeric_editor_status(control: AgentControl) -> str | None:
    """Planner-only comparison of this control's bounded numeric buffer/value."""
    if (normalize_role(control.role) != "spinbutton" or control.input_type != "number"
            or not control.editable or not control.focused or control.disabled
            or not valid_desktop_box(control.box) or control.edit_buffer is None):
        return None
    committed, buffer = _bounded_decimal(control.value), _bounded_decimal(control.edit_buffer)
    if committed is None or buffer is None:
        return "unknown"
    return "committed" if committed == buffer else "pending"


def _focusable(control: AgentControl) -> bool:
    return (
        isinstance(control.native, NativeDesktopEvidence) and control.native.focusable is True
    ) or control.editable or control.tag in {
        "button",
        "input",
        "textarea",
        "contenteditable",
        "select",
        "a",
    }


def observable_condition_kinds(control: AgentControl) -> tuple[str, ...]:
    """State predicates supported by this observation, independent of input legality."""
    if not isinstance(control, AgentControl):
        return ()
    kinds = ["exists", "absent"]
    if control.selectable or control.selected:
        kinds.append("selected")
    if control.expandable or control.expanded:
        kinds.append("expanded")
    if control.input_type in {"checkbox", "radio"}:
        kinds.extend(("checked", "unchecked"))
    if _value_bearing(control):
        kinds.append("value_equals")
    if _focusable(control):
        kinds.append("focused")
    return tuple(kinds)


def navigation_control_evidence(control: AgentControl) -> dict | None:
    """Stable affordances and supported state that justify further navigation.

    Static text remains available to explicit conditions and the VLM, but does
    not independently authorize another fast click. In particular, anonymous
    read-only containers observed only because they have focus carry no such
    evidence. Coordinates and focus are still checked separately before input.
    """
    if not isinstance(control, AgentControl) or control.input_type in {"password", "file", "hidden"}:
        return None
    operations = tuple(
        operation for operation in ("click", "fill", "check", "uncheck")
        if executable_control(control, operation)
    )
    capabilities = observable_condition_kinds(control)
    state = {
        kind: (observed_checked_state(control) if kind == "checked"
               else getattr(control, "value" if kind == "value_equals" else kind))
        for kind in ("checked", "selected", "expanded", "value_equals")
        if kind in capabilities
    }
    if not operations and not state:
        return None
    evidence = {
        "label": _normalized(control.label),
        "role": normalize_role(control.role),
        "input_type": control.input_type,
        "app": _normalized(control.app),
        "window": _normalized(control.window),
        "ancestors": tuple(_normalized(parent) for parent in control.ancestors),
        "operations": operations,
        "state": state,
    }
    if control.native is not None:
        evidence["click_modes"] = desktop_click_modes(control)
    if native_toggle_control(control):
        evidence["native_toggle"] = {
            "actions": control.native.actions,
            "complete": control.native.complete,
            "state_set_complete": control.native.state_set_complete,
        }
    return evidence


def condition_evidence(condition: AgentCondition, snapshot: AgentSnapshot) -> dict[str, str]:
    """Explain only public condition evidence; never include target or field values."""
    def evidence(outcome, reason):
        return {"outcome": outcome, "reason": reason}

    def observed(value):
        return evidence("true", "condition_met") if value else evidence("false", "state_mismatch")

    if not isinstance(condition, AgentCondition):
        return evidence("unknown", "invalid_condition")
    kind = condition.kind
    if kind == "ready_to_select":
        # Compiler-only selected-option readiness. Resolve every current
        # identity before state/capability; a partial tree proves no uniqueness.
        if not snapshot.a11y_complete:
            return evidence("unknown", "incomplete_observation")
        if condition.value:
            return evidence("unknown", "invalid_condition")
        matches = matching_controls(snapshot, condition.target)
        if not matches:
            return evidence("unknown", "target_missing")
        if len(matches) != 1:
            return evidence("unknown", "ambiguous_target")
        control = matches[0]
        if not control.selectable or control.editable:
            return evidence("unknown", "unsupported_state")
        return observed(executable_control(control, "click"))
    if kind in {"ready_to_fill", "ready_to_check", "ready_to_uncheck"}:
        # Compiler-only stage predicates, deliberately absent from the model's
        # condition allowlist. Never use a partial tree to prove uniqueness.
        if not snapshot.a11y_complete:
            return evidence("unknown", "incomplete_observation")
        operation = kind.removeprefix("ready_to_")
        if operation != "fill" and condition.value:
            return evidence("unknown", "invalid_condition")
        effect_kind = {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}[operation]
        effect = AgentCondition(effect_kind, condition.target, condition.value)
        result = condition_evidence(effect, snapshot)
        if result["outcome"] != "false":
            # Ordinary effect matching filters by observable capability, then
            # requires uniqueness BEFORE inspecting the desired value/state.
            # Already-satisfied readonly targets need no physical input.
            return result
        observable = tuple(
            control for control in matching_controls(snapshot, condition.target)
            if effect_kind in observable_condition_kinds(control)
        )
        executable = executable_controls(snapshot, operation, condition.target)
        if len(executable) > 1:
            return evidence("unknown", "ambiguous_target")
        return observed(len(observable) == len(executable) == 1 and observable[0] == executable[0])
    if kind in {"text_contains", "window_contains"}:
        if condition.target is not None or not _normalized(condition.value):
            return evidence("unknown", "invalid_condition")
        needle = _normalized(condition.value)
        if kind == "text_contains":
            if needle in _normalized(snapshot.text):
                return observed(True)
            if not snapshot.a11y_complete or not snapshot.text.strip():
                return evidence("unknown", "incomplete_observation")
            return observed(False)
        titles = [snapshot.title, *(control.window for control in snapshot.controls if control.active)]
        if any(needle in _normalized(title) for title in titles):
            return observed(True)
        if all(not title or title == "OSWorld desktop" for title in titles):
            return evidence("unknown", "target_missing")
        return observed(False)
    if kind not in {
        "exists", "absent", "checked", "unchecked", "value_equals", "focused", "selected", "expanded"
    }:
        return evidence("unknown", "invalid_condition")
    if (
        not isinstance(condition.target, AgentTarget)
        or condition.target.match not in {"exact", "contains"}
        or _empty_display_constraint(condition.target)
        or not (_normalized(condition.target.name) or _normalized(condition.target.role))
    ):
        return evidence("unknown", "invalid_condition")
    matches = matching_controls(snapshot, condition.target)
    if kind == "exists":
        return observed(True) if matches else evidence("unknown", "target_missing")
    if kind == "absent":
        if matches:
            return observed(False)
        if not snapshot.a11y_complete or not (snapshot.controls or snapshot.text.strip()):
            return evidence("unknown", "incomplete_observation")
        return observed(True)
    if not matches:
        return evidence("unknown", "target_missing")
    # Static labels often repeat their associated control's accessible name.
    # Disambiguate by the requested observable capability before uniqueness;
    # two actual inputs with the same name must still fail closed.
    matches = tuple(control for control in matches if kind in observable_condition_kinds(control))
    if not matches:
        return evidence("unknown", "unsupported_state")
    if len(matches) != 1:
        return evidence("unknown", "ambiguous_target")
    control = matches[0]
    if kind == "value_equals":
        return observed(value_matches(control, condition.value))
    if kind in {"checked", "unchecked"}:
        if native_toggle_control(control) and not snapshot.a11y_complete:
            return evidence("unknown", "incomplete_observation")
        checked = observed_checked_state(control)
        if checked is None:
            return evidence("unknown", "unsupported_state")
        return observed(checked is (kind == "checked"))
    if kind in {"focused", "selected", "expanded"}:
        return observed(bool(getattr(control, kind)))
    return evidence("unknown", "invalid_condition")


def condition_holds(condition: AgentCondition, snapshot: AgentSnapshot) -> bool:
    return condition_evidence(condition, snapshot)["outcome"] == "true"


def conditions_hold(conditions: tuple[AgentCondition, ...], snapshot: AgentSnapshot) -> bool:
    return all(condition_holds(condition, snapshot) for condition in conditions)
