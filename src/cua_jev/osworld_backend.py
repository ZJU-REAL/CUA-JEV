"""Typed desktop actions inside an already-owned OSWorld environment.

Only screenshots and accessibility observations reach the agent. Task configs,
evaluation functions and evaluator feedback are deliberately absent here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from .agent_planner import DESKTOP_KEYS, DESKTOP_MODIFIERS
from .agent_types import (
    AgentCondition,
    AgentControl,
    AgentSnapshot,
    AgentStep,
    AgentTarget,
    NativeDesktopEvidence,
    valid_click_parameters,
)
from .desktop_input_context import ObservedInputGeometry
from .desktop_property import property_effect_holds, property_field_steps, valid_property_stage
from .desktop_semantics import (
    checked_matches,
    condition_evidence,
    conditions_hold,
    desktop_click_modes,
    executable_control,
    executable_controls,
    matching_controls,
    native_navigation_click_modes,
    native_toggle_control,
    navigation_control_evidence,
    navigation_handoff_app,
    navigation_scope_controls,
    normalize_role,
    observed_checked_state,
    pending_numeric_buffer_match,
    valid_desktop_box,
    value_matches,
)
from .models import ActionCandidate, ActionReceipt, Channel, Risk, Verification
from .vision import SAMPLE_EDGE, WindowImage, changed_fraction

if TYPE_CHECKING:
    from .desktop_navigation_memory import NavigationState, NavigationTransition

_OPERATIONS = frozenset(
    {
        "visual_click",
        "visual_fill",
        "type_text",
        "hotkey",
        "key_press",
        "press",
        "scroll",
        "back",
        "wait",
        "click",
        "fill",
        "check",
        "uncheck",
    }
)
_BROWSER_KEYS = {
    "Enter": "enter",
    "Tab": "tab",
    "Escape": "esc",
    "Backspace": "backspace",
    "Delete": "delete",
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "Home": "home",
    "End": "end",
    "PageUp": "pageup",
    "PageDown": "pagedown",
    "Space": "space",
}

# Observation evidence is separate from the planner's shorter target-query and
# action-label contracts. Retain names consistently in controls and context.
_OBSERVED_NAME_LIMIT = 2000
_OBSERVED_BODY_LIMIT = 2000
_OBSERVED_TEXT_LIMIT = 96000
OBSERVATION_BUDGET_VERSION = "names2k_text96k_v1"


def _observed_name(value: str) -> str:
    return " ".join(value.split())[:_OBSERVED_NAME_LIMIT]


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def accessibility_context(raw: Any, *, screen_size: tuple[int, int] | None = None) -> tuple[str, str]:
    """Flatten bounded visible accessibility text; malformed XML is nonfatal."""
    title, text, _semantic = _accessibility_observation(raw, screen_size=screen_size)
    return title, text


def _top_bar_clock(role: str, attributes: dict, app: str, text: str) -> bool:
    """Recognize only GNOME Shell's small clock label at the top of the screen."""
    if app != "gnome-shell" or role not in {"label", "text"} or attributes.get("toolkit") != "clutter":
        return False
    pair = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    location = re.fullmatch(pair, attributes.get("screencoord", ""))
    size = re.fullmatch(pair, attributes.get("size", ""))
    if not location or not size:
        return False
    x, y = map(int, location.groups())
    width, height = map(int, size.groups())
    if not (x >= 0 and 0 <= y <= 48 and 0 < width <= 500 and 0 < height <= 64):
        return False
    normalized = re.sub(r"\\u(?:200[0-9a-fA-F]|202[89a-eA-E])", "", text)
    normalized = normalized.translate(dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e")))
    return bool(re.search(r"\d{1,2}\s*[:∶]\s*\d{2}", normalized))


def _component_bounds(attributes: dict, screen_size: tuple[int, int] | None) -> tuple | None:
    """Return observed pixel and normalized boxes only when fully on-screen."""
    if (
        screen_size is None
        or len(screen_size) != 2
        or any(type(value) is not int or value <= 0 for value in screen_size)
    ):
        return None
    pair = r"\(\s*(-?\d{1,9})\s*,\s*(-?\d{1,9})\s*\)"
    location = re.fullmatch(pair, attributes.get("screencoord", ""))
    size = re.fullmatch(pair, attributes.get("size", ""))
    if not location or not size:
        return None
    x, y = map(int, location.groups())
    width, height = map(int, size.groups())
    screen_width, screen_height = screen_size
    if not (0 <= x < x + width <= screen_width and 0 <= y < y + height <= screen_height):
        return None
    pixels = (x, y, x + width, y + height)
    normalized = tuple(
        round(value * 1000 / extent)
        for value, extent in zip(
            pixels,
            (screen_width, screen_height, screen_width, screen_height),
            strict=True,
        )
    )
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return None
    return pixels, normalized


_CONTROL_ROLES = frozenset(
    {
        "button",
        "push-button",
        "toggle-button",
        "toggle",
        "switch",
        "menu",
        "menu-item",
        "check-menu-item",
        "radio-menu-item",
        "check-box",
        "checkbox",
        "radio-button",
        "radio",
        "entry",
        "text",
        "text-entry",
        "combo-box",
        "page-tab",
        "list-item",
        "tree-item",
        "table-cell",
        "link",
        "hyperlink",
        "spin-button",
    }
)


def _shown(attributes: dict, atspi: bool) -> bool:
    return not any(attributes.get(key, "").lower() == "false" for key in ("visible", "showing")) and (
        not atspi or attributes.get("showing", "").lower() == "true"
    )


def _protocol_attribute(key: str, namespace: str) -> bool:
    # Native affordance authorization requires the recorded producer's exact
    # namespaces; a matching suffix on an unrelated host is not evidence.
    return key.startswith(f"{{https://accessibility.ubuntu.example.org/ns/{namespace}}}")


def _native_state(node: Any, name: str) -> bool | None:
    # Explicit negatives never disappear through namespace flattening. Positive
    # evidence must come from the actual state protocol, not a bare lookalike.
    values = [value.lower() for key, value in node.attrib.items() if _local_name(key) == name]
    if "false" in values:
        return False
    reported = [value.lower() for key, value in node.attrib.items()
                if _protocol_attribute(key, "state") and _local_name(key) == name]
    return True if reported == ["true"] else None


def _combine_native_state(previous: bool | None, current: bool | None) -> bool | None:
    if previous is False or current is False:
        return False
    return True if previous is True or current is True else None


def _input_parent_geometry(node, parent, control, screen_size, ancestor_states):
    """Retain actual ancestry for a tiny input proxy without changing its box."""
    if (parent is None or control.box is not None or control.disabled
            or not control.app or not control.window or not control.active
            or any(value is False for value in ancestor_states)
            or _local_name(parent.tag) not in {"section", "panel", "container", "static"}
            or screen_size is None or len(screen_size) != 2
            or any(type(value) is not int or value <= 0 for value in screen_size)):
        return None
    if any(_native_state(node, key) is not True
           for key in ("editable", "focused", "enabled", "sensitive", "showing", "visible")):
        return None
    if any(_native_state(parent, key) is not True
           for key in ("enabled", "sensitive", "showing", "visible")):
        return None

    def pixel_rect(element):
        component = {}
        for name in ("screencoord", "size"):
            values = [value for key, value in element.attrib.items() if _local_name(key) == name]
            native = [value for key, value in element.attrib.items()
                      if _local_name(key) == name and _protocol_attribute(key, "component")]
            if len(values) != 1 or values != native:
                return None
            pair = re.fullmatch(r"\(\s*(-?\d{1,9})\s*,\s*(-?\d{1,9})\s*\)", native[0])
            if pair is None:
                return None
            component[name] = tuple(map(int, pair.groups()))
        x, y = component["screencoord"]
        width, height = component["size"]
        if not (0 <= x < x + width <= screen_size[0] and 0 <= y < y + height <= screen_size[1]):
            return None
        return x, y, x + width, y + height

    inner, outer = pixel_rect(node), pixel_rect(parent)
    if (inner is None or outer is None or outer == (0, 0, *screen_size)
            or not (outer[0] <= inner[0] < inner[2] <= outer[2]
                    and outer[1] <= inner[1] < inner[3] <= outer[3])):
        return None
    classes = [value for key, value in parent.attrib.items() if _local_name(key) == "class"]
    return ObservedInputGeometry(
        control.ref, control.label, control.role, inner, _local_name(parent.tag),
        _observed_name(parent.get("name", ""))[:240], classes[0][:240] if len(classes) == 1 else "",
        outer, screen_size,
    )


def _native_evidence(
    node: Any, *, parent: Any, ancestor_states: tuple[bool | None, bool | None],
    bounds: tuple | None, screen_size: tuple[int, int] | None,
) -> NativeDesktopEvidence:
    action_fields: dict[str, set[str]] = {}
    complete = True
    for key in node.attrib:
        if not _protocol_attribute(key, "action"):
            continue
        match = re.fullmatch(r"([a-z][a-z0-9-]{0,63})_(desc|kb)", _local_name(key))
        if match is None:
            complete = False
            continue
        fields = action_fields.setdefault(match[1], set())
        if match[2] in fields:
            complete = False
        fields.add(match[2])
    if len(action_fields) > 16 or any(fields != {"desc", "kb"} for fields in action_fields.values()):
        complete = False
    actions = tuple(sorted(action_fields)[:16])
    component = {_local_name(key): value for key, value in node.attrib.items()
                 if _protocol_attribute(key, "component")}
    if bounds is None or _component_bounds(component, screen_size) != bounds:
        complete = False
    state_names = ("enabled", "sensitive", "focusable", "visible", "showing")
    # Unknown is absence, not an unrecognized value or conflicting serialization.
    for name in state_names:
        values = [value.lower() for key, value in node.attrib.items()
                  if _protocol_attribute(key, "state") and _local_name(key) == name]
        if len(values) > 1 or any(value not in {"true", "false"} for value in values):
            complete = False
    return NativeDesktopEvidence(
        actions=actions, complete=complete,
        **{name: _native_state(node, name) for name in state_names},
        parent_enabled=(_native_state(parent, "enabled")
                        if parent is not None and _local_name(parent.tag) != "application" else None),
        parent_sensitive=(_native_state(parent, "sensitive")
                          if parent is not None and _local_name(parent.tag) != "application" else None),
        ancestors_enabled=ancestor_states[0], ancestors_sensitive=ancestor_states[1],
        leaf=len(node) == 0,
    )


def _native_toggle_state(node: Any) -> tuple[bool, bool | None]:
    """Read this producer's complete positive state set, never bool(missing).

    OSWorld _create_atspi_node enumerates getState().get_states() before
    constructing a node and writes every returned state as true. Thus a
    missing checked flag is known off only within that recognized, intact
    state serialization. Mixed/defunct/foreign/conflicting evidence is unknown.
    """
    states = {_local_name(key): value for key, value in node.attrib.items()
              if _protocol_attribute(key, "state")}
    required = {"enabled", "sensitive", "visible", "showing"}
    relevant = required | {"checked", "indeterminate", "defunct", "invalid"}
    if (
        not required <= states.keys()
        or any(value not in {"true", "false"} for value in states.values())
        or any(_local_name(key) in relevant and not _protocol_attribute(key, "state")
               for key in node.attrib)
        or any(states.get(name) == "true" for name in ("indeterminate", "defunct", "invalid"))
    ):
        return False, None
    return True, states.get("checked") == "true"


def _descendant_label(node: Any, atspi: bool, app: str) -> str:
    """Use visible label descendants, without merging another control's text."""
    pending = [(child, 1) for child in reversed(list(node))]
    while pending:
        child, depth = pending.pop()
        if depth > 5:
            continue
        role = _local_name(child.tag)
        attrs = {_local_name(key): value for key, value in child.attrib.items()}
        if not _shown(attrs, atspi) or role in {"password", "password-text", "password-entry"}:
            continue
        if role in _CONTROL_ROLES and role != "text":
            continue
        label = _observed_name(attrs.get("name") or child.text or "")
        if role in {"label", "text", "static"} and label:
            return "GNOME top-bar clock" if _top_bar_clock(role, attrs, app, label) else label
        pending.extend((item, depth + 1) for item in reversed(list(child)))
    return ""


def _sibling_checkable_label(node: Any, parent: Any, atspi: bool, screen_size: tuple[int, int] | None) -> str:
    """Associate a unique visible label through bounded noninteractive wrappers."""
    wrappers = {"panel", "filler", "group", "grouping", "box"}
    if parent is None or _local_name(parent.tag) not in wrappers:
        return ""
    attributes = {_local_name(key): value for key, value in parent.attrib.items()}
    parent_bounds = _component_bounds(attributes, screen_size)
    if not _shown(attributes, atspi) or parent_bounds is None:
        return ""
    labels, checkables = [], []
    pending = [(sibling, parent_bounds[0], 1) for sibling in parent]
    inspected = 0
    while pending:
        inspected += 1
        if inspected > 64:
            return ""
        sibling, containing_bounds, depth = pending.pop()
        attrs = {_local_name(key): value for key, value in sibling.attrib.items()}
        if not _shown(attrs, atspi):
            continue
        role = _local_name(sibling.tag)
        if role not in wrappers and role not in {"label", "static", "text"} and normalize_role(role) not in {
            "checkbox", "radio"
        }:
            continue
        bounds = _component_bounds(attrs, screen_size)
        if bounds is None:
            return ""
        x0, y0, x1, y1 = bounds[0]
        px0, py0, px1, py1 = containing_bounds
        if not (px0 <= x0 < x1 <= px1 and py0 <= y0 < y1 <= py1):
            return ""
        if normalize_role(role) in {"checkbox", "radio"}:
            checkables.append((sibling, attrs))
        elif role in {"label", "static", "text"} and attrs.get("editable", "").lower() != "true":
            labels.append((sibling, attrs))
        elif role in wrappers:
            if depth >= 5 or attrs.get("focusable", "").lower() == "true":
                return ""
            pending.extend((child, bounds[0], depth + 1) for child in sibling)
    if len(labels) != 1 or len(checkables) != 1 or checkables[0][0] is not node:
        return ""
    for _, attrs in (labels[0], checkables[0]):
        bounds = _component_bounds(attrs, screen_size)
        if bounds is None:
            return ""
        x0, y0, x1, y1 = bounds[0]
        px0, py0, px1, py1 = parent_bounds[0]
        if not (px0 <= x0 < x1 <= px1 and py0 <= y0 < y1 <= py1):
            return ""
    label, attrs = labels[0]
    return _observed_name(attrs.get("name") or label.text or "")


def _editable_row_label(
    node: Any, parent_nodes: tuple[Any, ...], atspi: bool, screen_size: tuple[int, int] | None,
) -> str:
    """Name an anonymous field only through a unique bounded form-row pairing.

    Labels must be direct siblings of the field's transparent wrapper branch,
    with the same full vertical extent. Neither proximity nor tree order is a
    label relation. A named, noneditable units selector may share that branch.
    """
    wrappers = {"panel", "filler", "group", "grouping", "box"}
    label_roles = {"label", "static", "text"}

    def component(item):
        attrs = {_local_name(key): value for key, value in item.attrib.items()}
        bounds = _component_bounds(attrs, screen_size)
        return attrs, bounds[0] if bounds else None

    def inside(inner, outer):
        return (inner is not None and outer is not None
                and outer[0] <= inner[0] < inner[2] <= outer[2]
                and outer[1] <= inner[1] < inner[3] <= outer[3])

    def transparent(item, attrs):
        return (_local_name(item.tag) in wrappers
                and not any(attrs.get(flag, "").lower() == "true" for flag in ("focusable", "editable")))

    target_attrs, target_bounds = component(node)
    if (target_attrs.get("editable", "").lower() != "true" or target_bounds is None
            or not _shown(target_attrs, atspi)):
        return ""
    branch = node
    branch_bounds = target_bounds
    for parent in parent_nodes[:3]:
        parent_attrs, parent_bounds = component(parent)
        if (not transparent(parent, parent_attrs) or not _shown(parent_attrs, atspi)
                or not inside(branch_bounds, parent_bounds)):
            return ""
        siblings = []
        if len(parent) > 64:
            return ""
        for sibling in parent:
            attrs, bounds = component(sibling)
            if not _shown(attrs, atspi):
                continue
            if not inside(bounds, parent_bounds):
                return ""
            is_label = (_local_name(sibling.tag) in label_roles
                        and not any(attrs.get(flag, "").lower() == "true"
                                    for flag in ("editable", "focusable")))
            siblings.append((sibling, attrs, bounds, is_label))
        labels = [entry for entry in siblings if entry[3]]
        if not labels:
            branch, branch_bounds = parent, parent_bounds
            continue
        # Stop at the nearest container with labels, including an ambiguous or
        # unaligned row: a more distant ancestor must not override that evidence.
        labels = [entry for entry in labels
                  if entry[2][1::2] == branch_bounds[1::2] and entry[2][2] <= branch_bounds[0]]
        if len(labels) != 1 or target_bounds[1::2] != branch_bounds[1::2]:
            return ""
        label, label_attrs, label_bounds, _ = labels[0]
        owners = [entry[0] for entry in siblings if not entry[3]
                  and entry[2][1::2] == label_bounds[1::2] and entry[2][0] >= label_bounds[2]]
        if len(owners) != 1 or owners[0] is not branch:
            return ""
        pending, fields, selectors, inspected = [(branch, parent_bounds, 0)], [], [], 0
        while pending:
            child, containing_bounds, depth = pending.pop()
            inspected += 1
            if inspected > 64 or depth > 4:
                return ""
            attrs, bounds = component(child)
            if not _shown(attrs, atspi):
                continue
            if not inside(bounds, containing_bounds):
                return ""
            role = _local_name(child.tag)
            if attrs.get("editable", "").lower() == "true":
                if role != "text" and normalize_role(role) not in {"textbox", "spinbutton"}:
                    return ""
                fields.append(child)
                if len(child):
                    return ""
            elif transparent(child, attrs):
                pending.extend((item, bounds, depth + 1) for item in child)
            elif (normalize_role(role) == "combobox" and attrs.get("name", "").strip()
                  and not any(_shown(component(item)[0], atspi) for item in child)):
                # The units selector keeps its own accessible name. An editable
                # selector or a visible nested input would make this row ambiguous.
                selectors.append(bounds)
            else:
                return ""
        if len(fields) != 1 or fields[0] is not node:
            return ""
        if (len(selectors) > 1 or any(bounds[1::2] != target_bounds[1::2]
                                     or bounds[0] < target_bounds[2] for bounds in selectors)):
            return ""
        return _observed_name(label_attrs.get("name") or label.text or "")
    return ""


def _own_row_bounds(node: Any, attrs: dict, atspi: bool, screen_size: tuple[int, int] | None, bounds):
    """Ground an aggregate tree/list row in its own observed label region.

    The row keeps its identity and action. A label with only ``click-ancestor``
    delegates to that row; it is not promoted to an independent control.
    """
    if bounds is None:
        return None
    parent_box = bounds[0]
    name = " ".join(attrs.get("name", "").split())
    wrappers = {"panel", "filler", "group", "grouping", "box", "section", "tree", "list"}
    label_roles = {"label", "static", "text", "section"}
    labels, blocked = [], []
    pending = [(child, 1) for child in node]
    inspected = 0

    def inside(box):
        return (box is not None and parent_box[0] <= box[0] < box[2] <= parent_box[2]
                and parent_box[1] <= box[1] < box[3] <= parent_box[3])

    # Detect aggregate rows even behind an independently interactive wrapper.
    # Such a boundary cannot hide a child row and restore the unsafe center.
    while pending:
        child, depth = pending.pop()
        inspected += 1
        if inspected > 128 or depth > 6:
            return None
        child_attrs = {_local_name(key): value for key, value in child.attrib.items()}
        if not _shown(child_attrs, atspi):
            continue
        if normalize_role(_local_name(child.tag)) in {"treeitem", "listitem"}:
            child_bounds = _component_bounds(child_attrs, screen_size)
            blocked.append(child_bounds[0] if child_bounds else None)
        pending.extend((item, depth + 1) for item in child)
    if not blocked:
        return bounds
    if not all(inside(box) for box in blocked):
        return None
    pending = [(child, 1) for child in node]
    inspected = 0
    while pending:
        child, depth = pending.pop()
        inspected += 1
        if inspected > 128 or depth > 6:
            return None
        child_attrs = {_local_name(key): value for key, value in child.attrib.items()}
        if not _shown(child_attrs, atspi):
            continue
        raw_role = _local_name(child.tag)
        role = normalize_role(raw_role)
        child_bounds = _component_bounds(child_attrs, screen_size)
        box = child_bounds[0] if child_bounds else None
        if role in {"treeitem", "listitem"}:
            # Descendant rows own their own labels, including identical names.
            continue
        actions = {key.rsplit("_", 1)[0] for key in child_attrs if key.endswith(("_desc", "_kb"))}
        editable = child_attrs.get("editable", "").lower() == "true"
        focusable = child_attrs.get("focusable", "").lower() == "true"
        child_name = " ".join((child_attrs.get("name") or child.text or "").split())
        own_label = (
            name and child_name == name and raw_role in label_roles and not editable and not len(child)
            and actions <= {"click-ancestor"}
            and (not focusable or (raw_role == "section" and actions == {"click-ancestor"}))
        )
        if own_label:
            # Keep invalid/duplicate labels as ambiguity; never pick the closest.
            labels.append(child_bounds if inside(box) else None)
            continue
        if raw_role in _CONTROL_ROLES or editable or actions - {"click-ancestor"} or focusable:
            if not inside(box):
                return None
            blocked.append(box)
            continue
        if raw_role in wrappers:
            if not inside(box):
                return None
            pending.extend((item, depth + 1) for item in child)
    if len(labels) != 1 or labels[0] is None:
        return None
    own_box = labels[0][0]
    if any(own_box[0] < box[2] and box[0] < own_box[2]
           and own_box[1] < box[3] and box[1] < own_box[3] for box in blocked):
        return None
    return labels[0]


def _numeric_edit_buffer(
    node: Any, value: str, bounds: tuple | None, screen_size: tuple[int, int] | None,
    metadata: dict | None,
) -> str | None:
    """Retain same-node editor text only with precise producer evidence.

    AT-SPI Value and editable text are distinct observations. Neither a child
    label, valuetext, nor a foreign attribute establishes this editor buffer.
    The focused-only gate avoids treating an unrelated display as pending input.
    """
    if _local_name(node.tag) != "spin-button" or any(
        _native_state(node, state) is not True
        for state in ("editable", "focused", "enabled", "sensitive", "visible", "showing")
    ):
        return None
    committed = [
        text for key, text in node.attrib.items()
        if _local_name(key) in {"value", "current"}
    ]
    trusted_value = node.get("{https://accessibility.ubuntu.example.org/ns/value}value")
    geometry = {
        _local_name(key): text for key, text in node.attrib.items()
        if _protocol_attribute(key, "component")
    }
    if (
        trusted_value is None or committed != [trusted_value] or value != trusted_value
        or bounds is None or _component_bounds(geometry, screen_size) != bounds
        or node.text is None or len(node) != 0
    ):
        return None
    if len(node.text) > 128:
        if metadata is not None:
            metadata["complete"] = False
        return None
    return node.text


def _desktop_control(
    node: Any,
    attrs: dict,
    *,
    atspi: bool,
    app: str,
    window: str,
    ancestors: tuple[str, ...],
    active: bool,
    bounds: tuple | None,
    index: int,
    parent: Any,
    parent_nodes: tuple[Any, ...],
    screen_size: tuple[int, int] | None,
    native_ancestor_states: tuple[bool | None, bool | None] = (None, None),
    modal_context: bool = False,
    modal_owner_ref: str = "",
    metadata: dict | None = None,
) -> AgentControl | None:
    raw_role = _local_name(node.tag)
    editable = attrs.get("editable", "").lower() == "true"
    interactive = raw_role in _CONTROL_ROLES and (raw_role != "text" or editable)
    if raw_role in {"list-item", "tree-item", "table-cell"} and not any(
        attrs.get(key, "").lower() == "true" for key in ("selectable", "focusable", "selected")
    ):
        interactive = False
    role = "textbox" if raw_role == "text" and editable else normalize_role(raw_role)
    label = _observed_name(attrs.get("name", ""))
    checkable = raw_role in _CONTROL_ROLES and role in {"checkbox", "radio"}
    has_toggle_evidence = atspi and role == "checkbox" and any(
        _local_name(key) in {"toggle_desc", "toggle_kb"} for key in node.attrib
    )
    focusable = (_native_state(node, "focusable") is True if has_toggle_evidence
                 else attrs.get("focusable", "").lower() == "true")
    anonymous_readonly_checkable = checkable and not label and not focusable
    if not label and not editable and interactive:
        label = _descendant_label(node, atspi, app)
        if not label and role in {"checkbox", "radio"}:
            label = _sibling_checkable_label(node, parent, atspi, screen_size)
    if not label and editable and role in {"textbox", "spinbutton"}:
        label = _editable_row_label(node, parent_nodes, atspi, screen_size)
    if (
        not label
        and raw_role not in {"entry", "text", "text-entry", "spin-button"}
        and attrs.get("focusable", "").lower() != "true"
    ):
        interactive = False
    if anonymous_readonly_checkable:
        # Label association establishes whose state this is, not input support.
        interactive = False
    modal = bool(atspi and _native_state(node, "modal") is True)
    stateful = modal or any(
        attrs.get(key, "").lower() == "true"
        for key in ("active", "focused", "checked", "selected", "expanded")
    )
    if not label and not (node.text or "").strip() and not stateful and not interactive and not checkable:
        return None
    if _top_bar_clock(raw_role, attrs, app, label or node.text or ""):
        label = "GNOME top-bar clock"
    disabled = (
        any(attrs.get(key, "").lower() != "true" for key in ("enabled", "sensitive"))
        if atspi
        else any(attrs.get(key, "").lower() == "false" for key in ("enabled", "sensitive"))
    )
    # Do not upgrade actionless state copies. A direct action-shaped field
    # selects the separate evidence route even if its namespace is invalid;
    # invalid evidence must not fall back to ordinary flattened state claims.
    native_toggle = anonymous_readonly_checkable and has_toggle_evidence
    native = (
        _native_evidence(node, parent=parent, ancestor_states=native_ancestor_states,
                         bounds=bounds, screen_size=screen_size)
        if atspi and (raw_role == "canvas" or native_toggle) else None
    )
    if native is not None and native_toggle:
        state_complete, checked_state = _native_toggle_state(node)
        native = NativeDesktopEvidence(
            **{**asdict(native), "state_set_complete": state_complete, "checked_state": checked_state}
        )
    if native is not None:
        # Keep missing states as unknown in native evidence. A positive native
        # item affordance may authorize a physical gesture without inventing
        # enabled=true. The shared helper still rejects every unsupported item.
        disabled = not native.complete or any(state is False for state in (
            native.enabled, native.sensitive, native.ancestors_enabled, native.ancestors_sensitive,
        ))
        # This narrowly supported native item route grants pointer gestures,
        # not text input/value capability, even if an attribute says editable.
        # A real rename editor must appear as its own observed text control.
        editable = False
    input_type = role if role in {"checkbox", "radio"} else "number" if role == "spinbutton" else ""
    if raw_role == "toggle-button":
        input_type = "checkbox"
    tag = "input" if input_type or role == "textbox" else "a" if role == "link" else "button"
    if not interactive:
        tag = "text" if raw_role in {"label", "text", "static", "paragraph", "heading"} else "container"
        if not checkable:
            input_type = ""
    if role == "textbox":
        input_type = "text"
    value = ""
    if editable or role in {"textbox", "spinbutton"}:
        value = str(attrs.get("value", attrs.get("current", node.text or "")))[:2000]
    if interactive and role in {"treeitem", "listitem"}:
        bounds = _own_row_bounds(node, attrs, atspi, screen_size, bounds)

    def observed_state(name: str) -> bool:
        # Native item verification must use the same producer evidence as its
        # input gate. Flattening a foreign state namespace must not manufacture
        # focus/selection or a capability whose predicate could pass after input.
        return (_native_state(node, name) is True if native is not None
                else attrs.get(name, "").lower() == "true")

    return AgentControl(
        ref=f"d{index}",
        label=label,
        tag=tag,
        role=role,
        input_type=input_type,
        value=value,
        disabled=disabled,
        checked=(native.checked_state is True if native_toggle and native is not None
                 else attrs.get("checked", "").lower() == "true"),
        box=bounds[1] if bounds else None,
        app=app,
        window=window,
        ancestors=ancestors,
        focused=observed_state("focused"),
        selected=observed_state("selected"),
        expanded=observed_state("expanded"),
        editable=editable,
        active=active,
        selectable=any(observed_state(key) for key in ("selectable", "selected")),
        expandable=any(observed_state(key) for key in ("expandable", "expanded")),
        native=native,
        edit_buffer=(
            _numeric_edit_buffer(node, value, bounds, screen_size, metadata)
            if atspi and editable and not disabled else None
        ),
        modal_context=modal_context,
        modal=modal,
        modal_owner_ref=f"d{index}" if modal and tag == "container" else modal_owner_ref,
    )


def _accessibility_observation(
    raw: Any,
    *,
    screen_size: tuple[int, int] | None = None,
    controls_out: list[AgentControl] | None = None,
    metadata: dict | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    """Keep visible UI order while removing AT-SPI enumeration and clock churn."""
    if metadata is not None:
        metadata["complete"] = False
    if not isinstance(raw, str) or not raw or len(raw) > 2_000_000:
        return "OSWorld desktop", "", ()
    if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
        return "OSWorld desktop", "", ()
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, ValueError):
        return "OSWorld desktop", "", ()
    if metadata is not None:
        metadata["complete"] = True
    # OSWorld serializes AT-SPI's positive state set: absence of showing means
    # not currently showing. Plain XML adapters without state namespaces retain
    # their previous explicit-false convention.
    atspi = any(
        key.startswith(("{http://", "{https://")) and key.split("}", 1)[0].endswith("/ns/state")
        for node in root.iter()
        for key in node.attrib
    )
    lines: list[str] = []
    semantic: list[str] = []
    title = "OSWorld desktop"
    title_priority = -1
    roots = list(root) if _local_name(root.tag) in {"desktop", "desktop-frame"} else [root]
    # The desktop's application enumeration is not z-order. Only this level is
    # stabilized: order inside each application/window remains significant.
    applications = iter(
        sorted(
            (node for node in roots if _local_name(node.tag) == "application"),
            key=lambda node: node.get("name", ""),
        )
    )
    roots = [next(applications) if _local_name(node.tag) == "application" else node for node in roots]
    root_states = (
        tuple(_native_state(root, key) for key in ("enabled", "sensitive"))
        if _local_name(root.tag) in {"desktop", "desktop-frame"} else (None, None)
    )
    pending = [(node, "", None, "", (), False, None, (), root_states, False, "")
               for node in reversed(roots)]
    emitted_apps: set[int] = set()
    inspected = 0
    characters = 0
    input_geometry = []
    focused_inputs = 0
    while pending and inspected < 6000 and characters < _OBSERVED_TEXT_LIMIT:
        (node, app, app_node, window, ancestors, window_active, parent,
         parent_nodes, native_ancestor_states, modal_context, modal_owner_ref) = pending.pop()
        inspected += 1
        attributes = {_local_name(key): value for key, value in node.attrib.items()}
        role = _local_name(node.tag)
        full_name = " ".join(attributes.get("name", "").split())
        full_text = " ".join((node.text or "").split())
        name = full_name[:_OBSERVED_NAME_LIMIT]
        # Completeness describes retained normalized evidence, not raw whitespace
        # length. Any actual suffix loss remains incomplete and fails closed.
        if metadata is not None and (
            len(full_name) > _OBSERVED_NAME_LIMIT or len(full_text) > _OBSERVED_BODY_LIMIT
        ):
            metadata["complete"] = False
        if role == "application":
            application_states = tuple(
                _combine_native_state(prior, _native_state(node, key))
                for prior, key in zip(native_ancestor_states, ("enabled", "sensitive"), strict=True)
            )
            pending.extend((child, name, node, "", (), False, node, (node,), application_states, False, "")
                           for child in reversed(list(node)))
            continue
        if not _shown(attributes, atspi):
            continue
        if role in {"password", "password-text", "password-entry"}:
            continue
        modal_context = modal_context or (atspi and _native_state(node, "modal") is True)
        text = full_text[:_OBSERVED_BODY_LIMIT]
        active = attributes.get("active", "").lower() == "true"
        focused = attributes.get("focused", "").lower() == "true"
        if role in {"frame", "window", "dialog"}:
            window = name or window
            window_active = active or focused
        bounds = _component_bounds(attributes, screen_size)
        if controls_out is not None:
            control = _desktop_control(
                node,
                attributes,
                atspi=atspi,
                app=app,
                window=window,
                ancestors=ancestors,
                active=window_active,
                bounds=bounds,
                index=len(controls_out),
                parent=parent,
                parent_nodes=parent_nodes,
                screen_size=screen_size,
                native_ancestor_states=native_ancestor_states,
                modal_context=modal_context,
                modal_owner_ref=modal_owner_ref,
                metadata=metadata,
            )
            if control is not None:
                if len(controls_out) < 512:
                    controls_out.append(control)
                    if (atspi and window_active and role in {"entry", "text-entry", "text", "spin-button"}
                            and _native_state(node, "focused") is True):
                        focused_inputs += 1
                        context = _input_parent_geometry(
                            node, parent, control, screen_size, native_ancestor_states,
                        )
                        if context is not None:
                            input_geometry.append(context)
                    if control.modal is True and control.tag == "container":
                        modal_owner_ref = control.ref
                elif metadata is not None:
                    metadata["complete"] = False
        ancestor_name = "GNOME top-bar clock" if _top_bar_clock(role, attributes, app, name or text) else name
        parents = (*ancestors, f"{role}: {ancestor_name}")[-8:] if ancestor_name else ancestors
        pending.extend(
            (child, app, app_node, window, parents, window_active, node, (node, *parent_nodes[:2]),
             tuple(_combine_native_state(prior, _native_state(node, key))
                   for prior, key in zip(native_ancestor_states, ("enabled", "sensitive"), strict=True)),
             modal_context, modal_owner_ref)
            for child in reversed(list(node))
        )
        # A visible window alone does not identify the current keyboard context.
        # Keep the generic desktop title unless AT-SPI reports active/focused.
        priority = 3 if active else 2 if focused else -1
        if role in {"frame", "window", "dialog"} and name and priority > title_priority:
            title, title_priority = name, priority
        flags = [
            flag
            for flag in ("active", "focused", "checked", "selected", "expanded")
            if attributes.get(flag, "").lower() == "true"
        ]
        if attributes.get("enabled", "").lower() == "false":
            flags.append("disabled")
        if not name and not text and not flags:
            continue
        if app_node is not None and id(app_node) not in emitted_apps:
            header = f"application: {app}"
            lines.append(header)
            semantic.append(header)
            characters += len(header)
            emitted_apps.add(id(app_node))
        line = f"{role}: {name}"
        if text and text != name:
            line += f" | {text}"
        if flags:
            line += f" [{', '.join(flags)}]"
        geometry = f" box_1000={list(bounds[1])}" if bounds else ""
        line += geometry
        # Preserve order and focus so keyboard actions are bound to current context.
        lines.append(line)
        # The live clock remains readable in the prompt and screenshot. Only its
        # narrowly identified label is stable for plan freshness/effect checks.
        stable = (
            f"{role}: [GNOME top-bar clock]" if _top_bar_clock(role, attributes, app, name or text) else line
        )
        if stable != line and flags:
            stable += f" [{', '.join(flags)}]"
        if stable != line:
            stable += geometry
        if bounds:
            # Preserve sub-normalization pixel shifts in the freshness scope;
            # the model only needs the normalized coordinates it can output.
            stable += f" bounds_px={list(bounds[0])}"
        semantic.append(stable)
        characters += len(line)
    text = "\n".join(lines)
    if metadata is not None and (pending or len(text) > _OBSERVED_TEXT_LIMIT):
        metadata["complete"] = False
    if metadata is not None:
        metadata["input_geometry"] = (tuple(input_geometry)
                                      if metadata["complete"] and focused_inputs == 1 else ())
    return title, text[:_OBSERVED_TEXT_LIMIT], tuple(semantic)


def screenshot_image(raw: bytes) -> WindowImage:
    from PIL import Image

    if not isinstance(raw, bytes) or not raw or len(raw) > 32_000_000:
        raise ValueError("OSWorld screenshot is missing or exceeds its size limit")
    with Image.open(io.BytesIO(raw)) as source:
        width, height = source.size
        if width < 40 or height < 40 or width * height > 16_000_000:
            raise ValueError("OSWorld screenshot dimensions are outside allowed bounds")
        image = source.convert("RGB")
    sample = image.convert("L").resize((SAMPLE_EDGE, SAMPLE_EDGE)).tobytes()
    image.thumbnail((1280, 1280))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=75)
    return WindowImage(output.getvalue(), sample, (0, 0, width, height))


def _valid_box(box: Any) -> bool:
    return valid_desktop_box(box)


def _fresh(before: AgentSnapshot, after: AgentSnapshot) -> bool:
    if (
        before.scope_key() != after.scope_key()
        or before.a11y_complete != after.a11y_complete
        or before.controls != after.controls
        or before.image is None
        or after.image is None
    ):
        return False
    # Cursor movement/blinking is normal on desktop screenshots. Meaningful
    # frame changes still invalidate the decision immediately before input.
    return (
        before.image.region == after.image.region
        and changed_fraction(before.image.sample, after.image.sample) <= 0.01
    )


@dataclass(frozen=True)
class _Binding:
    candidate: dict
    step: AgentStep
    snapshot: AgentSnapshot
    target: AgentControl | None = None
    typing_target: AgentControl | None = None
    planned: AgentSnapshot | None = None
    navigation_app: str = ""
    button: str = "left"
    click_count: int = 1
    property_step: AgentStep | None = None
    confirmation: bool = False
    numeric_finalize: _NumericFinalizePending | None = None


@dataclass(frozen=True)
class _NumericFinalizePending:
    candidate: ActionCandidate
    candidate_data: dict
    receipt: ActionReceipt
    receipt_data: dict
    stage: AgentStep
    stage_data: dict
    field_step: AgentStep
    target: AgentControl
    snapshot: AgentSnapshot
    received_at: float
    environment_steps: int
    observation_count: int
    key: str


@dataclass(frozen=True)
class _RecoverySource:
    candidate: dict
    receipt: ActionReceipt
    receipt_data: dict
    before: AgentSnapshot
    environment_steps: int


@dataclass(frozen=True)
class _InputObservationEvidence:
    binding: _Binding
    step_data: dict
    candidate_data: dict
    receipt: ActionReceipt
    receipt_data: dict
    actual_before: AgentSnapshot
    actual_final_after: AgentSnapshot
    before_fingerprint: str
    after_fingerprint: str
    image_sha256: str
    owner_app: str
    received_at: float
    environment_steps: int
    observation_count: int
    input_steps: int
    verification: Verification
    verification_data: dict
    from_pending: bool


@dataclass(frozen=True)
class _StaleRejection:
    candidate: dict
    receipt: ActionReceipt
    receipt_data: dict
    environment_steps: int
    observation_count: int
    snapshot: AgentSnapshot
    received_at: float


@dataclass(frozen=True)
class _PropertyNavigationPending:
    binding: _Binding
    candidate: dict
    receipt: ActionReceipt
    receipt_data: dict
    environment_steps: int
    observation_count: int
    owner_app: str
    attempts: int = 0


@dataclass(frozen=True)
class _RecoveryContext:
    source: _RecoverySource
    app: str
    minimum_counts: dict
    new_control_count: int


def _actionable_structure(snapshot: AgentSnapshot, app: str, *, active_only: bool) -> Counter:
    """Count executable structure, without labels, values, focus, refs or boxes."""
    result: Counter = Counter()
    for control in snapshot.controls:
        if " ".join(control.app.split()).casefold() != app or (active_only and not control.active):
            continue
        evidence = navigation_control_evidence(control)
        if not evidence or not evidence["operations"]:
            continue
        shape = (control.tag, evidence["role"], evidence["input_type"], evidence["operations"],
                 evidence.get("click_modes", ()))
        context = (evidence["window"], evidence["ancestors"])
        result[(shape, context)] += 1
    return result


def _new_actionable_context(
    source: _RecoverySource, binding: _Binding, after: AgentSnapshot, verification: Verification,
) -> _RecoveryContext | None:
    facts = verification.details.get("postcondition_evidence", ())
    missing = [fact for fact in facts if not fact.get("satisfied")]
    if (
        verification.passed or not missing or any(fact.get("outcome") != "unknown" for fact in missing)
        or binding.step.operation != "click" or binding.target is None
        or not source.before.a11y_complete or not after.a11y_complete
    ):
        return None
    app = " ".join(binding.target.app.split()).casefold()
    if not app or binding.target not in source.before.controls:
        return None
    before = _actionable_structure(source.before, app, active_only=False)
    current = _actionable_structure(after, app, active_only=True)
    # A renamed label/window or moved control alone is not a newly opened UI.
    # Require a net increase of executable shapes as well as a new context key.
    old_shapes: Counter = Counter()
    new_shapes: Counter = Counter()
    for (shape, _), count in before.items():
        old_shapes[shape] += count
    for (shape, _), count in current.items():
        new_shapes[shape] += count
    remaining = new_shapes - old_shapes
    minimum_counts = {}
    new_count = 0
    for key, count in sorted((current - before).items()):
        added = min(count, remaining[key[0]])
        if added:
            minimum_counts[key] = current[key]
            new_count += added
            remaining[key[0]] -= added
    return _RecoveryContext(source, app, minimum_counts, new_count) if new_count else None


def _postcondition_evidence(conditions: tuple, snapshot: AgentSnapshot) -> list[dict]:
    facts = []
    for condition in conditions:
        evidence = condition_evidence(condition, snapshot)
        facts.append({
            "kind": condition.kind,
            "target": asdict(condition.target) if isinstance(condition.target, AgentTarget) else None,
            **evidence,
            "satisfied": evidence["outcome"] == "true",
        })
    return facts


def _navigation_scope(snapshot: AgentSnapshot, step: AgentStep, *, handoff_app: str = "") -> str:
    # Compare an affordance/state multiset, not the entire accessibility tree.
    # A focus-only anonymous panel can appear/disappear while the actual action
    # surface is unchanged (e.g. a popup omitted by the application toolkit).
    # That is a reason to ask the VLM, never permission to toggle the menu again.
    # Ref/order/focus/activation/geometry still participate in input preflight.
    controls = navigation_scope_controls(snapshot, step.target, handoff_app=handoff_app)
    public = sorted(
        json.dumps(evidence, sort_keys=True)
        for control in controls
        if (evidence := navigation_control_evidence(control)) is not None
    )
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def _navigation_progress(
    step: AgentStep, before: AgentSnapshot, after: AgentSnapshot,
    *, before_app: str = "", after_app: str = "",
) -> tuple[bool, str, str]:
    before_scope = _navigation_scope(before, step, handoff_app=before_app)
    after_scope = _navigation_scope(after, step, handoff_app=after_app)
    progressed = (
        before.a11y_complete
        and after.a11y_complete
        and bool(navigation_scope_controls(after, step.target, handoff_app=after_app))
        and before_scope != after_scope
    )
    return progressed, before_scope, after_scope


class _InputDeadlineExpired(TimeoutError):
    """No further local input may be submitted for this execution."""


class _TextFocusGuardRejected(ValueError):
    """Local focus-check failure only; never carries target identity or text."""

    def __init__(self, stage, snapshot, current):
        super().__init__("desktop focus changed before text input")

        def observed_bool(value):
            return value if type(value) is bool else None

        self.diagnostic = {
            "version": "text_focus_guard_diagnostic_v65", "stage": stage,
            "fresh_observe_returned": True,
            "a11y_complete": observed_bool(getattr(snapshot, "a11y_complete", None)),
            "image_present": getattr(snapshot, "image", None) is not None if snapshot is not None else None,
            "target_matched": current is not None,
            "observed_disabled": observed_bool(current.disabled) if current is not None else None,
            "observed_editable": observed_bool(current.editable) if current is not None else None,
            "observed_focused_true": observed_bool(current.focused) if current is not None else None,
            "typing_not_sent": True,
        }


def _bounded_text_focus_diagnostic(error):
    if type(error) is not _TextFocusGuardRejected:
        return None
    item = getattr(error, "diagnostic", None)
    facts = {"fresh_observe_returned", "a11y_complete", "image_present", "target_matched",
             "observed_disabled", "observed_editable", "observed_focused_true", "typing_not_sent"}
    if (type(item) is not dict or set(item) != facts | {"version", "stage"}
            or item["version"] != "text_focus_guard_diagnostic_v65"
            or type(item["stage"]) is not str
            or item["stage"] not in {"after_focus_click", "after_select_all", "unspecified"}
            or item["fresh_observe_returned"] is not True or item["typing_not_sent"] is not True
            or type(item["target_matched"]) is not bool
            or any(value is not None and type(value) is not bool
                   for key, value in item.items() if key in facts)):
        return None
    if not item["target_matched"] and any(item[key] is not None for key in (
            "observed_disabled", "observed_editable", "observed_focused_true")):
        return None
    if (item["target_matched"] and item["observed_disabled"] is False
            and item["observed_editable"] is True and item["observed_focused_true"] is True):
        return None
    return dict(item)


class OSWorldBackend:
    """Adapt screenshot+accessibility and computer_13; never own VM lifecycle."""

    fresh_preflight_guaranteed = True

    def __init__(self, env: Any, *, pause: float = 0.5, menu_text_provider: Any = None,
                 image_only_fill_intermediates: bool = False,
                 satisfied_property_readonly_fast_path: bool = False):
        if not 0 <= pause <= 10:
            raise ValueError("OSWorld action pause must be between zero and ten seconds")
        self.env = env
        self.pause = pause
        if type(image_only_fill_intermediates) is not bool:
            raise ValueError("image-only fill intermediates must be boolean")
        self.image_only_fill_intermediates = image_only_fill_intermediates
        if type(satisfied_property_readonly_fast_path) is not bool:
            raise ValueError("satisfied property read-only fast path must be boolean")
        self.satisfied_property_readonly_fast_path = satisfied_property_readonly_fast_path
        self.intermediate_observation_metrics = {"image_only_steps": 0}
        self._observation_phase_totals = {
            bucket: {phase: {"calls": 0, "failures": 0, "duration_ns": 0}
                     for phase in ("acquisition", "snapshot_build")}
            for bucket in ("full_observe", "full_step", "image_only_step")
        }
        if image_only_fill_intermediates:
            remote = (getattr(env, "supports_image_only_steps", None) is True
                      and callable(getattr(env, "step_image_only", None)))
            native = (type(getattr(env, "require_a11y_tree", None)) is bool
                      and env.require_a11y_tree)
            if not (remote or native):
                raise ValueError("environment has not enabled image-only fill observations")
        if menu_text_provider is not None and not callable(getattr(menu_text_provider, "recognize", None)):
            raise ValueError("menu text provider must expose recognize")
        self.menu_text_provider = menu_text_provider
        self.menu_grounding_metrics = {"attempts": 0, "failures": 0, "augmented_snapshots": 0,
                                       "added_controls": 0, "wall_ms": 0.0}
        self.last_observation: dict = {}
        self.environment_steps = 0
        self.fresh_observe_calls = 0
        self._issued: dict[str, _Binding] = {}
        self._verified: dict[str, tuple[dict, ActionReceipt, Verification]] = {}
        self._property_receipts: dict[str, tuple[dict, dict, AgentStep]] = {}
        self._readonly_property_receipts: set[str] = set()
        self._pending: dict[str, tuple[_Binding, int]] = {}
        self._property_navigation_pending: dict[str, _PropertyNavigationPending] = {}
        self._property_navigation_active: str | None = None
        self._property_navigation_post: tuple[_PropertyNavigationPending, AgentSnapshot, float] | None = None
        self._numeric_edit_source: tuple[dict, ActionReceipt, dict, int] | None = None
        self._numeric_planning_snapshot: tuple[tuple, AgentSnapshot, float, int] | None = None
        self._numeric_finalize_pending: _NumericFinalizePending | None = None
        self._numeric_finalize_snapshot: _NumericFinalizePending | None = None
        self._numeric_finalize_context: _NumericFinalizePending | None = None
        self._numeric_finalize_issued: _NumericFinalizePending | None = None
        self._launch_wait_until: dict[str, float] = {}
        self._navigation_planned: AgentSnapshot | None = None
        self._navigation_apps: dict[int, tuple[AgentStep, str]] = {}
        self._action_observation: AgentSnapshot | None = None
        self._action_observation_received_at: float | None = None
        self._post_snapshot: tuple[dict, ActionReceipt, dict, AgentSnapshot, float, int] | None = None
        self._input_evidence_enabled = False
        self._input_evidence: _InputObservationEvidence | None = None
        self._stale_rejection: _StaleRejection | None = None
        self._stale_decision_snapshot: _StaleRejection | None = None
        self._recovery_source: _RecoverySource | None = None
        self._recovery_context: _RecoveryContext | None = None
        self._recovery_confirmation: tuple[dict, dict, _RecoveryContext, int] | None = None
        self._last_observed_snapshot: AgentSnapshot | None = None
        self._last_observed_received_at: float | None = None
        self._execution_deadline: float | None = None
        self._closed = False
        self._original_command = None
        self._wrapped_command = None
        controller = getattr(env, "controller", None)
        original = getattr(controller, "execute_python_command", None)
        if callable(original):
            # Some OSWorld versions discard this error result in execute_action.
            # Convert an explicit remote error into a failed local receipt.
            self._original_command = original

            def checked_command(*args, **kwargs):
                result = original(*args, **kwargs)
                if result is None or (
                    isinstance(result, dict)
                    and (
                        str(result.get("status", "")).lower() in {"error", "failed", "failure"}
                        or result.get("success") is False
                        or result.get("returncode", 0) not in (0, None)
                    )
                ):
                    raise RuntimeError("OSWorld controller rejected the typed action")
                return result

            self._wrapped_command = checked_command
            controller.execute_python_command = checked_command

    def set_input_observation_evidence_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise ValueError("input observation evidence switch must be boolean")
        previous = self._input_evidence_enabled
        self._input_evidence_enabled = enabled
        self._input_evidence = None
        return previous

    def clear_input_observation_evidence(self) -> None:
        self._input_evidence = None

    def _remember_input_observation(self, binding, candidate, receipt, before, after, received_at,
                                    verification, *, from_pending=False) -> None:
        self._input_evidence = None
        if not self._input_evidence_enabled:
            return
        try:
            count = receipt.output.get("environment_steps")
            if (not receipt.success or type(count) is not int or count <= 0
                    or before.a11y_complete is not True or after.a11y_complete is not True
                    or before.image is None or after.image is None
                    or not isinstance(after.image.jpeg, bytes) or not after.image.jpeg
                    or type(received_at) not in (int, float) or not math.isfinite(received_at)):
                return
            self._input_evidence = _InputObservationEvidence(
                binding, binding.step.to_dict(), candidate.to_dict(), receipt, receipt.to_dict(),
                before, after, before.fingerprint(), after.fingerprint(),
                hashlib.sha256(after.image.jpeg).hexdigest(),
                binding.target.app if binding.target is not None else "", received_at,
                self.environment_steps, self.fresh_observe_calls, count, verification,
                verification.to_dict(), from_pending,
            )
        except Exception:
            # Optional data retention must never replace the real input/verification outcome.
            self._input_evidence = None

    def peek_input_observation_evidence(self, candidate, receipt):
        """Read one bounded actual-input record; no I/O, ticket or input permission."""
        evidence = self._input_evidence
        if not self._input_evidence_enabled or self._closed or evidence is None:
            return None
        known = self._verified.get(receipt.receipt_id)
        if (type(evidence.input_steps) is not int or evidence.input_steps <= 0
                or type(receipt.output.get("environment_steps")) is not int
                or receipt.output["environment_steps"] != evidence.input_steps
                or evidence.actual_before.a11y_complete is not True
                or evidence.actual_final_after.a11y_complete is not True
                or evidence.actual_before.image is None
                or receipt is not evidence.receipt or receipt.to_dict() != evidence.receipt_data
                or candidate.to_dict() != evidence.candidate_data
                or evidence.binding.step.to_dict() != evidence.step_data
                or known is None or known[0] != evidence.candidate_data or known[1] is not receipt
                or known[2] is not evidence.verification
                or known[2].to_dict() != evidence.verification_data
                or self.environment_steps != evidence.environment_steps
                or self.fresh_observe_calls != evidence.observation_count
                or evidence.actual_before.fingerprint() != evidence.before_fingerprint
                or evidence.actual_final_after.fingerprint() != evidence.after_fingerprint
                or evidence.actual_final_after.image is None
                or hashlib.sha256(evidence.actual_final_after.image.jpeg).hexdigest()
                != evidence.image_sha256):
            return None
        return evidence

    def take_verified_pending_snapshot(self, candidate, receipt):
        """Consume only a final true ordinary-pending full frame, never SDK receipt data."""
        evidence = self.peek_input_observation_evidence(candidate, receipt)
        self._input_evidence = None
        if evidence is None:
            return None
        age = time.perf_counter() - evidence.received_at
        if (not evidence.from_pending or evidence.verification.passed is not True
                or evidence.verification.details.get("postconditions_met") is not True
                or not 0 <= age <= 1.0
                or evidence.actual_final_after is not self._last_observed_snapshot
                or evidence.received_at != self._last_observed_received_at):
            return None
        return (evidence.actual_final_after, evidence.received_at, evidence.after_fingerprint)

    def clear_navigation_memory_sources(self) -> None:
        """Discard cached observation/repair authority; no I/O or receipt rewriting."""
        self.clear_input_observation_evidence()
        self._post_snapshot = self._property_navigation_post = None
        self._numeric_planning_snapshot = self._numeric_finalize_snapshot = None
        self._numeric_finalize_context = None
        self._stale_rejection = self._stale_decision_snapshot = None
        self._recovery_context = self._recovery_confirmation = None

    def navigation_memory_state(self, step: AgentStep, snapshot: AgentSnapshot) -> NavigationState | None:
        """Pure public-identity projection; no I/O or input permission."""
        from .desktop_navigation_memory import navigation_state

        return navigation_state(step, snapshot, _navigation_scope(snapshot, step))

    def navigation_transition_evidence(self, candidate, receipt) -> NavigationTransition | None:
        """Peek bounded actual input evidence without consuming another feature's record."""
        from .desktop_navigation_memory import NavigationTransition, fact_outcomes, navigation_action

        evidence = self.peek_input_observation_evidence(candidate, receipt)
        if evidence is None or not 0 <= time.perf_counter() - evidence.received_at <= 1.0:
            return None
        step = evidence.binding.step
        details = evidence.verification.details
        if (not step.navigation or step.operation != "click"
                or details.get("navigation") is not True
                or details.get("navigation_handoff") is not False
                or receipt.output.get("before_fingerprint") != evidence.before_fingerprint):
            return None
        before = self.navigation_memory_state(step, evidence.actual_before)
        after = self.navigation_memory_state(step, evidence.actual_final_after)
        if before is None or after is None:
            return None
        action = navigation_action(candidate, before)
        arrival = fact_outcomes(details.get("postcondition_evidence"))
        if action is None or arrival is None:
            return None
        return NavigationTransition(step.id, receipt.receipt_id, before, after, action,
                                    evidence.input_steps, evidence.verification.passed, arrival)

    def take_unexplained_navigation_context(self, candidate, receipt) -> dict | None:
        """Consume a verified-input information signal, never an input permission."""
        evidence = self.peek_input_observation_evidence(candidate, receipt)
        if evidence is None or not 0 <= time.perf_counter() - evidence.received_at <= 1.0:
            return None
        if evidence.from_pending and (
            evidence.actual_final_after is not self._last_observed_snapshot
            or evidence.received_at != self._last_observed_received_at
        ):
            return None
        from .desktop_unexplained_context import unexplained_navigation_context

        signal = unexplained_navigation_context(
            evidence.binding.step, evidence.actual_before, evidence.actual_final_after,
            evidence.verification, owner_app=evidence.owner_app,
        )
        if signal is None:
            return None
        self.clear_input_observation_evidence()
        return {
            **signal,
            "receipt_id": receipt.receipt_id,
            "candidate_id": candidate.id,
            "positive_environment_steps": evidence.input_steps,
            "evidence_source": "verified_pending_full" if evidence.from_pending else "sdk_after_input",
            "evidence_after_fingerprint": evidence.after_fingerprint,
            "sdk_receipt_after_fingerprint": receipt.output.get("after_fingerprint"),
        }

    @property
    def observation_phase_metrics(self) -> dict:
        """Detached bounded totals; these metrics never grant observation authority.

        Acquisition is env._get_obs or SDK input plus its response, not pure input.
        Full snapshot_build includes response validation, image/AX parsing and
        optional OCR; image-only snapshot_build validates its partial response.
        """
        return {
            "version": "observation_phase_timing_v84", "clock": "monotonic_ns",
            "phases": {bucket: {phase: dict(totals) for phase, totals in phases.items()}
                       for bucket, phases in self._observation_phase_totals.items()},
        }

    @contextmanager
    def _measure_observation_phase(self, bucket, phase):
        totals = self._observation_phase_totals[bucket][phase]
        totals["calls"] += 1
        started_ns = time.monotonic_ns()
        try:
            yield
        except BaseException:
            totals["failures"] += 1
            raise
        finally:
            totals["duration_ns"] += max(0, time.monotonic_ns() - started_ns)

    def observe(self) -> AgentSnapshot:
        self._post_snapshot = None
        self._property_navigation_post = None
        self._numeric_planning_snapshot = None
        self._numeric_finalize_snapshot = None
        self._numeric_finalize_context = None
        self._stale_rejection = self._stale_decision_snapshot = None
        self._recovery_context = None
        self._last_observed_snapshot = None
        self._last_observed_received_at = None
        if self._recovery_confirmation and self._recovery_confirmation[3] != self.fresh_observe_calls + 1:
            self._recovery_confirmation = None
        if self._closed:
            raise RuntimeError("OSWorld backend is closed")
        self.fresh_observe_calls += 1
        with self._measure_observation_phase("full_observe", "acquisition"):
            observation = self.env._get_obs()
            received_at = time.perf_counter()
        with self._measure_observation_phase("full_observe", "snapshot_build"):
            snapshot = self._snapshot(observation)
        self._last_observed_snapshot = snapshot
        self._last_observed_received_at = received_at
        return snapshot

    def current_full_observation_proof(self, snapshot, *, expected=None, source=None):
        """Pure existing-record proof for latest full observe or verified SDK-after.

        Union with the UNUSED goal-frame draft's same API; no second record or
        input permission. Pending origins retain their runner source label and
        use the actual latest full observe branch. Missing/consumed evidence is a miss.
        """
        if source is None:
            if snapshot is not self._last_observed_snapshot:
                return None
            received_at = self._last_observed_received_at
        elif type(source) is tuple and len(source) == 2:
            candidate, receipt = source
            evidence = self.peek_input_observation_evidence(candidate, receipt)
            if (evidence is None or evidence.from_pending
                    or evidence.actual_final_after is not snapshot
                    or evidence.verification.passed is not True
                    or receipt.success is not True
                    or receipt.output.get("after_fingerprint") != evidence.after_fingerprint):
                return None
            received_at = evidence.received_at
        else:
            return None
        if (self._closed or not isinstance(snapshot, AgentSnapshot)
                or snapshot.a11y_complete is not True or snapshot.image is None
                or type(snapshot.image.jpeg) is not bytes or not snapshot.image.jpeg
                or type(received_at) not in (int, float) or not math.isfinite(received_at)
                or type(self.environment_steps) is not int or self.environment_steps < 0
                or type(self.fresh_observe_calls) is not int or self.fresh_observe_calls <= 0):
            return None
        proof = (received_at, self.environment_steps, self.fresh_observe_calls,
                 snapshot.fingerprint(), hashlib.sha256(snapshot.image.jpeg).hexdigest())
        if (not 0 <= time.perf_counter() - received_at <= 1.0
                or (expected is not None and (type(expected) is not tuple or expected != proof))):
            return None
        return proof

    def _recovery_receipt_matches(self, source: _RecoverySource, candidate, receipt) -> bool:
        return (
            not self._closed and source.environment_steps == self.environment_steps
            and receipt is source.receipt and receipt.success
            and candidate.to_dict() == source.candidate and receipt.to_dict() == source.receipt_data
        )

    def _update_recovery_context(self, binding, candidate, receipt, after, verification) -> None:
        source = self._recovery_source
        self._recovery_context = None
        if source is not None and self._recovery_receipt_matches(source, candidate, receipt):
            self._recovery_context = _new_actionable_context(source, binding, after, verification)

    def take_unverified_recovery_context(self, candidate, receipt) -> dict | None:
        """Consume evidence for one fresh replan, never effect or goal success."""
        context, self._recovery_context = self._recovery_context, None
        self._recovery_source = None
        self._recovery_confirmation = None
        if context is None or not self._recovery_receipt_matches(context.source, candidate, receipt):
            return None
        known = self._verified.get(receipt.receipt_id)
        if known is None or known[2].passed:
            return None
        serialized = json.dumps((context.app, sorted(context.minimum_counts.items())), sort_keys=True)
        result = {
            "context_fingerprint": hashlib.sha256(serialized.encode()).hexdigest(),
            "new_control_count": context.new_control_count,
            "reason": "new_actionable_context_after_unconfirmed_input",
        }
        self._recovery_confirmation = (result, dict(result), context, self.fresh_observe_calls + 1)
        return result

    def confirm_unverified_recovery_context(self, context: dict, snapshot: AgentSnapshot) -> bool:
        """Confirm the ticket on exactly the next real observe, without doing I/O."""
        pending, self._recovery_confirmation = self._recovery_confirmation, None
        if pending is None:
            return False
        ticket, original, evidence, expected_observation = pending
        if (
            self._closed or context is not ticket or context != original
            or self.environment_steps != evidence.source.environment_steps
            or evidence.source.receipt.to_dict() != evidence.source.receipt_data
            or snapshot is not self._last_observed_snapshot
            or self.fresh_observe_calls != expected_observation or not snapshot.a11y_complete
        ):
            return False
        current = _actionable_structure(snapshot, evidence.app, active_only=True)
        return all(current[key] >= minimum for key, minimum in evidence.minimum_counts.items())

    def take_verified_post_snapshot(
        self, candidate: ActionCandidate, receipt: ActionReceipt
    ) -> tuple[AgentSnapshot, float] | None:
        """Consume this backend's last verified step response once, for decisions.

        The one-second age is measured since the local step response arrived,
        not since guest capture. Explicit observe/preflight never use this slot.
        Completion checks and the runner's final verification verdict remain
        the caller's responsibility; this method never establishes task success.
        """
        cached, self._post_snapshot = self._post_snapshot, None
        if cached is None or self._closed:
            return None
        candidate_data, original_receipt, receipt_data, snapshot, received_at, steps = cached
        age = time.perf_counter() - received_at
        if (
            receipt is not original_receipt or not receipt.success
            or candidate.to_dict() != candidate_data or receipt.to_dict() != receipt_data
            or steps != self.environment_steps or not 0 <= age <= 1.0
            or not self.verify(candidate, receipt).passed
        ):
            return None
        return snapshot, received_at

    def take_verified_pending_effect_snapshot(
        self, candidate: ActionCandidate, receipt: ActionReceipt,
    ) -> tuple[AgentSnapshot, float] | None:
        """Consume one actual successful navigation effect observation for decisions.

        This is a fresh pending-effect read, never an SDK step response. Input
        preflight and whole-goal checks still perform their independent reads.
        """
        cached, self._property_navigation_post = self._property_navigation_post, None
        if cached is None:
            return None
        source, snapshot, received_at = cached
        age = time.perf_counter() - received_at
        if (
            not self._property_navigation_receipt_matches(source, candidate, receipt)
            or self._property_navigation_active != receipt.receipt_id
            or self._property_navigation_pending.get(receipt.receipt_id) is not source
            or source.environment_steps != self.environment_steps
            or source.observation_count != self.fresh_observe_calls
            or snapshot is not self._last_observed_snapshot or not 0 <= age <= 1.0
            or not self.verify(candidate, receipt).passed
        ):
            return None
        return snapshot, received_at

    def consume_stale_rejection(self, candidate: ActionCandidate, receipt: ActionReceipt) -> bool:
        """Consume this exact zero-input refusal, allowing only fresh candidate generation.

        No input authority is supplied. Candidate generation can use the latest
        preflight frame once via take_stale_preflight_snapshot, or observe
        again. All selections still execute through a fresh ordinary preflight.
        """
        saved, self._stale_rejection = self._stale_rejection, None
        self._stale_decision_snapshot = None
        if saved is None or not self._stale_rejection_matches(saved, candidate, receipt):
            return False
        self._stale_decision_snapshot = saved
        return True

    def _stale_rejection_matches(self, saved: _StaleRejection, candidate, receipt) -> bool:
        return (
            not self._closed and receipt is saved.receipt and not receipt.success
            and candidate.to_dict() == saved.candidate and receipt.to_dict() == saved.receipt_data
            and saved.environment_steps == self.environment_steps
            and saved.observation_count == self.fresh_observe_calls
        )

    def take_stale_preflight_snapshot(
        self, candidate: ActionCandidate, receipt: ActionReceipt,
    ) -> tuple[AgentSnapshot, float] | None:
        """Consume the new read that refused an old candidate, for decisions only.

        The one-second local age includes snapshot processing. No explicit
        observation, input preflight or completion check uses this cache.
        """
        saved, self._stale_decision_snapshot = self._stale_decision_snapshot, None
        if saved is None or not self._stale_rejection_matches(saved, candidate, receipt):
            return None
        if not 0 <= time.perf_counter() - saved.received_at <= 1.0:
            return None
        return saved.snapshot, saved.received_at

    def _snapshot(self, observation: Any) -> AgentSnapshot:
        if not isinstance(observation, dict):
            raise ValueError("OSWorld observation must be a dictionary")
        image = screenshot_image(observation.get("screenshot"))
        controls: list[AgentControl] = []
        metadata: dict = {}
        title, text, semantic = _accessibility_observation(
            observation.get("accessibility_tree"),
            screen_size=image.region[2:],
            controls_out=controls,
            metadata=metadata,
        )
        # Keep exactly the public observation surfaces; never cache task metadata.
        self.last_observation = {
            "screenshot": observation["screenshot"],
            "accessibility_tree": observation.get("accessibility_tree"),
        }
        snapshot = AgentSnapshot(
            url="osworld://desktop",
            title=title,
            text=text,
            controls=tuple(controls),
            image=image,
            semantic_text=semantic,
            a11y_complete=metadata["complete"],
        )
        if metadata.get("input_geometry"):
            image_digest, state_digest = hashlib.sha256(image.jpeg).hexdigest(), snapshot.fingerprint()
            snapshot = replace(snapshot, input_geometry=tuple(
                replace(context, jpeg_sha256=image_digest, snapshot_fingerprint=state_digest)
                for context in metadata["input_geometry"]
            ))
        if self.menu_text_provider is None:
            return snapshot
        from .desktop_menu_grounding import ground_menu_items, native_menu_items

        items = native_menu_items(observation.get("accessibility_tree"), snapshot)
        if not any(item.can_click and item.needs_box for item in items):
            return snapshot
        self.menu_grounding_metrics["attempts"] += 1
        started = time.perf_counter()
        try:
            words = self.menu_text_provider.recognize(
                observation["screenshot"], deadline=self._execution_deadline,
            )
            augmented = ground_menu_items(snapshot, observation["screenshot"], words, items)
            if augmented is not snapshot:
                self.menu_grounding_metrics["augmented_snapshots"] += 1
                delta = len(augmented.controls) - len(snapshot.controls)
                self.menu_grounding_metrics["added_controls"] += delta
            return augmented
        except Exception as error:
            self.menu_grounding_metrics["failures"] += 1
            self.menu_grounding_metrics["last_error_type"] = type(error).__name__
            return snapshot
        finally:
            self.menu_grounding_metrics["wall_ms"] += (time.perf_counter() - started) * 1000

    def compile(
        self, step: AgentStep, planned: AgentSnapshot, current: AgentSnapshot
    ) -> tuple[ActionCandidate, ...]:
        self._stale_rejection = self._stale_decision_snapshot = None
        if self._navigation_planned is not planned:
            self._navigation_planned = planned
            self._navigation_apps.clear()
        property_stage = step.property_edit is not None
        if property_stage and (not valid_property_stage(step) or not current.a11y_complete):
            return ()
        navigation_app = ""
        semantic_op = step.operation in {"click", "fill", "check", "uncheck"}
        if (
            self._closed
            or not step.id
            or step.operation not in _OPERATIONS
            or (step.ref and step.target is not None)
            or (step.ref and not semantic_op)
            or (step.target is not None and not semantic_op)
            or current.image is None
            or not valid_click_parameters(step.operation, step.button, step.click_count)
        ):
            return ()
        if not isinstance(step.value, str) or len(step.value) > 2000 or "\x00" in step.value:
            return ()
        if type(step.navigation) is not bool:
            return ()
        if type(step.native_navigation_gestures) is not bool or (
            step.native_navigation_gestures
            and (not step.navigation or step.button != "left" or step.click_count != 1)
        ):
            return ()
        if step.navigation and (
            step.operation != "click"
            or not isinstance(step.label, str)
            or not step.label.strip()
            or step.ref
            or step.box is not None
            or step.value
            or (not step.postconditions and not property_stage)
            or not current.a11y_complete
        ):
            return ()
        # Keyboard and page actions also rely on the focus/context seen by the
        # planner. A screenshot-only desktop can change without changing its
        # textual fingerprint while the remote model is thinking.
        guarded = bool(step.preconditions and step.postconditions) and step.operation in {
            "type_text",
            "hotkey",
            "key_press",
            "press",
            "scroll",
            "back",
            "wait",
        }
        if step.target is None and not guarded and not _fresh(planned, current):
            return ()
        if not conditions_hold(step.preconditions, current):
            return ()
        typing_target = None
        if guarded and step.operation == "type_text":
            typing_target = self._typing_target(step, current)
            if typing_target is None:
                return ()
        targets: tuple[AgentControl | None, ...] = (None,)
        if semantic_op:
            if step.box is not None:
                return ()
            if step.target is not None:
                if step.navigation:
                    _, navigation_app = self._navigation_apps.setdefault(id(step), (step, ""))
                    targets = tuple(
                        control for control in navigation_scope_controls(
                            current, step.target, handoff_app=navigation_app
                        ) if executable_control(control, step.operation,
                                                button=step.button, click_count=step.click_count)
                    )
                else:
                    targets = executable_controls(current, step.operation, step.target,
                                                  button=step.button, click_count=step.click_count)
            elif step.ref:
                targets = tuple(
                    control
                    for control in current.controls
                    if control.ref == step.ref and executable_control(
                        control, step.operation, button=step.button, click_count=step.click_count,
                    )
                )
            else:
                return ()
            if any(native_toggle_control(control) for control in targets) and not current.a11y_complete:
                return ()
            if step.navigation:
                if len(targets) > 255:
                    return ()
            else:
                targets = targets[:64]
            if not targets and not property_stage:
                return ()
        visual = step.operation in {"visual_click", "visual_fill"}
        if visual:
            if not _valid_box(step.box) or not step.label.strip():
                return ()
        elif step.box is not None:
            return ()
        # This OSWorld computer_13 controller uses pyautogui.typewrite, which
        # cannot reliably enter non-ASCII characters. Reject before any click
        # or select-all instead of silently replacing a field with partial text.
        if step.operation in {"type_text", "visual_fill", "fill"} and (
            not step.value or not step.value.isascii()
        ):
            return ()
        if step.operation in {"visual_click", "back", "click", "check", "uncheck"} and step.value:
            return ()
        # Upstream PRESS/HOTKEY interpolate quoted key names without escaping.
        if step.operation == "key_press" and (step.value not in DESKTOP_KEYS or step.value in {"'", "\\"}):
            return ()
        if step.operation == "press" and step.value not in _BROWSER_KEYS:
            return ()
        if step.operation == "hotkey":
            keys = step.value.split("+")
            if (
                not 2 <= len(keys) <= 4
                or len(keys) != len(set(keys))
                or not all(key in DESKTOP_MODIFIERS for key in keys[:-1])
                or keys[-1] not in DESKTOP_KEYS
                or keys[-1] in DESKTOP_MODIFIERS
                or keys[-1] in {"'", "\\"}
            ):
                return ()
        if step.operation == "scroll" and step.value not in {"up", "down", "left", "right"}:
            return ()
        if step.operation == "wait" and step.value not in {"", "250", "500", "1000"}:
            return ()
        # Expand physical actions before registration so an over-budget step
        # cannot leave an executable prefix behind. Keep the logical step object
        # unchanged: receipt-authorized navigation scope is keyed by its identity.
        actions = []
        seen_actions = set()
        for target in targets:
            if step.native_navigation_gestures:
                modes = (
                    desktop_click_modes(target)
                    if step.full_legal_navigation_gestures is True and step.navigation
                    else native_navigation_click_modes(target)
                )
            else:
                modes = ((step.button, step.click_count),)
            for button, count in modes:
                key = (json.dumps(target.to_dict(), sort_keys=True) if target else None, button, count)
                if step.native_navigation_gestures and key in seen_actions:
                    continue
                seen_actions.add(key)
                actions.append((target, button, count, step))
        if property_stage:
            scope_controls = navigation_scope_controls(current, step.target, handoff_app=navigation_app)
            mutations = property_field_steps(step, current, scope_controls)
            if step.property_edit.prefer_verified_selection:
                # Keep the stronger effect-checked action when both choices send
                # exactly the same left click to a uniquely bound literal option.
                # Other gestures, focus clicks and Boolean navigation stay legal.
                selected_refs = {target.ref for target, mutation in mutations
                                 if mutation.operation == "click"
                                 and normalize_role(target.role) in {"cell", "listitem", "option"}
                                 and any(c.kind == "ready_to_select" for c in mutation.preconditions)}
                actions = [item for item in actions if not (
                    item[3] is step and item[0] is not None and item[0].ref in selected_refs
                    and item[1:3] == ("left", 1)
                )]
            if (self.satisfied_property_readonly_fast_path
                    and step.property_edit.prefer_verified_selection
                    and step.property_edit.operation == "set" and len(mutations) == 1):
                target, mutation = mutations[0]
                if (mutation.operation == "click" and not mutation.navigation
                        and target.label == step.property_edit.value
                        and target.app.strip() and target.window.strip()
                        and normalize_role(target.role) in {"cell", "listitem", "option"}
                        and mutation.target is not None
                        and mutation.target.name == step.property_edit.value
                        and mutation.target.match == "exact"
                        and {c.kind for c in mutation.preconditions} == {"ready_to_select"}
                        and {c.kind for c in mutation.postconditions}
                        == {"selected", "ready_to_select"}
                        and conditions_hold(mutation.preconditions, current)
                        and property_effect_holds(mutation, current, target)):
                    # Keep only the original locked, read-only field proof.
                    # Fresh preflight and later explicit commit/goal checks remain mandatory.
                    actions = []
            actions.extend((target, "left", 1, mutation)
                           for target, mutation in mutations)
        gesture_pool = "all_legal_v55"
        if (step.navigation and step.native_navigation_gestures
                and step.full_legal_navigation_gestures is True and len(actions) > 255):
            # Preserve the complete original pool when the expansion cannot fit.
            # Keep property mutations and observed native gestures; never register
            # an arbitrary prefix or drop a target to make extra gestures fit.
            actions = [item for item in actions if (
                item[3] is not step
                or item[1:3] in native_navigation_click_modes(item[0])
            )]
            gesture_pool = "native_affordances_fallback_v55"
        if step.navigation and len(actions) > 255:
            return ()
        offered = []
        for target, button, click_count, action_step in actions:
            confirmation = bool(property_stage and not action_step.navigation
                                and property_effect_holds(action_step, current, target))
            label = target.label if target else step.label or "desktop"
            detail = (
                f" ({len(action_step.value)} characters)"
                if action_step.operation in {"fill", "visual_fill", "type_text"}
                else (f" {action_step.value}" if action_step.value else "")
            )
            context = (
                (
                    f" [{target.ref}; {target.role}; {target.app}; {target.window}; "
                    f"{' / '.join(target.ancestors)}]"
                )
                if target
                else ""
            )
            goal = (f" toward property goal: {step.label}" if property_stage else
                    f" toward navigation goal: {step.label}" if step.navigation else "")
            click = (
                {"button": button, "click_count": click_count}
                if action_step.operation in {"click", "visual_click"} else {}
            )
            click_detail = (
                f" ({button} button, {click_count} click(s))"
                if click and (button != "left" or click_count != 1) else ""
            )
            operation_label = (
                "select option" if property_stage and step.property_edit.operation == "set"
                and not action_step.navigation and action_step.operation == "click"
                else action_step.operation
            )
            candidate = ActionCandidate(
                id=f"{step.id}:desktop:{uuid.uuid4().hex[:8]}",
                channel=Channel.GUI,
                capability=f"agent.osworld.{action_step.operation}",
                description=(
                    f"{'confirm already-satisfied ' if confirmation else ''}"
                    f"{operation_label} {label}{detail}{context}{click_detail}{goal} via OSWorld input"
                ),
                arguments={
                    "operation": action_step.operation,
                    "value": action_step.value,
                    "box": target.box if target else step.box,
                    "route": "osworld-a11y" if target else "osworld-computer-13",
                    "fingerprint": current.fingerprint(),
                    "bound_control": target.to_dict() if target else None,
                    "target": asdict(action_step.target) if action_step.target else None,
                    **click,
                    **({"navigation": True, "navigation_app": navigation_app}
                       if action_step.navigation else {}),
                    **({"navigation_gesture_pool": gesture_pool}
                       if action_step.navigation and step.full_legal_navigation_gestures is True else {}),
                    **({"property_stage": True, "property_phase": "navigation" if action_step.navigation
                        else "confirm" if confirmation else "edit"} if property_stage else {}),
                },
                risk=Risk.READ_ONLY if confirmation else Risk.LOCAL_WRITE,
                verifier="agent.effect",
                intent=step.id,
            )
            self._issued[candidate.id] = _Binding(
                candidate.to_dict(), action_step, current, target, typing_target, planned, navigation_app,
                button, click_count, step if property_stage else None, confirmation,
            )
            offered.append(candidate)
        return tuple(offered)

    def bound_property_step(self, candidate: ActionCandidate, receipt: ActionReceipt) -> AgentStep | None:
        """Return the actual typed field step for receipt-bound reporting only."""
        known = self._property_receipts.get(receipt.receipt_id)
        if self._closed or known is None or known[0] != candidate.to_dict() or known[1] != receipt.to_dict():
            return None
        if (receipt.receipt_id in self._readonly_property_receipts
                and not self._readonly_property_receipt_matches(candidate, receipt)):
            return None
        return known[2]

    def _readonly_property_receipt_matches(self, candidate: ActionCandidate, receipt: ActionReceipt) -> bool:
        """Check provenance and payload integrity, independently of effect success."""
        known = self._verified.get(receipt.receipt_id)
        bound = self._property_receipts.get(receipt.receipt_id)
        return bool(
            not self._closed and known is not None and bound is not None
            and receipt is known[1]
            and candidate.to_dict() == known[0] == bound[0] and receipt.to_dict() == bound[1]
        )

    def _navigation_details(
        self, binding: _Binding, before: AgentSnapshot, after: AgentSnapshot
    ) -> dict:
        """Propose scope evidence only on the execution/recovery receipt path."""
        app = binding.navigation_app
        known = self._navigation_apps.get(id(binding.step))
        if self._navigation_planned is binding.planned and known is not None and known[0] is binding.step:
            app = known[1] or app
            if not app and binding.button == "left" and binding.click_count == 1:
                app = navigation_handoff_app(before, after, binding.step.target, binding.target)
        progressed, before_scope, after_scope = _navigation_progress(
            binding.step, before, after, before_app=binding.navigation_app, after_app=app
        )
        return {
            "navigation": True,
            "before_scope": before_scope,
            "after_scope": after_scope,
            "accessibility_progress": progressed,
            "navigation_app": app,
            "navigation_handoff": bool(app and not binding.navigation_app),
        }

    def _commit_navigation_handoff(self, binding: _Binding, details: dict) -> None:
        app = details.get("navigation_app")
        known = self._navigation_apps.get(id(binding.step))
        if (
            details.get("accessibility_progress") and isinstance(app, str) and app
            and self._navigation_planned is binding.planned
            and known is not None and known[0] is binding.step and not known[1]
        ):
            self._navigation_apps[id(binding.step)] = (binding.step, app)

    @staticmethod
    def _application_wait_details(binding: _Binding, after: AgentSnapshot, facts: list[dict]) -> dict:
        """Identify a missing cross-app effect, never authorize another input.

        This evidence is emitted only from a successful typed action or its
        receipt-bound observations. A newly active application is evidence for
        repairing an exact predicate, not proof that the requested goal is done.
        Navigation retains its separate, existing handoff authorization.
        """
        step, before, launcher = binding.step, binding.snapshot, binding.target
        if (
            step.operation != "click" or binding.button != "left" or binding.click_count != 1
            or step.recovery != "observe" or launcher is None or launcher.window
            or launcher.role != "button" or not launcher.app.strip() or not launcher.label.strip()
            or not executable_control(launcher, "click") or not before.a11y_complete
        ):
            return {}
        missing = [fact for fact in facts if not fact["satisfied"]]
        if not missing or any(
            fact.get("outcome") != "unknown" or fact.get("reason") != "target_missing"
            or not isinstance(fact.get("target"), dict)
            or not fact["target"].get("app", "").strip()
            for fact in missing
        ):
            return {}
        expected = {" ".join(fact["target"]["app"].split()).casefold() for fact in missing}
        if len(expected) != 1:
            return {}
        expected_app = next(iter(expected))
        if expected_app in launcher.app.casefold() or any(
            expected_app in control.app.casefold() for control in before.controls
        ):
            return {}
        details = {"eligible": True, "reason": "awaiting_application", "expected_app": expected_app}
        scope = AgentTarget(app=launcher.app)
        app = navigation_handoff_app(before, after, scope, launcher)
        if app:
            windows = [control for control in navigation_scope_controls(after, scope, handoff_app=app)
                       if control.role in {"frame", "window", "dialog"}]
            details.update(
                eligible=False,
                reason="application_appeared_predicate_unconfirmed"
                if expected_app in app.casefold() else "unexpected_application_appeared",
                observed_app=app,
                observed_window=windows[0].window,
                observed_role=windows[0].role,
            )
        elif after.a11y_complete and any(
            control.active and control.app.strip()
            and not any(control.app.casefold() == old.app.casefold() for old in before.controls)
            for control in after.controls
        ):
            # Multiple active apps/windows, or no trustworthy executable window,
            # cannot establish a unique launch result. Ask the planner to inspect.
            details.update(eligible=False, reason="application_appearance_ambiguous")
        return {"application_wait": details}

    def _step(self, kind: str, parameters: dict, *, image_only: bool = False) -> None:
        from .osworld_step_observation import (
            IMAGE_ONLY_INFO_KEY,
            IMAGE_ONLY_VERSION,
            native_image_only_step,
            validate_intermediate_action,
        )

        action = {"action_type": kind, "parameters": parameters}
        if image_only:
            if not self.image_only_fill_intermediates:
                raise ValueError("image-only fill observations are disabled")
            validate_intermediate_action(action)
        self._post_snapshot = None
        self._stale_rejection = self._stale_decision_snapshot = None
        self._recovery_source = self._recovery_context = self._recovery_confirmation = None
        # Observe/focus checks and preceding inputs can block. Recheck at the
        # actual submission boundary; already submitted remote input cannot be
        # withdrawn and remains included in environment_steps.
        if self._execution_deadline is not None and time.perf_counter() >= self._execution_deadline:
            raise _InputDeadlineExpired()
        self.environment_steps += 1
        with self._measure_observation_phase(
                "image_only_step" if image_only else "full_step", "acquisition"):
            if image_only:
                if getattr(self.env, "supports_image_only_steps", None) is True:
                    response = self.env.step_image_only(action, pause=self.pause)
                else:
                    response = native_image_only_step(self.env, action, pause=self.pause)
            else:
                response = self.env.step(action, pause=self.pause)
            received_at = time.perf_counter()
        with self._measure_observation_phase(
                "image_only_step" if image_only else "full_step", "snapshot_build"):
            if not isinstance(response, tuple) or len(response) != 4:
                raise RuntimeError("OSWorld returned an invalid step response")
            observation, _reward, _done, info = response
            if isinstance(info, dict) and (info.get("error") or info.get("success") is False):
                raise RuntimeError("OSWorld reported a failed step")
            if not image_only and isinstance(info, dict) and IMAGE_ONLY_INFO_KEY in info:
                raise RuntimeError("OSWorld returned an intermediate observation for a full step")
            if image_only:
                if (not isinstance(info, dict) or info.get(IMAGE_ONLY_INFO_KEY) != IMAGE_ONLY_VERSION
                        or not isinstance(observation, dict)
                        or observation.get("accessibility_tree") is not None):
                    raise RuntimeError("OSWorld did not confirm an image-only fill observation")
                # A real screenshot is retained remotely, but this partial response
                # is never parsed as a planning/effect/focus snapshot or cached.
                screenshot_image(observation.get("screenshot"))
                self._action_observation = None
                self._action_observation_received_at = None
                self.intermediate_observation_metrics["image_only_steps"] += 1
                return
            # Discard reward and done: they are not effect or task-success evidence.
            self._action_observation = self._snapshot(observation)
            self._action_observation_received_at = received_at

    @staticmethod
    def _target_after(
        target: AgentControl, snapshot: AgentSnapshot, *, require_same_box: bool = False
    ) -> AgentControl | None:
        def identity(control):
            context = (control.identity(), control.app, control.window, control.ancestors)
            return (*context, control.box) if require_same_box else context

        matches = [control for control in snapshot.controls if identity(control) == identity(target)]
        return matches[0] if len(matches) == 1 else None

    def _pending_numeric_edit(self, binding: _Binding, after: AgentSnapshot) -> dict:
        if binding.step.operation != "fill" or binding.target is None or not after.a11y_complete:
            return {}
        target = self._target_after(binding.target, after)
        if target is None:
            return {}
        # Uniqueness precedes value matching. A second field cannot disappear
        # from the query just because it has another value, state or geometry.
        matches = (
            matching_controls(after, binding.step.target)
            if binding.step.target is not None else (target,)
        )
        fields = tuple(control for control in matches if control.editable
                       and normalize_role(control.role) in {"textbox", "spinbutton"})
        if fields != (target,):
            return {}
        match = pending_numeric_buffer_match(target, binding.step.value)
        if match is None:
            return {}
        return {"pending_numeric_edit": {
            "reason": "edit_buffer_matches_but_committed_value_differs",
            "target": {"name": target.label, "role": target.role,
                       "app": target.app, "window": target.window},
            "control_ref": target.ref,
            "committed_value": target.value,
            "edit_buffer": target.edit_buffer,
            "buffer_match": match,
            "focused": True,
            "task_success": False,
        }}

    def _numeric_receipt_matches(self, candidate: ActionCandidate, receipt: ActionReceipt) -> bool:
        source = self._numeric_edit_source
        return bool(source is not None and not self._closed and receipt is source[1]
                    and receipt.success and candidate.to_dict() == source[0]
                    and receipt.to_dict() == source[2] and self.environment_steps == source[3])

    def pending_numeric_replan(self, candidate: ActionCandidate, receipt: ActionReceipt) -> bool:
        """Recognize an unverified original typing buffer, without asserting commitment."""
        cached = self._numeric_planning_snapshot
        known = self._verified.get(receipt.receipt_id)
        return bool(
            cached is not None and cached[0] is self._numeric_edit_source
            and cached[3] == self.fresh_observe_calls and self._numeric_receipt_matches(candidate, receipt)
            and cached[1].a11y_complete and known is not None and known[1] is receipt
            and not self.verify(candidate, receipt).passed
            and known[2].details.get("pending_numeric_edit", {}).get("reason")
            == "edit_buffer_matches_but_committed_value_differs"
        )

    def take_pending_numeric_planning_snapshot(
        self, candidate: ActionCandidate, receipt: ActionReceipt,
    ) -> tuple[AgentSnapshot, float] | None:
        """Consume one original unverified typing frame for a single VLM recovery.

        It is never a verified effect, a decision snapshot or goal evidence.
        New observations, executions and the normal one-second age expire it.
        """
        cached = self._numeric_planning_snapshot
        eligible = self.pending_numeric_replan(candidate, receipt)
        self._numeric_planning_snapshot = None
        if cached is None or not eligible or not 0 <= time.perf_counter() - cached[2] <= 1.0:
            return None
        # Choosing ordinary model recovery abandons the original key policy;
        # this planning-only frame cannot later become its input context.
        self._numeric_finalize_pending = self._numeric_finalize_snapshot = None
        self._numeric_finalize_context = None
        return cached[1], cached[2]

    def _numeric_finalize_source_matches(self, source: _NumericFinalizePending) -> bool:
        known = self._verified.get(source.receipt.receipt_id)
        return bool(
            not self._closed and source.environment_steps == self.environment_steps
            and source.receipt.success and source.receipt.to_dict() == source.receipt_data
            and source.candidate.to_dict() == source.candidate_data
            and source.stage.to_dict() == source.stage_data
            and known is not None and known[0] == source.candidate_data
            and known[1] is source.receipt and not known[2].passed
            and known[2].details.get("pending_numeric_edit", {}).get("reason")
            == "edit_buffer_matches_but_committed_value_differs"
            and source.key in {"enter", "tab"}
            and getattr(source.stage.property_edit, "numeric_finalize_key", "") == source.key
        )

    def _numeric_finalize_control(
        self, source: _NumericFinalizePending, snapshot: AgentSnapshot, *, pending: bool,
    ) -> AgentControl | None:
        if not snapshot.a11y_complete or not source.target.app.strip() or not source.target.window.strip():
            return None
        target = self._target_after(source.target, snapshot, require_same_box=True)
        if target is None or target.disabled or not target.editable:
            return None
        fields = tuple(control for control in matching_controls(snapshot, source.field_step.target)
                       if control.editable and normalize_role(control.role) in {"textbox", "spinbutton"})
        if fields != (target,):
            return None
        if pending and (
            not target.focused or target.edit_buffer != source.target.edit_buffer
            or pending_numeric_buffer_match(target, source.field_step.value) is None
        ):
            return None
        return target

    def declared_numeric_finalize_pending(
        self, candidate: ActionCandidate, receipt: ActionReceipt, stage: AgentStep,
    ) -> bool:
        """Only an original explicit key policy can continue this unverified fill."""
        source = self._numeric_finalize_pending
        return bool(
            source is not None and source.candidate is candidate and source.receipt is receipt
            and source.stage is stage and self._numeric_finalize_source_matches(source)
        )

    def take_declared_numeric_finalize_snapshot(
        self, candidate: ActionCandidate, receipt: ActionReceipt, stage: AgentStep,
    ) -> tuple[AgentSnapshot, float] | None:
        """Consume one unverified response solely for its declared single key.

        This independent slot supplies no planning, ordinary decision, goal or
        acceptance evidence. Every key still performs its ordinary fresh input
        preflight. A new observation invalidates this slot, not source history.
        """
        source, self._numeric_finalize_snapshot = self._numeric_finalize_snapshot, None
        if (
            source is None or source is not self._numeric_finalize_pending
            or not self.declared_numeric_finalize_pending(candidate, receipt, stage)
            or source.observation_count != self.fresh_observe_calls
            or not 0 <= time.perf_counter() - source.received_at <= 1.0
            or self._numeric_finalize_control(source, source.snapshot, pending=True) is None
        ):
            return None
        self._numeric_finalize_context = source
        self._numeric_planning_snapshot = None
        return source.snapshot, source.received_at

    def compile_declared_numeric_finalize(
        self, candidate: ActionCandidate, receipt: ActionReceipt, stage: AgentStep,
        planned: AgentSnapshot, current: AgentSnapshot,
    ) -> tuple[AgentStep, tuple[ActionCandidate, ...]] | None:
        """Offer exactly the declared key, never a refill or a confirmation."""
        source = self._numeric_finalize_pending
        self._numeric_finalize_snapshot = None
        context, self._numeric_finalize_context = self._numeric_finalize_context, None
        if (
            source is None or not self.declared_numeric_finalize_pending(candidate, receipt, stage)
            or (current is source.snapshot and (
                context is not source or source.observation_count != self.fresh_observe_calls
                or not 0 <= time.perf_counter() - source.received_at <= 1.0
            ))
            or (current is not source.snapshot and current is not self._last_observed_snapshot)
            or self._numeric_finalize_control(source, current, pending=True) is None
        ):
            self._numeric_finalize_pending = None
            return None
        key_step = AgentStep(
            stage.id, "key_press", value=source.key, label="Finalize the explicitly declared numeric edit",
            preconditions=(AgentCondition("focused", source.field_step.target),),
            postconditions=source.field_step.postconditions, recovery="observe",
        )
        choices = self.compile(key_step, planned, current)
        self._numeric_finalize_pending = None
        self._numeric_planning_snapshot = None
        if len(choices) != 1:
            return None
        original = choices[0]
        choice = replace(original, arguments={
            **original.arguments, "declared_numeric_finalize": True,
            "source_receipt_id": source.receipt.receipt_id,
        })
        binding = self._issued.pop(original.id)
        self._issued[choice.id] = replace(binding, candidate=choice.to_dict(), numeric_finalize=source)
        self._numeric_finalize_issued = source
        return key_step, (choice,)

    def _click(self, box: tuple, snapshot: AgentSnapshot, *, button="left", click_count=1,
               image_only: bool = False) -> None:
        if not valid_click_parameters("click", button, click_count):
            raise ValueError("unsupported desktop click parameters")
        width, height = snapshot.image.region[2:]
        x = min(width - 1, int((box[0] + box[2]) * width / 2000))
        y = min(height - 1, int((box[1] + box[3]) * height / 2000))
        self._step("CLICK", {"x": x, "y": y, "button": button, "num_clicks": click_count},
                   image_only=image_only)

    def _focused_target(self, target: AgentControl, *, diagnostic_stage="unspecified") -> AgentControl:
        current = self._target_after(target, self.observe(), require_same_box=True)
        if current is None or current.disabled or not current.editable or not current.focused:
            error = _TextFocusGuardRejected(diagnostic_stage, self._last_observed_snapshot, current)
            error._focus_guard_owner = self  # Private local identity, never copied to output.
            raise error
        return current

    @staticmethod
    def _typing_target(step: AgentStep, snapshot: AgentSnapshot) -> AgentControl | None:
        """Resolve guarded typing to one observed editable text focus owner."""
        target = None
        for condition in step.preconditions:
            if condition.kind != "focused":
                continue
            if not conditions_hold((condition,), snapshot):
                return None
            matches = tuple(
                control
                for control in matching_controls(snapshot, condition.target)
                if control.focused
                and not control.disabled
                and control.editable
                and control.tag in {"input", "textarea", "contenteditable"}
                and normalize_role(control.role) in {"textbox", "spinbutton"}
            )
            if len(matches) != 1 or (target is not None and target != matches[0]):
                return None
            target = matches[0]
        return target

    def _perform(
        self, step: AgentStep, snapshot: AgentSnapshot, target: AgentControl | None,
        *, button: str, click_count: int,
    ) -> None:
        op = step.operation
        if target is not None:
            if op in {"check", "uncheck"}:
                if observed_checked_state(target) is None:
                    raise ValueError("desktop checked state is unknown before input")
                if checked_matches(target, op == "check", snapshot):
                    return
            partial = self.image_only_fill_intermediates and op == "fill"
            self._click(target.box, snapshot, button=button, click_count=click_count, image_only=partial)
            if op == "fill":
                self._focused_target(target, diagnostic_stage="after_focus_click")
                self._step("HOTKEY", {"keys": ["ctrl", "a"]}, image_only=partial)
                self._focused_target(target, diagnostic_stage="after_select_all")
                self._step("TYPING", {"text": step.value})
            return
        if op in {"visual_click", "visual_fill"}:
            partial = self.image_only_fill_intermediates and op == "visual_fill"
            self._click(step.box, snapshot, button=button, click_count=click_count, image_only=partial)
            if op == "visual_fill":
                self._step("HOTKEY", {"keys": ["ctrl", "a"]}, image_only=partial)
                self._step("TYPING", {"text": step.value})
        elif op == "type_text":
            self._step("TYPING", {"text": step.value})
        elif op == "hotkey":
            self._step("HOTKEY", {"keys": step.value.split("+")})
        elif op in {"key_press", "press"}:
            self._step("PRESS", {"key": _BROWSER_KEYS[step.value] if op == "press" else step.value})
        elif op == "scroll":
            dx, dy = {"up": (0, 3), "down": (0, -3), "left": (-3, 0), "right": (3, 0)}[step.value]
            self._step("SCROLL", {"dx": dx, "dy": dy})
        elif op == "back":
            self._step("HOTKEY", {"keys": ["alt", "left"]})
        elif op == "wait":
            time.sleep(int(step.value or "250") / 1000)

    def execute_with_deadline(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str, *, deadline: float
    ) -> ActionReceipt:
        """Limit new local input submissions by an absolute perf_counter deadline.

        This does not interrupt synchronous transports or cancel remote input.
        The context is private to this execution and restored even on failure.
        """
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("desktop input deadline must be finite")
        previous = self._execution_deadline
        self._execution_deadline = deadline
        try:
            return self.execute(candidate, observation_id, decision_id)
        finally:
            self._execution_deadline = previous

    def execute(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        self._input_evidence = None
        finalize_issued, self._numeric_finalize_issued = self._numeric_finalize_issued, None
        self._numeric_finalize_pending = self._numeric_finalize_snapshot = None
        self._numeric_finalize_context = None
        self._numeric_edit_source = None
        self._numeric_planning_snapshot = None
        self._property_navigation_active = None
        self._property_navigation_post = None
        self._post_snapshot = None
        self._stale_rejection = self._stale_decision_snapshot = None
        self._recovery_source = self._recovery_context = self._recovery_confirmation = None
        self._action_observation = None
        self._action_observation_received_at = None
        started = time.time()
        output: dict = {}
        verified = Verification(False, "agent.effect", {"task_success": False})
        failure = None
        initial_steps = self.environment_steps
        binding = None
        post_snapshot = None
        post_received_at = None
        readonly_confirmation = False
        try:
            preflight = time.perf_counter()
            binding = self._issued.get(candidate.id)
            if binding is None or candidate.to_dict() != binding.candidate:
                output["failure_code"] = "invalid_candidate"
                raise ValueError("unknown or altered desktop candidate")
            if binding.numeric_finalize is not None and (
                binding.numeric_finalize is not finalize_issued
                or not self._numeric_finalize_source_matches(binding.numeric_finalize)
                or binding.step.operation != "key_press"
                or binding.step.value != binding.numeric_finalize.key
            ):
                output["failure_code"] = "numeric_finalize_source_changed"
                raise ValueError("original numeric policy source changed")
            if binding.confirmation is True:
                if (candidate.risk != Risk.READ_ONLY
                        or candidate.arguments.get("property_phase") != "confirm"
                        or binding.property_step is None or not valid_property_stage(binding.property_step)
                        or binding.step.navigation or binding.target is None):
                    output["failure_code"] = "invalid_readonly_confirmation"
                    raise ValueError("invalid read-only property binding")
                fields = property_field_steps(
                    binding.property_step, binding.snapshot,
                    navigation_scope_controls(binding.snapshot, binding.property_step.target,
                                              handoff_app=binding.navigation_app),
                )
                if ((binding.target, binding.step) not in fields
                        or candidate.arguments.get("bound_control") != binding.target.to_dict()
                        or candidate.arguments.get("target") != asdict(binding.step.target)):
                    output["failure_code"] = "invalid_readonly_confirmation"
                    raise ValueError("altered read-only property binding")
                readonly_confirmation = True
            elif binding.confirmation is not False or candidate.risk == Risk.READ_ONLY:
                output["failure_code"] = "invalid_readonly_confirmation"
                raise ValueError("invalid read-only property binding")
            before = self.observe()
            if readonly_confirmation:
                output["observation_source"] = "fresh_preflight_read_only"
            if not _fresh(binding.snapshot, before):
                output["failure_code"] = "stale_observation"
                raise ValueError("stale desktop screenshot or accessibility context")
            if not conditions_hold(binding.step.preconditions, before):
                output["failure_code"] = "preconditions_unmet"
                raise ValueError("desktop action preconditions no longer hold")
            if (binding.numeric_finalize is not None and self._numeric_finalize_control(
                    binding.numeric_finalize, before, pending=True) is None):
                output["failure_code"] = "numeric_finalize_buffer_changed"
                raise ValueError("original numeric field or edit buffer changed before key input")
            if (
                binding.typing_target is not None
                and self._typing_target(binding.step, before) != binding.typing_target
            ):
                output["failure_code"] = "focus_changed"
                raise ValueError("bound desktop text input focus changed before typing")
            if binding.target is not None:
                fresh_target = next(
                    (control for control in before.controls if control.ref == binding.target.ref), None
                )
                if fresh_target != binding.target or not executable_control(
                    fresh_target, binding.step.operation,
                    button=binding.button, click_count=binding.click_count,
                ):
                    output["failure_code"] = "control_changed"
                    raise ValueError("bound desktop control changed before input")
                if native_toggle_control(fresh_target) and not before.a11y_complete:
                    output["failure_code"] = "incomplete_observation"
                    raise ValueError("native toggle observation is incomplete before input")
            action = time.perf_counter()
            self._action_observation = None
            property_confirmation = bool(
                binding.property_step is not None and not binding.step.navigation
                and property_effect_holds(binding.step, before, binding.target)
            )
            if readonly_confirmation and not property_confirmation:
                output["failure_code"] = "readonly_confirmation_changed"
                raise ValueError("read-only property effect no longer holds")
            if not property_confirmation:
                self._perform(binding.step, before, binding.target,
                              button=binding.button, click_count=binding.click_count)
            checking = time.perf_counter()
            # DesktopEnv.step already returns the actual post-input screenshot
            # and accessibility tree. Reuse only this execution's final typed
            # action response; waits and idempotent no-ops still observe afresh.
            # A locked read-only property uses its new preflight observation;
            # it never manufactures an action response or enters that cache.
            after = (before if readonly_confirmation else
                     self._action_observation if self._action_observation is not None else self.observe())
            post_snapshot = self._action_observation
            post_received_at = self._action_observation_received_at
            effects = []
            if before.scope_key() != after.scope_key():
                effects.append("accessibility")
            if (
                before.image.region != after.image.region
                or changed_fraction(before.image.sample, after.image.sample) > 0.001
            ):
                effects.append("image")
            if (
                binding.step.operation == "wait"
                and checking - action >= int(binding.step.value or "250") / 1000
            ):
                effects.append("time_elapsed")
            current_target = self._target_after(binding.target, after) if binding.target else None
            direct_condition = True
            if binding.step.operation == "fill":
                direct_condition = current_target is not None and value_matches(
                    current_target, binding.step.value,
                )
                if direct_condition:
                    effects.append("value")
            elif binding.step.operation in {"check", "uncheck"}:
                direct_condition = current_target is not None and checked_matches(
                    current_target, binding.step.operation == "check", after,
                )
                if native_toggle_control(binding.target):
                    direct_condition = direct_condition and (
                        current_target is not None and native_toggle_control(current_target)
                    )
                if direct_condition:
                    effects.append("checked")
            postcondition_evidence = _postcondition_evidence(binding.step.postconditions, after)
            postconditions_met = all(fact["satisfied"] for fact in postcondition_evidence)
            if binding.property_step is not None and binding.step.navigation:
                postconditions_met = False  # A changed route is never an applied property.
            elif binding.property_step is not None:
                satisfied = property_effect_holds(binding.step, after, current_target)
                direct_condition = direct_condition and satisfied
                postconditions_met = postconditions_met and satisfied
            if binding.step.postconditions and postconditions_met:
                effects.append("postconditions")
            navigation = {}
            passed = bool(effects) and direct_condition and postconditions_met
            if binding.numeric_finalize is not None:
                source = binding.numeric_finalize
                field = self._numeric_finalize_control(source, after, pending=False)
                committed = field is not None and property_effect_holds(source.field_step, after, field)
                passed = passed and committed
                postconditions_met = postconditions_met and committed
            if binding.step.navigation:
                navigation = self._navigation_details(binding, before, after)
                passed = navigation["accessibility_progress"] or postconditions_met
            if readonly_confirmation and not passed:
                output["failure_code"] = "readonly_confirmation_unconfirmed"
                raise ValueError("read-only property verification failed")
            application_wait = (
                self._application_wait_details(binding, after, postcondition_evidence)
                if self.environment_steps > initial_steps else {}
            )
            pending_numeric = (
                self._pending_numeric_edit(binding, after)
                if self.environment_steps > initial_steps else {}
            )
            verified = Verification(
                passed,
                "agent.effect",
                {
                    "effects": effects,
                    "operation": binding.step.operation,
                    "task_success": False,
                    "postconditions_met": postconditions_met,
                    "postcondition_evidence": postcondition_evidence,
                    **navigation,
                    **application_wait,
                    **pending_numeric,
                    **({"declared_numeric_finalize": True,
                        "source_receipt_id": binding.numeric_finalize.receipt.receipt_id}
                       if binding.numeric_finalize is not None else {}),
                    **({"property_stage": True, "property_confirmation": property_confirmation}
                       if binding.property_step is not None else {}),
                    **({"observation_source": "fresh_preflight_read_only"} if readonly_confirmation else {}),
                },
            )
            output = {
                "operation": binding.step.operation,
                **({"button": binding.button, "click_count": binding.click_count}
                   if binding.step.operation in {"click", "visual_click"} else {}),
                "route": candidate.arguments["route"],
                "effects": effects,
                "task_success": False,
                "postconditions_met": postconditions_met,
                "postcondition_evidence": postcondition_evidence,
                **navigation,
                **application_wait,
                **pending_numeric,
                **({"declared_numeric_finalize": True,
                    "source_receipt_id": binding.numeric_finalize.receipt.receipt_id}
                   if binding.numeric_finalize is not None else {}),
                **({"property_stage": True, "property_confirmation": property_confirmation}
                   if binding.property_step is not None else {}),
                **({"observation_source": "fresh_preflight_read_only"} if readonly_confirmation else {}),
                "before_fingerprint": before.fingerprint(),
                "after_fingerprint": after.fingerprint(),
                "timings_ms": {
                    "preflight": (action - preflight) * 1000,
                    "action": (checking - action) * 1000,
                    "effect_check": (time.perf_counter() - checking) * 1000,
                },
            }
            if current_target is not None and binding.step.operation in {"fill", "check", "uncheck"}:
                output["postcondition"] = {
                    "identity": list(current_target.identity()),
                    "value": current_target.value,
                    "checked": current_target.checked,
                }
        except Exception as exc:
            # Remote exceptions can embed request bodies. Never trace their text.
            failure = f"OSWorld action failed ({type(exc).__name__})"
            if type(exc) is _TextFocusGuardRejected and getattr(exc, "_focus_guard_owner", None) is self:
                # Preserve the old ValueError receipt label, refusal and counts.
                failure = "OSWorld action failed (ValueError)"
                diagnostic = _bounded_text_focus_diagnostic(exc)
                if diagnostic is not None:
                    output["failure_code"] = "text_focus_guard_rejected"
                    output["text_focus_guard"] = diagnostic
            if isinstance(exc, _InputDeadlineExpired):
                output["failure_code"] = "input_deadline_exceeded"
        finally:
            self._action_observation = None
            self._action_observation_received_at = None
            self._issued.clear()
        output["environment_steps"] = self.environment_steps - initial_steps
        receipt = ActionReceipt(
            observation_id,
            decision_id,
            candidate.id,
            candidate.channel,
            candidate.capability,
            failure is None,
            started,
            time.time(),
            output,
            failure,
        )
        self._verified[receipt.receipt_id] = (candidate.to_dict(), receipt, verified)
        if receipt.success and binding is not None and self.environment_steps > initial_steps:
            self._remember_input_observation(
                binding, candidate, receipt, before, after,
                post_received_at if post_received_at is not None else self._last_observed_received_at,
                verified,
            )
        if (not receipt.success and output.get("failure_code") == "stale_observation"
                and self.environment_steps == initial_steps and self._last_observed_snapshot is before
                and self._last_observed_received_at is not None):
            self._stale_rejection = _StaleRejection(
                candidate.to_dict(), receipt, receipt.to_dict(),
                self.environment_steps, self.fresh_observe_calls, before, self._last_observed_received_at,
            )
        if binding is not None and binding.property_step is not None:
            self._property_receipts[receipt.receipt_id] = (
                candidate.to_dict(), receipt.to_dict(), binding.step,
            )
            if binding.confirmation is True:
                self._readonly_property_receipts.add(receipt.receipt_id)
        if (binding is not None and binding.numeric_finalize is not None
                and receipt.success and verified.passed and self.environment_steps == initial_steps + 1):
            self._property_receipts[receipt.receipt_id] = (
                candidate.to_dict(), receipt.to_dict(), binding.numeric_finalize.field_step,
            )
        if (receipt.success and binding is not None and binding.step.operation == "fill"
                and self.environment_steps > initial_steps):
            self._numeric_edit_source = (
                candidate.to_dict(), receipt, receipt.to_dict(), self.environment_steps,
            )
            if (not verified.passed and post_snapshot is not None and post_snapshot.a11y_complete
                    and post_received_at is not None and self.environment_steps == initial_steps + 3
                    and verified.details.get("pending_numeric_edit", {}).get("reason")
                    == "edit_buffer_matches_but_committed_value_differs"):
                self._numeric_planning_snapshot = (
                    self._numeric_edit_source, post_snapshot, post_received_at, self.fresh_observe_calls,
                )
                stage = binding.property_step
                key = getattr(stage.property_edit, "numeric_finalize_key", "") if stage is not None else ""
                target = (self._target_after(binding.target, post_snapshot)
                          if binding.target is not None else None)
                if (key in {"enter", "tab"} and valid_property_stage(stage) and target is not None
                        and stage.recovery == "observe" and target.app.strip() and target.window.strip()):
                    source = _NumericFinalizePending(
                        candidate, candidate.to_dict(), receipt, receipt.to_dict(), stage, stage.to_dict(),
                        binding.step, target, post_snapshot, post_received_at, self.environment_steps,
                        self.fresh_observe_calls, key,
                    )
                    if self._numeric_finalize_control(source, post_snapshot, pending=True) is not None:
                        self._numeric_finalize_pending = self._numeric_finalize_snapshot = source
        if (
            receipt.success and binding is not None and binding.step.operation == "click"
            and self.environment_steps > initial_steps and not verified.passed
        ):
            self._recovery_source = _RecoverySource(
                candidate.to_dict(), receipt, receipt.to_dict(), before, self.environment_steps,
            )
            self._update_recovery_context(binding, candidate, receipt, after, verified)
        if (
            receipt.success and verified.passed and post_snapshot is not None
            and post_received_at is not None and self.environment_steps > initial_steps
            and not verified.details.get("application_wait", {}).get("eligible")
        ):
            self._post_snapshot = (
                candidate.to_dict(), receipt, receipt.to_dict(), post_snapshot, post_received_at,
                self.environment_steps,
            )
        if receipt.success and binding is not None and binding.step.navigation:
            self._commit_navigation_handoff(binding, verified.details)
        if (
            receipt.success and not verified.passed and binding is not None
            and binding.property_step is not None and valid_property_stage(binding.property_step)
            and binding.property_step.recovery == binding.step.recovery == "observe"
            and binding.step.navigation and binding.step.operation == "click"
            and not binding.step.postconditions and binding.target is not None
            and binding.property_step.target.app.strip() and not binding.navigation_app
            and candidate.risk != Risk.READ_ONLY and self.environment_steps == initial_steps + 1
            and before.a11y_complete and not self._property_navigation_affordances(binding, after)
        ):
            scope = navigation_scope_controls(before, binding.property_step.target)
            owners = {control.app.strip().casefold() for control in scope}
            if owners == {binding.target.app.strip().casefold()} and "" not in owners:
                self._property_navigation_pending[receipt.receipt_id] = _PropertyNavigationPending(
                    binding, candidate.to_dict(), receipt, receipt.to_dict(), self.environment_steps,
                    self.fresh_observe_calls, binding.target.app.strip().casefold(),
                )
                self._property_navigation_active = receipt.receipt_id
        if (
            receipt.success
            and binding is not None
            and binding.step.postconditions
            and binding.step.recovery == "observe"
            and (
                binding.target is not None
                or (binding.step.preconditions and not binding.step.operation.startswith("visual_"))
            )
        ):
            self._pending[receipt.receipt_id] = (binding, 0)
            if verified.details.get("application_wait", {}).get("eligible"):
                self._launch_wait_until[receipt.receipt_id] = time.perf_counter() + 60
        return receipt

    def verify(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        known = self._verified.get(receipt.receipt_id)
        if not receipt.success or known is None or known[0] != candidate.to_dict() or known[1] != receipt:
            return Verification(
                False, "agent.effect", {"reason": "unknown or failed receipt", "task_success": False}
            )
        pending_navigation = self._property_navigation_pending.get(receipt.receipt_id)
        if pending_navigation is not None and not self._property_navigation_receipt_matches(
            pending_navigation, candidate, receipt,
        ):
            return Verification(False, "agent.effect", {
                "reason": "unbound property navigation receipt", "task_success": False,
            })
        if (receipt.receipt_id in self._readonly_property_receipts
                and not self._readonly_property_receipt_matches(candidate, receipt)):
            return Verification(
                False, "agent.effect", {"reason": "unbound read-only property receipt", "task_success": False}
            )
        result = known[2]
        if "pending_numeric_edit" in result.details and not self._numeric_receipt_matches(candidate, receipt):
            return Verification(result.passed, result.verifier, {
                key: value for key, value in result.details.items() if key != "pending_numeric_edit"
            })
        return result

    def verify_pending(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        """Observe a receipt-bound effect; an eligible launcher gets a bounded lease.

        The 60-second deadline bounds observation starts, not synchronous
        transport completion. The runner separately clamps episode/plan time.
        """
        self._post_snapshot = None
        self._stale_rejection = self._stale_decision_snapshot = None
        self._recovery_context = self._recovery_confirmation = None
        if receipt.receipt_id in self._property_navigation_pending:
            return self._verify_property_navigation_pending(candidate, receipt)
        if receipt.receipt_id in self._readonly_property_receipts:
            return self.verify(candidate, receipt)
        known = self._verified.get(receipt.receipt_id)
        pending = self._pending.get(receipt.receipt_id)
        if (
            not receipt.success
            or known is None
            or known[0] != candidate.to_dict()
            or known[1] != receipt
            or pending is None
        ):
            return Verification(
                False, "agent.effect", {"reason": "no recoverable semantic receipt", "task_success": False}
            )
        # Early returns (including exhausted observation budgets) must not
        # re-expose a diagnostic for a copied/altered or superseded receipt.
        known = (known[0], known[1], self.verify(candidate, receipt))
        binding, attempts = pending
        if known[2].passed and (
            not binding.step.navigation or known[2].details.get("postconditions_met", False)
        ):
            return known[2]
        launch_until = self._launch_wait_until.get(receipt.receipt_id)
        application_wait = known[2].details.get("application_wait", {})
        if application_wait and not application_wait.get("eligible"):
            return known[2]
        if launch_until is not None and not application_wait.get("eligible"):
            return known[2]
        if attempts >= (20 if launch_until is not None else 2) or (
            launch_until is not None and time.perf_counter() >= launch_until
        ):
            return Verification(
                False,
                "agent.effect",
                {**known[2].details, "reason": "observation recovery budget exhausted",
                 "task_success": False},
            )
        input_evidence = self.peek_input_observation_evidence(candidate, receipt)
        self._input_evidence = None
        self._pending[receipt.receipt_id] = (binding, attempts + 1)
        after = None
        try:
            after = self.observe()
            direct_condition = True
            target = self._target_after(binding.target, after) if binding.target else None
            if binding.step.operation == "fill":
                direct_condition = target is not None and value_matches(target, binding.step.value)
            elif binding.step.operation in {"check", "uncheck"}:
                direct_condition = target is not None and checked_matches(
                    target, binding.step.operation == "check", after,
                )
                if native_toggle_control(binding.target):
                    direct_condition = direct_condition and (
                        target is not None and native_toggle_control(target)
                    )
            postcondition_evidence = _postcondition_evidence(binding.step.postconditions, after)
            satisfied = direct_condition and all(fact["satisfied"] for fact in postcondition_evidence)
            if binding.property_step is not None and not binding.step.navigation:
                satisfied = satisfied and property_effect_holds(binding.step, after, target)
            passed = satisfied
            effects = ["postconditions"] if satisfied else []
            navigation = {}
            if binding.step.navigation:
                navigation = self._navigation_details(binding, binding.snapshot, after)
                progressed = navigation["accessibility_progress"]
                passed = progressed or satisfied
                if progressed:
                    effects.append("accessibility")
            verification = Verification(
                passed,
                "agent.effect",
                {
                    "effects": effects,
                    "postconditions_met": satisfied,
                    "postcondition_evidence": postcondition_evidence,
                    "observation_recovery": attempts + 1,
                    "task_success": False,
                    **navigation,
                    **self._application_wait_details(binding, after, postcondition_evidence),
                    **(self._pending_numeric_edit(binding, after)
                       if self._numeric_receipt_matches(candidate, receipt) else {}),
                },
            )
        except Exception:
            verification = Verification(
                False, "agent.effect", {"reason": "effect observation failed", "task_success": False}
            )
        self._verified[receipt.receipt_id] = (known[0], receipt, verification)
        if (input_evidence is not None and input_evidence.binding is binding
                and after is not None and after is self._last_observed_snapshot
                and self.fresh_observe_calls == input_evidence.observation_count + 1
                and self.environment_steps == input_evidence.environment_steps):
            self._remember_input_observation(
                binding, candidate, receipt, input_evidence.actual_before, after,
                self._last_observed_received_at, verification, from_pending=True,
            )
        if after is not None:
            self._update_recovery_context(binding, candidate, receipt, after, verification)
        if binding.step.navigation:
            self._commit_navigation_handoff(binding, verification.details)
        return verification

    @staticmethod
    def _property_navigation_affordances(binding: _Binding, snapshot: AgentSnapshot) -> tuple:
        if binding.property_step is None:
            return ()
        return tuple(
            control for control in navigation_scope_controls(snapshot, binding.property_step.target)
            if any(executable_control(control, operation)
                   for operation in ("click", "fill", "check", "uncheck"))
        )

    def _property_navigation_receipt_matches(self, source, candidate, receipt) -> bool:
        return bool(
            not self._closed and receipt is source.receipt and receipt.success
            and candidate.to_dict() == source.candidate and receipt.to_dict() == source.receipt_data
        )

    def property_navigation_effect_pending(self, candidate: ActionCandidate, receipt: ActionReceipt) -> bool:
        """Report one strictly bound missing-scope wait; grant no input authority."""
        source = self._property_navigation_pending.get(receipt.receipt_id)
        known = self._verified.get(receipt.receipt_id)
        return bool(
            source is not None and self._property_navigation_receipt_matches(source, candidate, receipt)
            and self._property_navigation_active == receipt.receipt_id
            and source.environment_steps == self.environment_steps
            and source.observation_count == self.fresh_observe_calls and source.attempts < 2
            and known is not None and known[1] is receipt and not known[2].passed
        )

    def _verify_property_navigation_pending(self, candidate, receipt) -> Verification:
        self._property_navigation_post = None
        known = self._verified.get(receipt.receipt_id)
        if not self.property_navigation_effect_pending(candidate, receipt):
            return Verification(False, "agent.effect", {
                "reason": "property navigation observation ticket unavailable or exhausted",
                "task_success": False, "postconditions_met": False, "property_done": False,
            })
        source = self._property_navigation_pending[receipt.receipt_id]
        source = replace(source, attempts=source.attempts + 1, observation_count=source.observation_count + 1)
        self._property_navigation_pending[receipt.receipt_id] = source
        after = None
        try:
            after = self.observe()
            scope = navigation_scope_controls(after, source.binding.property_step.target)
            owners = {control.app.strip().casefold() for control in scope}
            windows = {control.window for control in scope if control.window.strip()}
            unknown_owner_affordances = any(
                not control.app.strip() and control.window in windows
                and any(executable_control(control, operation)
                        for operation in ("click", "fill", "check", "uncheck"))
                for control in after.controls
            )
            restored = bool(
                after.a11y_complete and owners == {source.owner_app}
                and self._property_navigation_affordances(source.binding, after)
                and not unknown_owner_affordances
                and self.fresh_observe_calls == source.observation_count
                and self.environment_steps == source.environment_steps
            )
            navigation = self._navigation_details(source.binding, source.binding.snapshot, after)
            progressed = restored and navigation["accessibility_progress"]
            verification = Verification(bool(progressed), "agent.effect", {
                **navigation, "accessibility_progress": bool(progressed),
                "effects": ["accessibility"] if progressed else [],
                "postconditions_met": False, "postcondition_evidence": [], "property_done": False,
                "property_navigation_wait": True, "observation_recovery": source.attempts,
                "pending_observation_fingerprint": after.fingerprint(),
                "observation_source": ("verified_pending_effect_observation" if progressed
                                       else "pending_effect_observation"),
                "task_success": False,
            })
        except Exception:
            verification = Verification(False, "agent.effect", {
                "reason": "property navigation effect observation failed", "task_success": False,
                "postconditions_met": False, "property_done": False,
                "property_navigation_wait": True, "observation_recovery": source.attempts,
            })
        self._verified[receipt.receipt_id] = (known[0], receipt, verification)
        if (verification.passed and after is self._last_observed_snapshot
                and self._last_observed_received_at is not None):
            self._property_navigation_post = (source, after, self._last_observed_received_at)
        return verification

    def close(self) -> None:
        self._input_evidence = None
        self._input_evidence_enabled = False
        self._numeric_finalize_pending = self._numeric_finalize_snapshot = None
        self._numeric_finalize_issued = None
        self._numeric_finalize_context = None
        self._numeric_edit_source = None
        self._numeric_planning_snapshot = None
        self._property_navigation_post = None
        self._property_navigation_active = None
        self._post_snapshot = None
        self._stale_rejection = self._stale_decision_snapshot = None
        self._recovery_source = self._recovery_context = self._recovery_confirmation = None
        self._last_observed_snapshot = None
        self._last_observed_received_at = None
        if not self._closed:
            controller = getattr(self.env, "controller", None)
            if (
                self._wrapped_command is not None
                and getattr(controller, "execute_python_command", None) is self._wrapped_command
            ):
                controller.execute_python_command = self._original_command
            self._issued.clear()
            self._verified.clear()
            self._property_receipts.clear()
            self._readonly_property_receipts.clear()
            self._pending.clear()
            self._property_navigation_pending.clear()
            self._launch_wait_until.clear()
            self._navigation_apps.clear()
            self._navigation_planned = None
            self._action_observation = None
            self._action_observation_received_at = None
            self._closed = True
