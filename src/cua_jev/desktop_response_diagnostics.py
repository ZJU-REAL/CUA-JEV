"""Bounded structural facts about an untrusted, already JSON-decoded response.

This is not a parser, schema validator, repair path or source of input authority.
Only fixed public field names and fixed enum values may reach the output.
"""

from __future__ import annotations

import hashlib
import json
import math

_MAX_NODES = 1024
_MAX_DEPTH = 12
_MAX_BYTES = 65536
_MAX_INTEGER_BITS = 4096

# Recognizing a name here does not make it legal at that path or for that intent.
# In particular, stage/reusable and misplaced prepare_repair remain parser errors.
_FIELDS = frozenset({
    "intent", "summary", "steps", "done", "completion_conditions", "label", "scope",
    "action", "prepare", "edit", "edits", "commit", "confirm", "completion", "stage",
    "goal_conditions", "arrival", "operation", "ref", "target", "value", "box", "box_space",
    "button", "click_count", "id", "depends_on", "preconditions", "postconditions", "recovery",
    "navigation", "reusable", "prepare_repair", "name", "role", "app", "window", "ancestor", "match",
})
_ENUMS = {
    "intent": frozenset({
        "pursue_property", "navigate", "edit_fields", "act_current", "complete", "reach_edit",
    }),
    "operation": frozenset({
        "click", "fill", "select", "check", "uncheck", "press", "scroll", "back", "wait",
        "visual_click", "visual_fill", "type_text", "hotkey", "key_press", "set",
    }),
    "completion": frozenset({"stage", "goal"}),
    "recovery": frozenset({"observe", "replan"}),
    "box_space": frozenset({"normalized_1000", "image_pixels"}),
    "button": frozenset({"left", "right"}),
    "match": frozenset({"exact", "contains"}),
}
_BOOL_FIELDS = frozenset({"done", "navigation", "reusable"})


class _Rejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def _scalar_size(value) -> int:
    # Only exact JSON primitive types reach this function. Never stringify an
    # unknown object, including subclasses of otherwise accepted builtins.
    kind = type(value)
    if kind is str and len(value) > _MAX_BYTES:
        raise _Rejected("budget_exceeded")
    if kind is int and value.bit_length() > _MAX_INTEGER_BITS:
        raise _Rejected("budget_exceeded")
    if kind is float and not math.isfinite(value):
        raise _Rejected("non_json_value")
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (UnicodeError, ValueError, OverflowError):
        raise _Rejected("non_json_value") from None
    if len(raw) > _MAX_BYTES:
        raise _Rejected("budget_exceeded")
    return len(raw)


def _bounded_tree(raw) -> tuple[int, int]:
    """Validate exact builtin JSON values iteratively, with an active-cycle set."""
    stack = [(raw, 0, False)]
    active = set()
    nodes = byte_count = 0
    while stack:
        value, depth, exiting = stack.pop()
        if exiting:
            active.remove(id(value))
            continue
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise _Rejected("budget_exceeded")
        kind = type(value)
        if kind is dict or kind is list:
            identity = id(value)
            if identity in active:
                raise _Rejected("cyclic_structure")
            count = len(value)
            if nodes + count > _MAX_NODES:
                raise _Rejected("budget_exceeded")
            active.add(identity)
            stack.append((value, depth, True))
            byte_count += 2 + max(0, count - 1) + (count if kind is dict else 0)
            if kind is dict:
                for key, child in reversed(tuple(value.items())):
                    if type(key) is not str:
                        raise _Rejected("non_json_key")
                    nodes += 1
                    byte_count += _scalar_size(key)
                    if nodes > _MAX_NODES or byte_count > _MAX_BYTES:
                        raise _Rejected("budget_exceeded")
                    stack.append((child, depth + 1, False))
            else:
                stack.extend((child, depth + 1, False) for child in reversed(value))
        elif kind is str or kind is int or kind is float or kind is bool or value is None:
            byte_count += _scalar_size(value)
        else:
            raise _Rejected("non_json_value")
        if byte_count > _MAX_BYTES:
            raise _Rejected("budget_exceeded")
    return nodes, byte_count


def _kind(value) -> str:
    kind = type(value)
    if kind is dict:
        return "object"
    if kind is list:
        return "array"
    if kind is str:
        return "string"
    if kind is bool:
        return "boolean"
    if kind is int or kind is float:
        return "number"
    return "null"  # The bounded-tree check has already rejected every other type.


def _bucket(size: int) -> str:
    if size == 0:
        return "0"
    for upper in (16, 64, 256, 1024, 4096, 16384, 65536):
        if size <= upper:
            return "<= " + str(upper)
    return "> 65536"


def _path_facts(value) -> dict:
    facts = {"type": _kind(value)}
    if type(value) is not dict:
        return facts
    facts["fields"] = {key: _kind(value[key]) for key in sorted(_FIELDS.intersection(value))}
    facts["unknown_field_count"] = sum(key not in _FIELDS for key in value)
    enums = {}
    for key in sorted(_ENUMS):
        if key in value and type(value[key]) is str:
            enums[key] = value[key] if value[key] in _ENUMS[key] else "other"
    for key in sorted(_BOOL_FIELDS):
        if key in value and type(value[key]) is bool:
            enums[key] = value[key]
    if "click_count" in value and type(value["click_count"]) is int:
        enums["click_count"] = value["click_count"] if value["click_count"] in (1, 2) else "other"
    if enums:
        facts["enums"] = enums
    return facts


def _finish(facts: dict) -> dict:
    # Hash the sanitized structural facts, never the original response or secrets.
    encoded = json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return {**facts, "structure_sha256": hashlib.sha256(encoded).hexdigest()}


def response_structure_facts(raw) -> dict:
    """Report fixed-path presence/type and safe enums; never accept or repair JSON."""
    base = {
        "version": "bounded_public_response_structure_v1", "untrusted": True,
        "schema_validated": False, "input_authority": False,
    }
    try:
        nodes, byte_count = _bounded_tree(raw)
    except _Rejected as error:
        return _finish({**base, "status": "unavailable", "error_type": error.reason})
    paths = {"root": _path_facts(raw)}
    if type(raw) is dict:
        for key in ("action", "prepare", "edit", "commit", "confirm"):
            if key in raw:
                paths[key] = _path_facts(raw[key])
        steps = raw.get("steps")
        if type(steps) is list and steps:
            paths["steps[0]"] = _path_facts(steps[0])
            if type(steps[0]) is dict and "action" in steps[0]:
                paths["steps[0].action"] = _path_facts(steps[0]["action"])
    return _finish({**base, "status": "observed", "node_count": nodes,
                    "json_bytes_bucket": _bucket(byte_count), "paths": paths})
