"""Bounded envelope structure, never response contents or input authority."""

from __future__ import annotations

_LIMIT = 4
_KNOWN_FUNCTIONS = frozenset({
    "declare_native_current_action", "declare_semantic_current_action",
    "declare_visual_current_action", "declare_key_page_action",
    "declare_current_completion", "declare_property_stage", "declare_navigation_stage",
    "declare_field_edits_stage", "declare_completion", "declare_stage_program",
    "declare_current_context_needed",
})
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "function_call", "content_filter"})
_TYPES = {dict: "object", list: "array", str: "string", bool: "boolean",
          int: "number", float: "number", type(None): "null"}


def _kind(value):
    return _TYPES.get(type(value), "non_json_type")


def _field(parent, key):
    present = type(parent) is dict and key in parent
    return {"present": present, "type": _kind(parent[key]) if present else "missing"}


def _length_bucket(value):
    if type(value) is not str:
        return "not_string"
    count = len(value)
    if count == 0:
        return "empty"
    for upper in (64, 256, 1024, 4096, 16384, 64000):
        if count <= upper:
            return "le_" + str(upper)
    return "gt_64000"


def _array_facts(parent, key):
    result = _field(parent, key)
    values = parent[key] if result["type"] == "array" else []
    result.update(count_capped=min(len(values), _LIMIT), overflow=len(values) > _LIMIT)
    return result, values


def _function_facts(function, offered):
    result = {"type": _kind(function)}
    if type(function) is not dict:
        return result
    result["name"] = _field(function, "name")
    name = function.get("name")
    # Never emit an unknown literal, including one supplied as an offered name.
    result["name"]["enum"] = name if type(name) is str and name in offered else "unoffered"
    result["arguments"] = _field(function, "arguments")
    result["arguments"]["length_bucket"] = _length_bucket(function.get("arguments"))
    return result


def _message_facts(message, offered):
    result = {"type": _kind(message)}
    if type(message) is not dict:
        return result
    result.update(content=_field(message, "content"), refusal=_field(message, "refusal"))
    result["function_call"] = _field(message, "function_call")
    if "function_call" in message:
        result["function_call"]["function"] = _function_facts(message["function_call"], offered)
    calls, values = _array_facts(message, "tool_calls")
    result["tool_calls"] = calls
    calls["items"] = []
    for call in values[:_LIMIT]:
        entry = {"type": _kind(call)}
        if type(call) is dict:
            entry["call_type"] = _field(call, "type")
            entry["call_type"]["enum"] = "function" if call.get("type") == "function" else "other"
            entry["function"] = _field(call, "function")
            if "function" in call:
                entry["function"]["shape"] = _function_facts(call["function"], offered)
        calls["items"].append(entry)
    return result


def declaration_envelope_facts(payload, *, offered_function_names=()):
    """Return whitelist-only facts without changing or validating the response.

    Offered names come from the exact trusted request body, never from payload.
    Only this module's fixed known names may be emitted; everything else maps to
    a fixed enum. JSON contents, IDs, unknown keys, literal names and hashes are
    never traversed or included. Legacy function_call is observed, not accepted.
    """
    names = offered_function_names if type(offered_function_names) in {tuple, list} else ()
    offered = frozenset(name for name in names[:len(_KNOWN_FUNCTIONS)]
                        if type(name) is str and name in _KNOWN_FUNCTIONS)
    result = {"version": "bounded_declaration_envelope_v1", "untrusted": True,
              "input_authority": False, "schema_validated": False, "payload_type": _kind(payload)}
    if type(payload) is not dict:
        return result
    result["error"] = _field(payload, "error")
    choices, values = _array_facts(payload, "choices")
    result["choices"] = choices
    choices["items"] = []
    for choice in values[:_LIMIT]:
        entry = {"type": _kind(choice)}
        if type(choice) is dict:
            entry["finish_reason"] = _field(choice, "finish_reason")
            reason = choice.get("finish_reason")
            entry["finish_reason"]["enum"] = (
                reason if type(reason) is str and reason in _FINISH_REASONS else "other"
            )
            entry["message"] = _field(choice, "message")
            entry["delta"] = _field(choice, "delta")
            if "message" in choice:
                entry["message"]["shape"] = _message_facts(choice["message"], offered)
        choices["items"].append(entry)
    return result
