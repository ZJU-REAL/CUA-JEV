"""Trace-only facts after existing declaration or prepared-property rejections.

These diagnostics accept no declaration, repair no response and grant no input.
Adapter facts contain shapes only. Scope facts allow bounded app/window/ancestor
text, never field values, arbitrary arguments, unknown fields or response content.
"""

import json
import math

_FUNCTIONS = frozenset({
    "declare_native_current_action", "declare_semantic_current_action",
    "declare_visual_current_action", "declare_key_page_action",
    "declare_current_completion", "declare_property_stage", "declare_navigation_stage",
    "declare_field_edits_stage", "declare_completion", "declare_stage_program",
    "declare_current_context_needed",
})
_FIELDS = (
    "summary", "id", "done", "depends_on", "completion_conditions", "label",
    "operation", "action_label", "recovery", "ref", "value", "button", "click_count",
    "target", "preconditions", "postconditions", "box", "box_space", "prepare_repair",
    "edit_operation", "edit_value", "numeric_finalize_key", "completion",
    "confirmation_label", "prepare", "scope", "goal_conditions", "edits", "commit",
    "conditions", "steps", "stages", "reason",
)
_KINDS = {dict: "object", list: "array", str: "string", bool: "boolean",
          int: "number", float: "number", type(None): "null"}
_ERROR_CODES = {
    KeyError: "adapter_key_error", IndexError: "adapter_index_error",
    TypeError: "adapter_type_error", ValueError: "adapter_value_error",
    RecursionError: "adapter_recursion_error",
}
_REJECTION_CODES = {
    "declaration response must be an object": ("declaration_envelope", "payload_object_required"),
    "one declaration choice required": ("declaration_envelope", "one_choice_required"),
    "missing message or refusal": ("declaration_envelope", "message_rejected"),
    "one declaration function required": ("declaration_envelope", "one_function_required"),
    "invalid declaration function envelope": ("declaration_envelope", "function_shape_rejected"),
    "unknown declaration function": ("declaration_envelope", "function_not_allowed"),
    "bounded declaration JSON argument string required": (
        "declaration_arguments", "argument_bounds_rejected",
    ),
    "invalid declaration argument JSON": ("declaration_arguments", "argument_decode_rejected"),
    "duplicate program JSON field": ("declaration_arguments", "program_duplicate_json_field"),
    "program_outer_fields": ("program_structure", "program_outer_fields_rejected"),
    "program_stage_count": ("program_structure", "program_stage_count_rejected"),
    "program_stage_fields": ("program_structure", "program_stage_fields_rejected"),
    "future_stage_requires_semantic_declarations_without_current_inputs": (
        "program_structure", "future_stage_requires_semantic_declarations_without_current_inputs",
    ),
    "program_field_stage_fields": ("program_structure", "program_field_stage_fields"),
    "program_field_edits_bound": ("program_structure", "program_field_edits_bound"),
    "program_empty_field_stage_requires_commit": (
        "program_structure", "program_empty_field_stage_requires_commit",
    ),
    "program_field_edit_fields": ("program_structure", "program_field_edit_fields"),
    "program_field_edit_operation": ("program_structure", "program_field_edit_operation"),
    "program_field_edit_value_shape": ("program_structure", "program_field_edit_value_shape"),
    "program_field_literal_text": ("program_structure", "program_field_literal_text"),
    "program_commit_fields": ("program_structure", "program_commit_fields"),
    "program_commit_operation": ("program_structure", "program_commit_operation"),
    "future_commit_requires_explicit_effect": (
        "program_structure", "future_commit_requires_explicit_effect",
    ),
    "program_commit_guard_bound": ("program_structure", "program_commit_guard_bound"),
    "program_current_commit_guards": ("program_structure", "program_current_commit_guards"),
    "program_requires_original_H6_mode": ("program_structure", "program_requires_original_H6_mode"),
    "program_unique_ordered_stage_ids": ("program_structure", "program_unique_ordered_stage_ids"),
    "program_semantic_stage_tools_only": ("program_structure", "program_semantic_stage_tools_only"),
    "program_members_cannot_claim_whole_goal": (
        "program_structure", "program_members_cannot_claim_whole_goal",
    ),
    "program_numeric_contingency_requires_numeric_literal": (
        "program_structure", "program_numeric_contingency_requires_numeric_literal",
    ),
    "program_navigation_fields": ("program_structure", "program_navigation_fields"),
    "program_navigation_arrival_required": (
        "program_structure", "program_navigation_arrival_required",
    ),
    "program_recovery_policy": ("program_structure", "program_recovery_policy"),
    "program_activation_identity": ("program_structure", "program_activation_identity"),
    "program_remaining_original_horizon_invalid": (
        "program_structure", "program_remaining_original_horizon_invalid",
    ),
    "program_activation_requires_complete_current_observation": (
        "program_structure", "program_activation_requires_complete_current_observation",
    ),
    "program_entry_conditions_not_true": ("program_structure", "program_entry_conditions_not_true"),
    "program_activation_requires_original_reusable_stage": (
        "program_structure", "program_activation_requires_original_reusable_stage",
    ),
    "program_active_stage_cannot_claim_whole_goal": (
        "program_structure", "program_active_stage_cannot_claim_whole_goal",
    ),
    "object arguments required": ("declaration_arguments", "argument_object_required"),
    "H1 empty advisory completion parameters required": ("flat_completion", "completion_header_rejected"),
    "original completion conditions required": ("flat_completion", "completion_conditions_shape_rejected"),
}
_REJECTION_CODES.update({code: ("program_wire", code) for code in (
    'program_wire_arguments_must_be_object',
    'program_wire_forbids_member_completion',
    'program_wire_forbids_member_goal_conditions',
    'program_wire_requires_original_stage_program',
    'program_wire_codec_version',
    'program_wire_provenance_mismatch',
    'program_wire_normalized_parser_input_mismatch',
)})
_ARGUMENT_CHARACTERS = 64000
_TOOLS = 16
_ROOT_KEYS = 512
_COUNT = 32
_ITEMS = 8
_NODES = 512
_DEPTH = 24


class _NonfiniteToken(ValueError):
    pass


def _reject_constant(_token):
    raise _NonfiniteToken()


def _kind(value):
    return _KINDS.get(type(value), "non_json_type")


def _shape(parent, key):
    if key not in parent:
        return {"present": False, "type": "missing"}
    value = parent[key]
    facts = {"present": True, "type": _kind(value)}
    if type(value) in {dict, list}:
        facts.update(count_capped=min(len(value), _COUNT), overflow=len(value) > _COUNT)
    if type(value) is list:
        counts = {}
        for item in value[:_ITEMS]:
            kind = _kind(item)
            counts[kind] = counts.get(kind, 0) + 1
        facts.update(item_types_prefix=counts, item_type_prefix_complete=len(value) <= _ITEMS)
    return facts


def _finiteness(raw):
    # Do not serialize/hash values or scan an unbounded decoded tree.
    pending, visited = [(raw, 0)], 0
    while pending:
        if visited >= _NODES:
            return "unverified_budget"
        value, depth = pending.pop()
        visited += 1
        if type(value) is float and not math.isfinite(value):
            return "nonfinite"
        if type(value) in {dict, list}:
            if depth >= _DEPTH or len(value) > _NODES - visited - len(pending):
                return "unverified_budget"
            children = value.values() if type(value) is dict else value
            pending.extend((child, depth + 1) for child in children)
    return "all_finite"


def _one_function(payload):
    if type(payload) is not dict:
        return None
    choices = payload.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        return None
    message = choices[0].get("message")
    if type(message) is not dict:
        return None
    calls = message.get("tool_calls")
    if type(calls) is not list or len(calls) != 1 or type(calls[0]) is not dict:
        return None
    call = calls[0]
    if call.get("type") != "function" or type(call.get("function")) is not dict:
        return None
    # No content, refusal, ID, legacy function_call or provider error is visited.
    return call["function"]


def _known_schema(function, offered_tools):
    name = function.get("name")
    if type(name) is not str or name not in _FUNCTIONS or type(offered_tools) is not list:
        return None
    if len(offered_tools) > _TOOLS:
        return None
    selected = []
    for tool in offered_tools:
        entry = tool.get("function") if type(tool) is dict else None
        if type(entry) is dict and entry.get("name") == name:
            selected.append(entry.get("parameters"))
    if len(selected) != 1 or type(selected[0]) is not dict:
        return None
    properties = selected[0].get("properties")
    required = selected[0].get("required", [])
    if type(properties) is not dict or type(required) is not list or len(required) > _COUNT:
        return None
    # Even trusted schema additions cannot turn unknown field names into output.
    known = tuple(key for key in _FIELDS if key in properties)
    required_known = tuple(key for key in known if key in required)
    return known, required_known, len(properties) - len(known)


def _program_member_completion_facts(raw):
    """Bounded known-field facts only; never copy member IDs/targets/values."""
    stages = raw.get("stages")
    if type(stages) is not list:
        return {"status": "stages_not_array"}
    limit, members = 12, []
    for index, member in enumerate(stages[:limit]):
        item = {"index": index, "member_type": _kind(member)}
        if type(member) is not dict:
            item["inspection"] = "member_not_object"
            members.append(item)
            continue
        item["arguments"] = _shape(member, "arguments")
        arguments = member.get("arguments")
        if type(arguments) is not dict:
            item["inspection"] = "arguments_not_object"
            members.append(item)
            continue
        item["inspection"] = "known_fields_only"
        item["completion"] = _shape(arguments, "completion")
        value = arguments.get("completion")
        item["completion"]["enum"] = (
            "missing" if "completion" not in arguments else
            value if type(value) is str and value in {"stage", "goal"} else "other"
        )
        item["goal_conditions"] = {"present": "goal_conditions" in arguments}
        members.append(item)
    return {"status": "bounded_prefix", "count_capped": min(len(stages), limit),
            "overflow": len(stages) > limit, "members": members}


def _facts(payload, offered_tools, result):
    function = _one_function(payload)
    if function is None:
        result["argument_parse_status"] = "envelope_unavailable"
        return result
    schema = _known_schema(function, offered_tools)
    if schema is None:
        result["argument_parse_status"] = "known_offered_schema_unavailable"
        return result
    arguments = function.get("arguments")
    result["arguments"] = {"present": "arguments" in function, "type": _kind(arguments)}
    if type(arguments) is not str or not 0 < len(arguments) <= _ARGUMENT_CHARACTERS:
        result["argument_parse_status"] = "outside_string_bounds"
        return result
    try:
        raw = json.loads(arguments, parse_constant=_reject_constant)
    except _NonfiniteToken:
        result["argument_parse_status"] = "nonfinite_token"
        return result
    except json.JSONDecodeError:
        result["argument_parse_status"] = "invalid_json"
        return result
    except RecursionError:
        result["argument_parse_status"] = "decoder_depth_exceeded"
        return result
    except (ValueError, TypeError, OverflowError, UnicodeError):
        result["argument_parse_status"] = "decoder_rejected"
        return result
    result.update(argument_parse_status="parsed", root_type=_kind(raw), finiteness=_finiteness(raw))
    if type(raw) is not dict:
        return result
    known, required, unrepresented_schema_keys = schema
    unknown, inspected = 0, 0
    for key in raw:
        if inspected == _ROOT_KEYS:
            break
        inspected += 1
        if key not in known:
            unknown += 1
    result["top_level"] = {
        "count_capped": min(len(raw), _COUNT), "overflow": len(raw) > _COUNT,
        "unknown_key_count_capped": min(unknown, _COUNT),
        "unknown_key_count_overflow": unknown > _COUNT,
        "unknown_key_scan_complete": inspected == len(raw),
        "unrepresented_schema_key_count_capped": min(unrepresented_schema_keys, _COUNT),
        "unrepresented_schema_key_count_overflow": unrepresented_schema_keys > _COUNT,
        "known_required_count": len(required),
        "known_required_missing_count": sum(key not in raw for key in required),
        "known_fields": {key: _shape(raw, key) for key in known},
    }
    if function.get("name") == "declare_stage_program":
        try:
            result["program_members"] = _program_member_completion_facts(raw)
        except Exception:
            # The extra facts must not mask the original fixed rejection code.
            result["program_members"] = {"status": "unavailable"}
    return result


def declaration_rejection_facts(payload, *, offered_tools, error):
    """Describe only the already-rejected adapter input; never return parsed data.

    Exact existing fixed decoder messages map to a fixed code; combined checks
    stay combined. Unknown messages are never rendered, copied or hashed. Their
    bounded argument is used only for whitelist equality. Structure facts are not
    schema validation and cannot locate an unrecorded historical failure.
    """
    stage, code = "declaration_adapter", _ERROR_CODES.get(type(error), "adapter_rejected")
    if (type(error) is ValueError and len(error.args) == 1
            and type(error.args[0]) is str and len(error.args[0]) <= 96):
        stage, code = _REJECTION_CODES.get(error.args[0], (stage, code))
    result = {
        "version": "bounded_declaration_rejection_v1",
        "stage": stage, "code": code,
        "untrusted": True, "input_authority": False, "schema_validated": False,
        "argument_parse_status": "not_inspected", "root_type": "not_decoded",
        "finiteness": "not_checked",
    }
    try:
        return _facts(payload, offered_tools, result)
    except Exception:
        # Diagnostics must not replace the original rejection with their own error.
        return {"version": "bounded_declaration_rejection_v1", "stage": "declaration_adapter",
                "code": "adapter_rejected", "details_status": "unavailable",
                "untrusted": True, "input_authority": False, "schema_validated": False}


_SCOPE_CHARACTERS = 240
_SCOPE_COUNT = 255
_SCOPE_REJECTION = "prepared property scope must identify a currently observed app/window"


def _scope_text_facts(source):
    values, sanitized, truncated = {}, False, False
    for field in ("app", "window", "ancestor"):
        value = source.get(field)
        value = value if type(value) is str else ""
        prefix = value[:_SCOPE_CHARACTERS]
        clean = "".join(character if character.isprintable() else " " for character in prefix)
        values[field] = clean
        sanitized = sanitized or clean != prefix
        truncated = truncated or len(value) > _SCOPE_CHARACTERS
    return values, sanitized, truncated


def prepared_property_scope_rejection_facts(plan, snapshot, *, offered_scopes, error):
    """Explain this existing refusal using bounded scope text, never edit values.

    The scopes are the actual already-offered request scopes, not replacement
    targets. Only app/window/ancestor text is allowed; no field values, labels,
    arguments, unknown fields or error content are retained. No input is granted.
    """
    result = {
        "version": "bounded_prepared_property_scope_rejection_v1",
        "stage": "execution_plan", "code": "prepared_property_scope_not_observed",
        "untrusted": True, "input_authority": False, "schema_validated": False,
    }
    try:
        from .agent_types import AgentPlan, AgentTarget
        from .desktop_semantics import matching_controls

        if (type(error) is not ValueError or error.args != (_SCOPE_REJECTION,)
                or type(plan) is not AgentPlan or len(plan.steps) not in {2, 3}):
            return None
        step = plan.steps[1]
        if (step.property_edit is None or step.property_prepare_dependency != "prepare"
                or step.depends_on != ("prepare",) or plan.steps[0].id != "prepare"
                or type(step.target) is not AgentTarget
                or matching_controls(snapshot, step.target, allow_scope=True)):
            return None
        declared, sanitized, truncated = _scope_text_facts({
            field: getattr(step.target, field) for field in ("app", "window", "ancestor")
        })
        available = type(offered_scopes) is list
        scopes = offered_scopes if available else []
        shown = []
        for item in scopes[:_ITEMS]:
            if type(item) is not dict or type(item.get("target")) is not dict:
                continue
            count = item.get("candidate_count")
            if type(count) is not int or not 1 <= count <= _SCOPE_COUNT:
                continue
            target, clean, clipped = _scope_text_facts(item["target"])
            if not (target["app"].strip() or target["window"].strip()):
                continue
            shown.append({"target": target, "candidate_count": count,
                          "scope_sanitized": clean, "scope_truncated": clipped})
        result.update(
            details_status="available", source="current_planning_snapshot_and_request_scopes",
            declared_scope=declared, declared_scope_sanitized=sanitized,
            declared_scope_truncated=truncated, declared_scope_match_count=0,
            offered_navigation_scopes=shown,
            offered_scope_status="available" if available else "unavailable",
            offered_scope_count_capped=min(len(scopes), _SCOPE_COUNT),
            offered_scope_count_overflow=len(scopes) > _SCOPE_COUNT,
            offered_scope_list_complete=available and len(scopes) == len(shown),
        )
        return result
    except Exception:
        # Diagnostic failures must preserve the original validation refusal.
        return {**result, "details_status": "unavailable"}
