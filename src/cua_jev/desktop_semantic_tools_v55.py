"""Opt-in flat current declarations around ancestor54 and the unchanged parser.

This module only projects one request and extracts one response. It sends no
HTTP, executes no function, supplies no fallback and grants no UI authority.
Reusable property/navigation/field-edit/completion mappings remain ancestor54.
"""

import copy
import hashlib
import json
import re

from .desktop_flat_current_codec import COMPLETION, FAMILIES, decode_flat_current
from .desktop_flat_current_prompt import rewrite_h1_wire_prompt, rewrite_h6_wire_prompt
from .desktop_flat_current_schema import flat_current_schemas
from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v54 import (
    decode_semantic_declaration as decode_ancestor,
)
from .desktop_semantic_tools_v54 import (
    encode_semantic_declaration as encode_ancestor,
)

FORMAT = "semantic_tools_v55"
ANCESTOR_FORMAT = "semantic_tools_v54"
RETAINED_FUNCTIONS = (
    "declare_property_stage", "declare_navigation_stage", "declare_field_edits_stage", "declare_completion",
)
CURRENT_DESCRIPTIONS = {
    "declare_native_current_action": (
        "Declare one current native-ref action with direct operation/ref and literal value when needed. "
        "No semantic target, geometry, predicates or execution occurs."
    ),
    "declare_semantic_current_action": (
        "Declare one current semantic control action with target and nonempty effect predicates. "
        "No direct ref, geometry, future navigation or execution occurs."
    ),
    "declare_visual_current_action": (
        "Declare one current uploaded-image action with explicit visible-object description and box/space. "
        "No ref, target or predicates; original parser and fresh input guards still apply."
    ),
    "declare_key_page_action": (
        "Declare one current key/page action with a literal value where needed and its original raw "
        "or complete guarded predicates. No target/ref/geometry or function execution."
    ),
    COMPLETION: (
        "Declare an empty advisory current completion with summary/done=true and optional observed "
        "whole-goal conditions. No action or official success is asserted."
    ),
}


def _tools_by_name(tools):
    _require(isinstance(tools, list) and bool(tools), "actual ancestor declaration tools required")
    found = {}
    for tool in tools:
        _require(isinstance(tool, dict) and tool.get("type") == "function"
                 and isinstance(tool.get("function"), dict), "original function envelope required")
        function = tool["function"]
        name = function.get("name")
        _require(isinstance(name, str) and name not in found
                 and isinstance(function.get("parameters"), dict), "unique named parameters required")
        found[name] = tool
    return found


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000"):
    """Replace only current families/wire using the actual complete ancestor54 request."""
    ancestor, ancestor_provenance = encode_ancestor(
        body, max_actions=max_actions, profile=profile, contract=contract, observation=observation,
        visual_box_space=visual_box_space,
    )
    _require(ancestor_provenance.get("response_transport") == ANCESTOR_FORMAT
             and ancestor_provenance.get("image_uploaded") is True,
             "actual ancestor54 full-image request required")
    encoded = copy.deepcopy(ancestor)
    by_name = _tools_by_name(ancestor.get("tools"))
    immediate = max_actions == 1
    expected_names = ({"declare_desktop_stage"} if immediate else
                      set(RETAINED_FUNCTIONS) | {"declare_current_action"})
    _require(set(by_name) == expected_names, "actual ancestor tool inventory changed")
    source_name = "declare_desktop_stage" if immediate else "declare_current_action"
    current_parameters = flat_current_schemas(
        by_name[source_name]["function"]["parameters"], max_actions=max_actions,
    )
    flat_tools = [{"type": "function", "function": {
        "name": name, "description": CURRENT_DESCRIPTIONS[name], "parameters": parameters,
    }} for name, parameters in current_parameters.items()]
    if immediate:
        encoded["tools"] = flat_tools
        rewritten = rewrite_h1_wire_prompt(
            ancestor["messages"][0]["content"], effective_max_actions=max_actions,
            visual_box_space=visual_box_space,
        )
    else:
        encoded["tools"] = []
        for tool in ancestor["tools"]:
            if tool["function"]["name"] == "declare_current_action":
                encoded["tools"].extend(flat_tools)
            else:
                encoded["tools"].append(copy.deepcopy(tool))
        rewritten = rewrite_h6_wire_prompt(
            ancestor["messages"][0]["content"], effective_max_actions=max_actions,
            visual_box_space=visual_box_space,
        )
        actual = _tools_by_name(encoded["tools"])
        _require(all(actual[name] == by_name[name] for name in RETAINED_FUNCTIONS),
                 "property/navigation/field-edit/completion tool changed")
    _require(len(encoded["tools"]) == (5 if immediate else 8), "unexpected flat declaration count")
    encoded["messages"][0]["content"] = rewritten
    _require(encoded.get("tool_choice") == "auto" and encoded.get("parallel_tool_calls") is False,
             "original inert nonparallel auto-selection required")
    restored = copy.deepcopy(encoded)
    restored["tools"] = copy.deepcopy(ancestor["tools"])
    restored["messages"][0]["content"] = ancestor["messages"][0]["content"]
    _require(restored == ancestor, "unexpected ancestor request mutation")
    return encoded, {
        **ancestor_provenance,
        "response_transport": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_request_provenance": ancestor_provenance,
        "ancestor_request_sha256": _digest(ancestor), "request_body_sha256": _digest(encoded),
        "ancestor_declaration_schema_sha256": _digest(ancestor["tools"]),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "declaration_tools_sha256": _digest(encoded["tools"]),
        "ancestor_parameter_schema_sha256": {
            name: _digest(tool["function"]["parameters"]) for name, tool in by_name.items()
        },
        "parameter_schema_sha256": {
            tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in encoded["tools"]
        },
        "ancestor_system_sha256": _digest(ancestor["messages"][0]["content"]),
        "projected_system_sha256": _digest(rewritten), "generation_projection_format": FORMAT,
        "declaration_tool_count": len(encoded["tools"]), "input_authority": False,
        "provider_schema_enforcement": "unverified", "strict_enforcement_claimed": False,
    }


def _reject_nonfinite(_value):
    raise ValueError("non-finite declaration JSON is not permitted")


def _extract_function(payload, allowed):
    """Reject the entire malformed envelope; accompanying content is never parsed."""
    _require(isinstance(payload, dict), "declaration response must be an object")
    choices = payload.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
             "one declaration choice required")
    message = choices[0].get("message")
    _require(isinstance(message, dict) and not message.get("refusal"), "missing message or refusal")
    calls = message.get("tool_calls")
    _require(isinstance(calls, list) and len(calls) == 1, "one declaration function required")
    call = calls[0]
    _require(isinstance(call, dict) and call.get("type") == "function"
             and isinstance(call.get("function"), dict), "invalid declaration function envelope")
    function = call["function"]
    name, arguments = function.get("name"), function.get("arguments")
    _require(isinstance(name, str) and name in allowed, "unknown declaration function")
    _require(isinstance(arguments, str) and 0 < len(arguments) <= 64000,
             "bounded declaration JSON argument string required")
    try:
        raw = json.loads(arguments, parse_constant=_reject_nonfinite)
        _require(isinstance(raw, dict), "object declaration arguments required")
        _digest(raw)  # Also reject finite-looking exponents that overflow to inf.
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ValueError("invalid declaration argument JSON") from None
    return name, arguments, {
        "tool_call_count": 1, "accompanying_content_present": message.get("content") not in (None, ""),
        "argument_characters": len(arguments),
        "argument_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
    }


def decode_semantic_declaration(payload, *, max_actions, prepare_repair_token=None):
    """Map one known declaration to the original parser object without executing it."""
    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    _require(prepare_repair_token is None or (max_actions == 1
             and isinstance(prepare_repair_token, str)
             and re.fullmatch(r"[0-9a-f]{32}", prepare_repair_token)), "original H1 repair offer required")
    allowed = set(FAMILIES) | ({COMPLETION} if max_actions == 1 else set(RETAINED_FUNCTIONS))
    name, arguments, envelope_provenance = _extract_function(payload, allowed)
    if name in FAMILIES or name == COMPLETION:
        mapped, mapped_provenance = decode_flat_current(
            name, arguments, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
        )
        return mapped, {
            **envelope_provenance, **mapped_provenance,
            "response_transport": FORMAT, "ancestor_response_transport": ANCESTOR_FORMAT,
            "mapped_intent": "act_current" if max_actions > 1 else "immediate_plan",
            "original_parser_required": True, "input_authority": False,
        }
    # Keep the exact ancestor54 stage mapper, including property numeric/confirm
    # flattening and the separate guarded-commit definitions. No action repair.
    mapped, ancestor_provenance = decode_ancestor(payload, max_actions=max_actions)
    return mapped, {
        **ancestor_provenance, **envelope_provenance,
        "response_transport": FORMAT, "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_mapping_provenance": ancestor_provenance,
        "mapped_object_sha256": _digest(mapped), "mapping_applied": True,
        "original_parser_required": True, "input_authority": False,
    }
