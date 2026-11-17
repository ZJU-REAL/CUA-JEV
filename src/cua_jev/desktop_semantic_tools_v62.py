"""Explicit normal high-level phase with a zero-input strong-H1 handoff."""

import copy
import json

from .desktop_flat_current_prompt import _FLAT_CURRENT_GRAMMAR
from .desktop_phase_handoff_v62 import CONTEXT_REASONS, CONTEXT_TOOL, HighLevelContextRequest
from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v55 import _extract_function
from .desktop_semantic_tools_v58 import decode_semantic_declaration as decode_ancestor
from .desktop_semantic_tools_v58 import encode_semantic_declaration as encode_ancestor
from .desktop_semantic_tools_v60 import PROGRAM_GUIDANCE, PROGRAM_TOOL
from .desktop_semantic_tools_v60 import decode_semantic_declaration as decode_program
from .desktop_semantic_tools_v60 import encode_semantic_declaration as encode_program

FORMAT = "semantic_tools_v62"
ANCESTOR_FORMAT = "semantic_tools_v58"
HIGH_LEVEL_TOOLS = (
    PROGRAM_TOOL, "declare_property_stage", "declare_navigation_stage",
    "declare_field_edits_stage", "declare_completion",
)
PHASE_GUIDANCE = (
    "\nNORMAL HIGH-LEVEL PHASE: Return exactly one of the offered declarations. "
    "A single complete property remains valid and preferred when sufficient; do not split it "
    "to manufacture program stages. Delegate known results and legal live navigation to Jev. "
    "A needed single current prerequisite may be explicitly declared inside the first property "
    "prepare under its original current-image/ref, focus, owner and effect guards. Do not invent "
    "a property merely to carry an isolated input, or wrap a current action without a genuine "
    "known requested property result. Future stages require original semantic declarations only; "
    "never guess future refs, coordinates, roles, names, owners or required arguments. "
    "If no valid high-level declaration fits, call declare_current_context_needed with ONLY "
    "reason: unsupported_high_level_action for an isolated action outside this high-level "
    "grammar, missing_future_semantics for required unknown future information, or "
    "transient_or_unexplained_state for an unexplained current state. This sends no input and "
    "requests the original strong single-current-action phase under the SAME remaining budgets. "
    "The handoff and its following VLM request both count; do not request it when a valid "
    "complete property or safely expressible program already fits. It grants no success, "
    "new observation, new TTL, failure reset or arbitrary-action authority. "
    "All original exact matching, validation, input and final fresh whole-goal checks remain."
)
PROGRAM_MEMBER_GUIDANCE = (
    "\nFor declare_stage_program ONLY: EVERY member's arguments, INCLUDING THE LAST member, "
    "must explicitly contain completion=stage. OMIT goal_conditions entirely, even when empty "
    "or null. A member never claims the whole user goal. The top-level single-property and "
    "completion guidance elsewhere does not apply to program members. The original final "
    "fresh whole-goal review remains required after the program."
)


def _normal_system(system):
    """Remove only unavailable current-family instructions; keep prepare grammar."""
    current_names = (
        "declare_navigation_stage, declare_field_edits_stage, declare_native_current_action, "
        "declare_semantic_current_action, declare_visual_current_action, declare_key_page_action, "
        "declare_completion. "
    )
    current_header = (
        "Each named current-action function takes one current grounded action with flat label,operation "
        "and only its declared family fields, as above. Property/navigation/field-edits/completion "
        "declarations retain their original parameters and mappings. "
    )
    prepare = _FLAT_CURRENT_GRAMMAR.split(" Only the separate current-declaration wire is flat:")[0]
    edits = (
        (current_names, "declare_navigation_stage, declare_field_edits_stage, declare_completion, "
                        "declare_current_context_needed. "),
        (_FLAT_CURRENT_GRAMMAR, prepare + " Only click/visual_click permit button/click_count. "),
        (current_header, "An isolated current action requires the explicit zero-input "
                         "declare_current_context_needed handoff; no current-action tool is offered here. "),
        ("Use a current action only when no reusable stage fits; do not force ",
         "Request current context only when no valid reusable stage with legitimate prepare fits; "
         "do not force "),
        (PROGRAM_GUIDANCE, PROGRAM_GUIDANCE.replace(
            "retain the original atomic/single-stage tools.",
            "request declare_current_context_needed, or retain a valid single-stage declaration.",
        )),
    )
    for old, new in edits:
        _require(system.count(old) == 1, "exact normal phase instruction fragment required")
        system = system.replace(old, new, 1)
    _require(not any(name in system for name in (
        "declare_native_current_action", "declare_semantic_current_action",
        "declare_visual_current_action", "declare_key_page_action",
    )), "unoffered current-family instructions remain")
    return system + PHASE_GUIDANCE


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000", normal_program_eligible=True):
    _require(type(normal_program_eligible) is bool, "normal phase eligibility must be boolean")
    normal = max_actions > 1 and normal_program_eligible
    encode = encode_program if normal else encode_ancestor
    options = {"normal_program_eligible": True} if normal else {}
    encoded, previous = encode(
        body, max_actions=max_actions, profile=profile, contract=contract,
        observation=observation, visual_box_space=visual_box_space, **options,
    )
    ancestor = copy.deepcopy(encoded)
    if normal:
        tools = {tool["function"]["name"]: tool for tool in encoded["tools"]}
        _require(set(HIGH_LEVEL_TOOLS) <= set(tools), "original high-level declarations required")
        encoded["tools"] = [copy.deepcopy(tools[name]) for name in HIGH_LEVEL_TOOLS]
        encoded["tools"][0]["function"]["description"] += (
            " Every member, including the last, requires arguments.completion=stage; "
            "omit arguments.goal_conditions entirely, including an empty value."
        )
        encoded["tools"].append({"type": "function", "function": {
            "name": CONTEXT_TOOL,
            "description": "Request the original strong H1 phase when no valid high-level stage fits. "
                           "Zero input; the handoff and following model request both count.",
            "parameters": {
                "type": "object", "properties": {
                    "reason": {"type": "string", "enum": sorted(CONTEXT_REASONS)},
                }, "required": ["reason"], "additionalProperties": False,
            },
        }})
        encoded["messages"][0]["content"] = _normal_system(encoded["messages"][0]["content"])
        encoded["messages"][0]["content"] += PROGRAM_MEMBER_GUIDANCE
    _require(encoded.get("tool_choice") == "auto" and encoded.get("parallel_tool_calls") is False,
             "original inert auto-selection required")
    actual = {tool["function"]["name"]: tool["function"] for tool in encoded["tools"]}
    return encoded, {
        **previous, "response_transport": FORMAT, "requested_format": FORMAT,
        "ancestor_response_transport": previous["response_transport"],
        "ancestor_request_provenance": previous, "ancestor_request_sha256": _digest(ancestor),
        "ancestor_declaration_schema_sha256": _digest(ancestor["tools"]),
        "request_body_sha256": _digest(encoded), "declaration_tools_sha256": _digest(encoded["tools"]),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "parameter_schema_sha256": {name: _digest(function["parameters"])
                                    for name, function in actual.items()},
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "ancestor_system_sha256": _digest(ancestor["messages"][0]["content"]),
        "generation_projection_format": FORMAT, "declaration_tool_count": len(actual),
        "phase": "normal_high_level" if normal else "original_current_or_review",
        "normal_program_eligible": normal_program_eligible, "normal_high_level_phase": normal,
        "program_tool_added": normal, "strong_H1_body_exact_ancestor58": max_actions == 1,
        "ineligible_body_exact_ancestor58": not normal,
        "context_handoff_tool_added": normal, "input_authority": False,
        "program_member_guidance": "explicit_stage_only_v63" if normal else None,
    }


def _unique_context(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate context declaration field")
        result[key] = value
    return result


def decode_semantic_declaration(payload, *, max_actions, prepare_repair_token=None,
                                normal_program_eligible=True):
    _require(type(normal_program_eligible) is bool, "normal phase eligibility must be boolean")
    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    normal = max_actions > 1 and normal_program_eligible
    if not normal:
        mapped, previous = decode_ancestor(
            payload, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
        )
    else:
        _require(prepare_repair_token is None, "normal phase cannot be a preparation repair")
        name, arguments, envelope = _extract_function(payload, {*HIGH_LEVEL_TOOLS, CONTEXT_TOOL})
        if name == CONTEXT_TOOL:
            raw = json.loads(arguments, object_pairs_hook=_unique_context)
            _require(type(raw) is dict and set(raw) == {"reason"}, "context declaration reason only")
            mapped = HighLevelContextRequest(
                raw["reason"], _digest({"tool": CONTEXT_TOOL, "arguments": raw}),
            )
            previous = {
                **envelope, "tool_name": CONTEXT_TOOL, "context_request": True,
                "declaration_sha256": mapped.declaration_sha256,
                "mapping_applied": False, "program_declaration": False,
            }
        else:
            mapped, previous = decode_program(payload, max_actions=max_actions)
    return mapped, {
        **previous, "response_transport": FORMAT, "requested_format": FORMAT,
        "ancestor_mapping_provenance": previous,
        "phase": "normal_high_level" if normal else "original_current_or_review",
        "normal_high_level_phase": normal, "input_authority": False,
        "context_request": isinstance(mapped, HighLevelContextRequest),
    }
