'Semantic declaration tools with ordered stage programs.'

from __future__ import annotations

import copy
import json

from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v55 import _extract_function
from .desktop_semantic_tools_v58 import decode_semantic_declaration as decode_ancestor
from .desktop_semantic_tools_v58 import encode_semantic_declaration as encode_ancestor
from .desktop_stage_program import PROGRAM_INTENT, PROGRAM_TOOL, STAGE_TOOLS, parse_stage_program

FORMAT = "semantic_tools_v60"
ANCESTOR_FORMAT = "semantic_tools_v58"
PROGRAM_GUIDANCE = (
    "\nNORMAL H6 PRIORITY: Assess ALL remaining requirements. Keep one property stage if it covers "
    "the complete result, navigation and required submission. Otherwise PREFER declare_stage_program "
    "for TWO OR MORE distinct, ordered, safely expressible known subgoals. Known means goal-supplied "
    "results, NOT that future UI is already visible. Later activation and guards prove its actual "
    "binding. Do not replan merely to declare another known result; if a required argument cannot be "
    "expressed without guessing, retain the original atomic/single-stage tools. Programs are optional. "
    "Arguments: label,stages; members: id,tool,arguments, optional entry_conditions/recovery. Use only "
    "original navigation/property/field_edits member tools, completion=stage, no goal_conditions. "
    "Future members compile only on their current full activation frame. Never invent future refs, "
    "control labels, roles, owners, apps, windows or coordinates. Omit optional unknown qualifiers, or "
    "delegate a known property result for Jev to bind. Only the first property may prepare from its "
    "uploaded ref/image; no future current actions, preparations or boxes. Later explicit commits "
    "still require real owner/field/effect guards. Entry is once-per-stage fresh admission, not input "
    "authority; ongoing guards remain original action/edit preconditions. Recovery defaults to observe "
    "only where original per-step bounded polling permits; replan skips it for original H1. Never "
    "upgrade an underlying replan step, retry false/unknown entry or add budget. Numeric-key/stale "
    "contingencies remain original. SHARE the original horizon/TTL across all stages. False/unknown "
    "current exit or invalid activation retires the remaining program under original recovery; a true "
    "future entry cannot rescue it. Finish with the original fresh whole-goal review. "
    "Same-app SYNTAX ONLY: substitute one observed app identity and goal-supplied results/literals; "
    "never execute placeholders. This example assumes neither result needs confirmation input. "
    'call declare_stage_program with arguments {"label":"Complete both known results","stages":['
    '{"id":"first","tool":"declare_property_stage","arguments":{'
    '"label":"FIRST_KNOWN_RESULT","scope":{"app":"OBSERVED_APP"},'
    '"edit_operation":"fill","edit_value":"FIRST_LITERAL","numeric_finalize_key":"none",'
    '"confirmation_label":"none","completion":"stage"}},'
    '{"id":"second","tool":"declare_property_stage","arguments":{'
    '"label":"SECOND_KNOWN_RESULT","scope":{"app":"OBSERVED_APP"},'
    '"edit_operation":"fill","edit_value":"SECOND_LITERAL","numeric_finalize_key":"none",'
    '"confirmation_label":"none","completion":"stage"}}]}'
)


def program_schema(ancestor_tools, *, max_actions):
    _require(type(max_actions) is int and 2 <= max_actions <= 12, "original multi-action horizon required")
    functions = {tool["function"]["name"]: tool["function"] for tool in ancestor_tools}
    _require(set(STAGE_TOOLS) <= set(functions), "original semantic stage tools required")
    definitions = {}
    branches, future_branches = [], []
    for ordinal, name in enumerate(STAGE_TOOLS):
        parameters = copy.deepcopy(functions[name]["parameters"])
        for key, definition in parameters.pop("$defs", {}).items():
            _require(key not in definitions or definitions[key] == definition,
                     "original shared stage schema definitions differ")
            definitions[key] = definition
        parameters.pop("$schema", None)
        parameters["properties"]["completion"] = {"const": "stage"}
        parameters["properties"].pop("goal_conditions", None)
        if "completion" not in parameters["required"]:
            parameters["required"].append("completion")
        key = f"program_stage_arguments_{ordinal}"
        _require(key not in definitions, "new program schema definition collision")
        definitions[key] = parameters
        branch = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1, "maxLength": 64,
                       "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$"},
                "tool": {"const": name},
                "arguments": {"$ref": "#/$defs/" + key},
                "entry_conditions": {
                    "type": "array", "items": {"$ref": "#/$defs/condition"}, "maxItems": 8,
                    "description": (
                        "Once-per-stage fresh admission; persistent conditions belong "
                        "in original action guards."
                    ),
                },
                "recovery": {
                    "enum": ["observe", "replan"], "default": "observe",
                    "description": (
                        "Inherit original per-step bounded observation, or explicitly skip "
                        "optional polling for H1; "
                        "no entry retry or extra budget."
                    ),
                },
            },
            "required": ["id", "tool", "arguments"],
            "additionalProperties": False,
        }
        branches.append(branch)
        future_parameters = copy.deepcopy(parameters)
        future_parameters["properties"].pop("prepare", None)
        if name == "declare_field_edits_stage":
            future_parameters["properties"]["commit"] = {"$ref": "#/$defs/program_future_commit"}
        future_key = key + "_future"
        definitions[future_key] = future_parameters
        future_branch = copy.deepcopy(branch)
        future_branch["properties"]["arguments"] = {"$ref": "#/$defs/" + future_key}
        future_branches.append(future_branch)
    future_commit = copy.deepcopy(definitions["commit"])
    future_commit["properties"].pop("ref", None)
    target = future_commit["properties"]["target"]
    target["anyOf"] = [branch for branch in target["anyOf"]
                       if branch.get("$ref") != "#/$defs/ref_target"]
    future_commit["properties"]["preconditions"]["items"] = {"$ref": "#/$defs/condition"}
    if "postconditions" not in future_commit["required"]:
        future_commit["required"].append("postconditions")
    definitions["program_future_commit"] = future_commit
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "label": {"type": "string", "minLength": 1, "maxLength": 240},
            "stages": {"type": "array", "minItems": 2, "maxItems": max_actions,
                       "prefixItems": [{"anyOf": branches}],
                       "items": {"anyOf": future_branches}},
        },
        "required": ["label", "stages"],
        "additionalProperties": False,
        "$defs": definitions,
    }


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000", normal_program_eligible=True):
    _require(type(normal_program_eligible) is bool, "normal program eligibility must be boolean")
    ancestor, previous = encode_ancestor(
        body, max_actions=max_actions, profile=profile, contract=contract,
        observation=observation, visual_box_space=visual_box_space,
    )
    _require(previous.get("response_transport") == ANCESTOR_FORMAT, "actual original58 wire required")
    encoded = copy.deepcopy(ancestor)
    program_tool_added = max_actions > 1 and normal_program_eligible
    if program_tool_added:
        encoded["tools"].insert(0, {"type": "function", "function": {
            "name": PROGRAM_TOOL,
            "description": (
                "Prefer for two or more safely expressible known subgoals when one complete property "
                "stage is insufficient. Share the original horizon and TTL; single-stage tools remain."
            ),
            "parameters": program_schema(ancestor["tools"], max_actions=max_actions),
        }})
        encoded["messages"][0]["content"] += PROGRAM_GUIDANCE
        restored = copy.deepcopy(encoded)
        restored["tools"].pop(0)
        restored["messages"][0]["content"] = restored["messages"][0]["content"][:-len(PROGRAM_GUIDANCE)]
        _require(restored == ancestor, "unexpected original58 request mutation")
    elif max_actions == 1:
        _require(encoded == ancestor, "original strong H1 request body must remain exact58")
    else:
        _require(encoded == ancestor, "ineligible H6 request body must remain exact58")
    actual = {tool["function"]["name"]: tool["function"] for tool in encoded["tools"]}
    return encoded, {
        **previous,
        "response_transport": FORMAT,
        "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_request_provenance": previous,
        "ancestor_request_sha256": _digest(ancestor),
        "request_body_sha256": _digest(encoded),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "declaration_tools_sha256": _digest(encoded["tools"]),
        "parameter_schema_sha256": {name: _digest(function["parameters"])
                                    for name, function in actual.items()},
        "ancestor_system_sha256": _digest(ancestor["messages"][0]["content"]),
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "declaration_tool_count": len(actual),
        "generation_projection_format": FORMAT,
        "schema_projection": "ordered_uncompiled_semantic_program_v1",
        "strong_H1_body_exact_ancestor58": max_actions == 1,
        "normal_program_eligible": normal_program_eligible,
        "program_tool_added": program_tool_added,
        "program_implementation_is_not_input_or_success_evidence": True,
        "input_authority": False,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate program JSON field")
        result[key] = value
    return result


def decode_semantic_declaration(payload, *, max_actions, prepare_repair_token=None):
    if max_actions == 1:
        mapped, previous = decode_ancestor(
            payload, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
        )
        return mapped, {
            **previous, "response_transport": FORMAT, "requested_format": FORMAT,
            "ancestor_response_transport": ANCESTOR_FORMAT,
            "ancestor_mapping_provenance": previous,
            "program_declaration": False, "strong_H1_mapping_exact_ancestor58": True,
        }
    allowed = set(STAGE_TOOLS) | {
        "declare_native_current_action", "declare_semantic_current_action",
        "declare_visual_current_action", "declare_key_page_action", "declare_completion", PROGRAM_TOOL,
    }
    name, arguments, envelope = _extract_function(payload, allowed)
    if name != PROGRAM_TOOL:
        mapped, previous = decode_ancestor(
            payload, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
        )
        return mapped, {
            **previous, "response_transport": FORMAT, "requested_format": FORMAT,
            "ancestor_response_transport": ANCESTOR_FORMAT,
            "ancestor_mapping_provenance": previous, "program_declaration": False,
        }
    _require(prepare_repair_token is None, "program cannot be a preparation repair response")
    raw = json.loads(arguments, object_pairs_hook=_unique_object)
    program = parse_stage_program(raw, max_actions=max_actions)
    mapped = {"intent": PROGRAM_INTENT, **raw}
    return mapped, {
        **envelope, "response_transport": FORMAT, "requested_format": FORMAT,
        "tool_name": PROGRAM_TOOL, "mapped_object_sha256": _digest(mapped),
        "mapping_applied": True, "mapping_strategy": "uncompiled_ordered_semantic_program_v1",
        "program_declaration": True, "program_declaration_sha256": program.declaration_sha256,
        "stage_count": len(program.stages), "future_compiled_steps": 0,
        "original_parser_required_per_activation": True,
        "shared_original_horizon": max_actions,
        "final_original_fresh_whole_goal_review_required": True,
        "ancestor_response_transport": ANCESTOR_FORMAT, "ancestor_mapping_applied": False,
        "input_authority": False,
    }
