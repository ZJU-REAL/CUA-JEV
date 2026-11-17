"""Isolated v58 declaration candidate; grants no input authority.

V57 remains the exact ancestor for every noncompletion mapping. H1 completion
has a new explicit bridge to the original empty advisory plan object.
"""

import copy
import json
import re

from .desktop_flat_current_codec import FAMILIES
from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v55 import _extract_function
from .desktop_semantic_tools_v57 import decode_semantic_declaration as decode_ancestor
from .desktop_semantic_tools_v57 import encode_semantic_declaration as encode_ancestor

FORMAT = "semantic_tools_v58"
ANCESTOR_FORMAT = "semantic_tools_v57"
COMPLETION = "declare_completion"
OLD_COMPLETION = "declare_current_completion"
STAGES = ("declare_property_stage", "declare_navigation_stage", "declare_field_edits_stage", COMPLETION)
PROPERTY_OPERATIONS = (
    "fill is only for editable text or numeric fields, including creation/renaming with literal text. "
    "set is for a real selectable noneditable option, not typing into an editable field. "
    "check/uncheck are only for actual checkbox/radio controls. Use current evidence; "
    "do not invent a role or force a property declaration when it does not fit."
)
NUMERIC_STRATEGY = (
    "numeric_finalize_key is the existing bounded numeric contingency: enter/tab only for a "
    "declared numeric edit under the original pending-buffer, committed-value, owner, focus and "
    "fresh-input checks; none disables that contingency. It never submits a URL, search, text "
    "field or form. No key is added implicitly. Declare a required submission separately under "
    "the original confirmation or current-action rules; no budget or TTL is extended."
)
_H1_EDITS = (
    (
        "entire goal already holds and you can call declare_current_completion with "
        "done=true and completion_conditions. ",
        "entire goal already holds and you can call declare_completion with label and "
        "optional conditions covering that entire goal. ",
    ),
    (
        "completion. Set done=true only when the entire task already holds now: call "
        "declare_current_completion with summary, done=true and observable completion_conditions "
        "covering the entire goal whenever available; this completion declaration has no id "
        "or action fields and sends no input. ",
        "completion. Only when the entire task already holds now, call declare_completion with "
        "label and optional observable conditions covering the entire goal whenever available. "
        "Its function identity explicitly maps to the original empty advisory done=true plan; "
        "do not emit summary, done, completion_conditions, id or action fields in this completion "
        "declaration. It sends no input. ",
    ),
    (
        "GOAL COMPLETION: optional flat header completion_conditions uses the same predicate schema ",
        "GOAL COMPLETION: current ACTION headers retain optional completion_conditions; the "
        "no-action declare_completion instead uses optional conditions with the same predicate schema ",
    ),
    (
        "user goal, or when the entire goal already holds now using "
        "declare_current_completion with done=true. ",
        "user goal, or when the entire goal already holds now using declare_completion. ",
    ),
    (
        "declare_current_completion with summary, done=true and these observable final-state conditions. ",
        "declare_completion with label and these observable final-state conditions in conditions. ",
    ),
    (
        "Return a no-action declare_current_completion only when done is true. ",
        "Return a no-action declare_completion only when the entire original goal already holds. ",
    ),
    (
        "declare_visual_current_action, declare_key_page_action, or declare_current_completion. ",
        "declare_visual_current_action, declare_key_page_action, or declare_completion. ",
    ),
)


def _functions(tools):
    result = {}
    for tool in tools:
        function = tool["function"]
        _require(function["name"] not in result, "unique ancestor functions required")
        result[function["name"]] = function
    return result


def _rewrite_once(system, edits):
    """Use disjoint exact spans of the actual strong ancestor; preserve all other slices."""
    spans = []
    for old, new in edits:
        _require(system.count(old) == 1, "exact ancestor wire fragment required")
        start = system.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    _require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)),
             "overlapping completion wire fragments")
    parts, cursor = [], 0
    for start, stop, new in spans:
        parts.extend((system[cursor:start], new))
        cursor = stop
    parts.append(system[cursor:])
    return "".join(parts)


def _completion_parameters(old):
    _require(old.get("required") == ["summary", "done"]
             and old["properties"]["done"] == {"const": True}
             and old["properties"]["summary"].get("maxLength") == 1000
             and set(old["properties"]) == {"summary", "done", "completion_conditions"},
             "actual original H1 completion parameters required")
    result = copy.deepcopy(old)
    result["properties"] = {
        "label": copy.deepcopy(old["properties"]["summary"]),
        "conditions": copy.deepcopy(old["properties"]["completion_conditions"]),
    }
    result["required"] = ["label"]
    _require(result["properties"]["conditions"].get("maxItems") == 8
             and result["properties"]["conditions"].get("minItems", 0) == 0,
             "original H1 optional empty-through-eight conditions required")
    return result


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000"):
    ancestor, previous = encode_ancestor(
        body, max_actions=max_actions, profile=profile, contract=contract,
        observation=observation, visual_box_space=visual_box_space,
    )
    _require(previous.get("response_transport") == ANCESTOR_FORMAT, "actual complete57 request required")
    encoded = copy.deepcopy(ancestor)
    old = _functions(ancestor["tools"])
    functions = _functions(encoded["tools"])
    immediate = max_actions == 1
    if immediate:
        _require(set(old) == set(FAMILIES) | {OLD_COMPLETION}, "actual five H1 tools required")
        completion = functions[OLD_COMPLETION]
        completion["name"] = COMPLETION
        completion["parameters"] = _completion_parameters(completion["parameters"])
        completion["description"] = (
            "Declare empty advisory completion only when the entire original goal already holds. "
            "Use label (nonblank, at most1000 characters) and optional conditions (zero through8); "
            "omit all other header/action fields. The new explicit bridge maps to the original empty "
            "done=true plan; original parsing and fresh whole-goal completion checks still apply."
        )
        encoded["messages"][0]["content"] = _rewrite_once(
            ancestor["messages"][0]["content"], _H1_EDITS,
        )
        _require(OLD_COMPLETION not in encoded["messages"][0]["content"], "old completion name residue")
    else:
        _require(set(old) == set(STAGES) | set(FAMILIES), "all eight original H6 tools required")
        encoded["tools"] = [next(t for t in encoded["tools"] if t["function"]["name"] == name)
                            for name in (*STAGES, *FAMILIES)]
        function = functions["declare_property_stage"]
        function["description"] += " " + PROPERTY_OPERATIONS + " " + NUMERIC_STRATEGY
        for name, text in (("edit_operation", PROPERTY_OPERATIONS),
                           ("edit_value", "Literal edit value: " + PROPERTY_OPERATIONS),
                           ("numeric_finalize_key", NUMERIC_STRATEGY)):
            parameter = function["parameters"]["properties"][name]
            parameter["description"] = (parameter.get("description", "") + " " + text).strip()
        encoded["messages"][0]["content"] = _rewrite_once(
            ancestor["messages"][0]["content"], ((
                "Creation/renaming is fill(final text), not selecting an existing same-name option. "
                "set supports real fields/options/on-off. ",
                PROPERTY_OPERATIONS + " ",
            ),),
        ) + "\nPROPERTY NUMERIC STRATEGY: " + NUMERIC_STRATEGY
    actual = _functions(encoded["tools"])
    _require(len(actual) == (5 if immediate else 8), "original five/eight capabilities required")
    for name in FAMILIES:
        _require(actual[name] == old[name], "noncompletion current tool must remain exact57")
    if not immediate:
        for name in STAGES[1:]:
            _require(actual[name] == old[name], "navigation/field/completion tool must remain exact57")
        restored_property = copy.deepcopy(actual[STAGES[0]])
        restored_property["description"] = old[STAGES[0]]["description"]
        for name in ("edit_operation", "edit_value", "numeric_finalize_key"):
            restored_property["parameters"]["properties"][name] = copy.deepcopy(
                old[STAGES[0]]["parameters"]["properties"][name],
            )
        _require(restored_property == old[STAGES[0]], "only adjacent property documentation may change")
    restored = copy.deepcopy(encoded)
    restored["tools"] = copy.deepcopy(ancestor["tools"])
    restored["messages"][0]["content"] = ancestor["messages"][0]["content"]
    _require(restored == ancestor, "unexpected non-wire request mutation")
    return encoded, {
        **previous, "response_transport": FORMAT, "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT, "ancestor_request_provenance": previous,
        "ancestor_request_sha256": _digest(ancestor), "request_body_sha256": _digest(encoded),
        "ancestor_declaration_schema_sha256": _digest(ancestor["tools"]),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "declaration_tools_sha256": _digest(encoded["tools"]),
        "ancestor_parameter_schema_sha256": {n: _digest(f["parameters"]) for n, f in old.items()},
        "parameter_schema_sha256": {n: _digest(f["parameters"]) for n, f in actual.items()},
        "ancestor_system_sha256": _digest(ancestor["messages"][0]["content"]),
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "generation_projection_format": FORMAT, "declaration_tool_count": len(actual),
        "schema_projection": "unified_completion_and_stage_first_documentation_v58",
        "input_authority": False, "provider_schema_enforcement": "unverified",
        "strict_enforcement_claimed": False,
    }


def decode_semantic_declaration(payload, *, max_actions, prepare_repair_token=None):
    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    _require(prepare_repair_token is None or (max_actions == 1
             and isinstance(prepare_repair_token, str)
             and re.fullmatch(r"[0-9a-f]{32}", prepare_repair_token)), "original H1 repair offer required")
    if max_actions == 1:
        name, arguments, envelope = _extract_function(payload, set(FAMILIES) | {COMPLETION})
        if name == COMPLETION:
            raw = json.loads(arguments)  # Original extractor proved finite JSON and object root.
            _require(set(raw) <= {"label", "conditions"} and "label" in raw
                     and isinstance(raw["label"], str) and bool(raw["label"].strip())
                     and len(raw["label"]) <= 1000, "v58 H1 completion header required")
            _require("conditions" not in raw or (isinstance(raw["conditions"], list)
                     and len(raw["conditions"]) <= 8
                     and all(isinstance(v, dict) for v in raw["conditions"])),
                     "v58 original H1 condition array required")
            mapped = {"summary": raw["label"], "steps": [], "done": True}
            if "conditions" in raw:
                mapped["completion_conditions"] = copy.deepcopy(raw["conditions"])
            return mapped, {
                **envelope, "response_transport": FORMAT, "requested_format": FORMAT,
                "tool_name": COMPLETION, "argument_sha256": envelope["argument_sha256"],
                "mapped_object_sha256": _digest(mapped), "mapping_applied": True,
                "mapping_strategy": "v58_H1_completion_header_to_original_empty_plan_v1",
                "ancestor_response_transport": ANCESTOR_FORMAT,
                "ancestor_mapping_applied": False, "original_parser_required": True,
                "input_authority": False,
            }
    mapped, previous = decode_ancestor(
        payload, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
    )
    return mapped, {
        **previous, "response_transport": FORMAT, "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT, "ancestor_mapping_provenance": previous,
        "ancestor_mapping_applied": True, "input_authority": False,
    }
