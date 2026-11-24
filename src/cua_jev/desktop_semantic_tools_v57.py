"""Structural current-declaration hints over the exact v56 mapping.

This wrapper declares one response, executes no tool or GUI action, sends no
HTTP, repairs no model output and grants no authority. Original parsing and
fresh execution guards remain required after the unchanged ancestor mapper.
"""

import copy

from .desktop_flat_current_schema import FAMILIES
from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v56 import decode_semantic_declaration as decode_ancestor
from .desktop_semantic_tools_v56 import encode_semantic_declaration as encode_ancestor

FORMAT = "semantic_tools_v57"
ANCESTOR_FORMAT = "semantic_tools_v56"
SEMANTIC_TOOL = "declare_semantic_current_action"
DESTINATION_CONTENT_GUIDANCE = (
    "\nDESTINATION CONTENT: Do not assume a search or query field is an address/location field. "
    "Typing URL-like text into an input does not by itself establish navigation to that destination. "
    "Use the observed field role, labels and current behavior to distinguish searching from "
    "navigation. A tab/window title or input value echoing the requested text alone does not "
    "establish arrival or completion; inspect the actual page content and required controls. "
    "A loading or searching view is not evidence that the requested destination has loaded. "
    "Use an already open relevant view when its current contents support the next subgoal."
)
H1_DEPENDENCY_HINT = (
    " This declaration has exactly one current action. Omit depends_on or use an empty array; "
    "earlier model calls are not preceding step ids in this declaration."
)
H6_SEMANTIC_CLICK_HINT = (
    " In a reusable horizon, semantic click requires either an explicit same-target "
    "selected/focused/checked/unchecked effect, or a final commit with a persistent "
    "value_equals/checked/unchecked/selected precondition and its original effect conditions. "
    "A same-target local-state click may omit preconditions. Opening another UI region uses "
    "a navigation stage; do not invent a guard, predicate or role to fit a tool. "
    "The separate native-current family keeps its original one-ref immediate exception."
)


def _functions(tools):
    _require(type(tools) is list, "actual ancestor tools required")
    found = {}
    for tool in tools:
        _require(type(tool) is dict and tool.get("type") == "function"
                 and type(tool.get("function")) is dict, "actual function envelope required")
        function = tool["function"]
        name = function.get("name")
        _require(type(name) is str and name not in found
                 and type(function.get("parameters")) is dict,
                 "unique original function parameters required")
        found[name] = function
    return found


def _local_or_commit():
    """Necessary structure only; original parser checks actual target equality."""
    return [
        {
            "properties": {"postconditions": {
                "contains": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["selected", "focused", "checked", "unchecked"]},
                        "target": {"type": "object"},
                    },
                    "required": ["kind", "target"],
                },
                "minContains": 1,
            }},
            "required": ["postconditions"],
        },
        {
            "properties": {"preconditions": {
                "contains": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["value_equals", "checked", "unchecked", "selected"]},
                    },
                    "required": ["kind"],
                },
                "minContains": 1,
            }},
            "required": ["preconditions"],
        },
    ]


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000"):
    ancestor, provenance = encode_ancestor(
        body, max_actions=max_actions, profile=profile, contract=contract,
        observation=observation, visual_box_space=visual_box_space,
    )
    _require(provenance.get("response_transport") == ANCESTOR_FORMAT,
             "actual complete v56 request required")
    original_system = ancestor["messages"][0]["content"]
    _require(type(original_system) is str and DESTINATION_CONTENT_GUIDANCE not in original_system,
             "original system must not already contain the new guidance suffix")
    encoded = copy.deepcopy(ancestor)
    encoded["messages"][0]["content"] = original_system + DESTINATION_CONTENT_GUIDANCE
    functions = _functions(encoded["tools"])
    original_functions = _functions(ancestor["tools"])
    changed = []
    _require(set(FAMILIES) <= set(functions), "all original current families required")
    if max_actions == 1:
        for name in FAMILIES:
            function = functions[name]
            dependencies = function["parameters"]["properties"].get("depends_on")
            _require(type(dependencies) is dict and dependencies.get("type") == "array",
                     "actual optional H1 dependency array required")
            dependencies["maxItems"] = 0
            function["description"] += H1_DEPENDENCY_HINT
            changed.append(name)
    else:
        function = functions[SEMANTIC_TOOL]
        branches = function["parameters"].get("anyOf")
        _require(type(branches) is list, "actual current operation branches required")
        clicks = [branch for branch in branches
                  if branch.get("properties", {}).get("operation", {}).get("const") == "click"]
        _require(len(clicks) == 1 and "anyOf" not in clicks[0],
                 "one original semantic-click branch required")
        clicks[0]["anyOf"] = _local_or_commit()
        function["description"] += H6_SEMANTIC_CLICK_HINT
        changed.append(SEMANTIC_TOOL)
    restored = copy.deepcopy(encoded)
    restored_functions = _functions(restored["tools"])
    for name in changed:
        restored_functions[name]["parameters"] = copy.deepcopy(original_functions[name]["parameters"])
        restored_functions[name]["description"] = original_functions[name]["description"]
    actual_system = restored["messages"][0]["content"]
    _require(type(actual_system) is str
             and actual_system == original_system + DESTINATION_CONTENT_GUIDANCE
             and actual_system.count(DESTINATION_CONTENT_GUIDANCE) == 1,
             "only one exact appended guidance suffix is permitted")
    restored["messages"][0]["content"] = actual_system[:-len(DESTINATION_CONTENT_GUIDANCE)]
    _require(restored == ancestor, "unexpected complete v56 body mutation")
    _require(encoded.get("tool_choice") == "auto"
             and encoded.get("parallel_tool_calls") is False,
             "original inert nonparallel auto-selection required")
    return encoded, {
        **provenance,
        "response_transport": FORMAT,
        "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_request_provenance": provenance,
        "ancestor_request_sha256": _digest(ancestor),
        "request_body_sha256": _digest(encoded),
        "ancestor_declaration_schema_sha256": _digest(ancestor["tools"]),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "declaration_tools_sha256": _digest(encoded["tools"]),
        "ancestor_parameter_schema_sha256": {
            name: _digest(function["parameters"]) for name, function in original_functions.items()
        },
        "parameter_schema_sha256": {
            name: _digest(function["parameters"]) for name, function in functions.items()
        },
        "ancestor_system_sha256": _digest(ancestor["messages"][0]["content"]),
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "generation_projection_format": FORMAT,
        "schema_projection": "current_structural_necessary_conditions_v1",
        "input_authority": False,
        "provider_schema_enforcement": "unverified",
        "strict_enforcement_claimed": False,
    }


def decode_semantic_declaration(payload, *, max_actions, prepare_repair_token=None):
    raw, provenance = decode_ancestor(
        payload, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
    )
    return raw, {
        **provenance,
        "response_transport": FORMAT,
        "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_mapping_provenance": provenance,
        "input_authority": False,
    }
