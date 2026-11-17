"""Project flat current-family schemas from actual ancestor54 parameters.

Only the new current wire changes. Reusable property, navigation, field edits
and their commit definitions stay outside this projection. Schema hints neither
prove provider enforcement nor replace the original parser or execution guards.
"""

import copy

FAMILIES = {
    "declare_native_current_action": ("click", "fill", "check", "uncheck", "press"),
    "declare_semantic_current_action": ("click", "fill", "check", "uncheck"),
    "declare_visual_current_action": ("visual_click", "visual_fill"),
    "declare_key_page_action": (
        "type_text", "hotkey", "key_press", "press", "scroll", "back", "wait",
    ),
}
FIELDS = {
    "declare_native_current_action": ("ref", "value", "button", "click_count"),
    "declare_semantic_current_action": (
        "target", "value", "button", "click_count", "preconditions", "postconditions",
    ),
    "declare_visual_current_action": ("box", "box_space", "value", "button", "click_count"),
    "declare_key_page_action": ("value", "preconditions", "postconditions"),
}
VALUE_REQUIRED = {
    "fill", "visual_fill", "type_text", "hotkey", "key_press", "press", "scroll", "wait",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnull(schema):
    """Remove only a recognized explicit null branch in this new wire shape."""
    _require(isinstance(schema, dict) and set(schema) == {"anyOf"},
             "expected original nullable property")
    branches = schema["anyOf"]
    _require(isinstance(branches, list) and len(branches) == 2
             and branches[1] == {"type": "null"}, "expected explicit trailing null")
    return copy.deepcopy(branches[0])


def _definitions(schema, definitions):
    """Copy exact transitive original defs; reject every unknown/nonlocal ref."""
    pending, found, visited = [schema], set(), set()
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if "$ref" in item:
                ref = item["$ref"]
                _require(isinstance(ref, str) and ref.startswith("#/$defs/")
                         and ref[8:] in definitions, "unknown or nonlocal definition")
                name = ref[8:]
                found.add(name)
                if name not in visited:
                    visited.add(name)
                    pending.append(definitions[name])
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return {key: copy.deepcopy(value) for key, value in definitions.items() if key in found}


def _branch(operation, *, key_family, immediate, repair_offered):
    properties, required = {"operation": {"const": operation}}, []
    if operation in VALUE_REQUIRED:
        required.append("value")
    # The codec rejects click parameters on every other operation.
    branch = {"properties": properties, "required": required}
    if operation not in {"click", "visual_click"}:
        branch["not"] = {"anyOf": [{"required": [key]} for key in ("button", "click_count")]}
    if key_family:
        raw = {"properties": {"preconditions": {"maxItems": 0},
                              "postconditions": {"maxItems": 0}}}
        if immediate:
            raw["properties"]["completion_conditions"] = {"maxItems": 0}
        branch["anyOf"] = [raw]
        if operation != "back":
            branch["anyOf"].append({
                "properties": {"preconditions": {"minItems": 1},
                               "postconditions": {"minItems": 1}},
                "required": ["preconditions", "postconditions"],
            })
    if repair_offered:
        _require(not key_family, "repair is visual-family only")
        if operation == "visual_fill":
            branch["anyOf"] = [
                {"not": {"required": ["prepare_repair"]}},
                {"properties": {"done": {"const": False},
                                "completion_conditions": {"maxItems": 0}},
                 "required": ["prepare_repair"]},
            ]
        else:
            branch["not"] = {"required": ["prepare_repair"]}
    return branch


def flat_current_schemas(parameters, *, max_actions):
    """Return flat parameter schemas without mutating the actual input.

    H1 input: V54 declare_desktop_stage parameters. H>1 input: V54
    declare_current_action parameters. Property/navigation/commit tools are
    deliberately outside this projection, and must be retained independently.
    """
    _require(type(max_actions) is int and 1 <= max_actions <= 12,
             "actual original horizon required")
    _require(isinstance(parameters, dict) and parameters.get("type") == "object"
             and isinstance(parameters.get("$defs"), dict)
             and isinstance(parameters.get("properties"), dict),
             "actual original parameter object required")
    definitions, root = parameters["$defs"], parameters["properties"]
    immediate = max_actions == 1
    if immediate:
        _require({"summary", "steps", "done", "completion_conditions"} <= set(root)
                 and root["steps"].get("maxItems") == 1,
                 "actual H1 plan properties required")
        action = root["steps"]["items"]["properties"]
        header = {key: copy.deepcopy(root[key])
                  for key in ("summary", "done", "completion_conditions")}
        header.update({key: copy.deepcopy(action[key]) for key in ("id", "depends_on")})
        header_required = ["summary", "id", "done"]
    else:
        _require(set(root) == {"label", "action"}
                 and root["action"] == {"$ref": "#/$defs/action"},
                 "actual H6 current-action properties required")
        action = definitions["action"]["properties"]
        header, header_required = {"label": copy.deepcopy(root["label"])}, ["label"]
    _require({"operation", "label", "recovery", "target", "box", "ref",
              "preconditions", "postconditions"} <= set(action),
             "actual original action properties required")
    output = {}
    for name, operations in FAMILIES.items():
        _require(set(operations) <= set(action["operation"]["enum"]),
                 "current family needs actual original operations")
        properties = copy.deepcopy(header)
        properties.update({key: copy.deepcopy(action[key]) for key in FIELDS[name]})
        properties.update(operation={"enum": list(operations)},
                          action_label=copy.deepcopy(action["label"]),
                          recovery=copy.deepcopy(action["recovery"]))
        properties["action_label"]["minLength"] = 1
        required = header_required + ["operation"]
        if name == "declare_native_current_action":
            properties["ref"]["minLength"] = 1
            required.append("ref")
        elif name == "declare_semantic_current_action":
            properties["target"] = _nonnull(properties["target"])
            properties["postconditions"]["minItems"] = 1
            required += ["target", "postconditions"]
        elif name == "declare_visual_current_action":
            properties["box"] = _nonnull(properties["box"])
            required += ["box", "box_space"]
            if immediate:
                required.append("action_label")
        if immediate and name in {"declare_native_current_action", "declare_visual_current_action"}:
            properties["completion_conditions"]["maxItems"] = 0
        repair_offered = (immediate and name == "declare_visual_current_action"
                          and "prepare_repair" in root)
        if repair_offered:
            properties["prepare_repair"] = copy.deepcopy(root["prepare_repair"])
        result = {"type": "object", "properties": properties,
                  "required": required, "additionalProperties": False,
                  "anyOf": [_branch(op, key_family=name == "declare_key_page_action",
                                    immediate=immediate, repair_offered=repair_offered)
                            for op in operations]}
        result["$defs"] = _definitions(result, definitions)
        result["$schema"] = parameters["$schema"]
        output[name] = result
    if immediate:
        completion = {"type": "object", "properties": {
            "summary": copy.deepcopy(root["summary"]), "done": {"const": True},
            "completion_conditions": copy.deepcopy(root["completion_conditions"]),
        }, "required": ["summary", "done"], "additionalProperties": False}
        completion["$defs"] = _definitions(completion, definitions)
        completion["$schema"] = parameters["$schema"]
        output["declare_current_completion"] = completion
    return output
