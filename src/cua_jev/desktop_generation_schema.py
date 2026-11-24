"""Generation-only intersections for opt-in semantic_tools_v54.

This is neither a parser nor an input permission. It adds only three structural
restrictions to copied, actual tool parameters. It never repairs an instance,
resolves a UI target, changes a guard, executes a tool, or sends a request.
"""

import copy

FORMAT = "semantic_tools_v54"
TARGET_OPERATIONS = ("click", "fill", "check", "uncheck")
GUARDED_OPERATIONS = ("type_text", "hotkey", "key_press", "press", "scroll", "wait")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _semantic_constraint(operation_order):
    """A reusable action has no current ref/box and must declare an effect."""
    controls = [operation for operation in operation_order if operation in TARGET_OPERATIONS]
    guarded = [operation for operation in operation_order if operation in GUARDED_OPERATIONS]
    _require(controls and guarded, "expected original desktop control/guarded operations")
    return {
        "properties": {
            "ref": {"const": ""},
            "box": {"type": "null"},
            "operation": {"enum": [operation for operation in operation_order
                                   if operation in TARGET_OPERATIONS + GUARDED_OPERATIONS]},
            "postconditions": {"minItems": 1},
        },
        "required": ["postconditions"],
        "anyOf": [
            {"properties": {"operation": {"enum": controls}, "target": {"type": "object"}},
             "required": ["target"]},
            {"properties": {"operation": {"enum": guarded}, "target": {"type": "null"},
                            "preconditions": {"minItems": 1}},
             "required": ["preconditions"]},
        ],
    }


def _restrict_action(action):
    """Retain all original properties/defaults/refs and intersect via anyOf."""
    _require(isinstance(action, dict) and action.get("type") == "object"
             and isinstance(action.get("properties"), dict), "original action object schema required")
    _require(not any(key in action for key in ("anyOf", "allOf", "oneOf")),
             "unsupported existing action union; never overwrite it")
    properties = action["properties"]
    _require({"operation", "target", "ref", "box", "preconditions", "postconditions"} <= set(properties),
             "original desktop action fields required")
    operation_order = properties["operation"].get("enum")
    _require(isinstance(operation_order, list) and operation_order
             and all(isinstance(value, str) for value in operation_order),
             "explicit original operations required")
    semantic = _semantic_constraint(operation_order)
    action["anyOf"] = [
        {"properties": {"target": {"type": "null"},
                        "preconditions": {"maxItems": 0}, "postconditions": {"maxItems": 0}}},
        semantic,
    ]
    return semantic


def project_parameters(original_parameters, *, effective_max_actions, format_tag):
    """Copy/project actual H1/H>1 tool parameters only for an explicit new tag.

    Each added branch is an intersection with its unchanged containing schema.
    Missing properties stay optional. The raw branch retains bare keyboard/page
    actions; this helper does not force them to claim guards. Completion omitted
    or [] leaves every other original H1 default alone, including done+steps.
    """
    _require(format_tag == FORMAT, "generation projection requires explicit semantic_tools_v54")
    _require(type(effective_max_actions) is int and 1 <= effective_max_actions <= 12,
             "actual effective desktop horizon required")
    result = copy.deepcopy(original_parameters)
    _require(isinstance(result, dict) and result.get("type") == "object",
             "actual original function parameter object required")
    definitions = result.get("$defs")
    _require(isinstance(definitions, dict), "actual original definitions required")
    if "action" in definitions:
        _restrict_action(definitions["action"])
    # Deliberately do not traverse field_edit/commit or arbitrary object shapes.
    # Current-ref guarded commits have their own legitimate parser exception.
    if effective_max_actions == 1:
        _require(not any(key in result for key in ("anyOf", "allOf", "oneOf")),
                 "unsupported existing H1 root union; never overwrite it")
        properties = result.get("properties")
        _require(isinstance(properties, dict)
                 and {"steps", "done", "completion_conditions"} <= set(properties),
                 "actual H1 plan parameter schema required")
        steps = properties["steps"]
        _require(isinstance(steps, dict) and steps.get("type") == "array" and steps.get("maxItems") == 1,
                 "original single-action steps schema required")
        semantic = _restrict_action(steps.get("items"))
        result["anyOf"] = [
            {"properties": {"completion_conditions": {"maxItems": 0}}},
            {"properties": {"steps": {"minItems": 1, "maxItems": 1, "items": copy.deepcopy(semantic)}},
             "required": ["steps"]},
            {"properties": {"done": {"const": True}, "steps": {"maxItems": 0}},
             "required": ["done", "steps"]},
        ]
    return result


def project_declaration_tools(original_tools, *, effective_max_actions, format_tag):
    """Optional pure list adapter; no tool selection, transport or execution."""
    _require(isinstance(original_tools, list) and original_tools, "actual declaration tools required")
    tools = copy.deepcopy(original_tools)
    for tool in tools:
        _require(isinstance(tool, dict) and tool.get("type") == "function"
                 and isinstance(tool.get("function"), dict), "declaration function envelope required")
        function = tool["function"]
        function["parameters"] = project_parameters(
            function.get("parameters"), effective_max_actions=effective_max_actions, format_tag=format_tag,
        )
    return tools
