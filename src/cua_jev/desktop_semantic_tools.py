"""Flat semantic declarations; no HTTP, function execution or parser authority."""

import copy
import hashlib
import json

FORMAT = "semantic_tools_v53"
FUNCTION_INTENTS = {
    "declare_property_stage": "pursue_property",
    "declare_navigation_stage": "navigate",
    "declare_field_edits_stage": "edit_fields",
    "declare_current_action": "act_current",
    "declare_completion": "complete",
}
PROPERTY_REQUIRED = {
    "label", "scope", "edit_operation", "numeric_finalize_key", "confirmation_label", "completion",
}
PROPERTY_OPTIONAL = {"edit_value", "prepare", "goal_conditions"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_wire(value)).hexdigest()


def _reject_nonfinite_json(_value):
    raise ValueError("non-finite JSON is not permitted")


def referenced_definitions(schema, definitions):
    """Keep only original transitive local refs; never rewrite or expand a definition."""
    def refs(value):
        pending, found = [value], set()
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                if "$ref" in item:
                    ref = item["$ref"]
                    _require(isinstance(ref, str) and ref.startswith("#/$defs/")
                             and ref[8:] in definitions, "unknown or nonlocal schema reference")
                    found.add(ref[8:])
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return found

    pending, found = list(refs(schema)), set()
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(refs(definitions[name]) - found)
    return {name: copy.deepcopy(value) for name, value in definitions.items() if name in found}


def semantic_function_schemas(root):
    _require(isinstance(root, dict) and root.get("type") == "object"
             and isinstance(root.get("anyOf"), list) and len(root["anyOf"]) == 5
             and isinstance(root.get("$defs"), dict), "original52 five-object schema required")
    by_intent = {}
    for branch in root["anyOf"]:
        marker = branch["properties"]["intent"]
        _require(set(marker) == {"const"} and marker["const"] in FUNCTION_INTENTS.values()
                 and marker["const"] not in by_intent, "unique original intent required")
        parameters = copy.deepcopy(branch)
        parameters["properties"].pop("intent")
        parameters["required"].remove("intent")
        if marker["const"] == "pursue_property":
            properties = parameters["properties"]
            edit = root["$defs"]["literal_edit"]["anyOf"][0]
            numeric = properties.pop("numeric_finalize")
            confirmation = properties.pop("confirm")["anyOf"][0]
            properties.pop("edit")
            properties.update(
                edit_operation={"enum": ["fill", "set", "check", "uncheck"]},
                edit_value=copy.deepcopy(edit["properties"]["value"]),
                numeric_finalize_key=copy.deepcopy(numeric["properties"]["key"]),
                confirmation_label=copy.deepcopy(confirmation["properties"]["label"]),
            )
            parameters["required"] = [key for key in parameters["required"]
                                      if key not in {"edit", "numeric_finalize", "confirm"}]
            parameters["required"] += ["edit_operation", "numeric_finalize_key", "confirmation_label"]
            parameters["anyOf"] = [
                {"properties": {"edit_operation": {"enum": ["fill", "set"]}}, "required": ["edit_value"]},
                {"properties": {"edit_operation": {"enum": ["check", "uncheck"]}},
                 "not": {"required": ["edit_value"]}},
            ]
        parameters["$defs"] = referenced_definitions(parameters, root["$defs"])
        parameters["$schema"] = root["$schema"]
        by_intent[marker["const"]] = parameters
    return {name: by_intent[intent] for name, intent in FUNCTION_INTENTS.items()}


def semantic_stage_instructions(limit, *, visual_box_space="normalized_1000"):
    from .desktop_property_protocol import _shared

    _require(type(limit) is int and 2 <= limit <= 12, "original multi-action horizon required")
    return (
        "Return exactly ONE inert declaration call, not an action execution. The function name supplies "
        "the stage; arguments MUST NOT contain intent. Functions: declare_property_stage, "
        "declare_navigation_stage, declare_field_edits_stage, declare_current_action, declare_completion. "
        "PRIORITIZE one property declaration for an expressible requested property/text result, covering "
        "navigation/edit/required submission rather than separate menu calls. Otherwise prefer reusable "
        "navigation/current-field edits. Use a current action only when no reusable stage fits; do not force "
        "property on unsuitable tasks. Scopes come from current navigation_scopes, not future "
        "windows/fields.\n"
        "Complete text example: call declare_property_stage with arguments "
        '{"label":"Apply requested text","scope":{"app":"CURRENT_APP"},'
        '"prepare":{"operation":"click","ref":"CURRENT_REF"},"edit_operation":"fill",'
        '"edit_value":"FINAL_TEXT","numeric_finalize_key":"none",'
        '"confirmation_label":"Submit requested text change","completion":"goal"}. '
        "Placeholders are syntax, never grounding. Replace with actual current evidence and requested data. "
        "Property uses label,scope,edit_operation,numeric_finalize_key,confirmation_label,completion; "
        "prepare/goal_conditions optional. fill/set REQUIRE edit_value; check/uncheck MUST OMIT it, even "
        "an empty value. No edit/confirm/numeric_finalize wrappers or extra fields. Creation/renaming is "
        "fill(final text), not selecting an existing same-name option. set supports real "
        "fields/options/on-off. "
        "Nonnumeric text/path/on-off/check/uncheck use numeric_finalize_key=none. Never repair an invalid "
        "strategy by silently changing its value.\n"
        "Numeric enter/tab is optional strategy, not text/form submission: only bounded numeric fill/set "
        "may declare it, reserving ONE original action. Declare this contingency before its field appears; "
        "the key is sent only if the later native checks hold. Offer that ONE key only after an actual "
        "unverified "
        "fill receipt on the SAME unique focused native numeric field with matching pending edit buffer "
        "and different committed value. Fresh recheck owner/field/focus/buffer immediately before key; "
        "verify actual committed value afterward, retaining original confirm/goal checks. A bound field "
        "later becoming unknown/closed needs VLM. No guessed key, implicit refill, buffer-as-persistence or "
        "TTL/time/action extension.\n"
        "confirmation_label is exact none for NO confirmation input; otherwise explicitly name the "
        "confirmation intention with a nonempty label. It is not a button/ref/key. Declare required "
        "submission "
        "before the form opens. Jev selects only fresh legal buttons in the receipt-verified field's unique "
        "observed owner with persistent field/result/closure guards. Unknown owner needs VLM, never "
        "guessing. "
        "A current prepare is one action object, operation direct, no wrapper/id. Retain the property stage; "
        "for search, prepare value is query and edit_value is final data. Bind the real field after "
        "preparation. Native preparation requires that control's listed operation; otherwise ground a "
        "visual action in the uploaded image. "
        "Visual input including prepare requires the actual uploaded current image; no future coordinates.\n"
        "declare_navigation_stage arguments: label,scope,arrival,completion. Arrival must be a meaningful "
        "unmet observable final view/state. declare_field_edits_stage arguments: label,edits,completion and "
        "optional explicit commit, using the original field/action/condition grammar. edits=[] still "
        "requires "
        "a current commit guarded by persistent field state. Resolve current refs uniquely; preserve owner "
        "and actual closure evidence. Property/navigation/field edits require completion=stage|goal: goal "
        "finishes the ENTIRE task; stage leaves work. Nonempty goal_conditions remain binding. A field, "
        "click or open form alone is not persistence. declare_current_action arguments: label,action for "
        "one current grounded action. declare_completion arguments: label and optional observable "
        "conditions, "
        "only if the whole task is already finished. A declaration never proves success. "
        f"At most {limit} logical actions under the original budgets; tool_choice is auto, no function "
        "executes and no tool results or internal second request follow.\n"
        + _shared(visual_box_space)
    )


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000"):
    from .desktop_output_schema import desktop_response_format
    from .desktop_property_protocol import completion_policy_property_protocol_instructions
    from .desktop_tool_declaration import encode_tool_declaration, validate_tool_declaration_selection

    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    if max_actions == 1:
        return encode_tool_declaration(body, profile=profile, contract=contract, observation=observation)
    validate_tool_declaration_selection(profile, contract, observation)
    expected = desktop_response_format(
        max_actions, property_contract="property_intent_v52", image_uploaded=True,
    )
    _require(body.get("response_format") == expected, "actual original52 response_format required")
    _require(not any(k in body for k in ("tools", "tool_choice", "parallel_tool_calls")),
             "preexisting tool wire")
    encoded = copy.deepcopy(body)
    parameters = semantic_function_schemas(encoded.pop("response_format")["json_schema"]["schema"])
    messages = encoded.get("messages")
    _require(isinstance(messages, list) and messages and isinstance(messages[0], dict)
             and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str),
             "original system instruction required")
    uploaded = any(isinstance(m, dict) and isinstance(m.get("content"), list)
                   and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m["content"])
                   for m in messages)
    _require(uploaded, "flat semantic tools require an actual uploaded image")
    original_system = messages[0]["content"]
    original_base = completion_policy_property_protocol_instructions(
        max_actions, visual_box_space=visual_box_space, image_uploaded=True)
    _require(original_system.startswith(original_base), "original52 instruction base changed")
    messages[0]["content"] = semantic_stage_instructions(max_actions, visual_box_space=visual_box_space)
    messages[0]["content"] += original_system[len(original_base):]  # Original encoding/geometry guidance.
    tools = [{"type": "function", "function": {
        "name": name, "description": "Declare " + intent + " as data; no function or GUI execution.",
        "parameters": parameters[name],
    }} for name, intent in FUNCTION_INTENTS.items()]
    encoded.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
    restored = {k: copy.deepcopy(v) for k, v in encoded.items()
                if k not in {"tools", "tool_choice", "parallel_tool_calls"}}
    restored["messages"][0]["content"], restored["response_format"] = original_system, expected
    _require(restored == body, "unexpected original request mutation")
    return encoded, {"response_transport": FORMAT, "image_uploaded": True,
                     "original_schema_sha256": _digest(expected),
                     "original_system_sha256": _digest(original_system),
                     "projected_system_sha256": _digest(messages[0]["content"]),
                     "parameter_schema_sha256": {n: _digest(s) for n, s in parameters.items()}}


def _property_from_flat(raw):
    _require(PROPERTY_REQUIRED <= set(raw) and set(raw) <= PROPERTY_REQUIRED | PROPERTY_OPTIONAL,
             "flat property requires only its declared fields")
    op = raw["edit_operation"]
    _require(isinstance(op, str) and op in {"fill", "set", "check", "uncheck"}, "invalid flat edit operation")
    _require(("edit_value" in raw) == (op in {"fill", "set"}), "flat edit operation/value mismatch")
    if "edit_value" in raw:
        value = raw["edit_value"]
        _require(isinstance(value, str) and 0 < len(value) <= 2000
                 and value.isascii() and "\x00" not in value,
                 "flat edit value must be a bounded literal ASCII string")
    for key in ("label", "confirmation_label"):
        _require(isinstance(raw[key], str) and bool(raw[key].strip()) and len(raw[key]) <= 240,
                 "flat declaration label must be a bounded nonempty string")
    _require(isinstance(raw["numeric_finalize_key"], str)
             and raw["numeric_finalize_key"] in {"enter", "tab", "none"},
             "invalid explicit numeric strategy")
    _require(isinstance(raw["completion"], str) and raw["completion"] in {"stage", "goal"},
             "invalid completion")
    scope = raw["scope"]
    _require(isinstance(scope, dict) and set(scope) <= {"app", "window", "ancestor"}
             and all(isinstance(v, str) and len(v) <= 240 for v in scope.values())
             and any(scope.get(k, "").strip() for k in ("app", "window")), "invalid flat property scope")
    _require("prepare" not in raw or isinstance(raw["prepare"], dict), "prepare must remain an action object")
    _require("goal_conditions" not in raw or (isinstance(raw["goal_conditions"], list)
             and 1 <= len(raw["goal_conditions"]) <= 8
             and all(isinstance(v, dict) for v in raw["goal_conditions"])),
             "goal conditions must remain original condition objects")
    result = {k: copy.deepcopy(raw[k])
              for k in ("label", "scope", "completion", "prepare", "goal_conditions") if k in raw}
    result["edit"] = {"operation": op, **({"value": raw["edit_value"]} if "edit_value" in raw else {})}
    result["numeric_finalize"] = {"key": raw["numeric_finalize_key"]}
    result["confirm"] = (
        "none" if raw["confirmation_label"] == "none" else {"label": raw["confirmation_label"]}
    )
    return result


def decode_semantic_declaration(payload, *, max_actions):
    from .desktop_tool_declaration import decode_tool_declaration

    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    if max_actions == 1:
        return decode_tool_declaration(payload)
    _require(isinstance(payload, dict), "declaration response must be an object")
    choices = payload.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
             "one choice required")
    message = choices[0].get("message")
    _require(isinstance(message, dict) and not message.get("refusal"), "missing message or refusal")
    calls = message.get("tool_calls")
    _require(isinstance(calls, list) and len(calls) == 1, "one function required")
    call = calls[0]
    _require(isinstance(call, dict) and call.get("type") == "function"
             and isinstance(call.get("function"), dict), "invalid function envelope")
    function = call["function"]
    name = function.get("name")
    _require(isinstance(name, str) and name in FUNCTION_INTENTS, "unknown declaration function")
    arguments = function.get("arguments")
    _require(isinstance(arguments, str) and 0 < len(arguments) <= 64000, "bounded JSON arguments required")
    try:
        raw = json.loads(arguments, parse_constant=_reject_nonfinite_json)
    except (ValueError, RecursionError):
        raise ValueError("invalid argument JSON") from None
    _require(isinstance(raw, dict) and "intent" not in raw, "object arguments without intent required")
    intent = FUNCTION_INTENTS[name]
    mapped = {"intent": intent, **(_property_from_flat(raw) if intent == "pursue_property" else raw)}
    return mapped, {"response_transport": FORMAT, "tool_name": name, "mapped_intent": intent,
                    "tool_call_count": 1,
                    "accompanying_content_present": message.get("content") not in (None, ""),
                    "argument_characters": len(arguments),
                    "argument_sha256": hashlib.sha256(arguments.encode()).hexdigest(),
                    "mapped_object_sha256": _digest(mapped), "input_authority": False}
