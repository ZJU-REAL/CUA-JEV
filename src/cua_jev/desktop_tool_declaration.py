"""Inert desktop plan declarations; no HTTP, tool dispatch or parser replacement."""

import copy
import hashlib
import json

FORMAT = "tool_declaration_v52"
FUNCTION = "declare_desktop_stage"
TOOL_GUIDANCE = (
    "\nDECLARATION OUTPUT TRANSPORT: Return exactly one call to declare_desktop_stage. "
    "Its arguments must be the one JSON plan object described above. This is an inert "
    "declaration, not tool execution. Do not return extra calls or a separate content answer."
)


def _require(ok, code):
    if not ok:
        raise ValueError(code)


def validate_tool_declaration_selection(profile, contract, observation):
    _require(profile == "desktop" and contract == "property_intent_v52"
             and observation == "full_multimodal_v1",
             "tool declaration requires desktop v52 full multimodal planning")


def function_parameters(production_schema):
    """Same redundant-root rule as reviewed probe e570; never modify its input."""
    parameters = copy.deepcopy(production_schema)
    _require(isinstance(parameters, dict), "function parameters must be an object schema")
    if "type" not in parameters:
        branches = parameters.get("anyOf")
        _require(isinstance(branches, list) and bool(branches)
                 and all(isinstance(branch, dict) and branch.get("type") == "object"
                         for branch in branches),
                 "every root anyOf branch must explicitly be object")
        parameters["type"] = "object"
    _require(parameters["type"] == "object", "function parameters root must be object")
    return parameters


def encode_tool_declaration(body, *, profile, contract, observation):
    """Adapt one already-built production request; never make a request itself."""
    validate_tool_declaration_selection(profile, contract, observation)
    encoded = copy.deepcopy(body)
    _require(not any(key in encoded for key in ("tools", "tool_choice", "parallel_tool_calls")),
             "unexpected preexisting tool transport")
    production_format = encoded.pop("response_format", None)
    _require(isinstance(production_format, dict) and production_format.get("type") == "json_schema"
             and isinstance(production_format.get("json_schema"), dict),
             "original desktop schema format required")
    parameters = function_parameters(production_format["json_schema"].get("schema"))
    messages = encoded.get("messages")
    _require(isinstance(messages, list) and bool(messages) and isinstance(messages[0], dict)
             and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str),
             "original system instruction required")
    image_uploaded = any(
        isinstance(message, dict) and isinstance(message.get("content"), list)
        and any(isinstance(block, dict) and block.get("type") == "image_url"
                for block in message["content"])
        for message in messages
    )
    _require(image_uploaded, "full image upload required for declaration transport")
    messages[0]["content"] += TOOL_GUIDANCE
    encoded.update(tools=[{"type": "function", "function": {
        "name": FUNCTION,
        "description": "Declare one desktop plan as data; no function or GUI execution occurs.",
        "parameters": parameters,
    }}], tool_choice="auto", parallel_tool_calls=False)
    restored = {key: copy.deepcopy(value) for key, value in encoded.items()
                if key not in {"tools", "tool_choice", "parallel_tool_calls"}}
    restored["messages"][0]["content"] = restored["messages"][0]["content"].removesuffix(TOOL_GUIDANCE)
    restored["response_format"] = production_format
    _require(restored == body, "unexpected production request change")
    return encoded, {"response_transport": FORMAT, "tool_name": FUNCTION,
                     "image_uploaded": image_uploaded, "function_parameters_root_type": "object",
                     "redundant_root_object_added": "type" not in production_format["json_schema"]["schema"]}


def decode_tool_declaration(payload):
    """Extract data only. Caller must still run unchanged parse_plan and guards."""
    _require(isinstance(payload, dict), "declaration response must be an object")
    choices = payload.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
             "declaration response requires exactly one choice")
    message = choices[0].get("message")
    _require(isinstance(message, dict), "declaration response message missing")
    _require(not message.get("refusal"), "declaration response must not contain a refusal")
    accompanying_content_present = message.get("content") not in (None, "")
    calls = message.get("tool_calls")
    _require(isinstance(calls, list) and len(calls) == 1,
             "declaration response requires exactly one tool call")
    call = calls[0]
    _require(isinstance(call, dict) and call.get("type") == "function"
             and isinstance(call.get("function"), dict), "invalid declaration function envelope")
    function = call["function"]
    _require(function.get("name") == FUNCTION, "unexpected declaration function name")
    arguments = function.get("arguments")
    _require(isinstance(arguments, str) and 0 < len(arguments) <= 64000,
             "declaration arguments must be a bounded JSON string")
    try:
        raw = json.loads(arguments)
    except (ValueError, RecursionError):
        raise ValueError("declaration arguments must be valid JSON") from None
    _require(isinstance(raw, dict), "declaration arguments must be one JSON object")
    return raw, {"response_transport": FORMAT, "tool_name": FUNCTION, "tool_call_count": 1,
                 "accompanying_content_present": accompanying_content_present,
                 "argument_characters": len(arguments),
                 "argument_sha256": hashlib.sha256(arguments.encode()).hexdigest()}
