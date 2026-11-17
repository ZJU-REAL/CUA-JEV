"""Explicit normal-H6 JSON-schema declarations; original strong-H1 transport."""

import copy
import hashlib
import json
from pathlib import Path

from .desktop_phase_handoff_v62 import CONTEXT_TOOL
from .desktop_program_wire_v64 import canonical_json
from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v64 import HIGH_LEVEL_TOOLS, PROGRAM_GUIDANCE
from .desktop_semantic_tools_v64 import decode_semantic_declaration as decode_ancestor
from .desktop_semantic_tools_v64 import encode_semantic_declaration as encode_ancestor

FORMAT = "semantic_json_v66"
SCHEMA_NAME = "cua_jev_declaration_v66"
DECLARATION_TOOLS = (*HIGH_LEVEL_TOOLS, CONTEXT_TOOL)
UNSUPPORTED_KEYWORDS = frozenset(
    {
        "uniqueItems",
        "contains",
        "minContains",
        "maxContains",
        "additionalItems",
        "multipleOf",
    }
)
MAX_CONTENT_BYTES = 64_000
_KINDS = {
    dict: "object",
    list: "array",
    str: "string",
    bool: "boolean",
    int: "number",
    float: "number",
    type(None): "null",
}
_SEMANTIC_FIELDS = (
    "tool_name",
    "mapped_object_sha256",
    "mapping_applied",
    "mapping_strategy",
    "program_declaration",
    "program_declaration_sha256",
    "stage_count",
    "future_compiled_steps",
    "original_parser_required",
    "original_parser_required_per_activation",
    "shared_original_horizon",
    "mapped_intent",
    "final_original_fresh_whole_goal_review_required",
    "context_request",
    "declaration_sha256",
    "wire_declaration_json",
    "wire_declaration_sha256",
    "normalized_declaration_json",
    "normalized_declaration_sha256",
    "normalized_argument_sha256",
    "wire_stage_mapping",
    "codec_version",
    "codec_source_hashes",
)
_REQUEST_BINDING_FIELDS = (
    "response_format_json",
    "response_format_sha256",
    "schema_name",
    "declaration_schema_sha256",
    "parameter_schema_sha256",
    "offered_tool_names",
    "content_codec_source_hashes",
    "provider_schema_enforcement",
    "schema_namespace",
)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_hashes():
    return {
        "src/cua_jev/desktop_semantic_json_v66.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }


def _schema_projection(value, *, prefix, removed, path=""):
    """Keep each parameter root in its own namespace; rewrite EVERY local ref."""
    if type(value) is list:
        return [
            _schema_projection(item, prefix=prefix, removed=removed, path=f"{path}/{index}")
            for index, item in enumerate(value)
        ]
    if type(value) is not dict:
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        current = path + "/" + key
        if key in UNSUPPORTED_KEYWORDS:
            removed.append({"path": current, "keyword": key})
        elif key == "$schema":
            continue  # Dialect marker, not a generation constraint.
        elif key == "$ref":
            _require(
                type(item) is str and (item == "#" or item.startswith("#/")),
                "structured_schema_local_pointer_required",
            )
            result[key] = prefix + item[1:]
        elif key == "$id":
            raise ValueError("structured_schema_id_not_supported")
        elif key in {"const", "enum", "default", "examples"}:
            result[key] = copy.deepcopy(item)  # Instance data must not be rewritten as schema.
        elif key in {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}:
            _require(type(item) is dict, "structured_schema_mapping_required")
            result[key] = {
                name: _schema_projection(child, prefix=prefix, removed=removed, path=current + "/" + name)
                for name, child in item.items()
            }
        else:
            result[key] = _schema_projection(item, prefix=prefix, removed=removed, path=current)
    return result


def declaration_schema(tools):
    _require(
        type(tools) is list and [item["function"]["name"] for item in tools] == list(DECLARATION_TOOLS),
        "structured_actual_six_tools_required",
    )
    definitions, branches, parameters, removed = {}, [], {}, []
    for index, tool in enumerate(tools):
        function = tool["function"]
        name, key = function["name"], f"v66_tool_{index}_arguments"
        projected = _schema_projection(
            function["parameters"], prefix="#/$defs/" + key, removed=removed, path="/$defs/" + key
        )
        definitions[key] = projected
        parameters[name] = _sha(canonical_json(projected))
        branches.append(
            {
                "type": "object",
                "properties": {
                    "tool": {"const": name},
                    "arguments": {"$ref": "#/$defs/" + key},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
        )
    return (
        {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(DECLARATION_TOOLS)},
                "arguments": {"type": "object"},
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
            "oneOf": branches,
            "$defs": definitions,
        },
        parameters,
        removed,
    )


def _pool_schema_definitions(schema):
    """Share only exact dependency-closed copies with the untouched p0 catalog.

    This is limited to the six-root schema emitted above, not a general schema
    optimizer. Unsupported scope, references, layout or cycles retain the full
    original layout. Generation behavior under a different layout is unverified.
    """
    original = copy.deepcopy(schema)
    original_json = canonical_json(original)
    metadata = {
        "version": "p0_dependency_closed_def_pool_v67",
        "anchor_root": "v66_tool_0_arguments",
        "applied": False,
        "status": "unchanged_unsupported",
        "mappings": [],
        "removed_definition_count": 0,
        "original_schema_sha256": _sha(original_json),
        "pooled_schema_sha256": _sha(original_json),
        "original_schema_bytes": len(original_json.encode("utf-8")),
        "pooled_schema_bytes": len(original_json.encode("utf-8")),
        "provider_generation_equivalence": "unverified",
    }

    def supported(condition):
        if not condition:
            raise ValueError("unsupported_schema_pooling_layout")

    names = [f"v66_tool_{index}_arguments" for index in range(6)]
    keywords = {
        "$ref",
        "$defs",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "enum",
        "const",
        "default",
        "examples",
        "description",
        "title",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minProperties",
        "maxProperties",
        "format",
    }
    literals = {"const", "enum", "default", "examples"}
    try:
        supported(
            type(original) is dict
            and set(original)
            == {
                "type",
                "properties",
                "required",
                "additionalProperties",
                "oneOf",
                "$defs",
            }
        )
        supported(
            original["type"] == "object"
            and original["additionalProperties"] is False
            and original["required"] == ["tool", "arguments"]
            and original["properties"]
            == {
                "tool": {"type": "string", "enum": list(DECLARATION_TOOLS)},
                "arguments": {"type": "object"},
            }
        )
        roots = original["$defs"]
        supported(
            type(roots) is dict
            and set(roots) == set(names)
            and type(original["oneOf"]) is list
            and len(original["oneOf"]) == 6
        )
        nodes = {}
        for index, name in enumerate(names):
            supported(
                original["oneOf"][index]
                == {
                    "type": "object",
                    "properties": {
                        "tool": {"const": DECLARATION_TOOLS[index]},
                        "arguments": {"$ref": "#/$defs/" + name},
                    },
                    "required": ["tool", "arguments"],
                    "additionalProperties": False,
                }
            )
            argument = roots[name]
            supported(type(argument) is dict)
            nodes["#/$defs/" + name] = argument
            catalog = argument.get("$defs", {})
            supported(type(catalog) is dict)
            for key, value in catalog.items():
                supported(
                    type(key) is str
                    and key.isascii()
                    and bool(key)
                    and all(char.isalnum() or char == "_" for char in key)
                )
                nodes["#/$defs/" + name + "/$defs/" + key] = value
        anchor = roots[names[0]]["$defs"]
        supported(len([key for key in anchor if key.startswith("program_stage_arguments_")]) == 6)

        def references(value, *, catalog_allowed=False):
            if type(value) is bool:
                return set()
            supported(type(value) is dict and set(value) <= keywords)
            found = set()
            for key, item in value.items():
                if key in literals:
                    continue
                if key == "$ref":
                    supported(type(item) is str and item in nodes)
                    found.add(item)
                elif key in {"properties", "$defs"}:
                    supported(type(item) is dict and (key != "$defs" or catalog_allowed))
                    for child in item.values():
                        found.update(references(child))
                elif key in {"oneOf", "anyOf", "allOf", "prefixItems"}:
                    supported(type(item) is list)
                    for child in item:
                        found.update(references(child))
                elif key in {"not", "if", "then", "else", "items", "additionalProperties"}:
                    found.update(references(item))
            return found

        graph = {ref: references(node, catalog_allowed=ref.count("/") == 2) for ref, node in nodes.items()}
        state = {}

        def acyclic(ref):
            supported(state.get(ref) != 1)
            if state.get(ref) == 2:
                return
            state[ref] = 1
            for target in graph[ref]:
                acyclic(target)
            state[ref] = 2

        for ref in graph:
            acyclic(ref)
        mapping = {
            "#/$defs/" + name + "/$defs/" + key: "#/$defs/" + names[0] + "/$defs/" + key
            for name in names[1:]
            for key in roots[name].get("$defs", {})
            if key in anchor
        }

        def rewrite(value):
            if type(value) is list:
                return [rewrite(item) for item in value]
            if type(value) is not dict:
                return copy.deepcopy(value)
            result = {}
            for key, item in value.items():
                if key in literals:
                    result[key] = copy.deepcopy(item)
                elif key == "$ref":
                    result[key] = mapping.get(item, item)
                elif key in {"properties", "$defs"}:
                    result[key] = {name: rewrite(child) for name, child in item.items()}
                else:
                    result[key] = rewrite(item)
            return result

        while True:
            rejected = [
                source
                for source, target in mapping.items()
                if canonical_json(rewrite(nodes[source])) != canonical_json(nodes[target])
            ]
            if not rejected:
                break
            for source in rejected:
                del mapping[source]
        if not mapping:
            metadata["status"] = "unchanged_no_duplicates"
            return original, metadata
        pooled = rewrite(original)
        for source in mapping:
            parts = source[2:].split("/")
            del pooled["$defs"][parts[1]]["$defs"][parts[3]]
        supported(pooled["$defs"][names[0]] == original["$defs"][names[0]])
        pooled_json = canonical_json(pooled)
        metadata.update(
            applied=True,
            status="pooled",
            removed_definition_count=len(mapping),
            mappings=[{"source_ref": source, "anchor_ref": mapping[source]} for source in sorted(mapping)],
            pooled_schema_sha256=_sha(pooled_json),
            pooled_schema_bytes=len(pooled_json.encode("utf-8")),
        )
        return pooled, metadata
    except (ValueError, KeyError, TypeError, RecursionError):
        return original, metadata


def encode_semantic_declaration(
    body,
    *,
    max_actions,
    profile,
    contract,
    observation,
    visual_box_space="normalized_1000",
    normal_program_eligible=True,
):
    encoded, previous = encode_ancestor(
        body,
        max_actions=max_actions,
        profile=profile,
        contract=contract,
        observation=observation,
        visual_box_space=visual_box_space,
        normal_program_eligible=normal_program_eligible,
    )
    normal = max_actions > 1 and normal_program_eligible
    if not normal:
        return encoded, {
            **previous,
            "response_transport": FORMAT,
            "requested_format": FORMAT,
            "normal_json_schema": False,
            "transport_kind": "original_function_tools",
            "phase": "original_current_or_review",
        }
    schema, parameters, removed = declaration_schema(encoded["tools"])
    schema, pooling = _pool_schema_definitions(schema)
    parameters = {
        name: _sha(canonical_json(schema["$defs"][f"v66_tool_{index}_arguments"]))
        for index, name in enumerate(DECLARATION_TOOLS)
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": SCHEMA_NAME,
            "strict": True,
            "schema": schema,
        },
    }
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        del encoded[key]
    _require("response_format" not in encoded, "structured_template_already_has_response_format")
    encoded["response_format"] = response_format
    first = PROGRAM_GUIDANCE.split("\n", 1)[0]
    system = encoded["messages"][0]["content"]
    _require(system.startswith(first), "structured_original_clean_phase_prefix_required")
    encoded["messages"][0]["content"] = (
        "You plan the user's COMPLETE requested desktop result; Jev selects fresh legal controls under "
        "the original guards. Return ONLY one entire JSON object with exactly two keys: tool and "
        "arguments. tool is ONE of the six names in the response schema; arguments is that named "
        "declaration's object. No function/tool call, prose, Markdown fence, extra object, intent "
        "wrapper or executor payload. The JSON is inert declaration data, not execution."
        + system[len(first) :]
    )
    return encoded, {
        "response_transport": FORMAT,
        "requested_format": FORMAT,
        "generation_projection_format": FORMAT,
        "phase": "normal_high_level",
        "normal_high_level_phase": True,
        "normal_json_schema": True,
        "normal_program_eligible": True,
        "declaration_tool_count": 6,
        "offered_tool_names": list(DECLARATION_TOOLS),
        "response_format_json": canonical_json(response_format),
        "response_format_sha256": _sha(canonical_json(response_format)),
        "schema_name": SCHEMA_NAME,
        "schema_namespace": "per_offered_tool_parameter_root_v66",
        "schema_pooling": pooling,
        "declaration_schema_sha256": _sha(canonical_json(schema)),
        "parameter_schema_sha256": parameters,
        "request_body_sha256": _digest(encoded),
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "content_codec_source_hashes": _source_hashes(),
        "provider_schema_enforcement": "unverified",
        "http_tools_present": False,
        "http_tool_choice_present": False,
        "http_parallel_tool_calls_present": False,
        "removed_unsupported_schema_keywords": removed,
        "source_variant": "v67_restricted_p0_schema_pooling",
        "internal_template": {
            "kind": "function_schema_template_not_actual_HTTP_body",
            "provenance": previous,
        },
        "input_authority": False,
    }


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "structured_duplicate_json_key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("structured_content_nonfinite_json")


def _content_object(content):
    _require(type(content) is str, "structured_content_string_required")
    _require(0 < len(content) <= MAX_CONTENT_BYTES, "structured_content_bound")
    try:
        encoded = content.encode("utf-8")
    except UnicodeError:
        raise ValueError("structured_content_invalid_utf8") from None
    _require(0 < len(encoded) <= MAX_CONTENT_BYTES, "structured_content_bound")
    try:
        raw = json.loads(content, object_pairs_hook=_unique, parse_constant=_nonfinite)
    except ValueError as error:
        if str(error) in {"structured_duplicate_json_key", "structured_content_nonfinite_json"}:
            raise
        raise ValueError("structured_content_invalid_json") from None
    except (RecursionError, TypeError, UnicodeError):
        raise ValueError("structured_content_invalid_json") from None
    try:
        canonical_json(raw).encode("utf-8")
    except UnicodeError:
        raise ValueError("structured_content_invalid_utf8") from None
    except (ValueError, RecursionError, TypeError):
        raise ValueError("structured_content_nonfinite_json") from None
    return raw, encoded


def _message(payload):
    _require(type(payload) is dict, "structured_response_object_required")
    _require(payload.get("error") is None, "structured_response_error")
    choices = payload.get("choices")
    _require(
        type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
        "structured_one_choice_required",
    )
    choice = choices[0]
    _require(choice.get("finish_reason") == "stop", "structured_finish_stop_required")
    message = choice.get("message")
    _require(type(message) is dict, "structured_message_object_required")
    _require(message.get("refusal") is None or message.get("refusal") == "", "structured_refusal")
    calls = message.get("tool_calls")
    _require(calls is None or (type(calls) is list and not calls), "structured_provider_tool_calls_forbidden")
    _require(message.get("function_call") is None, "structured_legacy_function_call_forbidden")
    return message


def _request_binding(provenance, offered):
    _require(
        type(provenance) is dict
        and provenance.get("response_transport") == FORMAT
        and provenance.get("normal_json_schema") is True,
        "structured_request_provenance_required",
    )
    names = provenance.get("offered_tool_names")
    actual = names if offered is None else list(offered) if type(offered) in {list, tuple} else None
    _require(names == list(DECLARATION_TOOLS) and actual == names, "structured_actual_tool_mask_required")
    text = provenance.get("response_format_json")
    _require(type(text) is str, "structured_response_schema_provenance_required")
    response_format = json.loads(text)
    _require(
        canonical_json(response_format) == text
        and _sha(text) == provenance.get("response_format_sha256")
        and response_format.get("type") == "json_schema"
        and provenance.get("schema_name") == SCHEMA_NAME
        and response_format["json_schema"].get("name") == SCHEMA_NAME
        and response_format["json_schema"].get("strict") is True
        and response_format["json_schema"]["schema"]["properties"]["tool"]["enum"] == names
        and _sha(canonical_json(response_format["json_schema"]["schema"]))
        == provenance.get("declaration_schema_sha256")
        and provenance.get("content_codec_source_hashes") == _source_hashes(),
        "structured_response_schema_provenance_mismatch",
    )
    return {
        **{key: copy.deepcopy(provenance[key]) for key in _REQUEST_BINDING_FIELDS},
        **(
            {"schema_pooling": copy.deepcopy(provenance["schema_pooling"])}
            if "schema_pooling" in provenance else {}
        ),
    }


def decode_semantic_declaration(
    payload,
    *,
    max_actions,
    prepare_repair_token=None,
    normal_program_eligible=True,
    offered_tool_names=None,
    request_provenance=None,
):
    _require(type(normal_program_eligible) is bool, "normal phase eligibility must be boolean")
    _require(type(max_actions) is int and 1 <= max_actions <= 12, "original horizon required")
    normal = max_actions > 1 and normal_program_eligible
    if not normal:
        mapped, previous = decode_ancestor(
            payload,
            max_actions=max_actions,
            prepare_repair_token=prepare_repair_token,
            normal_program_eligible=normal_program_eligible,
        )
        return mapped, {
            **previous,
            "response_transport": FORMAT,
            "requested_format": FORMAT,
            "phase": "original_current_or_review",
            "normal_json_schema": False,
            "transport_kind": "original_function_tools",
            "provider_tool_call_count": previous["tool_call_count"],
        }
    _require(prepare_repair_token is None, "normal phase cannot be a preparation repair")
    binding = _request_binding(request_provenance, offered_tool_names)
    message = _message(payload)
    raw, content_bytes = _content_object(message.get("content"))
    _require(type(raw) is dict and set(raw) == {"tool", "arguments"}, "structured_declaration_exact_fields")
    tool, arguments = raw["tool"], raw["arguments"]
    _require(type(tool) is str and tool in binding["offered_tool_names"], "structured_unoffered_tool")
    _require(type(arguments) is dict, "structured_arguments_object_required")
    argument_json = canonical_json(arguments)
    internal = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": argument_json,
                            },
                        }
                    ]
                }
            }
        ]
    }
    mapped, previous = decode_ancestor(internal, max_actions=max_actions, normal_program_eligible=True)
    return mapped, {
        **{key: copy.deepcopy(previous[key]) for key in _SEMANTIC_FIELDS if key in previous},
        **binding,
        "response_transport": FORMAT,
        "requested_format": FORMAT,
        "phase": "normal_high_level",
        "normal_high_level_phase": True,
        "normal_json_schema": True,
        "transport_kind": "json_schema_content",
        "provider_tool_call_count": 0,
        "tool_call_count": 0,
        "selected_tool": tool,
        "tool_name": tool,
        "content_declaration_json": canonical_json(raw),
        "content_declaration_sha256": _sha(canonical_json(raw)),
        "observed_content_utf8_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "content_utf8_bytes": len(content_bytes),
        "canonical_arguments_json": argument_json,
        "canonical_arguments_sha256": _sha(argument_json),
        "argument_sha256": _sha(argument_json),
        "argument_hash_kind": "adapter_canonical_arguments",
        "original_parser_required": True,
        "internal_adapter": {"kind": "synthetic_legacy_mapper_payload_not_provider", "provenance": previous},
        "input_authority": False,
    }


def structured_response_facts(payload):
    """Bounded actual shape/parse facts; never emit content, unknown names or argument values."""
    result = {
        "version": "bounded_structured_content_response_v66",
        "untrusted": True,
        "input_authority": False,
        "payload_type": _KINDS.get(type(payload), "non_json_type"),
    }
    if type(payload) is not dict:
        return result
    result["error_type"] = _KINDS.get(type(payload.get("error")), "non_json_type")
    result["error_present"] = "error" in payload
    choices = payload.get("choices")
    result["choices_count_capped"] = min(len(choices), 4) if type(choices) is list else 0
    result["choices_overflow"] = type(choices) is list and len(choices) > 4
    result["choices"] = []
    for choice in choices[:4] if type(choices) is list else []:
        fact = {"type": _KINDS.get(type(choice), "non_json_type")}
        if type(choice) is dict:
            reason = choice.get("finish_reason")
            fact["finish_reason"] = (
                reason
                if type(reason) is str and reason in {"stop", "length", "tool_calls", "content_filter"}
                else "other"
            )
            message = choice.get("message")
            fact["message_type"] = _KINDS.get(type(message), "non_json_type")
            if type(message) is dict:
                content = message.get("content")
                fact["content_present"] = "content" in message
                fact["content_type"] = _KINDS.get(type(content), "non_json_type")
                fact["tool_calls_present"] = "tool_calls" in message
                calls = message.get("tool_calls")
                fact["tool_calls_type"] = _KINDS.get(type(calls), "non_json_type")
                fact["tool_calls_count_capped"] = min(len(calls), 4) if type(calls) is list else 0
                fact["function_call_present"] = "function_call" in message
                fact["function_call_type"] = _KINDS.get(type(message.get("function_call")), "non_json_type")
                fact["refusal_type"] = _KINDS.get(type(message.get("refusal")), "non_json_type")
                fact["refusal_present"] = "refusal" in message
                if type(content) is str:
                    count = (
                        MAX_CONTENT_BYTES + 1
                        if len(content) > MAX_CONTENT_BYTES
                        else len(content.encode("utf-8", errors="replace"))
                    )
                    fact["content_bytes_bucket"] = next(
                        (
                            "le_" + str(limit)
                            for limit in (0, 64, 256, 1024, 4096, 16384, MAX_CONTENT_BYTES)
                            if count <= limit
                        ),
                        "over_bound",
                    )
                    try:
                        raw, _encoded = _content_object(content)
                        fact["content_json_status"] = (
                            "finite_object" if type(raw) is dict else "finite_non_object"
                        )
                        if type(raw) is dict:
                            fact["known_root_keys"] = sorted(set(raw) & {"tool", "arguments"})
                            fact["extra_root_keys_present"] = bool(set(raw) - {"tool", "arguments"})
                            tool = raw.get("tool")
                            fact["selected_tool"] = (
                                tool if type(tool) is str and tool in DECLARATION_TOOLS else "unoffered"
                            )
                            fact["arguments_type"] = _KINDS.get(type(raw.get("arguments")), "non_json_type")
                    except (ValueError, TypeError, RecursionError) as error:
                        code = str(error)
                        fact["content_json_status"] = (
                            code
                            if code
                            in {
                                "structured_content_bound",
                                "structured_content_invalid_utf8",
                                "structured_content_invalid_json",
                                "structured_duplicate_json_key",
                                "structured_content_nonfinite_json",
                            }
                            else "unavailable"
                        )
        result["choices"].append(fact)
    return result
