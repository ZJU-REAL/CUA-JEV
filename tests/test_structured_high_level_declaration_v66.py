"""Explicit structured-content wire, no provider-function or input evidence fabrication."""

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_clean_high_level_phase_v65 import property_result
from test_program_wire_v64 import actual_input, wire_declaration
from test_semantic_tools_v57 import FIELD, current, packet
from test_semantic_tools_v57 import observed as observed
from test_stage_program_contract_v60 import navigation

from cua_jev import desktop_semantic_json_v66 as new
from cua_jev.desktop_phase_handoff_v62 import HighLevelContextRequest
from cua_jev.desktop_program_wire_v64 import bind_program_provenance, canonical_json
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_stage_program import parse_stage_program

PRIVATE = "PRIVATE_LITERAL_OR_HEADER_MUST_NOT_LEAK_6619"


def response(raw=None, *, text=None):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(raw) if text is None else text,
                },
            }
        ]
    }


@pytest.fixture
def offered(observed, monkeypatch):
    body, options = actual_input(observed, monkeypatch)
    original, _ = new.encode_ancestor(body, **options)
    encoded, provenance = new.encode_semantic_declaration(body, **options)
    return original, encoded, provenance


def decode(payload, offered):
    return new.decode_semantic_declaration(
        payload,
        max_actions=6,
        request_provenance=offered[2],
        offered_tool_names=offered[2]["offered_tool_names"],
    )


def declarations():
    return [
        ("declare_stage_program", wire_declaration()),
        ("declare_property_stage", property_result()),
        ("declare_navigation_stage", navigation()["arguments"]),
        (
            "declare_field_edits_stage",
            {
                "label": "Set observed field",
                "completion": "stage",
                "edits": [
                    {"operation": "fill", "target": FIELD, "value": "requested literal"},
                ],
            },
        ),
        (
            "declare_completion",
            {
                "label": "Observed editor",
                "conditions": [
                    {"kind": "text_contains", "value": "Editor"},
                ],
            },
        ),
        ("declare_current_context_needed", {"reason": "missing_future_semantics"}),
    ]


def test_actual_generation_is_strict_closed_six_way_schema_with_every_local_ref_resolved(offered):
    original, body, provenance = offered
    assert all(key not in body for key in ("tools", "tool_choice", "parallel_tool_calls"))
    schema = body["response_format"]["json_schema"]["schema"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["name"] == "cua_jev_declaration_v66"
    assert schema["required"] == ["tool", "arguments"] and schema["additionalProperties"] is False
    assert len(schema["oneOf"]) == len(schema["$defs"]) == 6
    assert [branch["properties"]["tool"]["const"] for branch in schema["oneOf"]] == list(
        new.DECLARATION_TOOLS
    )
    assert schema["properties"]["tool"]["enum"] == list(new.DECLARATION_TOOLS)
    Draft202012Validator.check_schema(schema)
    references = []

    def inspect(item):
        if type(item) is dict:
            assert not new.UNSUPPORTED_KEYWORDS & set(item)
            if "$ref" in item:
                pointer = item["$ref"]
                assert pointer.startswith("#/$defs/v66_tool_")
                target = schema
                for part in pointer[2:].split("/"):
                    key = part.replace("~1", "/").replace("~0", "~")
                    target = target[int(key)] if type(target) is list else target[key]
                assert type(target) in {dict, bool}
                references.append(pointer)
            for child in item.values():
                inspect(child)
        elif type(item) is list:
            for child in item:
                inspect(child)

    inspect(schema)
    assert len(references) > 50  # Includes references deep inside program/future/prepare definitions.
    restored = copy.deepcopy(body)
    del restored["response_format"]
    restored.update({key: original[key] for key in ("tools", "tool_choice", "parallel_tool_calls")})
    restored["messages"][0]["content"] = original["messages"][0]["content"]
    assert restored == original  # Actual goal/state/history/image/model/thinking stay unchanged.
    assert "Return ONLY one entire JSON object" in body["messages"][0]["content"]
    assert "Return exactly ONE offered inert declaration call" not in body["messages"][0]["content"]
    assert provenance["request_body_sha256"] == _digest(body)
    assert provenance["response_format_json"] == canonical_json(body["response_format"])
    assert (
        provenance["response_format_sha256"]
        == hashlib.sha256(provenance["response_format_json"].encode()).hexdigest()
    )
    assert provenance["provider_schema_enforcement"] == "unverified"
    assert "declaration_tools_sha256" not in provenance


@pytest.mark.parametrize(
    "tool,arguments", declarations(), ids=lambda value: value if type(value) is str else None
)
def test_all_six_choices_validate_and_map_through_original_mapper_without_provider_function_claim(
    offered,
    tool,
    arguments,
):
    declaration = {"tool": tool, "arguments": copy.deepcopy(arguments)}
    payload = response(declaration)
    before = copy.deepcopy(payload)
    schema = offered[1]["response_format"]["json_schema"]["schema"]
    Draft202012Validator(schema).validate(declaration)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate({**declaration, "unoffered_root_field": 1})
    mapped, provenance = decode(payload, offered)
    canonical_packet = packet(json.loads(canonical_json(arguments)), tool)
    original, _ = new.decode_ancestor(canonical_packet, max_actions=6)
    assert mapped == original and payload == before
    assert provenance["provider_tool_call_count"] == provenance["tool_call_count"] == 0
    assert provenance["transport_kind"] == "json_schema_content" and provenance["input_authority"] is False
    assert provenance["selected_tool"] == provenance["tool_name"] == tool
    assert provenance["internal_adapter"]["kind"] == "synthetic_legacy_mapper_payload_not_provider"
    assert provenance["internal_adapter"]["provenance"]["tool_call_count"] == 1
    assert (
        "ancestor_mapping_provenance" not in provenance and "observed_wire_argument_sha256" not in provenance
    )
    assert provenance["argument_hash_kind"] == "adapter_canonical_arguments"
    assert provenance["argument_sha256"] == provenance["canonical_arguments_sha256"]
    for key in ("content_declaration", "canonical_arguments"):
        assert hashlib.sha256(provenance[key + "_json"].encode()).hexdigest() == provenance[key + "_sha256"]
    actual_content = payload["choices"][0]["message"]["content"]
    assert hashlib.sha256(actual_content.encode()).hexdigest() == provenance["observed_content_utf8_sha256"]
    if tool == "declare_stage_program":
        program = parse_stage_program(
            {key: value for key, value in mapped.items() if key != "intent"}, max_actions=6
        )
        public = bind_program_provenance(program, provenance).to_dict()
        assert json.loads(public["wire_declaration_json"]) == arguments
        assert public["source_declaration_status"] == "codec_normalized_canonical_json"
        assert public["normalized_declaration_sha256"] == program.declaration_sha256
    if tool == "declare_current_context_needed":
        assert isinstance(mapped, HighLevelContextRequest) and provenance["context_request"]
        assert provenance["mapping_applied"] is False


@pytest.mark.parametrize(
    "placeholders",
    [
        {},
        {"tool_calls": None, "function_call": None, "refusal": None},
        {"tool_calls": [], "function_call": None, "refusal": ""},
    ],
)
def test_confirmed_zero_call_null_and_empty_SDK_placeholders_are_valid(offered, placeholders):
    payload = response(
        {"tool": "declare_current_context_needed", "arguments": {"reason": "missing_future_semantics"}}
    )
    payload["error"] = None
    payload["choices"][0]["message"].update(placeholders)
    mapped, provenance = decode(payload, offered)
    assert isinstance(mapped, HighLevelContextRequest) and provenance["provider_tool_call_count"] == 0


@pytest.mark.parametrize(
    "text,code",
    [
        ("ordinary prose", "structured_content_invalid_json"),
        ('```json\n{"tool":"declare_completion","arguments":{}}\n```', "structured_content_invalid_json"),
        ("{} afterword", "structured_content_invalid_json"),
        ("{}{}", "structured_content_invalid_json"),
        ("[]", "structured_declaration_exact_fields"),
        ("{}", "structured_declaration_exact_fields"),
        ('{"tool":"declare_completion","arguments":{},"extra":1}', "structured_declaration_exact_fields"),
        (
            '{"tool":"declare_completion","tool":"declare_completion","arguments":{}}',
            "structured_duplicate_json_key",
        ),
        ('{"tool":"declare_completion","arguments":{"x":1,"x":2}}', "structured_duplicate_json_key"),
        ('{"tool":"declare_completion","arguments":{"x":NaN}}', "structured_content_nonfinite_json"),
        ('{"tool":"declare_completion","arguments":{"x":1e9999}}', "structured_content_nonfinite_json"),
        ('{"tool":[],"arguments":{}}', "structured_unoffered_tool"),
        ('{"tool":"declare_native_current_action","arguments":{}}', "structured_unoffered_tool"),
        ('{"tool":"declare_completion","arguments":null}', "structured_arguments_object_required"),
        ('{"tool":"declare_completion","arguments":[]}', "structured_arguments_object_required"),
        ('{"tool":"declare_completion","arguments":{"label":"\\ud800"}}', "structured_content_invalid_utf8"),
        (" " * 64001, "structured_content_bound"),
    ],
    ids=lambda value: "over_bound_content" if type(value) is str and len(value) > 128 else None,
)
def test_illegal_content_is_rejected_without_fences_prose_duplicate_or_nonfinite_fallback(
    offered, text, code
):
    with pytest.raises(ValueError, match="^" + code + "$"):
        decode(response(text=text), offered)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"tool_calls": [{}]}, "structured_provider_tool_calls_forbidden"),
        ({"tool_calls": {}}, "structured_provider_tool_calls_forbidden"),
        ({"tool_calls": "bad"}, "structured_provider_tool_calls_forbidden"),
        ({"tool_calls": False}, "structured_provider_tool_calls_forbidden"),
        ({"function_call": {}}, "structured_legacy_function_call_forbidden"),
        ({"refusal": PRIVATE}, "structured_refusal"),
        ({"refusal": {}}, "structured_refusal"),
        ({"content": []}, "structured_content_string_required"),
    ],
)
def test_function_refusal_and_malformed_placeholders_do_not_downgrade_to_content(offered, change, code):
    payload = response({"tool": "declare_completion", "arguments": {"label": "Observed result"}})
    payload["choices"][0]["message"].update(change)
    with pytest.raises(ValueError, match="^" + code + "$"):
        decode(payload, offered)


def test_provider_error_nonstop_and_multiple_choices_fail_before_any_declaration(offered):
    good = response({"tool": "declare_completion", "arguments": {"label": "Observed result"}})
    for error in ("", False, {}):
        payload = copy.deepcopy(good)
        payload["error"] = error
        with pytest.raises(ValueError, match="^structured_response_error$"):
            decode(payload, offered)
    payload = copy.deepcopy(good)
    payload["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="^structured_finish_stop_required$"):
        decode(payload, offered)
    payload = copy.deepcopy(good)
    payload["choices"] *= 2
    with pytest.raises(ValueError, match="^structured_one_choice_required$"):
        decode(payload, offered)


def test_actual_request_mask_and_schema_hash_must_bind_the_content_decoder(offered):
    payload = response({"tool": "declare_completion", "arguments": {"label": "Observed result"}})
    with pytest.raises(ValueError, match="^structured_actual_tool_mask_required$"):
        new.decode_semantic_declaration(
            payload, max_actions=6, request_provenance=offered[2], offered_tool_names=["declare_completion"]
        )
    altered = copy.deepcopy(offered[2])
    altered["response_format_sha256"] = "00" * 32
    with pytest.raises(ValueError, match="^structured_response_schema_provenance_mismatch$"):
        new.decode_semantic_declaration(payload, max_actions=6, request_provenance=altered)


def test_unsupported_schema_keywords_removed_recursively_without_rewriting_literal_instance_data():
    schema = {
        "type": "object",
        "properties": {"contains": {"type": "number", "multipleOf": 2}},
        "allOf": [
            {
                "uniqueItems": True,
                "contains": {"const": "x"},
                "minContains": 1,
                "maxContains": 2,
                "additionalItems": False,
            }
        ],
        "const": {"contains": "literal remains", "$ref": "literal remains"},
        "$defs": {"nested": {"$ref": "#/$defs/leaf"}, "leaf": {"type": "string"}},
    }
    removed = []
    projected = new._schema_projection(schema, prefix="#/$defs/tool", removed=removed)
    assert projected["properties"]["contains"] == {"type": "number"}
    assert projected["const"] == schema["const"] and projected["allOf"] == [{}]
    assert {row["keyword"] for row in removed} == new.UNSUPPORTED_KEYWORDS
    assert projected["$defs"]["nested"]["$ref"] == "#/$defs/tool/$defs/leaf"


@pytest.mark.parametrize("limit,formal", [(1, False), (1, True), (6, True)])
def test_non_normal_encode_body_and_decode_are_exact_original58_semantics(
    observed, monkeypatch, limit, formal
):
    body, options = actual_input(observed, monkeypatch, limit=limit, formal=formal)
    original, _ = new.encode_ancestor(body, **options)
    encoded, provenance = new.encode_semantic_declaration(body, **options)
    assert encoded == original and encoded["tool_choice"] == "auto" and "response_format" not in encoded
    assert provenance["normal_json_schema"] is False
    tool = "declare_native_current_action" if limit == 1 else "declare_completion"
    raw = (
        current(1)
        if limit == 1
        else {
            "label": "Observed editor",
            "conditions": [
                {"kind": "text_contains", "value": "Editor"},
            ],
        }
    )
    original, _ = new.decode_ancestor(
        packet(raw, tool), max_actions=limit, normal_program_eligible=options["normal_program_eligible"]
    )
    mapped, transport = new.decode_semantic_declaration(
        packet(raw, tool), max_actions=limit, normal_program_eligible=options["normal_program_eligible"]
    )
    assert mapped == original and transport["transport_kind"] == "original_function_tools"
    assert transport["provider_tool_call_count"] == 1


def test_bounded_real_structure_facts_distinguish_content_failure_without_leaking_values():
    payload = response({"tool": PRIVATE, "arguments": {PRIVATE: PRIVATE}, PRIVATE: PRIVATE})
    payload["headers"] = {"Authorization": PRIVATE}
    facts = new.structured_response_facts(payload)
    fact = facts["choices"][0]
    assert fact["content_json_status"] == "finite_object" and fact["extra_root_keys_present"]
    assert fact["selected_tool"] == "unoffered" and fact["known_root_keys"] == ["arguments", "tool"]
    assert PRIVATE not in json.dumps(facts) and "Authorization" not in json.dumps(facts)
    assert fact["tool_calls_present"] is False and fact["finish_reason"] == "stop"
    for text, expected in (
        ("prose " + PRIVATE, "structured_content_invalid_json"),
        ('{"x":1,"x":2}', "structured_duplicate_json_key"),
        ('{"x":NaN}', "structured_content_nonfinite_json"),
    ):
        facts = new.structured_response_facts(response(text=text))
        assert facts["choices"][0]["content_json_status"] == expected
        assert PRIVATE not in json.dumps(facts)
    payload["choices"][0]["finish_reason"] = {PRIVATE: PRIVATE}
    assert new.structured_response_facts(payload)["choices"][0]["finish_reason"] == "other"
