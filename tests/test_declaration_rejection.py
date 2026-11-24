"""Bounded declaration rejection privacy and structural metadata tests."""

import copy
import json

import pytest

from cua_jev import desktop_declaration_rejection as diagnostic

SECRET = "PRIVATE_VALUE_NAME_ID_HEADER_REFUSAL_d901"
TOOLS = [{"type": "function", "function": {
    "name": "declare_current_completion", "parameters": {
        "type": "object", "required": ["summary", "done"],
        "properties": {"summary": {"type": "string"}, "done": {"const": True},
                       "completion_conditions": {"type": "array"}},
    },
}}]


def payload(arguments, name="declare_current_completion"):
    return {"id": SECRET, "headers": {"Authorization": SECRET},
            "choices": [{"message": {"content": SECRET, "refusal": SECRET,
                "tool_calls": [{"id": SECRET, "type": "function", "function": {
                    "name": name, "arguments": arguments,
                }}]}}]}


def facts(arguments, *, error=None, tools=None):
    return diagnostic.declaration_rejection_facts(
        payload(arguments), offered_tools=TOOLS if tools is None else tools,
        error=ValueError("fixture rejection") if error is None else error,
    )


def test_metadata_does_not_copy_values_unknown_names_ids_headers_refusal_or_hashes():
    raw = {"summary": SECRET, "done": False, SECRET: SECRET,
           "completion_conditions": [{SECRET: SECRET}]}
    source, offered = payload(json.dumps(raw)), copy.deepcopy(TOOLS)
    before = copy.deepcopy((source, offered))
    result = diagnostic.declaration_rejection_facts(source, offered_tools=offered, error=ValueError(SECRET))
    assert (source, offered) == before
    rendered = json.dumps(result)
    assert SECRET not in rendered
    assert all(word not in rendered for word in ("arguments_sha", "sha256", "Authorization", "refusal"))
    assert result["input_authority"] is result["schema_validated"] is False
    assert result["top_level"]["unknown_key_count_capped"] == 1
    assert result["top_level"]["known_fields"]["done"] == {"present": True, "type": "boolean"}
    assert result["top_level"]["known_fields"]["summary"] == {"present": True, "type": "string"}


@pytest.mark.parametrize(("argument", "status", "root", "finite"), [
    ('{"summary":', "invalid_json", "not_decoded", "not_checked"),
    ('{"summary":NaN}', "nonfinite_token", "not_decoded", "not_checked"),
    ('{"summary":Infinity}', "nonfinite_token", "not_decoded", "not_checked"),
    ('{"summary":1e999}', "parsed", "object", "nonfinite"),
    ('[]', "parsed", "array", "all_finite"),
    ('null', "parsed", "null", "all_finite"),
    ('false', "parsed", "boolean", "all_finite"),
    ('0', "parsed", "number", "all_finite"),
    ('"private"', "parsed", "string", "all_finite"),
    ('{}', "parsed", "object", "all_finite"),
])
def test_argument_json_root_and_finiteness_are_fixed_enums(argument, status, root, finite):
    result = facts(argument)
    assert (result["argument_parse_status"], result["root_type"], result["finiteness"]) == (
        status, root, finite,
    )


def test_required_presence_and_condition_count_only_not_predicate_contents():
    result = facts(json.dumps({"completion_conditions": [{SECRET: SECRET}] * 40}))
    root = result["top_level"]
    assert root["known_required_count"] == root["known_required_missing_count"] == 2
    assert root["known_fields"]["summary"] == {"present": False, "type": "missing"}
    conditions = root["known_fields"]["completion_conditions"]
    assert conditions["count_capped"] == 32 and conditions["overflow"]
    assert conditions["item_types_prefix"] == {"object": 8}
    assert conditions["item_type_prefix_complete"] is False
    assert SECRET not in json.dumps(result)


def test_unknown_schema_names_never_become_output_keys():
    tools = copy.deepcopy(TOOLS)
    tools[0]["function"]["parameters"]["properties"][SECRET] = {}
    result = facts(json.dumps({SECRET: SECRET}), tools=tools)
    assert result["top_level"]["unrepresented_schema_key_count_capped"] == 1
    assert SECRET not in json.dumps(result)


def test_tree_and_root_key_work_is_bounded_and_output_does_not_grow_with_data():
    result = facts(json.dumps({"summary": SECRET, "done": True, "completion_conditions": [{}] * 700}))
    assert result["finiteness"] == "unverified_budget"
    result = facts(json.dumps({f"{SECRET}{i}": i for i in range(600)}))
    assert result["top_level"]["unknown_key_count_capped"] == 32
    assert result["top_level"]["unknown_key_count_overflow"] is True
    assert result["top_level"]["unknown_key_scan_complete"] is False
    assert len(json.dumps(result)) < 12000 and SECRET not in json.dumps(result)
    result = facts('[' * 40 + '0' + ']' * 40)
    assert result["argument_parse_status"] == "parsed" and result["finiteness"] == "unverified_budget"


def test_out_of_bounds_or_unknown_function_does_not_parse(monkeypatch):
    monkeypatch.setattr(diagnostic.json, "loads", lambda *_a, **_k: pytest.fail("must not parse"))
    assert facts("x" * 64001)["argument_parse_status"] == "outside_string_bounds"
    result = diagnostic.declaration_rejection_facts(payload("{}", SECRET), offered_tools=TOOLS,
                                                   error=ValueError(SECRET))
    assert result["argument_parse_status"] == "known_offered_schema_unavailable"
    assert SECRET not in json.dumps(result)


@pytest.mark.parametrize(("message", "stage", "code"), [
    ("invalid declaration argument JSON", "declaration_arguments", "argument_decode_rejected"),
    ("H1 empty advisory completion parameters required", "flat_completion", "completion_header_rejected"),
    ("original completion conditions required", "flat_completion", "completion_conditions_shape_rejected"),
    (SECRET, "declaration_adapter", "adapter_value_error"),
])
def test_only_exact_existing_static_error_messages_select_codes(message, stage, code):
    result = facts("{}", error=ValueError(message))
    assert (result["stage"], result["code"]) == (stage, code)
    assert SECRET not in json.dumps(result)


def test_diagnostic_internal_error_returns_fixed_fallback(monkeypatch):
    def fail(*_args):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(diagnostic, "_facts", fail)
    assert facts("{}") == {
        "version": "bounded_declaration_rejection_v1", "stage": "declaration_adapter",
        "code": "adapter_rejected", "details_status": "unavailable",
        "untrusted": True, "input_authority": False, "schema_validated": False,
    }
