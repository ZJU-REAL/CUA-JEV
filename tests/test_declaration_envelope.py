"""Regression tests for declaration envelope."""

import copy
import json

import httpx
import pytest
from test_desktop_output_schema_contract import observed as observed
from test_semantic_tools_planner import model
from test_semantic_tools_v55 import capture

from cua_jev.desktop_declaration_envelope import declaration_envelope_facts

NATIVE = "declare_native_current_action"
SECRET = "PRIVATE_ARGUMENT_CONTENT_AND_UNKNOWN_NAME_79df"


def envelope():
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": SECRET, "refusal": None, "tool_calls": [{"id": SECRET, "type": "function",
        "function": {"name": NATIVE, "arguments": json.dumps({"private": SECRET})}}],
    }}], "id": SECRET, "headers": {"Authorization": SECRET}, "error": {"message": SECRET}}


def test_fixed_facts_exclude_text_ids_unknown_keys_hashes_and_arguments():
    payload = envelope()
    before = copy.deepcopy(payload)
    facts = declaration_envelope_facts(payload, offered_function_names=(NATIVE,))
    assert payload == before
    rendered = json.dumps(facts)
    assert SECRET not in rendered and "Authorization" not in rendered and "sha256" not in rendered
    assert facts["choices"]["count_capped"] == 1 and facts["choices"]["overflow"] is False
    message = facts["choices"]["items"][0]["message"]["shape"]
    assert message["content"] == {"present": True, "type": "string"}
    assert message["refusal"] == {"present": True, "type": "null"}
    function = message["tool_calls"]["items"][0]["function"]["shape"]
    assert function["name"]["enum"] == NATIVE
    assert function["arguments"] == {"present": True, "type": "string", "length_bucket": "le_64"}


def test_four_by_four_caps_overflow_and_unknown_names_never_escape():
    payload = envelope()
    call = payload["choices"][0]["message"]["tool_calls"][0]
    call["function"]["name"] = SECRET
    payload["choices"][0]["message"]["tool_calls"] = [call] * 99
    payload["choices"] *= 99
    facts = declaration_envelope_facts(payload, offered_function_names=(SECRET, NATIVE))
    assert facts["choices"]["count_capped"] == 4 and facts["choices"]["overflow"] is True
    assert len(facts["choices"]["items"]) == 4
    for choice in facts["choices"]["items"]:
        calls = choice["message"]["shape"]["tool_calls"]
        assert calls["count_capped"] == 4 and calls["overflow"] is True and len(calls["items"]) == 4
        assert all(c["function"]["shape"]["name"]["enum"] == "unoffered" for c in calls["items"])
    assert SECRET not in json.dumps(facts) and len(json.dumps(facts)) < 16000


@pytest.mark.parametrize("value,kind", [(None, "null"), (False, "boolean"), (0, "number"),
                                        ("", "string"), ([], "array"), ({}, "object")])
def test_argument_type_is_reported_without_parsing_even_malformed_argument_json(value, kind):
    payload = envelope()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = value
    facts = declaration_envelope_facts(payload, offered_function_names=())
    function = facts["choices"]["items"][0]["message"]["shape"]["tool_calls"]["items"][0]
    assert function["function"]["shape"]["arguments"]["type"] == kind
    assert function["function"]["shape"]["name"]["enum"] == "unoffered"


def test_missing_legacy_function_call_and_unknown_finish_reason_are_distinct_fixed_facts():
    payload = {"choices": [{"finish_reason": SECRET, "message": {
        "function_call": {"name": NATIVE, "arguments": SECRET}, "refusal": SECRET,
    }}]}
    facts = declaration_envelope_facts(payload, offered_function_names=(NATIVE,))
    choice = facts["choices"]["items"][0]
    assert choice["finish_reason"]["enum"] == "other"
    assert choice["message"]["shape"]["tool_calls"] == {
        "present": False, "type": "missing", "count_capped": 0, "overflow": False, "items": [],
    }
    assert choice["message"]["shape"]["function_call"]["function"]["name"]["enum"] == NATIVE
    assert SECRET not in json.dumps(facts)


@pytest.mark.parametrize("output_format", ["semantic_tools_v55", "semantic_tools_v56", "semantic_tools_v57"])
def test_real_planner_decode_failure_remains_one_post_and_original_rejection(observed, output_format):
    calls = []
    payload = envelope()
    payload["choices"][0]["message"]["tool_calls"] = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = model(client, desktop_output_format=output_format)
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            planner.plan("Original complete goal", observed, [], max_actions=1)
    assert len(calls) == planner.calls == 1
    assert planner.last_response_transport is None
    assert planner.last_validation_error == (
        "Response must contain exactly one named desktop declaration with JSON-object arguments."
    )
    if output_format == "semantic_tools_v57":
        assert planner.last_response_structure["version"] == "bounded_declaration_envelope_v1"
        assert planner.last_response_structure["choices"]["items"][0]["message"]["shape"][
            "tool_calls"]["count_capped"] == 0
    else:
        assert planner.last_response_structure is None


def test_v57_valid_current_plan_sends_one_body_and_exposes_only_safe_envelope(observed):
    raw = {"summary": "Current input", "id": "s1", "done": False, "operation": "click", "ref": "help"}
    plan, body, planner = capture(observed, raw, NATIVE, limit=1, output_format="semantic_tools_v57")
    assert plan.steps[0].operation == "click" and planner.calls == 1
    assert planner.last_response_transport["declaration_envelope"]["input_authority"] is False
    assert "declaration_envelope" not in json.dumps(body)
    # The semantic57 suite separately retains frozen55/56 whole-request golden checks.
