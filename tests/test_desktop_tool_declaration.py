"""Pure production declaration transport boundaries; no HTTP or GUI."""

import copy
import json

import pytest

from cua_jev import desktop_tool_declaration as DRAFT


def packet(arguments, *, name=DRAFT.FUNCTION):
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "type": "function", "function": {"name": name, "arguments": arguments}}]}}]}


def test_encoder_preserves_full_image_thinking_and_exact_original_body():
    original = {"model": "test-only", "stream": False, "enable_thinking": True,
        "messages": [{"role": "system", "content": "Original grammar"},
            {"role": "user", "content": [{"type": "text", "text": "Saved public AX"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fixture"}}]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "schema": {"anyOf": [{"type": "object"}, {"type": "object"}]}}}}
    before = copy.deepcopy(original)
    body, facts = DRAFT.encode_tool_declaration(original, profile="desktop",
        contract="property_intent_v52", observation="full_multimodal_v1")
    assert original == before and body["enable_thinking"] is True and "thinking_budget" not in body
    assert body["messages"][1] == original["messages"][1] and facts["image_uploaded"]
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert len(body["tools"]) == 1 and "strict" not in body["tools"][0]["function"]
    assert "response_format" not in body and facts["redundant_root_object_added"]


def test_original_object_schema_and_unsupported_roots():
    immediate = {"type": "object", "properties": {"steps": {"type": "array"}}}
    assert DRAFT.function_parameters(immediate) == immediate
    for unsupported in ({}, {"anyOf": []}, {"anyOf": [{"type": "object"}, {}]},
                        {"anyOf": [{"type": "array"}]}):
        with pytest.raises(ValueError):
            DRAFT.function_parameters(unsupported)


def test_decoder_extracts_data_without_parser_execution_or_cleanup():
    for raw in ({"intent": "complete", "completion": "goal"},
                {"summary": "Immediate", "steps": [], "done": True},
                {"intent": "pursue_property", "numeric_finalize": {"key": None}}):
        decoded, facts = DRAFT.decode_tool_declaration(packet(json.dumps(raw)))
        assert decoded == raw and facts["tool_call_count"] == 1
        # In particular key:null remains null; the original semantic parser must reject it.


def test_accompanying_content_is_ignored_and_cannot_replace_arguments():
    raw = {"summary": "Immediate", "steps": [], "done": True}
    payload = packet(json.dumps(raw))
    payload["choices"][0]["message"]["content"] = '{"intent":"unrelated-content-only"}'
    decoded, facts = DRAFT.decode_tool_declaration(payload)
    assert decoded == raw and facts["accompanying_content_present"] is True
    assert "unrelated-content-only" not in json.dumps(facts)


def test_decoder_rejects_absent_multiple_wrong_name_bad_json_and_nonobject():
    good = packet('{"intent":"complete"}')
    multi = copy.deepcopy(good)
    multi["choices"][0]["message"]["tool_calls"] *= 2
    multi_choice = {"choices": good["choices"] * 2}
    refusal = copy.deepcopy(good)
    refusal["choices"][0]["message"]["refusal"] = "Refused"
    for payload in ({"choices": [{"message": {"content": '{"intent":"complete"}'}}]},
                    multi, multi_choice, refusal,
                    packet("{}", name="wrong"), packet("{"), packet("[]"), packet({}),
                    packet(" " * 64001)):
        with pytest.raises(ValueError):
            DRAFT.decode_tool_declaration(payload)


def test_configuration_is_optin_desktop52_full_multimodal_only():
    for profile, contract, observation in (
        ("browser", "property_intent_v52", "full_multimodal_v1"),
        ("desktop", "property_intent_v51", "full_multimodal_v1"),
        ("desktop", "property_intent_v52", "native_first_v51"),
    ):
        with pytest.raises(ValueError):
            DRAFT.validate_tool_declaration_selection(profile, contract, observation)
