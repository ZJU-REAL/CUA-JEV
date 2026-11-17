"""Flat declaration contract boundaries; no provider, VM or GUI operations."""

import copy
import json

import pytest

from cua_jev import desktop_tool_declaration as original_transport
from cua_jev.agent_planner import parse_plan
from cua_jev.agent_types import AgentSnapshot
from cua_jev.desktop_output_schema import desktop_response_format


@pytest.fixture
def prototype():
    from cua_jev import desktop_semantic_tools

    return desktop_semantic_tools


def flat():
    return {"label": "Requested text change", "scope": {"app": "Editor"}, "completion": "goal",
            "edit_operation": "fill", "edit_value": "Requested literal", "numeric_finalize_key": "none",
            "confirmation_label": "Submit requested change"}


def packet(arguments, name="declare_property_stage", content=None):
    return {"choices": [{"message": {"content": content, "tool_calls": [{"type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("nested", [False, True])
def test_nonfinite_json_is_rejected_before_any_stage_mapping(prototype, token, nested):
    payload = packet(flat())
    arguments = '{"label":' + token + '}'
    if nested:
        arguments = '{"label":"Goal","scope":{"app":"Editor","unexpected":[' + token + ']}}'
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(ValueError, match="invalid argument JSON"):
        prototype.decode_semantic_declaration(payload, max_actions=6)


def test_explicit_flat_mapping_preserves_values_and_none_means_no_confirmation(prototype):
    raw = flat()
    untouched = copy.deepcopy(raw)
    mapped, facts = prototype.decode_semantic_declaration(packet(raw, content="ignored"), max_actions=6)
    assert raw == untouched and facts["accompanying_content_present"] and facts["input_authority"] is False
    assert mapped["intent"] == "pursue_property"
    assert mapped["edit"] == {"operation": "fill", "value": raw["edit_value"]}
    assert mapped["numeric_finalize"] == {"key": "none"}
    assert mapped["confirm"] == {"label": raw["confirmation_label"]}
    raw["confirmation_label"] = "none"
    assert prototype.decode_semantic_declaration(packet(raw), max_actions=6)[0]["confirm"] == "none"


@pytest.mark.parametrize("field,value", [
    ("intent", "pursue_property"), ("edit", {"operation": "fill", "value": "Requested literal"}),
    ("confirm", "Submit requested change"), ("numeric_finalize", {"key": "none"}), ("arguments", {}),
])
def test_old_or_extra_fields_are_rejected_even_when_they_match(prototype, field, value):
    with pytest.raises(ValueError):
        prototype.decode_semantic_declaration(packet(flat() | {field: value}), max_actions=6)


@pytest.mark.parametrize("operation,has_value", [
    ("fill", False), ("set", False), ("check", True), ("uncheck", True),
])
def test_operation_value_presence_is_strict_without_defaults(prototype, operation, has_value):
    raw = flat() | {"edit_operation": operation}
    if not has_value:
        raw.pop("edit_value")
    with pytest.raises(ValueError, match="operation/value mismatch"):
        prototype.decode_semantic_declaration(packet(raw), max_actions=6)


def test_text_enter_is_not_coerced_and_still_fails_the_original_numeric_boundary(prototype):
    mapped, _facts = prototype.decode_semantic_declaration(
        packet(flat() | {"numeric_finalize_key": "enter"}), max_actions=6)
    assert mapped["numeric_finalize"] == {"key": "enter"}
    with pytest.raises(ValueError, match="bounded numeric edit"):
        parse_plan(mapped, AgentSnapshot("osworld://desktop", "Editor", "", ()),
                   profile="desktop", max_actions=6,
                   allow_property_value_binding=True, allow_observed_property_goal=True,
                   allow_property_numeric_finalization=True, require_numeric_finalize_choice=True,
                   require_explicit_stage_policy=True)


def test_transitive_defs_and_nonproperty_branch_constraints_remain_original(prototype):
    root = desktop_response_format(6, property_contract="property_intent_v52")["json_schema"]["schema"]
    functions = prototype.semantic_function_schemas(root)
    for name, intent in prototype.FUNCTION_INTENTS.items():
        params = functions[name]
        assert all(params["$defs"][k] == root["$defs"][k] for k in params["$defs"])
        assert params["$defs"] == prototype.referenced_definitions(
            {k: v for k, v in params.items() if k != "$defs"}, root["$defs"])
        if intent != "pursue_property":
            branch = next(b for b in root["anyOf"] if b["properties"]["intent"]["const"] == intent)
            assert params["properties"] == {k: v for k, v in branch["properties"].items() if k != "intent"}
            assert params["required"] == [k for k in branch["required"] if k != "intent"]


def test_H1_still_delegates_original_transport_without_wire_or_prompt_adaptation(prototype, monkeypatch):
    sentinel, calls = object(), []
    monkeypatch.setattr(original_transport, "encode_tool_declaration",
                        lambda body, **kwargs: calls.append((body, kwargs)) or sentinel)
    monkeypatch.setattr(original_transport, "decode_tool_declaration", lambda value: sentinel)
    body = {"unchanged_original_H1": True}
    assert prototype.encode_semantic_declaration(body, max_actions=1, profile="desktop",
        contract="property_intent_v52", observation="full_multimodal_v1") is sentinel
    assert prototype.decode_semantic_declaration({}, max_actions=1) is sentinel
    assert calls[0][0] is body


@pytest.mark.parametrize("field,value", [("confirmation_label", None), ("numeric_finalize_key", False),
                                        ("edit_value", 42), ("completion", "Goal")])
def test_flat_types_and_enums_are_not_coerced(prototype, field, value):
    with pytest.raises(ValueError):
        prototype.decode_semantic_declaration(packet(flat() | {field: value}), max_actions=6)


@pytest.mark.parametrize("kind", ["empty_tools", "multiple_choices", "refusal", "argument_limit"])
def test_response_envelope_is_bounded_and_never_uses_content_as_declaration(prototype, kind):
    payload = packet(flat(), content=json.dumps(flat()))
    message = payload["choices"][0]["message"]
    if kind == "empty_tools":
        message["tool_calls"] = []
    elif kind == "multiple_choices":
        payload["choices"] *= 2
    elif kind == "refusal":
        message["refusal"] = "Provider refusal"
    else:
        message["tool_calls"][0]["function"]["arguments"] = " " * 64001
    with pytest.raises(ValueError):
        prototype.decode_semantic_declaration(payload, max_actions=6)
