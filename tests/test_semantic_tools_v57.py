"""Regression tests for semantic tools v57."""

import copy
import importlib.util
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import Draft202012Validator, ValidationError
from PIL import Image

from cua_jev import desktop_semantic_tools_v57 as new
from cua_jev.agent_planner import MultimodalPlanner, _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.desktop_declaration_envelope import declaration_envelope_facts
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_semantic_tools_v56 import decode_semantic_declaration as decode56
from cua_jev.osworld_backend import screenshot_image

ROOT = Path(__file__).resolve().parents[1]
CURRENT = {
    "declare_native_current_action", "declare_semantic_current_action",
    "declare_visual_current_action", "declare_key_page_action",
}
HELP = {"name": "Help", "role": "button", "app": "Editor", "window": "Main"}
OTHER = {"name": "Other", "role": "button", "app": "Editor", "window": "Main"}
CHECK = {"name": "Wrap", "role": "checkbox", "app": "Editor", "window": "Main"}
FIELD = {"name": "Field", "role": "textbox", "app": "Editor", "window": "Main"}


@pytest.fixture
def observed():
    output = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(output, "PNG")
    owner = {"app": "Editor", "window": "Main", "active": True}
    controls = (
        AgentControl("root", "Main", "container", role="frame", box=(0, 0, 1000, 1000), **owner),
        AgentControl("help", "Help", "button", role="button", box=(600, 100, 700, 180), **owner),
        AgentControl("other", "Other", "button", role="button", box=(600, 200, 700, 280), **owner),
        AgentControl("check", "Wrap", "input", role="checkbox", input_type="checkbox",
                     box=(600, 300, 700, 380), **owner),
        AgentControl("field", "Field", "input", role="textbox", input_type="text", value="old",
                     focused=True, editable=True, box=(100, 100, 400, 180), **owner),
    )
    return AgentSnapshot("osworld://desktop", "Main", "Editor Main", controls,
                         image=screenshot_image(output.getvalue()))


def packet(raw, name="declare_native_current_action", *, content="PRIVATE_NARRATION"):
    return {"usage": {"prompt_tokens": 20, "completion_tokens": 7}, "choices": [{
        "finish_reason": "tool_calls", "message": {"content": content, "tool_calls": [{
            "type": "function", "function": {"name": name, "arguments": json.dumps(raw)},
        }]},
    }]}


def current(limit, name="declare_native_current_action"):
    header = {"summary": "Current action", "id": "only", "done": False} if limit == 1 else {
        "label": "Current action",
    }
    action = {
        "declare_native_current_action": {"operation": "click", "ref": "help"},
        "declare_semantic_current_action": {"operation": "click", "target": HELP,
            "postconditions": [{"kind": "focused", "target": HELP}]},
        "declare_visual_current_action": {"operation": "visual_click", "action_label": "Visible current Help",
            "box": [100, 100, 300, 200], "box_space": "normalized_1000"},
        "declare_key_page_action": {"operation": "key_press", "value": "ENTER"},
    }[name]
    return {**header, **copy.deepcopy(action)}


def planner(client, fmt="semantic_tools_v57", cls=MultimodalPlanner):
    return cls("https://offline.invalid", "strong-fixture", api_key="offline", client=client,
               profile="desktop", desktop_contract="property_intent_v52", desktop_output_format=fmt,
               desktop_planning_observation="full_multimodal_v1", enable_thinking=True,
               request_timeout_s=90)


def capture(observed, raw, *, limit=6, name="declare_native_current_action",
            fmt="semantic_tools_v57", cls=MultimodalPlanner):
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        instance = planner(client, fmt, cls)
        plan = instance.plan("Use the current public state", observed, [], max_actions=limit)
    assert len(requests) == instance.calls == 1
    return plan, json.loads(requests[0].content), instance


def parameters(body, name):
    return next(tool["function"]["parameters"] for tool in body["tools"] if tool["function"]["name"] == name)


def oracle(raw, observed, limit):
    options = {"profile": "desktop"}
    if limit > 1:
        options.update(allow_property_goal_conditions=True, allow_advisory_completion=True,
                       inherit_current_action_label=True, allow_guarded_commit_only=True,
                       allow_property_preparation=True, allow_property_value_binding=True,
                       allow_dialog_commit_effect=True, allow_observed_property_goal=True,
                       allow_current_commit_refs=True, allow_property_numeric_finalization=True)
    parsed = parse_plan(copy.deepcopy(raw), observed, max_actions=limit, **options)
    _validate_desktop_execution_plan(parsed, observed, limit)
    return parsed


@pytest.mark.parametrize("name", sorted(CURRENT))
def test_h1_four_current_schemas_allow_only_omitted_or_empty_dependencies(observed, name):
    raw = current(1, name)
    _, body, _ = capture(observed, raw, limit=1, name=name)
    schema = parameters(body, name)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(raw)
    validator.validate({**raw, "depends_on": []})
    assert schema["properties"]["depends_on"]["maxItems"] == 0
    invalid = {**raw, "depends_on": ["prior_call_is_not_a_step"]}
    with pytest.raises(ValidationError):
        validator.validate(invalid)
    mapped, _ = new.decode_semantic_declaration(packet(invalid, name), max_actions=1)
    with pytest.raises(ValueError):
        oracle(mapped, observed, 1)


@pytest.mark.parametrize("kind", ["selected", "focused", "checked", "unchecked"])
def test_h6_same_target_local_click_stays_legal_without_any_preconditions(observed, kind):
    target = CHECK if kind in {"checked", "unchecked"} else HELP
    raw = {"label": "Set an observed local state", "operation": "click", "target": target,
           "postconditions": [{"kind": kind, "target": copy.deepcopy(target)}]}
    plan, body, _ = capture(observed, raw, name=new.SEMANTIC_TOOL)
    assert plan.steps[0].preconditions == () and not plan.steps[0].navigation
    Draft202012Validator(parameters(body, new.SEMANTIC_TOOL)).validate(raw)
    Draft202012Validator(parameters(body, new.SEMANTIC_TOOL)).validate({**raw, "preconditions": []})
    mapped57, _ = new.decode_semantic_declaration(packet(raw, new.SEMANTIC_TOOL), max_actions=6)
    mapped56, _ = decode56(packet(raw, new.SEMANTIC_TOOL), max_actions=6)
    assert mapped57 == mapped56 and oracle(mapped57, observed, 6) == plan


def test_necessary_schema_does_not_replace_same_target_parser_check(observed):
    valid = current(6, new.SEMANTIC_TOOL)
    _, body, _ = capture(observed, valid, name=new.SEMANTIC_TOOL)
    wrong = {**valid, "postconditions": [{"kind": "focused", "target": OTHER}]}
    Draft202012Validator(parameters(body, new.SEMANTIC_TOOL)).validate(wrong)
    mapped, _ = new.decode_semantic_declaration(packet(wrong, new.SEMANTIC_TOOL), max_actions=6)
    with pytest.raises(ValueError, match="same-target|hybrid UI navigation"):
        oracle(mapped, observed, 6)
    wrong["postconditions"] = [{"kind": "expanded", "target": HELP}]
    with pytest.raises(ValidationError):
        Draft202012Validator(parameters(body, new.SEMANTIC_TOOL)).validate(wrong)


def test_persistent_precondition_final_commit_and_native_current_exception_remain(observed):
    guard = {"kind": "value_equals", "target": FIELD, "value": "old"}
    effect = {"kind": "absent", "target": {"name": "Unsaved", "app": "Editor"}}
    raw = {"label": "Commit an observed persistent value", "operation": "click", "target": HELP,
           "preconditions": [guard], "postconditions": [effect]}
    plan, body, _ = capture(observed, raw, name=new.SEMANTIC_TOOL)
    Draft202012Validator(parameters(body, new.SEMANTIC_TOOL)).validate(raw)
    assert len(plan.steps) == 1 and plan.steps[0].preconditions
    native, _, _ = capture(observed, current(6))
    assert len(native.steps) == 1 and native.steps[0].ref == "help"
    assert not native.steps[0].preconditions and not native.steps[0].postconditions
    illegal = {"summary": "A guarded commit followed by more work", "steps": [
        {"id": "commit", "operation": "click", "target": HELP,
         "preconditions": [guard], "postconditions": [effect]},
        {"id": "later", "operation": "fill", "target": FIELD, "value": "new", "depends_on": ["commit"],
         "postconditions": [{"kind": "value_equals", "target": FIELD, "value": "new"}]},
    ], "done": False}
    with pytest.raises(ValueError, match="guarded exact click must end"):
        oracle(illegal, observed, 6)


@pytest.mark.parametrize("limit", [1, 6])
def test_full_strong_system_and_unrelated_tool_definitions_are_exact_v56(observed, limit):
    raw = current(limit)
    _, old, _ = capture(observed, raw, limit=limit, fmt="semantic_tools_v56")
    _, body, instance = capture(observed, raw, limit=limit)
    suffix = new.DESTINATION_CONTENT_GUIDANCE
    assert suffix not in old["messages"][0]["content"]
    assert body["messages"][0]["content"] == old["messages"][0]["content"] + suffix
    assert body["messages"][0]["content"].count(suffix) == 1
    assert body["messages"][1:] == old["messages"][1:]  # State/history and full image exact.
    assert len(body["tools"]) == len(old["tools"]) == (5 if limit == 1 else 8)
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    untouched = set(tool["function"]["name"] for tool in body["tools"]) - (
        CURRENT if limit == 1 else {new.SEMANTIC_TOOL}
    )
    for name in untouched:
        assert next(t for t in body["tools"] if t["function"]["name"] == name) == next(
            t for t in old["tools"] if t["function"]["name"] == name
        )
    metadata = instance.last_request_provenance
    assert metadata["response_transport"] == "semantic_tools_v57"
    assert metadata["ancestor_response_transport"] == "semantic_tools_v56"
    assert metadata["ancestor_system_sha256"] == _digest(old["messages"][0]["content"])
    assert metadata["projected_system_sha256"] == _digest(body["messages"][0]["content"])
    assert metadata["strict_enforcement_claimed"] is False and metadata["input_authority"] is False


def test_destination_guidance_cannot_be_appended_twice_and_is_generic(observed, monkeypatch):
    _, ancestor, instance = capture(observed, current(1), limit=1, fmt="semantic_tools_v56")
    ancestor["messages"][0]["content"] += new.DESTINATION_CONTENT_GUIDANCE
    before = copy.deepcopy(ancestor)
    monkeypatch.setattr(new, "encode_ancestor", lambda *_args, **_kwargs: (
        ancestor, instance.last_request_provenance,
    ))
    with pytest.raises(ValueError):
        new.encode_semantic_declaration({}, max_actions=1, profile="desktop", contract="property_intent_v52",
                                         observation="full_multimodal_v1")
    assert ancestor == before
    guidance = new.DESTINATION_CONTENT_GUIDANCE
    assert "Do not assume" in guidance
    for task_or_route in ("Thunderbird", "about:", "://", "Ctrl+", "7b1e1ff9", "GIMP", "VS Code"):
        assert task_or_route not in guidance


@pytest.mark.parametrize("corruption", ["changed_prefix", "missing_suffix", "duplicate_suffix"])
def test_roundtrip_guard_checks_actual_system_before_stripping_the_unique_suffix(
    observed, monkeypatch, corruption,
):
    _, ancestor, instance = capture(observed, current(1), limit=1, fmt="semantic_tools_v56")
    before = copy.deepcopy(ancestor)
    monkeypatch.setattr(new, "encode_ancestor", lambda *_args, **_kwargs: (
        ancestor, instance.last_request_provenance,
    ))
    root_copies = 0

    def copy_with_bad_restored_system(value):
        nonlocal root_copies
        result = copy.deepcopy(value)
        if type(value) is dict and "messages" in value and "tools" in value:
            root_copies += 1
            if root_copies == 2:  # The local restoration copy, not the ancestor or encoder output.
                original = ancestor["messages"][0]["content"]
                suffix = new.DESTINATION_CONTENT_GUIDANCE
                result["messages"][0]["content"] = {
                    "changed_prefix": "UNEXPECTED_PREFIX" + original + suffix,
                    "missing_suffix": original,
                    "duplicate_suffix": original + suffix + suffix,
                }[corruption]
        return result

    monkeypatch.setattr(new, "copy", SimpleNamespace(deepcopy=copy_with_bad_restored_system))
    with pytest.raises(ValueError, match="exact appended guidance suffix"):
        new.encode_semantic_declaration({}, max_actions=1, profile="desktop", contract="property_intent_v52",
                                         observation="full_multimodal_v1")
    assert ancestor == before


@pytest.fixture(scope="module")
def frozen56_planner_type():
    # Future test only: load the actual frozen class under the same package namespace.
    # This code has NOT been executed during this review.
    path = ROOT / "tests/fixtures/references/agent_planner_v56.py"
    spec = importlib.util.spec_from_file_location("cua_jev._frozen56_planner_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MultimodalPlanner


@pytest.mark.parametrize("fmt", ["semantic_tools_v55", "semantic_tools_v56"])
@pytest.mark.parametrize("limit", [1, 6])
def test_old55_56_whole_body_mapper_and_envelope_shape_are_frozen56(
    observed, frozen56_planner_type, fmt, limit,
):
    raw = current(limit)
    expected, old_body, old = capture(observed, raw, limit=limit, fmt=fmt, cls=frozen56_planner_type)
    parsed, body, instance = capture(observed, raw, limit=limit, fmt=fmt)
    assert parsed == expected and body == old_body
    assert instance.last_raw == old.last_raw
    assert "declaration_envelope" not in instance.last_response_transport
    assert "declaration_envelope" not in old.last_response_transport


def test_new_format_navigation_keeps_original_compiler_owned_legal_gesture_projection(observed):
    raw = {"label": "Reach observed controls", "scope": {"app": "Editor"},
           "arrival": [{"kind": "exists", "target": HELP}],
           "completion": "stage"}
    plan, _, instance = capture(observed, raw, name="declare_navigation_stage")
    assert plan.steps[0].navigation and plan.steps[0].full_legal_navigation_gestures
    assert instance.last_response_transport["response_transport"] == "semantic_tools_v57"


def test_v56_mapping_and_offered_repair_token_are_not_rewritten():
    token = "ab" * 16
    raw = {"summary": "Repair current preparation", "id": "only", "done": False,
           "operation": "visual_fill", "action_label": "Visible current field", "value": "literal query",
           "box": [100, 100, 300, 200], "box_space": "normalized_1000", "prepare_repair": token}
    payload = packet(raw, "declare_visual_current_action")
    expected, old_metadata = decode56(payload, max_actions=1, prepare_repair_token=token)
    actual, metadata = new.decode_semantic_declaration(payload, max_actions=1, prepare_repair_token=token)
    assert actual == expected and metadata["ancestor_mapping_provenance"] == old_metadata
    assert token not in json.dumps(metadata)
    for missing_or_wrong in (None, "cd" * 16):
        with pytest.raises(ValueError):
            new.decode_semantic_declaration(payload, max_actions=1, prepare_repair_token=missing_or_wrong)
    key = packet(current(6, "declare_key_page_action"), "declare_key_page_action")
    assert new.decode_semantic_declaration(key, max_actions=6)[0] == decode56(key, max_actions=6)[0]


def test_envelope_is_bounded_whitelist_only_and_never_interprets_raw_content():
    secret = "PRIVATE_PROVIDER_BODY" * 10000
    call = {"id": secret, "type": secret, "function": {"name": secret, "arguments": secret}}
    choice = {"finish_reason": secret, "message": {"content": secret, "refusal": secret,
              "function_call": {"name": secret, "arguments": {"private": secret}},
              "tool_calls": [copy.deepcopy(call)] * 80}}
    payload = {"choices": [choice] * 80, "error": {"headers": {"Authorization": secret}}}
    before = copy.deepcopy(payload)
    facts = declaration_envelope_facts(
        payload, offered_function_names=(secret, "declare_native_current_action")
    )
    text = json.dumps(facts)
    assert payload == before and "PRIVATE_" not in text and len(text) < 15000
    assert facts["choices"]["count_capped"] == 4 and facts["choices"]["overflow"] is True
    assert len(facts["choices"]["items"]) == 4
    for item in facts["choices"]["items"]:
        calls = item["message"]["shape"]["tool_calls"]
        assert calls["count_capped"] == 4 and calls["overflow"] is True and len(calls["items"]) == 4
        assert calls["items"][0]["function"]["shape"]["name"]["enum"] == "unoffered"
    assert facts["input_authority"] is False and facts["schema_validated"] is False


@pytest.mark.parametrize("failure", ["missing_tools", "multiple_choices", "multiple_calls", "legacy_only",
                                     "unknown_name", "object_arguments", "refusal", "nonfinite"])
def test_current_extraction_failure_keeps_only_current_envelope_and_one_http(observed, failure):
    good = packet(current(1))
    bad = packet(current(1), content="PRIVATE_RESPONSE_CONTENT")
    message = bad["choices"][0]["message"]
    if failure == "missing_tools":
        del message["tool_calls"]
    elif failure == "multiple_choices":
        bad["choices"] *= 2
    elif failure == "multiple_calls":
        message["tool_calls"] *= 2
    elif failure == "legacy_only":
        message["function_call"] = message.pop("tool_calls")[0]["function"]
    elif failure == "unknown_name":
        message["tool_calls"][0]["function"]["name"] = "PRIVATE_UNKNOWN_FUNCTION"
    elif failure == "object_arguments":
        message["tool_calls"][0]["function"]["arguments"] = {"private": "PRIVATE_ARGUMENT"}
    elif failure == "refusal":
        message["refusal"] = "PRIVATE_REFUSAL"
    else:
        message["tool_calls"][0]["function"]["arguments"] = '{"operation":NaN}'
    replies = iter([good, bad])
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=next(replies))
    )) as client:
        instance = planner(client)
        instance.plan("Current", observed, [], max_actions=1)
        assert instance.last_response_transport["declaration_envelope"]
        with pytest.raises(ValueError) as caught:
            instance.plan("Current", observed, [], max_actions=1)
    assert len(requests) == instance.calls == 2 and instance.last_response_transport is None
    facts = instance.last_response_structure
    assert facts["version"] == "bounded_declaration_envelope_v1"
    assert "PRIVATE_" not in json.dumps(facts) + str(caught.value) + instance.last_validation_error


@pytest.mark.parametrize("kind", ["HTTP", "JSON"])
def test_transport_or_json_failure_after_success_clears_old_envelope(observed, kind):
    requests = []

    def answer(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=packet(current(1)))
        return (httpx.Response(400, json={"error": "PRIVATE_PROVIDER_TEXT"}) if kind == "HTTP" else
                httpx.Response(200, content=b"PRIVATE_NON_JSON_BODY"))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        instance = planner(client)
        instance.plan("Current", observed, [], max_actions=1)
        with pytest.raises(RuntimeError) as caught:
            instance.plan("Current", observed, [], max_actions=1)
    assert len(requests) == instance.calls == 2
    assert instance.last_response_structure is instance.last_response_transport is None
    assert "PRIVATE_" not in str(caught.value) + instance.last_validation_error


@pytest.mark.parametrize("failure", ["different_target", "H1_dependency"])
def test_original_parser_still_rejects_residual_illegal_output_after_one_current_transport(observed, failure):
    limit = 6 if failure == "different_target" else 1
    name = new.SEMANTIC_TOOL if limit == 6 else "declare_native_current_action"
    bad = current(limit, name)
    if failure == "different_target":
        bad["postconditions"] = [{"kind": "focused", "target": OTHER}]
    else:
        bad["depends_on"] = ["prior_model_call"]
    replies = iter([packet(current(limit)), packet(bad, name, content="PRIVATE_IRRELEVANT_CONTENT")])
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=next(replies))
    )) as client:
        instance = planner(client)
        instance.plan("Current", observed, [], max_actions=limit)
        with pytest.raises(ValueError) as caught:
            instance.plan("Current", observed, [], max_actions=limit)
    assert len(requests) == instance.calls == 2
    # Extraction succeeded; parsing did not. Keep these stages distinct.
    transport = instance.last_response_transport
    assert transport["tool_name"] == name and transport["input_authority"] is False
    assert transport["declaration_envelope"]["version"] == "bounded_declaration_envelope_v1"
    assert "PRIVATE_" not in json.dumps(transport["declaration_envelope"]) + str(caught.value)


@pytest.mark.parametrize("failure", ["missing_image", "invalid_bool"])
def test_v57_preflight_failure_after_success_drops_all_old_current_call_fields_without_http(
    observed, failure,
):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=packet(current(1)))
    )) as client:
        instance = planner(client)
        instance.plan("Current", observed, [], max_actions=1)
        totals = dict(instance.usage_totals)
        assert instance.last_response_transport["declaration_envelope"]
        # Semantic transport currently leaves last_raw empty; seed this legacy
        # mutable diagnostic to verify the new early reset clears it too.
        instance.last_raw = "PRIVATE_STALE_RAW"
        bad_snapshot = replace(observed, image=None) if failure == "missing_image" else observed
        flag = True if failure == "missing_image" else "PRIVATE_INVALID_BOOL"
        with pytest.raises(ValueError):
            instance.plan("Current", bad_snapshot, [], max_actions=1, whole_goal_recheck=flag)
    assert len(requests) == instance.calls == 1 and instance.usage_totals == totals
    assert instance.last_response_structure is instance.last_response_transport is instance.last_raw is None
    assert (
        instance.last_request_provenance is instance.last_request_image_uploaded
        is instance.last_http_wall_ms is None
    )
    assert instance.last_usage == {}
