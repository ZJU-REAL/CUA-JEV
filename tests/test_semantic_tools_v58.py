"""Mock-only isolated v58 wire, mapping, provenance and unchanged ancestor tests."""

import copy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_semantic_tools_v57 import capture, current, packet, planner
from test_semantic_tools_v57 import observed as observed
from test_whole_goal_recheck import InputBackend, finished, run

from cua_jev import desktop_semantic_tools_v58 as new
from cua_jev.agent_planner import parse_plan
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_semantic_tools_v57 import decode_semantic_declaration as decode57


def functions(body):
    return {t["function"]["name"]: t["function"] for t in body["tools"]}


def request(observed, limit, fmt="semantic_tools_v58"):
    _plan, body, instance = capture(observed, current(limit), limit=limit, fmt=fmt)
    return instance, body


@pytest.mark.parametrize("limit", [1, 2, 6, 12])
def test_actual_mock_inventory_order_hashes_and_full_ancestor_body(observed, limit):
    instance, body = request(observed, limit)
    _, old = request(observed, limit, "semantic_tools_v57")
    actual, ancestor = functions(body), functions(old)
    assert list(actual) == (list(new.FAMILIES) + [new.COMPLETION] if limit == 1
                            else list(new.STAGES) + list(new.FAMILIES))
    assert len(actual) == (5 if limit == 1 else 8)
    assert all(actual[name] == ancestor[name] for name in new.FAMILIES)
    if limit > 1:
        assert all(actual[name] == ancestor[name] for name in new.STAGES[1:])
    restored = copy.deepcopy(body)
    restored["tools"] = old["tools"]
    restored["messages"][0]["content"] = old["messages"][0]["content"]
    assert restored == old
    p = instance.last_request_provenance
    assert p["response_transport"] == p["requested_format"] == "semantic_tools_v58"
    assert p["ancestor_response_transport"] == "semantic_tools_v57"
    assert p["ancestor_request_sha256"] == _digest(old)
    assert p["request_body_sha256"] == _digest(body)
    assert p["declaration_schema_sha256"] == p["declaration_tools_sha256"] == _digest(body["tools"])
    assert p["ancestor_declaration_schema_sha256"] == _digest(old["tools"])
    assert p["parameter_schema_sha256"] == {n: _digest(f["parameters"]) for n, f in actual.items()}
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False


def test_actual_strong_H1_wire_changes_only_reviewed_exact_spans(observed):
    _, body = request(observed, 1)
    _, old = request(observed, 1, "semantic_tools_v57")
    system = body["messages"][0]["content"]
    original = old["messages"][0]["content"]
    assert new.OLD_COMPLETION not in system
    assert "summary (short string), id (short STRING such as s1), done=false" in system
    assert "do not emit summary, done, completion_conditions, id or action fields" in system
    reverse = [(replacement, previous) for previous, replacement in new._H1_EDITS]
    assert new._rewrite_once(system, reverse) == original
    assert system.count("IMMEDIATE ACTION CONTRACT: ") == original.count("IMMEDIATE ACTION CONTRACT: ") == 1


def test_property_docs_only_do_not_change_noncompletion_parameter_constraints(observed):
    _, body = request(observed, 6)
    _, old = request(observed, 6, "semantic_tools_v57")
    actual, ancestor = functions(body)["declare_property_stage"], functions(old)["declare_property_stage"]
    restored = copy.deepcopy(actual)
    restored["description"] = ancestor["description"]
    for name in ("edit_operation", "edit_value", "numeric_finalize_key"):
        assert "description" in restored["parameters"]["properties"][name]
        restored["parameters"]["properties"][name] = ancestor["parameters"]["properties"][name]
    assert restored == ancestor
    assert "editable text or numeric" in actual["parameters"]["properties"]["edit_operation"]["description"]
    assert "selectable noneditable" in actual["parameters"]["properties"]["edit_value"]["description"]
    assert "checkbox/radio" in actual["description"]
    assert "never submits a URL, search" in (
        actual["parameters"]["properties"]["numeric_finalize_key"]["description"]
    )
    assert "No key is added implicitly" in body["messages"][0]["content"]


@pytest.mark.parametrize("conditions", [None, [], [{"kind": "text_contains", "value": "ready"}]])
def test_H1_explicit_new_bridge_preserves_label_and_presence_without_old_mapper(conditions, monkeypatch):
    raw = {"label": " exact label "}
    if conditions is not None:
        raw["conditions"] = conditions
    monkeypatch.setattr(new, "decode_ancestor", lambda *_a, **_kw: pytest.fail("H1 completion is new bridge"))
    mapped, p = new.decode_semantic_declaration(packet(raw, new.COMPLETION), max_actions=1)
    assert mapped == {"summary": " exact label ", "steps": [], "done": True,
                      **({"completion_conditions": conditions} if conditions is not None else {})}
    assert p["ancestor_response_transport"] == "semantic_tools_v57"
    assert p["ancestor_mapping_applied"] is False and "ancestor_mapping_provenance" not in p
    assert p["mapping_strategy"] == "v58_H1_completion_header_to_original_empty_plan_v1"
    assert p["tool_name"] == "declare_completion" and p["original_parser_required"]
    assert p["input_authority"] is False


@pytest.mark.parametrize("raw", [
    {}, {"label": ""}, {"label": " "}, {"label": 1}, {"label": "x" * 1001},
    {"label": "ok", "done": True}, {"label": "ok", "summary": "extra"},
    {"label": "ok", "conditions": None}, {"label": "ok", "conditions": {}},
    {"label": "ok", "conditions": [{}] * 9}, {"label": "ok", "conditions": ["bad"]},
    {"label": "ok", "private": "never repaired"},
])
def test_H1_rejects_incompatible_headers_without_dropping_truncating_or_adding_fields(raw):
    before = copy.deepcopy(raw)
    with pytest.raises(ValueError):
        new.decode_semantic_declaration(packet(raw, new.COMPLETION), max_actions=1)
    assert raw == before


@pytest.mark.parametrize("arguments", ['[]', 'null', '{"label":NaN}', '{"label":1e999}', '{"label":'])
def test_original_envelope_JSON_finite_and_root_rejection_is_retained(arguments):
    p = packet({}, new.COMPLETION)
    p["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(ValueError):
        new.decode_semantic_declaration(p, max_actions=1)


def test_old_H1_completion_name_is_not_a_silent_alias():
    with pytest.raises(ValueError, match="unknown declaration function"):
        new.decode_semantic_declaration(packet({"summary": "old", "done": True}, new.OLD_COMPLETION),
                                        max_actions=1)


def test_H1_1000_vs_H6_240_and_empty_array_difference_preserved(observed):
    _, one = request(observed, 1)
    _, many = request(observed, 6)
    h1 = functions(one)[new.COMPLETION]["parameters"]
    h6 = functions(many)[new.COMPLETION]["parameters"]
    assert h1["properties"]["label"]["maxLength"] == 1000
    assert h6["properties"]["label"]["maxLength"] == 240
    assert h1["additionalProperties"] is h6["additionalProperties"] is False
    Draft202012Validator(h1).validate({"label": "x" * 1000, "conditions": []})
    for raw in [{"label": "x" * 241}, {"label": "ok", "conditions": []}]:
        with pytest.raises(ValidationError):
            Draft202012Validator(h6).validate(raw)
        mapped, _ = new.decode_semantic_declaration(packet(raw, new.COMPLETION), max_actions=6)
        assert mapped == decode57(packet(raw, new.COMPLETION), max_actions=6)[0]
        assert mapped == {"intent": "complete", **raw}  # No truncation or deleting an explicit [].
        requests = []
        def answer(request, *, responses=requests, parameters=raw):
            responses.append(request)
            return httpx.Response(200, json=packet(parameters, new.COMPLETION))
        plans = []
        for fmt in ("semantic_tools_v57", "semantic_tools_v58"):
            with httpx.Client(transport=httpx.MockTransport(answer), trust_env=False) as client:
                instance = planner(client, fmt)
                if "conditions" in raw:
                    with pytest.raises(ValueError):
                        instance.plan("Whole goal", observed, [], max_actions=6)
                else:
                    # Legacy compile maps label to summary, whose real runtime limit is1000.
                    plans.append(instance.plan("Whole goal", observed, [], max_actions=6))
                assert instance.calls == 1
        assert len(requests) == 2
        if plans:
            assert plans[0] == plans[1] and plans[0].summary == raw["label"]
    accepted = {"label": "ok", "conditions": [{"kind": "text_contains", "value": "ready"}]}
    assert new.decode_semantic_declaration(packet(accepted, new.COMPLETION), max_actions=6)[0] == (
        decode57(packet(accepted, new.COMPLETION), max_actions=6)[0])


@pytest.mark.parametrize("limit", [1, 6])
@pytest.mark.parametrize("name", list(new.FAMILIES))
def test_every_current_family_preserves_actual_ancestor_mapping(limit, name):
    reply = packet(current(limit, name), name)
    before = copy.deepcopy(reply)
    mapped, p = new.decode_semantic_declaration(reply, max_actions=limit)
    original, _ = decode57(reply, max_actions=limit)
    assert mapped == original and reply == before
    assert p["ancestor_mapping_applied"] is True
    assert p["ancestor_mapping_provenance"]["response_transport"] == "semantic_tools_v57"


@pytest.mark.parametrize("name,raw", [
    ("declare_property_stage", {"label": "Edit", "scope": {"app": "Editor"}, "edit_operation": "fill",
                               "edit_value": "literal", "numeric_finalize_key": "none",
                               "confirmation_label": "none", "completion": "stage"}),
    ("declare_navigation_stage", {"label": "Reach", "scope": {"app": "Editor"},
                                 "arrival": [{"kind": "text_contains", "value": "target"}],
                                 "completion": "stage"}),
    ("declare_field_edits_stage", {"label": "Edit", "edits": [
        {"target": {"name": "Field", "app": "Editor"}, "operation": "fill", "value": "literal"}],
                                  "completion": "stage"}),
])
def test_reusable_stage_mapper_is_exact57(name, raw):
    reply = packet(raw, name)
    assert new.decode_semantic_declaration(reply, max_actions=6)[0] == (
        decode57(reply, max_actions=6)[0])


def test_actual_new_completion_reaches_original_parser_and_trace_only_error_reset(observed):
    requests = []
    replies = iter([{"label": "ready", "conditions": []}, {"label": "bad", "done": True}])
    def answer(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=packet(next(replies), new.COMPLETION))
    with httpx.Client(transport=httpx.MockTransport(answer), trust_env=False) as client:
        instance = planner(client, "semantic_tools_v58")
        plan = instance.plan("Whole goal", observed, [], max_actions=1)
        assert plan.done and not plan.steps
        assert instance.last_response_transport["ancestor_mapping_applied"] is False
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            instance.plan("Whole goal", observed, [], max_actions=1)
        facts = instance.last_declaration_rejection
        assert facts["top_level"]["known_fields"]["label"] == {"present": True, "type": "string"}
        assert facts["top_level"]["unknown_key_count_capped"] == 1
        with pytest.raises(ValueError, match="whole_goal_recheck must be a boolean"):
            instance.plan("Whole goal", observed, [], max_actions=1, whole_goal_recheck="invalid")
        assert instance.last_declaration_rejection is None
    assert len(requests) == instance.calls == 2
    assert requests[0] == requests[1] and "declaration_rejection" not in json.dumps(requests)


def test_mapped_completion_does_not_bypass_original_predicate_parser(observed):
    raw, _ = new.decode_semantic_declaration(
        packet({"label": "ready", "conditions": [{"kind": "not_a_predicate"}]}, new.COMPLETION),
        max_actions=1,
    )
    with pytest.raises(ValueError):
        parse_plan(raw, observed, profile="desktop", max_actions=1)


def test_runner_v58_diagnostic_stays_trace_only_and_does_not_change_call_budget():
    marker = {"version": "bounded_declaration_rejection_v1", "code": "adapter_rejected"}
    class Planner:
        profile = "desktop"
        desktop_output_format = "semantic_tools_v58"
        last_validation_error = "original rejection"
        last_declaration_rejection = marker
        def __init__(self):
            self.histories = []
        def plan(self, goal, snapshot, history, *, max_actions=1):
            self.histories.append(copy.deepcopy(history))
            if len(self.histories) == 1:
                raise ValueError("invalid multimodal planning response (ValueError)")
            return finished()
    backend, instance = InputBackend(), Planner()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    result, _, _, trace = run([], enabled=False, mode="vlm", backend=backend, planner=instance, max_plans=2)
    assert result.planner_calls == len(instance.histories) == 2 and not backend.executed
    assert "declaration_rejection" not in json.dumps(instance.histories)
    errors = [e["payload"] for e in trace.events if e["kind"] == "agent_recovery"]
    assert errors[0]["declaration_rejection"] == marker


@pytest.mark.parametrize("options,marker", [
    ({"whole_goal_recheck": True}, "whole_goal_recheck"),
    ({"whole_goal_inband_epoch": 0}, "whole_goal_inband_review"),
])
def test_actual_whole_goal_variants_keep_full_image_and_no_old_completion_wire(observed, options, marker):
    requests = []
    def answer(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=packet({"label": "Whole goal already observed"}, new.COMPLETION))
    with httpx.Client(transport=httpx.MockTransport(answer), trust_env=False) as client:
        instance = planner(client, "semantic_tools_v58")
        plan = instance.plan("Whole original goal", observed, [], max_actions=1, **options)
    assert instance.calls == len(requests) == 1 and plan.done and not plan.steps
    assert instance.last_request_provenance[marker]
    assert new.OLD_COMPLETION not in requests[0]["messages"][0]["content"]
    assert any(item["type"] == "image_url" for item in requests[0]["messages"][1]["content"])


@pytest.mark.parametrize("case", ["missing_effect", "nonlocal_effect", "cross_target", "false_guard"])
def test_v58_does_not_bypass_actual_v57_H6_semantic_click_or_precondition_guards(observed, case):
    raw = current(6, "declare_semantic_current_action")
    if case == "missing_effect":
        raw.pop("postconditions")
    elif case == "nonlocal_effect":
        raw["postconditions"] = [{"kind": "exists", "target": raw["target"]}]
    elif case == "cross_target":
        raw["postconditions"] = [{"kind": "focused", "target": {
            "name": "Other", "role": "button", "app": "Editor", "window": "Main",
        }}]
    else:
        raw["preconditions"] = [{"kind": "value_equals", "target": {
            "name": "Field", "role": "textbox", "app": "Editor", "window": "Main",
        }, "value": "different from the actual current value"}]
    validations = []
    for fmt in ("semantic_tools_v57", "semantic_tools_v58"):
        requests = []
        def answer(request, *, collected=requests):
            collected.append(request)
            return httpx.Response(200, json=packet(raw, "declare_semantic_current_action"))
        with httpx.Client(transport=httpx.MockTransport(answer), trust_env=False) as client:
            instance = planner(client, fmt)
            with pytest.raises(ValueError, match="invalid multimodal planning response"):
                instance.plan("Whole goal", observed, [], max_actions=6)
        assert instance.calls == len(requests) == 1
        validations.append(instance.last_validation_error)
    assert validations[0] == validations[1] and validations[0]
