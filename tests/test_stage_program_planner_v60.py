'Regression tests for test stage program planner v60.'

import copy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_semantic_tools_v57 import current, packet
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_contract_v60 import declaration, query_panel

from cua_jev.desktop_semantic_tools_v60 import (
    FORMAT,
    PROGRAM_GUIDANCE,
    PROGRAM_TOOL,
    decode_semantic_declaration,
)
from cua_jev.desktop_stage_program import StageProgramPlan


def captured(observed, *, fmt, limit, raw, tool):
    requests = []

    def response(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, tool))

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        instance = make_planner(client, fmt=fmt)
        plan = instance.plan("Use the requested generic desktop workflow", observed, [], max_actions=limit)
    assert len(requests) == instance.calls == 1
    return plan, requests[0], instance


def test_one_program_HTTP_returns_only_first_plan_then_activation_makes_zero_HTTP(observed):
    requests = []
    raw = declaration()

    def response(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, PROGRAM_TOOL))

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        instance = make_planner(client, fmt=FORMAT)
        wrapped = instance.plan("Reach and edit the requested generic query", observed, [], max_actions=6)
        assert isinstance(wrapped, StageProgramPlan)
        assert wrapped.initial_plan.steps[0].navigation
        assert not any(control.label == "Requested query" for control in observed.controls)
        before_provenance = copy.deepcopy(instance.last_request_provenance)
        before_usage = copy.deepcopy(instance.usage_totals)
        child = instance.activate_program_stage(
            wrapped.program, 1, query_panel(observed), remaining_actions=1)
        assert child.reusable and child.steps[0].value == "requested literal"
        assert child.steps[0].target.window == "Query panel"
        assert not child.done and not child.completion_conditions
        assert len(requests) == instance.calls == 1
        assert instance.usage_totals == before_usage
        assert instance.last_request_provenance == before_provenance
        assert instance.last_response_transport["program_declaration"] is True
        assert instance.last_response_transport["future_compiled_steps"] == 0


def test_H6_puts_program_first_and_preserves_all_original_eight_tools_in_order(observed):
    _, old_request, _ = captured(observed, fmt="semantic_tools_v58", limit=6,
                                 raw=current(6), tool="declare_native_current_action")
    _, new_request, _ = captured(observed, fmt=FORMAT, limit=6,
                                 raw=current(6), tool="declare_native_current_action")
    old, body = json.loads(old_request.content), json.loads(new_request.content)
    assert len(old["tools"]) == 8 and len(body["tools"]) == 9
    assert body["tools"][1:] == old["tools"]
    assert body["tools"][0]["function"]["name"] == PROGRAM_TOOL
    parameters = body["tools"][0]["function"]["parameters"]
    Draft202012Validator.check_schema(parameters)
    Draft202012Validator(parameters).validate(declaration())
    restored = copy.deepcopy(body)
    restored["tools"].pop(0)
    assert restored["messages"][0]["content"].endswith(PROGRAM_GUIDANCE)
    restored["messages"][0]["content"] = restored["messages"][0]["content"][:-len(PROGRAM_GUIDANCE)]
    assert restored == old


def test_strong_H1_request_bytes_and_current_plan_remain_exact58(observed):
    raw = current(1)
    original, old_request, _ = captured(observed, fmt="semantic_tools_v58", limit=1,
                                        raw=raw, tool="declare_native_current_action")
    actual, request, instance = captured(observed, fmt=FORMAT, limit=1,
                                         raw=raw, tool="declare_native_current_action")
    assert request.content == old_request.content and actual == original
    assert PROGRAM_TOOL not in {tool["function"]["name"] for tool in json.loads(request.content)["tools"]}
    assert instance.last_response_transport["strong_H1_mapping_exact_ancestor58"] is True
    with pytest.raises(ValueError):
        decode_semantic_declaration(packet(declaration(), PROGRAM_TOOL), max_actions=1)


def test_program_JSON_cannot_silently_overwrite_duplicate_fields():
    response = packet(declaration(), PROGRAM_TOOL)
    response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"label":"original","label":"replacement","stages":[]}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        decode_semantic_declaration(response, max_actions=6)


@pytest.mark.parametrize("field,value", [("label", "private accidental"), ("stages", [])])
def test_program_is_not_accepted_as_preparation_repair_even_if_otherwise_well_formed(field, value):
    raw = declaration()
    # Preparation repair has one original H1 action authority, never a program.
    raw[field] = value
    with pytest.raises(ValueError, match="preparation repair"):
        decode_semantic_declaration(packet(raw, PROGRAM_TOOL), max_actions=6,
                                    prepare_repair_token="ab" * 16)
