"""Regression tests for semantic v57 inband envelope."""

import json
from dataclasses import replace

import httpx
import pytest
from test_desktop_output_schema_contract import observed as observed
from test_semantic_tools_planner import model, packet


def completion_arguments(limit):
    conditions = [{"kind": "window_contains", "value": "Main"}]
    if limit == 1:
        return "declare_current_completion", {"summary": "Complete original goal", "done": True,
                                               "completion_conditions": conditions}
    return "declare_completion", {"label": "Complete original goal", "conditions": conditions}


@pytest.mark.parametrize("limit", [1, 6])
def test_semantic_v57_and_inband_review_use_one_original_request_and_safe_current_envelope(observed, limit):
    sent = []
    name, arguments = completion_arguments(limit)

    def answer(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=packet(arguments, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v57")
        plan = planner.plan("Every requirement of the complete original public goal", observed, [],
                            max_actions=limit, whole_goal_inband_epoch=4)
    assert len(sent) == planner.calls == 1 and plan.done and plan.completion_conditions
    assert planner.last_request_provenance["whole_goal_inband_review"] == {
        "version": "all_original_goal_review_v1", "input_epoch": 4,
    }
    assert "EVERY requirement in the original" in sent[0]["messages"][0]["content"]
    state = json.loads(sent[0]["messages"][1]["content"][0]["text"])
    assert state["goal"] == "Every requirement of the complete original public goal"
    assert len(sent[0]["tools"]) == (5 if limit == 1 else 8)
    envelope = planner.last_response_transport["declaration_envelope"]
    assert envelope["version"] == "bounded_declaration_envelope_v1"


@pytest.mark.parametrize("failure", ["bool", "image", "inband_epoch"])
def test_v57_prefail_clears_last_successful_envelope_and_marker_without_another_http(observed, failure):
    sent = []
    name, arguments = completion_arguments(1)

    def answer(request):
        sent.append(request)
        return httpx.Response(200, json=packet(arguments, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v57")
        planner.plan("Original goal", observed, [], max_actions=1, whole_goal_inband_epoch=0)
        assert planner.last_response_transport["declaration_envelope"]
        current = replace(observed, image=None) if failure == "image" else observed
        kwargs = ({"whole_goal_recheck": 1} if failure == "bool" else
                  {"whole_goal_recheck": True} if failure == "image" else
                  {"whole_goal_inband_epoch": False})
        with pytest.raises(ValueError):
            planner.plan("Original goal", current, [], **kwargs)
        assert len(sent) == planner.calls == 1
        assert planner.last_response_structure is planner.last_response_transport is None
        assert planner.last_request_provenance is planner.last_request_image_uploaded is None
        assert planner.last_http_wall_ms is planner.last_raw is None
        assert planner.last_usage == {}


def test_bad_declaration_remains_rejected_and_bounded_diagnostic_never_copies_private_content(observed):
    sent = []
    private = "DO_NOT_ECHO_PRIVATE_RESPONSE_CONTENT"

    def answer(request):
        sent.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                     "message": {"content": private}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v57")
        with pytest.raises((ValueError, RuntimeError)):
            planner.plan("Original goal", observed, [], max_actions=1)
    assert len(sent) == planner.calls == 1 and planner.last_response_transport is None
    diagnostic = planner.last_response_structure
    assert diagnostic["version"] == "bounded_declaration_envelope_v1"
    assert diagnostic["input_authority"] is diagnostic["schema_validated"] is False
    assert private not in json.dumps(diagnostic)
