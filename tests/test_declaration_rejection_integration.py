"""Trace-only declaration rejection integration with the actual planner and runner."""

import copy
import json

import httpx
import pytest
from test_desktop_output_schema_contract import observed as observed
from test_semantic_tools_planner import model, packet
from test_whole_goal_recheck import InputBackend, finished, run

from cua_jev import desktop_declaration_rejection as diagnostic

VALIDATION = "Response must contain exactly one named desktop declaration with JSON-object arguments."


@pytest.mark.parametrize("format_name", ["semantic_tools_v55", "semantic_tools_v56", "semantic_tools_v57"])
@pytest.mark.parametrize("raw,code", [
    ({"done": True}, "completion_header_rejected"),
    ({"summary": "Advisory", "done": True, "private_unknown": "private"}, "completion_header_rejected"),
    ({"summary": "Advisory", "done": True, "completion_conditions": None},
     "completion_conditions_shape_rejected"),
])
def test_actual_original_adapter_rejection_keeps_one_HTTP_error_and_safe_envelope(
    observed, format_name, raw, code,
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=packet(raw, name="declare_current_completion"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = model(client, desktop_output_format=format_name)
        with pytest.raises(ValueError, match=r"^invalid multimodal planning response \(ValueError\)$"):
            planner.plan("Whole original goal", observed, [], max_actions=1)
    assert len(requests) == planner.calls == 1
    assert planner.last_response_transport is None and planner.last_validation_error == VALIDATION
    assert "declaration_rejection" not in json.dumps(requests)
    if format_name == "semantic_tools_v57":
        assert planner.last_declaration_rejection["code"] == code
        assert planner.last_response_structure["version"] == "bounded_declaration_envelope_v1"
        assert "declaration_rejection" not in planner.last_response_structure
    else:
        assert planner.last_declaration_rejection is None and planner.last_response_structure is None


def test_accepted_v57_completion_and_pre_request_failure_clear_old_diagnostic(observed):
    requests = []
    replies = iter([{"done": True}, {"summary": "Advisory", "done": True}])

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=packet(next(replies), name="declare_current_completion"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v57")
        with pytest.raises(ValueError):
            planner.plan("Whole original goal", observed, [], max_actions=1)
        assert planner.last_declaration_rejection is not None
        with pytest.raises(ValueError, match="whole_goal_recheck must be a boolean"):
            planner.plan("Whole original goal", observed, [], max_actions=1, whole_goal_recheck="invalid")
        assert planner.last_declaration_rejection is None and len(requests) == 1
        plan = planner.plan("Whole original goal", observed, [], max_actions=1)
        assert plan.done and not plan.steps and planner.last_declaration_rejection is None
    assert len(requests) == planner.calls == 2
    # Same goal/snapshot/history/horizon: diagnostic collection changes no request body.
    assert requests[0] == requests[1]


def test_diagnostic_failure_cannot_replace_original_error_or_add_HTTP(observed, monkeypatch):
    def fail(*_a, **_kw):
        raise RuntimeError("private diagnostic bug")

    monkeypatch.setattr(diagnostic, "declaration_rejection_facts", fail)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=packet({"done": True}, name="declare_current_completion"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = model(client, desktop_output_format="semantic_tools_v57")
        with pytest.raises(ValueError, match=r"^invalid multimodal planning response \(ValueError\)$"):
            planner.plan("Whole original goal", observed, [], max_actions=1)
    assert planner.last_validation_error == VALIDATION and len(requests) == planner.calls == 1
    assert planner.last_declaration_rejection["details_status"] == "unavailable"
    assert "private" not in json.dumps(planner.last_declaration_rejection)


@pytest.mark.parametrize("format_name", ["semantic_tools_v55", "semantic_tools_v56", "semantic_tools_v57"])
def test_rejection_trace_only_never_enters_history_recent_actions_or_call_count(format_name):
    marker = {"version": "bounded_declaration_rejection_v1", "stage": "declaration_adapter",
              "code": "adapter_rejected", "input_authority": False}

    class Planner:
        profile = "desktop"
        desktop_output_format = format_name
        last_validation_error = VALIDATION
        last_response_structure = {"version": "existing_bounded_envelope"}
        last_usage = {}
        last_declaration_rejection = marker

        def __init__(self):
            self.histories = []

        def plan(self, goal, snapshot, history, *, max_actions=1):
            self.histories.append(copy.deepcopy(history))
            if len(self.histories) == 1:
                raise ValueError("invalid multimodal planning response (ValueError)")
            return finished()

    backend, planner = InputBackend(), Planner()
    backend.values = {"Name": "Ada", "City": "Hangzhou"}
    result, _, _, trace = run([], enabled=False, mode="vlm", backend=backend,
                              planner=planner, max_plans=2)
    assert result.planner_calls == len(planner.histories) == 2 and not backend.executed
    assert planner.histories[1] == [{
        "feedback": "Planning failed validation; return a valid grounded plan.",
        "error_type": "ValueError", "validation": VALIDATION,
        "response_structure": {"version": "existing_bounded_envelope"}, "model_usage": {},
    }]
    assert "declaration_rejection" not in json.dumps(planner.histories)
    errors = [event["payload"] for event in trace.events if event["kind"] == "agent_recovery"]
    assert len(errors) == 1
    if format_name == "semantic_tools_v57":
        assert errors[0]["declaration_rejection"] == marker
    else:
        assert "declaration_rejection" not in errors[0]
