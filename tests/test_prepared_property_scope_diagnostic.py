"""Offline evidence for trace-only diagnostics after the original scope refusal."""

import json
import socket
from dataclasses import replace

import httpx
import pytest
from test_desktop_output_schema_contract import observed as observed
from test_explicit_stage_completion_protocol import parse
from test_semantic_tools_planner import model

from cua_jev import desktop_declaration_rejection as diagnostic
from cua_jev.agent_planner import _validate_desktop_execution_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.trace import JsonlTrace

ERROR = "prepared property scope must identify a currently observed app/window"
WRAPPED = r"^invalid multimodal planning response \(ValueError\)$"
VERSION = "bounded_prepared_property_scope_rejection_v1"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("Only httpx.MockTransport is permitted")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


def rejected_property():
    return {
        "intent": "pursue_property", "label": "Set requested property",
        "scope": {"app": "Unobserved Editor", "window": "Future panel", "ancestor": "Future owner"},
        "prepare": {"operation": "click", "ref": "help"},
        "edit": {"operation": "fill", "value": "100"},
        "numeric_finalize": {"key": "enter"},
        "confirm": {"label": "Confirm requested property"}, "completion": "goal",
    }


def responder(requests, raw):
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(raw), "reasoning_content": "PRIVATE_REASONING_MARKER",
        }}]})

    return respond


class RefusalBackend:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.inputs = self.compiles = 0

    def observe(self):
        return self.snapshot

    def compile(self, *_args):
        self.compiles += 1
        raise AssertionError("A refused plan must not reach compilation")

    def execute(self, *_args):
        self.inputs += 1
        raise AssertionError("A refused plan must not send input")

    def verify(self, *_args):
        raise AssertionError("A refused plan has no receipt")

    def close(self):
        pass


def run_rejected(snapshot, *, max_plans=1):
    requests, trace = [], JsonlTrace()
    backend = RefusalBackend(snapshot)
    with httpx.Client(transport=httpx.MockTransport(responder(requests, rejected_property()))) as client:
        planner = model(client, desktop_output_format="unconstrained_v1")
        result = AgentRunner(
            backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, max_plans=max_plans, max_failures=max_plans),
        ).run("Apply and confirm the requested property")
    assert result.planner_calls == planner.calls == len(requests) == max_plans
    assert backend.inputs == backend.compiles == 0
    return planner, requests, trace


def test_actual_original_scope_guard_and_raw_planner_runner_rejection_send_zero_inputs(observed):
    parsed = parse(rejected_property(), observed)
    with pytest.raises(ValueError, match="^" + ERROR + "$"):
        _validate_desktop_execution_plan(parsed, observed, 6)
    planner, requests, trace = run_rejected(observed)
    assert planner.last_validation_error == ERROR
    facts = planner.last_declaration_rejection
    assert facts["version"] == VERSION and facts["declared_scope"] == rejected_property()["scope"]
    assert facts["declared_scope_match_count"] == 0 and facts["input_authority"] is False
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert facts["offered_navigation_scopes"] == [{
        "target": {"app": item["target"]["app"], "window": "", "ancestor": ""},
        "candidate_count": item["candidate_count"], "scope_sanitized": False, "scope_truncated": False,
    } for item in state["navigation_scopes"]]
    errors = [row["payload"] for row in trace.events if row["kind"] == "agent_recovery"]
    assert len(errors) == 1 and errors[0]["declaration_rejection"] == facts
    assert "PRIVATE_REASONING_MARKER" not in json.dumps(facts)


def test_scope_values_items_counts_are_bounded_sanitized_and_ignore_unknown_fields(observed):
    parsed = parse(rejected_property(), observed)
    long_text = "\x1b" + "X" * 500
    target = replace(parsed.steps[1].target, app=long_text, window="Future\n\u202e", ancestor="Owner\t")
    parsed = replace(parsed, steps=(
        parsed.steps[0], replace(parsed.steps[1], target=target), *parsed.steps[2:],
    ))
    marker = "PRIVATE_UNKNOWN_MARKER"
    offered = [{
        "target": {"app": long_text, "window": "Main\n\u202e", "ancestor": "Group\t", "token": marker},
        "candidate_count": 255, "headers": marker, "reasoning_content": marker, "value": marker,
    } for _ in range(300)]
    facts = diagnostic.prepared_property_scope_rejection_facts(
        parsed, observed, offered_scopes=offered, error=ValueError(ERROR),
    )
    assert facts["offered_scope_count_capped"] == 255 and facts["offered_scope_count_overflow"]
    assert len(facts["offered_navigation_scopes"]) == 8 and not facts["offered_scope_list_complete"]
    assert facts["declared_scope_sanitized"] and facts["declared_scope_truncated"]
    for scope in [facts["declared_scope"], *(row["target"] for row in facts["offered_navigation_scopes"])]:
        assert set(scope) == {"app", "window", "ancestor"}
        assert all(len(value) <= 240 and all(c.isprintable() for c in value) for value in scope.values())
    assert marker not in json.dumps(facts) and len(json.dumps(facts)) < 11000
    assert all(row["candidate_count"] == 255 for row in facts["offered_navigation_scopes"])


def test_unknown_raw_scope_field_is_original_parse_refusal_without_scope_content_diagnostic(observed):
    raw, requests = rejected_property(), []
    marker = "PRIVATE_UNKNOWN_SCOPE_MARKER"
    raw["scope"][marker] = marker
    with httpx.Client(transport=httpx.MockTransport(responder(requests, raw))) as client:
        planner = model(client, desktop_output_format="unconstrained_v1")
        with pytest.raises(ValueError, match=WRAPPED):
            planner.plan("Requested property", observed, [], max_actions=6)
    assert planner.calls == len(requests) == 1 and planner.last_declaration_rejection is None
    assert planner.last_validation_error != ERROR
    assert marker not in planner.last_validation_error


def test_diagnostic_bug_preserves_original_error_one_post_and_zero_inputs(observed, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("PRIVATE_DIAGNOSTIC_MARKER")

    monkeypatch.setattr(diagnostic, "prepared_property_scope_rejection_facts", fail)
    planner, _, trace = run_rejected(observed)
    assert planner.last_validation_error == ERROR
    assert planner.last_declaration_rejection["details_status"] == "unavailable"
    assert "PRIVATE_DIAGNOSTIC_MARKER" not in json.dumps(trace.events)
    requests = []
    with httpx.Client(transport=httpx.MockTransport(responder(requests, rejected_property()))) as client:
        direct = model(client, desktop_output_format="unconstrained_v1")
        with pytest.raises(ValueError, match=WRAPPED):
            direct.plan("Requested property", observed, [], max_actions=6)
    assert direct.last_validation_error == ERROR and len(requests) == direct.calls == 1


def test_trace_diagnostic_does_not_change_recovery_history_prompt_or_request(observed, monkeypatch):
    _, actual_requests, actual_trace = run_rejected(observed, max_plans=2)
    monkeypatch.setattr(diagnostic, "prepared_property_scope_rejection_facts", lambda *_a, **_kw: None)
    _, control_requests, control_trace = run_rejected(observed, max_plans=2)
    assert actual_requests == control_requests
    assert VERSION not in json.dumps(actual_requests)
    assert any("declaration_rejection" in row["payload"] for row in actual_trace.events)
    assert not any("declaration_rejection" in row["payload"] for row in control_trace.events)


def test_new_call_clears_previous_scope_diagnostic_even_before_http_failure(observed):
    requests, raw = [], rejected_property()
    with httpx.Client(transport=httpx.MockTransport(responder(requests, raw))) as client:
        planner = model(client, desktop_output_format="unconstrained_v1")
        with pytest.raises(ValueError, match=WRAPPED):
            planner.plan("Requested property", observed, [], max_actions=6)
        assert planner.last_declaration_rejection is not None
        with pytest.raises(ValueError, match="whole_goal_recheck must be a boolean"):
            planner.plan("Requested property", observed, [], max_actions=6, whole_goal_recheck="invalid")
        assert planner.last_declaration_rejection is None and len(requests) == 1
