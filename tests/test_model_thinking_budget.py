"""Requested reasoning limits and reported provider usage must stay distinct."""

import json
import time

import httpx
import pytest

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentSnapshot

SNAPSHOT = AgentSnapshot("osworld://desktop", "Editor", "", ())
DONE = {"intent": "complete", "label": "Inspect current evidence"}


def planner(client, **settings):
    return MultimodalPlanner("https://offline.invalid", "mock", profile="desktop",
                             desktop_contract="property_intent_v33", client=client, **settings)


@pytest.mark.parametrize("budget", [1, 512, 1024, 262144, 262145])
def test_positive_budget_is_a_top_level_http_parameter_without_a_generic_model_ceiling(budget):
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(DONE)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner(client, enable_thinking=True, thinking_budget=budget).plan("Inspect", SNAPSHOT, [])
    body, = requests
    assert body["thinking_budget"] == budget and type(body["thinking_budget"]) is int
    assert body["enable_thinking"] is True
    assert not {"extra_body", "max_tokens", "max_completion_tokens", "reasoning_effort"}.intersection(body)
    # A mock acceptance says nothing about a real provider's model-specific ceiling.


def test_default_request_does_not_claim_or_send_a_reasoning_budget():
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(DONE)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner(client, enable_thinking=True).plan("Inspect", SNAPSHOT, [])
    assert "thinking_budget" not in requests[0]


@pytest.mark.parametrize("budget", [True, False, 0, -1, 512.0, "512"])
def test_budget_rejects_invalid_types_and_nonpositive_values_before_request(budget):
    with pytest.raises(ValueError, match="positive integer"):
        MultimodalPlanner("https://offline.invalid", "mock", enable_thinking=True, thinking_budget=budget)


@pytest.mark.parametrize("thinking", [False, None])
def test_budget_cannot_be_silently_ignored_when_thinking_is_not_explicitly_on(thinking):
    with pytest.raises(ValueError, match="enable_thinking=True"):
        MultimodalPlanner("https://offline.invalid", "mock", enable_thinking=thinking, thinking_budget=512)


def test_usage_preserves_provider_reasoning_and_text_counts_without_reinterpreting_them():
    usage = {"completion_tokens": 1604, "prompt_tokens": 10985, "total_tokens": 12589,
             "completion_tokens_details": {"reasoning_tokens": 1429, "text_tokens": 174}}

    def reply(_request):
        return httpx.Response(200, json={"usage": usage,
                                        "choices": [{"message": {"content": json.dumps(DONE)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        model = planner(client, enable_thinking=True, thinking_budget=512)
        model.plan("Inspect", SNAPSHOT, [])
        assert model.last_usage["completion_tokens"] == 1604
        assert model.last_usage["completion_details_reasoning_tokens"] == 1429
        assert model.last_usage["completion_details_text_tokens"] == 174
        assert model.usage_totals == model.last_usage
        # This deliberately inconsistent mock shows that request configuration
        # cannot be substituted for returned usage or prove honoring the limit.


@pytest.mark.parametrize("details", [
    {"reasoning_tokens": True}, {"reasoning_tokens": -1}, {"reasoning_tokens": 1.0},
    {"reasoning_tokens": None}, {"reasoning_tokens": "10"}, [1, 2],
])
def test_invalid_nested_usage_is_not_a_reasoning_measurement(details):
    def reply(_request):
        return httpx.Response(200, json={
            "usage": {"completion_tokens": 5, "completion_tokens_details": details},
            "choices": [{"message": {"content": json.dumps(DONE)}}],
        })

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        model = planner(client)
        model.plan("Inspect", SNAPSHOT, [])
        assert model.last_usage == {"completion_tokens": 5}


def test_response_structure_is_bounded_and_cleared_after_a_transport_failure():
    calls = 0
    secret = "private-field-data-must-never-enter-the-structure-facts"

    def reply(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise httpx.ReadTimeout(secret, request=request)
        raw = {"intent": "act_current", "label": secret, "completion": "stage",
               "action": {"operation": "click", "ref": secret}}
        return httpx.Response(200, json={"usage": {"completion_tokens": 23},
                                        "choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        model = planner(client)
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            model.plan("Inspect", SNAPSHOT, [])
        facts = model.last_response_structure
        assert facts["untrusted"] is True and facts["input_authority"] is False
        assert facts["schema_validated"] is False and secret not in json.dumps(facts)
        assert model.last_usage == {"completion_tokens": 23}
        with pytest.raises(RuntimeError, match="multimodal planning request failed"):
            model.plan("Inspect", SNAPSHOT, [])
        assert model.last_response_structure is None and model.last_usage == {}
        assert model.usage_totals == {"completion_tokens": 23}


def test_expiry_before_http_does_not_reuse_previous_response_usage_or_structure():
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(200, json={"usage": {"completion_tokens": 7},
                                        "choices": [{"message": {"content": json.dumps(DONE)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        model = planner(client)
        model.plan("Inspect", SNAPSHOT, [])
        model.set_episode_deadline(time.perf_counter() - 1)
        with pytest.raises(TimeoutError, match="expired before model request"):
            model.plan("Inspect", SNAPSHOT, [])
        assert model.last_response_structure is None and model.last_usage == {}
        assert model.usage_totals == {"completion_tokens": 7} and len(calls) == 1
