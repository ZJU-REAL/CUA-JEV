import json

import httpx
import pytest

from cua_jev.errors import PolicyError
from cua_jev.models import ActionCandidate, Channel, Observation, Risk
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.policy import JevPolicy, JevRequestError


def capture(items, *, layout="deduplicated_v1", public=False):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        keys = list(body["questions"]["action"]["criteria"])
        return httpx.Response(200, json={"answers": {"action": {
            "choice": keys[-1], "probabilities": {key: float(key == keys[-1]) for key in keys},
            "confidence": 1.0,
        }}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="test", request_layout=layout, client=client, retries=0)
        chooser = PublicDecisionPolicy(policy) if public else policy
        observation = Observation("goal", "stage", {"controls": [], "text": "private OCR"})
        decision = chooser.choose(observation, items)
    return requests[0], decision


@pytest.mark.parametrize("count", [1, 107, 255])
def test_deduplicated_request_retains_every_option_and_full_candidate_information(count):
    items = [ActionCandidate(
        str(i), Channel.GUI, "desktop.click", "Click target " + str(i),
        arguments={"target": i}, risk=Risk.LOCAL_WRITE, preconditions=("fresh", "enabled"),
        verifier="state-check", expected={"changed": i}, requires_confirmation=True, intent="navigate",
    ) for i in range(count)]
    body, decision = capture(items)
    assert "available_actions" not in body["state"]
    criteria = body["questions"]["action"]["criteria"]
    assert [{"id": key, **value} for key, value in criteria.items()] == [i.to_dict() for i in items]
    assert decision.candidate_id == items[-1].id
    assert set(decision.probabilities) == set(criteria)


def test_default_full_layout_and_public_redaction_are_preserved():
    items = [ActionCandidate("one", Channel.GUI, "fill", "Fill field",
                             arguments={"secret": "typed literal"}, expected={"value": "typed literal"})]
    full, _ = capture(items, layout="full_v1")
    assert full["state"]["available_actions"] == [items[0].to_dict()]
    assert isinstance(full["questions"]["action"]["criteria"]["one"], str)
    public, _ = capture(items, public=True)
    assert "typed literal" not in json.dumps(public)
    assert "private OCR" not in json.dumps(public)
    candidate = public["questions"]["action"]["criteria"]["one"]
    assert candidate["arguments"] == candidate["expected"] == {}


@pytest.mark.parametrize("count", [0, 256])
def test_new_layout_never_truncates_out_of_range_candidates(count):
    items = [ActionCandidate(str(i), Channel.GUI, "click", "click") for i in range(count)]
    with pytest.raises(PolicyError, match="1 and 255"):
        capture(items)


def test_new_layout_still_rejects_duplicate_ids_and_unknown_layout():
    one = ActionCandidate("same", Channel.GUI, "click", "click")
    with pytest.raises(PolicyError, match="unique"):
        capture([one, one])
    with pytest.raises(ValueError, match="request layout"):
        JevPolicy(api_key="test", request_layout="unknown")


@pytest.mark.parametrize("body,expected", [
    ({"error": "max_tokens_exceeded"}, "max_tokens_exceeded"),
    ({"detail": {"code": "max_tokens_exceeded", "message": "private response"}}, "max_tokens_exceeded"),
    ({"error": {"code": "max_tokens_exceeded"}}, "max_tokens_exceeded"),
    ({"error": "private response"}, None),
    ({"message": "max_tokens_exceeded"}, None),
    ({"error": {"message": "private max_tokens_exceeded response"}}, None),
    ({"error": "max_tokens_exceeded", "padding": "x" * 4096}, None),
])
def test_context_error_is_sanitized_permanent_and_never_retried_or_recovered(body, expected):
    requests, sleeps = [], []

    def handle(request):
        requests.append(request)
        return httpx.Response(400, json=body)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="test", request_layout="deduplicated_v1", client=client,
                           retries=4, fallback_on_transport=True, sleep=sleeps.append)
        with pytest.raises(JevRequestError) as error:
            policy.choose(Observation("goal", "stage", {}), [
                ActionCandidate("one", Channel.GUI, "click", "click")])
    assert len(requests) == 1 and not sleeps
    failure = error.value.failure
    assert failure.service_code == expected and not failure.network_retryable and not failure.retryable
    assert policy.last_failure.get("service_code") == expected
    assert failure.to_dict().get("service_code") == expected
    assert "private" not in str(failure.to_dict()) + str(policy.last_exchange) + str(error.value)
    assert "fallback" not in policy.last_exchange
