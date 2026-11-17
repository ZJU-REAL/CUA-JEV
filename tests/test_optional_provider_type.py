"""Provider metadata regression from actual previously accepted response shapes.

Both requests below must really use v7 and have a local alias map. A v4 fallback
would not exercise the defective field gate and must not satisfy these tests.
"""

import copy
import json

import httpx
import pytest
from test_jev_wire_handle_encoding import public_inputs
from test_jev_wire_handle_policy import capture

from cua_jev.guard import ActionGuard
from cua_jev.policy import JevPolicy, JevRequestError


def offered_handles(request):
    body = json.loads(request.content)
    assert body["state"]["encoding"]["version"] == "shared_fields_v7"
    assert body["state"]["encoding"]["wire_criterion_handles"] == "opaque_local_v1"
    assert body["state"]["encoding"]["control_description_all"] is True
    handles = list(body["questions"]["action"]["criteria"])
    assert len(handles) == 72 and handles[0] == "h0"
    return handles


@pytest.mark.parametrize("include_type", [False, True])
def test_optional_provider_type_metadata_maps_original_offers_and_keeps_raw_response(include_type):
    observation, candidates = public_inputs()
    raw_responses = []

    def respond(request):
        handles = offered_handles(request)
        # Keep the real provider's observed metadata order when it is present.
        answer = {"type": "choice"} if include_type else {}
        answer.update(choice=handles[-1], confidence=1.0,
                      probabilities={key: float(key == handles[-1]) for key in handles})
        payload = {"answers": {"action": answer}}
        raw_responses.append(copy.deepcopy(payload))
        return httpx.Response(200, json=payload)

    requests, decision, policy, _, returned_candidates = capture(
        "shared_fields_v7", handler=respond, inputs=(observation, candidates), retries=0,
    )
    assert len(requests) == 1
    assert policy.last_exchange["request_encoding"]["actual_layout"] == "shared_fields_v7"
    assert policy.last_exchange["wire_alias"]["aliases"]
    assert decision.candidate_id == candidates[-1].id
    assert list(decision.probabilities) == [item.id for item in candidates]
    assert returned_candidates is candidates
    assert policy.last_exchange["response"] == raw_responses[0]
    assert ("type" in policy.last_exchange["response"]["answers"]["action"]) is include_type
    assert ActionGuard().approve(observation, decision, candidates) is candidates[-1]


@pytest.mark.parametrize("problem", [
    "unknown_handle", "original_ID_echo", "missing_probability", "extra_probability",
    "missing_choice", "missing_confidence", "nonfinite_probability", "nonfinite_confidence",
    "boolean_probability", "boolean_confidence", "extra_answer_field", "nonmaximum", "bad_sum",
    "type_null", "type_bool", "type_number", "type_object", "type_array",
])
def test_type_metadata_never_repairs_an_invalid_choice_or_probability_vector(problem):
    observation, candidates = public_inputs()
    requests, provider_payloads = [], []

    def respond(request):
        requests.append(bytes(request.content))
        handles = offered_handles(request)
        probabilities = {key: float(key == handles[-1]) for key in handles}
        answer = {"type": "choice", "choice": handles[-1], "confidence": 1.0,
                  "probabilities": probabilities}
        if problem == "unknown_handle":
            answer["choice"] = "unoffered"
        elif problem == "original_ID_echo":
            answer["choice"] = candidates[-1].id
        elif problem == "missing_probability":
            del probabilities[handles[0]]
        elif problem == "extra_probability":
            probabilities["unoffered"] = 0
        elif problem == "missing_choice":
            del answer["choice"]
        elif problem == "missing_confidence":
            del answer["confidence"]
        elif problem == "nonfinite_probability":
            probabilities[handles[0]] = float("nan")
        elif problem == "nonfinite_confidence":
            answer["confidence"] = float("inf")
        elif problem == "boolean_probability":
            probabilities[handles[0]] = False
        elif problem == "boolean_confidence":
            answer["confidence"] = True
        elif problem == "extra_answer_field":
            answer["unrecognized"] = "still rejected"
        elif problem == "nonmaximum":
            answer["choice"] = handles[0]
        elif problem == "bad_sum":
            probabilities[handles[0]] = 1.0
        else:
            answer["type"] = {"type_null": None, "type_bool": False, "type_number": 0,
                              "type_object": {}, "type_array": []}[problem]
        payload = {"answers": {"action": answer}}
        provider_payloads.append(copy.deepcopy(payload))
        # This intentionally preserves invalid JSON constants for the existing
        # v7 response validator. No real service is involved.
        return httpx.Response(200, content=json.dumps(payload, allow_nan=True).encode("utf-8"))

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout="shared_fields_v7", retries=3)
        with pytest.raises(JevRequestError) as caught:
            policy.choose(observation, candidates)
    assert len(requests) == caught.value.failure.attempts == 1
    assert caught.value.failure.code == "invalid_response" and not caught.value.failure.retryable
    assert caught.value.failure.request_encoding.to_dict()["actual_layout"] == "shared_fields_v7"
    assert policy.last_exchange["wire_alias"]["aliases"]
    # Compare serialization so deliberate NaN remains comparable without mutating
    # the original parsed provider object into a repaired temporary answer.
    assert json.dumps(policy.last_exchange["response"], sort_keys=True, allow_nan=True) == json.dumps(
        provider_payloads[0], sort_keys=True, allow_nan=True,
    )


def test_optional_type_does_not_allow_duplicate_probability_keys_or_a_content_retry():
    requests = []

    def respond(request):
        requests.append(bytes(request.content))
        handles = offered_handles(request)
        members = [json.dumps(key) + ":" + str(int(key == handles[-1])) for key in handles]
        # Without this duplicate the vector is complete, sums to one, and its
        # selected known handle is maximal. Rejection therefore needs the kept
        # duplicate-key guard rather than an unrelated incomplete-vector failure.
        members.insert(0, json.dumps(handles[0]) + ":0")
        content = ('{"answers":{"action":{"type":"choice","choice":'
                   + json.dumps(handles[-1]) + ',"confidence":1,"probabilities":{'
                   + ",".join(members) + "}}}}")
        return httpx.Response(200, content=content.encode("utf-8"))

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout="shared_fields_v7", retries=3)
        with pytest.raises(JevRequestError) as caught:
            policy.choose(*public_inputs())
    assert len(requests) == caught.value.failure.attempts == 1
    assert caught.value.failure.code == "invalid_response"
    assert policy.last_exchange["request_encoding"]["actual_layout"] == "shared_fields_v7"
    assert policy.last_exchange["wire_alias"]["aliases"]
