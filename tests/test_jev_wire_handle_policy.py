"""Actual JevPolicy Mock requests, raw responses, original IDs and unchanged guards."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from test_jev_wire_handle_encoding import public_inputs

from cua_jev.errors import GuardRejected
from cua_jev.guard import ActionGuard
from cua_jev.jev_wire_handle_encoding import WireHandleAlias, _wire, restore_wire_handle_request
from cua_jev.models import Decision
from cua_jev.policy import JevPolicy, JevRequestError

LAYOUTS = ("full_v1", "deduplicated_v1", "shared_fields_v2", "shared_fields_v3",
           "shared_fields_v4", "shared_fields_v5", "shared_fields_v6")


def capture(layout, *, handler=None, inputs=None, retries=0):
    observation, candidates = inputs or public_inputs()
    requests = []

    def respond(request):
        requests.append((bytes(request.content), request.headers.get("Authorization")))
        if handler is not None:
            return handler(request)
        ids = list(json.loads(request.content)["questions"]["action"]["criteria"])
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"action": {
            "choice": ids[-1], "probabilities": {key: float(key == ids[-1]) for key in ids},
            "confidence": 1.0,
        }}, "usage": {"known": 1}})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        policy = JevPolicy(api_key="offline-only-key", client=client, request_layout=layout,
                           model="jev-1.13.0", retries=retries, sleep=lambda _seconds: None)
        decision = policy.choose(observation, candidates)
    return requests, decision, policy, observation, candidates


def test_new_wire_preserves_all_facts_guard_pool_IDs_and_raw_provider_response():
    observation, candidates = public_inputs()
    before = _wire([observation.to_dict(), [item.to_dict() for item in candidates]])
    old_requests, _, _, _, _ = capture("shared_fields_v6", inputs=(observation, candidates))
    requests, decision, policy, _, _ = capture("shared_fields_v7", inputs=(observation, candidates))
    assert len(requests) == len(old_requests) == 1
    raw, authorization = requests[0]
    body = json.loads(raw)
    assert authorization == "Bearer offline-only-key" and "offline-only-key" not in raw.decode()
    assert body["model"] == "jev-1.13.0"
    assert body["state"]["encoding"]["version"] == "shared_fields_v7"
    metadata = policy.last_exchange["wire_alias"]
    alias = WireHandleAlias(tuple(pair[1] for pair in metadata["aliases"]),
                            tuple(pair[0] for pair in metadata["aliases"]), metadata["observation_id"],
                            metadata["ancestor_ordered_sha256"], metadata["selected_ordered_sha256"])
    restored = restore_wire_handle_request(body, alias, offered_ids=[item.id for item in candidates],
                                          observation_id=observation.observation_id)
    assert _wire(restored) == old_requests[0][0]
    assert decision.candidate_id == candidates[-1].id
    assert list(decision.probabilities) == [item.id for item in candidates]
    assert ActionGuard().approve(observation, decision, candidates) is candidates[-1]
    assert policy.last_exchange["response"]["answers"]["action"]["choice"] == alias.handles[-1]
    assert "wire_alias" not in body
    assert _wire([observation.to_dict(), [item.to_dict() for item in candidates]]) == before
    # An unconverted provider handle does not acquire any permission in the original guard.
    wire_decision = Decision(observation.observation_id, alias.handles[-1],
                             {key: float(key == alias.handles[-1]) for key in alias.handles},
                             1.0, "offline", 0.0)
    with pytest.raises(GuardRejected, match="offered candidate"):
        ActionGuard().approve(observation, wire_decision, candidates)
    diagnostic = policy.last_exchange["request_encoding"]
    assert diagnostic["version"] == "jev_request_encoding_cost_v3"
    assert diagnostic["requested_layout"] == diagnostic["actual_layout"] == "shared_fields_v7"
    assert diagnostic["compact_utf8_body_bytes"] == len(raw)
    assert diagnostic["wire_handle_encoding"]["selected_ordered_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("problem", ["unknown", "original_ID", "missing_probability", "extra_probability",
                                     "extra_answer_field", "bool_probability", "bool_confidence",
                                     "nonmaximum", "bad_sum", "nonfinite"])
def test_bad_alias_responses_never_guess_replace_retry_or_mutate_raw_response(problem):
    observation, candidates = public_inputs()
    responses = []

    def respond(request):
        ids = list(json.loads(request.content)["questions"]["action"]["criteria"])
        probabilities = {key: float(key == ids[-1]) for key in ids}
        answer = {"choice": ids[-1], "confidence": 1.0, "probabilities": probabilities}
        if problem == "unknown":
            answer["choice"] = "invented"
        elif problem == "original_ID":
            answer["choice"] = candidates[-1].id
        elif problem == "missing_probability":
            del probabilities[ids[0]]
        elif problem == "extra_probability":
            probabilities["invented"] = 0
        elif problem == "extra_answer_field":
            answer["extra"] = "ignored is not allowed"
        elif problem == "bool_probability":
            probabilities[ids[0]] = False
        elif problem == "bool_confidence":
            answer["confidence"] = True
        elif problem == "nonmaximum":
            answer["choice"] = ids[0]
        elif problem == "bad_sum":
            probabilities[ids[0]] = 1.0
        else:
            content = b'{"answers":{"action":{"choice":"h0","confidence":NaN,"probabilities":{}}}}'
            responses.append(None)
            return httpx.Response(200, content=content)
        payload = {"answers": {"action": answer}}
        responses.append(copy.deepcopy(payload))
        return httpx.Response(200, json=payload)

    requests = []

    def count(request):
        requests.append(request)
        return respond(request)

    with httpx.Client(transport=httpx.MockTransport(count), trust_env=False) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout="shared_fields_v7", retries=3)
        with pytest.raises(JevRequestError) as caught:
            policy.choose(observation, candidates)
    assert len(requests) == caught.value.failure.attempts == 1
    assert caught.value.failure.code == "invalid_response" and caught.value.failure.retryable is False
    if responses[0] is not None:
        assert policy.last_exchange["response"] == responses[0]
    assert "wire_alias" in policy.last_exchange
    assert "aliases" not in caught.value.failure.to_dict()["request_encoding"]["wire_handle_encoding"]


def test_duplicate_provider_probability_keys_are_detected_without_content_retry():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=(
            b'{"answers":{"action":{"choice":"h0","confidence":1,'
            b'"probabilities":{"h0":0,"h0":1}}}}'
        ))

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout="shared_fields_v7", retries=4)
        with pytest.raises(JevRequestError) as caught:
            policy.choose(*public_inputs())
    assert len(calls) == caught.value.failure.attempts == 1
    assert caught.value.failure.code == "invalid_response"


def test_unknown_reference_falls_back_full_v6_body_and_old_selection_IDs():
    observation, candidates = public_inputs()
    observation.state["navigation_memory"]["edges"] = [{"candidate_id": candidates[0].id}]
    old, _, _, _, _ = capture("shared_fields_v6", inputs=(observation, candidates))
    new, decision, policy, _, _ = capture("shared_fields_v7", inputs=(observation, candidates))
    assert new[0][0] == old[0][0]
    assert decision.candidate_id == candidates[-1].id
    diagnostic = policy.last_exchange["request_encoding"]
    assert diagnostic["requested_layout"] == "shared_fields_v7"
    assert diagnostic["actual_layout"] == "shared_fields_v6"
    assert diagnostic["wire_handle_encoding"]["fallback_reason"]
    assert "wire_alias" not in policy.last_exchange


def test_actual_v4_ancestor_fallback_and_success_then_capacity_failure_are_current_call_only():
    observation, candidates = public_inputs(1)
    requests, _, policy, _, _ = capture("shared_fields_v7", inputs=(observation, candidates[:1]))
    diagnostic = policy.last_exchange["request_encoding"]
    assert diagnostic["actual_layout"] in {"shared_fields_v4", "shared_fields_v5"}
    assert diagnostic["wire_handle_encoding"]["fallback_reason"] == "ancestor_not_v6"
    assert len(requests) == 1

    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            ids = list(calls[-1]["questions"]["action"]["criteria"])
            return httpx.Response(200, json={"answers": {"action": {
                "choice": ids[0], "confidence": 1.0,
                "probabilities": {key: float(key == ids[0]) for key in ids},
            }}})
        return httpx.Response(400, json={"error": {"code": "max_tokens_exceeded", "message": "PRIVATE"}})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        policy = JevPolicy(api_key="PRIVATE_KEY", client=client, request_layout="shared_fields_v7", retries=3)
        policy.choose(*public_inputs())
        with pytest.raises(JevRequestError) as caught:
            policy.choose(observation, candidates[:1])
    failure = caught.value.failure
    assert len(calls) == 2 and failure.attempts == 1 and not failure.retryable
    assert failure.service_code == "max_tokens_exceeded" and not failure.network_retryable
    assert "wire_alias" not in policy.last_exchange
    assert "PRIVATE" not in json.dumps(failure.to_dict())
    assert failure.request_encoding.to_dict() == policy.last_exchange["request_encoding"]


@pytest.mark.parametrize("failure_kind,expected_attempts", [("capacity", 1), ("authentication", 1),
                                                          ("timeout", 2)])
def test_new_layout_keeps_existing_HTTP_retry_classification_and_local_map(failure_kind, expected_attempts):
    calls, sleeps = [], []

    def respond(request):
        calls.append(bytes(request.content))
        if failure_kind == "timeout":
            raise httpx.ReadTimeout("PRIVATE_PROVIDER_TEXT", request=request)
        status = 400 if failure_kind == "capacity" else 401
        return httpx.Response(status, json={"error": {"code": "max_tokens_exceeded", "message": "PRIVATE"}})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        policy = JevPolicy(api_key="PRIVATE_KEY", client=client, request_layout="shared_fields_v7",
                           retries=1, sleep=sleeps.append)
        with pytest.raises(JevRequestError) as caught:
            policy.choose(*public_inputs())
    failure = caught.value.failure
    assert len(calls) == failure.attempts == expected_attempts
    assert len(set(calls)) == 1
    assert failure.request_encoding.to_dict()["actual_layout"] == "shared_fields_v7"
    assert "wire_alias" in policy.last_exchange and "wire_alias" not in json.loads(calls[0])
    assert failure.request_encoding.to_dict()["compact_utf8_body_bytes"] == len(calls[0])
    assert "PRIVATE" not in json.dumps(failure.to_dict())
    assert (len(sleeps), failure.retryable) == ((1, True) if failure_kind == "timeout" else (0, False))
    if failure_kind == "capacity":
        assert failure.service_code == "max_tokens_exceeded" and not failure.network_retryable


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_old_layout_bodies_decoders_and_default_diagnostics_match_before_v7_golden(layout):
    golden = json.loads((Path(__file__).parent / "fixtures/jev_wire_handles_legacy_golden.json").read_bytes())
    requests, decision, policy, _, candidates = capture(layout)
    raw = requests[0][0]
    assert hashlib.sha256(raw).hexdigest() == golden[layout]["ordered_body_sha256"]
    assert len(raw) == golden[layout]["body_bytes"]
    diagnostic = copy.deepcopy(policy.last_exchange.get("request_encoding"))
    if diagnostic is not None:
        del diagnostic["layout_encoding_wall_ms"]
    assert diagnostic == golden[layout]["request_encoding_without_wall_time"]
    assert {key: getattr(decision, key) for key in (
        "candidate_id", "probabilities", "confidence", "model",
    )} == golden[layout]["decoded_decision"]
    assert decision.candidate_id == candidates[-1].id
    assert "wire_alias" not in policy.last_exchange
    assert policy.last_exchange["response"] == golden[layout]["raw_provider_response"]
