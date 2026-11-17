"""Real policy serialization and current-call diagnostics, with offline HTTP only."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import httpx
import pytest

from cua_jev.jev_argument_encoding import restore_argument_bases
from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.policy import JevPolicy, JevRequestError


def _json(value, *, ascii=False):
    return json.dumps(value, ensure_ascii=ascii, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _actions():
    return [ActionCandidate(
        f"{target}:{gesture}", Channel.GUI, "desktop.click", f"Select {target} by {gesture}",
        arguments={"operation": "click", "ref": target, "button": button, "click_count": count,
                   "bound_control": {"ref": target, "description": "exact observed context " * 90,
                                     "ancestors": ["pane: current", f"row: {target}"]}},
    ) for target in ("a", "b")
        for gesture, button, count in (("left", "left", 1), ("right", "right", 1),
                                       ("double", "left", 2))]


def _capture(layout, observation, actions, *, public=False):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        ids = [item.id for item in actions]
        return httpx.Response(200, json={"answers": {"action": {
            "choice": ids[-1], "probabilities": {key: float(key == ids[-1]) for key in ids},
            "confidence": 1.0,
        }}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout=layout, retries=0)
        decision = (PublicDecisionPolicy(policy) if public else policy).choose(observation, actions)
    assert len(requests) == 1
    return requests[0], decision, policy.last_exchange


def test_complete_policy_body_restores_exactly_to_v4_and_keeps_original_selection():
    obs = Observation("Open the current item", "Navigate", {"controls": []})
    actions = _actions()
    original_actions = _json([item.to_dict() for item in actions])
    v4, selected4, exchange4 = _capture("shared_fields_v4", obs, actions)
    v5, selected5, exchange5 = _capture("shared_fields_v5", obs, actions)
    assert _json(restore_argument_bases(v5)) == _json(v4)
    assert list(v5["questions"]["action"]["criteria"]) == [item.id for item in actions]
    assert selected4.candidate_id == selected5.candidate_id == actions[-1].id
    assert _json([item.to_dict() for item in actions]) == original_actions
    assert "request_encoding" not in exchange4
    diag = exchange5["request_encoding"]
    assert diag["requested_layout"] == diag["actual_layout"] == "shared_fields_v5"
    assert diag["argument_base_applied"] is True
    assert (diag["candidate_count"], diag["marked_candidate_count"], diag["argument_base_count"]) == (6, 6, 2)
    assert diag["compact_utf8_body_bytes"] == len(_json(v5).encode("utf-8"))
    assert diag["byte_metric_is_captured_HTTP_bytes"] is False
    assert diag["layout_encoding_wall_ms"] >= 0
    assert len(_json(v5)) < len(_json(v4))
    assert exchange5["request_hash"] == hashlib.sha256(_json(v5, ascii=True).encode()).hexdigest()


def test_requested_v5_unprofitable_request_records_actual_v4_without_changing_body():
    obs = Observation("Read", "Current", {"title": "窗口"})
    actions = _actions()[:1]
    original, _, _ = _capture("shared_fields_v4", obs, actions)
    body, _, exchange = _capture("shared_fields_v5", obs, actions)
    assert body == original
    diag = exchange["request_encoding"]
    assert diag["requested_layout"] == "shared_fields_v5"
    assert diag["actual_layout"] == "shared_fields_v4"
    assert diag["argument_base_applied"] is False
    assert diag["marked_candidate_count"] == diag["argument_base_count"] == 0
    assert diag["compact_utf8_body_bytes"] == len(_json(body).encode("utf-8"))


@pytest.mark.parametrize("layout", ["full_v1", "deduplicated_v1", "shared_fields_v2",
                                   "shared_fields_v3", "shared_fields_v4", "shared_fields_v5"])
def test_capacity_refusal_keeps_current_call_failure_and_never_retries(layout):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json={"answers": {"action": {
                "choice": "a:left", "probabilities": {
                    item.id: float(item.id == "a:left") for item in _actions()
                }, "confidence": 1.0,
            }}})
        return httpx.Response(400, json={"error": {
            "code": "max_tokens_exceeded", "message": "PRIVATE_PROVIDER_TEXT",
        }})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="PRIVATE_API_KEY", client=client, request_layout=layout, retries=3)
        first = Observation("First", "large", {})
        policy.choose(first, _actions())
        second = Observation("Second", "small", {})
        with pytest.raises(JevRequestError) as caught:
            policy.choose(second, _actions()[:1])
    failure = caught.value.failure
    assert len(requests) == 2 and failure.attempts == 1
    assert failure.observation_id == second.observation_id != first.observation_id
    assert failure.request_hash == hashlib.sha256(_json(requests[-1], ascii=True).encode()).hexdigest()
    assert failure.http_status == 400 and failure.service_code == "max_tokens_exceeded"
    assert failure.retryable is False and failure.network_retryable is False
    safe = failure.to_dict()
    assert "PRIVATE_" not in _json(safe)
    if layout == "shared_fields_v5":
        assert safe["request_encoding"] == policy.last_exchange["request_encoding"]
        assert safe["request_encoding"]["actual_layout"] == "shared_fields_v4"
        assert safe["request_encoding"]["candidate_count"] == 1
        with pytest.raises(FrozenInstanceError):
            failure.request_encoding.candidate_count = 99
        safe["request_encoding"]["candidate_count"] = 99
        assert failure.to_dict()["request_encoding"]["candidate_count"] == 1
    else:
        assert "request_encoding" not in safe
        assert "request_encoding" not in policy.last_exchange


def test_invalid_choice_diagnostics_keep_original_failure_without_a_new_request():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"action": {
            "choice": "invented", "probabilities": {}, "confidence": 1.0,
        }}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout="shared_fields_v5", retries=3)
        with pytest.raises(JevRequestError) as caught:
            policy.choose(Observation("Read", "current", {}), _actions())
    failure = caught.value.failure
    assert len(requests) == failure.attempts == 1
    assert failure.code == "invalid_response" and failure.retryable is False
    assert failure.to_dict()["request_encoding"] == policy.last_failure["request_encoding"]


def test_public_policy_shares_retained_descriptions_after_removing_private_arguments():
    from cua_jev.jev_description_encoding import restore_description_fragments

    actions = [replace(
        item, description=(f"Open target {index // 3} " + "visible current window context " * 40
                           + f"gesture {index % 3} toward the requested view"),
        arguments={**item.arguments, "private": "PRIVATE_INPUT_LITERAL"},
    ) for index, item in enumerate(_actions())]
    obs = Observation("Open a view", "Navigate", {"controls": []})
    old, selected4, _ = _capture("shared_fields_v4", obs, actions, public=True)
    new, selected5, exchange = _capture("shared_fields_v5", obs, actions, public=True)
    assert "PRIVATE_INPUT_LITERAL" not in _json(new)
    assert exchange["request_encoding"]["argument_base_applied"] is False
    assert exchange["request_encoding"]["description_fragment_applied"] is True
    assert _json(restore_description_fragments(new)) == _json(old)
    assert selected4.candidate_id == selected5.candidate_id == actions[-1].id
    assert list(new["questions"]["action"]["criteria"]) == [item.id for item in actions]
    assert len(_json(new).encode("utf-8")) < len(_json(old).encode("utf-8"))
